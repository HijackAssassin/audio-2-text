"""Audio2Text - batch transcribe audio files into a single text file."""

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request
import webbrowser
import zipfile
from datetime import datetime

import update_check

APP_VERSION = update_check.APP_VERSION  # set in version.txt
FROZEN = getattr(sys, "frozen", False)
APP_DIR = os.path.dirname(sys.executable if FROZEN else os.path.abspath(__file__))
CUDA_DIR = os.path.join(APP_DIR, "cuda")
SETTINGS_PATH = os.path.join(os.environ.get("APPDATA", APP_DIR), "Audio2Text", "settings.json")

# A windowed exe has no console; give libraries that print somewhere harmless to write.
for _stream in ("stdout", "stderr"):
    if getattr(sys, _stream) is None:
        setattr(sys, _stream, open(os.devnull, "w"))

# Python 3.13.0 venvs can't locate Tcl/Tk; point them at the base install.
_tcl_root = os.path.join(sys.base_prefix, "tcl")
for _var, _name in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
    if _var not in os.environ and os.path.isdir(os.path.join(_tcl_root, _name)):
        os.environ[_var] = os.path.join(_tcl_root, _name)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3")
OUTPUT_TYPES = [
    ("Text file", "*.txt"),
    ("Log file", "*.log"),
    ("Markdown file", "*.md"),
    ("Text file (.text)", "*.text"),
    ("All files", "*.*"),
]
# (PyPI package, required major version) - CUDA libraries CTranslate2 needs for GPU inference.
GPU_PACKAGES = [("nvidia-cublas-cu12", "12"), ("nvidia-cudnn-cu12", "9")]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The models are public; Hugging Face's server-sent "unauthenticated requests" notice is harmless.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# (model id, download size in MB, dropdown tagline, longer description)
MODELS = [
    ("large-v3-turbo", 1620, "RECOMMENDED, most accurate (~1.6 GB)",
     "Best accuracy in testing, including with accents and shouting. "
     "Fast on a GPU, usable on CPU."),
    ("medium", 1530, "Accurate, slower (~1.5 GB)",
     "Nearly as accurate as turbo but slower. Try it if turbo mishears a particular voice."),
    ("base", 148, "Fast, rough draft (~145 MB)",
     "Small and quick, for older PCs without a GPU. Expect roughly 1 wrong word in 8."),
]
MODEL_SIZE_MB = {m: size for m, size, _, _ in MODELS}
MODEL_INFO = {m: desc for m, _, _, desc in MODELS}
DEFAULT_MODEL = "large-v3-turbo"
INSTALLED_MARK = "✓"  # check mark

LANGUAGES = {"English": "en", "Auto-detect": None}
GPU_DEVICE, CPU_DEVICE = "GPU (Recommended)", "CPU"

START_TEXT = "Start Audio To Text"
INSTALL_TEXT = "Model not installed - Download it now?"
DELETE_ICON, DOWNLOAD_ICON = "✖", "\U0001F847"  # heavy X, heavy down arrow


# ---------------------------------------------------------------- settings

def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data):
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def scan_folder(folder, recursive):
    """Return audio files in `folder` (optionally its subfolders), sorted by name."""
    if recursive:
        found = []
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort(key=str.lower)
            found += [os.path.join(dirpath, f) for f in sorted(filenames, key=str.lower)]
    else:
        found = [os.path.join(folder, f) for f in sorted(os.listdir(folder), key=str.lower)
                 if os.path.isfile(os.path.join(folder, f))]
    return [f for f in found if f.lower().endswith(AUDIO_EXTENSIONS)]


# ---------------------------------------------------------------- internet

def internet_available(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def no_internet(what):
    messagebox.showerror(
        "No Internet Connection",
        f"Could not download {what}.\n\n"
        "An internet connection is required to download it. "
        "Please connect to the internet and try again.")


# ---------------------------------------------------------------- models

def model_cache_dir(name):
    from faster_whisper.utils import _MODELS
    from huggingface_hub import constants
    return os.path.join(constants.HF_HUB_CACHE, "models--" + _MODELS[name].replace("/", "--"))


def is_model_installed(name):
    from faster_whisper.utils import download_model
    try:
        path = download_model(name, local_files_only=True)
    except Exception:
        return False
    return os.path.isfile(os.path.join(path, "model.bin"))


def _link_targets(folder):
    """Real files that the symlinks under `folder` point to (following link chains)."""
    targets = set()
    for dirpath, _dirs, files in os.walk(folder):
        for f in files:
            p = os.path.join(dirpath, f)
            if os.path.islink(p):
                targets.add(os.path.normcase(os.path.realpath(p)))
    return targets


def delete_model(name):
    """Delete a model's folder plus any shared-store files that no other model links to.

    Deliberately conservative: a stored file is only removed after checking every other
    repo in the cache, so deleting one model can never break another."""
    from huggingface_hub import constants
    cache = constants.HF_HUB_CACHE
    repo_dir = model_cache_dir(name)
    if not os.path.isdir(repo_dir):
        return
    mine = _link_targets(repo_dir)
    others = set()
    for entry in os.listdir(cache):
        path = os.path.join(cache, entry)
        if "--" in entry and os.path.isdir(path) and os.path.normcase(path) != os.path.normcase(repo_dir):
            others |= _link_targets(path)
    shutil.rmtree(repo_dir)
    store = os.path.normcase(os.path.join(cache, "blobs")) + os.sep
    for target in mine - others:
        if target.startswith(store) and os.path.isfile(target):
            os.remove(target)
            if os.path.exists(target + ".refs"):
                os.remove(target + ".refs")


def cache_drive_free():
    """Free bytes on the model cache's drive. Download progress is measured from this because
    NTFS doesn't update a file's listed size until the downloader closes it."""
    from huggingface_hub import constants
    os.makedirs(constants.HF_HUB_CACHE, exist_ok=True)
    return shutil.disk_usage(constants.HF_HUB_CACHE).free


# ---------------------------------------------------------------- hardware

def _add_cuda_dll_dirs():
    """Make CUDA libraries loadable from the app's cuda folder or pip-installed nvidia packages."""
    dirs = [CUDA_DIR]
    try:
        import nvidia
        for base in nvidia.__path__:
            dirs += [os.path.join(base, lib, "bin") for lib in os.listdir(base)]
    except ImportError:
        pass
    for d in dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _latest_windows_wheel(package, major):
    """Find the newest win_amd64 wheel of `package` with the given major version on PyPI."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as r:
        releases = json.load(r)["releases"]
    versions = [v for v in releases if v.split(".")[0] == major and v.replace(".", "").isdigit()]
    for v in sorted(versions, key=lambda v: [int(x) for x in v.split(".")], reverse=True):
        for f in releases[v]:
            if f["filename"].endswith("win_amd64.whl") and not f.get("yanked"):
                return f["url"], f["size"], f["filename"]
    raise RuntimeError(f"No Windows download found for {package} {major}.x")


def install_cuda_libraries(log):
    """Download NVIDIA's CUDA wheels from PyPI and extract their DLLs into CUDA_DIR."""
    os.makedirs(CUDA_DIR, exist_ok=True)
    for package, major in GPU_PACKAGES:
        url, size, filename = _latest_windows_wheel(package, major)
        log(f"Downloading {filename} ({size / 1e6:.0f} MB)...")
        with tempfile.TemporaryDirectory() as tmp:
            wheel = os.path.join(tmp, filename)
            with urllib.request.urlopen(url, timeout=60) as r, open(wheel, "wb") as f:
                done, next_report = 0, 0.25
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if done / size >= next_report:
                        log(f"  {done / size:.0%}")
                        next_report += 0.25
            with zipfile.ZipFile(wheel) as z:
                for member in z.namelist():
                    if member.lower().endswith(".dll") and "/bin/" in member:
                        with z.open(member) as src, open(
                                os.path.join(CUDA_DIR, os.path.basename(member)), "wb") as dst:
                            shutil.copyfileobj(src, dst)
        log(f"Installed {package}.")


def _cuda_libs_loadable():
    import ctypes
    _add_cuda_dll_dirs()
    try:
        ctypes.WinDLL("cublas64_12.dll")
        ctypes.WinDLL("cudnn64_9.dll")
        return True
    except OSError:
        return False


def check_hardware():
    """Return a dict describing the CPU/GPU and whether GPU transcription is ready."""
    info = {"cpu_threads": os.cpu_count() or 1, "gpu": None, "vram_gb": None, "gpu_ready": False}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW,
        ).stdout.strip().splitlines()
        if out:
            name, mem = [p.strip() for p in out[0].split(",")]
            info["gpu"], info["vram_gb"] = name, round(int(mem) / 1024, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if info["gpu"]:
        try:
            import ctranslate2
            info["gpu_ready"] = ctranslate2.get_cuda_device_count() > 0 and _cuda_libs_loadable()
        except Exception:
            pass
    return info


# ---------------------------------------------------------------- app

class Audio2TextApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Audio2Text v{APP_VERSION}")
        self.root.minsize(700, 760)

        s = load_settings()
        self.last_output_dir = s.get("output_dir") if os.path.isdir(s.get("output_dir") or "") else None
        self.audio_files = []  # full paths, in display order
        self.output_path = tk.StringVar()
        self.include_subfolders = tk.BooleanVar(value=bool(s.get("include_subfolders", False)))
        self.model_label = tk.StringVar()
        self.model_desc = tk.StringVar()
        self.language = tk.StringVar(value=s.get("language") if s.get("language") in LANGUAGES else "English")
        self.device_choice = tk.StringVar(value=CPU_DEVICE if s.get("device") == CPU_DEVICE else GPU_DEVICE)
        self.skip_silence = tk.BooleanVar(value=bool(s.get("skip_silence", False)))
        self.vocabulary = tk.StringVar(value=str(s.get("vocabulary", "")))
        self.hw_text = tk.StringVar(value="Checking hardware...")
        self.status = tk.StringVar(value="Add or drag in audio files or folders, then choose where to save the text file.")

        self.hardware = None
        self.installed = None  # set of installed model ids once checked
        self.selected_model = s.get("model") if s.get("model") in MODEL_INFO else DEFAULT_MODEL
        self.busy = False
        self.events = queue.Queue()
        self.cancel_requested = threading.Event()

        self._build_ui()
        self._enable_drag_and_drop()
        self._refresh_model_ui()
        if s.get("geometry"):
            try:
                self.root.geometry(s["geometry"])
            except tk.TclError:
                pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)
        self._in_background(self._check_system)
        self._in_background(self._check_for_update)

    def _save_settings(self):
        save_settings({
            "model": self.selected_model,
            "language": self.language.get(),
            "device": self.device_choice.get(),
            "skip_silence": self.skip_silence.get(),
            "include_subfolders": self.include_subfolders.get(),
            "vocabulary": self.vocabulary.get().strip(),
            "output_dir": self.last_output_dir,
            "geometry": self.root.geometry(),
        })

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    def _enable_drag_and_drop(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except (ImportError, AttributeError, tk.TclError):
            pass  # Plain Tk root (e.g. tkinterdnd2 missing): Add Files / Add Folder still work.

    def _on_drop(self, event):
        files, folders = [], []
        for p in self.root.tk.splitlist(event.data):
            (folders if os.path.isdir(p) else files).append(p)
        found = list(files)
        for folder in folders:
            found += scan_folder(folder, self.include_subfolders.get())
        if self._add_paths(found):
            self._suggest_output(os.path.dirname(os.path.normpath(found[0])))
        else:
            messagebox.showwarning("Audio2Text", "No .flac, .wav or .mp3 files were found in what you dropped.")
        return event.action

    def _in_background(self, target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    def _check_system(self):
        self.events.put(("models", {m for m, *_ in MODELS if is_model_installed(m)}))
        self.events.put(("hardware", check_hardware()))

    def _check_for_update(self):
        """Ask the website for the latest version. Quiet unless there is an update."""
        try:
            offer = update_check.check()
        except Exception:
            return  # offline or the site is down: no bar
        if update_check.is_newer(offer["version"]):
            self.events.put(("update", offer))

    def _show_update(self, offer):
        """A bar above everything else: what's out, Download (opens it in the browser) and Later."""
        style = ttk.Style(self.root)
        bold = ("Segoe UI", 9, "bold")
        style.configure("Update.TFrame", background="#E3F2FD")
        style.configure("Update.TLabel", background="#E3F2FD", foreground="#1565c0", font=bold)
        style.configure("UpdateMuted.TLabel", background="#E3F2FD", foreground="#555")
        bar = ttk.Frame(self.root, style="Update.TFrame", padding=(12, 6))
        bar.pack(fill="x", padx=12, pady=(12, 0), before=self.main_frame)
        ttk.Label(bar, text=f"Update available: Audio2Text v{offer['version']}",
                  style="Update.TLabel").pack(side="left")
        ttk.Label(bar, text=f"You have v{APP_VERSION}.", style="UpdateMuted.TLabel").pack(
            side="left", padx=(8, 0))
        ttk.Button(bar, text="Later", command=bar.destroy).pack(side="right")
        ttk.Button(bar, text="Download", command=lambda: webbrowser.open(offer["download"])).pack(
            side="right", padx=(0, 6))
        self._log(f"Audio2Text v{offer['version']} is available (you have v{APP_VERSION}).")

    # ---- layout

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        self.main_frame = frame
        frame.columnconfigure(0, weight=1)

        # Input
        inp = ttk.LabelFrame(frame, text="Audio input (.flac, .wav, .mp3)", padding=8)
        inp.grid(row=0, column=0, sticky="nsew")
        inp.columnconfigure(0, weight=1)
        inp.rowconfigure(0, weight=1)
        self.file_list = tk.Listbox(inp, height=7, selectmode="extended", activestyle="none")
        self.file_list.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(inp, orient="vertical", command=self.file_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.bind("<Delete>", lambda e: self._remove_selected())

        side = ttk.Frame(inp)
        side.grid(row=0, column=2, sticky="n", padx=(8, 0))
        ttk.Button(side, text="Add Files...", command=self._add_files).pack(fill="x")
        ttk.Button(side, text="Add Folder...", command=self._add_folder).pack(fill="x", pady=4)
        ttk.Checkbutton(side, text="Include subfolders", variable=self.include_subfolders).pack(anchor="w")
        ttk.Button(side, text="Remove Selected", command=self._remove_selected).pack(fill="x", pady=(12, 4))
        ttk.Button(side, text="Clear All", command=self._clear_files).pack(fill="x")
        self.count_label = ttk.Label(inp)
        self.count_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Output
        out = ttk.LabelFrame(frame, text="Output text file", padding=8)
        out.grid(row=1, column=0, sticky="ew", pady=8)
        out.columnconfigure(0, weight=1)
        ttk.Entry(out, textvariable=self.output_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(out, text="Save As...", command=self._pick_output).grid(row=0, column=1, padx=(8, 0))

        # Settings
        opts = ttk.LabelFrame(frame, text="Settings", padding=8)
        opts.grid(row=2, column=0, sticky="ew")
        opts.columnconfigure(1, weight=1)
        ttk.Label(opts, text="Model:").grid(row=0, column=0, sticky="w")
        model_row = ttk.Frame(opts)
        model_row.grid(row=0, column=1, sticky="w", padx=6)
        self.model_box = ttk.Combobox(model_row, textvariable=self.model_label,
                                      state="readonly", width=58)
        self.model_box.pack(side="left")
        self.model_box.bind("<<ComboboxSelected>>", self._on_model_change)
        self.model_button = tk.Button(model_row, width=3, pady=0, relief="groove", cursor="hand2",
                                      font=("Segoe UI Symbol", 11, "bold"))
        self.model_button.pack(side="left", padx=(6, 0))
        ttk.Label(opts, textvariable=self.model_desc, foreground="#555", wraplength=580).grid(
            row=1, column=1, sticky="w", padx=6, pady=(2, 8))

        ttk.Label(opts, text="Language:").grid(row=2, column=0, sticky="w")
        row2 = ttk.Frame(opts)
        row2.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Combobox(row2, textvariable=self.language, values=list(LANGUAGES),
                     state="readonly", width=12).pack(side="left")
        ttk.Label(row2, text="Device:").pack(side="left", padx=(16, 4))
        ttk.Combobox(row2, textvariable=self.device_choice, values=(GPU_DEVICE, CPU_DEVICE),
                     state="readonly", width=20).pack(side="left")
        ttk.Checkbutton(row2, text="Skip silence (for long recordings)",
                        variable=self.skip_silence).pack(side="left", padx=(16, 0))

        ttk.Label(opts, text="Vocabulary:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opts, textvariable=self.vocabulary).grid(row=3, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Label(opts, text="Optional: names or unusual words the audio contains, separated by commas "
                             "(e.g. Richtofen, Blondie, sinew).",
                  foreground="#555", wraplength=580).grid(row=4, column=1, sticky="w", padx=6, pady=(2, 0))

        hw = ttk.Frame(opts)
        hw.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(hw, text="Hardware:").pack(side="left")
        ttk.Label(hw, textvariable=self.hw_text).pack(side="left", padx=6)
        self.gpu_button = ttk.Button(hw, text="Install GPU Support", command=self._install_gpu)

        # Run
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, pady=10)
        self.start_button = ttk.Button(buttons, text=START_TEXT, command=self._on_start_click)
        self.start_button.pack(side="left", ipadx=12, ipady=4)
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8, ipady=4)

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(row=4, column=0, sticky="ew")
        ttk.Label(frame, textvariable=self.status).grid(row=5, column=0, sticky="w", pady=4)

        self.log = tk.Text(frame, height=6, state="disabled", wrap="word")
        self.log.grid(row=6, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(6, weight=1)
        self._update_count()

    # ---- input list

    def _add_paths(self, paths):
        known = {os.path.normcase(p) for p in self.audio_files}
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            if os.path.normcase(p) not in known and p.lower().endswith(AUDIO_EXTENSIONS):
                known.add(os.path.normcase(p))
                self.audio_files.append(p)
                self.file_list.insert("end", p)
                added += 1
        self._update_count()
        return added

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio files", "*.flac *.wav *.mp3"), ("All files", "*.*")])
        if paths and self._add_paths(paths):
            self._suggest_output(os.path.dirname(paths[0]))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing audio files")
        if not folder:
            return
        found = scan_folder(folder, self.include_subfolders.get())
        if self._add_paths(found):
            self._suggest_output(folder)
        elif not found:
            messagebox.showwarning("Audio2Text", "No .flac, .wav or .mp3 files found in that folder.")

    def _remove_selected(self):
        for i in reversed(self.file_list.curselection()):
            self.file_list.delete(i)
            del self.audio_files[i]
        self._update_count()

    def _clear_files(self):
        self.file_list.delete(0, "end")
        self.audio_files.clear()
        self._update_count()

    def _update_count(self):
        n = len(self.audio_files)
        text = f"{n} file{'s' if n != 1 else ''}"
        if n == 0:
            text += "  -  you can also drag and drop audio files or folders onto this window"
        self.count_label.configure(text=text)

    # ---- output

    def _default_name(self):
        return f"Audio2Text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def _suggest_output(self, folder):
        if not self.output_path.get().strip():
            folder = self.last_output_dir or folder
            self.output_path.set(os.path.join(os.path.normpath(folder), self._default_name()))

    def _pick_output(self):
        current = self.output_path.get().strip()
        path = filedialog.asksaveasfilename(
            title="Save transcription as",
            defaultextension=".txt",
            filetypes=OUTPUT_TYPES,
            initialdir=os.path.dirname(current) if current else self.last_output_dir,
            initialfile=os.path.basename(current) if current else self._default_name())
        if path:
            self.output_path.set(os.path.normpath(path))

    # ---- models

    def _model_installed(self, name):
        return self.installed is not None and name in self.installed

    def _on_model_change(self, _event=None):
        self.selected_model = self.model_label.get().split()[0]
        self._refresh_model_ui()

    def _refresh_model_ui(self):
        """Sync dropdown check marks, the delete/download button and the start button."""
        labels = []
        for m, _size, tag, _desc in MODELS:
            label = f"{m}  -  {tag}" + (f"  {INSTALLED_MARK}" if self._model_installed(m) else "")
            labels.append(label)
            if m == self.selected_model:
                self.model_label.set(label)
        self.model_box.configure(values=labels)

        name = self.selected_model
        checking = self.installed is None
        installed = self._model_installed(name)
        if checking:
            self.model_desc.set(MODEL_INFO[name])
        elif installed:
            self.model_desc.set(f"{MODEL_INFO[name]}\nInstalled and ready to use.")
        else:
            self.model_desc.set(f"{MODEL_INFO[name]}\nNot installed - click the blue arrow "
                                "to download it (one time only).")

        if installed:
            self.model_button.configure(text=DELETE_ICON, fg="#c62828", activeforeground="#c62828",
                                        command=self._delete_model)
        else:
            self.model_button.configure(text=DOWNLOAD_ICON, fg="#1565c0", activeforeground="#1565c0",
                                        command=self._download_model)
        idle = "disabled" if self.busy or checking else "normal"
        self.model_button.configure(state=idle)
        self.model_box.configure(state="disabled" if self.busy else "readonly")
        self.start_button.configure(text=START_TEXT if installed or checking else INSTALL_TEXT,
                                    state=idle)
        self.gpu_button.configure(state="disabled" if self.busy else "normal")

    def _set_busy(self, busy):
        self.busy = busy
        self._refresh_model_ui()

    def _delete_model(self):
        name = self.selected_model
        size = MODEL_SIZE_MB[name]
        size_text = f"{size / 1000:.1f} GB" if size >= 1000 else f"{size} MB"
        if not messagebox.askyesno(
                "Delete Model",
                f"Delete the {name} model and free up {size_text}?\n\n"
                "You can download it again later."):
            return
        delete_model(name)
        if is_model_installed(name):
            messagebox.showerror("Delete Model", f"Could not delete the {name} model.\n\n"
                                                 f"Folder: {model_cache_dir(name)}")
        else:
            self.installed.discard(name)
            self._log(f"Deleted model '{name}'.")
        self._refresh_model_ui()

    def _download_model(self):
        name = self.selected_model
        if not internet_available("huggingface.co"):
            no_internet(f"the {name} model")
            return
        self._set_busy(True)
        self.progress.configure(mode="determinate", maximum=MODEL_SIZE_MB[name], value=0)
        self.status.set(f"Downloading {name} model...")
        self._log(f"Downloading model '{name}' (~{MODEL_SIZE_MB[name]} MB)...")
        self._in_background(self._run_model_download, name)

    def _run_model_download(self, name):
        from faster_whisper.utils import download_model
        baseline = cache_drive_free()
        done = threading.Event()

        def report():
            while not done.wait(0.5):
                mb = max(0, baseline - cache_drive_free()) / 1e6
                self.events.put(("model_progress", (name, mb)))

        self._in_background(report)
        try:
            download_model(name)
            ok = is_model_installed(name)
            error = None if ok else "The download did not complete."
        except Exception as e:
            ok, error = False, str(e)
        done.set()
        offline = not ok and not internet_available("huggingface.co")
        self.events.put(("model_downloaded", (name, ok, offline, error)))

    # ---- hardware / GPU support

    def _show_hardware(self, info):
        self.hardware = info
        cpu = f"CPU {info['cpu_threads']} threads"
        self.gpu_button.pack_forget()
        if not info["gpu"]:
            self.hw_text.set(f"{cpu}  |  No NVIDIA GPU found - using CPU")
            self.device_choice.set(CPU_DEVICE)
        elif info["gpu_ready"]:
            self.hw_text.set(f"{cpu}  |  {info['gpu']} ({info['vram_gb']} GB) - GPU ready")
        else:
            self.hw_text.set(f"{cpu}  |  {info['gpu']} ({info['vram_gb']} GB) - GPU support not installed")
            self.gpu_button.pack(side="left", padx=6)

    def _install_gpu(self):
        if not messagebox.askyesno(
                "Install GPU Support",
                "This downloads NVIDIA's CUDA libraries (about 1.3 GB) into the app's "
                "'cuda' folder so transcription can run on your graphics card, which is "
                "much faster.\n\nContinue?"):
            return False
        if not internet_available("pypi.org"):
            no_internet("GPU support")
            return False
        self._set_busy(True)
        self.hw_text.set("Installing GPU support - this can take several minutes...")
        self._in_background(self._run_gpu_install)
        return True

    def _run_gpu_install(self):
        try:
            install_cuda_libraries(lambda msg: self.events.put(("log", msg)))
            result = "ok"
        except Exception as e:
            self.events.put(("log", f"GPU support install failed: {e}"))
            result = "offline" if not internet_available("pypi.org") else "failed"
        self.events.put(("gpu_installed", result))

    # ---- run

    def _on_start_click(self):
        if self._model_installed(self.selected_model):
            self._start()
        else:
            self._download_model()

    def _choose_device(self):
        """Return 'cuda', 'cpu', or None if the user cancelled."""
        if self.device_choice.get() == CPU_DEVICE:
            return "cpu"
        hw = self.hardware
        if hw is None or hw["gpu_ready"]:
            return "cuda"
        if not hw["gpu"]:
            messagebox.showinfo("Audio2Text", "No NVIDIA graphics card was found, so the CPU will be used.")
            self.device_choice.set(CPU_DEVICE)
            return "cpu"
        answer = messagebox.askyesnocancel(
            "GPU Support Not Installed",
            "GPU support isn't installed yet, so transcription would run on the slower CPU.\n\n"
            "Yes - install GPU support now (about 1.3 GB)\n"
            "No - use the CPU for this run")
        if answer is None:
            return None
        if answer:
            self._install_gpu()
            return None
        return "cpu"

    def _start(self):
        if not self.audio_files:
            messagebox.showerror("Audio2Text", "Add some audio files or a folder first.")
            return
        missing = [p for p in self.audio_files if not os.path.isfile(p)]
        if missing:
            messagebox.showerror("Audio2Text", "These files no longer exist:\n" + "\n".join(missing[:10]))
            return

        out_file = self.output_path.get().strip()
        if not out_file:
            self._pick_output()
            out_file = self.output_path.get().strip()
            if not out_file:
                return
        if not os.path.splitext(out_file)[1]:
            out_file += ".txt"
            self.output_path.set(out_file)
        if not os.path.isdir(os.path.dirname(os.path.abspath(out_file))):
            messagebox.showerror("Audio2Text", "The output folder does not exist.")
            return
        if os.path.exists(out_file) and not messagebox.askyesno(
                "Audio2Text", f"{os.path.basename(out_file)} already exists. Overwrite it?"):
            return

        device = self._choose_device()
        if device is None:
            return

        self.cancel_requested.clear()
        self._set_busy(True)
        self.cancel_button.configure(state="normal")
        self.progress.configure(mode="determinate", maximum=len(self.audio_files), value=0)
        self._log(f"Transcribing {len(self.audio_files)} file(s) to {out_file}")

        self.last_output_dir = os.path.dirname(os.path.abspath(out_file))
        self._save_settings()
        options = {
            "language": LANGUAGES[self.language.get()],
            "vad_filter": self.skip_silence.get(),
            "hotwords": self.vocabulary.get().strip() or None,
        }
        self._in_background(self._run, list(self.audio_files), out_file, self.selected_model, device, options)

    def _cancel(self):
        self.cancel_requested.set()
        self.status.set("Cancelling after the current file...")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---- worker thread

    @staticmethod
    def _load_model(model_name, device):
        from faster_whisper import WhisperModel
        # Models are downloaded up front, so loading never touches the network.
        if device == "cuda":
            _add_cuda_dll_dirs()
            return WhisperModel(model_name, device="cuda", compute_type="int8_float16",
                                local_files_only=True)
        return WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)

    @staticmethod
    def _transcribe(model, path, options):
        segments, _info = model.transcribe(path, beam_size=5, **options)
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    def _run(self, files, out_file, model_name, device, options):
        emit = self.events.put
        try:
            emit(("status", f"Loading '{model_name}' model..."))
            try:
                model = self._load_model(model_name, device)
            except Exception as e:
                if device != "cuda":
                    raise
                emit(("log", f"GPU unavailable ({e}); using CPU."))
                device = "cpu"
                model = self._load_model(model_name, device)
            emit(("log", f"Model '{model_name}' loaded on {device.upper()}."))

            with open(out_file, "w", encoding="utf-8") as out:
                for i, path in enumerate(files):
                    if self.cancel_requested.is_set():
                        emit(("log", "Cancelled."))
                        break
                    name = os.path.basename(path)
                    emit(("status", f"Transcribing {i + 1}/{len(files)}: {name}"))
                    try:
                        try:
                            text = self._transcribe(model, path, options)
                        except RuntimeError as e:
                            # Missing CUDA libraries (cuBLAS/cuDNN) only surface at inference time.
                            if device != "cuda":
                                raise
                            emit(("log", f"GPU failed ({e}); switching to CPU."))
                            device = "cpu"
                            model = self._load_model(model_name, device)
                            text = self._transcribe(model, path, options)
                    except Exception as e:
                        text = f"[ERROR: {e}]"
                        emit(("log", f"Error on {name}: {e}"))

                    if i > 0:
                        out.write("\n")
                    out.write(f"{name}\n{text}\n")
                    out.flush()
                    emit(("log", f"Done: {name}"))
                    emit(("progress", i + 1))

            emit(("finished", out_file))
        except Exception as e:
            emit(("failed", str(e)))

    # ---- UI thread

    def _poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "log":
                    self._log(value)
                elif kind == "progress":
                    self.progress.configure(value=value)
                elif kind == "models":
                    self.installed = value
                    self._refresh_model_ui()
                elif kind == "hardware":
                    self._show_hardware(value)
                elif kind == "model_progress":
                    name, mb = value
                    self.progress.configure(value=min(mb, MODEL_SIZE_MB[name]))
                    self.status.set(f"Downloading {name} model... {mb:,.0f} / {MODEL_SIZE_MB[name]:,} MB")
                elif kind == "model_downloaded":
                    self._on_model_downloaded(*value)
                elif kind == "gpu_installed":
                    self._on_gpu_installed(value)
                elif kind == "update":
                    self._show_update(value)
                elif kind == "finished":
                    self._set_busy(False)
                    self.cancel_button.configure(state="disabled")
                    self.status.set("Finished.")
                    self._log(f"Saved: {value}")
                    messagebox.showinfo("Audio2Text", f"Transcription saved to:\n{value}")
                elif kind == "failed":
                    self._set_busy(False)
                    self.cancel_button.configure(state="disabled")
                    self.status.set("Failed.")
                    self._log(f"Failed: {value}")
                    messagebox.showerror("Audio2Text", value)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_model_downloaded(self, name, ok, offline, error):
        self.progress.configure(value=0)
        if ok:
            self.installed.add(name)
            self.status.set(f"{name} model installed.")
            self._log(f"Model '{name}' installed.")
        else:
            self.status.set(f"Could not download the {name} model.")
            self._log(f"Model download failed: {error}")
            if offline:
                no_internet(f"the {name} model")
            else:
                messagebox.showerror("Download Failed", f"Could not download the {name} model.\n\n{error}")
        self._set_busy(False)

    def _on_gpu_installed(self, result):
        self._set_busy(False)
        if result == "ok":
            self._log("GPU support installed.")
            self._in_background(lambda: self.events.put(("hardware", check_hardware())))
            return
        self.hw_text.set("GPU support was not installed - using CPU.")
        self.gpu_button.pack(side="left", padx=6)
        if result == "offline":
            no_internet("GPU support")
        else:
            messagebox.showerror("Install Failed", "Could not install GPU support. See the log for details.")


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except (ImportError, tk.TclError):
        root = tk.Tk()
    Audio2TextApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
