# Audio2Text

A Windows app that transcribes a batch of audio files into one text file. Drop in your files, press go, and get a
single transcript with every file's text - offline, on your own PC, using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). An optional GPU download makes it much faster on
NVIDIA cards.

**Download:** https://teamwolfgamers.online/audio-2-text

The app checks that page for a newer version each time it starts (pattywatty.duckdns.org is the backup address).

## Running from source

Python 3.9+ on Windows. `Audio2Text.bat` makes a `.venv`, installs `requirements.txt` and starts the app.

## Building

`build-app.bat <version> <project folder>` builds `<project folder>\Build\Audio2Text.exe` (+ `_internal\`) with
PyInstaller in `.venv`. The CUDA libraries are not bundled; the app downloads them itself ("Install GPU Support").

## More

Made by Patrick - more apps and tools at https://teamwolfgamers.online
