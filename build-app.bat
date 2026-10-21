@echo off
rem ==========================================================================
rem  Audio2Text's build steps. build.bat (in the project folder) calls:
rem      Code\build-app.bat <version> "<project folder>"
rem  ->  Build\Audio2Text.exe (+ its _internal folder), with version.txt bundled in (the update check reads it).
rem
rem  Build\cuda (GPU support the app downloaded itself, "Install GPU Support") is kept and never packed: CUDA
rem  libraries are left out of the build (1.3 GB).
rem  Needs Python 3 (the libraries go into Code\.venv on first run).
rem ==========================================================================
setlocal
set "VERSION=%~1"
set "ROOT=%~2"
set "CODE=%ROOT%\Code"
set "BUILD=%ROOT%\Build"
set "WORK=%TEMP%\audio2text-build"
if "%VERSION%"=="" echo Run build.bat in the project folder, not this. & exit /b 1
cd /d "%CODE%"

if not exist ".venv\Scripts\python.exe" (
    echo First build: creating Code\.venv and installing the libraries ...
    python -m venv .venv || exit /b 1
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
)
".venv\Scripts\python.exe" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install pyinstaller || exit /b 1
)

rem Python 3.13.0 venvs can't find Tcl/Tk on their own (PyInstaller needs it to bundle tkinter).
for /f "delims=" %%P in ('.venv\Scripts\python.exe -c "import sys; print(sys.base_prefix)"') do set "PYBASE=%%P"
if exist "%PYBASE%\tcl\tcl8.6" set "TCL_LIBRARY=%PYBASE%\tcl\tcl8.6"
if exist "%PYBASE%\tcl\tk8.6" set "TK_LIBRARY=%PYBASE%\tcl\tk8.6"

if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%"
> "%WORK%\version.txt" echo %VERSION%
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed ^
    --name Audio2Text ^
    --distpath "%WORK%\dist" --workpath "%WORK%\work" --specpath "%WORK%" ^
    --add-data "%WORK%\version.txt;." ^
    --collect-data faster_whisper ^
    --collect-binaries ctranslate2 ^
    --collect-all tkinterdnd2 ^
    --exclude-module nvidia ^
    audio2text.py || exit /b 1

rem Replace the previous build but keep Build\cuda (downloaded GPU support).
if not exist "%BUILD%" mkdir "%BUILD%"
if exist "%BUILD%\_internal" rmdir /s /q "%BUILD%\_internal" || exit /b 1
if exist "%BUILD%\Audio2Text.exe" del "%BUILD%\Audio2Text.exe" || exit /b 1
robocopy "%WORK%\dist\Audio2Text" "%BUILD%" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 exit /b 1
rmdir /s /q "%WORK%"
exit /b 0
