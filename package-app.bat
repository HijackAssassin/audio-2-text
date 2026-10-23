@echo off
rem ==========================================================================
rem  Audio2Text's download. Website\"2 archive this build.bat" calls:
rem      Code\package-app.bat <version> "<project folder>"
rem  ->  Website\Audio-2-Text-V<version>.rar: Audio2Text.exe + _internal at the top of the .rar (the same layout as
rem  every download before; Build\cuda is never in Website\Audio2Text\ - KEEP_OUT in project.cfg).
rem ==========================================================================
setlocal
set "VERSION=%~1"
set "WEB=%~2\Website"
set "RAR=%ProgramFiles%\WinRAR\Rar.exe"
if not exist "%RAR%" echo WinRAR was not found: %RAR% & exit /b 1
if exist "%WEB%\Audio-2-Text-V%VERSION%.rar" del /q "%WEB%\Audio-2-Text-V%VERSION%.rar"
pushd "%WEB%\Audio2Text"
"%RAR%" a -m5 -idq "%WEB%\Audio-2-Text-V%VERSION%.rar" Audio2Text.exe _internal
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
