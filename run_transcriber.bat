@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    call python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing / updating dependencies...
pip install faster_whisper -q

if "%1"=="" (
    echo Usage: run_transcriber.bat ^<audio_path^> [language]
    echo Example: run_transcriber.bat test.mp3 zh
    goto :eof
)

if "%2"=="" (set "LANG=zh") else (set "LANG=%2")
python audio_transcriber.py "%~1" -l %LANG%

pause
