@echo off
cd %~dp0xtts

set pypath=home = %cd%\python
set venvpath=_ENV=%cd%\venv
if exist venv (powershell -command "$text = (gc venv\pyvenv.cfg) -replace 'home = .*', $env:pypath; $Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding($False);[System.IO.File]::WriteAllLines('venv\pyvenv.cfg', $text, $Utf8NoBomEncoding);$text = (gc venv\scripts\activate.bat) -replace '_ENV=.*', $env:venvpath; $Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding($False);[System.IO.File]::WriteAllLines('venv\scripts\activate.bat', $text, $Utf8NoBomEncoding);")

set appdata=%cd%\tmp
set userprofile=%cd%\tmp
set temp=%cd%\tmp
set path=%cd%\venv\scripts;%cd%\venv\Lib\site-packages\torch\lib;%cd%\SillyTavern-Extras\ffmpeg
call %cd%\venv\scripts\activate.bat

start /b cmd /c python -m xtts_api_server --bat-dir %~dp0 -d=cuda --deepspeed --stream-to-wavs --call-wav2lip --output SillyTavern-Extras\\tts_out\\ --extras-url http://127.0.0.1:5100/ --wav-chunk-sizes=10,20,40,100,200,300,400,9999

cd SillyTavern-Extras
start /b cmd /c python server.py  --enable-modules wav2lip

:Again
%systemroot%\system32\curl.exe http://127.0.0.1:8020/ >nul 2>nul
if %errorlevel% neq 0 (
    %systemroot%\system32\timeout.exe /t 5>nul 2>nul
    goto Again
)

cd ..
cd ..
cd TalkLlama

start "TalkLlama" cmd /c talk-llama.exe -mw ggml-medium-q5_0.bin -ml mistral-7b-instruct-v0.2.Q5_0.gguf --language ru -p "Христ" --vad-last-ms 500 --vad-start-thold 0.001 --bot-name "Глеб" --xtts-voice Глеб --prompt-file assistant_ru.txt --temp 0.3 --ctx_size 2000  --multi-chars --allow-newline --stop-words Христ:;```;---; -ngl 99 -n 80 --threads 4 --split-after 1 --sleep-before-xtts 800
