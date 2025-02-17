@echo off
cd /d "%~dp0fish-speech"
call "..\fishenv\conda\condabin\conda.bat" activate "..\fishenv\env"
python dialogue-app.py
pause
