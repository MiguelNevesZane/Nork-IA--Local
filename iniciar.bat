@echo off
cd /d "%~dp0"
call venv_cuda\Scripts\activate.bat
python main.py
pause
