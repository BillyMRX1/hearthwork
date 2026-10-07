@echo off
rem Windows: download a GGUF model from Hugging Face into your models folder:  model.bat <link>
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0model.py" %*) else (python "%~dp0model.py" %*)
