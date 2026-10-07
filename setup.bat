@echo off
rem Windows: re-run setup (hardware check, llama.cpp download/update, models folder). Add --update-llama for the newest build.
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0onboard.py" %*) else (python "%~dp0onboard.py" %*)
pause
