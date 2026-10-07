@echo off
rem Windows: double-click, or run from a terminal. Needs Python 3.8+ (python.org or the Microsoft Store).
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0start.py" %*) else (python "%~dp0start.py" %*)
pause
