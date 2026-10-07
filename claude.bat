@echo off
rem Windows: run from the project you want to work on:  C:\path\to\hearthwork\claude.bat
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0claude.py" %*) else (python "%~dp0claude.py" %*)
