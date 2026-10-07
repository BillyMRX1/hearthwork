@echo off
rem Windows: run from the project you want the agent to work on:  C:\path\to\hearthwork\hearthwork.bat  [claude|codex]
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0hearthwork.py" %*) else (python "%~dp0hearthwork.py" %*)
