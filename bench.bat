@echo off
rem Windows: benchmark the running model through a coding agent:  bench.bat [--agent claude|codex] [--show]
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0bench.py" %*) else (python "%~dp0bench.py" %*)
