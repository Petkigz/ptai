@echo off
REM ============================================================================
REM  PTAI - run on your PC (Windows)  -  THE ONE FILE TO RUN
REM
REM  Double-click to:
REM    1. check the Python dependencies (only slow on the first run)
REM    2. start the trading agent in PAPER mode (no real money is spent)
REM    3. open the PTAI console in your browser (only when it is ready)
REM
REM  Want to know WHY it will not start? Run this file with the argument
REM  check (or from a command prompt: run_ptai.bat check) and it prints one
REM  PASS/WARN/FAIL line per requirement instead of opening windows that die.
REM
REM  Runs directly on your system Python - no venv is created.
REM  The agent's memory lives in the data\ folder next to this file and
REM  survives restarts; it is not touched by Python or dependency updates.
REM
REM  To stop: close the "PTAI Agent (paper)" and "PTAI Console" windows
REM  (Ctrl-C inside them), and close this window.
REM
REM  Ports: this PC already uses 3000 and 8000 for another project, so the
REM  console runs on 8010 by default. Change it below if 8010 is taken.
REM
REM  There used to be several .bat files (setup / start / start_dashboard).
REM  They are gone - this file is the only one, and it does all of them.
REM ============================================================================
setlocal
cd /d "%~dp0"

REM ---------------- the things you may want to change ------------------------
set PTAI_DASHBOARD_PORT=8010
set BANKROLL=50
set INTERVAL_MIN=10
REM -------------------------------------------------------------------------

echo.
echo  === PTAI ===
echo  Console   : http://localhost:%PTAI_DASHBOARD_PORT%
echo  Agent     : PAPER mode (no real money), $%BANKROLL%, one cycle every %INTERVAL_MIN% minutes
echo.

REM ---------------- find python (system python, no venv) ---------------------
set "PYTHON="
where py >nul 2>nul && set "PYTHON=py -3"
if not defined PYTHON (
  where python >nul 2>nul && set "PYTHON=python"
)
if not defined PYTHON (
  echo [ERROR] Python was not found.
  echo         Install it from https://www.python.org/downloads/ and tick
  echo         "Add python.exe to PATH" during the install, then re-run.
  goto fail
)

REM ---------------- what the agent imports (one folder, no venv) -------------
set "PYTHONPATH=%~dp0src"

REM ---------------- optional: run_ptai.bat check -----------------------------
REM The pre-flight report. Nothing is started, nothing is written: it reads.
if /i "%~1"=="check" (
  echo Running the pre-flight check ...
  %PYTHON% -m ptai.doctor
  echo.
  pause
  exit /b 0
)

REM ---------------- dependencies (system python, installed only if missing) ---
REM This used to run pip on every double-click, which needs the internet every
REM time and can fail for reasons that have nothing to do with PTAI. The
REM imports are checked first, so a machine that is already installed starts
REM immediately and works offline.
echo Checking what is installed ...
%PYTHON% -c "import loguru, pydantic, pydantic_settings, httpx, rich, typer, fastapi, uvicorn, numpy, pandas, sklearn" >nul 2>nul
if not errorlevel 1 (
  echo All packages are present.
  goto deps_ok
)
echo Some packages are missing - installing them now (first run, needs internet) ...
%PYTHON% -m pip install --quiet --user -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Installing dependencies failed. Read the messages above.
  echo         Check the internet connection, then run: run_ptai.bat check
  goto fail
)
:deps_ok

REM ---------------- folders + optional browser --------------------------------
if not exist data mkdir data
if not exist logs mkdir logs
REM Chromium is only needed if a venue ever asks for a browser login; the
REM paper Polymarket run works without it. Best effort, never blocks startup.
%PYTHON% -m playwright install chromium >nul 2>nul
if errorlevel 1 (
  echo Note: the Chromium browser was not installed - fine for paper trading;
  echo       it is only needed later for browser-based venue logins.
)

REM ---------------- start the agent (paper mode, own window) ------------------
start "PTAI Agent (paper)" /min /d "%~dp0" cmd /k "%PYTHON% main.py run --bankroll %BANKROLL% --interval %INTERVAL_MIN%"

REM ---------------- console (own window, then wait for it) ---------------------
REM The console is the one front end: the agent's state, the money, the venue,
REM orders, activity and setup. The old diagnostic dashboard (logs, V2/V3
REM internals, backtest) is a lab tool and is NOT started here - Launching the
REM whole toolbox next to the trader is what made the product feel like a pile
REM of loose parts.
start "PTAI Console" /d "%~dp0" cmd /k "set PYTHONPATH=%~dp0src && %PYTHON% -m ptai.ui.console"

echo Waiting for the console to come up on port %PTAI_DASHBOARD_PORT% ...
set /a TRIES=0
:wait_port
%PYTHON% -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',%PTAI_DASHBOARD_PORT%));s.close()" >nul 2>nul
if not errorlevel 1 goto port_up
set /a TRIES+=1
if %TRIES% geq 90 (
  echo.
  echo [ERROR] The console did not answer on port %PTAI_DASHBOARD_PORT%
  echo         within 90 seconds. Check the "PTAI Console" window for the
  echo         error message. If the port is already used by something else,
  echo         change PTAI_DASHBOARD_PORT at the top of this file.
  goto fail
)
timeout /t 1 /nobreak >nul
goto wait_port

:port_up
echo Console is up. Opening your browser ...
start "" http://localhost:%PTAI_DASHBOARD_PORT%

echo.
echo  ============================================================
echo   PTAI is running:
echo.
echo   - Console   : http://localhost:%PTAI_DASHBOARD_PORT%  (in its own window)
echo   - Agent     : PAPER mode, one cycle every %INTERVAL_MIN% minutes
echo                  (in the minimized "PTAI Agent (paper)" window)
echo.
echo   To stop PTAI: close the "PTAI Agent (paper)" window and the
echo   "PTAI Console" window (Ctrl-C inside each). Then close this.
echo.
echo   To see what PTAI checked before starting: run_ptai.bat check
echo  ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo  ============================================================
echo   PTAI did not start. Read the message above.
echo   Common fixes:
echo     - install Python 3 from python.org (tick "Add to PATH")
echo     - close other programs using port %PTAI_DASHBOARD_PORT%
echo  ============================================================
pause
exit /b 1
