@echo off
REM ============================================================================
REM  PTAI - run on your PC (Windows)  -  THE ONE FILE TO RUN
REM
REM  Double-click to:
REM    1. check the Python dependencies (only slow on the first run)
REM    2. open the PTAI console in your browser (only when it is ready)
REM    3. the console starts the trading agent in PAPER mode (no real money
REM       is spent) IN ITS OWN PROCESS - one window, one agent, one place
REM       where the settings, the round and the log are visible
REM
REM  Want to know WHY it will not start? Run this file with the argument
REM  check (or from a command prompt: run_ptai.bat check) and it prints one
REM  PASS/WARN/FAIL line per requirement instead of opening windows that die.
REM
REM  Runs directly on your system Python - no venv is created.
REM  The agent's memory lives in the data\ folder next to this file and
REM  survives restarts; it is not touched by Python or dependency updates.
REM
REM  To stop: press Stop in the page (the record then says the agent was
REM  stopped on purpose), or close the "PTAI" window.
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
echo              started by the console itself - one window, one agent
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
REM ---------------- ONE window: the console, which runs the agent ------------
REM There used to be TWO windows here: the agent (`main.py run --bankroll ...`)
REM and the console. That split is what the operator saw as the web page being
REM disconnected from the command line: settings typed in the page could not
REM change the loop running in the other window, the page ran an engine of its
REM own inside the web process, and the agent's output was in a window the page
REM could not show. Now this process IS the agent's host:
REM
REM   * PTAI_AGENT_AUTOSTART=1 tells it to start the loop at startup;
REM   * Start / Stop / interval / "Run a round now" all act on that one loop;
REM   * the agent's log is on the page, from the process doing the work.
REM
REM The CLI is still the same loop and still works (`main.py run`); if it is
REM already running, this console says so and names its pid instead of starting
REM a second engine over the same database.
REM
REM The old diagnostic dashboard (logs, V2/V3 internals, backtest) is a lab tool
REM and is NOT started here - launching the whole toolbox next to the trader is
REM what made the product feel like a pile of loose parts.
start "PTAI" /d "%~dp0" cmd /k "set PYTHONPATH=%~dp0src && set PTAI_AGENT_AUTOSTART=1 && set BANKROLL=%BANKROLL% && set INTERVAL_MIN=%INTERVAL_MIN% && %PYTHON% -m ptai.ui.console"

echo Waiting for the console to come up on port %PTAI_DASHBOARD_PORT% ...
set /a TRIES=0
:wait_port
%PYTHON% -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',%PTAI_DASHBOARD_PORT%));s.close()" >nul 2>nul
if not errorlevel 1 goto port_up
set /a TRIES+=1
if %TRIES% geq 90 (
  echo.
  echo [ERROR] The console did not answer on port %PTAI_DASHBOARD_PORT%
  echo         within 90 seconds. Check the "PTAI" window for the
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
echo   - Console   : http://localhost:%PTAI_DASHBOARD_PORT%  (in the "PTAI" window)
echo   - Agent     : PAPER mode, running INSIDE the console process, one
echo                  cycle every %INTERVAL_MIN% minutes to start with. The
echo                  page's interval box changes that while it runs.
echo.
echo   Stop the agent from the page (Stop), or close the "PTAI" window.
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
