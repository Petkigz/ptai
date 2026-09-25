@echo off
REM ============================================================================
REM  PTAI - run on your PC (Windows)  -  THE ONE FILE TO RUN
REM
REM  Double-click to:
REM    1. create a clean Python environment (.venv) on first run
REM    2. install dependencies (only slow on the first run)
REM    3. start the trading agent in PAPER mode (no real money is spent)
REM    4. open the product dashboard in your browser (only when it is ready)
REM
REM  To stop: close the "PTAI Agent (paper)" and "PTAI Dashboard" windows
REM  (Ctrl-C inside them), and close this window.
REM
REM  Ports: this PC already uses 3000 and 8000 for another project, so the
REM  dashboard runs on 8010 by default. Change it below if 8010 is taken.
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
echo  Dashboard : http://localhost:%PTAI_DASHBOARD_PORT%
echo  Agent     : PAPER mode (no real money), $%BANKROLL%, one cycle every %INTERVAL_MIN% minutes
echo.

REM ---------------- find python ----------------------------------------------
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

REM ---------------- environment (first run only) ------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a clean Python environment in .venv ...
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create .venv. Check that "venv" is available
    echo         (on Windows: python.org installers include it).
    goto fail
  )
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Could not activate .venv
  goto fail
)

echo Installing dependencies (only slow on the first run) ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Installing dependencies failed. Read the messages above.
  goto fail
)

REM ---------------- folders + optional browser --------------------------------
if not exist data mkdir data
if not exist logs mkdir logs
REM Chromium is only needed if a venue ever asks for a browser login; the
REM paper Polymarket run works without it. Best effort, never blocks startup.
python -m playwright install chromium >nul 2>nul
if errorlevel 1 (
  echo Note: browser (Chromium) not installed - fine for paper trading;
  echo       it is only needed later for browser-based venue logins.
)

REM ---------------- start the agent (paper mode, own window) ------------------
start "PTAI Agent (paper)" /min /d "%~dp0" cmd /k ".venv\Scripts\python.exe main.py run --bankroll %BANKROLL% --interval %INTERVAL_MIN%"

REM ---------------- dashboard (own window, then wait for it) ------------------
start "PTAI Dashboard" /d "%~dp0" cmd /k "set PYTHONPATH=%~dp0src && .venv\Scripts\python.exe -m ptai.dashboard"

echo Waiting for the dashboard to come up on port %PTAI_DASHBOARD_PORT% ...
set /a TRIES=0
:wait_port
python -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',%PTAI_DASHBOARD_PORT%));s.close()" >nul 2>nul
if not errorlevel 1 goto port_up
set /a TRIES+=1
if %TRIES% geq 90 (
  echo.
  echo [ERROR] The dashboard did not answer on port %PTAI_DASHBOARD_PORT%
  echo         within 90 seconds. Check the "PTAI Dashboard" window for the
  echo         error message. If the port is already used by something else,
  echo         change PTAI_DASHBOARD_PORT at the top of this file.
  goto fail
)
timeout /t 1 /nobreak >nul
goto wait_port

:port_up
echo Dashboard is up. Opening your browser ...
start "" http://localhost:%PTAI_DASHBOARD_PORT%

echo.
echo  ============================================================
echo   PTAI is running:
echo.
echo   - Dashboard : http://localhost:%PTAI_DASHBOARD_PORT%  (in its own window)
echo   - Agent     : PAPER mode, one cycle every %INTERVAL_MIN% minutes
echo                  (in the minimized "PTAI Agent (paper)" window)
echo.
echo   To stop PTAI: close the "PTAI Agent (paper)" window and the
echo   "PTAI Dashboard" window (Ctrl-C inside each). Then close this.
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
