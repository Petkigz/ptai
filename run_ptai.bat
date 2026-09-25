@echo off
REM ============================================================================
REM  PTAI - run on your PC (Windows)  -  THE ONE FILE TO RUN
REM
REM  Double-click to:
REM    1. create a clean Python environment (.venv) on first run
REM    2. install dependencies (only slow on the first run)
REM    3. start the trading agent in PAPER mode (no real money is spent)
REM    4. open the product dashboard in your browser
REM
REM  To stop: close the "PTAI Agent (paper)" window (or Ctrl-C inside it),
REM  and close this window.
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
  pause
  exit /b 1
)

REM ---------------- environment (first run only) ------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a clean Python environment in .venv ...
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create .venv. Check that "venv" is available
    echo         (on Windows: python.org installers include it).
    pause
    exit /b 1
  )
)
call ".venv\Scripts\activate.bat"

echo Installing dependencies (only slow on the first run) ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Installing dependencies failed. Read the messages above.
  pause
  exit /b 1
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
start "PTAI Agent (paper)" /min "%~dp0.venv\Scripts\python.exe" main.py run --bankroll %BANKROLL% --interval %INTERVAL_MIN%

REM ---------------- dashboard (this window) -----------------------------------
set PYTHONPATH=%CD%\src
timeout /t 2 /nobreak >nul
start "" http://localhost:%PTAI_DASHBOARD_PORT%
python -m ptai.dashboard

echo.
echo Dashboard stopped. The agent window keeps the agent running.
pause
endlocal
