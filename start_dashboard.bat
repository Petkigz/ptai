@echo off
echo ============================================================
echo  PTAI - PRODUCT DASHBOARD - http://localhost:8000
echo  Full UI for linking wallet, LLM setup, health checks
echo  Local only, no cloud - Product for other people
echo ============================================================
echo.

if not exist .venv\Scripts\activate.bat (
    echo ERROR: .venv not found! Run setup.bat first!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Starting PRODUCT dashboard at http://localhost:8000
echo.
echo NEW PRODUCT UI FEATURES:
echo  - Overview: bankroll, PnL, trades, health status
echo  - Onboarding: 7-step wizard auto-detected progress
echo  - Wallet: Link Polymarket account in UI (no .env editing)
echo  - AI Brain: LM Studio status, models, speed warning R1 vs fast
echo  - Trading: Controls, risk settings, run cycle
echo  - Logs: Live logs tail, is it working?
echo  - Settings: Advanced .env management
echo.
echo For other people: just run this bat, open browser, follow onboarding
echo Open browser to http://localhost:8000
echo.
echo Press Ctrl+C to stop dashboard
echo.

python run_dashboard.py

pause
