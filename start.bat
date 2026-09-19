@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  PTAI - START TRADING - 48GB RAM ULTIMATE - BEST CONFIG
echo  "Here is 50 dollars earn enough to pay for yourself or shut down"
echo  Best models: Qwen2.5-72B / Llama-70B / DeepSeek-R1-70B
echo ============================================================
echo.

if not exist .venv\Scripts\activate.bat (
    echo ERROR: .venv not found! Run setup.bat first!
    pause
    exit /b 1
)

echo [1/4] Activating venv...
call .venv\Scripts\activate.bat
echo Venv OK
echo.

echo [2/4] Checking LM Studio at http://localhost:1234...
curl -s http://localhost:1234/v1/models >nul 2>&1
if %errorlevel% equ 0 (
    echo LM Studio running - GREAT! Models:
    curl -s http://localhost:1234/v1/models
    echo.
    echo YOU HAVE (from log): deepseek-r1-distill-qwen-32b, qwen/qwen3-32b, qwen/qwen3.8-27b
    echo   R1 32B = 7-8 MIN per market SLOW due to reasoning!
    echo   qwen3-32b = 2-3 SEC per market FAST - BEST FOR TRADING!
    echo   RECOMMEND: unload R1, load qwen/qwen3-32b for 10-min cycle
    echo.
    echo BEST FOR 48GB:
    echo   qwen/qwen3-32b YOU HAVE - FASTEST 3 sec - BEST FOR TRADING
    echo   Qwen2.5-72B Q4_K_M (42GB) - BEST OVERALL 12 sec
    echo   Llama-3.1-70B Q4_K_M (40GB) - BEST STABLE 15 sec
    echo   DeepSeek-R1-70B Q4_K_M (40GB) - BEST REASONING but SLOW 8 min
    echo   Qwen2.5-32B Q8_0 (34GB) - BEST SPEED+QUALITY 6 sec
    echo.
) else (
    echo WARNING: LM Studio NOT running at http://localhost:1234
    echo PTAI will use heuristic fallback (works, but smarter with LM Studio)
    echo.
    echo TO FIX: Open LM Studio -^> Developer -^> Start Server (port 1234)
    echo Continuing in 5 seconds...
    timeout /t 5 >nul
)
echo.

echo [3/4] Config check...
findstr BANKROLL .env
findstr LLM_PROVIDER .env
findstr SCAN_MARKETS_COUNT .env
findstr MAX_DEEP_ANALYZE .env
findstr SENTIMENT_USE_X .env
echo.

echo [4/4] Starting PTAI - 48GB ULTIMATE - X + R1 FIX APPLIED...
echo.
echo ============================================================
echo  CONFIG FOR 48GB RAM - ULTIMATE + X + R1 FIX:
echo  - YOU HAVE: deepseek-r1-distill-qwen-32b (SLOW 8 min), qwen/qwen3-32b (FAST 3 sec)
echo  - BEST FOR TRADING: qwen/qwen3-32b - 2-3 sec, 50 deep = 2.5 min total!
echo  - R1 WARNING: R1 is 7-8 MIN per market due to think chain!
echo  - R1 Fix: Auto-detect R1 -^> MAX_DEEP_ANALYZE=5 not 30 (5*8 min=40 min still slow)
echo  - RECOMMENDED: In LM Studio unload R1, load qwen/qwen3-32b
echo  - Scan: 500 markets every 10 min (1.2 sec OK!)
echo  - Deep: 30 for 70B, 50 for 32B fast, 5 for R1 slow
echo  - Sentiment: 60 with circuit breaker, or 0 with SENTIMENT_USE_X=false
echo  - X FIX: After 3 fails circuit opens 10 min, fast skip 40 sec not 21 min
echo  - Risk: Kelly Half 0.5, max 6%% bankroll, min edge 8%%
echo  - Max open: 6 positions (36%% max exposure)
echo  - Self-preservation: Earn $5/day or shutdown
echo ============================================================
echo.
echo CRITICAL FIXES APPLIED:
echo  1. X blocking 404 - circuit breaker - 21 min -^> 40 sec
echo  2. R1 slowness - auto-detect R1 -^> MAX_DEEP_ANALYZE=5, max_tokens=500
echo  3. Prompt shortened - no think tag, direct JSON for speed
echo.
echo YOUR LOG SHOWS OLD CODE - YOU MUST PULL:
echo   git pull origin arena/01a0b42e-ptai
echo   Then restart. Log shows 100 markets sentiment (old) not 60 (new)
echo   And 30 deep for R1 should be 5 deep with warning
echo.
echo RECOMMENDED .env FOR FAST TRADING (fits 10 min):
echo   SENTIMENT_USE_X=false
echo   MAX_DEEP_ANALYZE=50
echo   LM_STUDIO_MODEL=qwen/qwen3-32b
echo   Then: 500 scan + 0 sentiment + 50 deep*3 sec = 4 min total!
echo.
echo IF YOU KEEP R1 (slow):
echo   SENTIMENT_USE_X=false
echo   MAX_DEEP_ANALYZE=5
echo   SCAN_INTERVAL_MINUTES=60 (hourly not 10 min)
echo   5 * 8 min = 40 min per cycle
echo.
echo Starting: 50 dollars - earn enough to pay for yourself or shut down
echo Command: python main.py pay-for-yourself 50 --llm lm_studio --daily-cost 5 --interval 10
echo Dashboard: http://localhost:8000 (run start_dashboard.bat in 2nd terminal)
echo Press Ctrl+C to stop
echo.

python main.py pay-for-yourself 50 --llm lm_studio --daily-cost 5 --interval 10

echo.
echo ============================================================
echo  PTAI Stopped
echo  Status: python main.py status
echo  Logs: logs\ptai.log
echo  Data: data\ptai.db
echo ============================================================
pause
