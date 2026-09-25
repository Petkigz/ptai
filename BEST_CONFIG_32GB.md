# BEST CONFIG FOR 32GB RAM - PTAI

> **Note:** the entry point is now a single file: `run_ptai.bat` (the `setup.bat` / `start.bat` / `start_dashboard.bat` files are removed).

You have 32GB RAM - this is the sweet spot for local trading agent. Here's the best possible setup.

## Best Model for 32GB RAM

### Option 1: BEST STABLE (Recommended for 32GB)

**Model:** `Qwen2.5-32B-Instruct Q4_K_M`
- **LM Studio ID:** `lmstudio-community/Qwen2.5-32B-Instruct-GGUF`
- **File:** `Qwen2.5-32B-Instruct-Q4_K_M.gguf` (19.2 GB)
- **RAM needed:** ~20-22GB (leaves 10GB for Windows + PTAI + browser)
- **Speed:** ~6-8 sec per market, 50 markets = ~6 min
- **Quality:** Excellent reasoning, great JSON, very calibrated
- **Why best:** Fits comfortably in 32GB, fast enough for 10 min interval, smarter than any 7B/8B

### Option 2: BEST QUALITY (If you close Chrome/Discord)

**Model:** `Qwen2.5-32B-Instruct Q6_K`
- **File:** `Qwen2.5-32B-Instruct-Q6_K.gguf` (27 GB)
- **RAM needed:** ~28-30GB (close other apps, need 32GB free)
- **Speed:** ~8-12 sec per market
- **Quality:** Better than Q4, best quality for 32B
- **Use if:** You close browser and other RAM heavy apps while trading

### Option 3: BEST REASONING (Chain-of-thought)

**Model:** `DeepSeek-R1-Distill-Qwen-32B Q4_K_M`
- **ID:** `lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF`
- **File:** `DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf` (19GB)
- **RAM:** ~20GB
- **Why:** This model THINKS step-by-step before answering. Shows its reasoning like:
  "Base rate 60%... Recent news... Sentiment bullish... Market at 50% seems low because... Fair value 68%"
  Perfect for prediction markets. Best for fair value accuracy.
- **Tradeoff:** Slightly slower, more verbose, but most accurate

### What NOT to use on 32GB:

- ❌ `Qwen2.5-72B` - Needs 45GB+ RAM, won't fit
- ❌ `Llama-3.1-70B Q4` - Needs 40GB, tight on 32GB, will swap and be super slow
- ❌ `Llama-3.1-405B` - Needs 200GB+, impossible
- ❌ `Qwen2.5-32B Q8_0` - Needs 34GB+, too big for 32GB

## Best Config for 32GB (Applied by setup.bat)

```ini
# .env - BEST FOR 32GB
LLM_PROVIDER=lm_studio
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=local-model
SCAN_MARKETS_COUNT=500
MAX_DEEP_ANALYZE=50
MAX_OPEN_POSITIONS=6
MAX_POSITION_PCT=0.06
MIN_EDGE_PCT=0.08
KELLY_FRACTION=0.5
BROWSER_HEADLESS=false
```

Why 500 not 750? 32B model slower, 500 markets scanning + 50 deep analyze fits in 10 min interval. 750 would take 15 min.

Why 50 deep not 200? 50 * 8 sec = 400 sec = 6.5 min, plus 500 scan + 100 sentiment + research = ~10 min total. 200 * 8 sec = 26 min too slow.

Why 6 open positions not 8? With 32B, you want lower exposure, more selective. 6 * 6% = 36% max exposure.

## LM Studio Settings for 32GB (Best)

1. Open LM Studio -> Settings (gear icon)
2. **GPU Offload:** If you have NVIDIA GPU with 8GB+ VRAM, set GPU offload to 50% or 100% - makes it MUCH faster (2 sec vs 8 sec per market)
   - If you have Apple Silicon Mac, enable Metal
   - If no GPU, CPU only is OK, just slower
3. **Context Length:** Set to 2048 (PTAI prompt is ~1500 tokens, don't need 4096)
4. **Threads:** Set to max CPU cores (e.g. 8 or 12)
5. **Flash Attention:** Enable if available

In Developer tab:
- **Temperature:** 0.2 (PTAI sets this, but you can leave default)
- **Top P:** 0.95
- **Repeat Penalty:** 1.1

## Performance Expectations on 32GB

With Qwen2.5-32B Q4_K_M on 32GB RAM, CPU only (no GPU):

- Scan 500 markets: 0.5 sec (API) or 1 sec (mock)
- X sentiment 100 markets: 30 sec (snscrape) + 100 * 2 sec LLM sentiment = 3 min
- Research 20 markets: 20 * 2 sec = 40 sec
- Fair value 50 markets: 50 * 7 sec = 5.8 min
- Kelly + execution: 5 sec
- **Total: ~10 min** - fits interval!

With GPU offload (8GB VRAM):
- Fair value 50 markets: 50 * 2 sec = 1.6 min
- **Total: ~5 min** - even better, can increase to 750 markets / 100 deep

## Best Possible Recommendations - Everything

### Hardware:
- 32GB RAM is perfect - you have it
- If you have GPU (NVIDIA 8GB+ or Apple Silicon), enable GPU offload in LM Studio - 3x faster
- SSD not HDD (models load faster)
- Good internet for Polymarket API and X scraping

### Software:
- Windows 10/11 with WSL2 optional
- Python 3.11 (best for PTAI)
- LM Studio latest version
- Chrome for Playwright browser profile (log into Polymarket once)

### Models to download in LM Studio (in order of preference):

1. **First:** `Qwen2.5-32B-Instruct Q4_K_M` - your daily driver, best stable
2. **Second:** `DeepSeek-R1-Distill-Qwen-32B Q4_K_M` - for important markets, best reasoning
3. **Optional:** `Qwen2.5-7B-Instruct Q4` - fast screening model (if you want 2-tier: 7B screens 200, 32B re-scores top 20)

### PTAI Settings for Best Results:

```bash
# In .env
BANKROLL=50
DAILY_COST_TO_COVER=5
SCAN_MARKETS_COUNT=500
MAX_DEEP_ANALYZE=50
MAX_OPEN_POSITIONS=6
MAX_POSITION_PCT=0.06
MIN_EDGE_PCT=0.08
KELLY_FRACTION=0.5
```

- Start with `DRY_RUN=true` for 1-2 days, watch logs, see if edge >8% opportunities look sensible
- Then set `DRY_RUN=false` with real Polymarket private key and small bankroll $50
- Monitor `python main.py status` and dashboard http://localhost:8000
- If profitable after 3 days, increase bankroll

### Trading Strategy Best:

- PTAI hunts mispricing >8% - this is good, filters noise
- Kelly 6% cap prevents wipeout - critical
- Half-Kelly 0.5 reduces variance - recommended by pros
- Don't override risk rules - one wrong 50% bet wipes account
- Let self-preservation shutdown if not profitable - saves money

### To Make It Even Better (Future):

- Add second model: Use Qwen2.5-7B fast to screen 200, then Qwen2.5-32B smart to re-score top 20 - I can add this 2-tier mode if you want
- Add Kalshi API for more markets
- Add stop-loss monitor loop (already in code, but needs live price feed)
- Add Telegram notifications

## Quick Start Best Config

```bat
:: First time
setup.bat
:: Then in LM Studio: Download Qwen2.5-32B Q4_K_M, Start Server
:: Then:
start.bat
:: In second terminal:
start_dashboard.bat
```

That's the best possible for 32GB RAM.

## Model Download Links in LM Studio

Search in Discover tab:

- `lmstudio-community/Qwen2.5-32B-Instruct-GGUF` -> Download `Qwen2.5-32B-Instruct-Q4_K_M.gguf` (19GB)
- `lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF` -> Download `Q4_K_M` (19GB)
- `lmstudio-community/Qwen2.5-7B-Instruct-GGUF` -> Download `Q4_K_M` (4.5GB) for fast screening

Download will take 10-30 min depending on internet (19GB file).

After download, Developer tab -> Select model -> Start Server -> Should show running on port 1234.

Done.
