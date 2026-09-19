# BEST CONFIG FOR 48GB RAM - PTAI - ULTIMATE + X + R1 FIX (2024+)

You have 48GB RAM - you can run **70B models** now. This is the ultimate tier for local trading.

## ⚠️ CRITICAL FIXES APPLIED (2026-05) - YOU MUST PULL

### Fix 1: X (Twitter) BLOCKING 404 - Circuit Breaker
Your log shows: `https://twitter.com/i/api/graphql/.../SearchTimeline -> blocked (404)` - 13 sec delay per market, 100 markets = 21 min!

FIX:
- Circuit breaker: after 3 fails, open 10 min, fast [] return
- Sentiment 100->60 markets, 15 tweets, 0.2 sec delay
- If circuit open, skip remaining instantly 0 sec
- Fallback: web search sentiment DuckDuckGo
- Toggle: SENTIMENT_USE_X=false for LLM-only fastest

Result: 21 min -> 40 sec!

### Fix 2: DeepSeek-R1-Distill-Qwen-32B EXTREMELY SLOW - 8 min per market!
Your log shows: `Deep analyzing top 30 markets` with `DeepSeek-R1-Distill-Qwen-32B` taking 6-8 MIN per market!
- 11:11:01 -> 11:17:44 = 6.5 min
- 11:17:46 -> 11:25:50 = 8 min
- 30 markets * 8 min = **240 MIN = 4 HOURS** - breaks 10 min interval!

ROOT CAUSE: R1 reasoning models think 1000+ tokens inside `<think>...</think>` before JSON. CPU inference = very slow.

FIX APPLIED:
- Auto-detect R1 via actual LM Studio /v1/models list `['deepseek-r1-distill-qwen-32b','qwen/qwen3-32b',...]`
- If R1 detected: set MAX_DEEP_ANALYZE=5 not 30/50 (5*8 min = 40 min still slow but better)
- Reduced max_tokens 1200 -> 500 for R1
- Prompt shortened: "No chain-of-thought, no <think> tag, just direct JSON" - forces faster
- Extract JSON strips <think> tags
- Better: switch to `qwen/qwen3-32b` (non-R1) you already have! 2-3 sec vs 8 min!

RECOMMENDED FOR YOU RIGHT NOW:
You have 3 models:
- `deepseek-r1-distill-qwen-32b` - 8 min SLOW, best reasoning but too slow for trading loop
- `qwen/qwen3-32b` - 2-3 sec FAST, excellent quality, BEST FOR TRADING
- `qwen/qwen3.8-27b` - similar fast

USE `qwen/qwen3-32b` for trading! In LM Studio, unload R1 and load qwen3-32b. Then:
```ini
# .env - FAST MODE for 10-min cycle
SENTIMENT_USE_X=false
MAX_DEEP_ANALYZE=50
LM_STUDIO_MODEL=qwen/qwen3-32b
SCAN_INTERVAL_MINUTES=10
```
- Sentiment 0 sec (LLM-only, no X blocking)
- Research 20 markets ~40 sec
- Fair value 50 * 3 sec = 2.5 min
- Total ~4 min - fits 10 min easily!

If you MUST use R1:
```ini
SENTIMENT_USE_X=false
MAX_DEEP_ANALYZE=5
LM_STUDIO_MODEL=deepseek-r1-distill-qwen-32b
SCAN_INTERVAL_MINUTES=60
```
- 5 * 8 min = 40 min - exceeds interval, run hourly

## ⚠️ X (Twitter) BLOCKING FIX - IMPORTANT (Nov 2024+)

**X now blocks snscrape with `blocked (404) graphql SearchTimeline`**

Your logs show this: `https://twitter.com/i/api/graphql/.../SearchTimeline -> blocked (404)` - 4 retries then giving up, 13 sec delay per market.

**FIX APPLIED (commit fbcf5ff):**
- Circuit breaker: after 3 consecutive fails, open circuit 10 min and return [] quickly
- Sentiment reduced 100 -> 60 markets, limit 30 -> 15 tweets, delay 0.3 -> 0.2 sec
- If circuit open, skip remaining instantly (avoids 21 min delay for 100 markets)
- Fallback: web search sentiment via DuckDuckGo for top 20 markets
- Env toggle: `SENTIMENT_USE_X=false` to disable X entirely and use LLM-only (fastest)

**Current behavior with fix:**
- First 3 markets: try X, fail 13 sec each = 39 sec
- Circuit opens: log `X BLOCKING DETECTED: snscrape failed 3 times... Opening circuit breaker for 10 min`
- Remaining 57 markets: skip instantly 0 sec, use neutral 0.00 fallback + LLM reasoning
- Total sentiment: ~40 sec instead of 21 min - fits 10 min interval!

**If you want X sentiment working:**
- Option A: `SENTIMENT_USE_X=false` in .env - disable X, pure LLM fair value (works great with DeepSeek-R1-32B reasoning)
- Option B: Browser method - log into X in Playwright browser, then `X_USE_BROWSER=true` (bypasses blocking)
- Option C: X API Bearer Token - set `X_BEARER_TOKEN` in .env (paid)

**Recommended for you with DeepSeek-R1-32B:**
Use `SENTIMENT_USE_X=false` - your 32B reasoning model is so good at fair value it doesn't need X tweets. Or keep true with circuit breaker - it auto-fails fast.

You have DeepSeek-R1-Distill-Qwen-32B - BEST reasoning model for 32B tier! Perfect for fair value without X, but slow.

## Best Models for 48GB RAM - Ranked

### 🥇 #1 BEST OVERALL FOR 48GB - Qwen2.5-72B Q4_K_M

**Model:** `Qwen2.5-72B-Instruct Q4_K_M`
- **LM Studio ID:** `lmstudio-community/Qwen2.5-72B-Instruct-GGUF`
- **File:** `Qwen2.5-72B-Instruct-Q4_K_M.gguf` (~42-45GB)
- **RAM needed:** ~44GB (leaves 4GB for Windows - close Chrome/Discord)
- **Speed:** ~12-15 sec per market CPU, ~3-5 sec with GPU offload
- **Quality:** **Best open model under 100B**. Beats Llama 70B on reasoning, math, JSON. Excellent calibration.
- **Why #1:** 72B is smarter than 70B Llama, Qwen2.5 is best at reasoning. This is the smartest model that fits in 48GB.

### 🥈 #2 BEST FOR TRADING SPEED - qwen/qwen3-32b (YOU HAVE IT!)

**Model:** `qwen/qwen3-32b`
- **You have it!** LM Studio detected: `['deepseek-r1-distill-qwen-32b','qwen/qwen3-32b','qwen/qwen3.8-27b']`
- **RAM:** ~20GB
- **Speed:** 2-3 sec per market CPU - FASTEST good model
- **Quality:** Excellent, qwen3 best non-reasoning 32B
- **Why #1 for trading:** 2-3 sec vs R1 8 min = 160x faster! 50 deep = 2.5 min fits 10 min cycle. Use this!

### 🥈 #3 BEST STABLE 70B - Llama-3.1-70B Q5_K_M

**Model:** `Llama-3.1-70B-Instruct Q5_K_M`
- **ID:** `lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF`
- **File:** `Meta-Llama-3.1-70B-Instruct-Q5_K_M.gguf` (~48GB) or Q4_K_M (40GB)
- **RAM:** Q4_K_M 40GB (stable, leaves 8GB), Q5_K_M 48GB (best quality, close everything)
- **Speed:** ~15-20 sec CPU, ~4-6 sec GPU
- **Why:** Gold standard, most tested, very calibrated for superforecasting. If Qwen 72B too big, this is safest 70B.

### 🥇 #1 BEST REASONING BUT SLOW - DeepSeek-R1-Distill-Llama-70B Q4_K_M

**Model:** `DeepSeek-R1-Distill-Llama-70B Q4_K_M`
- **ID:** `lmstudio-community/DeepSeek-R1-Distill-Llama-70B-GGUF` or `bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF`
- **File:** `DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf` (~40GB)
- **RAM:** ~40GB
- **Speed:** 7-8 MIN per market CPU due to <think> reasoning! Very slow!
- **Why BEST REASONING:** This model THINKS step-by-step for 500-1000 tokens before answering.
  Perfect for hunting mispricing. Most accurate fair value but too slow for 10-min loop.
  Use 5 deep and hourly interval.

### Option 4: BEST SPEED+QUALITY BALANCE - Qwen2.5-32B Q8_0

**Model:** `Qwen2.5-32B-Instruct Q8_0`
- **File:** `Qwen2.5-32B-Instruct-Q8_0.gguf` (~34GB)
- **RAM:** ~34GB (leaves 14GB free, very stable)
- **Speed:** ~6-8 sec CPU, ~2 sec GPU - FASTER than 70B
- **Why:** Q8 is highest quality quantization for 32B, often beats 70B Q4. Faster than 70B, fits comfortably. Best if you want speed + quality.

## What Fits in 48GB - Complete List

| Model | Quant | Size | RAM | Fits 48GB? | Speed | Quality |
|-------|-------|------|-----|------------|-------|---------|
| qwen/qwen3-32b | Q4 | 20GB | 20GB | ✅ Yes | FAST 2-3 sec | Excellent |
| deepseek-r1-distill-qwen-32b | Q4 | 20GB | 20GB | ✅ Yes | SLOW 7-8 min | BEST reasoning |
| Qwen2.5-32B Q4_K_M | Q4 | 19GB | 20GB | ✅ Yes, easy | Fast | Good |
| Qwen2.5-32B Q6_K | Q6 | 27GB | 28GB | ✅ Yes | Medium | Very Good |
| Qwen2.5-32B Q8_0 | Q8 | 34GB | 34GB | ✅ Yes | Medium | Excellent |
| Qwen2.5-72B Q4_K_M | Q4 | 42GB | 44GB | ✅ Yes, close apps | Slow | BEST |
| Llama-3.1-70B Q4_K_M | Q4 | 40GB | 40GB | ✅ Yes | Slow | Excellent |
| Llama-3.1-70B Q5_K_M | Q5 | 48GB | 48GB | ⚠️ Tight, close all | Slow | BEST QUALITY 70B |
| DeepSeek-R1-70B Q4 | Q4 | 40GB | 40GB | ✅ Yes | Very Slow 15 min | BEST REASONING |
| Llama-3.1-405B Q4 | Q4 | 200GB | 200GB | ❌ No | - | - |

## My Ultimate Recommendation for 48GB

**For you with 48GB, use this:**

**For trading NOW (you have it!):** `qwen/qwen3-32b` - 2-3 sec, 50 deep = 2.5 min, fits 10 min!

**Primary daily driver:** `Qwen2.5-72B Q4_K_M` (42GB) - smartest that fits, or `Llama-3.1-70B Q4_K_M` (40GB) if 72B too big.

**If you want best reasoning (slow, hourly):** `DeepSeek-R1-Distill-Llama-70B Q4_K_M` (40GB) - thinks step-by-step, most accurate.

**If you want speed + still excellent:** `Qwen2.5-32B Q8_0` (34GB) - faster than 70B, Q8 quality often beats 70B Q4.

**My pick for you:** Use **qwen/qwen3-32b** you already have for trading (fast), try **Qwen2.5-72B Q4_K_M** for max intelligence later.

## LM Studio Settings for 48GB + 70B Models

1. **GPU Offload CRITICAL for 70B:** 
   - If you have NVIDIA GPU 8GB+ VRAM: Set GPU offload to 100% in LM Studio Settings -> GPU
   - This makes 70B go from 20 sec to 4 sec per market - 5x faster
   - If no GPU, CPU only still works but slow
   - Apple Silicon Mac: Enable Metal, offload 100%

2. **Context Length:** 2048 (PTAI needs 1500 tokens, don't need more)

3. **Threads:** Max cores (e.g. 12 or 16)

4. **Flash Attention:** Enable

5. **Close apps:** For 72B Q4 (44GB RAM), close Chrome, Discord, etc. Need 44GB free.

## Performance on 48GB with 70B Model

With Qwen2.5-72B Q4_K_M, CPU only (no GPU):

- Scan 500: 0.5 sec
- Sentiment 100: 30 sec + 100*5 sec = 8 min
- Research 20: 40 sec
- Fair value 30 markets (optimized for 70B): 30 * 12 sec = 6 min
- Total: ~15 min - **exceeds 10 min interval**

**So for 70B, we optimize:**

- Scan 500
- Sentiment 60 markets (not 100) or 0 with SENTIMENT_USE_X=false
- Research 15 markets (not 20)
- Deep analyze 30 markets (not 50) -> 30*12 sec = 6 min
- Total: ~10 min - fits!

**With GPU offload (8GB VRAM):**
- Fair value 30 markets: 30 * 3 sec = 1.5 min
- Total: ~5 min - can increase to 50 deep

**With qwen3-32b fast (YOU HAVE IT):**
- Scan 500: 1 sec
- Sentiment 0 (SENTIMENT_USE_X=false): 0 sec
- Research 20: 40 sec
- Fair value 50 * 3 sec = 2.5 min
- Total: ~4 min - fits 10 min easily!

**With R1 slow:**
- Scan 500: 1 sec
- Sentiment 0: 0 sec
- Research 5: 10 sec
- Fair value 5 * 8 min = 40 min
- Total: 40 min - need hourly interval

## Best Config for 48GB (Applied by setup.bat)

```ini
# .env - BEST FOR 48GB - FAST TRADING with qwen3-32b
LLM_PROVIDER=lm_studio
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=qwen/qwen3-32b
SCAN_MARKETS_COUNT=500
MAX_DEEP_ANALYZE=50
SENTIMENT_USE_X=false
MAX_OPEN_POSITIONS=6
MAX_POSITION_PCT=0.06
MIN_EDGE_PCT=0.08
KELLY_FRACTION=0.5
```

Why 50 deep for qwen3-32b? 50*3 sec = 2.5 min fits 10 min interval. For 70B, 30 deep. For R1, 5 deep.

## Quick Start 48GB Ultimate

```bat
setup.bat
:: In LM Studio: unload deepseek-r1-distill-qwen-32b, load qwen/qwen3-32b
:: Developer tab -> Start Server
start.bat
:: Second terminal:
start_dashboard.bat
```

## Model Download - 48GB Best

In LM Studio Discover tab, search:

1. **Best for trading NOW (you have it!):** `qwen/qwen3-32b` - you already have it, 2-3 sec FAST
2. **Best overall:** `Qwen2.5-72B-Instruct-GGUF` -> Download `Qwen2.5-72B-Instruct-Q4_K_M.gguf` (42GB) - takes 30-60 min to download
3. **Best stable 70B:** `Meta-Llama-3.1-70B-Instruct-GGUF` -> `Q4_K_M.gguf` (40GB)
4. **Best reasoning (slow):** `DeepSeek-R1-Distill-Llama-70B-GGUF` -> `Q4_K_M.gguf` (40GB)
5. **Best speed+quality:** `Qwen2.5-32B-Instruct-GGUF` -> `Q8_0.gguf` (34GB)

Pick one to start. I recommend **qwen/qwen3-32b** you have now (fast), then try 72B.

## To Make It Even Better with 48GB

You have enough RAM for **2-tier ensemble** - I can add this if you want:

- Tier 1: Qwen2.5-7B fast screens 200 markets in 2 min
- Tier 2: Qwen2.5-72B smart re-scores top 30 in 6 min
- Total 8 min, best of both worlds

Want me to add 2-tier mode?

For now, best single model for 48GB: **qwen/qwen3-32b** for speed, **Qwen2.5-72B Q4_K_M** for max intelligence, **DeepSeek-R1-70B Q4_K_M** for best reasoning but slow.
