# PTAI with LM Studio - You use LM Studio

This guide is specifically for you since you use LM Studio.

## Why LM Studio?

- Fully local, no cloud
- OpenAI compatible API at `http://localhost:1234`
- Easy model management
- Works on Windows/Mac/Linux
- Better than Ollama for many users (GUI, easier)

## Setup LM Studio for PTAI

### 1. Install LM Studio

Download from https://lmstudio.ai - available for Windows, Mac, Linux.

### 2. Download a Model

Open LM Studio -> Discover tab (magnifying glass) -> Search:

**Recommended for PTAI (fair value reasoning):**
- `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` - Good balance, 8B, needs ~6GB RAM
- `lmstudio-community/Qwen2.5-7B-Instruct-GGUF` - Excellent reasoning, 7B
- `lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF` - Fast, good
- `TheBloke/Llama-2-7B-Chat-GGUF` - Older but works

**For low RAM (4-8GB):**
- `lmstudio-community/Qwen2.5-3B-Instruct-GGUF` - 3B, fast
- `microsoft/Phi-3-mini-4k-instruct-gguf` - 3.8B, very fast

**For high accuracy (if you have 32GB+ RAM or 24GB VRAM):**
- `lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF` - Best reasoning
- `lmstudio-community/Qwen2.5-32B-Instruct-GGUF`

Download one (click download icon). Wait for download.

### 3. Start Local Server

1. Go to Developer tab (left sidebar, `</>` icon)
2. Top dropdown: Select your downloaded model
3. Left panel: 
   - Check "Cross-Origin-Resource-Sharing (CORS)" if you want dashboard to work
   - Port: 1234 (default)
4. Click "Start Server" - should show "Server running on http://localhost:1234"
5. You should see logs of requests

Test in browser: http://localhost:1234/v1/models - should return JSON with models.

### 4. Configure PTAI

Copy env:

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=auto
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=local-model
```

`local-model` means auto-detect first model loaded in LM Studio. Or set specific name like `llama-3.1-8b-instruct`.

### 5. Test LLM Connection

```bash
python main.py check-llm --provider lm_studio
```

Should show:

```
LLM Status
Provider | Host | Available | Models
LM Studio | http://localhost:1234 | ✅ | ['llama-3.1-8b-instruct']
Active provider: LMStudioProvider
LLM response: {"answer": 0.42}
```

If ❌, check:
- LM Studio server running?
- Port 1234? Check http://localhost:1234/v1/models in browser
- Firewall blocking?

### 6. Run PTAI with LM Studio

```bash
# Init
python main.py init --bankroll 50 --llm lm_studio --model local-model

# One cycle test (dry run, no real money)
python main.py run --bankroll 50 --once --llm lm_studio --dry-run

# Full autonomous - THE COMMAND
python main.py pay-for-yourself 50 --llm lm_studio --daily-cost 5 --interval 10

# With live trading (real money, need private key in .env and DRY_RUN=false)
python main.py pay-for-yourself 50 --llm lm_studio --daily-cost 5
```

### 7. What PTAI does with LM Studio

Every 10 minutes:

1. Scans 750 markets from Polymarket Gamma API (public, no key)
2. Scrapes X sentiment via snscrape (local, no key)
3. For each top market, calls LM Studio:

```
POST http://localhost:1234/v1/chat/completions
{
  "model": "local-model",
  "messages": [{"role": "user", "content": "Market: Will BTC hit 100k? Price 60c... Estimate fair value..."}],
  "temperature": 0.2
}
```

LM Studio runs model locally on your GPU/CPU, returns JSON:

```json
{
  "fair_value": 0.75,
  "edge": 0.15,
  "confidence": 0.72,
  "side": "YES",
  "should_trade": true,
  "reasoning": "Base rate 60%..."
}
```

4. Filters edge >8%, calculates Kelly max 6%, executes via API or browser.

All local: LM Studio + PTAI + Polymarket API + X scraping. No OpenAI, no cloud.

### Troubleshooting LM Studio

**"No LLM detected"**

- Make sure LM Studio server is running (Developer tab -> Start Server)
- Check http://localhost:1234/v1/models in browser - should return JSON
- Try `curl http://localhost:1234/v1/models`
- Check .env LM_STUDIO_HOST matches (default http://localhost:1234)
- Restart LM Studio

**"Model not found"**

- Use `local-model` as model name for auto-detect
- Or set exact model ID from /v1/models response
- In LM Studio Developer tab, make sure model is loaded (shows in top dropdown)

**Slow responses**

- Use smaller model: qwen2.5-3b or phi-3-mini
- Reduce `max_tokens` in config to 800
- In LM Studio settings, enable GPU acceleration if you have GPU
- Close other apps to free RAM

**Out of memory**

- Use 3B model instead of 7B/8B
- In LM Studio, set context length to 2048 instead of 4096
- Close browser tabs

**Still not working - use heuristic fallback**

PTAI works even without LLM - uses heuristic that still finds mispricing via base rate reversion. LLM just makes it smarter. So you can run without LM Studio, but with LM Studio it's much better.

### LM Studio vs Ollama

Both work, PTAI auto-detects:

- **LM Studio**: GUI, easier, Windows friendly, OpenAI compatible, you already use it -> use this
- **Ollama**: CLI, lighter, good for Linux servers, needs `ollama serve`

PTAI tries LM Studio first (port 1234) then Ollama (port 11434). So if you run LM Studio server, it will use it automatically with `LLM_PROVIDER=auto`.

### Performance Tips for LM Studio

- **Model**: Qwen2.5-7B is best for reasoning/fair value, Llama 3.1 8B good too
- **Quantization**: Q4_K_M is good balance size/quality, Q5/Q6 better quality but larger
- **Context**: 2048 enough for PTAI, don't need 8192
- **GPU**: Enable GPU offload in LM Studio settings if you have NVIDIA/Apple Silicon
- **Batch**: PTAI analyzes top 200 markets per cycle, each needs one LLM call ~2-5 sec, so 200 * 3 sec = 10 min - fits in 10 min interval. If slower, reduce `scan_markets_count` to 500 or top analysis to 100.

### Example .env for LM Studio

```
BANKROLL=50
LLM_PROVIDER=lm_studio
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=local-model
DRY_RUN=true
SCAN_MARKETS_COUNT=750
MIN_EDGE_PCT=0.08
MAX_POSITION_PCT=0.06
```

That's it! PTAI + LM Studio = fully local autonomous trader.
