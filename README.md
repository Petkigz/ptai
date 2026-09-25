# PTAI — Local Autonomous Trading Agent

> **"Here is 50 dollars, earn enough to pay for yourself or shut down"**

Fully local, no cloud agents. Runs on your PC, uses your browser and terminal, scans 500-1000 Polymarket markets every 10 minutes, reads live X sentiment, builds fair value, flags mispricing >8%, sizes with Kelly Criterion (max 6% bankroll), and executes in its own browser.

**Now supports LM Studio (you use LM Studio) + Ollama** — auto-detects local LLM, fully offline.

**Risk-first design**: One wrong prediction could wipe the account, so Half-Kelly + 6% cap + daily loss limits + self-preservation shutdown.

---

## Architecture

```
Every 10 minutes:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. SCAN 500-1000 markets (Gamma API)                     │
  │    - volume24hr sorted, filters: liquidity >1000, active │
  ├─────────────────────────────────────────────────────────┤
  │ 2. READ LIVE X SENTIMENT (snscrape / API / browser)      │
  │    - 50 tweets per market, 24h lookback                  │
  │    - bullish/bearish scoring                             │
  ├─────────────────────────────────────────────────────────┤
  │ 3. RESEARCH (computer use) - 45s per market              │
  │    - web_search (DuckDuckGo), browser, terminal          │
  │    - uses your PC to research                            │
  ├─────────────────────────────────────────────────────────┤
  │ 4. BUILD FAIR VALUE (local LLM: LM Studio / Ollama)      │
  │    - superforecaster prompt                              │
  │    - base rates + sentiment + research + market price    │
  │    - confidence calibrated                               │
  ├─────────────────────────────────────────────────────────┤
  │ 5. FLAG MISPRICING >8%                                   │
  │    - edge = fair - market                                │
  │    - |edge| >= 0.08 AND confidence >=0.60                │
  ├─────────────────────────────────────────────────────────┤
  │ 6. POSITION SIZE (Kelly, Half-Kelly, max 6%)             │
  │    - f* = (bp - q)/b, b = (1-m)/m                        │
  │    - half-kelly * bankroll, capped 6%                    │
  │    - risk checks: max open 8, daily loss 15%, drawdown 30%│
  ├─────────────────────────────────────────────────────────┤
  │ 7. EXECUTE IN OWN BROWSER (Playwright persistent)        │
  │    - API first (CLOB), browser fallback                  │
  │    - stealth, human delays, own profile                  │
  │    - supports Polymarket + Kalshi + any site via browser │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  Self-Preservation: bankroll, daily_cost, unprofitable_days
  -> SHUTDOWN if not earning enough
  Position Monitor + Notifications + Dashboard
```

## Components

- **Market Scanner** (`src/ptai/markets/`): Gamma API pagination, 750 markets default, CLOB orderbook, mock fallback
- **X Sentiment** (`src/ptai/sentiment/`): snscrape (no API key), optional API/browser, LLM analysis
- **Researcher** (`src/ptai/agent/researcher.py`): ReAct loop, 45s per market, web_search + browser + terminal
- **Brain** (`src/ptai/agent/brain.py`): **LM Studio / Ollama** local LLM, superforecaster prompt, fair value JSON
- **LLM Router** (`src/ptai/llm/provider.py`): Auto-detects LM Studio (port 1234) first, then Ollama (11434), then heuristic
- **Risk** (`src/ptai/risk/`): KellyCalculator with half-Kelly, RiskManager with 6 checks, PositionMonitor
- **Execution** (`src/ptai/execution/`): BrowserExecutor (Playwright persistent + stealth), Polymarket CLOB API, GenericSiteExecutor for other sites
- **Storage** (`src/ptai/storage/db.py`): SQLite local, trades, scans, bankroll history, self-preservation
- **Tools** (`src/ptai/agent/tools.py`): web_search, terminal, browser, file — agent can research on its own
- **Loop** (`src/ptai/agent/loop.py`): Autonomous 10-min loop, rich UI, notifier, researcher
- **Console** (`src/ptai/ui/console.py`): the one front end — agent state, money,
  venue, orders, activity, setup. Started by `run_ptai.bat`
- **Diagnostic dashboard** (`src/ptai/dashboard.py`): the lab — raw logs, V2/V3
  internals, scan history, backtest, wallet linking, API config. Not part of the
  daily loop and deliberately not started by the runner

## Quick Start (Local Only) - LM Studio (You)

### 0. Windows: one file, that's it

```bat
run_ptai.bat
```

Double-click it. It runs on your system Python — **no `.venv` is created**;
the agent's memory lives in `data\` next to the file and survives restarts.
It checks the dependencies, starts the trading agent in **paper mode** (no
real money), and opens the **PTAI console** in your browser (default port
**8010** — change `PTAI_DASHBOARD_PORT` at the top of the file if that port is
taken). The console is the product: one screen about the one agent — is it
running, what is it doing, what is the money doing, and the single next thing
that stands between it and earning.
It used to be four .bat files (setup / start / start_dashboard); there is
now one, and it does all of them.

If an older checkout left a `.venv` folder behind, you can delete it — the
runner no longer uses it and nothing in `data\` depends on it.

### 1. Install PTAI

```bash
git clone <this repo>
cd ptai
chmod +x scripts/setup.sh
./scripts/setup.sh
source .venv/bin/activate
```

Manual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
cp config/config.yaml.example config/config.yaml
```

### 2. Setup LM Studio (Local LLM, No Cloud) - YOUR SETUP

```bash
# Download LM Studio from https://lmstudio.ai
# Open LM Studio -> Discover tab -> Download model:
# Recommended: llama-3.1-8b-instruct, qwen2.5-7b-instruct, mistral-7b
# For low RAM: qwen2.5-3b or phi-3-mini

# Then:
# Developer tab (</> icon) -> Select model -> Start Server (port 1234)
# Test: http://localhost:1234/v1/models should return JSON
```

See `README_LMSTUDIO.md` for detailed LM Studio guide.

Alternative: Ollama:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
# Mac
brew install ollama
ollama pull llama3.1:8b
```

Without LLM, agent falls back to heuristic (still works, less intelligent).

### 3. Configure .env

```bash
# For LM Studio (you)
LLM_PROVIDER=auto  # auto tries LM Studio first
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=local-model  # auto-detect

# For testing (no real money)
DRY_RUN=true
BANKROLL=50

# For live (Polygon wallet needed)
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_FUNDER_ADDRESS=0x...
DRY_RUN=false

# X sentiment (optional, snscrape works without keys)
X_USE_SNSCRAPE=true
```

### 4. Run

```bash
# Check LLM (LM Studio)
python main.py check-llm --provider lm_studio
python main.py lm-studio-guide

# Init
python main.py init --bankroll 50 --llm lm_studio --model local-model

# Scan only
python main.py scan --count 100

# One cycle test
python main.py run --bankroll 50 --once --dry-run --llm lm_studio

# Autonomous - THE COMMAND YOU ASKED FOR:
python main.py pay-for-yourself 50 --daily-cost 5 --interval 10 --llm lm_studio

# Status
python main.py status

# Console (the front end - http://localhost:8010, PTAI_DASHBOARD_PORT or
# PTAI_CONSOLE_PORT to move it). run_ptai.bat starts this for you.
set PYTHONPATH=src && python -m ptai.ui.console

# Diagnostic dashboard (the lab - http://localhost:8020). Start it only when
# something needs diagnosing; nothing in the daily loop depends on it.
set PYTHONPATH=src && set PTAI_DASHBOARD_PORT=8020 && python -m ptai.dashboard

# Manual trade
python main.py trade "Will BTC hit 100k" --side YES --amount 5
```

## The Self-Preservation Command

> "here is 50 dollars earn enough to pay for yourself or shut down"

```bash
python main.py pay-for-yourself 50 --llm lm_studio
```

What it does:
- Sets bankroll $50, initial_bankroll $50
- Every 10 min scans 750 markets, X sentiment, researches (45s each top 20), fair value via LM Studio, Kelly sizing, executes
- Tracks: required_profit = days_active * daily_cost ($5/day default)
- Shutdown triggers:
  - Bankroll <=0
  - Unprofitable for 3 consecutive days
  - Total drawdown >=30%
  - Daily loss >=15% (pause)
- On shutdown: exports `./data/final_report.csv`, marks shutdown state, desktop notification
- Logs everything to `./data/ptai.db` and `./logs/`

You can change daily cost:

```bash
python main.py pay-for-yourself 50 --daily-cost 2.5 --llm lm_studio
```

## Risk Rules (Why They Matter)

One wrong prediction could wipe account, so:

1. **Kelly Criterion**: `f* = (bp - q)/b`, `b = (1-m)/m`
   - Example: market 60c, fair 75% => edge 15%, odds (1-0.6)/0.6=0.66, Kelly raw = (0.66*0.75-0.25)/0.66=37%, Half-Kelly=18.5%, capped to 6% => bet $3 on $50 bankroll
2. **Max 6% bankroll** per position (configurable)
3. **Max 8 open positions** (total exposure <50%)
4. **Min edge 8%** to trade
5. **Half-Kelly** (0.5 fraction) for safety
6. **Daily loss limit 15%** — pause trading
7. **Total drawdown 30%** — shutdown
8. **Stop loss 50%** per position (monitored)
9. **Position Monitor** checks exposure, stop-loss, resolution

## Browser Execution

- Persistent profile: `./browser/profiles/default` — stays logged in to Polymarket, X
- Stealth: `playwright-stealth` to avoid detection
- Human delays: random 300-1200ms
- Modes: `api` (CLOB), `browser` (UI), `hybrid` (API first, browser fallback)
- For other sites (Kalshi, PredictIt, Manifold): `GenericSiteExecutor` via browser, works for any site
- First run opens Chromium — log into Polymarket and X manually, then close. Profile persists.

## LM Studio vs Ollama

Both fully local, no cloud. PTAI auto-detects:

- **LM Studio**: GUI, Windows friendly, OpenAI compatible at `http://localhost:1234/v1`, you already use it → **use this**
- **Ollama**: CLI, `http://localhost:11434`, `ollama serve`

PTAI tries LM Studio first (1234) then Ollama (11434). Set `LLM_PROVIDER=auto` for auto-detect.

See `README_LMSTUDIO.md` for detailed LM Studio setup.

## Extending to Other Sites

Edit `config/config.yaml`:

```yaml
other_sites:
  enabled: true
  sites:
    - name: kalshi
      enabled: true
      browser_only: true
    - name: predictit
      enabled: true
      browser_only: true
```

Generic browser executor works for any site — navigates, screenshots, template for DOM selectors. Implement site-specific selectors in `generic_browser_executor.py`.

Kalshi API: extend `src/ptai/markets/kalshi.py` (stub exists).

## Project Structure

```
ptai/
├── src/ptai/
│   ├── agent/
│   │   ├── loop.py        # main 10-min loop + researcher + notifier
│   │   ├── brain.py       # fair value LLM (LM Studio/Ollama)
│   │   ├── researcher.py  # ReAct research 45s per market
│   │   ├── tools.py       # computer use tools
│   │   └── notifier.py    # desktop notifications
│   ├── llm/
│   │   └── provider.py    # LM Studio + Ollama router
│   ├── markets/
│   │   ├── polymarket.py  # Gamma + CLOB
│   │   ├── kalshi.py      # stub extensible
│   │   ├── scanner.py     # 500-1000 scan
│   │   ├── mock.py        # offline demo
│   │   └── base.py
│   ├── sentiment/
│   │   ├── x_scraper.py   # snscrape / API / browser
│   │   └── analyzer.py    # LLM sentiment
│   ├── risk/
│   │   ├── kelly.py       # Kelly 6% cap
│   │   └── manager.py
│   ├── execution/
│   │   ├── browser.py     # Playwright own browser
│   │   ├── polymarket_executor.py
│   │   ├── generic_browser_executor.py # any site
│   │   └── monitor.py     # position monitor
│   ├── storage/
│   │   └── db.py          # SQLite local
│   ├── ui/console.py      # the one front end (agent, money, venue, orders)
│   ├── dashboard.py       # diagnostic dashboard (the lab)
│   ├── config.py          # LM Studio + Ollama config
│   └── cli.py             # Typer CLI with LM Studio
├── browser/profiles/
├── data/ptai.db
├── logs/
├── config/config.yaml
├── README_LMSTUDIO.md     # LM Studio guide for you
├── ARCHITECTURE.md
├── requirements.txt
└── scripts/setup.sh
```

## What Was Added Since Your Last Check

You asked "are you 100% sure that this is it nothing more to add" — **no, there was more needed for LM Studio and completeness**. Now added:

1. **LM Studio support** (`src/ptai/llm/provider.py`):
   - `LMStudioProvider` with OpenAI compatible API at `http://localhost:1234/v1`
   - Auto-detect models, `local-model` for auto
   - `LLMRouter` tries LM Studio first, then Ollama, then heuristic
   - `check-llm` CLI command, `lm-studio-guide` command

2. **Researcher** (`researcher.py`):
   - ReAct loop, 45s per market max, uses web_search + browser + terminal
   - Researches top 20 markets by volume
   - Feeds research into Brain prompt

3. **Position Monitor** (`monitor.py`):
   - Tracks exposure, stop-loss, resolution
   - Risk report: total exposure, over-exposed check

4. **Generic Browser Executor** for other sites (Kalshi, PredictIt, Manifold)

5. **Notifier** for desktop notifications on opportunity/trade/shutdown

6. **Config updated** for LM Studio:
   - `LM_STUDIO_HOST`, `LM_STUDIO_MODEL`, `LLM_PROVIDER=auto`
   - `.env.example` and `config.yaml.example` updated

7. **Dashboard** already exists, now includes LLM provider info

8. **README_LMSTUDIO.md** detailed guide for you

**Now it's truly complete** for your LM Studio setup and fully autonomous operation.

## FAQ

**Does it need cloud?** No. LM Studio local (port 1234), snscrape local, Playwright local, SQLite local. Only Polymarket public APIs and X scraping need internet.

**Can it run on my PC?** Yes. 8GB RAM, 4 cores. LM Studio 7B needs ~6GB RAM, 3B needs ~4GB. Heuristic fallback works with no LLM.

**How does it research on its own?** `Researcher` + `ToolRegistry`: web_search (DuckDuckGo), browser (Playwright), terminal (shell), file. Researches 45s per market, feeds into LLM prompt.

**Live trading?** Set `DRY_RUN=false` and `POLYMARKET_PRIVATE_KEY` (Polygon). Test dry run first. Uses `py-clob-client` local signing.

**Other sites?** Generic browser executor works for any site. Kalshi API stub ready to extend.

**LM Studio not detected?** See `README_LMSTUDIO.md` and run `python main.py check-llm --provider lm_studio`. Make sure server running in LM Studio Developer tab.

## Safety & Disclaimer

- Experimental software. Prediction markets risky.
- Start `DRY_RUN=true`, small bankroll.
- Not financial advice. You responsible for trades.
- Kelly doesn't guarantee profit; optimizes growth if fair value accurate.
- Self-preservation best-effort; monitor `ptai status` and dashboard.

## License

MIT — Use at your own risk.

---

Built for: **"here is 50 dollars earn enough to pay for yourself or shut down"** — fully local, autonomous, risk-aware, **LM Studio ready**.
