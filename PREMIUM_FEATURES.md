# PTAI PREMIUM - AI Teammates Product - All Features Built

> "AI teammates you can give real work to. Bots can sign in to your tools, use them just like you do, and come back with finished work."

PTAI now at premium product level like Lindy / SmythOS / Relay - built for other people to use, not just you.

## ✅ Built Features - Done

### 1. 🤖 AI Teammates - Give Real Work to Bots (Core Premium)

**7 Specialized Teammates:**

| Teammate | Role | Tools Signed In | What It Does |
|----------|------|-----------------|--------------|
| 🔍 Scout | Market Scanner | gamma_api, clob_api | Scans 500-1000 markets every 10 min, filters by volume/liquidity, assigns to others |
| 🐦 Sentiment Analyst | X/Twitter & News Reader | x_snscrape, web_search, duckduckgo | Reads live X sentiment (with circuit breaker for 404 blocking), web news fallback |
| 🔬 Researcher | Deep Researcher | browser, terminal, web_search | Uses computer, browser, terminal to research top 20 markets, 45s each, ReAct loop |
| 🧠 Quant | Superforecaster | lm_studio, ollama, grok | Builds fair value, base rates, flags mispricing >8%, confidence calibrated |
| 🛡️ Risk Officer | Risk Manager | risk_manager, kelly, monitor | Kelly 6% cap prevents wipeout, Half-Kelly 0.5, max 6 positions, daily loss 15% pause |
| ⚡ Trader | Executor | polymarket_clob, browser, generic | Executes in own browser (Playwright persistent), API first then browser fallback |
| 📈 Coach | Performance Coach | memory, storage, calibration | Learns from past trades, tracks Brier score calibration, improves future fair values |

**How it works:**
```python
# Give real work to teammates
task = Task(id="scan_1", type="scan_markets", description="Scan 500 markets by volume", payload={"count": 500})
result = await scout.assign(task)  # Scout signs into gamma_api, scans, comes back with finished work

# Team collaborates
coordinator.run_full_cycle()  # Scout -> Sentiment -> Researcher -> Quant -> Risk -> Trader -> Coach
```

**CLI:**
```bash
python main.py teammates status  # Show team status table
python main.py teammates run --count 500  # Run team cycle
python main.py premium 50  # Premium mode with all teammates
```

**Dashboard:** Tab "AI Teammates" shows 7 bots, status IDLE/BUSY, tasks done, avg time, tools signed in, plus "Give Work to Team" form.

**Files:**
- `src/ptai/agent/teammates/base.py` - BaseTeammate, Task, TeammateStatus
- `src/ptai/agent/teammates/*.py` - Each teammate
- `src/ptai/agent/teammates/coordinator.py` - Orchestrates team
- `src/ptai/agent/premium_loop.py` - PremiumTradingAgent using team

### 2. 🔐 Vault - Tool Sign-In (Bots sign in like you do)

**Premium: Bots can sign in to your tools just like you do**

- Secure local vault `./data/vault.json` encrypted via `cryptography` Fernet (or XOR fallback)
- Each teammate signs into tools it needs, credentials stay local, never cloud
- Supports: polymarket_clob, kalshi, manifold, x_api, x_snscrape, discord_webhook, telegram_bot, lm_studio, ollama, grok_api, browser, web_search, terminal, etc

```python
vault = Vault()
vault.store_tool_credentials("Trader", "polymarket_clob", {"private_key": "0x...", "funder": "0x..."})
creds = vault.get_tool_credentials("Trader", "polymarket_clob")
```

**CLI:**
```bash
python main.py vault status  # Show all tool sign-ins
python main.py vault list
```

**Dashboard:** Tab "Vault" shows signed-in tools by teammate, add tool sign-in form, encrypted storage note.

**Files:** `src/ptai/vault/vault.py`

### 3. 🧠 Memory & Learning - Intelligence

**Premium: Bots learn from past work, track calibration, improve**

- SQLite `./data/memory.db` with `memories` and `calibrations` tables
- Types: task, insight, trade, calibration, research
- `remember()` - save insight, `recall()` - keyword search (can add embeddings), `remember_task()`
- `track_calibration()` - Brier score = (predicted - actual)^2, good <0.2, bad >0.3
- Coach teammate uses memory to generate insights: "Win rate low, tighten edge >10%"

```python
memory = Memory()
memory.remember("Quant found opportunity: Will BTC hit 100k? edge 15% conf 0.72", type="insight", importance=0.8)
memory.track_calibration(market_id="123", question="Will BTC...", predicted=0.75, actual=1, confidence=0.72, edge=0.15)
insights = memory.get_insights(limit=20)
```

**CLI:**
```bash
python main.py memory insights --limit 20
python main.py memory recall --query "BTC" --limit 10
python main.py memory stats
```

**Dashboard:** Tab "Memory & Learning" shows calibration stats (count, avg Brier, avg conf), insights table, recall search box.

**Files:** `src/ptai/memory/memory.py`

### 4. 📈 Backtest - Test Before Live (Premium)

**Premium: Test strategies on historical data before risking real money**

- Mock historical: 30 days * 10 markets per day with random fair value + actual outcome based on fair prob
- Kelly sizing, win rate, PnL, max drawdown, Sharpe, equity curve
- Future: real Polymarket historical data from Gamma API archive

```python
engine = BacktestEngine()
result = engine.run(strategy_config={"bankroll": 50, "min_edge": 0.08, "max_pos_pct": 0.06}, days=30)
# result: initial $50 -> final $65 PnL $15 30% win rate 55% max DD 10% Sharpe 2.1
```

**CLI:**
```bash
python main.py backtest --days 30 --bankroll 50 --edge 8
```

**Dashboard:** Tab "Backtest" has form (days, bankroll, min edge, max pos), Run Backtest button, result display with PnL, win rate, equity curve text.

**Files:** `src/ptai/backtest/engine.py`

### 5. 👥 Multi-User - Product for Other People (Premium)

**Premium: Each user isolated - own bankroll, wallet, browser profiles, memory, vault**

- SQLite `./data/users.db` with users table
- Per-user isolated paths:
  - `data/users/{id}/ptai.db` - trades, scans
  - `data/users/{id}/vault.json` - encrypted keys
  - `data/users/{id}/memory.db` - learning
  - `browser/profiles/{id}` - persistent logins (Polymarket, X)
  - `logs/users/{id}/ptai.log` - logs
- Plans: free, pro $29/mo, premium $99/mo (for billing future)

```python
manager = UserManager()
user = manager.create_user(email="user@example.com", name="John", bankroll=50, plan="pro")
paths = manager.get_user_paths(user.id)  # isolated paths
agent = PremiumTradingAgent(user_id=user.id, bankroll=50)  # uses user paths
```

**CLI:**
```bash
python main.py users list
python main.py users create --email user@example.com --name "John" --bankroll 50 --plan pro
python main.py users paths --user-id abc123
python main.py premium 50 --user abc123  # Run for specific user
```

**Dashboard:** Tab "Users" has create user form (email, name, bankroll, plan), all users table, isolated paths explanation.

**Files:** `src/ptai/users/manager.py`

### 6. 🔔 Notifications Premium - Discord, Telegram, Desktop

**Premium: Notify via multiple channels when bots finish work**

- Desktop: Linux notify-send, Mac osascript, Windows win10toast
- Discord: webhook URL from env `DISCORD_WEBHOOK_URL` - embeds with color
- Telegram: bot token + chat_id from env `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Channels: desktop, discord, telegram, log
- Methods: opportunity_found, trade_executed, shutdown_alert, error_alert, teammate_update

```python
notifier = Notifier(enabled=True)
notifier.notify(title="Opportunity", message="Edge 15%", urgency="normal", channels=["discord", "telegram"])
notifier.teammate_update(teammate="Quant", task="Build fair value", result="Found 3 opps")
```

**Env:**
```ini
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
```

**Files:** `src/ptai/agent/notifier.py`

### 7. 🎨 Product Dashboard - Full UI for Non-Technical Users (Premium)

**Premium: All in UI, no .env editing, product for other people**

**7 Original Tabs (from previous product UI):**
- Overview: bankroll, PnL, trades, system health (is it working?), self-preservation, recent trades, scans
- Onboarding: 7-step wizard auto-detected progress bar
- Wallet: Link Polymarket account IN UI (private key password field, funder, dry_run toggle, test connection, save to .env)
- AI Brain: LM Studio status, models, R1 slow warning, best models table, setup steps
- Trading: Controls (run cycle, refresh, export), risk explanation, trading config form
- Logs: Tail 100/200 lines, good vs bad patterns
- Settings: .env display, danger zone

**5 New Premium Tabs:**
- AI Teammates: 7 bots status table, give work to team form (full cycle, scan only, etc), roles explanation
- Memory & Learning: Calibration stats, insights table, recall search
- Vault: Signed-in tools by teammate, add tool sign-in form, supported tools list
- Backtest: Run backtest form, result display, equity curve
- Users: Create user form, all users table, isolation explanation

**APIs:**
- `/api/config` GET/POST (masked keys)
- `/api/llm/status` - LM Studio check, R1 detection
- `/api/system/health` - health + onboarding progress
- `/api/logs/tail`
- `/api/wallet/test` - validates key format + CLOB client
- `/api/teammates/status` - 7 teammates status
- `/api/teammates/run` - give work to team
- `/api/memory/insights`, `/api/memory/recall`
- `/api/vault/status`
- `/api/backtest/run`
- `/api/users/list`, `/api/users/create`

**Design:** Inter font, dark theme #0a0e13, cards, metrics, status pills (ok/warn/error), progress bars, alerts, responsive.

**Files:** `src/ptai/dashboard.py` (79KB product UI), `start_dashboard.bat`

### 8. 🚀 Premium Loop - Main Entry for Premium Product

```python
from src.ptai.agent.premium_loop import PremiumTradingAgent

agent = PremiumTradingAgent(user_id="abc123", bankroll=50)
await agent.run_autonomous(interval_minutes=10)  # Team collaborates every 10 min
```

**CLI:** `python main.py premium 50 --daily-cost 5 --interval 10 --user abc123`

**Files:** `src/ptai/agent/premium_loop.py`

## How to Use as Product for Other People

```bash
# 1. Setup
git clone ... && cd ptai && setup.bat  # or ./scripts/setup.sh
# Installs venv, deps, playwright, creates .env with fast mode SENTIMENT_USE_X=false MAX_DEEP_ANALYZE=50

# 2. Start Dashboard (Product UI)
start_dashboard.bat
# Open http://localhost:8000

# 3. Onboarding Tab - Follow 7 steps auto-detected:
# - LM Studio: Download LM Studio, load qwen/qwen3-32b (fast 3s not R1 8 min slow), Start Server port 1234
# - Wallet Tab: Enter private key (0x...) + funder (0x...) + Save + Test Connection + DRY_RUN=true first
# - Trading Tab: Save config (500 scan, 50 deep for fast model)
# - Overview: Check System Health = Healthy, Run Cycle button -> 4 min cycle, check Logs

# 4. Start Autonomous Trading (Team)
# Terminal 1: start.bat (or python main.py premium 50 --interval 10)
# Terminal 2: start_dashboard.bat (dashboard)
# Team collaborates every 10 min: Scout -> Sentiment -> Researcher -> Quant -> Risk -> Trader -> Coach

# 5. For other users (multi-tenant):
# Dashboard Users tab -> Create User -> email, bankroll, plan
# Or CLI: python main.py users create --email user@example.com --plan pro
# Then: python main.py premium 50 --user {id}

# 6. Premium features in dashboard:
# - Teammates tab: Give work to team, see each bot status
# - Memory tab: See what bots learned, calibration Brier score, search past insights
# - Vault tab: See tool sign-ins, add new tools
# - Backtest tab: Test strategy 30 days before live, see equity curve
# - Users tab: Manage users for product

# 7. Go Live:
# Wallet tab -> DRY_RUN toggle OFF (false) -> Save -> Restart start.bat
# Now real orders via CLOB, check https://polymarket.com/portfolio
# Kelly 6% max = $3 on $50 bankroll prevents wipeout
```

## Files Created for Premium Product

```
src/ptai/agent/teammates/
  base.py - BaseTeammate, Task, TeammateStatus
  scout.py - Market Scanner
  sentiment_analyst.py - X/News Reader
  researcher.py - Deep Researcher (browser/terminal)
  quant.py - Superforecaster Fair Value
  risk_officer.py - Kelly 6% Cap
  trader.py - Browser Execution
  coach.py - Learning & Calibration
  coordinator.py - Orchestrates team
  __init__.py

src/ptai/vault/
  vault.py - Secure encrypted vault for tool sign-ins
  __init__.py

src/ptai/memory/
  memory.py - Learning, Brier score, insights, recall
  __init__.py

src/ptai/backtest/
  engine.py - Backtest on historical data
  __init__.py

src/ptai/users/
  manager.py - Multi-user isolated paths
  __init__.py

src/ptai/agent/
  premium_loop.py - PremiumTradingAgent with team
  notifier.py - Discord, Telegram, desktop, teammate_update

src/ptai/
  dashboard.py - Full product UI with 12 tabs (7 original + 5 premium)
  cli.py - Added premium, teammates, memory, vault, backtest, users commands

requirements.txt - Added cryptography
PREMIUM_FEATURES.md - This file
LINK_ACCOUNT_GUIDE.md - How to link wallet & check if working
BEST_CONFIG_48GB.md - R1 + X blocking fix + fast mode
```

## Premium vs Original

| Feature | Original (Single Agent) | Premium (AI Teammates Product) |
|---------|------------------------|-------------------------------|
| Agents | 1 TradingAgent does all | 7 teammates specialized, can give real work to each |
| Tool Sign-In | .env manual, single browser profile | Vault encrypted, per-teammate per-user, UI managed |
| Learning | No memory, no calibration | Memory SQLite, Brier score tracking, Coach learns, insights |
| Backtest | No | Mock historical backtest, equity curve, Sharpe, before live |
| Multi-User | Single user, single DB | UserManager, isolated paths per user, plans free/pro/premium |
| Notifications | Desktop only | Desktop + Discord webhook + Telegram bot + teammate updates |
| Dashboard | Basic stats, trades, scans | 12 tabs product UI: Overview, Onboarding wizard, Wallet linking in UI, LLM setup, Teammates team status + give work, Memory insights + recall, Vault tool sign-ins, Backtest, Users multi-tenant, Trading controls, Logs, Settings |
| CLI | run, status, scan, trade | + premium, teammates, memory, vault, backtest, users |
| Product Ready | For you only, need .env editing | For other people, all in UI, onboarding wizard auto-detected |

## Self-Preservation Still Core

> "Here is 50 dollars earn enough to pay for yourself or shut down"

- Bankroll $50 initial
- Must earn $5/day or shutdown after 3 unprofitable days
- Kelly Half 0.5, max 6% bankroll ($3 on $50) prevents wipeout
- Daily loss 15% pause, total drawdown 30% shutdown
- Coach teammate tracks and learns to improve

## Done - Ready for Product

All premium features built, committed, pushed to `arena/01a0b42e-ptai` branch at `cc6a94e`.

To run premium product:
```bash
git pull origin arena/01a0b42e-ptai
setup.bat
# Load qwen/qwen3-32b in LM Studio (fast not R1 slow)
start_dashboard.bat  # Product UI http://localhost:8000 - all features in UI
# Follow Onboarding wizard
# Wallet tab link account
# Teammates tab give work
# Then start.bat or premium command for autonomous team
```
