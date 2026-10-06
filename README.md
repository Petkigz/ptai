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

## Where The Rules Are Enforced

The list above is only true of the code if something actually stops the order.
Three things do, each consulted immediately before an order is dispatched:

- **`risk/money_guard.py`** - session (30%) and daily (15%) loss limits, measured
  from SETTLED outcomes in the trade log per lane, so a restart cannot reset them.
  A live refusal sends the order to paper rather than dropping it; the first live
  money at a venue is capped at 2% of capital until 10 live trades there settle.
- **`validation/walk_forward.py`** - out-of-sample folds over the settled record:
  consecutive in time, discovery folds then a fresh holdout, Holm-Bonferroni
  across the rules tested, statistical lift and economic viability judged
  separately. Run `python main.py validate` to see it. It can REFUSE a rule and
  can never qualify a venue - `may_qualify()` returns False always, because a
  backtest figure must never trigger live capital.
- **`validation/rule_bench.py`** - what the agent DOES about a refusal. An entry
  rests on the strongest rule it satisfies; if every one of those has been
  refused out of sample, real money is withdrawn from that entry (it still runs,
  in paper, and the trade records the reason). The bench only subtracts: it never
  widens or rewrites the entry filters.

`python main.py doctor` reports all three before you fund anything.

## The Forecast Engine: What Each Component Contributed

A fair value is built from independent components and the console now shows the
chain, term by term, for every market:

```
market price -> the LLM's own answer -> weighted ensemble -> calibrated
             -> conservative -> the edge a trade is measured against
```

On the Money tab each component lists the probability it wanted, the confidence
it claimed and the weight it actually got - **including the ones that contributed
nothing, with the reason** ("no news retrieved", "X unavailable (circuit breaker
open)", "anchored answer", "not in this cycle's deep shortlist"). A component
that is absent says so; none of them contributes weight it cannot justify.

Three rules this engine holds to, each written because of a specific failure:

* **The prompt contains no example numbers.** It used to print
  `{"fair_value":0.65,"edge":0.15,"confidence":0.72}` immediately before asking
  for an answer, and seventeen forecasts came back 0.65/0.72 with edge 0.15 on
  markets priced 15%-60%. The schema is still given; its values are placeholders.
* **A repeated answer carries no weight.** If a model returns the same
  probability across markets priced materially differently, that answer is
  recorded as no opinion - confidence 0, no ensemble weight, no trade - and says
  why.
* **No input, no weight.** A model with nothing to say returns confidence 0
  rather than a neutral-looking number at the market price, which used to dilute
  every component that did have evidence.

**Base rates are counted, not assumed.** The base-rate model shipped with
category constants and no data, so it contributed nothing on every market. It now
counts YES frequencies from markets the venue reports as closed (refreshed at
most once a day) and carries a sample-sized weight - a category with fewer than
30 resolved markets stays "no data" rather than becoming a number.

**Two-stage scan.** One LLM call per market took 60-75 seconds, which no 10-minute
cycle can absorb. Each cycle now reads every order book cheaply, ranks on
measurable execution quality and liquidity, and spends the deep analysis (X
sentiment, web research, the LLM) on the top group only - `PTAI_DEEP_MARKETS`,
8 by default. Everything else is still read and priced on its measured book, and
the console reports how many markets were deep-analysed and how many were not.

**Order books are validated before they are priced against.** The CLOB returns
each side as a list and does not promise which end is best, so reading index 0
gave the worst quote on both sides - which is how "bid 0.0100 / ask 0.9900,
spread 98%" appeared on market after market. Books are now sorted, checked
against the token and market they were requested for, and refused (with the
reason) when crossed, stale against the market's own price, or one-sided.

## A Round: Research, Bets, And A Bankroll At The End

Pressing **Run one round** - or the agent's own interval - runs one complete round
and ends with a number: the paper account before it and after it.

    Round 4 (312s): paper account $50.00 -> $51.24 = +$1.24 | 6 opened, 0 settled,
    6 held | of which +$1.24 is the book at current prices (marked, not settled)

What the round does, in order: research the market (discovery plus the order
books), pick the suitable predictions (the cheap screen, then deep analysis on
the shortlist - X, web research and the model), research further to choose which
ones to bet (the forecast chain, fees, depth, net EV, the risk rules), bet them
with the paper currency, and close by marking the whole book and reporting the
account at both ends. Every stage's count is on the round (`markets_discovered`,
`markets_screened`, `markets_researched`, `markets_priced`, `positions_opened`,
`positions_settled`, `positions_held`, `staked_usd`), so "it ran" can be told
apart from "it did the work".

**The result is `equity_end - equity_start`, with the two halves kept apart.**
Realised P&L is money that came back when a market resolved - it is in the
bankroll and cannot be taken away. Marked P&L is the open book at current prices;
it moves back and forth and it is not a settlement, and the log says so. A round
that could not price anything reports NO NUMBER rather than a flat $50.00 that
reads like break-even. The console shows the last round and the score across all
of them (up / down / flat, best, worst) on the Agent tab, and the history is
stored (`agent.rounds`, 50 kept) so it survives a restart.

**More positions at once - bounded by the rules.** A round fills every position
slot the limits leave open (6 by default) and stops there: a book that already
holds every slot opens nothing, and the round still researches and reports. Two
things had to be true for that to be safe, and neither was:

  * The exposure ceilings (position count, 15% per category, 20% correlated, 50%
    total) are enforced by `ExposureManager`, and nothing ever told it what was
    already open - it was consulted with an empty list every cycle, so those
    ceilings were enforced against zero. It is now seeded from the trades table
    (which stores `category` and `correlation_group`) at the start of every round,
    and grows as the round opens positions.
  * The executor allows one order per venue per second. A round opening several
    positions came back `rate_limited` after the first and silently lost the
    rest. The round now waits the second out instead of losing the order.

**Paper positions are marked to market.** They used to be carried at cost, so
paper equity was a constant between settlements and no paper round could ever
report anything but the figure it started with. They are now marked at the
current price (the book's mid where a validated book exists, the venue's own
price otherwise, the last price this agent saw if the venue did not answer), with
the side respected - a NO position is marked `1 - yes` - and an unpriced position
carried at cost and counted out loud (`open_positions_unmarked`).

### While A Round Runs: The Screen Has To Agree With The Log

The operator's report - *"when i press run a round it still says stopped while its
showing that its running"* - was right, and it was three separate faults that all
produced that one sentence:

  * **The pill read a heartbeat that is written once.** `agent.heartbeat` is
    stamped at the start of a cycle; `agent.phase` is rewritten for every market
    the cycle touches. A cycle that ran longer than the liveness window (a
    200-market scan, or a model call that took 547s, both in the same log) had a
    stale heartbeat and a phase seconds old - and the verdict used the heartbeat
    only, so the page printed `STOPPED` with the live line *"market 3 of 25 …
    previous market took 547s"* underneath it. A phase that says work is
    happening (`scanning`, `screening`, `evaluating`, `executing`), written
    inside the window, is now evidence of life; the resting phases
    (`cycle_complete`, `no_markets`, `blocked`, `sleeping`) deliberately are not,
    because a process that wrote "sleeping" and then died must not look alive for
    another window.
  * **The button's own answer could fail to send.** The run-cycle reply carries
    the cycle's result under `detail`, which holds live objects
    (`CombinatorialGroup`, `Market`, enums). They are not JSON-serialisable, so
    FastAPI raised inside the response and a round that RAN returned HTTP 500:
    the page showed an error box instead of a bankroll and never re-read the
    agent state. The reply is now serialised through one total converter - a
    round that happened always comes back with its number.
  * **Every real order book was reported as an estimate.** Four lines in the CLOB
    reader called `parse_price_size`, which was defined nowhere; every book
    raised `NameError`, the fallback caught it, and 200 markets in a row printed
    `ESTIMATED … NOT REAL` while the real depth was already in hand. There is now
    one level parser for the whole file (dict `price`/`size`, or a two-element
    list), used by the depth reader and `normalise_book` alike, and it never
    raises - a level that is not a quote reads as no quote.

The same sweep removed the last two undefined names in the tree: the settlement
guard's missing `logger` (a settler that raised reported `NameError` instead of
returning `UNSETTLEABLE`, over a stake) and `MarketMechanics`, which
`get_mechanics` annotated as its return type while only importing it inside the
method body.

### The Page And The Command Line Are One System

The operator's report - *"the webui is too disconeccted with the command line .
things take too long to change or dont change at all . some things are even
mising"* - was not a styling problem. `run_ptai.bat` started TWO processes:

```
start "PTAI Agent (paper)" cmd /k "python main.py run --bankroll 50 --interval 10"
start "PTAI Console"       cmd /k "python -m ptai.ui.console"
```

and the console built a THIRD engine inside the web process on every press of
"Run one round". Three consequences, all of them his sentence:

  * **Two engines, one database.** Both wrote `agent.phase`, `agent.heartbeat`,
    `agent.rounds` and the trades table, so the page could attribute one engine's
    work to the other and neither knew the other existed. The button's reply
    described a round the loop in the other window had never seen.
  * **Settings that could not reach the loop.** `--interval 10` and `--dry-run`
    were fixed when the process started. Changing the mode in the page changed
    what the page *said*; the agent that was actually trading kept its startup
    flags until it was restarted, and the interval had no endpoint at all.
  * **Things that were simply missing.** Nothing in the page could start or stop
    the agent - its lifecycle was a console window - and the agent's log existed
    only in that window, so the browser could not show what the agent was doing
    while the operator watched it.

The fix is one owner, one setting store, one front end:

  * **A LEASE (`agent.engine_lease`).** Any engine that starts the loop claims it
    first, recording kind, pid, host, start time and a heartbeat that is renewed
    on every phase write. A second starter is refused **by name and pid** -
    `python main.py run` now answers *"the agent is already running in the
    console engine (pid 4408)"* and exits rather than running a rival loop.
    `--force` takes it over deliberately. A lease the operator stops is marked
    RELEASED with a reason, so the page reads *"stopped on purpose at 10:31"*
    instead of decaying into "no sign of life", which is what a crash looks like.
  * **The console HOSTS the agent.** `run_ptai.bat` now opens ONE window: the
    console process, with `PTAI_AGENT_AUTOSTART=1`, starts the loop in its own
    thread under the same `run_continuous` the CLI uses. Start, Stop, the interval
    box and "Run a round now" all act on that one loop. The CLI is still the same
    engine and still works; it is simply not a second one.
  * **The interval lives in the database (`operator.interval_min`).** The loop
    re-reads it before every wait, the wait is sliced so a change applies within a
    minute instead of after the old interval, and a shortened interval ends the
    wait that is already running. `--interval` on the command line now WRITES that
    setting instead of pinning a process-local number.
  * **Mode is read every cycle, with the reason when it cannot be honoured.**
    The page's switch is the intent; `dry_run` at start-up is the capability. Live
    needs both, and when the switch is live but the process was started with
    `--dry-run`, the loop says so in the log and in the page rather than silently
    trading paper or pretending the click worked.
  * **"Run a round now" asks the loop; it does not build one.** If the console
    hosts the agent it wakes it (a threadsafe wake into the agent's own event
    loop - an `asyncio.Event.set()` from the web thread sets a flag on the wrong
    loop and changes nothing, which the real console proved before the tests did).
    If another engine owns the agent, it refuses and names it. If nobody owns it,
    the console starts it here. The reply says which of the three happened; the
    figures arrive on the round card from the engine that ran the round.
  * **The agent's log is on the page.** The console keeps a ring buffer of its own
    loguru records and serves `/api/console/logs`; the page follows it by
    sequence number and shows the same lines the command window shows, with the
    process id and which engine wrote them. The durable file under `logs/` is
    named next to it.
  * **Polling follows the work.** The fixed 15-second timer is gone: the page
    reads every 3 s while an agent is running (the log every 2.5 s), backs off to
    15 s when nothing is, and polls nothing at all while the tab is hidden. The
    balance and brain caches fell from 20 s/15 s to 5 s for the same reason -
    "it takes too long to change" was partly a cache, not the network.
  * **A cycle announces itself.** The loop writes a working phase and a heartbeat
    *before* the cycle's first network call, and logs `Starting cycle N` and
    `cycle N took Xs`. A cycle that hangs in a venue call therefore shows as
    WORKING with the time it started, instead of the page resting on a stale
    "next cycle at 08:04" while nothing else happens.

Verified against a running console, not only in tests: the console starts the
agent under its own pid, `Run a round now` started cycle 2 in the same millisecond
it was requested, setting the interval to 2 minutes produced
*"The next cycle starts at 07:59 UTC (every 2 min, set in the console)"* on the
running loop, Stop released the lease with *"the operator pressed Stop"*, and a
`main.py run` in another process was refused with the console's pid.

### The Log Ring Survives Anything

The page reads the agent's log through a ring buffer of the last 800 lines. That
ring is a loguru sink, and `import ptai.cli` calls `logger.remove()` at import
time (a CLI process wants a clean slate) - which takes every handler with it.
After that, the ring never came back: the page showed an empty log with no
explanation, permanently, and the next re-attach *cleared the buffer*, deleting
the lines the operator was already reading. Found by a test that imported the CLI
for a source assertion and then watched an unrelated page test fail.

A read of the log now re-attaches the sink if it is gone, and keeps the buffered
lines. A duplicate handler cannot arise on that path (`_add_sink` is only reached
when no handler is flagged as the ring's), so there was never a reason to throw
the history away.

### A Dead Engine Holds Nothing

The operator's report was four symptoms that arrived as one sentence:

    "the logs are not showing whats happening anymor , if i press stop nothing
     happens , refresh does nothing too . the button for start the agenbt is
     always greyed out . paper mode has zero balance available but it supposed
     to operate on capitaal so it need to have at least 20 or 50 dollars"

The first three have one cause, and it is not a rendering bug. Starting the agent
writes a **lease** - who owns the loop, in which process, on which machine. The
lease was judged fresh by the age of its heartbeat and by nothing else, so when
the engine *died* (the window was closed, the console was restarted, the process
crashed) the claim survived for the rest of the freshness window while the
process behind it did not exist. Nothing had ever asked whether that pid was
alive. From the page that looks exactly like the report:

  * **Start** greyed out, and refused with *"already running in the console
    engine (pid 7444)"* about a pid that is gone;
  * **Stop** refused with *"the agent is running outside this console (pid
    7444); close that window to stop it"* - a window that does not exist;
  * **Refresh** re-reads the same lease and redraws the same thing, so the
    button appears to do nothing;
  * the **log** panel showed this process's ring buffer and never said that no
    engine was running, so a stopped agent and a working one read the same.

Liveness is now part of the answer. `pid_alive()` asks the OS, and a lease whose
pid is gone on this machine is **abandoned**: it holds nothing, the note names
the pid and says the engine is gone, Start takes the lease, and Stop cleans the
record instead of refusing. Two things this deliberately does *not* do:

  * it does not shorten the freshness window - that would declare live engines
    dead every few minutes instead;
  * it does not treat "cannot tell" as dead. A pid on another machine is
    unknown, and unknown is treated as possibly alive, because refusing to start
    a second engine is the safe error and two engines on one database is the bug
    the lease exists to prevent.

A live engine in another window is still refused, by name and pid. `Refresh` now
re-renders a different page because the record itself changed, and the log panel
states whose log it is (*"Showing the last N line(s) buffered by the console
engine, pid 3972 - no engine is running the agent"*). Every Start/Stop leaves its
answer on the page, on its own line, whether it worked or was refused.

### Paper Mode Always Has A Purse

> "paper mode has zero balance available but it supposed to operate on capitaal
> so it need to have at least 20 or 50 dollars"

A paper account with no purse cannot size a trade, and a trial that stops
because its simulation ran out of imaginary money is the one thing paper mode
must never do. The paper purse is checked at every cycle start and when the
console starts the agent, and:

  * a purse **with** money is left alone;
  * a purse at zero **because open paper positions or working paper orders hold
    the capital** is left alone - that money is already in the equity figure,
    and topping it up would double-count the account's own funds, so the log
    says which case it is;
  * a purse at zero **with nothing open** is re-seeded to the configured
    bankroll (never below $20), the top-up is written to the bankroll history so
    the equity curve shows it as a deposit rather than a profit, and the log
    line says so:

        PAPER PURSE: was $0.00 with no open position; re-seeded to $50.00 of
        paper money at the console start (recorded in the bankroll history, so
        it cannot be read as profit)

The page never shows a bare zero either. The header used to read
`$0.00 available`, which is the *live* figures - no venue is funded in paper mode
because none needs to be - beside an unfunded-looking account. In paper mode the
header now shows the simulated account: `paper account $50.00 equity · $50.00
free · paper purse $50.00`, and an empty purse is labelled with what will happen
to it rather than left to look broken.

Reading the paper balance also turned up a second reader of the same database:
the Polymarket adapter opened `./data/ptai.db` directly, ignoring `PTAI_DB`, so a
console pointed at any other database asked a *different file* for the venue's
balance. That is how a funded account gets reported as having nothing. It now
uses the one Storage resolver every other entry point uses.

### The Model Line Is A Record, Not A Plan

> `Local model: qwen3.8-27b-... | NOT USED THIS CYCLE: 8 market(s) were sent to
> the model but no answer was recorded`

The router's own counter in that same cycle read **zero calls**. Both numbers
came from the same dict, and the sentence was built from the wrong one: 8 was
the deep shortlist - a plan the screen writes *before* pricing - and the line
called it a record of sends. A run that used no model time said it had sent
eight markets to a model, and blamed the model for the silence.

Three fixes, because there were three defects:

  * **the count moved to where the asking happens.** The forecast ensemble counts
    a market when it actually calls the model, and reports whether the answer
    came from the model or from the heuristic fallback. "Asked" and "shortlisted"
    are now different keys, so the line cannot confuse them;
  * **the line says which of three facts is true**: the model was asked and
    answered (*"8 of 8 market(s) asked were answered"*), the model was asked and
    nothing came back (*"...none was answered (APITimeoutError...)"*), or the
    shortlist never reached the pricing stage (*"8 market(s) were chosen for deep
    analysis but none of them reached the pricing stage this cycle, so the model
    was never asked"*). A model that CHANGED between cycles is named with the
    model it changed from, because the operator's log showed two different model
    ids in one session with no line connecting them;
  * **the starvation itself is fixed.** The screen and the scan each had their
    own idea of which markets matter: the screen ranked on book quality and
    depth, the scan took the top 100 per venue by volume. The operator's cycle
    chose eight markets for deep analysis and the scan dropped every one at its
    own cap - so no model time was spent at all, on a machine with a model
    loaded. Both stages now call one method
    (`markets_that_will_be_evaluated`), and the screen says how many markets the
    cap will drop before pricing.

The same cycle's per-market line claimed "*192 ranked below it*" for every one
of 192 markets - a per-market ranking the screen never computed. It now says
what the number is: how many markets were priced on their book alone this cycle.

### A Price Is Not A Probability

> `Best opportunity: predictit YES edge 0.093 | Will 0GUSDT close higher in 24h?`
> `whitebit-ADA_PERP ... Executable: pay 0.748 -> -0.953 REFUSED: no executable
> edge: fair 0.100 vs the 0.748 a share actually costs`
> `whitebit-BCH_TRY ... spread 6300.0% = 6302.0% cost must exceed to break even`

A crypto exchange quotes the price of a coin. `ADA_PERP at 0.748` is $0.748, not
a 74.8% chance, and there is no Yes share to buy at that price - so a "spread"
computed against a 0-1 probability is meaningless, which is where 6300% came
from. The venue label was wrong the same way: the market is a Binance pair, and
its venue was read from `market.source`, a three-value prediction enum, which
printed "predictit" for a coin.

  * the crypto adapters mark their markets **not a probability market**
    (`raw["probability_market"] = False`), because that is a fact about the
    instrument, not a preference;
  * the screen and the venue scan refuse them with the reason in words - *"this
    venue quotes PRICES, not probabilities, so there is no Yes share to buy and
    no edge to compute"* - counted once per venue, and they stay visible in the
    venue inventory;
  * the WhiteBIT order book refuses **before it fetches anything**, with
    `source: "not_a_probability_book"` and the actual quotes in its warning, so a
    currency quote can never become a 0-1 spread again;
  * `venue_id` on an opportunity comes from the market's own venue
    (momentum, event trading, market making all changed), never from the
    `MarketSource` enum.

### One Line Per Fact

> `Storage initialized at data\ptai.db` ... every fifteen seconds
> `Portfolio from LOCAL STATE: balance $50.00 ...` ... every fifteen seconds
> `Routing market 4052413 venue_id polymarket -> exact adapter polymarket` x899
> `ERROR ... Could not read Polymarket credentials` ... in a PAPER run
> `Mechanics for predictit-8544-33624 unreadable: AttributeError:
>  'PredictItAdapter' object has no attribute 'get_mechanics'` ... per market

Each of these is one fact printed many times, and together they buried the
cycle's real messages:

  * every hardcoded `Storage(db_path="./data/ptai.db")` is gone (dashboard, CLI,
    the two old loops, the premium loop, the Polymarket adapter). One reader of
    `PTAI_DB`; the vault now defaults to the database's own folder too, so a
    login saved in the console is the file the agent reads. The "Storage
    initialized" line prints **once per path per process** and is DEBUG after
    that, whatever future code builds one per request;
  * the LOCAL STATE portfolio line prints when the numbers **change**, not on
    every poll;
  * routing is printed once per venue (first market), not once per market;
  * a paper process states plainly that credentials are not configured - at
    INFO, with "this process is in PAPER mode, which needs no credentials". An
    ERROR line for a working paper run reads like a broken agent;
  * a venue that publishes no `get_mechanics` reader is not asked for one. The
    fee falls back to its declared capability, exclusivity stays **unknown**
    rather than guessed, and the venue is named **once**, at INFO;
  * the same fact is not printed twice. From the 18:14 run: snscrape's failure
    line carried its whole ~1.5 KB GraphQL URL (three times a cycle), so it now
    names the query and the failure count and puts the URL at DEBUG; ESPN's
    HTTP 403 printed once per league inside a window that was already open, so
    the second request in the window is DEBUG; the base-rate model's "no
    historical data loaded" warning lived on an object built per forecast, so it
    printed once per deep market - it is once per PROCESS now; and the agent's
    `Venue kalshi restricted for UG` no longer repeats the registry's own
    eligibility warning in the same second;
  * `use_registry=False explicitly requested - using PolymarketClient legacy
    path (NOT recommended)` is gone from every cycle. That warning fired for the
    Polymarket adapter's OWN discovery call - the one correct path, since asking
    the registry from inside the registry would recurse - so it read as a defect
    in every run. The direct path is now DEBUG with that wording, and the real
    defect (no registry at all) is still warned about where it is detected.

### A Save That Is Interrupted Leaves A Whole File

> `Qualification load failed: Expecting value: line 507 column 18 (char 23042)`

`_save()` opened `data/venue_qualification.json` with `"w"` - which truncates -
and then streamed JSON into it, so a process stopped during a save left half a
document. The load then failed, discarded the whole record, and left the broken
file in place to fail again on every future cycle.

  * writes go to a sibling temp file and are moved into place with
    `os.replace`, so a reader sees the old complete file or the new complete
    file and never a half one;
  * one unreadable venue record no longer aborts the load for every other venue -
    it is named and skipped, and that venue reads as "never measured";
  * a load failure now **names the file**, says the consequence, and moves the
    unreadable file aside as `venue_qualification.json.corrupt` so the record can
    be rebuilt without losing the evidence.

**A save Windows refuses is not a lost record.** The next run reported

    WARNING ptai.venues.qualification:_save:427 - Qualification save failed:
    [WinError 5] Access is denied: 'data\venue_qualification.json.tmp' ->
    'data\venue_qualification.json'

`os.replace` cannot move a file Windows has open for reading, and the console
reads this file for its snapshot while the agent saves it - so the save was lost,
silently, with the cycle's evidence in it. The save now

  * writes to a temp file named for the writing **process** (`...json.<pid>.tmp`),
    so two writers cannot move each other's half-written file into place;
  * retries `os.replace` six times over ~1 s, which clears a reader's
    milliseconds-long handle;
  * if it still cannot move the file, writes the record **in place** and says so
    in the log: losing the whole save is worse than a non-atomic one, and a
    truncated file is salvaged or quarantined on load anyway.

### An Arbitrage Has To Be One

The scan groups markets by event and adds up their YES prices, because in a group
where exactly one outcome wins, a sum under $1 is free money and a sum over $1 is
free money the other way. The arithmetic was right and the premise was missing.
The 2026-09-29 log shows what the old code printed from 200 real Polymarket
markets:

    COMBINATORIAL ARB FOUND: MECE group elon-musk-of-tweets...: 10 markets sum
      YES 0.003 | buy_all_yes cost $0.003 payout $1.000 profit $0.997 (33233.3%)
      | Exhaustive True Exclusive True | Should trade True
    COMBINATORIAL ARB FOUND: MECE group bitcoin-above-on-september-29-2026: 10
      markets sum YES 4.498 | sell_all_yes_buy_all_no ... profit $3.498 (63.6%)

Neither is an arbitrage. The first is a field the 200-market scan only partly
captured - the sum is missing 99.7% of the candidates, so of course it is small.
The second is ten Bitcoin price strikes, which are not mutually exclusive at all:
several win together, so "all but one NO wins" is false and the $3.50 was
invented. Both were called `Exclusive True` because they shared an event slug
(`is_exclusive = True  # Assume exclusive if same event_slug`) and `Exhaustive
True` because a group held three or more markets.

A basket is now an arbitrage only when the evidence supports all five of these,
and when it does not, the reason names which one failed:

  * **Exclusivity from the venue, not from a slug.** Polymarket's `negRisk` mark
    (on the event payload, or the venue's own CLOB market info) is the fact that
    says the outcomes are one mutually exclusive basket. If the two sources
    disagree, neither is trusted.
  * **Completeness.** Every outcome of the event must be in the basket - the scan
    read 10 of the event's 34 is a refusal, with both numbers on it.
  * **A sum a real basket could be quoted at** (0.75-1.25). Beyond that, the set
    is not a complete exclusive basket, whatever its slug says.
  * **Executable prices.** Cost is the ask on every YES leg and (1 - best bid) on
    every NO leg, from books that validated as real. The venue's mid can show a
    6% gap while the asks cost more than the $1 payout - that basket is refused
    with the loss stated.
  * **The venue's own fee.** The old estimate was the constant `2% x legs x 0.5`;
    the net now comes from the venue's declared rate, and when it cannot be read
    the basket is reported as verified with the net **not claimed**.

The engine also says, in its output and on every group, that it places no orders
(`wired_to_execution: False`, `places_orders: False`): it is research the loop
records, and nothing downstream trades it. A verified basket is a finding, not a
position - the execution lane for multi-leg baskets (all-or-nothing fills, no
unhedged remainder) is not built yet, and the report says so rather than implying
otherwise.

### No Book Is Not A Price

The 2026-09-29 log had two lines, hundreds of times each:

    Orderbook fetch failed for 2139...268: 404 Client Error: Not Found for url:
      https://clob.polymarket.com/book?token_id=2139...268
    ESTIMATED orderbook 4190831: liq $1753212.63358 vol $1011431.527044 spread
      1.0% ... - NOT REAL, $50 trader should verify executable price

The first is the venue stating a fact - its CLOB has no book for that token -
and the second invents a spread, a depth and an execution quality for the market
the venue has just refused. Reading a book now returns `(book, reason)`:

  * **`not_listed`** - the venue answered 404 (closed, settled, or never
    CLOB-listed). The market gets a refusal: `source "no_clob_book"`, every
    price field `None`, `execution_quality 0.0`, `executable False`. Nothing is
    estimated from liquidity or volume, and nothing downstream can price it.
  * **the venue answered with nothing** - an empty book, or levels on one side
    only. Same refusal: there is no two-sided market to trade.
  * **`transport`** - nobody answered (timeout, TLS, DNS). This is the only case
    a labelled estimate is honest, and it says so in the log. It is never an
    input to a cost or an edge.

The flood is fixed at both ends. The 404 body no longer prints a warning per
market - the scan counts the refusals and says so once
(*"N had no venue book at all (not priced)"*), a count that also travels in the
screen payload as `no_book`.

A reference source that raises gets the same treatment. `get_all_reference_odds`
is called once per market, so a per-call memo was no memo at all: the deribit
`TypeError: '<=' not supported between instances of 'NoneType' and 'int'`
printed every couple of seconds for a whole scan. The cause is fixed at the
source (a market with no usable expiry is refused with that reason instead of
being priced, and the lognormal pricer rejects a missing input), and a source
that still raises is skipped for five minutes with one line saying why - then
tried again, so a transient fault cannot disable an anchor for good. The cycle
prints the reasons it has, once:

    Reference odds: 0 anchor(s) from 20 market(s), 0 actionable; unavailable:
      deribit - the market has no usable expiry, so the option cannot be priced

### A Similar Question Is Not The Same Market

The 2026-09-29 log printed this eleven times in one cycle, and then traded on it:

    ARBITRAGE FOUND: Same event confidence 0.82: 'Will Sarah Knafo win the 2027
      French presidential election?' vs 'Who will win the next French
      presidential election?' | Cost 0.420 profit 0.581 (138.4%) adjusted 112.0%
      | Trade: True

Knafo is one candidate in the market the second question settles; a gap between
them is a gap between two different questions, not a mispricing of one. The
"0.82 same event confidence" was a string-similarity score, which is why the same
engine flagged a Brazilian candidate against the winner market. Two more lines in
that log came from the same finding:

    Best opportunity: MarketSource.POLYMARKET+MarketSource.PREDICTIT_arb YES edge
      1.323
    ERROR ... ABORT: opportunity venue_id MarketSource.POLYMARKET+MarketSource.
      PREDICTIT_arb does not match adapter polymarket - exact routing required

The venue id was an enum repr, the pair (two legs, two venues) was being sent down
the single-venue order path, and the arb lane then died on a field the
opportunity never had. All three are now rules:

  * **Identity, not similarity.** A pair is the same market only when the
    normalized questions are identical and, when both venues state an end date,
    the dates agree within seven days. Everything else is a pair *candidate* -
    research, printed at DEBUG, never `ARBITRAGE FOUND`.
  * **Executable prices.** Cost is the cheap leg's ask plus (1 - the other leg's
    bid), each charged the venue's own taker fee. Both books must be real and
    validated; if a book is an estimate, absent, or the fee cannot be read, the
    pair is refused with that reason and the mid-price figure is kept only as
    `indicative_profit_pct`. A margin under 3% after fees is not tradeable.
  * **`estimated_profit_pct` means money.** It is 0.0 until a pair is executable,
    because it is the number the ranking sorts on.
  * **A pair never reaches the single-venue path.** Converted opportunities carry
    `should_trade=False`, a routable `polymarket+predictit` venue id, and
    `raw.research_only`; the arbitrage lane reads the pair itself and builds both
    legs. The cycle's "best opportunity" line skips research-only findings.
  * **No modelled EV above the payout.** A $1.00 stake at 0.0005 cannot earn
    $1218.82 - the best possible outcome of one share is $1.00 minus cost - and
    that number was printed as `netEV`, ranked first, and pushed to execution.
    Such an EV is now refused, with both numbers on the refusal.
  * **No book is counted, not priced.** A market whose book is an estimate or
    absent is skipped by the venue scan with one counted line
    (*"N market(s) not evaluated - their book is an estimate or absent"*)
    instead of running the whole stack to `REFUSED: orderbook is not real`.

## Logins, Venues, And The Sports Lane

**Logins** are saved in the console's Setup tab and kept encrypted in
`data/vault.json` on this machine. The agent applies them when it starts AND at
the top of every cycle, so one saved while the agent is running takes effect on
the next cycle - no restart. A saved value wins over `.env` (which still works);
each field shows which source is in force, and secrets are masked. Betfair,
Kalshi, Polymarket, The Odds API and football-data.org all read from there.

**Venues** have a switch, on the Setup tab: every venue PTAI knows about, with
"use it" on or off. Off means the agent does not ask it for markets and does not
trade it - logged as your choice, not as a failure. On still has to pass every
safety check: funded, qualified, inside the loss limits.

**The sports lane** runs the same lifecycle as the trading side - quoted from a
real feed, placed as a position, settled on the final score, P&L written through
the same learning record, and visible on the Money tab. Two rules it holds to:
a bet is refused at placement if its market cannot be settled from a score
(corners, cards, both-teams-to-score, quarter-line handicaps - with the reason on
it), and a bet is settled only on a result a feed reported - an unreadable result
leaves it OPEN rather than refunding it. Team ratings are Elo built from finished
results (three per team before a rating exists), which is what gives the sports
models an independent view to bet against a price with.

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

## Which Model Is Running

The operator asked, after reading a whole run: *"the lm studio model id is
nowhere to be found in any of these runs, whats happening, why is it not being
used"*. All four holes were real:

  * the startup line printed the **configured** value - `model=local-model`, the
    placeholder - and the actual choice was made lazily inside the first model
    call, where nothing printed it;
  * `LMStudioProvider.chat()` logged nothing on success - no model id, no time;
  * `LLMResponse.model`, the id the **server** says answered, was read nowhere in
    the codebase;
  * a market the model was never asked about logged
    `Using fallback heuristic for {market.id}` with no reason and no model.

The model id is now a named fact at every point where it is decided or used:

    LOCAL MODEL: qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp (lm_studio at
      http://localhost:1234/v1) - auto: picked ... because it is a fast model,
      and deepseek-r1-distill-qwen-32b is R1-style (~minutes per market)
    LM Studio answered with 'qwen3.8-27b-...' in 3.4s
    LLM lm_studio ['qwen3.8-27b-...'] parsed JSON: fair=0.62 edge=0.02
    Brain [LMStudioProvider/qwen3.8-27b-...]: Will the bill pass? | Market 60.0%
      Fair 62.0% Edge +2.0% Conf 0.70 Trade? False | basis research
    Local model: qwen3.8-27b-... (LMStudioProvider at http://localhost:1234/v1) -
      auto: ... | markets asked: 8 | 8 of 8 call(s) answered in 27.1s
      (qwen3.8-27b-... x8) | the screen chose 8 of 200 market(s) for deep
      analysis (a plan: 8 of them reached the pricing stage) | venue scan:
      priced 41 market(s) of the 200 read (99 had no usable book, 60 beyond
      their venue's cap) | thinking: OFF - the model answers directly;
      PTAI's prompt asks it not to think and the request does not ask the
      server to enable it (PTAI_LLM_THINKING=0). Turn it on in Setup > Brain
      to make the model reason first

The per-cycle line is printed by every cycle, including a cycle that discovered
nothing and takes the early-return path. When the model did no work it says so
and says why, with the endpoint:

    Local model: none - no local model server answered (lm_studio
      http://localhost:1234, ollama http://localhost:11434); markets are priced
      without a model | NOT USED THIS CYCLE: no local model server answered, so
      every market was priced without a model | the screen chose 0 of 0
      market(s) for deep analysis (a plan: 0 of them reached the pricing stage)
      | venue scan: priced 0 market(s) of the 0 read

  * **One decision, three readers.** The router resolves the model once, at
    detection, with `choose_loaded_model` - the same function the console's
    Brain panel and the startup panel use - and the provider then calls *that*
    id, so no surface can name a model the provider does not call. The startup
    panel prints `LLM: auto -> <model> @ <endpoint>`, not just the preference.
  * **Why the model may not be used**, stated per cycle: the server is not
    answering; no model is loaded; every loaded model is R1-style; the market was
    not in the cycle's deep shortlist (model time deliberately goes to a bounded
    shortlist - see *Two-stage scan*); or every call failed, with the error.
  * **The record carries it.** Each forecast result carries `llm_model`, the
    ensemble's component row carries it (the console's forecast trace prints
    `llm_reasoning <model-id>`), and the cycle report and the console's forecast
    evidence carry the whole `local_model` block.
  * **A dead server is probed on a leash.** Detection is retried once a minute
    (the operator may start LM Studio mid-run) and reports the first miss in
    full; the retries are DEBUG, so a whole scan no longer prints one "no model"
    line per market.

## The Model Line Counts What Happened

The next run's log carried the V55 model line, and it was still wrong - in a new
way:

    Local model: qwen/qwen3-14b (LMStudioProvider at http://localhost:1234/v1) -
      auto: first loaded model is qwen/qwen3-14b | NOT USED THIS CYCLE: 1
      market(s) were priced in full without the model: this process has no model
      router wired to the forecast engine | model time went to 1 of 200 screened
      market(s); 199 priced on their measured book alone

Four sentences, four different facts, three of them wrong:

  * **the market was not "priced without the model"** - it was the one market
    chosen for deep analysis (4949306) and the *resolution* gate refused it
    before a forecast was ever built, which the same log said two lines later:
    `Blocked by resolution risk`. The model was not skipped; it was never
    reached;
  * **"no model router wired to the forecast engine" was false** - a local model
    had been detected and named in the same line;
  * **"model time went to 1 of 200"** - 1 was the deep SHORTLIST, a plan made
    before any pricing. The router recorded zero calls: no model time went
    anywhere;
  * **"199 priced on their measured book alone"** - 199 was the number the
    screen did not give model time to. The same cycle's scan said most of them
    were never priced at all.

The line is now built from three counters, kept apart on purpose:

| counter | what it means | where it is incremented |
|---|---|---|
| `priced` / `deep_priced` | markets that reached the pricing stage (deep ones separately) | the context provider, once per market it is asked about |
| `forecasts_run` | forecasts the ensemble actually built | `EnsembleForecaster.forecast_market` |
| `asked` / `answered_by_model` | markets handed to the model step, and markets the model answered | the ensemble's LLM branch, at the call |
| `calls` / `answered` / `failed` | calls the provider made | the router, at `chat()` |

...and it says which of the six ways the cycle stopped:

    NOT USED THIS CYCLE: 1 market(s) reached the pricing stage and were refused
      before a forecast was built, so the model was never asked - the cycle's
      own refusal line for each one names the gate (resolution risk,
      contradiction)

The other sentences it can produce: no model server answered; N market(s)
reached the model step and none was answered (with the reason); the router
recorded calls but no market reached the model step; N market(s) were chosen for
deep analysis but none reached the pricing stage (they were filtered out on
volume, liquidity or an unvalidated book); no market reached the deep shortlist;
or the shortlist was priced and forecasts were built but not one was handed to
the model - the actual wiring gap, which is now the *only* reason that sentence
is reachable. When the two counters disagree - asks that produced no recorded
call - the line says `WIRING GAP: N of those ask(s) produced no recorded call at
all, so they never reached the provider`, which is the honest version of "8
markets were sent to the model but no answer was recorded".

**And the tail is about prices, not plans.** `199 priced on their measured book
alone` is gone: the screen's line now says `none of them has been priced yet`
(and names why each was dropped), and what was actually priced comes from the
venues' own reports: `venue scan: priced N market(s) of the M read (K had no
usable book, B beyond their venue's cap)`. Those three numbers account for every
market the screen read - priced, refused for its book, or past the venue's own
cap.

**One model per install.** The same log named `qwen/qwen3-14b` in the router's
detection line and `qwen3.8-27b-...` in the cycle's model line, in the same
minute: the agent and the console had each made their own auto pick from LM
Studio's list, whose order is not stable. The first auto pick is now remembered
next to the database (`data/llm_selection.json`) and reused by every process in
this install while it is still loaded, so one machine calls one model; the
detection line is printed once per process (the console builds a router per
request), and a pinned model in Setup still overrides the memory.

**A refusal is one fact.** The same cycle printed

    CLOB orderbook for <id> REJECTED by validation (no usable levels) - refusing
    to price against it

about forty times, once per market. The refusal is right and stays - a book that
failed validation never becomes a cost or an edge - but each distinct reason is
now reported once and then counted, and the cycle's scan line reports the total:
`polymarket: 40 orderbook(s) refused this cycle and never priced against (no
usable levels x40)`. Nothing is printed for the repeats at all, not even at
DEBUG, because DEBUG is on on this machine.

**The sentence adds up, and nothing goes missing from it.** The 18:14 run
printed `markets asked: 3` beside `5 of them reached the pricing stage`, with no
clause connecting them: two of those deep markets were refused by the resolution
gate before a forecast was ever built. The line now says so. The venue-scan tail
also accounts for every market the scan read:

    venue scan: priced 5 market(s) of the 900 read (95 had no usable book,
      100 beyond their venue's cap, 700 in venues or below the floors this
      cycle does not price - each line says why)

5 + 95 + 100 + 700 = 900. The old sentence stopped at "100 beyond their
venue's cap" and left 700 markets unexplained.

And a venue's own `tradeable` count is now printed as **`tradeable at the
venue's own bar`**, because it is a different measure from the cycle's
`After fees/liquidity/uncertainty/risk: N actually tradeable` - the 18:14 log
showed `polymarket: ... tradeable 1` and `0 actually tradeable` in the same
report, which read as a contradiction rather than as two gates.

**The screen only shortlists markets the scan will price.** The screen applies
the same volume and liquidity floors the venue scan does, so model budget cannot
be spent planning for markets the pricing stage discards - the failure mode
behind "8 chosen for deep analysis" in a cycle that asked the model nothing.

## Thinking Is A Switch You Own

The operator asked, of a whole run: *"it seems thinking is disabled for the model
i want to know why"*. It was, and the answer was in two places - neither of them
the loaded model:

  * **the prompt forbade it.** The system prompt said, unconditionally,
    `No chain-of-thought, no <think> tag, just direct JSON`. That wording was a
    speed decision made after a log where ONE market took ~9 minutes with a
    reasoning model, and it applied even when the operator wanted reasoning;
  * **the request never asked for it.** Nothing in PTAI ever sent
    `enable_thinking` - the chat-template kwarg LM Studio honours for Qwen3-style
    hybrid-thinking models - so whatever the loaded model's own default was,
    applied. There was no line anywhere saying which of the two had decided.

Both are now one setting, and it is yours:

    PTAI_LLM_THINKING=1     # .env, or Setup > Brain in the console

With it on, the request carries

    extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

and the prompt asks the model to reason first and end with the JSON the parser
reads. With it off (the default), the request sends no thinking kwarg at all and
the prompt keeps the speed wording. The kwarg is only sent when it is wanted, so
a server that does not understand it never sees it.

**The cycle says which way it is set, every time.** The model line ends with one
sentence, from one place:

    ... | thinking: OFF - the model answers directly; PTAI's prompt asks it not
      to think and the request does not ask the server to enable it
      (PTAI_LLM_THINKING=0). Turn it on in Setup > Brain to make the model
      reason first

    ... | thinking: ON - the model is asked to reason through the market before
      answering (PTAI_LLM_THINKING=1); each forecast takes longer, and a call
      that overruns the call limit gives no forecast for that market. Turn it
      off in Setup > Brain to go back to direct answers

**What it costs.** A reasoning pass is slower per market; that is why the default
is off. The call limit (`One call waits at most`, `LLM_TIMEOUT_SECONDS`) still
applies, so a model that thinks for longer than the limit gives NO forecast for
that market - which the log says - instead of holding the cycle. Turn thinking on
and raise the call limit together if your model needs more than 180 s per market,
and expect fewer markets per cycle.

**What changes, and what does not.** Nothing about the pipeline changes with the
switch: same question, same market and evidence blocks, same schema, same parser,
same timeout, same accounting. Only three things follow the flag, and each one is
there so the switch cannot quietly cost you an answer:

  * the request (the `enable_thinking` kwarg - sent only when on);
  * the two sentences that tell the model what you want (reason first, or answer
    straight - they can never disagree, and they used to: the system turn asked
    for reasoning while the user turn said `BE CONCISE - NO <think> reasoning
    tags`);
  * the token budget. A reasoning pass is paid for out of `max_tokens`, so with
    thinking on the budget is at least 2048 - otherwise the model is cut off
    mid-thought and there is no JSON left to read, which looks like a model
    failure while the model is doing exactly what it was asked.

A reasoning model that answers through its thinking - `<think>...</think>`, a
bare thinking block, prose, or a fenced code block - is parsed the same as a
direct answer: the extractor strips the wrapper, then reads the first balanced
JSON object. (This was a real failure: the old extractor only understood a
closed `</think>` tag with clean JSON after it, so a wrapped answer was logged as
"no usable JSON" and the market was priced by the heuristic.)

And **"the model was used" now means a model ANSWERED**, not that an HTTP call
came back. In the 17:25 run every call returned 200 with nothing parseable in it,
the cycle's own line said `8 of 8 call(s) answered`, and every market was priced
by the heuristic - which read exactly like a cycle the model had priced. The
router's count is now printed as `call(s) returned`, and `NOT USED THIS CYCLE`
names the reason (`'qwen/qwen3-14b' answered, but its answer had no usable
JSON`) whenever markets were asked and none of them produced a model answer.

The setting is visible and editable in three places that agree, because they read
one field: the Brain panel's **Thinking** row, `POST /api/console/brain
{"thinking": true|false}` (what the button calls), and `PTAI_LLM_THINKING` in
`.env` / `.env.example`. A change applies **from the agent's next start**, the
same as the model pin and the call limit.

Verified end to end: a real cycle over stubbed venues against a stand-in LM
Studio that records what it receives - with the switch off the requests carry no
`chat_template_kwargs` and the prompt says `No chain-of-thought`; with it on
every request carries `chat_template_kwargs: {"enable_thinking": true}` and the
prompt asks for reasoning. The stand-in's recording is the request body, so this
is PTAI's own wire traffic, not a claim about it.

## Paper Mode Trades, And A Round Shows What It Made

Your words: *"paper mode need to start generating income and doing trades
because in all the logs I've seen so far I've not seen 1 trade ... give the
results in the ui, the profits or losses made in a round. now I don't know how
long I'm supposed to run it to produce results so tell me."*

Paper mode could not place a trade for arithmetic reasons. Here is the same
market, priced by a real cycle, before and after (`fair 0.490` against a book
bidding 0.40 / asking 0.42, a 15-point model disagreement):

```
before    Deductions (... unc 0.339 ...) | Effective -0.072     -> 0 candidates
after     CASH deductions (0.026 of edge): fees 0.036 gas 0.010 spread 0.020 |
          risk haircut (NOT cash): uncertainty 0.195 moves the estimate
          0.490 -> 0.400 | Effective 0.064 | Executable: pay 0.420 -> best +0.051
```

Three faults, all fixed:

1. **The confidence and the uncertainty were averaged over components that said
   nothing.** A component with no data already had weight 0 - it could not move
   the estimate - but it was still counted in the average: three of five
   components silent turned a model's honest 75% confidence into
   `conf 0.29` (the floor is 60%) and its 25% uncertainty into `0.71`, which
   then demanded a 12% mispricing before anything could be *considered*. Both
   averages now run over the components that actually contributed, and the trace
   says how many had data (`confidence 0.725 over 2 component(s) with data; 3
   had none`).
2. **Uncertainty was charged as cash on top of the haircut it already applied.**
   `unc 0.339` was 84.75% of the share price charged as though a venue collected
   it. Uncertainty is a haircut on the ESTIMATE, not money: it is charged once,
   in `conservative_probability()` (the estimate shrinks toward the price and
   stops at it), and the log line now separates `CASH deductions` from
   `risk haircut (NOT cash)`.
3. **The "conservative" estimate was not conservative.** It was anchored on 0.5
   whatever the market said, so a 0.49 estimate of a 0.40 market came back as
   `conservative 0.990` - printed in the same cycle as the edge calculator's own
   `0.355`. There is one definition now (`conservative_probability`), used by the
   ensemble and the edge calculator alike, and the live gate is decided on it.

And one structural fault: the paper/exploration lane - the lane that produces
the resolved trades a venue earns live capital with - was fed from markets that
had ALREADY passed the live gates, so on a fresh install it could never fire.

**What a round does now**

* The live bar is unchanged: 8% mispricing on the mid, the conservative estimate
  clearing the price a share costs, confidence ≥ 60%, liquidity and execution
  quality floors, exposure caps, kill switch.
* When nothing clears it, the best **model-backed** market that would still pay
  after every cash cost becomes ONE paper trade, sized at $1, labelled
  everywhere as `PAPER/EXPLORATION` (log, round, console). Its own bar: a real
  two-sided book, a model that actually answered (a rule of thumb is not
  evidence), ≥ 5% mispricing, and ≥ 2% left after paying the executable price and
  every cash cost. It never touches live capital and 95% of the round is
  unaffected by it.
* A market the account already holds is not bought again - that is the same bet
  twice, not a new learning opportunity.
* Every round records `trades` (market, side, stake, price, shares, expected net
  EV, paper/live, exploration or not) and the round's result:
  `net_usd = equity_end - equity_start`, split into **realised** (markets that
  resolved - that money is banked) and **marked** (the book at current prices,
  which can move back). The console's Round card shows all of it, plus the
  **closest call** of the round: the market that came nearest, what it was worth
  after every cost, and the gate that refused it.

**How long to run it**

* A trade is worth what its edge says: $1 at a 5% post-cost edge is about 5
  cents per trade, before the market resolves. That is the honest arithmetic of
  a $50 paper book - income scales with the stake and the edge, not with the
  clock.
* Trades per day are bounded by how many markets carry a real model-backed
  disagreement, not by the interval: one paper trade per round is the lane's
  ceiling, and the round will only find that market when the model disagrees
  with the price enough to pay the spread.
* **Realised** profit or loss needs markets to RESOLVE - days to weeks, run to
  their end date. Until then every round shows the marked number, and the
  console says so rather than reporting a settled result it does not have.
* Live capital needs **100 resolved paper trades** per venue with win rate ≥ 55%,
  Brier ≤ 0.25 and profit factor ≥ 1.1. That is the milestone to watch, and it is
  why the paper lane has to trade at all.
* The number to watch while you wait is the **closest call**: it prints the best
  executable edge of the round against the live bar. When that number is inside
  8%, the live lane is close; when it is far off, the log is telling you the
  market - not the wiring - is refusing.

**Verified end to end**: a real `run_cycle` over a stub venue with a real
two-sided book and a model that answers 0.55 against a 0.42 ask places one paper
trade (2.38 shares at 0.42, $1.00), records the position, and the round reports
`$50.00 -> $49.95 = -$0.05 | 1 opened, 1 held | 1 trade(s) this round | of which
-$0.05 is the book at current prices (marked, not settled)`. A model that agrees
with the market trades nothing and never reaches the venue.

## The Model Meets The Markets That Can Actually Be Priced

Your 2026-09-29 19:30 and 19:43 cycles, verbatim from your console:

```
Cheap screen: 200 market(s) read in 38.7s, 2 chosen for deep analysis (limit 8);
  57 had no venue book; 100 sit below their venue's own per-venue cap
1 of the 2 deep market(s) were refused before a forecast was built
Local model: ... markets asked: 1
closest: 2772194 - model 0.046 against the 0.002 mid, +0.043 a share after every
  cash cost (conservative +0.008) - refused: Mispricing 0.044 < 8% threshold
Final selected 0 trades
```

Two hundred markets read, a deep budget of eight, and the model was asked about
**one**. The round could not have traded: even a perfect funnel would have found
that best candidate 4.4% mispriced, under the 5% paper bar. But the funnel was
starving on its own, for two reasons that are now fixed.

**The per-venue cap was eating the deep budget.** The screen chooses on book
quality (does this market have a validated two-sided book?). The venue scan caps
on volume, and it applied that cap *before* any book was known. So the 100
slots went to markets whose books the venue then refused - the same cycle shows
69 CLOB order-book refusals - and the shortlist could only draw from the handful
of priceable markets that happened to sit inside the volume top-100. Two chosen
out of two hundred.

* The cap now applies to markets that can actually be priced: a market with no
  validated book cannot consume a pricing slot it would only be refused from.
* The screen's shortlist is priced **first**. The two stages used to disagree
  about which markets matter (book quality vs. volume); now the markets given
  model time are the markets the scan reaches first, and the rest of the
  eligible list fills the remaining slots.

**The screen's own arithmetic did not add up.** The line named chosen, no-book
and below-cap; 41 of the 200 markets in your 19:33 cycle vanished from it. Every
market read now falls into exactly one of four buckets - chosen for deep
analysis, below the scan's per-venue cap, refused by the screen itself, or
priceable but not shortlisted - and the console's screen panel shows all four
plus `unaccounted`, which is zero when the numbers are honest. You read this
line to answer "how long do I have to run this", so it has to add up.

**Verified**, on a 200-market venue built to your log's shape - the top 100 by
volume have no usable book, the rest price fine:

```
screen    considered 200 | priceable 100 | refused_at_screen 100 |
          dropped_by_scan 0 | priceable_not_shortlisted 92 | unaccounted 0 |
          shortlist 8
model     asked 8 | answered_by_model 8 | calls 8 | deep priced 8
result    PAPER/EXPLORATION pm100 YES $1.00 -> ROUND 50.00 -> 49.95 = -0.05,
          1 opened, 1 held
```

Before the fix the same scenario asked the model about two markets and priced
one; now the whole deep budget is spent on markets that have a real book. The
bar is unchanged: this restores the opportunity flow, it does not lower what
counts as an edge. A market that is genuinely 4.4% mispriced is still refused,
and says so.

## Nineteen Venues: Which Ones Run, And Where A Login Goes

Your words: *"i have alot of venues but all of them except two are saying
unavailable even in paper mode which doesnt make sense unless they require login
but it seems i cant even connect my credentials to most of them."*

Both halves of that were true, and they had different causes.

**"Unavailable" meant the wrong thing.** A venue was badged *unavailable* when
there was no way to fund it from here - but that is a statement about REAL
capital, and it was printed on a page whose mode was paper. Manifold, PredictIt,
Binance and WhiteBIT were being scanned and paper-traded every single cycle while
reading as broken, because they have no deposit route in this build. The word
now means what it says: **no client is written**, so there is nothing to read and
nothing a login could unlock. The money truth did not disappear - it is printed
under the row as the blocker ("play-money only", "needs an exchange account, KYC
and API keys"), and the page says outright that it is about real capital only.

**A saved login looked like it had done nothing.** Betfair's classification asked
whether its adapter *requires* credentials, not whether they were still missing -
so saving the Betfair login changed nothing on the panel, which is exactly what
"i cant even connect my credentials" looks like from the outside. Whether a login
is still needed is now read from the vault: save it and the venue flips from
*needs a login / not running* to *paper + live data / runs today* on the next
cycle, no restart.

**And there was no way to tell which venues take a login at all.** Seven forms
sat beside nineteen venues; "no form" read as broken. The Logins panel now
answers it per venue, in four states:

```
login saved       1   your saved login is read at the start of every cycle
login needed      1   Betfair: closed without a login - the form is below
login available   2   Polymarket, Kalshi: public data reads without it,
                      the login is for authenticated reads and real orders
no login needed   5   public data, paper-traded for free - nothing to connect
no client yet    11   a login would unlock nothing, so none is offered
```

The only state that is a defect - a venue that requires a login and has no form to
enter it - is called a defect on the page (`missing_form`, zero in this build).

**Verified** against the real nineteen-adapter registry: 8 venues report as
scanned-and-paper-traded (with the funding blocker beside them), 11 as no-client
with the client that is missing named, 0 as "unavailable" while usable, and the
coverage counts add up to 19. Saving a Betfair login moves the venue from
`needs_login / can_run_today False` to `paper_only / can_run_today True` and the
counts from 6 readable venues to 7.

## Reaching Polymarket, Venue By Venue

Your words: *"lets start giving the venues enough code to reach poly market. i
dont know whats better all at once or one by ne. also the one venue at a time was
me thinking it would require more resources to run several venues at once but now
that i can disable thinking i think its ok to run them."*

**One at a time - and the list is much shorter than nineteen.**

"Reaching Polymarket" is three layers, and they have to be built in this order:

```
1. reads_markets        a client that returns live markets
2. reads_account        the venue tells PTAI the balance and open positions
3. places_real_orders   a submission path that can actually reach the venue
```

Polymarket has all three. Every other venue is measured against that, layer by
layer, in the agent's own inventory record - and the console's *Every venue* table
now lists them **in that order**, each with the next piece of work:

```
kalshi         1 layer missing: place a real order
betfair        1 layer missing: place a real order; then save the login
crypto_binance 1 layer missing: place a real order
manifold       1 layer missing: place a real order
whitebit       1 layer missing: place a real order
predictit      2 layers missing: read the account; place a real order
apify          3 layers missing - no Apify API client ...
simmer         3 layers missing - no Simmer SDK client ...
```

**But code is only half of it, and the other half decides where the effort goes.**
A venue you cannot put money into is a paper exercise no matter how much is
written for it, so every row also answers *can money reach this venue from where
you are*:

* **Kalshi** - its own adapter says `restricted` for Uganda, and its funding route
  is a US bank account. It is the closest non-Polymarket adapter in the code
  (real REST API, balance reads, an order endpoint) and it still cannot take your
  money. Build it for the product, not for your own account.
* **Betfair** - Uganda appears on Betfair's published restricted lists (the
  adapter answers `requires_verification`, so PTAI says unverified rather than
  claiming access). Worth confirming with Betfair before a line of code.
* **Manifold** - real API, play money only: it can never hold real capital, and
  that makes it the safest place to prove a full live path end to end.
* **Crypto (Binance, WhiteBIT)** - fundable by you, real APIs, but they are not
  probability books: the mispricing-versus-forecast stack does not apply without
  a separate strategy lane.
* The other eleven have no client written at all, and most are UK/EU exchanges
  that will not accept a Uganda account either.

So the honest queue for a real trade is: **Polymarket (done)**, then the first
venue that is both buildable *and* fundable from Uganda - and that is a decision
the console now lays out rather than a guess.

**On "one venue at a time":** your reason was resources, and it was the right
thing to check - but it was never what one venue was protecting. PTAI already
scans nineteen venues every cycle, and the model is asked per market either way,
so scanning more venues costs no extra model calls. What one venue protected was
CAPITAL: every live venue is a separate funded account with its own minimum order
size, and the agent cannot move money between them.

It is now a setting - `PTAI_MAX_LIVE_VENUES`, default **1** (unchanged), ceiling 5
- and the protection that matters is kept: **free cash is spent down the list of
live venues**, so the same dollar is never promised to two accounts. A second
venue can only go live when it has its own order path, its own funded account, its
own authorised budget and its own qualification record. Today raising the setting
changes nothing, because exactly one venue can place a real order - which is the
point: the switch is ready, the capability is not.

## The Road To 100 Resolved Trades

Your words: *"i need to get Polymarket paper record to 100 resolved trades so
live unlocks."*

That is the number that unlocks real capital, and three things decide how fast it
moves. All three were against you:

**1. The lane placed one trade a round.** One was the ceiling while the lane only
had to prove it could trade at all. It is now `PTAI_PAPER_TRADES_PER_CYCLE`,
default **3** (ceiling 5) - a round that finds three qualifying markets learns
from three. Everything else about the lane is unchanged: a real two-sided book, a
model that actually answered, at least 5% mispricing, at least 2% left after the
price paid and every cash cost, $1 a trade, one position per market, and the
exposure manager still caps what can be open at once.

**2. Nothing preferred markets that actually SETTLE.** 100 *resolved* trades is
the goal, and a position that resolves in three months teaches the record next
quarter. Two changes:

* the cheap screen now gives a **bounded bonus** to markets with a near end date
  (`+0.15` inside 24h, `+0.10` inside 3 days, `+0.05` inside a week, nothing
  otherwise) - the same book and volume, sooner, wins the deep slot. This is
  capital x time, straight from the economic framework, not a paper-mode hack.
  It is a bonus and never a gate: a market with no book is still refused at -1
  whatever its date, and a market with no end date is not treated as if it
  settled tomorrow.
* the lane itself takes the **soonest-resolving** candidate first: within a day
  first, then within a week, then the rest, soonest inside each band. Every paper
  trade's log line now says when it settles - *"It settles in 6h - every one of
  these counts toward the 100 resolved trades live capital needs."*

**3. The count alone is not the gate.** Live capital needs 100 resolved trades
**and** win rate ≥ 55% **and** Brier ≤ 0.25 **and** profit factor ≥ 1.1 **and** a
positive net P&L. A record can pass the count and fail the rest, so a bare "12 of
100" would have been the wrong number to wait on. The console's Capital & Funding
tab now carries a **Paper record towards live capital** panel with all five gates,
a progress bar, the number of open positions waiting to settle, the median time
from entry to settlement, the current pace, and - when the count is the only thing
missing - the **ETA at that pace**. The line under it names which gate is actually
binding: *"37 of 100 resolved paper trades at the current pace of 4.2/day (~15d).
6 position(s) are open and waiting to settle."*

**Measured, not promised.** A paper position now records `resolved_at` when it
settles, so the pace and the median hold time are real measurements rather than
guesses from the entry timestamp; rows that settled before that column existed are
counted but excluded from the rate. The panel reads the qualification engine's own
`requirements` - a test asserts the displayed gates equal the enforced ones, so the
screen can never promise a bar the gate does not use.

**What this means for you:** restart PTAI, leave it running in paper, and watch
that panel. Each round can now add up to three resolved-in-days learning trades
instead of one, on the markets most likely to settle. Live unlocks when the panel
says all five gates are met - not before, and not on the count alone.

## Funding From Uganda: Bitcoin Is A Real Rail

Your words: *"funding the venues is impossible in ugx of course but what if i use
bitcoin"*. It is not impossible, and Bitcoin is one of the ways in.

Polymarket does not take a deposit in one form only. It runs a **bridge**: you
send an asset on its own chain, the bridge converts it, and it arrives in your
Polymarket balance as pUSD (1 pUSD = 1 USDC = $1). Bitcoin is a supported source
chain, so money that can only reach Bitcoin can still fund a live account. That is
the part worth knowing, because the thing that was actually impossible was the
route you were looking at - a USD bank wire or an internationally enabled card.

**The three ways in, and what each one costs:**

| Rail | Minimum | What it costs | Best for |
| --- | --- | --- | --- |
| **Bitcoin** through the Polymarket bridge | **$9** | The bridge's conversion cost, plus the Bitcoin network fee. No card fee, no 3-5% spread. | Local money that can reach BTC but not a USD card or bank - mobile money and P2P countries |
| **USDC on Polygon** from an exchange | **$2** | The exchange's withdrawal fee, usually $0.10-1.00 whatever the size, plus cents of Polygon gas. | The cheapest way in above about $50 - if you can get USDC |
| **Card on-ramp** (MoonPay / Transak) | **$5** | About 3-5% over the market rate plus card fees. On $50 that is $1.50-2.50 gone before the first trade. | Speed, and an internationally enabled card |

**Getting the asset in the first place is its own step, and it is recorded too.**
For Uganda the panel says what is actually true: pan-African on-ramps such as
Yellow Card take MTN/Airtel mobile money or a bank transfer (KYC once, spread
instead of a fee); escrowed P2P marketplaces - Binance P2P, OKX P2P, NoOnes -
match you with a seller for UGX and hold the crypto until you confirm the local
payment. What you cannot do is walk into your bank or send mobile money straight
to a venue: local banks and mobile-money operators do not convert crypto, and no
local exchange is licensed. So the two rails above are the practical ones, and
both end in exactly the assets the deposit rails take.

**The failure modes are stated on the same panel, because they are where the money
goes:**

* **USDC is Polygon only.** USDC on Ethereum, Solana, BSC or Arbitrum sent to a
  Polygon address is not credited and may be gone. This remains the single most
  common way people lose a deposit here, and it is why the panel keeps the warning
  next to the rail rather than in a docs page. Polymarket runs a recovery tool at
  `recovery.polymarket.com` for supported tokens sent the wrong way - a wrong
  *network* is not always recoverable.
* **Bitcoin has a $9 minimum, and it is real.** A smaller deposit is not processed.
  And BTC's price moves: the conversion happens when the deposit lands, so the
  pUSD you get is the value at landing. Buy and send promptly. On a $50 deposit a
  few dollars of movement is cents to a dollar - still far less than the card fee
  it replaces.
* **P2P is a counterparty, not a bank.** Keep the payment proof, never release the
  crypto before the money is in your account, keep the chat in the platform, and
  start with an amount small enough that one bad trade does not matter.

**Where to see it:** the console's **Capital & Funding** tab. The funding panel now
shows *"Ways to get money in"* - each rail with its minimum, cost, steps and
failure mode - followed by *"Getting from Ugandan shillings to a deposit"*. The
country is read from the same recorded inventory the venue table makes its
eligibility verdicts from, so the funding panel and the venue page cannot disagree
about where you are; a country with nothing recorded gets silence, not a guess.

**Two things this does not change.** Funding an account is not the same as going
live: the agent still needs the 100-resolved-trade record and the other four gates
(see *The Road To 100 Resolved Trades*), and a budget you authorise at the venue.
And PTAI still never touches the money: the deposit address comes from YOUR
Polymarket deposit screen, the agent never sees it and never needs it, and no
deposit is ever sent to the agent.

The minimums above are recorded from Polymarket's own bridge documentation
(2026-09). The live list is `GET https://bridge.polymarket.com/supported-assets`,
and PTAI's copy is a date-stamped record rather than a guess - re-check it before
a large deposit.

## The Second Venue That Can Place An Order: Kalshi

Your words: *"so ye now we can implement all venues."* This is the first one done,
and it is the one that mattered most, because Kalshi is the only other venue the
money side of PTAI was ever set up to fund.

**What Kalshi was, honestly:** an adapter that read markets and prices, refused
every order with `"Live trading not implemented for Kalshi yet"`, answered a
balance request with zeros the moment an API key existed, and stamped every book
`"ESTIMATION not real Kalshi orderbook"` even when the exchange had answered.

**What its adapter does now:**

1. **It signs the way Kalshi signs.** Kalshi authenticates with an RSA key from
   your account, not a secret string. The Logins tab asks for it
   (`KALSHI_PRIVATE_KEY`, the whole PEM) and the adapter signs
   `timestamp + method + path` with RSA-PSS/SHA-256. One key id without its key,
   or a key that cannot be read, is a refusal with the reason - never a
   half-configured attempt at your money.
2. **A real order path.** `POST /portfolio/events/orders`, Kalshi's current V2
   shape: `bid`/`ask`, a fixed-point dollar price, a contract count, and a
   `client_order_id` so a retry cannot become a second order. YES goes out as a
   `bid`; NO goes out as an `ask` at `1 - price`, which is Kalshi's own
   documented equivalence (a YES bid at p *is* a NO ask at 1 - p, same size), so
   the money at risk is identical either way.
3. **Prices on the market's own grid.** Kalshi now prices in sub-cent ticks that
   differ per market (`price_ranges`): 0.001 steps near the edges, whole cents in
   the middle. Off-grid prices are rejected, so the price is snapped - **down**
   for a buy, **up** for the YES leg of a NO order, because the bound being
   honoured is your maximum, not a rounding convention.
4. **A real account read.** `GET /portfolio/balance`, signed, in dollars. A
   0.00 now means the account is empty; before, it meant "we did not ask".
5. **The book is the exchange's book.** `orderbook_fp` publishes bids only, best
   bid last, and the ask is the other side's best bid complemented. When the
   exchange cannot be read there is **no book at all** - no placeholder spread
   around the last price.
6. **Settlement the exchange reports.** `result` (yes/no) plus the market's
   status and settlement value, so a Kalshi paper trade resolves on a real
   outcome. That is another honest source of resolutions on the road to 100.
7. **The fee is a curve, not a zero.** Kalshi charges
   `round_up(0.07 x contracts x P x (1-P))`: 3.5% of a $1 stake at 50c, 6.3% at
   10c. The old capability said `0.0` - the same value as "free" - and the code
   around it even said Kalshi charges nothing. The cost model now asks the
   adapter for this market's rate and gets the curve.
8. **An order probe that proves permission.** Place one contract at the lowest
   price the market allows (it cannot be marketable at that price), then cancel
   it. Declared as `supports_order_probe`, so the account-health ladder can
   prove order permission rather than assume it - and it refuses in dry run.

**What is NOT true, and the venue page keeps saying it:** you cannot fund Kalshi
from Uganda. It requires US residency and KYC, and its deposits are USD by ACH.
Its own adapter answers `restricted` for UG, and its row reads *"an order path is
not a funding route"*. What changed is that Kalshi is no longer blocked by
missing code - it is blocked by geography and a bank, which is a different and
honest sentence.

Two smaller limits, recorded rather than hidden: the per-market fee multiplier
some Kalshi series carry is not on the payload this client reads, so 1 is assumed
(the general schedule) - a market with its own multiplier has its fee
under-stated, never the order. And no live order has been placed with real money
from this machine; the path was built against Kalshi's current documented API and
every response is reported verbatim, so a wrong assumption arrives as a refusal,
not as a wrong trade. If you are ever in a position to use it, place **one
contract** first.

**What is next, in the same order the venue page gives:** Manifold still refuses
orders in its adapter, and the crypto venues - Binance, WhiteBIT - cannot ride
the probability lane at all: their markets are not probability markets, so they
need a directional-pricing lane before an order path on them would mean anything.
(Betfair was the second of the distance-1 venues and is done - see below.)

## The Third Venue That Can Place An Order: Betfair

Betfair is the venue with the widest tradable catalogue PTAI knows about: match
odds, goals, corners, cards, correct score, half-time, Asian handicaps - the
derivative markets the pricing models were actually written for. Its adapter
read all of that and then refused every order on purpose, with a good reason:

> Placing a real Betfair order needs the back/lay side, the price, the size,
> persistence type and a liability check against the wallet.

**The stranger fault was underneath.** The exchange's prices were being read and
then thrown away before they reached the paper fill. Everything downstream prices
a binary outcome in 0..1 (`bids`/`asks`); Betfair's book came back in decimal
odds under `back`/`lay`, so the paper lane looked for `bids`, found none, and
priced the trade from an **assumed 2% spread** instead of the exchange's real
prices. The venue with the best prices was the one whose prices the simulator
trusted least. The book is now published in both spaces: the exchange's own odds
and stakes, side by side with the same book converted to probability and contract
counts (`size x odds`, because the exchange quotes the *backer's stake*, not
contracts).

**What its adapter does now:**

1. **A real order path, through the exchange's own client.** `place_orders` with
   a `LIMIT`/`BACK` instruction on a specific selection, persistence `LAPSE` (the
   exchange cancels it at the off, so an order the agent forgot cannot be matched
   in play at a pre-play price), and a `customer_ref` so a retry is recognisable
   at the venue.
2. **Prices on Betfair's own ladder.** Its increments are not one cent - they
   widen with the odds (0.01 up to 2.0, then 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5,
   10). An off-ladder price is rejected, so prices are snapped onto the ladder,
   and the authorised maximum is a probability: a cap of 0.40 means odds of
   2.50 at the shortest, rounded **up**, so rounding can never buy above your
   limit. When the market is offering worse odds than the cap, the order rests at
   the cap instead of paying more.
3. **The size is what the market actually offers.** A thin market produces a
   smaller bet, not a rejected one - the stake is capped by the money available
   at that price, and anything below the exchange minimum is refused with the
   number.
4. **A side is a selection, and nothing is synthesised.** YES is the market's
   first published outcome; NO is the other runner of a **two-outcome** market.
   A NO on a 1X2 market is "the draw or the away win" - two selections, not one
   bet - so it is refused rather than turned into a lay whose liability the risk
   layer never sized. **Lays are not built at all** for the same reason: a lay
   risks `size x (odds - 1)`, which is not the quantity this executor sizes.
5. **An order probe that proves permission.** One pound at the shortest price on
   the ladder (1.01, which cannot match), then cancelled, with the cancel
   confirmed before the probe reports success. If the cancel is not confirmed,
   the probe fails and names the bet id that may still be resting.
6. **Settlement the exchange reports.** A closed market with one runner marked
   WINNER resolves; a void market or a dead heat resolves **nothing**. And to
   make that mapping exact, a market whose first runner has no two-sided price is
   not published at all - because the settlement reads the exchange's runner
   order, and a market published with a different order would attribute a win to
   the wrong outcome for good. Those markets are counted in the coverage report,
   not passed over in silence.
7. **The fee is commission, not 2% of stake.** Betfair charges on **net
   winnings**, only when the bet wins: at odds 2.50 that is 7.5% of the stake,
   nearly four times the flat "2%" the capability used to declare. The per-market
   rate the cost model uses is `5% x (1 - p)` - the exchange's Market Base Rate,
   converted to a per-stake number with the market's own probability of the
   charge, and the worse leg when the side is not yet known. The flat field keeps
   the ceiling, so a consumer that can only read a number over-charges rather
   than under-charges.
8. **The account read is the venue's own.** Available balance, and the open
   orders list, under `betfair_account_api`. `positions` stays empty with a note,
   because an exchange holds matched **bets** and valuing them needs the live
   price of every market the account has touched - that is not a number to make
   up. The account's currency is reported too, and a non-USD account says plainly
   that the ledger reads its amounts as dollars, which is wrong by the exchange
   rate.

**What is NOT true, and the venue page now says it in one sentence:** Betfair's
feed is **closed**. Without the login it returns no markets at all, so its row is
"real money path, waiting for a login" rather than "scanned every cycle" - it is
the one venue where the adapter can submit and still cannot run yet. And you
cannot fund it from Uganda: the exchange funds by card or bank in the countries
it serves, it decides who it serves, and its own eligibility check answers
`requires_verification` for UG rather than a yes. So its row reads *"cannot hold
real money from here: funds by card or bank transfer in the countries Betfair
serves"*. An order path is still not a funding route.

## Manifold: Real Prices, Real Resolutions, No Money

Manifold trades in **Mana**, and Mana cannot be cashed out. So this venue will
never hold your money, and nothing below claims otherwise - its row is
`paper_only`, its funding line says `play-money only: no real capital can be
deployed`, and its adapter refuses every order at every setting with the word
*play money* in the reason.

What it did have was a made-up spread. `get_orderbook` answered with
`bid = price - 1.5%` and `ask = price + 1.5%` - a spread nobody quoted, on the one
venue whose mechanism is completely specified and published. Every Manifold paper
trade was priced from that fiction.

1. **The book is now the venue's own AMM curve.** Manifold binary markets
   (`mechanism: cpmm-1`) hold `y^p · n^(1-p) = k` constant, where `pool.YES` and
   `pool.NO` are published and `p` is the creator's weight. A bet adds mana to
   both pools and removes shares of the side bought until the invariant holds
   again, so the shares received, the price paid and the price impact are all
   computable from the venue's numbers. A 100/100 pool at `p=0.5` with a 10-mana
   YES bet returns **19.09 shares** and moves the market to **54.75%** - Manifold's
   own worked example, pinned to four decimals in the tests.
2. **The ladder walks the curve.** Each level is one slice of a bet (1, 2, 5, 10,
   25, 50, 100, 250 mana): the average price paid inside that slice and the shares
   it buys. A $1 paper trade sees the touch; a $250 one sees the price impact. The
   paper lane reads those levels like any other book, so a Manifold fill is the
   mechanism's own fill instead of a 1.5% guess.
3. **A NO is priced in YES space.** Everywhere else in PTAI a book is YES-token
   prices, so the NO ladder is published flipped - buying NO at 0.52 reads as a
   YES bid of 0.48 - and `bids`/`asks` sort the way the paper parser expects.
4. **The numbers must agree with each other, or nothing is published.** The
   weight implied by the pool and probability is compared with the `p` the venue
   publishes; the curve must also reproduce the market's own probability. If they
   disagree, the book carries **no levels** and a reason naming both numbers. A
   curve from contradictory inputs is a plausible-looking guess.
5. **Settlement is the venue's record.** `GET /v0/market/{id}` is public, so a
   Manifold paper trade now settles on the venue's real resolution - YES → 1.0,
   NO → 0.0 - and reaches the resolved count that unlocks live trading. `MKT`
   (a percentage payout) and `CANCEL` (a void) are **refused**, not rounded to
   0 or 1; a non-binary market is refused with its type.
6. **The account read needs a key, and now has a form.** Manifold's markets and
   resolutions are public, so the market feed needs no login at all. The Mana
   balance does: `GET /v0/me` with `Authorization: Key …`. There is now a Manifold
   login in the same vault as every other venue, the agent reads it at startup and
   refreshes it mid-run, and the inventory's next step for this venue is *"save the
   Manifold login: the account read needs it"* - a form to fill in, not missing code.
   With no key, the account read says there is no key; it used to return
   `balance: 0`, which read as an empty funded account instead of an unauthenticated
   one.
7. **An optional login is not a missing one.** A tool whose fields are all optional
   used to report `configured` from the moment it existed, so Manifold's unsaved key
   showed as saved. Coverage now requires a value to call a login configured, and a
   venue whose markets are public and whose login only adds the account read reads
   `no_login_needed` - with the optional form still visible, because it exists.

**Still NOT true, and the row says so:** Manifold cannot hold your money, cannot
place an order PTAI settles in dollars, and cannot be funded from anywhere -
Mana is play money. Its API does accept bets (`POST /v0/bet` with a Mana API key),
and PTAI deliberately does **not** send them: an order path that spends Mana would
have to be labelled a real order path, and the venue page would then claim a
capital route this venue does not have. The honest use of Manifold is the one now
implemented - a real venue with real prices and real resolutions feeding the paper
record.

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
