# PTAI v2 - Market-Agnostic Autonomous Trading Engine

> **Mission**: Seek positive expected value while preserving capital. Trade only when evidence, calibration, liquidity, and risk constraints agree. If the system's forecasting advantage cannot be demonstrated, stop trading. **DO NOTHING is a successful outcome.**

## Why v2?

Previous PTAI was Polymarket-only, single LLM decides 75%, raw edge >=8% = trade, no calibration, no correlation, no kill switch hierarchy.

**v2 is market-agnostic, ensemble intelligence, separated execution, with capital preservation first.**

## Architecture

```
                         ┌──────────────────────┐
                         │      LOCAL BRAIN     │
                         │ LM Studio / GGUF LLM  │
                         └──────────┬───────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  MARKET INTELLIGENCE ENGINE                 │
│                                                             │
│  Polymarket ─┐                                              │
│  Kalshi ─────┤                                              │
│  Other ──────┤──► Market Normalizer ─► 500-1000 markets    │
│              │                                              │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
       ┌─────────────────┐
       │ Opportunity     │
       │ Scanner         │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────────────────┐
       │ INFORMATION ENGINE          │
       │                             │
       │ News                       │
       │ X sentiment (credibility)  │
       │ Market history             │
       │ Order book                 │
       │ Resolution rules           │
       │ Base rates                 │
       │ Web research (bull/bear)   │
       │ Contradiction detection    │
       └──────────────┬──────────────┘
                      │
                      ▼
             ┌────────────────┐
             │ FAIR VALUE     │
             │ ENGINE         │
             │                │
             │ P(event)=?     │
             │ confidence=?   │
             │ uncertainty=?  │
             └───────┬────────┘
                     │
                     ▼
              EDGE CALCULATOR
                     │
              ┌──────┴──────┐
              │             │
          < 8% edge      ≥ 8% edge
              │             │
             PASS            ▼
                       INDEPENDENT
                       VALIDATION
                             │
                    ┌────────┴────────┐
                    │                 │
                 reject             pass
                                      │
                                      ▼
                              RISK ENGINE
                                      │
                           ┌──────────┴─────────┐
                           │                    │
                       Half-Kelly          Exposure
                           │                    │
                           └─────────┬──────────┘
                                     ▼
                              max 6% bankroll
                                     │
                                     ▼
                            EXECUTION ENGINE
                                     │
                     ┌───────────────┼───────────────┐
                     │               │               │
                  CLOB API        Browser        Limit order
                     │               │               │
                     └───────────────┴───────────────┘
                                     │
                                     ▼
                             POSITION MONITOR
                                     │
                    ┌────────────────┼───────────────┐
                    │                │               │
                  P/L            resolution      exposure
                    │                │               │
                    └────────────────┼───────────────┘
                                     ▼
                              LEARNING ENGINE
                                     │
                                     ▼
                              calibration DB
```

### Biggest Change: Intelligence Separated from Execution

LLM **proposes**:
```json
{
  "market": "...",
  "side": "YES",
  "fair_probability": 0.73,
  "market_probability": 0.61,
  "edge": 0.12,
  "confidence": 0.81,
  "reasoning": "...",
  "sources": [...],
  "trade": true
}
```

Deterministic Python **decides** if allowed, Guard **enforces** max $2.71, max price 0.615.

Even if LLM goes insane, it can't BUY $50k.

## Key Improvements

### 1. Effective Edge (Sophisticated 8% Rule)

Don't just `fair - market >= 0.08`.

```
raw_edge
  ↓ fees (0.8-2%)
  ↓ spread (orderbook)
  ↓ slippage (amount/liquidity)
  ↓ liquidity penalty
  ↓ uncertainty
  ↓ correlation
  ↓ time
  ↓ effective_edge
```

Market YES 0.61, fair 0.73, raw 12%, fees 0.8%, slippage 1.2%, uncertainty 2% → effective 6% → NO TRADE.

### 2. Ensemble Forecasting (Most Important)

Not just LLM saying 75%.

- **Base-rate**: historical frequencies, event type, time remaining, analogues
- **News**: current news, official announcements, reputable sources, timestamps
- **X**: volume, credibility, sentiment, acceleration, bot likelihood, narratives, engagement quality
- **Market microstructure**: bid/ask, spread, depth, recent trades, velocity, volume, liquidity, large orders
- **LLM**: Qwen/other synthesizes

```
Base-rate  0.68
News       0.75
X          0.71
Market     0.69
LLM        0.73
  → Ensemble 0.712
```

Fair 71.2%, Market 61%, Edge +10.2% - defensible.

### 3. Calibration (Extremely Important)

Track 1000 predictions: for 70% forecasts, does 70% win?

If 70% forecasts only win 57%, overconfident.

Maintain:
```
Prediction → forecast prob → resolves → actual → calibration DB
  → Brier score, log loss, calibration curve, accuracy by confidence/category/time/source
```

Learns:
- Politics 70% → historically 64%
- Crypto 70% → 72%
- Sports 70% → 69%

Then calibrate.

### 4. Kelly with Calibration

```
LLM prob → calibration → uncertainty penalty → ensemble → Kelly → Half-Kelly → 6% cap → liquidity cap → correlation cap
```

Portfolio:
- Single ≤6%
- Category ≤15%
- Correlated ≤20%
- Total ≤40-50%

Because 8 independent 6% positions can be same bet: "Will Trump...", "Will Republican...", "Will Senate...", "Will X policy..." all correlate.

### 5. Uncertainty Margin

Market 60%, Model 71% → +11% edge, but uncertainty ±8% → conservative 63% → edge 3% → NO TRADE.

Prevents LLM turning uncertainty into fake edge.

### 6. X as Information Source, NOT Truth

X sentiment +0.82 doesn't mean probability 82%.

```
X signal → credibility analysis → corroboration → novelty → time decay → model input
```

Detect: bot bursts, duplicates, farming, old info, fake accounts, coordinated narratives, credibility.

### 7. Researcher Tries to DISPROVE Trade

- Researcher A: Supporting YES
- Researcher B: Supporting NO
- Researcher C: Info making both wrong
- Researcher D: Resolution rules
- Researcher E: Info after latest market move

Final: SUPPORTING, CONTRADICTING, UNKNOWN, RESOLUTION RISKS, FINAL FORECAST → reduces confirmation bias.

### 8. Resolution-Risk Detection

"What causes contract to resolve YES?"

Question, source, date, timezone, criteria, ambiguous language, cancellation, early resolution, oracle.

If ambiguous → TRADE=FALSE regardless of edge.

### 9. Execution Deterministic

Receives `{market_id, token_id, side, max_price, max_spend}` and nothing else.

```
LLM: "Buy this"
Risk: "Allowed $2.71"
Execution: "Cannot exceed $2.71 or 0.615"
```

### 10. 6% as Ceiling, Not Normal

```
Edge  Confidence  Kelly  Actual
8.2%  0.61        1.1%   0%
10%   0.72        4.2%   2.1%
15%   0.85        18%    6% cap
```

Less aggressive with marginal.

### 11. $50 Mission Correction

Not "earn $5/day or shutdown" implying guaranteed returns.

Mission: Attempt profitability. Stop conditions: bankroll floor, drawdown, consecutive losses, model degradation, calibration collapse, execution anomalies, data-feed failure.

Self-sustainability target separate.

DO NOTHING is successful.

### 12. Kill Switch Hierarchy L0-5

- L0 Normal
- L1 No new trades (daily loss, consecutive losses, LLM unavailable, calibration collapse, internet)
- L2 Cancel orders (execution mismatch, duplicate)
- L3 Exit positions (bankroll <50%)
- L4 Persist state (drawdown 30%)
- L5 Terminate

### 13. Local Architecture (Fast + Deep)

- Fast model: screen 500-1000 (classification, news extraction, sentiment, duplicate)
- Deep model: analyze top 10-50 (forecast, contradiction, synthesis, uncertainty, resolution)
- Deterministic Python: Kelly, risk, limits, orders, portfolio, accounting, shutdown

### 14. Realistic Pipeline

```
1000 → cheap filters → 500 → liquidity → 200 → fast model → 50 → deep research → 10 → ensemble → 3 → risk → 0-3 trades
```

Scan 500 in 1.2s, deep 50*3s=2.5 min fits 10 min for fast model. R1 slow 8 min per market needs 5 deep + hourly.

### 15. Market-Agnostic Opportunity Ranking

Common score:
```
expected_edge × prob_correct × liquidity × execution × calibration × time_efficiency
/ (fees + slippage + uncertainty + risk)
```

15% theoretical edge with poor liquidity loses to 7% edge with excellent liquidity, high confidence, short resolution, low correlation.

### 16. Learn Which Markets Good At

```
Venue/Strategy  Skill  Avg EV  Risk
Weather         0.81   +7.2%   Low
Economic        0.76   +5.1%   Med
Prediction      0.71   +3.8%   Med
Crypto          0.58   -1.4%   High
```

Measured from own historical results, not assumed. Concentrates research where demonstrated skill strongest.

## Market-Agnostic Design

```
                    ┌─────────────────────┐
                    │     PTAI MISSION    │
                    │ Preserve capital +  │
                    │ seek positive EV    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  MARKET DISCOVERY   │
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       │                       │                        │
       ▼                       ▼                        ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Prediction   │        │   Financial  │        │ Other Legal  │
│ Markets      │        │    Markets   │        │ Opportunities│
├──────────────┤        ├──────────────┤        ├──────────────┤
│ Polymarket   │        │ Stocks       │        │ Auctions     │
│ Kalshi       │        │ ETFs         │        │ Marketplaces │
│ Other venues │        │ Crypto       │        │ Other venues │
└──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       │                       │                        │
       └───────────────────────┼────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ OPPORTUNITY ENGINE  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   OPPORTUNITY RANK  │
                    │ "Where is my edge?" │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   RISK ENGINE       │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ EXECUTION ENGINE    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ MONITOR + LEARN     │
                    └──────────┬──────────┘
                               │
                               └──────► back to discovery
```

Instead of "PTAI trades Polymarket", mission is:

"PTAI searches permitted markets for statistically defensible positive expected-value opportunities, compares them on a common basis, and deploys capital only when the estimated advantage survives fees, liquidity, uncertainty and risk constraints."

Each venue must pass qualification via paper trading.

## Module Structure

```
src/ptai/
├── intelligence/
│   ├── ensemble.py (weighted ensemble)
│   ├── forecaster.py (base-rate, news, X with credibility, microstructure)
│   ├── calibration.py (Brier, log loss, curve)
│   ├── uncertainty.py (conservative 71%±8%→63%)
│   ├── contradiction.py (bull/bear)
│   └── resolution_analyzer.py (ambiguous? → NO TRADE)
├── markets/
│   ├── market_normalizer.py (Polymarket, Kalshi → common)
│   └── orderbook.py (spread, imbalance, manipulation detection)
├── information/
│   ├── x_engine.py (credibility, bot likelihood, novelty, decay, corroboration)
│   ├── news_engine.py (reputable sources, recency)
│   ├── web_researcher.py (bull/bear 45s)
│   ├── source_quality.py
│   └── event_timeline.py
├── strategy/
│   ├── opportunity.py (1000→0-3 pipeline)
│   ├── fair_value.py (ensemble+calibration+uncertainty+contradiction+resolution)
│   ├── edge.py (fees, spread, slippage, liquidity, uncertainty, correlation, time)
│   └── strategy_selector.py (mispricing, arbitrage, event, market making)
├── risk/
│   ├── exposure.py (single 6%, category 15%, correlated 20%, total 50%)
│   ├── correlation.py (Trump, Republican, Senate same bet)
│   ├── drawdown.py
│   ├── limits.py (deterministic validation)
│   └── kill_switch.py (L0-5)
├── execution/
│   ├── order_manager.py (guard checks)
│   ├── execution_guard.py (max $2.71, max price 0.615 even if LLM insane)
│   └── reconciliation.py (verify fill, balance mismatch)
├── learning/
│   ├── calibration_db.py
│   ├── trade_outcomes.py (venue/category/strategy performance)
│   ├── performance.py
│   └── model_evaluation.py
├── venues/
│   ├── adapter.py (MarketAdapter interface, common score)
│   ├── registry.py (learns which venues work)
│   └── polymarket_adapter.py (one venue among many)
└── agent/
    └── v2_loop.py (full cycle, DO NOTHING success)
```

## Autonomous Cycle v2

```
EVERY 10 MINUTES
  ↓ Check health (LLM, storage, internet, kill switch, calibration)
  ↓ Check portfolio
  ↓ Check eligibility (Uganda? Never bypass)
  ↓ Scan 500-1000 markets (all eligible venues)
  ↓ Remove illiquid
  ↓ Remove ambiguous resolution
  ↓ Fast model screening (200→50)
  ↓ Historical/base-rate
  ↓ News (reputable, recency)
  ↓ X (credibility, bot, novelty, decay, corroboration) - info source, NOT truth
  ↓ Orderbook (spread, imbalance, large orders, manipulation)
  ↓ Deep research (bull, bear, contradicting both, unknown, resolution risks, new info after move)
  ↓ Bull vs Bear
  ↓ Resolution verification
  ↓ Ensemble (0.68,0.75,0.71,0.69,0.73→0.712)
  ↓ Calibration (Politics 70%→64%)
  ↓ Uncertainty (71%±8%→63%)
  ↓ Effective edge (raw 12% - fees 0.8% - spread 2% - slippage 1.2% - uncertainty 2% = 6%)
  ↓ Is edge >8%? NO → DO NOTHING (successful)
  ↓ YES → Kelly → Half-Kelly → 6% cap → Liquidity cap → Correlation cap → Portfolio risk (total 50%)
  ↓ Execution simulation
  ↓ Place order with guard (max $2.71, max price 0.615)
  ↓ Verify fill
  ↓ Reconcile account
  ↓ Monitor position
  ↓ Record prediction for calibration
  ↓ LEARN (venue, category, strategy performance, where to concentrate)
```

## Geographic & Market Rules Compliance

- **Uganda**: Not in Polymarket blocked list per current check, but verify live at https://polymarket.com/restricted and terms before funding. Never use VPN bypass.
- **Market Rules**: PUBLIC INFO ONLY, NO manipulation, NO spoofing, NO wash trading, NO self-dealing, NO front-running, NO insider info.

## Production Readiness

```
Architecture              █████████░  ~90%  → v2 95%
Local autonomy            █████████░  ~90%  → v2 95%
Risk framework            ████████░░  ~80%  → v2 95%
Market intelligence       ███████░░░  ~70%  → v2 85%
Forecast quality          ██████░░░░  ~60%  → v2 85%
Calibration               ███░░░░░░░  ~30%  → v2 80%
Execution reliability     ██████░░░░  ~60%  → v2 85%
Backtesting               ███░░░░░░░  ~30%  → v2 60%
Production readiness      ████░░░░░░  ~40%  → v2 75%
```

Still not 100% because:
- Need real historical data for backtesting (currently mock)
- Need live orderbook imbalance (currently mock)
- Need multi-model ensemble with real LLM calls (heuristic + LLM)
- Need Kalshi real integration (skeleton)
- Need 100+ resolved forecasts for meaningful calibration

**Recommendation**: Run v2 in DRY_RUN paper trading 30 days, collect 100+ forecasts, measure Brier, venue performance, then cautiously allocate real capital only to qualified venues/categories where skill >0.6 and Brier <0.25.

## Dashboard v2

New endpoints:
- `GET /api/v2/status` - mission, bankroll, kill switch, exposure, calibration, venue leaderboard, learning
- `GET /api/v2/venues` - venues, eligibility, leaderboard, concentration
- `GET /api/v2/calibration` - Brier, log loss, curve, degrading check
- `GET /api/v2/risk` - kill switch L0-5, exposure caps, limits
- `GET /api/v2/opportunities` - pipeline 1000→0-3
- `GET /api/v2/learning` - venue/category performance, where to concentrate
- `POST /api/v2/run-cycle` - trigger v2 cycle

UI tab "PTAI v2 Engine (Market-Agnostic)" with architecture diagram, kill switch, venues, exposure, effective edge explanation, calibration, pipeline, learning, principles.

## Tests

235 tests passing:
- 31 new v2 tests covering adapter, opportunity scoring, eligibility, registry ranking, base-rate model, ensemble, calibration, uncertainty (71%±8%→63%), contradiction, resolution (ambiguous detection), edge (deductions), fair value, opportunity pipeline (cheap, liquidity, fast), exposure (6% cap), correlation (Trump/Republican), kill switch L0-5, limits (6% cap), drawdown, order manager (absolute max $1000), execution guard (LLM cannot override $50k), reconciliation, trade outcomes, calibration DB, X engine (duplicate, bot), market normalizer (Gamma JSON strings), orderbook (spread, imbalance, manipulation), v2 endpoints.

## Philosophy

**Before**: "Make money at all costs or shut down" - dangerous incentive to force trades.

**v2**: "Seek positive EV while preserving capital. Trade only when evidence, calibration, liquidity, risk agree. If advantage cannot be demonstrated, stop trading."

**DO NOTHING is a successful outcome.**

With $50, capital preservation first, then identifying repeatable +EV, then compounding when advantage justifies.

PTAI shouldn't assume venues profitable - prove through paper trading, then cautiously allocate.

Don't optimize profit alone - common score balancing edge, confidence, liquidity, execution, calibration, time vs fees, slippage, uncertainty, risk.
