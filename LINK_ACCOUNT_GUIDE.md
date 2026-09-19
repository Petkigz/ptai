# How to Link Polymarket Account & Check If PTAI Is Working

## PART 1: Link Your Polymarket Account

### Step 1: Create Polymarket Account (if you don't have)
1. Go to https://polymarket.com
2. Connect with MetaMask / WalletConnect / Email
3. This creates a Polygon wallet (chain 137) - this is your funder address

### Step 2: Get Your Wallet Keys

**You need 2 things for PTAI .env:**
- `POLYMARKET_PRIVATE_KEY` - your Polygon wallet private key (0x...)
- `POLYMARKET_FUNDER_ADDRESS` - your public wallet address (0x...)

**How to get them:**

**Option A: If you use MetaMask (easiest):**
1. MetaMask -> Account Details -> Show Private Key -> Enter password -> Copy 0x...
2. Your funder address is your MetaMask address (0x... shown at top)
3. Example:
   ```
   POLYMARKET_PRIVATE_KEY=0x123abc... (64 hex chars, keep secret!)
   POLYMARKET_FUNDER_ADDRESS=0xYourMetaMaskAddress
   ```

**Option B: If you use Polymarket Email Login (Magic Link):**
- Email login uses custodial wallet - you need to export via Polymarket settings or create new MetaMask and deposit to it
- Better: Create new MetaMask, deposit USDC on Polygon to it, use that as funder

**Option C: Create New Wallet for Trading (Recommended for safety):**
1. Create new MetaMask wallet just for PTAI (so you isolate risk)
2. Send $50 USDC on Polygon network to it (Polygon chain, not Ethereum)
3. Use that new wallet's private key + address

**Security:**
- NEVER share private key
- .env is gitignored (not pushed)
- Keep backup offline
- Start with $50 as you said

### Step 3: Put Keys in .env

Edit `.env` file (in ptai folder):

```ini
# For testing - no real money (DEFAULT)
DRY_RUN=true
POLYMARKET_PRIVATE_KEY=
POLYMARKET_FUNDER_ADDRESS=

# For LIVE trading - real money
DRY_RUN=false
POLYMARKET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE_64_CHARS
POLYMARKET_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS
POLYMARKET_CHAIN_ID=137
POLYMARKET_SIGNATURE_TYPE=1
```

**Test first with DRY_RUN=true:**
- Agent will scan, find opportunities, calculate sizes, but NOT place orders
- It logs `Would place BUY ...` instead
- This lets you verify it works without risking money

**When confident, go live:**
- Set `DRY_RUN=false`
- Agent will use `py-clob-client` to sign and place orders via CLOB API
- It creates API creds automatically `create_or_derive_api_creds()`

### Step 4: Fund Your Wallet

- You need USDC on Polygon (not Ethereum mainnet)
- Buy USDC on exchange, withdraw to Polygon network to your funder address
- Or bridge from Ethereum to Polygon via https://portal.polygon.technology
- Check balance: https://polygonscan.com/address/0xYourAddress
- PTAI checks bankroll from DB ($50 default), but real on-chain balance matters for execution

### Step 5: Pull Latest Code & Configure Fast Mode

You are on old code - MUST pull:

```bash
git pull origin arena/01a0b42e-ptai
```

Then edit `.env` for FAST trading (fits 10 min):

```ini
# FAST CONFIG - YOU HAVE qwen/qwen3-32b which is 3 sec per market!
LM_STUDIO_HOST=http://localhost:1234
LM_STUDIO_MODEL=qwen/qwen3-32b
SENTIMENT_USE_X=false
MAX_DEEP_ANALYZE=50
SCAN_MARKETS_COUNT=500
SCAN_INTERVAL_MINUTES=10
DRY_RUN=true   # start true, set false later for live
```

**Why qwen/qwen3-32b not R1?**
- R1 = 8 MIN per market due to <think> reasoning
- qwen3-32b = 3 SEC per market, 160x faster
- You have both - use qwen3-32b for trading!

In LM Studio:
- Unload `deepseek-r1-distill-qwen-32b`
- Load `qwen/qwen3-32b`
- Developer tab -> Start Server (port 1234)
- Test http://localhost:1234/v1/models should show qwen3-32b

## PART 2: How to Know If It's Working

### 1. Terminal Logs (Main Check)

Run `start.bat` - you should see:

```
PTAI Cycle 1 | Bankroll $50.00 | LLM: LMStudioProvider
=== CYCLE 1 SCAN START ===
Scanned 500 markets in 1.2s | Avg Vol 24h $...
Analyzing X sentiment for 60 markets... (or 0 if SENTIMENT_USE_X=false)
Sentiment analyzed for 60 markets | X circuit open: False
Research done for 20 markets
Model detection: 'qwen/qwen3-32b ...' -> is_r1=False is_32b=True -> top_n=50
Deep analyzing top 50 markets by volume
Brain [LMStudioProvider]: Will BTC hit 100k? | Market 60.0% Fair 75.0% Edge 15.0% Conf 0.72 Trade? True
Fair value done: 50 analyzed, 3 opportunities >8% edge (LLM: LMStudioProvider)
Calculating position sizes bankroll=$50
Sized Will BTC hit... | Edge 15.0% | Size $3.00 | Kelly 6% capped
Executing 3 trades
[DRY RUN] Would place BUY 3 @ 0.6 for token ...
Executed Will BTC... -> dry_run
Cycle 1 Summary (180.5s)
```

**Good signs:**
- Scanned 500 markets in ~1 sec ✅
- Sentiment fast (40 sec or 0 sec if disabled) ✅
- Deep analyzing top 50 (not 100) ✅
- Brain logs with Fair values ✅
- Opportunities >8% found ✅
- Sized with $ amounts ✅
- Executed dry_run or executed_api ✅
- Cycle Summary table ✅

**Bad signs (old code you had):**
- Sentiment 100 markets, 13 sec each, 83 min total ❌ -> pull new code
- Deep 30 markets for R1 taking 6-8 min each ❌ -> switch to qwen3-32b
- `local-model` detection ❌ -> new code detects actual model list

### 2. Dashboard - http://localhost:8000

In second terminal, run `start_dashboard.bat`, then open browser:

- **http://localhost:8000** - main dashboard
- **http://localhost:8000/health** - should return `{"status":"ok", "bankroll":50.0, ...}` (you had 404 before, now fixed)
- **http://localhost:8000/api/status** - JSON status

Dashboard shows:
- Bankroll $50.00
- Total PnL $0.00 (0.0%)
- Trades 0, Open 0, Win Rate N/A
- Days Active, Required Profit $5/day, Status OK (or SHUTDOWN if self-preservation triggers)
- Recent Trades table (empty at start, fills after cycles)
- Scans table: Time, Scanned 500, Opps 3, Avg Edge 12%, Time s 180, Bankroll $50

**If dashboard shows data updating every 3 sec -> working!**

If you see:
- `No trades yet - run start.bat to start trading` -> you haven't run start.bat yet, normal at first
- `No scans yet` -> same, run start.bat

### 3. CLI Status Command

In third terminal:

```bash
# Activate venv first
.venv\Scripts\activate.bat  # Windows
# or source .venv/bin/activate on Linux/Mac

python main.py status
python main.py check-llm --provider lm_studio
```

`status` shows:
```
Bankroll: $50.00
Trades: 3 total, 0 open
PnL: $0.00
Days Active: 0
Required: $0.00
Self-Preservation: OK
LLM: LMStudioProvider - models: ['qwen/qwen3-32b', ...]
```

`check-llm` shows:
```
LM Studio detected at http://localhost:1234
Models: ['qwen/qwen3-32b', 'deepseek-r1-distill-qwen-32b', ...]
Test prompt: OK, response: {"fair_value":0.65...}
```

If LLM not detected -> start LM Studio server

### 4. Database & Logs

Check files:
- `data/ptai.db` - SQLite, contains trades, scans, market_analysis, bankroll history
- `logs/ptai.log` - detailed logs

Open DB:
```bash
python main.py status --verbose
# or
sqlite3 data/ptai.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5;"
sqlite3 data/ptai.db "SELECT * FROM market_scans ORDER BY timestamp DESC LIMIT 5;"
```

Logs:
```
tail -f logs/ptai.log  # Linux
type logs\ptai.log     # Windows
```

Look for:
- `X BLOCKING DETECTED` -> X is blocked, circuit breaker working (expected, 40 sec not 21 min)
- `R1 reasoning 32B - VERY SLOW` -> you are using R1, switch to qwen3-32b
- `Fair value done` -> brain working
- `Sized ... | Edge 15% | Size $3.00` -> Kelly sizing working
- `Executed ... -> dry_run` -> execution working (dry run)

### 5. Dry Run vs Live Test

**Start with DRY_RUN=true (safe):**
- No real orders placed
- You see `Would place BUY` logs
- Dashboard shows trades with status `dry_run`
- Verify 2-3 cycles (20-30 min) work, cycle time <10 min

**Then go live (real money):**
- Edit .env: `DRY_RUN=false`, add private key + funder
- Restart start.bat
- Now logs show `Order placed: {...}` with real orderID, not dry_run
- Check Polymarket portfolio: https://polymarket.com/portfolio
- Start with small size: bankroll $50, Kelly 6% max = $3 per trade, safe

### 6. Self-Preservation Check

Agent must "earn enough to pay for yourself or shut down":

- Default: must earn $5/day
- Check dashboard: Required Profit = days_active * $5
- If bankroll $50 and after 3 days you have $40 (lost $10), required is $15, you have -$10 PnL -> unprofitable
- After 3 unprofitable days, it shuts down, exports `data/final_report.csv`, logs SHUTDOWN
- This is safety - prevents infinite loss

You can change daily cost in .env: `DAILY_COST_TO_COVER=5.0` or via CLI `pay-for-yourself 50 --daily-cost 2.5`

### Quick Checklist - Is It Working?

- [ ] `git pull origin arena/01a0b42e-ptai` done, now at c6c2221 or newer
- [ ] LM Studio running at http://localhost:1234/v1/models returns your models
- [ ] In LM Studio loaded `qwen/qwen3-32b` (fast) not R1 (slow)
- [ ] .env has `SENTIMENT_USE_X=false`, `MAX_DEEP_ANALYZE=50`, `DRY_RUN=true` initially
- [ ] `start.bat` shows Cycle 1, Scanned 500 in 1.2s, Model detection qwen3-32b -> top_n=50
- [ ] No 83 min sentiment delay, sentiment 0 sec or 40 sec with circuit breaker
- [ ] Fair value 50 markets in ~2.5 min (3 sec each), not 4 hours
- [ ] Dashboard http://localhost:8000 shows bankroll, auto-refresh 3s, health 200 OK
- [ ] `python main.py check-llm` shows LM Studio detected
- [ ] After 1 cycle, `data/ptai.db` has scans and trades
- [ ] Logs show Opportunities >8% and Sized with $ amounts
- [ ] Then set DRY_RUN=false and private key to go live, check Polymarket portfolio

If all checked -> **WORKING!** ✅

If stuck, share:
- Terminal log of 1 cycle
- Dashboard screenshot
- `python main.py check-llm --provider lm_studio` output
- .env (hide private key!)

