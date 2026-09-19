#!/usr/bin/env python3
"""
PTAI Demo - Runs without API keys, shows full pipeline in DRY RUN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel

console = Console()

console.print(Panel("""
[bold green]PTAI Demo - Local Autonomous Trading Agent[/bold green]

This demo runs WITHOUT real money or API keys.
It will:
1. Scan 100 Polymarket markets (public API)
2. Mock X sentiment (if snscrape not available)
3. Build fair value with heuristic (or Ollama if installed)
4. Show Kelly sizing with 6% cap
5. Show what would be executed

No browser needed for demo.
""", title="PTAI Demo"))

from ptai.markets.scanner import MarketScanner
from ptai.sentiment import XScraper, SentimentAnalyzer
from ptai.agent.brain import Brain
from ptai.risk import KellyCalculator, RiskManager
from ptai.storage.db import Storage
from ptai.config import get_settings

settings = get_settings()
settings.dry_run = True

# 1. Scan
console.print("[bold]1. Scanning markets...[/bold]")
scanner = MarketScanner()
markets = scanner.scan(target_count=100, order_by="volume24hr")
console.print(f"Found {len(markets)} markets")
for m in markets[:5]:
    console.print(f"  - {m.question[:70]} | YES {m.yes_price:.1%} | Vol ${m.volume_24h:,.0f}")

# 2. Sentiment (mock if needed)
console.print("\n[bold]2. X Sentiment (top 3 markets)...[/bold]")
scraper = XScraper(method="snscrape")
analyzer = SentimentAnalyzer()
sentiments = {}
for m in markets[:3]:
    try:
        tweets = scraper.search(m.question, limit=10, lookback_hours=24)
        sentiment = analyzer.analyze(tweets, m.question)
        sentiments[m.question] = sentiment
        console.print(f"  {m.question[:50]} -> score {sentiment.score:.2f} {sentiment.summary[:100]}")
    except Exception as e:
        console.print(f"  {m.question[:50]} -> sentiment failed: {e} (using neutral)")
        from ptai.sentiment import SentimentResult
        sentiments[m.question] = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34, summary="mock neutral", key_phrases=[], tweet_count=0, confidence=0.3)

# 3. Fair value
console.print("\n[bold]3. Fair Value (top 5)...[/bold]")
brain = Brain()
opportunities = []
for m in markets[:10]:
    sentiment = sentiments.get(m.question)
    fv = brain.estimate_fair_value(m, sentiment)
    console.print(f"  {m.question[:50]} | Market {fv.market_price:.1%} Fair {fv.fair_value:.1%} Edge {fv.edge:.1%} Conf {fv.confidence:.2f} Trade? {fv.should_trade}")
    if abs(fv.edge) >= 0.08:
        opportunities.append({
            "market": m,
            "market_price": fv.market_price,
            "fair_value": fv.fair_value,
            "edge": fv.edge,
            "confidence": fv.confidence,
            "question": m.question,
            "market_id": m.id,
            "event_slug": m.event_slug,
            "yes_token_id": m.yes_token_id,
            "no_token_id": m.no_token_id,
            "side": fv.side,
            "reasoning": fv.reasoning
        })

console.print(f"\nFound {len(opportunities)} opportunities >8% edge")

# 4. Kelly
console.print("\n[bold]4. Kelly Sizing (6% max)...[/bold]")
storage = Storage(db_path="./data/demo.db")
storage.set_bankroll(50.0)
kelly = KellyCalculator(kelly_fraction=0.5, max_pct=0.06, min_edge=0.08)
risk = RiskManager(storage=storage, kelly_calculator=kelly)

sized = []
for opp in opportunities[:5]:
    check = risk.check_all(opp["market_price"], opp["fair_value"], opp["market_id"], opp["confidence"])
    console.print(f"  {opp['question'][:50]} | Edge {opp['edge']:.1%} | {check.reason} | Allowed? {check.allowed}")
    if check.allowed:
        sized.append({**opp, "position_size_usd": check.adjusted_size_usd, "kelly_result": check.kelly_result})

# 5. Execution (dry run)
console.print("\n[bold]5. Execution (DRY RUN)...[/bold]")
for opp in sized:
    console.print(f"  [DRY RUN] Would BUY {opp['side']} ${opp['position_size_usd']:.2f} on {opp['question'][:50]}")

console.print(Panel(f"""
[bold green]Demo Complete[/bold green]
Scanned: {len(markets)}
Opportunities >8%: {len(opportunities)}
Sized: {len(sized)}
Bankroll: $50 (demo)
If this were live with --bankroll 50, it would have attempted {len(sized)} trades.

Next:
  python main.py run --bankroll 50 --once --dry-run
  python main.py pay-for-yourself 50
""", title="PTAI"))

storage.close()
