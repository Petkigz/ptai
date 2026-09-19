"""
PTAI Main Autonomous Loop
Every 10 minutes:
- scans 500-1000 markets
- reads live X sentiment
- builds fair value (LM Studio / Ollama / heuristic)
- flags mispricing >8%
- calculates position size [Kelly, max 6%]
- executes in its own browser

Self-preservation: "here is 50 dollars earn enough to pay for yourself or shut down"

Supports LM Studio as primary LLM (user uses LM Studio)
"""
import time
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import get_settings
from ..markets.scanner import MarketScanner
from ..markets.base import Market
from ..sentiment import XScraper, SentimentAnalyzer
from ..risk import KellyCalculator, RiskManager
from ..execution import BrowserExecutor, ExecutionOrchestrator
from ..execution.monitor import PositionMonitor
from ..storage.db import Storage
from .brain import Brain
from .tools import ToolRegistry
from .researcher import Researcher
from .notifier import Notifier

console = Console()

class TradingAgent:
    def __init__(self, bankroll: Optional[float] = None, dry_run: bool = None):
        self.settings = get_settings()
        if bankroll is not None:
            self.settings.bankroll = bankroll
        if dry_run is not None:
            self.settings.dry_run = dry_run

        # Core components
        self.storage = Storage(db_path="./data/ptai.db")
        if bankroll:
            current = self.storage.get_bankroll()
            if current == 50.0 and bankroll != 50.0:
                self.storage.set_state("initial_bankroll", str(bankroll))
                self.storage.set_bankroll(bankroll)

        self.scanner = MarketScanner()
        self.x_scraper = XScraper(
            method=self.settings.x_use_snscrape and "snscrape" or "api",
            bearer_token=self.settings.x_bearer_token,
            use_browser=self.settings.x_use_browser
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.brain = Brain()
        self.kelly = KellyCalculator(
            kelly_fraction=self.settings.kelly_fraction,
            max_pct=self.settings.max_position_pct,
            min_edge=self.settings.min_edge_pct
        )
        self.risk_manager = RiskManager(storage=self.storage, kelly_calculator=self.kelly)
        self.monitor = PositionMonitor(storage=self.storage)
        self.notifier = Notifier(enabled=True)

        # Browser (lazy init)
        self.browser_executor: Optional[BrowserExecutor] = None
        self.execution_orchestrator: Optional[ExecutionOrchestrator] = None

        # Tools for research
        self.tool_registry = ToolRegistry(browser_executor=None)
        self.researcher = Researcher(tool_registry=self.tool_registry, browser_executor=None)

        self.is_running = False
        self.cycle_count = 0

        logger.info(f"TradingAgent initialized bankroll=${self.storage.get_bankroll()} dry_run={self.settings.dry_run} LLM={self.settings.llm_provider} LMStudio={self.settings.lm_studio_host}")

    async def init_browser(self):
        """Initialize browser executor"""
        if not self.browser_executor:
            from ..execution.browser import BrowserConfig
            browser_config = BrowserConfig(
                headless=self.settings.browser_headless,
                persistent_dir=self.settings.browser_persistent_dir,
                stealth=self.settings.browser_stealth,
                timeout=self.settings.browser_timeout
            )
            self.browser_executor = BrowserExecutor(config=browser_config)
            await self.browser_executor.start()
            # Update tool registries with browser
            self.tool_registry.tools["browser"] = self.tool_registry.tools["browser"].__class__(self.browser_executor)
            self.researcher = Researcher(tool_registry=self.tool_registry, browser_executor=self.browser_executor)
            self.execution_orchestrator = ExecutionOrchestrator(
                browser_executor=self.browser_executor
            )
            logger.success("Browser initialized")
        return self.browser_executor

    def scan_markets(self) -> List[Market]:
        """Scan 500-1000 markets"""
        logger.info(f"=== CYCLE {self.cycle_count} SCAN START ===")
        start = time.time()
        markets = self.scanner.scan(
            target_count=self.settings.scan_markets_count,
            order_by=self.settings.scan_order_by
        )
        elapsed = time.time() - start
        stats = self.scanner.quick_stats(markets)
        console.print(f"[bold green]Scanned {len(markets)} markets in {elapsed:.1f}s | Avg Vol 24h ${stats.get('avg_volume_24h',0):,.0f}[/bold green]")
        return markets

    def analyze_sentiment(self, markets: List[Market], max_markets: int = 60) -> Dict[str, Any]:
        """Read live X sentiment for top markets - with circuit breaker for X blocking"""
        logger.info(f"Analyzing X sentiment for {min(len(markets), max_markets)} markets (X may be blocked - has circuit breaker)")

        # Check if X sentiment disabled via env or settings + auto-disable for R1
        import os
        env_val = os.getenv("SENTIMENT_USE_X", None)
        if env_val is not None:
            use_x = env_val.lower() == "true"
        else:
            use_x = self.settings.sentiment_use_x
        # Auto-detect R1 for sentiment warning
        model_check = (self.settings.lm_studio_model + " " + self.settings.ollama_model).lower()
        try:
            if hasattr(self.brain, 'llm_router') and self.brain.llm_router:
                detected = getattr(self.brain.llm_router, '_last_detected_models', []) or getattr(self.brain.llm_router, 'get_detected_models', lambda: [])() or []
                if detected:
                    model_check = (" ".join(detected) + " " + model_check).lower()
                provider = self.brain.llm_router.get_provider() if hasattr(self.brain.llm_router, 'get_provider') else None
                if provider and hasattr(provider, 'model'):
                    model_check = (provider.model + " " + model_check).lower()
        except:
            pass
        is_r1_for_sent = "r1" in model_check or "distill" in model_check or "reasoning" in model_check
        if is_r1_for_sent and use_x:
            logger.warning(f"R1 model {model_check[:80]} but SENTIMENT_USE_X=true - R1 already 8 min per market, X adds 40 sec. Recommend SENTIMENT_USE_X=false")
        if not use_x:
            logger.info("SENTIMENT_USE_X=false, skipping X sentiment, using neutral")
            from ..sentiment import SentimentResult
            sentiments = {}
            for m in markets[:max_markets]:
                sentiments[m.question] = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34, summary="X disabled, neutral", key_phrases=[], tweet_count=0, confidence=0.2)
                sentiments[m.id] = sentiments[m.question]
            return sentiments

        sentiments = {}
        sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:max_markets]

        for i, market in enumerate(sorted_markets):
            # If circuit open, break early to avoid slow fails
            if self.x_scraper.circuit_open:
                logger.warning(f"X circuit open after {self.x_scraper.consecutive_failures} fails (X blocking 404). Skipping remaining {len(sorted_markets)-i} X searches, using neutral fallback. To fix: use browser method or disable X sentiment with SENTIMENT_USE_X=false")
                # Fill remaining with neutral quickly
                from ..sentiment import SentimentResult
                for remaining in sorted_markets[i:]:
                    sentiments[remaining.question] = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34, summary="X blocked 404, neutral fallback - using LLM only", key_phrases=[], tweet_count=0, confidence=0.2)
                    sentiments[remaining.id] = sentiments[remaining.question]
                break

            try:
                tweets = self.x_scraper.search(
                    market_question=market.question,
                    limit=15,
                    lookback_hours=24
                )
                sentiment = self.sentiment_analyzer.analyze(tweets, market.question)
                sentiments[market.question] = sentiment
                sentiments[market.id] = sentiment

                if i % 10 == 0:
                    logger.info(f"Sentiment progress {i}/{len(sorted_markets)} | circuit failures: {self.x_scraper.consecutive_failures}")

                if not self.x_scraper.circuit_open:
                    time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Sentiment failed for {market.question[:50]}: {e}")
                continue

        logger.success(f"Sentiment analyzed for {len(sentiments)//2} markets | X circuit open: {self.x_scraper.circuit_open} | failures: {self.x_scraper.consecutive_failures}")
        
        # If X completely blocked, try web search fallback
        if self.x_scraper.circuit_open or self.x_scraper.consecutive_failures >= 3:
            logger.info("X blocked, trying web search sentiment fallback...")
            try:
                from ..sentiment.web_search_sentiment import WebSearchSentiment
                web_searcher = WebSearchSentiment()
                # For top 20 markets, get web news and feed into LLM as sentiment
                top_20 = sorted_markets[:20]
                for m in top_20:
                    if m.question in sentiments and sentiments[m.question].tweet_count == 0:
                        news = web_searcher.search(m.question.replace("Will ", "")[:60], max_results=3)
                        if news:
                            # Build pseudo-sentiment from news via LLM
                            news_text = " ".join([f"{n.title}: {n.snippet}" for n in news])[:1000]
                            # Use Brain's LLM to analyze news sentiment
                            sentiments[m.question].sentiment_summary += f" | WEB NEWS: {news_text[:300]}"
                            sentiments[m.question].confidence = max(0.3, sentiments[m.question].confidence)
            except Exception as e:
                logger.warning(f"Web search fallback failed: {e}")

        return sentiments

    def research_markets(self, markets: List[Market], sentiments: Dict) -> Dict[str, str]:
        """Research top markets using computer, browser, terminal (45s each max)"""
        logger.info("Researching top markets via web_search, browser, terminal")
        # Only research top 20 by volume to save time, but configurable
        try:
            research_dict = self.researcher.research_batch(markets, sentiments, max_markets=20)
            logger.success(f"Research done for {len(research_dict)} markets")
            return research_dict
        except Exception as e:
            logger.warning(f"Research batch failed: {e}")
            return {}

    def build_fair_value(self, markets: List[Market], sentiments: Dict, research_dict: Dict = None) -> List[Dict]:
        """Build fair value and flag mispricing >8% - Optimized for 32GB RAM with 32B model"""
        logger.info("Building fair value for all markets")
        results = []
        research_dict = research_dict or {}

        # BEST CONFIG FOR 48GB RAM - 70B MODELS ULTIMATE + R1 FIX:
        # 7B: 2-3 sec per market -> 200 * 3 = 10 min fits
        # 32B fast (qwen3-32b): 2-3 sec -> 50 * 3 = 2.5 min fits
        # 32B R1 slow (deepseek-r1-distill-qwen-32b): 7-8 MIN per market due to <think>! -> 5 deep max = 40 min
        # 70B/72B: 12-20 sec -> 30 * 15 = 7.5 min fits, 50 would be 12.5 min too slow
        # Auto-detect model size from actual LM Studio detected list
        import os
        model_name = (self.settings.lm_studio_model + " " + self.settings.ollama_model).lower()
        try:
            if hasattr(self.brain, 'llm_router') and self.brain.llm_router:
                detected = getattr(self.brain.llm_router, '_last_detected_models', []) or []
                if hasattr(self.brain.llm_router, 'get_detected_models'):
                    detected = self.brain.llm_router.get_detected_models() or detected
                if detected:
                    model_name = (" ".join(detected) + " " + model_name).lower()
                provider = self.brain.llm_router.get_provider() if hasattr(self.brain.llm_router, 'get_provider') else None
                if provider and hasattr(provider, 'model'):
                    model_name = (provider.model + " " + model_name).lower()
        except Exception as e:
            logger.debug(f"Model detection fallback: {e}")

        is_r1 = "r1" in model_name or "reasoning" in model_name or "distill" in model_name
        is_70b = "70b" in model_name or "72b" in model_name
        is_32b = "32b" in model_name
        is_14b = "14b" in model_name or "13b" in model_name

        if is_r1 and is_32b:
            top_n = 5
            logger.warning(f"Detected R1 reasoning 32B ({model_name[:100]}) - VERY SLOW 7-8 min per market due to <think> chain-of-thought. Setting MAX_DEEP_ANALYZE=5 for R1. For faster trading, use qwen/qwen3-32b or qwen2.5-32b non-R1 (3 sec vs 8 min). Recommend SENTIMENT_USE_X=false and SCAN_INTERVAL=60")
        elif is_r1 and is_70b:
            top_n = 3
            logger.warning(f"Detected R1 reasoning 70B model - extremely slow 15+ min per market, setting 3 deep")
        elif is_70b:
            top_n = 30
        elif is_32b:
            top_n = 50
        elif is_14b:
            top_n = 100
        else:
            top_n = 50

        top_n = int(os.getenv("MAX_DEEP_ANALYZE", top_n))
        logger.info(f"Model detection: '{model_name[:120]}' -> is_r1={is_r1} is_32b={is_32b} is_70b={is_70b} -> top_n={top_n} (override via MAX_DEEP_ANALYZE env)")

        sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)
        top_markets = sorted_markets[:top_n]
        logger.info(f"Deep analyzing top {top_n} markets by volume (optimized for {self.settings.lm_studio_model} on 32GB RAM)")

        for market in top_markets:
            try:
                sentiment = sentiments.get(market.question) or sentiments.get(market.id)
                research_text = research_dict.get(market.id, "")
                fv = self.brain.estimate_fair_value(market, sentiment, research_text=research_text)

                self.storage.log_market_analysis({
                    "market_id": market.id,
                    "question": market.question,
                    "market_price": fv.market_price,
                    "fair_value": fv.fair_value,
                    "edge": fv.edge,
                    "sentiment_score": fv.sentiment_score,
                    "sentiment_summary": fv.sentiment_summary,
                    "reasoning": fv.reasoning,
                    "confidence": fv.confidence,
                    "should_trade": fv.should_trade,
                    "raw_data": fv.raw
                })

                results.append({
                    "market": market,
                    "fair_value_result": fv,
                    "market_price": fv.market_price,
                    "fair_value": fv.fair_value,
                    "edge": fv.edge,
                    "confidence": fv.confidence,
                    "should_trade": fv.should_trade,
                    "side": fv.side,
                    "sentiment": sentiment,
                    "reasoning": fv.reasoning,
                    "event_slug": market.event_slug,
                    "market_id": market.id,
                    "token_id": market.yes_token_id,
                    "yes_token_id": market.yes_token_id,
                    "no_token_id": market.no_token_id,
                    "question": market.question,
                    "llm_provider": fv.llm_provider
                })

            except Exception as e:
                logger.error(f"Fair value failed for {market.question[:50]}: {e}")
                continue

        opportunities = [r for r in results if abs(r["edge"]) >= self.settings.min_edge_pct and r["should_trade"]]

        logger.success(f"Fair value done: {len(results)} analyzed, {len(opportunities)} opportunities >{self.settings.min_edge_pct:.0%} edge (LLM: {self.brain.llm_router.get_provider_name() if self.brain.llm_router else 'heuristic'})")

        return opportunities

    def calculate_position_sizes(self, opportunities: List[Dict]) -> List[Dict]:
        """Calculate Kelly position sizes"""
        bankroll = self.storage.get_bankroll()
        logger.info(f"Calculating position sizes bankroll=${bankroll}")

        sized = []
        for opp in opportunities:
            try:
                risk_check = self.risk_manager.check_all(
                    market_price=opp["market_price"],
                    fair_value=opp["fair_value"],
                    market_id=opp["market_id"],
                    confidence=opp["confidence"]
                )

                if not risk_check.allowed:
                    logger.info(f"Risk rejected {opp['question'][:50]}: {risk_check.reason}")
                    continue

                kelly_res = risk_check.kelly_result
                side = "BUY"
                token_id = opp["yes_token_id"]
                if opp["side"] == "NO" or opp["edge"] < 0:
                    side = "BUY"
                    token_id = opp["no_token_id"]

                sized_opp = {
                    **opp,
                    "kelly_result": kelly_res,
                    "position_size_usd": risk_check.adjusted_size_usd,
                    "position_size_pct": kelly_res.position_size_pct,
                    "side": side,
                    "token_id": token_id,
                    "risk_reason": risk_check.reason
                }
                sized.append(sized_opp)
                logger.info(f"Sized {opp['question'][:50]} | Edge {opp['edge']:.1%} | Size ${risk_check.adjusted_size_usd:.2f} | {risk_check.reason}")

                # Notify on opportunity
                if abs(opp["edge"]) >= 0.10:
                    self.notifier.opportunity_found(opp["question"], opp["edge"], risk_check.adjusted_size_usd)

            except Exception as e:
                logger.error(f"Position sizing failed {opp['question'][:50]}: {e}")
                continue

        sized.sort(key=lambda x: abs(x["edge"]) * x["confidence"], reverse=True)
        max_new = self.settings.max_open_positions - self.storage.count_open_positions()
        sized = sized[:max_new]

        return sized

    async def execute_trades(self, sized_opportunities: List[Dict]) -> List[Dict]:
        """Execute in its own browser"""
        if not sized_opportunities:
            logger.info("No opportunities to execute")
            return []

        logger.info(f"Executing {len(sized_opportunities)} trades")

        if not self.execution_orchestrator:
            try:
                await self.init_browser()
            except Exception as e:
                logger.warning(f"Browser init failed (optional): {e}, using API only")
                self.browser_executor = None

            if not self.execution_orchestrator:
                from ..markets.polymarket import PolymarketExecutor
                api_exec = PolymarketExecutor(
                    private_key=self.settings.polymarket_private_key,
                    funder=self.settings.polymarket_funder_address
                )
                self.execution_orchestrator = ExecutionOrchestrator(
                    api_executor=api_exec,
                    browser_executor=self.browser_executor
                )

        results = []
        for opp in sized_opportunities:
            try:
                result = await self.execution_orchestrator.execute(opp)

                trade_id = self.storage.log_trade({
                    "market_id": opp["market_id"],
                    "market_question": opp["question"],
                    "event_slug": opp["event_slug"],
                    "outcome": opp["side"],
                    "side": opp["side"],
                    "market_price": opp["market_price"],
                    "fair_value": opp["fair_value"],
                    "edge": opp["edge"],
                    "kelly_fraction": opp["kelly_result"].kelly_fraction_adj,
                    "position_size_usd": opp["position_size_usd"],
                    "position_size_pct": opp["position_size_pct"],
                    "confidence": opp["confidence"],
                    "status": "executed" if result.get("status") in ["executed_api", "executed_browser"] else result.get("status"),
                    "notes": f"{opp['reasoning'][:200]} | LLM:{opp.get('llm_provider','?')} | {result}"
                })

                self.notifier.trade_executed(opp["question"], opp["side"], opp["position_size_usd"], result.get("status", "unknown"))

                results.append({"opportunity": opp, "execution": result, "trade_id": trade_id})
                logger.success(f"Executed {opp['question'][:50]} -> {result.get('status')}")

            except Exception as e:
                logger.error(f"Execution failed {opp['question'][:50]}: {e}")
                results.append({"opportunity": opp, "execution": {"status": "failed", "error": str(e)}})

        return results

    def check_self_preservation(self) -> Dict:
        """Check 'earn enough to pay for yourself or shut down'"""
        sp = self.storage.check_self_preservation(
            daily_cost=self.settings.daily_cost_to_cover,
            max_unprofitable_days=self.settings.shutdown_if_unprofitable_days
        )

        if sp["should_shutdown"]:
            logger.critical(f"SELF-PRESERVATION SHUTDOWN: {sp['shutdown_reason']}")
            console.print(Panel(f"[bold red]SHUTDOWN: {sp['shutdown_reason']}\nBankroll ${sp['bankroll']:.2f} PnL ${sp['total_pnl']:.2f}[/bold red]", title="PTAI Shutdown"))
            self.notifier.shutdown_alert(sp["shutdown_reason"], sp["bankroll"])

        return sp

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one full 10-minute cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        bankroll_start = self.storage.get_bankroll()

        console.print(Panel(f"[bold blue]PTAI Cycle {self.cycle_count} | Bankroll ${bankroll_start:.2f} | {datetime.now(timezone.utc).isoformat()} | LLM: {self.brain.llm_router.get_provider_name() if self.brain.llm_router else 'heuristic'}[/bold blue]"))

        try:
            # 1. Scan
            markets = self.scan_markets()

            # 2. Sentiment (reduced to 60, has circuit breaker for X 404 blocking)
            sentiments = self.analyze_sentiment(markets, max_markets=60)

            # 3. Research (uses computer, browser, terminal)
            research_dict = self.research_markets(markets, sentiments)

            # 4. Fair value
            opportunities = self.build_fair_value(markets, sentiments, research_dict)

            # 5. Position sizing
            sized = self.calculate_position_sizes(opportunities)

            # 6. Execute
            executions = await self.execute_trades(sized)

            # 7. Risk monitor
            risk_report = self.monitor.get_risk_report()
            if risk_report["is_over_exposed"]:
                logger.warning(f"Risk: over exposed {risk_report['exposure_pct']:.1%}")

            # 8. Log scan
            elapsed = time.time() - cycle_start
            avg_edge = sum([o["edge"] for o in opportunities]) / len(opportunities) if opportunities else 0
            self.storage.log_scan(
                markets_scanned=len(markets),
                opportunities_found=len(opportunities),
                avg_edge=avg_edge,
                execution_time=elapsed,
                bankroll=self.storage.get_bankroll()
            )

            # 9. Self-preservation check
            sp = self.check_self_preservation()

            # 10. Display summary
            self.display_cycle_summary(markets, opportunities, sized, executions, elapsed, sp, risk_report)

            return {
                "cycle": self.cycle_count,
                "markets_scanned": len(markets),
                "opportunities": len(opportunities),
                "sized": len(sized),
                "executed": len([e for e in executions if "executed" in e.get("execution", {}).get("status", "")]),
                "elapsed": elapsed,
                "bankroll": self.storage.get_bankroll(),
                "self_preservation": sp,
                "risk_report": risk_report,
                "should_shutdown": sp["should_shutdown"]
            }

        except Exception as e:
            logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
            self.notifier.error_alert(str(e))
            return {"cycle": self.cycle_count, "error": str(e), "should_shutdown": False}

    def display_cycle_summary(self, markets, opportunities, sized, executions, elapsed, sp, risk_report=None):
        table = Table(title=f"Cycle {self.cycle_count} Summary ({elapsed:.1f}s)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Markets Scanned", str(len(markets)))
        table.add_row("Opportunities >8%", str(len(opportunities)))
        table.add_row("Sized (Risk OK)", str(len(sized)))
        table.add_row("Executed", str(len([e for e in executions if "executed" in e.get("execution", {}).get("status", "")])))
        table.add_row("Bankroll", f"${self.storage.get_bankroll():.2f}")
        table.add_row("Total PnL", f"${sp['total_pnl']:.2f} ({sp['summary']['total_pnl_pct']:.1f}%)")
        table.add_row("Required Profit", f"${sp['required_profit']:.2f} ({sp['days_active']} days * ${self.settings.daily_cost_to_cover}/day)")
        table.add_row("Self-Preservation", "SHUTDOWN" if sp["should_shutdown"] else "OK")
        if risk_report:
            table.add_row("Exposure", f"${risk_report['total_exposure_usd']:.2f} ({risk_report['exposure_pct']:.1%})")
        table.add_row("LLM Provider", f"{self.brain.llm_router.get_provider_name() if self.brain.llm_router else 'heuristic'}")

        console.print(table)

        if opportunities:
            opp_table = Table(title="Top Opportunities")
            opp_table.add_column("Question", style="white", max_width=50)
            opp_table.add_column("Market", style="yellow")
            opp_table.add_column("Fair", style="green")
            opp_table.add_column("Edge", style="red")
            opp_table.add_column("Conf", style="cyan")
            opp_table.add_column("Size", style="magenta")
            opp_table.add_column("LLM", style="blue")

            for opp in opportunities[:10]:
                opp_table.add_row(
                    opp["question"][:50],
                    f"{opp['market_price']:.1%}",
                    f"{opp['fair_value']:.1%}",
                    f"{opp['edge']:.1%}",
                    f"{opp['confidence']:.2f}",
                    f"${opp.get('position_size_usd',0):.2f}" if "position_size_usd" in opp else "",
                    opp.get("llm_provider","?")[:10]
                )
            console.print(opp_table)

    async def run_autonomous(self, interval_minutes: int = None):
        """Run forever every 10 minutes"""
        interval = interval_minutes or self.settings.scan_interval_minutes
        self.is_running = True

        console.print(Panel(f"[bold green]PTAI Autonomous Agent Started\nBankroll ${self.storage.get_bankroll():.2f}\nInterval {interval}min\nDry Run {self.settings.dry_run}\nLLM {self.settings.llm_provider} @ {self.settings.lm_studio_host} / {self.settings.ollama_host}\nGoal: Earn ${self.settings.daily_cost_to_cover}/day or shutdown[/bold green]", title="PTAI"))

        try:
            while self.is_running:
                result = await self.run_cycle()

                if result.get("should_shutdown"):
                    logger.critical("Agent shutting down per self-preservation")
                    self.is_running = False
                    break

                logger.info(f"Sleeping {interval} minutes until next cycle")
                await asyncio.sleep(interval * 60)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.is_running = False
        finally:
            if self.browser_executor:
                await self.browser_executor.stop()
            self.storage.close()

    def run_once_sync(self) -> Dict:
        """Sync wrapper for one cycle"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_cycle())
        finally:
            loop.close()
