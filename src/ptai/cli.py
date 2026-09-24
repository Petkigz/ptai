"""
PTAI CLI - Command line interface
Supports: "here is 50 dollars earn enough to pay for yourself or shut down"
Supports LM Studio (primary) and Ollama
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from .config import get_settings
from .agent.loop import TradingAgent  # legacy; retained for `ptai premium` and history
from .agent.v3_loop import TradingAgentV3
from .storage.db import Storage
from .markets.scanner import MarketScanner
from .risk import KellyCalculator

app = typer.Typer(
    name="ptai",
    help="PTAI - Local Autonomous Trading Agent for Polymarket (LM Studio + Ollama)",
    add_completion=False
)
console = Console()

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
logger.add(
    "./logs/ptai.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

@app.command()
def init(
    bankroll: float = typer.Option(50.0, "--bankroll", "-b", help="Initial bankroll in USD"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run vs live trading"),
    model: str = typer.Option("local-model", "--model", "-m", help="LM Studio model (local-model for auto) or Ollama model"),
    llm_provider: str = typer.Option("auto", "--llm", help="LLM provider: auto, lm_studio, ollama"),
):
    """Initialize PTAI with bankroll and config - supports LM Studio"""
    settings = get_settings()
    storage = Storage()

    storage.set_state("initial_bankroll", str(bankroll))
    storage.set_bankroll(bankroll)
    storage.set_state("model", model)
    storage.set_state("llm_provider", llm_provider)

    console.print(Panel(f"[bold green]PTAI Initialized\nBankroll: ${bankroll}\nDry Run: {dry_run}\nModel: {model}\nLLM Provider: {llm_provider}\nDB: ./data/ptai.db[/bold green]", title="PTAI Init"))

    # Check LLM providers
    try:
        from .llm.provider import LLMRouter
        router = LLMRouter(preferred=llm_provider, ollama_host=settings.ollama_host, lm_studio_host=settings.lm_studio_host, model=model)
        console.print(f"[bold]LLM Router: {router.get_provider_name()}[/bold]")
        if router.is_available():
            console.print(f"[green]LLM available: {router.get_provider_name()}[/green]")
            if "LMStudio" in router.get_provider_name():
                models = router.provider.list_models() if hasattr(router.provider, 'list_models') else []
                console.print(f"LM Studio models: {models[:5]}")
        else:
            console.print(f"[yellow]No LLM detected. For LM Studio:\n1. Open LM Studio\n2. Developer tab -> Start Server (port 1234)\n3. Load a model\n4. Will use heuristic fallback until LLM available[/yellow]")
    except Exception as e:
        console.print(f"[yellow]LLM check failed: {e} - will use heuristic fallback[/yellow]")

    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright
        console.print("[green]Playwright installed[/green]")
    except:
        console.print("[yellow]Playwright not installed, run: pip install playwright && playwright install chromium[/yellow]")

    # Check snscrape
    try:
        import snscrape
        console.print("[green]snscrape installed (X sentiment)[/green]")
    except:
        console.print("[yellow]snscrape not installed, run: pip install snscrape (optional, for X sentiment)[/yellow]")

    storage.close()

@app.command()
def check_llm(
    provider: str = typer.Option("auto", "--provider", help="auto, lm_studio, ollama"),
    host: str = typer.Option(None, "--host", help="Override host, e.g. http://localhost:1234"),
):
    """Check which local LLM is available (LM Studio / Ollama)"""
    settings = get_settings()
    from .llm.provider import LLMRouter

    lm_host = host if provider in ["lm_studio", "auto"] else settings.lm_studio_host
    if provider == "lm_studio" and host is None:
        lm_host = settings.lm_studio_host
    elif provider == "ollama" and host is None:
        lm_host = settings.ollama_host

    router = LLMRouter(
        preferred=provider,
        ollama_host=settings.ollama_host,
        lm_studio_host=lm_host,
        model=settings.lm_studio_model
    )

    table = Table(title="LLM Status")
    table.add_column("Provider")
    table.add_column("Host")
    table.add_column("Available")
    table.add_column("Models")

    # Check LM Studio
    from .llm.provider import LMStudioProvider, OllamaProvider
    lm = LMStudioProvider(host=settings.lm_studio_host, model=settings.lm_studio_model)
    ollama = OllamaProvider(host=settings.ollama_host, model=settings.ollama_model)

    table.add_row("LM Studio", settings.lm_studio_host, "✅" if lm.is_available() else "❌", str(lm.list_models()[:3]) if lm.is_available() else "Not running - start LM Studio server")
    table.add_row("Ollama", settings.ollama_host, "✅" if ollama.is_available() else "❌", "llama3.1:8b etc" if ollama.is_available() else "Not running - ollama serve")

    console.print(table)
    console.print(f"\n[bold]Active provider: {router.get_provider_name()}[/bold]")

    if router.is_available():
        # Test chat
        console.print("\nTesting LLM with fair value prompt...")
        resp = router.chat("What is 0.6 * 0.7? Respond JSON {\"answer\": 0.42}", system="You are a math expert, respond only JSON")
        if resp:
            console.print(f"[green]LLM response: {resp.content[:500]}[/green]")
        else:
            console.print("[red]LLM test failed[/red]")
    else:
        console.print("\n[yellow]No LLM available. Agent will use heuristic fallback (still works, but less intelligent).\nFor LM Studio: Open LM Studio -> Developer -> Start Server -> Load model[/yellow]")

@app.command()
def scan(
    count: int = typer.Option(750, "--count", "-c", help="Number of markets to scan"),
    order: str = typer.Option("volume24hr", "--order", help="Order by: volume24hr, liquidity, volume")
):
    """Scan markets only (no trading)"""
    scanner = MarketScanner()
    markets = scanner.scan(target_count=count, order_by=order)

    table = Table(title=f"Scanned {len(markets)} Markets")
    table.add_column("Question", max_width=60)
    table.add_column("YES Price")
    table.add_column("Vol 24h")
    table.add_column("Liquidity")
    table.add_column("Slug")

    for m in markets[:20]:
        table.add_row(
            m.question[:60],
            f"{m.yes_price:.1%}",
            f"${m.volume_24h:,.0f}",
            f"${m.liquidity:,.0f}",
            m.event_slug[:30]
        )

    console.print(table)
    console.print(f"Stats: {scanner.quick_stats(markets)}")

# ---------------------------------------------------------------------------
# V3 runtime helpers
# ---------------------------------------------------------------------------

def _seed_bankroll(amount: float) -> None:
    """
    Persist the bankroll before constructing V3.

    TradingAgentV3 sizes everything - Kelly, the 6% cap, exposure limits,
    drawdown - from `storage.get_performance_summary()["bankroll"]`, not from a
    constructor argument. Without this the CLI's --bankroll flag would be
    ignored and the agent would size against whatever was last stored.
    """
    try:
        from .storage.db import Storage
        storage = Storage()
        storage.set_bankroll(float(amount))
        storage.close()
        logger.debug(f"Seeded bankroll ${amount}")
    except Exception as e:
        logger.warning(f"Could not seed bankroll ${amount}: {type(e).__name__}: {e}")


def _print_cycle_result(result: dict) -> None:
    """
    Print a cycle summary rather than the raw dict.

    A V3 result is large (per-venue reports, alpha output, betting card,
    execution detail); dumping it to the terminal buries the one line the
    operator needs.
    """
    status = result.get("status", "unknown")
    if status == "blocked":
        console.print(f"[red]Cycle blocked:[/red] {result.get('reason')}")
        return

    disc = result.get("discovery", {}) or {}
    opps = result.get("opportunities", {}) or {}
    execs = result.get("execution", []) or []

    table = Table(title=f"V3 Cycle: {status}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Markets scanned", str(disc.get("total_scanned", 0)))
    table.add_row("Venues reporting", str(len(disc.get("per_venue", {}) or {})))
    table.add_row("Candidates", str(opps.get("total_candidates", 0)))
    table.add_row("Tradeable after costs", str(opps.get("total_tradeable", 0)))
    table.add_row("Executed", str(len(execs)))
    table.add_row("DO NOTHING", "yes" if result.get("do_nothing_success") else "no")
    table.add_row("Time", f"{result.get('execution_time', 0):.1f}s")
    console.print(table)

    best = opps.get("best") or {}
    if best.get("question") and best.get("question") != "DO NOTHING":
        console.print(
            f"[green]Best:[/green] {best.get('venue')} / {best.get('strategy')} "
            f"- edge {best.get('edge', 0):.3f} score {best.get('score', 0):.3f}")
        if best.get("reasoning"):
            console.print(f"[dim]{best['reasoning'][:200]}[/dim]")
    else:
        console.print("[dim]No opportunity cleared the gates. DO NOTHING is a "
                      "successful outcome, not a failure.[/dim]")

    betting = result.get("betting") or {}
    if betting.get("events"):
        console.print(
            f"[dim]Betting: {betting.get('events')} fixtures, "
            f"{betting.get('markets_scanned_by_type', {})} markets, "
            f"mode {betting.get('data_mode')}[/dim]")


@app.command()
def run(
    bankroll: Optional[float] = typer.Option(None, "--bankroll", "-b", help="Bankroll override"),
    once: bool = typer.Option(False, "--once", help="Run one cycle only"),
    interval: int = typer.Option(10, "--interval", "-i", help="Interval minutes"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run"),
    headless: bool = typer.Option(False, "--headless", help="Browser headless"),
    llm: str = typer.Option("auto", "--llm", help="LLM provider: auto, lm_studio, ollama"),
    country: str = typer.Option(
        "UG", "--country",
        help="Jurisdiction used for venue eligibility (e.g. UG, GB, US)"),
):
    """
    Run autonomous agent

    Example: here is 50 dollars earn enough to pay for yourself or shut down
    -> ptai run --bankroll 50 --interval 10 --llm lm_studio
    """
    settings = get_settings()
    if bankroll:
        settings.bankroll = bankroll
    if headless:
        settings.browser_headless = True
    settings.dry_run = dry_run
    settings.llm_provider = llm

    console.print(Panel(
        f"[bold]Bankroll: ${bankroll or settings.bankroll}\n"
        f"Interval: {interval}min\n"
        f"Dry Run: {dry_run}\n"
        f"Country: {country}\n"
        f"Headless: {headless}\n"
        f"LLM: {llm} (LM Studio {settings.lm_studio_host} / Ollama {settings.ollama_host})\n"
        f"Objective: grow capital under a risk budget[/bold]",
        title="PTAI Autonomous Agent"
    ))

    # V3 is the canonical pipeline (qualification -> multi-venue discovery ->
    # multi-strategy evaluation -> expected net EV -> Kelly -> exposure ->
    # ExecutionGuard -> MultiVenueExecutor -> exact adapter). This command used
    # to launch the legacy agent/loop.py TradingAgent, so every V3 improvement
    # sat beside the runtime instead of underneath it.
    _seed_bankroll(bankroll or settings.bankroll)

    agent = TradingAgentV3(country_code=country, dry_run=dry_run)

    async def _run():
        if once:
            result = await agent.run_cycle()
            _print_cycle_result(result)
        else:
            console.print(
                "[dim]V3 runs the qualification pipeline each cycle. "
                "Ctrl-C to stop.[/dim]")
            await agent.run_continuous(interval_minutes=interval)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("[yellow]Stopped by user[/yellow]")

@app.command("console")
def console_command(
    port: int = typer.Option(8101, "--port", help="Port for the console"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
):
    """
    Open the operator console: capital, mode switch, orders, results.

    Starts in PAPER mode. Paper simulates the whole loop against the real
    orderbook and moves no money, so it needs neither capital nor credentials -
    which is why it is where a new install should start.

    Live mode is refused from the UI until a venue is both funded and authorised,
    because an armed system that cannot fire is worse than an honest paper one.
    """
    from .ui.console import main as console_main

    console.print(Panel(
        f"[bold]PTAI Console[/bold]\n"
        f"http://localhost:{port}\n\n"
        f"Starts in PAPER mode. Nothing is sent to a venue until you switch to "
        f"live, and live is refused unless a venue is funded and authorised.\n\n"
        f"[dim]Funding: money goes to the VENUE account, not to the agent. "
        f"See the Capital & Funding tab.[/dim]",
        title="Operator Console"
    ))
    console_main(host=host, port=port)

@app.command()
def status():
    """Show agent status and performance"""
    storage = Storage()
    perf = storage.get_performance_summary()
    sp = storage.check_self_preservation()

    # Capital first, as separate quantities. "Bankroll: $47" does not tell an
    # operator how much of it is already committed to an open position.
    from .execution.position_ledger import PositionLedgerBuilder
    ledger = PositionLedgerBuilder(storage=storage).build()

    capital = Table(title="Capital")
    capital.add_column("Metric", style="cyan")
    capital.add_column("Value", style="magenta")
    capital.add_row("Equity", f"${ledger.equity:.2f}")
    capital.add_row("Free capital", f"${ledger.free_cash:.2f}")
    capital.add_row("Reserved capital",
                    f"${ledger.reserved_capital:.2f} ({ledger.reserved_pct:.1f}%)")
    capital.add_row("Open position value", f"${ledger.open_position_value:.2f}")
    capital.add_row("Realised P&L", f"${ledger.realised_pnl:+.2f}")
    capital.add_row("Unrealised P&L",
                    f"${ledger.unrealised_pnl:+.2f}"
                    + (" [dim](unmarked positions carried at cost)[/dim]"
                       if any("carried at cost" in w for w in ledger.warnings) else ""))
    capital.add_row("Positions", f"{ledger.live_position_count} live, "
                                 f"{ledger.paper_position_count} paper")
    console.print(capital)

    performance = Table(title="Performance")
    performance.add_column("Metric", style="cyan")
    performance.add_column("Value", style="magenta")
    performance.add_row("Initial", f"${perf['initial_bankroll']:.2f}")
    performance.add_row("Total return",
                        f"${perf['total_pnl']:+.2f} ({perf['total_pnl_pct']:+.1f}%)")
    performance.add_row("Total trades", str(perf["total_trades"]))
    performance.add_row("Win rate", f"{perf['win_rate']:.1f}%" if perf['win_rate'] else "no resolved trades yet")
    performance.add_row("Days active", str(sp["days_active"]))
    # Drawdown is the risk figure that matters; the daily-cost comparison is
    # advisory context about running costs, not a target the agent is judged on.
    performance.add_row("Max drawdown", f"{min(0.0, perf['total_pnl_pct']):.1f}%")
    performance.add_row("Running cost comparison",
                        f"${sp['required_profit']:.2f} budgeted "
                        f"({'covered' if sp['is_profitable_enough'] else 'not covered'}) "
                        f"[dim]- operator's budget, not a target[/dim]")
    console.print(performance)

    risk = Table(title="Risk state")
    risk.add_column("Metric", style="cyan")
    risk.add_column("Value", style="magenta")
    risk.add_row("Unprofitable streak", str(sp["unprofitable_streak"]))
    risk.add_row("Trading halted",
                 "[red]YES[/red]" if sp["should_shutdown"] else "[green]NO[/green]")
    if sp["should_shutdown"]:
        risk.add_row("Reason", str(sp.get("shutdown_reason", "")))
    console.print(risk)

    for warning in ledger.warnings:
        console.print(f"[yellow]Ledger warning: {warning}[/yellow]")

    trades = storage.get_recent_trades(10)
    if trades:
        t_table = Table(title="Recent Trades")
        t_table.add_column("Time")
        t_table.add_column("Question", max_width=40)
        t_table.add_column("Edge")
        t_table.add_column("Size")
        t_table.add_column("Status")

        for tr in trades:
            t_table.add_row(
                tr["timestamp"][:16],
                tr["market_question"][:40] if tr["market_question"] else tr["market_id"][:20],
                f"{tr['edge']:.1%}" if tr["edge"] else "N/A",
                f"${tr['position_size_usd']:.2f}" if tr["position_size_usd"] else "N/A",
                tr["status"]
            )
        console.print(t_table)

    storage.close()

@app.command()
def trade(
    market_id: str = typer.Argument(..., help="Market ID or slug"),
    side: str = typer.Option("YES", "--side", help="YES or NO"),
    amount: float = typer.Option(5.0, "--amount", "-a", help="Amount USD"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run")
):
    """Manually trade a market"""
    from .markets.polymarket import PolymarketClient, PolymarketExecutor
    client = PolymarketClient()
    markets = client.scan_markets(target_count=100, order_by="volume24hr")
    target = None
    for m in markets:
        if market_id in m.id or market_id in m.slug or market_id in m.event_slug or market_id.lower() in m.question.lower():
            target = m
            break

    if not target:
        console.print(f"[red]Market not found: {market_id}[/red]")
        return

    console.print(f"Found: {target.question} YES {target.yes_price:.1%} Vol ${target.volume_24h:,.0f}")

    storage = Storage()
    bankroll = storage.get_bankroll()
    kelly = KellyCalculator()
    fair = typer.prompt("Enter your fair value (0-1)", default=0.6, type=float)
    res = kelly.calculate(target.yes_price, fair, bankroll)
    console.print(f"Kelly: {res.reason} Size ${res.position_size_usd:.2f}")

    if typer.confirm(f"Execute {side} ${amount} on {target.question[:50]}?"):
        executor = PolymarketExecutor(
            private_key=get_settings().polymarket_private_key,
            funder=get_settings().polymarket_funder_address
        )
        token_id = target.yes_token_id if side == "YES" else target.no_token_id
        result = executor.place_order(token_id, target.yes_price if side=="YES" else target.no_price, amount/target.yes_price, side="BUY", dry_run=dry_run)
        console.print(f"Result: {result}")

    storage.close()

@app.command()
def shutdown(
    force: bool = typer.Option(False, "--force", help="Force shutdown even if profitable")
):
    """Shutdown agent and export report"""
    storage = Storage()
    sp = storage.check_self_preservation()
    perf = storage.get_performance_summary()

    console.print(Panel(f"[bold]Shutting down PTAI\nBankroll ${perf['bankroll']:.2f}\nPnL ${perf['total_pnl']:.2f}[/bold]", title="Shutdown"))

    import csv
    trades = storage.get_recent_trades(1000)
    Path("./data").mkdir(exist_ok=True)
    with open("./data/final_report.csv", "w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

    console.print(f"Exported {len(trades)} trades to ./data/final_report.csv")

    if sp["should_shutdown"] or force:
        storage.set_state("shutdown", "true")
        storage.set_state("shutdown_reason", sp.get("shutdown_reason", "manual"))
        console.print("[red]Agent marked as shutdown[/red]")
    else:
        console.print("[yellow]Agent not in shutdown condition, use --force to override[/yellow]")

    storage.close()

@app.command()
def pay_for_yourself(
    bankroll: float = typer.Argument(50.0, help="Here is X dollars"),
    daily_cost: float = typer.Option(5.0, "--daily-cost", help="Daily cost to cover"),
    interval: int = typer.Option(10, "--interval", help="Scan interval minutes"),
    llm: str = typer.Option("auto", "--llm", help="LLM provider: auto, lm_studio, ollama"),
    country: str = typer.Option(
        "UG", "--country",
        help="Jurisdiction used for venue eligibility (e.g. UG, GB, US)"),
):
    """
    The self-preservation command: "here is 50 dollars earn enough to pay for yourself or shut down"

    This is the main entry point for the requested behavior.
    Supports LM Studio (you use LM Studio).
    """
    console.print(Panel(
        f"[bold green]Received ${bankroll} - Task: Earn enough to pay for yourself or shut down\n"
        f"Daily cost: ${daily_cost}\n"
        f"Bankroll: ${bankroll}\n"
        f"Interval: {interval}min\n"
        f"Country: {country}\n"
        f"LLM: {llm} (LM Studio at http://localhost:1234)\n"
        f"Strategy: Scan 500-1000 markets, X sentiment, fair value, mispricing >8%, Kelly max 6%[/bold green]",
        title="PTAI Self-Preservation Mode - LM Studio Ready"
    ))

    storage = Storage()
    storage.set_state("initial_bankroll", str(bankroll))
    storage.set_bankroll(bankroll)
    storage.set_state("daily_cost", str(daily_cost))
    storage.close()

    settings = get_settings()
    settings.llm_provider = llm

    # The self-preservation command is the one the user actually runs, and it
    # was launching the legacy loop. It now runs V3, which is the pipeline that
    # carries expected net EV, the execution guard and exact venue routing.
    agent = TradingAgentV3(country_code=country, dry_run=settings.dry_run)

    console.print(
        f"[dim]Data mode {agent.data_mode.value} "
        f"(dry_run={agent.dry_run}). Real orders require dry_run=False AND a "
        f"verified account; until the account probe can place and cancel a test "
        f"order, live capital stays disabled.[/dim]")

    async def _run():
        await agent.run_continuous(interval_minutes=interval)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("[yellow]Stopped[/yellow]")

@app.command()
def premium(
    bankroll: float = typer.Argument(50.0, help="Here is X dollars"),
    daily_cost: float = typer.Option(5.0, "--daily-cost", help="Daily cost"),
    interval: int = typer.Option(10, "--interval", help="Interval minutes"),
    user_id: str = typer.Option(None, "--user", help="User ID for multi-tenant"),
):
    """
    PREMIUM MODE: AI Teammates you can give real work to
    Bots sign in to tools, use them just like you do, come back with finished work
    Team: Scout, SentimentAnalyst, Researcher, Quant, RiskOfficer, Trader, Coach
    Premium features: Vault, Memory, Learning, Backtest, Multi-user, Notifications
    """
    console.print(Panel(
        f"[bold green]PTAI PREMIUM - AI Teammates Mode\n"
        f"Bankroll: ${bankroll} | Daily cost: ${daily_cost} | Interval: {interval}min\n"
        f"Team: 7 AI Teammates - Scout, Sentiment, Researcher, Quant, RiskOfficer, Trader, Coach\n"
        f"Premium: Vault (tool sign-in), Memory (learning), Backtest, Multi-user, Discord/Telegram\n"
        f"User: {user_id or 'default'} | Goal: Earn enough to pay for yourself or shutdown[/bold green]",
        title="PTAI PREMIUM"
    ))
    
    from .agent.premium_loop import PremiumTradingAgent
    
    agent = PremiumTradingAgent(user_id=user_id, bankroll=bankroll)
    
    async def _run():
        await agent.run_autonomous(interval_minutes=interval)
    
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("[yellow]Premium stopped[/yellow]")

@app.command()
def teammates(
    action: str = typer.Argument("status", help="status, run, list"),
    count: int = typer.Option(500, "--count", help="Markets to scan"),
):
    """Manage AI Teammates - give real work to bots"""
    if action == "status":
        from .vault import Vault
        from .memory import Memory
        from .storage.db import Storage
        from .agent.teammates.coordinator import TeamCoordinator
        from .agent.brain import Brain
        from .risk import KellyCalculator, RiskManager
        from .execution.monitor import PositionMonitor
        from .agent.notifier import Notifier
        from .config import get_settings
        
        settings = get_settings()
        storage = Storage(db_path="./data/ptai.db")
        vault = Vault()
        memory = Memory()
        brain = Brain()
        kelly = KellyCalculator(kelly_fraction=settings.kelly_fraction, max_pct=settings.max_position_pct, min_edge=settings.min_edge_pct)
        risk_manager = RiskManager(storage=storage, kelly_calculator=kelly)
        monitor = PositionMonitor(storage=storage)
        notifier = Notifier(enabled=True)
        
        coordinator = TeamCoordinator(vault=vault, memory=memory, storage=storage, brain=brain, kelly=kelly, risk_manager=risk_manager, monitor=monitor, notifier=notifier)
        coordinator.display_team_status()
        
        # Vault status
        vault_status = vault.list_all()
        console.print(f"\n[bold]Vault: {sum(len(v) for v in vault_status.values())} tool sign-ins across {len(vault_status)} teammates[/bold]")
        for teammate, tools in vault_status.items():
            console.print(f"  {teammate}: {', '.join(tools)}")
        
        storage.close()
        memory.close()
    
    elif action == "run":
        console.print(f"[bold]Running team cycle with {count} markets...[/bold]")
        from .agent.premium_loop import PremiumTradingAgent
        agent = PremiumTradingAgent(bankroll=50)
        result = agent.run_once_sync()
        console.print(f"Result: {result}")

@app.command()
def memory(
    action: str = typer.Argument("insights", help="insights, recall, calibration, stats"),
    query: str = typer.Option("", "--query", help="Search query"),
    limit: int = typer.Option(20, "--limit", help="Limit"),
):
    """Memory & Learning - intelligence that improves over time"""
    from .memory import Memory
    mem = Memory()
    
    if action == "insights":
        insights = mem.get_insights(limit=limit)
        table = Table(title=f"Memory Insights ({len(insights)})")
        table.add_column("Time")
        table.add_column("Type")
        table.add_column("Content", max_width=60)
        table.add_column("Importance")
        for ins in insights:
            table.add_row(ins.created_at[:16], ins.type, ins.content[:60], str(ins.importance))
        console.print(table)
        
        cal = mem.get_calibration_stats()
        console.print(f"\nCalibration: {cal}")
    
    elif action == "recall":
        results = mem.recall(query=query, limit=limit)
        table = Table(title=f"Recall '{query}' ({len(results)})")
        table.add_column("Time")
        table.add_column("Type")
        table.add_column("Content", max_width=80)
        for r in results:
            table.add_row(r.created_at[:16], r.type, r.content[:80])
        console.print(table)
    
    elif action == "stats":
        cal = mem.get_calibration_stats()
        console.print(f"Calibration Stats: {cal}")
        console.print(f"Total memories: {mem.conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]}")
    
    mem.close()

@app.command()
def vault(
    action: str = typer.Argument("status", help="status, list, clear"),
    teammate: str = typer.Option("", "--teammate", help="Teammate name"),
    tool: str = typer.Option("", "--tool", help="Tool name"),
):
    """Vault - Tool sign-in management, bots sign in like you do"""
    from .vault import Vault
    v = Vault()
    
    if action == "status" or action == "list":
        all_tools = v.list_all()
        table = Table(title=f"Vault - {sum(len(t) for t in all_tools.values())} tool sign-ins")
        table.add_column("Teammate")
        table.add_column("Tools")
        table.add_column("Count")
        for tm, tools in all_tools.items():
            table.add_row(tm, ", ".join(tools), str(len(tools)))
        console.print(table)
        if not all_tools:
            console.print("[yellow]No tools signed in yet - teammates auto sign in when they work[/yellow]")
    
    elif action == "clear":
        if teammate and tool:
            if v.revoke(teammate, tool):
                console.print(f"[green]Revoked {teammate} -> {tool}[/green]")
            else:
                console.print(f"[red]Not found {teammate} -> {tool}[/red]")
        else:
            console.print("Use --teammate and --tool to revoke specific")

@app.command()
def backtest(
    days: int = typer.Option(30, "--days", help="Days to backtest"),
    bankroll: float = typer.Option(50.0, "--bankroll", help="Initial bankroll"),
    edge: float = typer.Option(8.0, "--edge", help="Min edge %"),
):
    """Backtest - Test strategies on historical data before live (premium)"""
    from .backtest import BacktestEngine, HistoricalDataProvider
    engine = BacktestEngine()
    config = {"name": f"Backtest {days}d edge {edge}%", "bankroll": bankroll, "min_edge": edge/100, "max_pos_pct": 0.06, "kelly_fraction": 0.5}

    # The V9 gate refuses synthetic data, and it is right to - but that means
    # real resolved markets have to be fetched first, or the command raises on
    # every invocation.
    provider = HistoricalDataProvider()
    dataset = provider.build_dataset(days_back=max(days, 1))
    if not dataset.ok:
        console.print(f"[yellow]No historical data available: {(dataset.warnings or ['unknown'])[0]}[/yellow]")
        console.print("[yellow]Backtest needs real resolved markets; synthetic data is refused by the V9 gate.[/yellow]")
        raise typer.Exit(code=1)
    if dataset.is_baseline:
        console.print("[yellow]No recorded strategy signal for these markets, so this is a[/yellow]")
        console.print("[yellow]NO-SKILL BASELINE: edge is zero and no trades are taken. It[/yellow]")
        console.print("[yellow]measures costs, not skill, and cannot authorise live trading.[/yellow]")

    result = engine.run(strategy_config=config, dataset=dataset, days=days)
    for w in (result.warnings or [])[:3]:
        console.print(f"[dim]{w}[/dim]")
    
    table = Table(title=f"Backtest Result: {result.strategy_name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Initial", f"${result.initial_bankroll:.2f}")
    table.add_row("Final", f"${result.final_bankroll:.2f}")
    table.add_row("PnL", f"${result.total_pnl:.2f} ({result.total_pnl_pct:.1%})")
    table.add_row("Trades", str(result.total_trades))
    table.add_row("Win Rate", f"{result.win_rate:.1f}% ({result.winning_trades}W/{result.losing_trades}L)")
    table.add_row("Avg Edge", f"{result.avg_edge:.1%}")
    table.add_row("Max DD", f"{result.max_drawdown:.1%}")
    table.add_row("Sharpe", f"{result.sharpe:.2f}")
    console.print(table)
    
    # Equity curve
    console.print("\n[bold]Equity Curve (last 20 points):[/bold]")
    for point in result.equity_curve[-20:]:
        console.print(f"  {point['date'][:10]} -> ${point['bankroll']:.2f}")

@app.command()
def users(
    action: str = typer.Argument("list", help="list, create, paths"),
    email: str = typer.Option("", "--email", help="User email"),
    name: str = typer.Option("", "--name", help="User name"),
    bankroll: float = typer.Option(50.0, "--bankroll", help="Bankroll"),
    plan: str = typer.Option("free", "--plan", help="free, pro, premium"),
    user_id: str = typer.Option("", "--user-id", help="User ID"),
):
    """Multi-user management for product - each user isolated"""
    from .users import UserManager
    manager = UserManager()
    
    if action == "list":
        users_list = manager.list_users()
        table = Table(title=f"Users ({len(users_list)}) - Multi-tenant Product")
        table.add_column("ID")
        table.add_column("Email")
        table.add_column("Name")
        table.add_column("Bankroll")
        table.add_column("Plan")
        table.add_column("Created")
        for u in users_list:
            table.add_row(u.id[:8], u.email, u.name, f"${u.bankroll:.2f}", u.plan, u.created_at[:16])
        console.print(table)
    
    elif action == "create":
        if not email or "@" not in email:
            console.print("[red]Valid --email required[/red]")
            return
        user = manager.create_user(email=email, name=name, bankroll=bankroll, plan=plan)
        console.print(f"[green]Created user {user.email} ID {user.id} plan {user.plan} bankroll ${user.bankroll}[/green]")
        paths = manager.get_user_paths(user.id)
        console.print(f"Paths: {paths}")
    
    elif action == "paths":
        if not user_id:
            console.print("[red]--user-id required[/red]")
            return
        paths = manager.get_user_paths(user_id)
        console.print(f"User {user_id} paths:")
        for k, v in paths.items():
            console.print(f"  {k}: {v} exists={v.exists()}")
    
    manager.close()

@app.command()
def bots(
    action: str = typer.Argument("list", help="list, create, default_team, run"),
    name: str = typer.Option("", "--name", help="Bot name"),
    bot_type: str = typer.Option("custom", "--type", help="project, outbound, systems, scout, researcher, trader, custom"),
    role: str = typer.Option("", "--role", help="Role description"),
    bot_id: str = typer.Option("", "--bot-id", help="Bot ID for task"),
    title: str = typer.Option("", "--title", help="Task title"),
):
    """Dynamic Bots - Create a Bot, give task, add another when work grows - parallel 24/7"""
    from .bots import BotManager
    from .vault import Vault
    from .memory import Memory
    
    vault = Vault()
    memory = Memory()
    manager = BotManager(vault=vault, memory=memory)
    
    if action == "list":
        if len(manager.bots) == 0:
            console.print("[yellow]No bots - creating default team: Project, Outbound, Systems, Scout, Researcher, Trader[/yellow]")
            manager.create_default_team()
        
        table = Table(title=f"Dynamic Bots ({len(manager.bots)}) - Work in Parallel 24/7")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Role", max_width=30)
        table.add_column("Status")
        table.add_column("Tasks")
        table.add_column("Learning")
        for bot in manager.list_bots():
            s = bot.get_status()
            table.add_row(s.id, s.name, s.type, s.role[:30], "BUSY" if s.is_busy else "IDLE", f"{s.tasks_completed} done", f"{s.learning_score:.2f} ctx {s.context_size}")
        console.print(table)
    
    elif action == "create":
        if not name:
            console.print("[red]--name required[/red]")
            return
        bot = manager.create_bot(name=name, bot_type=bot_type, role=role)
        console.print(f"[green]Created bot {bot.name} ID {bot.id} type {bot.type} - works in parallel 24/7, signs into tools[/green]")
    
    elif action == "default_team":
        bots = manager.create_default_team()
        console.print(f"[green]Default team created: {len(bots)} bots - Project, Outbound, Systems, Scout, Researcher, Trader[/green]")
    
    memory.close()

@app.command()
def projects(
    action: str = typer.Argument("list", help="list, create"),
    name: str = typer.Option("", "--name", help="Project name"),
    goal: str = typer.Option("", "--goal", help="Project goal"),
    description: str = typer.Option("", "--description", help="Description"),
):
    """Projects - Bots take projects from start to end, keep context, get smarter"""
    from .projects import ProjectManager
    pm = ProjectManager()
    
    if action == "list":
        projects_list = pm.list_projects()
        table = Table(title=f"Projects ({len(projects_list)}) - Start to End")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Goal", max_width=40)
        table.add_column("Status")
        table.add_column("Progress")
        table.add_column("Tasks")
        table.add_column("Bots")
        for p in projects_list:
            table.add_row(p.id, p.name, p.goal[:40], p.status, f"{p.progress*100:.0f}%", f"{len([t for t in p.tasks if t.status=='completed'])}/{len(p.tasks)}", ",".join(p.assigned_bots[:2]))
        console.print(table)
    
    elif action == "create":
        if not name:
            console.print("[red]--name required[/red]")
            return
        project = pm.create_project(name=name, description=description, goal=goal)
        console.print(f"[green]Created project {project.name} ID {project.id} goal: {goal} - bots will take from start to end[/green]")

@app.command()
def routines(
    action: str = typer.Argument("list", help="list, record, run"),
    name: str = typer.Option("", "--name", help="Routine name"),
    routine_id: str = typer.Option("", "--id", help="Routine ID"),
):
    """Routines - Show a Bot how it's done, it saves as routine and runs on own next time"""
    from .routines import RoutineManager, RoutineRecorder
    
    if action == "list":
        rm = RoutineManager()
        routines_list = rm.list_routines()
        table = Table(title=f"Routines ({len(routines_list)}) - Show How It's Done")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Description", max_width=40)
        table.add_column("Steps")
        table.add_column("Runs")
        table.add_column("Success")
        for r in routines_list:
            table.add_row(r.id, r.name, r.description[:40], str(len(r.steps)), str(r.run_count), str(r.success_count))
        console.print(table)
        if not routines_list:
            console.print("[yellow]No routines yet - show Bot how it's done: recorder.start_recording(), record steps, stop[/yellow]")
    
    elif action == "record":
        console.print("[bold]Routine recording - Show Bot how it's done:[/bold]")
        console.print("Use dashboard Routines tab for interactive recording via UI")
        console.print("Or in code: recorder = RoutineRecorder(bot); recorder.start_recording(name); record_step(); stop_recording()")

@app.command()
def approvals(
    action: str = typer.Argument("list", help="list, pending, approve, reject"),
    approval_id: str = typer.Option("", "--id", help="Approval ID"),
):
    """Approvals - Bots come back when your approval is needed"""
    from .approvals import ApprovalManager
    am = ApprovalManager()
    
    if action == "list" or action == "pending":
        pending = am.list_pending()
        table = Table(title=f"Pending Approvals ({len(pending)}) - Bots Waiting for You")
        table.add_column("ID")
        table.add_column("Bot")
        table.add_column("Task")
        table.add_column("Description", max_width=40)
        table.add_column("Created")
        for a in pending:
            table.add_row(a.id, a.bot_name, a.task_title[:30], a.description[:40], a.created_at[:16])
        console.print(table)
        if not pending:
            console.print("[green]No pending approvals - bots working autonomously 24/7[/green]")
        
        if action == "list":
            all_a = am.list_all()
            console.print(f"\nTotal approvals: {len(all_a)} (including resolved)")
    
    elif action == "approve":
        if not approval_id:
            console.print("[red]--id required[/red]")
            return
        if am.approve(approval_id):
            console.print(f"[green]Approved {approval_id} - bot can continue[/green]")
        else:
            console.print(f"[red]Approval {approval_id} not found[/red]")
    
    elif action == "reject":
        if not approval_id:
            console.print("[red]--id required[/red]")
            return
        if am.reject(approval_id):
            console.print(f"[yellow]Rejected {approval_id}[/yellow]")
        else:
            console.print(f"[red]Approval {approval_id} not found[/red]")

@app.command()
def lm_studio_guide():
    """Show LM Studio setup guide"""
    console.print(Panel("""
[bold green]LM Studio Setup for PTAI (Local, No Cloud)[/bold green]

1. Download LM Studio: https://lmstudio.ai
2. Open LM Studio
3. Go to Discover tab, download a model:
   - Recommended: llama-3.1-8b-instruct, qwen2.5-7b-instruct, mistral-7b-instruct
   - For better accuracy: llama-3.1-70b if you have 48GB+ RAM/VRAM
   - Fast: qwen2.5-3b or phi-3-mini for low RAM

4. Go to Developer tab (left side, </> icon)
5. Top: Select your model
6. Click "Start Server" - should run on http://localhost:1234
7. Check "Cross-Origin" if needed
8. Test: curl http://localhost:1234/v1/models

9. Run PTAI:
   python main.py check-llm --provider lm_studio
   python main.py init --bankroll 50 --llm lm_studio --model local-model
   python main.py run --bankroll 50 --once --llm lm_studio
   python main.py pay-for-yourself 50 --llm lm_studio

LM Studio provides OpenAI compatible API at:
  http://localhost:1234/v1/chat/completions
  http://localhost:1234/v1/models

PTAI auto-detects LM Studio first, then Ollama, then heuristic fallback.
No API key needed, uses "lm-studio" as dummy key.

Troubleshooting:
- If "No LLM detected", make sure server is running in LM Studio
- Try http://localhost:1234/v1/models in browser
- Check .env LM_STUDIO_HOST=http://localhost:1234
- Use --model local-model for auto-detect

Heuristic fallback still works without LLM, but LLM is much smarter for fair value.
""", title="LM Studio Guide"))

if __name__ == "__main__":
    app()
