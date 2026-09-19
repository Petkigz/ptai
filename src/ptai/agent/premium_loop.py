"""
PTAI Premium Loop - AI Teammates you can give real work to
Bots sign in to tools, use them just like you do, come back with finished work
Premium product level with learning, vault, memory, backtest, multi-user

This is the premium version of loop.py using TeamCoordinator
"""
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from ..config import get_settings
from ..storage.db import Storage
from ..vault import Vault
from ..memory import Memory
from ..users import UserManager
from .brain import Brain
from .tools import ToolRegistry
from .researcher import Researcher
from .notifier import Notifier
from ..risk import KellyCalculator, RiskManager
from ..execution.monitor import PositionMonitor
from .teammates.coordinator import TeamCoordinator

console = Console()

class PremiumTradingAgent:
    """
    Premium Trading Agent - Team of AI Teammates
    - Scout, SentimentAnalyst, Researcher, Quant, RiskOfficer, Trader, Coach
    - Each signs into tools via Vault
    - Learns via Memory
    - Multi-user via UserManager
    - Backtest, Vault, Notifications premium
    """
    def __init__(self, user_id: str = None, bankroll: float = None, dry_run: bool = None):
        self.settings = get_settings()
        if bankroll is not None:
            self.settings.bankroll = bankroll
        if dry_run is not None:
            self.settings.dry_run = dry_run
        
        self.user_id = user_id or "default"
        
        # Multi-user paths
        self.user_manager = UserManager()
        if user_id:
            user = self.user_manager.get_user(user_id)
            if user:
                paths = self.user_manager.get_user_paths(user_id)
                self.storage = Storage(db_path=str(paths["db"]))
                self.vault = Vault(vault_path=str(paths["vault"]))
                self.memory = Memory(db_path=str(paths["memory"]))
                self.browser_profile = str(paths["browser_profile"])
                logger.info(f"Premium agent for user {user_id} ({user.email}) bankroll ${user.bankroll}")
            else:
                # Fallback to default paths
                self.storage = Storage(db_path="./data/ptai.db")
                self.vault = Vault()
                self.memory = Memory()
                self.browser_profile = self.settings.browser_persistent_dir
        else:
            self.storage = Storage(db_path="./data/ptai.db")
            self.vault = Vault()
            self.memory = Memory()
            self.browser_profile = self.settings.browser_persistent_dir
        
        if bankroll:
            current = self.storage.get_bankroll()
            if current == 50.0 and bankroll != 50.0:
                self.storage.set_state("initial_bankroll", str(bankroll))
                self.storage.set_bankroll(bankroll)
        
        # Core components for teammates
        self.brain = Brain()
        self.kelly = KellyCalculator(kelly_fraction=self.settings.kelly_fraction, max_pct=self.settings.max_position_pct, min_edge=self.settings.min_edge_pct)
        self.risk_manager = RiskManager(storage=self.storage, kelly_calculator=self.kelly)
        self.monitor = PositionMonitor(storage=self.storage)
        self.notifier = Notifier(enabled=True, vault=self.vault)
        
        self.tool_registry = ToolRegistry(browser_executor=None)
        self.researcher = Researcher(tool_registry=self.tool_registry, browser_executor=None)
        
        # Team coordinator - the premium teammates
        self.coordinator = TeamCoordinator(
            vault=self.vault,
            memory=self.memory,
            storage=self.storage,
            brain=self.brain,
            kelly=self.kelly,
            risk_manager=self.risk_manager,
            monitor=self.monitor,
            notifier=self.notifier
        )
        
        self.is_running = False
        self.cycle_count = 0
        
        logger.success(f"PremiumTradingAgent initialized with {len(self.coordinator.teammates)} teammates: {', '.join(self.coordinator.teammates.keys())} | bankroll=${self.storage.get_bankroll()} dry_run={self.settings.dry_run} user={self.user_id}")
    
    async def run_cycle_with_team(self) -> Dict[str, Any]:
        """Run one cycle using AI Teammates collaborating"""
        self.cycle_count += 1
        cycle_start = time.time()
        bankroll_start = self.storage.get_bankroll()
        
        console.print(Panel(f"[bold green]PTAI PREMIUM Cycle {self.cycle_count} | Bankroll ${bankroll_start:.2f} | {datetime.now(timezone.utc).isoformat()} | Team: {len(self.coordinator.teammates)} teammates | User: {self.user_id}[/bold green]"))
        
        try:
            # Use team coordinator to run full cycle
            result = await self.coordinator.run_full_cycle(
                target_count=self.settings.scan_markets_count,
                use_x=self.settings.sentiment_use_x
            )
            
            elapsed = time.time() - cycle_start
            
            # Log scan
            avg_edge = 0
            if result.get("opportunities"):
                # Can't easily get avg edge from opportunities list without recalc, use 0
                pass
            
            try:
                self.storage.log_scan(
                    markets_scanned=result.get("markets_scanned", 0),
                    opportunities_found=result.get("opportunities", 0),
                    avg_edge=avg_edge,
                    execution_time=elapsed,
                    bankroll=self.storage.get_bankroll()
                )
            except Exception as e:
                logger.debug(f"Log scan failed: {e}")
            
            # Self-preservation check via RiskOfficer teammate
            sp = self.storage.check_self_preservation(daily_cost=self.settings.daily_cost_to_cover, max_unprofitable_days=self.settings.shutdown_if_unprofitable_days)
            
            if sp["should_shutdown"]:
                logger.critical(f"SELF-PRESERVATION SHUTDOWN: {sp['shutdown_reason']}")
                console.print(Panel(f"[bold red]SHUTDOWN: {sp['shutdown_reason']}\nBankroll ${sp['bankroll']:.2f} PnL ${sp['total_pnl']:.2f}[/bold red]", title="PTAI Shutdown"))
                self.notifier.shutdown_alert(sp["shutdown_reason"], sp["bankroll"])
            
            # Display summary
            console.print(Panel(f"[bold blue]Cycle {self.cycle_count} Complete in {elapsed:.1f}s | Scanned {result.get('markets_scanned',0)} | Opps {result.get('opportunities',0)} | Sized {result.get('sized',0)} | Executed {result.get('executed',0)} | Insights {len(result.get('coach_insights',[]))}[/bold blue]"))
            
            if result.get("coach_insights"):
                for insight in result["coach_insights"][:3]:
                    console.print(f"[yellow]Coach Insight: {insight}[/yellow]")
            
            return {
                "cycle": self.cycle_count,
                "elapsed": elapsed,
                "bankroll": self.storage.get_bankroll(),
                "self_preservation": sp,
                "should_shutdown": sp["should_shutdown"],
                **result
            }
        
        except Exception as e:
            logger.error(f"Premium cycle {self.cycle_count} failed: {e}", exc_info=True)
            self.notifier.error_alert(str(e))
            return {"cycle": self.cycle_count, "error": str(e), "should_shutdown": False}
    
    async def run_autonomous(self, interval_minutes: int = None):
        interval = interval_minutes or self.settings.scan_interval_minutes
        self.is_running = True
        
        console.print(Panel(f"[bold green]PTAI PREMIUM Autonomous Agent Started\nTeam: {len(self.coordinator.teammates)} AI Teammates\nBankroll ${self.storage.get_bankroll():.2f}\nInterval {interval}min\nDry Run {self.settings.dry_run}\nUser {self.user_id}\nGoal: Earn ${self.settings.daily_cost_to_cover}/day or shutdown\nBots sign in to tools, use them like you do, come back with finished work[/bold green]", title="PTAI PREMIUM"))
        
        try:
            while self.is_running:
                result = await self.run_cycle_with_team()
                
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
            self.storage.close()
            self.memory.close()
            self.user_manager.close()
    
    def run_once_sync(self) -> Dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_cycle_with_team())
        finally:
            loop.close()
