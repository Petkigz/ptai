"""
Team Coordinator - Orchestrates AI Teammates
Gives real work to bots, they sign in to tools, use them, come back with finished work
Premium product - CrewAI-like but local, no cloud
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .base import Task, BaseTeammate
from .scout import ScoutTeammate
from .sentiment_analyst import SentimentAnalystTeammate
from .researcher import ResearcherTeammate
from .quant import QuantTeammate
from .risk_officer import RiskOfficerTeammate
from .trader import TraderTeammate
from .coach import CoachTeammate

console = Console()

class TeamCoordinator:
    """
    Coordinates AI Teammates - premium feature
    You can give real work to teammates, they sign in to tools and come back with finished work
    """
    def __init__(self, vault=None, memory=None, storage=None, browser_executor=None, tool_registry=None, brain=None, kelly=None, risk_manager=None, monitor=None, notifier=None):
        self.vault = vault
        self.memory = memory
        self.storage = storage
        self.browser_executor = browser_executor
        self.tool_registry = tool_registry
        self.brain = brain
        self.kelly = kelly
        self.risk_manager = risk_manager
        self.monitor = monitor
        self.notifier = notifier
        
        # Initialize teammates
        self.scout = ScoutTeammate(vault=vault, memory=memory)
        self.sentiment_analyst = SentimentAnalystTeammate(vault=vault, memory=memory)
        self.researcher = ResearcherTeammate(vault=vault, memory=memory, browser_executor=browser_executor, tool_registry=tool_registry)
        self.quant = QuantTeammate(vault=vault, memory=memory, brain=brain, storage=storage)
        self.risk_officer = RiskOfficerTeammate(vault=vault, memory=memory, storage=storage, kelly=kelly, risk_manager=risk_manager, monitor=monitor)
        self.trader = TraderTeammate(vault=vault, memory=memory, storage=storage, browser_executor=browser_executor, notifier=notifier)
        self.coach = CoachTeammate(vault=vault, memory=memory, storage=storage, brain=brain)
        
        self.teammates: Dict[str, BaseTeammate] = {
            "scout": self.scout,
            "sentiment_analyst": self.sentiment_analyst,
            "researcher": self.researcher,
            "quant": self.quant,
            "risk_officer": self.risk_officer,
            "trader": self.trader,
            "coach": self.coach
        }
        
        logger.success(f"TeamCoordinator initialized with {len(self.teammates)} teammates: {', '.join(self.teammates.keys())}")
    
    def get_team_status(self) -> Dict[str, Any]:
        status = {}
        for name, teammate in self.teammates.items():
            status[name] = teammate.get_status()
        return status
    
    def display_team_status(self):
        table = Table(title="AI Teammates Status")
        table.add_column("Teammate", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Status", style="magenta")
        table.add_column("Tasks Done", style="green")
        table.add_column("Avg Time", style="yellow")
        table.add_column("Tools Signed In", style="blue")
        
        for name, teammate in self.teammates.items():
            s = teammate.get_status()
            status_str = f"BUSY: {s.current_task[:30]}..." if s.is_busy else "IDLE"
            table.add_row(
                s.name,
                s.role[:40],
                status_str,
                str(s.tasks_completed),
                f"{s.avg_duration:.1f}s",
                ", ".join(s.tools_signed_in[:3])
            )
        console.print(table)
    
    async def run_full_cycle(self, target_count=500, max_deep=None, use_x=True) -> Dict[str, Any]:
        """
        Run full trading cycle with teammates collaborating
        Each teammate does its job and passes work to next
        """
        cycle_start = time.time()
        logger.info("=== TEAM CYCLE START - AI Teammates collaborating ===")
        
        try:
            # 1. Scout: Scan markets
            console.print(Panel("[bold blue]1. Scout scanning markets...[/bold blue]"))
            scout_task = Task(id=f"scout_{int(time.time())}", type="scan_markets", description=f"Scan {target_count} markets by volume", payload={"count": target_count, "order_by": "volume24hr"})
            scout_result = await self.scout.assign(scout_task)
            markets = scout_result["markets"]
            logger.success(f"Scout found {len(markets)} markets")
            
            # 2. Sentiment Analyst: Analyze sentiment
            console.print(Panel(f"[bold blue]2. Sentiment Analyst reading X for {min(len(markets), 60)} markets...[/bold blue]"))
            sentiment_task = Task(id=f"sentiment_{int(time.time())}", type="analyze_sentiment_batch", description=f"Analyze sentiment for {min(len(markets), 60)} markets", payload={"markets": markets, "max_markets": 60, "use_x": use_x})
            sentiment_result = await self.sentiment_analyst.assign(sentiment_task)
            sentiments = sentiment_result["sentiments"]
            
            # 3. Researcher: Research top markets
            console.print(Panel("[bold blue]3. Researcher researching top 20 markets...[/bold blue]"))
            research_task = Task(id=f"research_{int(time.time())}", type="research_batch", description="Research top 20 markets via web, browser, terminal", payload={"markets": markets, "sentiments": sentiments, "max_markets": 20})
            research_result = await self.researcher.assign(research_task)
            research_dict = research_result["research"]
            
            # 4. Quant: Build fair value
            console.print(Panel("[bold blue]4. Quant building fair value...[/bold blue]"))
            quant_task = Task(id=f"quant_{int(time.time())}", type="build_fair_value", description=f"Build fair value for top markets, flag >8% edge", payload={"markets": markets, "sentiments": sentiments, "research": research_dict})
            quant_result = await self.quant.assign(quant_task)
            opportunities = quant_result["opportunities"]
            
            # 5. Risk Officer: Size positions
            console.print(Panel(f"[bold blue]5. Risk Officer sizing {len(opportunities)} opportunities...[/bold blue]"))
            risk_task = Task(id=f"risk_{int(time.time())}", type="size_positions", description=f"Size {len(opportunities)} opportunities with Kelly 6% cap", payload={"opportunities": opportunities})
            risk_result = await self.risk_officer.assign(risk_task)
            sized = risk_result["sized"]
            
            # 6. Trader: Execute
            console.print(Panel(f"[bold blue]6. Trader executing {len(sized)} trades...[/bold blue]"))
            trader_task = Task(id=f"trader_{int(time.time())}", type="execute_trades", description=f"Execute {len(sized)} trades in own browser", payload={"sized_opportunities": sized})
            trader_result = await self.trader.assign(trader_task)
            executions = trader_result["executed"]
            
            # 7. Coach: Review and learn
            console.print(Panel("[bold blue]7. Coach reviewing performance...[/bold blue]"))
            coach_task = Task(id=f"coach_{int(time.time())}", type="review_performance", description="Review performance, track calibration, generate insights", payload={})
            coach_result = await self.coach.assign(coach_task)
            
            elapsed = time.time() - cycle_start
            
            # Display team status
            self.display_team_status()
            
            return {
                "cycle_time": elapsed,
                "markets_scanned": len(markets),
                "sentiments": len(sentiments)//2,
                "research": len(research_dict),
                "opportunities": len(opportunities),
                "sized": len(sized),
                "executed": len([e for e in executions if "executed" in e.get("execution", {}).get("status", "")]),
                "executions": executions,
                "coach_insights": coach_result.get("insights", []),
                "team_status": self.get_team_status()
            }
        
        except Exception as e:
            logger.error(f"Team cycle failed: {e}", exc_info=True)
            return {"error": str(e), "cycle_time": time.time() - cycle_start}
