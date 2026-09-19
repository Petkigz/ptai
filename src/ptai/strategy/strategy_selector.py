"""
Strategy Selector - chooses which strategy to use per opportunity
Mispricing, arbitrage, event trading, market making, other validated strategies
"""
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class StrategyType(str, Enum):
    MISPRICING = "mispricing"  # Fair value vs market price
    ARBITRAGE = "arbitrage"    # Same event across venues
    EVENT_TRADING = "event_trading"  # News-driven
    MARKET_MAKING = "market_making"  # Provide liquidity
    MOMENTUM = "momentum"  # Price velocity
    MEAN_REVERSION = "mean_reversion"


@dataclass
class StrategyPerformance:
    strategy: StrategyType
    total_trades: int
    win_rate: float
    avg_edge: float
    profit: float
    brier_score: float
    skill: float


class StrategySelector:
    """
    Selects strategy based on historical performance.
    PTAI shouldn't assume strategies are profitable - prove via paper trading.
    """
    def __init__(self):
        self.strategy_performance: Dict[StrategyType, StrategyPerformance] = {}
        # Initialize with neutral performance
        for st in StrategyType:
            self.strategy_performance[st] = StrategyPerformance(
                strategy=st,
                total_trades=0,
                win_rate=0.5,
                avg_edge=0.0,
                profit=0.0,
                brier_score=0.5,
                skill=0.5
            )

    def select_strategy(self, opportunity) -> StrategyType:
        """Select best strategy for opportunity based on historical performance"""
        # Simple: mispricing is default, but check if arbitrage possible
        # If same event exists on multiple venues with price discrepancy, arbitrage
        
        # For now, always mispricing - would need cross-venue detection for arbitrage
        # Check opportunity category to choose
        category = getattr(opportunity, 'category', 'unknown')
        
        # If high volume and tight spread, market making might work
        # If news-driven, event trading
        
        # Rank strategies by skill for this category
        # For demo, return mispricing
        return StrategyType.MISPRICING

    def update_performance(self, strategy: StrategyType, outcome: Dict):
        """Update after trade resolution"""
        perf = self.strategy_performance[strategy]
        perf.total_trades += 1
        if outcome.get("win"):
            perf.win_rate = (perf.win_rate * (perf.total_trades - 1) + 1) / perf.total_trades
        else:
            perf.win_rate = (perf.win_rate * (perf.total_trades - 1)) / perf.total_trades
        
        # Update avg edge, profit, etc
        edge = outcome.get("edge", 0)
        perf.avg_edge = (perf.avg_edge * (perf.total_trades - 1) + edge) / perf.total_trades
        perf.profit += outcome.get("pnl", 0)

    def get_leaderboard(self) -> List[Dict]:
        """Get strategy leaderboard"""
        leaderboard = []
        for st, perf in self.strategy_performance.items():
            leaderboard.append({
                "strategy": st.value,
                "total": perf.total_trades,
                "win_rate": perf.win_rate,
                "avg_edge": perf.avg_edge,
                "profit": perf.profit,
                "skill": perf.skill
            })
        return sorted(leaderboard, key=lambda x: x["skill"], reverse=True)
