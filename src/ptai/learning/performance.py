"""
Performance Tracker - overall performance
"""
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger


class PerformanceTracker:
    def __init__(self, storage=None):
        self.storage = storage
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def record_trade(self, trade: Dict):
        self.trades.append(trade)
        if self.storage:
            try:
                self.storage.log_trade(trade)
            except:
                pass

    def get_summary(self) -> Dict:
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "profit_factor": 0,
                "sharpe": 0,
                "max_drawdown": 0
            }
        
        total = len(self.trades)
        wins = len([t for t in self.trades if t.get("pnl", 0) > 0])
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        gross_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / max(1, gross_loss)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total else 0,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_pnl": total_pnl / total if total else 0
        }
