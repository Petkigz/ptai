"""
Drawdown Manager - tracks drawdown and triggers kill switches
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class DrawdownState:
    peak_bankroll: float
    current_bankroll: float
    drawdown_pct: float
    max_drawdown_pct: float
    daily_pnl: float
    daily_loss_pct: float
    consecutive_losses: int
    is_in_drawdown: bool


class DrawdownManager:
    def __init__(self, initial_bankroll: float = 50.0):
        self.initial_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_drawdown_pct = 0.0
        self.daily_pnls: List[Dict] = []  # List of daily PnL
        self.consecutive_losses = 0
        self.last_update = datetime.now(timezone.utc)

        # Thresholds
        self.daily_loss_pause_pct = 0.15  # 15% daily loss -> pause
        self.total_drawdown_shutdown_pct = 0.30  # 30% total drawdown -> shutdown
        self.consecutive_loss_limit = 5

    def update_bankroll(self, new_bankroll: float, pnl: float = 0):
        self.current_bankroll = new_bankroll
        if new_bankroll > self.peak_bankroll:
            self.peak_bankroll = new_bankroll
        
        drawdown = (self.peak_bankroll - new_bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if drawdown > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0

        self.last_update = datetime.now(timezone.utc)

    def record_daily_pnl(self, date: str, pnl: float, bankroll: float):
        self.daily_pnls.append({
            "date": date,
            "pnl": pnl,
            "bankroll": bankroll,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        # Keep last 30 days
        if len(self.daily_pnls) > 30:
            self.daily_pnls = self.daily_pnls[-30:]

    def get_state(self) -> DrawdownState:
        drawdown_pct = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        
        # Daily PnL
        daily_pnl = 0
        daily_loss_pct = 0
        if self.daily_pnls:
            today = self.daily_pnls[-1]
            daily_pnl = today["pnl"]
            daily_loss_pct = abs(daily_pnl) / self.peak_bankroll if daily_pnl < 0 else 0

        return DrawdownState(
            peak_bankroll=self.peak_bankroll,
            current_bankroll=self.current_bankroll,
            drawdown_pct=drawdown_pct,
            max_drawdown_pct=self.max_drawdown_pct,
            daily_pnl=daily_pnl,
            daily_loss_pct=daily_loss_pct,
            consecutive_losses=self.consecutive_losses,
            is_in_drawdown=drawdown_pct > 0.05
        )

    def check_triggers(self) -> Dict[str, any]:
        """Check if any kill switch triggers"""
        state = self.get_state()
        triggers = []

        if state.daily_loss_pct >= self.daily_loss_pause_pct:
            triggers.append({
                "level": 1,
                "reason": f"Daily loss {state.daily_loss_pct*100:.1f}% >= {self.daily_loss_pause_pct*100}% pause threshold",
                "action": "No new trades"
            })

        if state.drawdown_pct >= self.total_drawdown_shutdown_pct:
            triggers.append({
                "level": 4,
                "reason": f"Total drawdown {state.drawdown_pct*100:.1f}% >= {self.total_drawdown_shutdown_pct*100}% shutdown threshold",
                "action": "Persist state and terminate"
            })

        if state.consecutive_losses >= self.consecutive_loss_limit:
            triggers.append({
                "level": 1,
                "reason": f"Consecutive losses {state.consecutive_losses} >= {self.consecutive_loss_limit}",
                "action": "No new trades, review strategy"
            })

        if state.current_bankroll < self.initial_bankroll * 0.5:
            triggers.append({
                "level": 3,
                "reason": f"Bankroll {state.current_bankroll} < 50% of initial {self.initial_bankroll}",
                "action": "Exit eligible positions"
            })

        return {
            "state": state,
            "triggers": triggers,
            "should_pause": len([t for t in triggers if t["level"] >= 1]) > 0,
            "should_shutdown": len([t for t in triggers if t["level"] >= 4]) > 0
        }
