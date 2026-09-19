"""
Circuit Breaker - Critical Risk Rules for $50 Account
From blueprint: daily loss limit, max open positions, kill switch, cut losses, take profits, beware thin markets

Most important module - one wrong call must not wipe account
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
from pathlib import Path


@dataclass
class Position:
    market_id: str
    entry_price: float
    current_price: float
    amount_usd: float
    side: str
    entry_time: datetime
    category: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0


@dataclass
class CircuitBreakerState:
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_loss_limit: float = -5.0  # Stop if daily loss exceeds $5
    max_open_positions: int = 3  # Max 3 open positions (18% exposure)
    max_exposure_pct: float = 0.18  # 3×6%
    open_positions: List[Position] = field(default_factory=list)
    is_halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[datetime] = None
    consecutive_losses: int = 0
    max_consecutive_losses: int = 5


class CircuitBreaker:
    """
    Circuit breaker for $50 account
    - Daily loss limit $5
    - Max open positions 3
    - Cut losses fast 30-40%
    - Take profits 50% sell half
    - Beware thin markets >$10k volume
    - Never leverage
    - Diversify 2-3 uncorrelated
    """
    def __init__(self, 
                 daily_loss_limit: float = -5.0,
                 max_open_positions: int = 3,
                 stop_loss_pct: float = 0.35,  # Cut losses at 35%
                 take_profit_pct: float = 0.50,  # Take profits at 50%
                 min_volume: float = 10000,  # Beware thin markets
                 data_dir: str = "./data"):
        self.daily_loss_limit = daily_loss_limit
        self.max_open_positions = max_open_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_volume = min_volume
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state_file = self.data_dir / "circuit_breaker.json"
        self.state = CircuitBreakerState(
            daily_loss_limit=daily_loss_limit,
            max_open_positions=max_open_positions
        )
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                # Load daily PnL if same day
                last_date = data.get("date")
                today = datetime.now(timezone.utc).date().isoformat()
                if last_date == today:
                    self.state.daily_pnl = data.get("daily_pnl", 0.0)
                    self.state.daily_trades = data.get("daily_trades", 0)
                    self.state.consecutive_losses = data.get("consecutive_losses", 0)
            except Exception as e:
                logger.warning(f"Circuit breaker load failed: {e}")

    def _save_state(self):
        try:
            data = {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "daily_pnl": self.state.daily_pnl,
                "daily_trades": self.state.daily_trades,
                "consecutive_losses": self.state.consecutive_losses,
                "open_positions": len(self.state.open_positions),
                "is_halted": self.state.is_halted,
                "halt_reason": self.state.halt_reason
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Circuit breaker save failed: {e}")

    def check_daily_loss_limit(self, additional_pnl: float = 0.0) -> tuple[bool, str]:
        """Check if daily loss limit hit"""
        projected_pnl = self.state.daily_pnl + additional_pnl
        if projected_pnl <= self.daily_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = f"Daily loss limit hit: {projected_pnl:.2f} <= {self.daily_loss_limit:.2f}"
            self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=24)
            logger.warning(f"CIRCUIT BREAKER HALT: {self.state.halt_reason}")
            self._save_state()
            return False, self.state.halt_reason
        return True, "OK"

    def check_max_positions(self) -> tuple[bool, str]:
        """Check max open positions"""
        if len(self.state.open_positions) >= self.max_open_positions:
            return False, f"Max open positions hit: {len(self.state.open_positions)} >= {self.max_open_positions} (18% exposure)"
        return True, "OK"

    def check_thin_market(self, volume_24h: float, liquidity: float) -> tuple[bool, str]:
        """Beware thin markets - stick to >$10k volume"""
        if volume_24h < self.min_volume:
            return False, f"Thin market: volume ${volume_24h:.0f} < ${self.min_volume:.0f} minimum - high slippage risk"
        if liquidity < 1000:
            return False, f"Thin market: liquidity ${liquidity:.0f} < $1000 - high slippage"
        return True, "OK"

    def check_can_trade(self) -> tuple[bool, str]:
        """Overall check if can trade"""
        if self.state.is_halted:
            if self.state.halt_until and datetime.now(timezone.utc) < self.state.halt_until:
                return False, f"Halted: {self.state.halt_reason} until {self.state.halt_until}"
            else:
                # Halt expired, reset
                self.state.is_halted = False
                self.state.halt_reason = ""
                self.state.halt_until = None
                # Reset daily if new day
                today = datetime.now(timezone.utc).date().isoformat()
                try:
                    data = json.loads(self.state_file.read_text())
                    if data.get("date") != today:
                        self.state.daily_pnl = 0.0
                        self.state.daily_trades = 0
                except:
                    pass

        # Daily loss
        ok, reason = self.check_daily_loss_limit()
        if not ok:
            return False, reason

        # Max positions
        ok, reason = self.check_max_positions()
        if not ok:
            return False, reason

        return True, "OK - can trade"

    def evaluate_position(self, position: Position) -> Dict[str, str]:
        """
        Evaluate open position for cut losses / take profits
        - Cut losses fast if drops 30-40%
        - Take profits if gains 50% sell half
        """
        entry = position.entry_price
        current = position.current_price
        side = position.side
        
        # Calculate PnL %
        if side.upper() == "YES":
            pnl_pct = (current - entry) / entry if entry > 0 else 0
        else:  # NO or SELL
            pnl_pct = (entry - current) / entry if entry > 0 else 0
        
        action = "HOLD"
        reason = f"PnL {pnl_pct*100:.1f}% within limits"
        
        if pnl_pct <= -self.stop_loss_pct:
            action = "CUT_LOSS"
            reason = f"Cut losses fast: PnL {pnl_pct*100:.1f}% <= -{self.stop_loss_pct*100:.0f}% - thesis wrong, close position"
            logger.warning(f"CUT LOSS: {position.market_id} PnL {pnl_pct*100:.1f}% entry {entry:.3f} current {current:.3f}")
        elif pnl_pct >= self.take_profit_pct:
            action = "TAKE_PROFIT_HALF"
            reason = f"Take profits: PnL {pnl_pct*100:.1f}% >= {self.take_profit_pct*100:.0f}% - sell half, lock in gains"
            logger.info(f"TAKE PROFIT: {position.market_id} PnL {pnl_pct*100:.1f}% - sell half")
        elif pnl_pct >= self.take_profit_pct * 0.6:  # 30% gain
            action = "TAKE_PROFIT_QUARTER"
            reason = f"Take partial profits: PnL {pnl_pct*100:.1f}% >= {self.take_profit_pct*0.6*100:.0f}% - consider selling quarter"
        
        return {
            "market_id": position.market_id,
            "pnl_pct": pnl_pct,
            "pnl_usd": position.pnl_usd,
            "action": action,
            "reason": reason,
            "entry_price": entry,
            "current_price": current
        }

    def add_position(self, position: Position):
        """Add open position"""
        self.state.open_positions.append(position)
        self._save_state()

    def remove_position(self, market_id: str, pnl_usd: float = 0.0):
        """Remove closed position and update daily PnL"""
        self.state.open_positions = [p for p in self.state.open_positions if p.market_id != market_id]
        self.state.daily_pnl += pnl_usd
        self.state.daily_trades += 1
        if pnl_usd < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Check consecutive losses halt
        if self.state.consecutive_losses >= self.state.max_consecutive_losses:
            self.state.is_halted = True
            self.state.halt_reason = f"Consecutive losses {self.state.consecutive_losses} >= {self.state.max_consecutive_losses}"
            self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=12)
            logger.warning(f"CIRCUIT BREAKER: {self.state.halt_reason}")
        
        self._save_state()

    def record_trade(self, pnl: float, position_id: str = ""):
        """Record trade PnL for testing - wrapper around remove_position logic"""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        if self.state.daily_pnl <= self.daily_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = f"Daily loss limit hit: {self.state.daily_pnl:.2f} <= {self.daily_loss_limit:.2f}"
            from datetime import timedelta, timezone, datetime
            self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=24)
        
        if self.state.consecutive_losses >= self.state.max_consecutive_losses:
            self.state.is_halted = True
            self.state.halt_reason = f"Consecutive losses {self.state.consecutive_losses} >= {self.state.max_consecutive_losses}"
            from datetime import timedelta, timezone, datetime
            self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=12)
        
        self._save_state()

    def can_trade(self) -> bool:
        ok, _ = self.check_can_trade()
        return ok

    def get_status(self) -> Dict:
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_loss_limit": self.state.daily_loss_limit,
            "open_positions": len(self.state.open_positions),
            "max_open_positions": self.state.max_open_positions,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
            "consecutive_losses": self.state.consecutive_losses,
            "can_trade": self.check_can_trade()[0],
            "can_trade_reason": self.check_can_trade()[1],
            "critical_rules": [
                f"Daily loss limit ${self.daily_loss_limit} - stop trading day if hit",
                f"Max open positions {self.max_open_positions} (18% exposure)",
                f"Cut losses at {self.stop_loss_pct*100:.0f}% - thesis wrong",
                f"Take profits at {self.take_profit_pct*100:.0f}% sell half",
                f"Beware thin markets volume >${self.min_volume:.0f}",
                "Never leverage - buying shares not margin",
                "Diversify 2-3 uncorrelated events"
            ]
        }

    def reset_daily(self):
        """Reset daily counters - called at midnight UTC"""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.consecutive_losses = 0
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.halt_until = None
        self._save_state()
        logger.info("Circuit breaker daily reset")
