"""
Kill Switch Hierarchy - LEVEL 0-5
LEVEL 0 Normal operation
LEVEL 1 No new trades, existing monitored
LEVEL 2 Cancel outstanding orders
LEVEL 3 Exit eligible positions
LEVEL 4 Persist state
LEVEL 5 Terminate trading engine

Triggers: daily loss, drawdown, LLM unavailable, market API inconsistent, price feed stale,
browser compromised, unexpected balance, execution mismatch, DB corruption, calibration collapse,
internet instability, duplicate order detected
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timezone
from loguru import logger
import json
from pathlib import Path


class KillLevel(int, Enum):
    NORMAL = 0
    NO_NEW_TRADES = 1
    CANCEL_ORDERS = 2
    EXIT_POSITIONS = 3
    PERSIST_STATE = 4
    TERMINATE = 5


@dataclass
class KillTrigger:
    level: KillLevel
    reason: str
    timestamp: datetime
    data: Dict = field(default_factory=dict)
    triggered_by: str = "system"


@dataclass
class KillSwitchState:
    current_level: KillLevel
    triggers: List[KillTrigger] = field(default_factory=list)
    is_active: bool = False
    last_trigger: Optional[KillTrigger] = None


class KillSwitch:
    """
    Kill switch hierarchy with multiple triggers.
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state_file = self.data_dir / "kill_switch.json"
        self.current_level = KillLevel.NORMAL
        self.triggers: List[KillTrigger] = []
        self.handlers: Dict[KillLevel, List[Callable]] = {
            level: [] for level in KillLevel
        }
        self.load_state()

        # Thresholds
        self.thresholds = {
            "daily_loss_pct": 0.15,
            "drawdown_pct": 0.30,
            "max_consecutive_losses": 5,
            "llm_timeout_seconds": 300,
            "price_feed_stale_seconds": 600,
            "duplicate_order_window_seconds": 60
        }

    def load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.current_level = KillLevel(data.get("current_level", 0))
                logger.info(f"Kill switch loaded: level {self.current_level}")
            except Exception as e:
                logger.warning(f"Kill switch load failed: {e}")

    def save_state(self):
        try:
            data = {
                "current_level": int(self.current_level),
                "triggers": [
                    {
                        "level": int(t.level),
                        "reason": t.reason,
                        "timestamp": t.timestamp.isoformat(),
                        "data": t.data,
                        "triggered_by": t.triggered_by
                    }
                    for t in self.triggers[-20:]  # last 20
                ],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Kill switch save failed: {e}")

    def register_handler(self, level: KillLevel, handler: Callable):
        """Register handler to execute when level triggered"""
        self.handlers[level].append(handler)

    def trigger(self, level: KillLevel, reason: str, data: Dict = None, triggered_by: str = "system") -> KillLevel:
        """Trigger kill switch at level"""
        data = data or {}
        trigger = KillTrigger(
            level=level,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
            data=data,
            triggered_by=triggered_by
        )
        self.triggers.append(trigger)
        
        # Only escalate, never de-escalate automatically (requires manual reset)
        if level > self.current_level:
            old_level = self.current_level
            self.current_level = level
            logger.warning(f"KILL SWITCH ESCALATED: {old_level} -> {level} reason: {reason}")
            
            # Execute handlers for this level and all lower levels that haven't been executed?
            # Actually execute handlers for new level
            for handler in self.handlers.get(level, []):
                try:
                    handler(trigger)
                except Exception as e:
                    logger.error(f"Kill switch handler failed: {e}")
            
            self.save_state()
        
        return self.current_level

    def check_daily_loss(self, daily_loss_pct: float):
        if daily_loss_pct >= self.thresholds["daily_loss_pct"]:
            self.trigger(
                KillLevel.NO_NEW_TRADES,
                f"Daily loss {daily_loss_pct*100:.1f}% >= {self.thresholds['daily_loss_pct']*100}%",
                {"daily_loss_pct": daily_loss_pct}
            )

    def check_drawdown(self, drawdown_pct: float):
        if drawdown_pct >= self.thresholds["drawdown_pct"]:
            self.trigger(
                KillLevel.PERSIST_STATE,
                f"Drawdown {drawdown_pct*100:.1f}% >= {self.thresholds['drawdown_pct']*100}%",
                {"drawdown_pct": drawdown_pct}
            )

    def check_llm_unavailable(self, last_success_seconds_ago: float):
        if last_success_seconds_ago > self.thresholds["llm_timeout_seconds"]:
            self.trigger(
                KillLevel.NO_NEW_TRADES,
                f"LLM unavailable for {last_success_seconds_ago:.0f}s",
                {"seconds_ago": last_success_seconds_ago}
            )

    def check_price_feed_stale(self, last_update_seconds_ago: float):
        if last_update_seconds_ago > self.thresholds["price_feed_stale_seconds"]:
            self.trigger(
                KillLevel.NO_NEW_TRADES,
                f"Price feed stale {last_update_seconds_ago:.0f}s",
                {"seconds_ago": last_update_seconds_ago}
            )

    def check_execution_mismatch(self, expected: float, actual: float):
        if abs(expected - actual) / max(1, expected) > 0.1:  # 10% mismatch
            self.trigger(
                KillLevel.CANCEL_ORDERS,
                f"Execution mismatch expected ${expected} actual ${actual}",
                {"expected": expected, "actual": actual}
            )

    def check_duplicate_order(self, market_id: str, recent_orders: List[Dict]):
        # Check for duplicate orders within window
        now = datetime.now(timezone.utc)
        recent_same = [
            o for o in recent_orders
            if o.get("market_id") == market_id and
            (now - datetime.fromisoformat(o.get("timestamp", now.isoformat()).replace("Z", "+00:00"))).total_seconds() < self.thresholds["duplicate_order_window_seconds"]
        ]
        if len(recent_same) >= 2:
            self.trigger(
                KillLevel.CANCEL_ORDERS,
                f"Duplicate order detected for {market_id}: {len(recent_same)} in {self.thresholds['duplicate_order_window_seconds']}s",
                {"market_id": market_id, "count": len(recent_same)}
            )

    def check_calibration_collapse(self, brier_score: float):
        if brier_score > 0.35:
            self.trigger(
                KillLevel.NO_NEW_TRADES,
                f"Calibration collapse Brier {brier_score:.3f} > 0.35",
                {"brier_score": brier_score}
            )

    def check_internet_instability(self, failure_count: int):
        if failure_count > 5:
            self.trigger(
                KillLevel.NO_NEW_TRADES,
                f"Internet instability {failure_count} failures",
                {"failure_count": failure_count}
            )

    def can_trade(self) -> bool:
        """Can we open new trades?"""
        return self.current_level < KillLevel.NO_NEW_TRADES

    def should_cancel_orders(self) -> bool:
        return self.current_level >= KillLevel.CANCEL_ORDERS

    def should_exit_positions(self) -> bool:
        return self.current_level >= KillLevel.EXIT_POSITIONS

    def should_terminate(self) -> bool:
        return self.current_level >= KillLevel.TERMINATE

    def reset(self, manual_reason: str = "manual reset"):
        """Manual reset - requires human intervention"""
        logger.warning(f"Kill switch MANUAL RESET from {self.current_level} to NORMAL: {manual_reason}")
        self.current_level = KillLevel.NORMAL
        self.triggers.append(KillTrigger(
            level=KillLevel.NORMAL,
            reason=f"Manual reset: {manual_reason}",
            timestamp=datetime.now(timezone.utc),
            triggered_by="human"
        ))
        self.save_state()

    def get_state(self) -> KillSwitchState:
        return KillSwitchState(
            current_level=self.current_level,
            triggers=self.triggers[-10:],
            is_active=self.current_level > KillLevel.NORMAL,
            last_trigger=self.triggers[-1] if self.triggers else None
        )

    def get_status_report(self) -> Dict:
        state = self.get_state()
        return {
            "current_level": int(state.current_level),
            "level_name": state.current_level.name,
            "is_active": state.is_active,
            "can_trade": self.can_trade(),
            "should_cancel": self.should_cancel_orders(),
            "should_exit": self.should_exit_positions(),
            "should_terminate": self.should_terminate(),
            "triggers_count": len(self.triggers),
            "last_trigger": {
                "level": int(state.last_trigger.level),
                "reason": state.last_trigger.reason,
                "timestamp": state.last_trigger.timestamp.isoformat()
            } if state.last_trigger else None,
            "thresholds": self.thresholds
        }
