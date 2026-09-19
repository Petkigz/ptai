"""
Position Monitor - Tracks open positions, stop-loss, resolution
Critical for risk: one wrong prediction could wipe account
"""
import time
from typing import List, Dict
from datetime import datetime, timezone
from loguru import logger

from ..storage.db import Storage
from ..markets.polymarket import PolymarketClient
from ..config import get_settings

class PositionMonitor:
    def __init__(self, storage: Storage = None):
        self.settings = get_settings()
        self.storage = storage or Storage()
        self.client = PolymarketClient()
        self.stop_loss_pct = self.settings.max_total_drawdown_pct  # Reuse or separate
        logger.info("PositionMonitor initialized")

    def check_stop_loss(self) -> List[Dict]:
        """Check if any open position hit stop loss (50% down)"""
        open_positions = self.storage.get_open_positions()
        triggered = []
        for pos in open_positions:
            # For now, we don't have live PnL, so this is placeholder
            # In real implementation, fetch current market price and compare to entry
            market_price = pos.get("market_price", 0.5)
            # If we had current price, check:
            # if pos is YES and current_price < market_price * (1 - stop_loss_pct): trigger
            # For demo, just log
            pass
        return triggered

    def check_resolutions(self) -> List[Dict]:
        """Check if any open markets have resolved, update PnL"""
        open_positions = self.storage.get_open_positions()
        resolved = []
        for pos in open_positions:
            market_id = pos.get("market_id")
            # Try to fetch market status from Gamma API
            try:
                # This would need to fetch event and check if closed/resolved
                # For now, placeholder
                pass
            except Exception as e:
                logger.warning(f"Resolution check failed for {market_id}: {e}")
        return resolved

    def update_bankroll_from_resolved(self, trade_id: int, pnl: float):
        """Update bankroll when trade resolves"""
        current = self.storage.get_bankroll()
        new_bankroll = current + pnl
        self.storage.set_bankroll(new_bankroll)
        # Mark trade as resolved
        self.storage.conn.execute("UPDATE trades SET resolved=1, pnl=? WHERE id=?", (pnl, trade_id))
        self.storage.conn.commit()
        logger.info(f"Trade {trade_id} resolved PnL ${pnl:.2f}, bankroll ${current:.2f} -> ${new_bankroll:.2f}")

    def get_risk_report(self) -> Dict:
        """Get current risk status"""
        perf = self.storage.get_performance_summary()
        open_positions = self.storage.get_open_positions()
        total_exposure = sum([p.get("position_size_usd", 0) for p in open_positions])
        bankroll = perf["bankroll"]
        exposure_pct = total_exposure / bankroll if bankroll > 0 else 0

        return {
            "bankroll": bankroll,
            "open_positions": len(open_positions),
            "total_exposure_usd": total_exposure,
            "exposure_pct": exposure_pct,
            "max_exposure_allowed": bankroll * 0.5,  # Max 50% total exposure
            "is_over_exposed": exposure_pct > 0.5,
            "performance": perf
        }

    def run_monitor_loop(self, interval_seconds: int = 300):
        """Run monitor loop every 5 minutes"""
        logger.info(f"Starting position monitor loop every {interval_seconds}s")
        while True:
            try:
                report = self.get_risk_report()
                if report["is_over_exposed"]:
                    logger.warning(f"OVER EXPOSED: {report['exposure_pct']:.1%} > 50%")

                self.check_stop_loss()
                self.check_resolutions()

                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(interval_seconds)
