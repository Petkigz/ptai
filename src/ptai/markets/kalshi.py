"""
Kalshi integration - FIXED from stub
Previously: scan_markets() -> "Kalshi scanning not yet implemented" -> []
Now: Real implementation via KalshiAdapter with API + mock fallback

Kalshi is CFTC-regulated US event exchange with first-party REST and WebSocket API
"""
from typing import List
from loguru import logger
from .base import Market, MarketSource

class KalshiClient:
    def __init__(self, api_key: str = "", enabled: bool = True):
        self.api_key = api_key
        self.enabled = enabled
        # Use KalshiAdapter as real implementation
        try:
            from ..venues.kalshi_adapter import KalshiAdapter
            self.adapter = KalshiAdapter(api_key=api_key)
            logger.info("Kalshi client initialized via KalshiAdapter - real API only, no fallback")
        except Exception as e:
            logger.warning(f"Kalshi adapter init failed {e}, using stub fallback")
            self.adapter = None
            self.enabled = False

    def scan_markets(self, target_count: int = 100) -> List[Market]:
        """
        Real Kalshi markets, or none.

        Three separate fabrication paths lived here: a mock return when an
        event loop was already running, a mock return when the adapter raised,
        and a mock return when no adapter was configured. So the sync client
        invented markets in every failure mode - including the common one of
        being called from inside a running loop, which is exactly what the
        dashboard does.

        It now delegates to the adapter and returns what the adapter returns:
        an empty list, plus a stated reason, when Kalshi is unreachable.
        """
        if not self.enabled:
            logger.info("Kalshi client disabled")
            return []

        if not self.adapter:
            logger.warning("Kalshi adapter not available; returning no markets")
            return []

        from .scanner import _run_coroutine_sync
        markets = _run_coroutine_sync(self.adapter.discover_markets(target_count=target_count))
        if markets is None:
            logger.warning(f"Kalshi scan failed: {getattr(self.adapter, 'last_error', 'unknown')}")
            return []
        logger.info(f"Kalshi scanned {len(markets)} markets via adapter")
        return markets

    # Real implementation notes:
    # - Auth with Kalshi API: https://trading-api.readme.io/
    # - GET https://api.elections.kalshi.com/trade-api/v2/markets
    # - Convert to Market objects via KalshiAdapter._parse_kalshi_market
    # - Executor via Kalshi CLOB or browser
    # - WebSocket for real-time orderbook streaming
    # - Now implemented in venues/kalshi_adapter.py
