"""
CCXT Adapter - unified data layer across prediction markets and crypto venues.

No client exists. ccxt is not installed, and the premise is questionable: ccxt
covers crypto exchanges, not Polymarket or Kalshi. Those already have working
adapters of their own, so this layer would add an unmaintained translation step
in front of venues that already work.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class CCXTUnifiedAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, venues: List[str] = None):
        super().__init__(
            venue_id="ccxt_unified",
            venue_type=VenueType.OTHER,
            note=("ccxt is not installed and no unified reader was written. Polymarket and Kalshi have implemented adapters of their own, so this layer is redundant"),
        )
        self.venues = venues or ["polymarket", "kalshi", "binance", "whitebit", "pionex"]
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
