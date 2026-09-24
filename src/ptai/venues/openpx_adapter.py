"""
OpenPX Adapter - Rust client with sub-millisecond WebSocket support across
Polymarket and Kalshi.

No client exists. The Rust library is not installed and no Python interop was
written.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class OpenPXAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, polymarket_key: str = None, kalshi_key: str = None):
        super().__init__(
            venue_id="openpx",
            venue_type=VenueType.PREDICTION,
            note=("no OpenPX client; the Rust library is not installed and no Python binding was written"),
        )
        self.polymarket_key = polymarket_key
        self.kalshi_key = kalshi_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
