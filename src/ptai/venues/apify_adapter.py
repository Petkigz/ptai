"""
Apify Adapter - paid arbitrage scanners across Polymarket, Kalshi and PredictIt.

No API client exists. The adapter previously returned two hardcoded arb rows
("Fed cut June", "Trump win") with invented cross-venue prices as though a paid
scanner had produced them.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_SCANNER, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class ApifyAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_token: str = None):
        super().__init__(
            venue_id="apify",
            venue_type=VenueType.PREDICTION,
            note=("no Apify API client; the paid arbitrage scanner is never called. Cross-venue arbitrage is available directly through the Polymarket and Kalshi adapters, which are implemented and charge no per-pair fee"),
        )
        self.api_token = api_token
        self.price_per_1000 = 2.0
        self.capabilities.implementation_status = STATUS_SCANNER
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
