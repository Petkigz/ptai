"""
Veynor Adapter - prediction market intelligence API for Kalshi and Polymarket.

No client exists. Whale and smart-money signals are now implemented directly in
markets/whale_tracker.py from public Polymarket activity, which supersedes this
venue for that purpose.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class VeynorAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_key: str = None):
        super().__init__(
            venue_id="veynor",
            venue_type=VenueType.PREDICTION,
            note=("no Veynor API client. Whale signals are implemented in markets/whale_tracker.py from public Polymarket data, so this venue is redundant as well as unimplemented"),
        )
        self.api_key = api_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
