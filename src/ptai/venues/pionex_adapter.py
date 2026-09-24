"""
Pionex Adapter - REST and WebSocket API for spot, futures and built-in bots.

No client exists. Grid and DCA bot control via API was described and never
implemented.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class PionexAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(
            venue_id="pionex",
            venue_type=VenueType.FINANCIAL,
            note=("no Pionex REST or WebSocket client; bot control is unimplemented"),
        )
        self.api_key = api_key
        self.api_secret = api_secret
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
