"""
Stock Adapter - equities, with a broker selection.

No broker client exists. `broker` defaulted to the literal string "mock", so the
default configuration was a broker that returned invented quotes.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class StockAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, broker: str = "none", api_key: str = None):
        super().__init__(
            venue_id=f"stock_{broker}",
            venue_type=VenueType.FINANCIAL,
            note=("no broker client. The default broker was literally named 'mock'; there is now no mock broker, and no real one either"),
        )
        self.broker = broker
        self.api_key = api_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
