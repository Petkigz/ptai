"""
Cymetica Event Trader Adapter - perpetual prediction markets with an official Python SDK.

No SDK client exists. The SDK is not installed and neither the orderbook path
nor order placement was ever written.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class CymeticaAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_key: str = None):
        super().__init__(
            venue_id="cymetica",
            venue_type=VenueType.PREDICTION,
            note=("no Cymetica SDK client; orderbook streaming and order placement are unimplemented"),
        )
        self.api_key = api_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
