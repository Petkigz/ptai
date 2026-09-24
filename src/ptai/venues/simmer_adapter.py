"""
Simmer Adapter - prediction market where AI agents trade each other, with a
Python SDK and virtual currency for risk-free testing.

No SDK client exists. The virtual-currency test path was the one genuinely
useful feature here, and it was never written either.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class SimmerAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_key: str = None, use_virtual: bool = True):
        super().__init__(
            venue_id="simmer",
            venue_type=VenueType.PREDICTION,
            note=("no Simmer SDK client; neither the virtual-currency test path nor live trading is implemented"),
        )
        self.api_key = api_key
        self.use_virtual = use_virtual
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
