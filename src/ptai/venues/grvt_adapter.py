"""
GRVT Adapter - hybrid derivatives exchange with a central limit order book.

No client exists. The Hummingbot integration guide is cited in the original
docstring, but no integration was written.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class GRVTAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, api_key: str = None, private_key: str = None):
        super().__init__(
            venue_id="grvt",
            venue_type=VenueType.FINANCIAL,
            note=("no GRVT client; CLOB access and the Hummingbot integration are unimplemented. This venue also wants $50+ for testing and $200+ live, against a $50 bankroll"),
        )
        self.api_key = api_key
        self.private_key = private_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
