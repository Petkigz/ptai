"""
Cymetica Event Trader Adapter - perpetual prediction markets.

No client exists. The original docstring claimed an official Python SDK; none
could be found for it - not on PyPI under `cymetica` or any of the usual
`cymetica-*` names, and on GitHub only a website repository - so there is no
SDK to build on and neither the orderbook path nor order placement was ever
written. The row says exactly that rather than citing an SDK nobody can install.
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
            note=("no Cymetica client: no SDK found on PyPI or GitHub to "
                  "build on; orderbook streaming and order placement are "
                  "unimplemented"),
        )
        self.api_key = api_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
