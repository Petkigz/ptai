"""
GRVT Adapter - hybrid derivatives exchange with a central limit order book.

No client exists. GRVT publishes an official Python SDK on GitHub
(gravity-technologies/grvt-pysdk; not on PyPI), and the Hummingbot integration
guide is cited in the original docstring, but no integration was written and
the SDK is not installed. The venue also wants $50+ for testing and $200+ live,
against a $50 bankroll, and it is a crypto-derivatives exchange: an account
there needs KYC and crypto, which is not money this build can reach from
Uganda. The note says all of that rather than a bare "unimplemented".
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
            note=("no GRVT client: the official Python SDK "
                  "(gravity-technologies/grvt-pysdk, GitHub - not PyPI) is not "
                  "installed and no integration was written; the venue wants "
                  "$50+ to test and $200+ live against a $50 bankroll, and an "
                  "account needs KYC and crypto this build cannot reach from "
                  "Uganda"),
        )
        self.api_key = api_key
        self.private_key = private_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
