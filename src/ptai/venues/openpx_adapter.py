"""
OpenPX Adapter - Rust client with sub-millisecond WebSocket support across
Polymarket and Kalshi.

No venue client exists, and the reason is not a missing binding: the `openpx`
package on PyPI (0.3.1) IS a Python binding - a compiled Rust core with a
Python wrapper - but it is a UNIFIED Kalshi+Polymarket SDK, not a venue. There
is no OpenPX exchange behind it to read markets from or place orders on.
Polymarket and Kalshi already have first-party adapters in this build
(polymarket_adapter.py, kalshi_adapter.py), so a second client for the same two
exchanges would be an unmaintained translation step in front of venues that
already work - the same reasoning as the ccxt_unified row. This adapter stays
unimplemented by design, and the note says so rather than citing a binding that
exists for a different purpose.
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
            note=("no OpenPX venue: the `openpx` PyPI package is a unified "
                  "Kalshi+Polymarket SDK, not a venue, and both exchanges "
                  "already have first-party adapters - redundant by design"),
        )
        self.polymarket_key = polymarket_key
        self.kalshi_key = kalshi_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
