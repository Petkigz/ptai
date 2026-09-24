"""
AFX DEX Adapter - perpetual contract DEX with wallet-signed on-chain settlement.

No client exists. Requires EIP-712 order signing and a 10 USDC minimum deposit,
neither of which is implemented.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, STATUS_UNIMPLEMENTED,
                      UnimplementedVenueAdapter, VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class AFXAdapter(UnimplementedVenueAdapter):
    """
    Registered so the venue is visible, but it returns no markets and places no
    orders. See the module docstring for what is missing.
    """

    def __init__(self, wallet_address: str = None, private_key: str = None):
        super().__init__(
            venue_id="afx_dex",
            venue_type=VenueType.FINANCIAL,
            note=("no wallet-signed request client; EIP-712 order signing, on-chain settlement and the deposit flow are all unimplemented"),
        )
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.capabilities.implementation_status = STATUS_UNIMPLEMENTED
        logger.info(
            f"{type(self).__name__} registered as "
            f"{self.capabilities.implementation_status} - "
            f"{self.capabilities.implementation_note}"
        )
