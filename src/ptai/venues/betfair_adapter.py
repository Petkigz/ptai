"""
Legacy Betfair stub — superseded by betfair_exchange.py.

This module used to hold three adapters that returned invented football
questions ("Will Team A win? BETDAQ", "Will Over 2.5 goals?") as though an
exchange had published them. Its `BetfairAdapter` registered itself under
venue_id "betfair", which meant the registry was handed this stub while the
real exchange client in betfair_exchange.py sat unused.

That is the specific way the betting layer was unreachable: the one feed that
carries goals, corners, cards and player props was never connected to the
trading path. Real Betfair markets now come from
`betfair_exchange.BetfairExchangeAdapter`.

Betdaq and BetConnect are separate exchanges listed in the original docstring as
flumine integrations. Neither has a client here.
"""
from typing import Any, Dict, List

from loguru import logger

from .adapter import (EligibilityStatus, MarketAdapter, UnimplementedVenueAdapter,
                      VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


class BetfairAdapter(UnimplementedVenueAdapter):
    """
    Deprecated. Do not register this.

    Kept under the same name so existing imports keep working, but it no longer
    claims venue_id "betfair" — that belongs to BetfairExchangeAdapter, which
    issues real requests. Registering this would shadow it.
    """

    def __init__(self, use_flumine: bool = False, **kwargs):
        super().__init__(
            venue_id="betfair_legacy_stub",
            venue_type=VenueType.OTHER,
            note=("deprecated stub; use venues.betfair_exchange.BetfairExchangeAdapter "
                  "for real Betfair data. Registering this under venue_id 'betfair' "
                  "would shadow the working adapter."),
        )
        self.use_flumine = use_flumine
        logger.warning("BetfairAdapter is a deprecated stub - use BetfairExchangeAdapter")


class BetdaqAdapter(UnimplementedVenueAdapter):
    """Betdaq exchange: no client exists."""

    def __init__(self, username: str = None, password: str = None, **kwargs):
        super().__init__(
            venue_id="betdaq",
            venue_type=VenueType.OTHER,
            note=("no Betdaq client; the flumine integration referenced in the original "
                  "docstring was never written"),
        )
        self.username = username
        self.password = password


class BetConnectAdapter(UnimplementedVenueAdapter):
    """BetConnect: no client exists."""

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(
            venue_id="betconnect",
            venue_type=VenueType.OTHER,
            note=("no BetConnect client; the flumine integration referenced in the "
                  "original docstring was never written"),
        )
        self.api_key = api_key
