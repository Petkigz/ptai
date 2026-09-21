
"""
Betfair Adapter - world's largest betting exchange mature Python ecosystem
flumine framework open-source event-based trading framework for sports betting with support Betfair Betdaq Betconnect
Risk management, simulation, paper trading, multi-venue support
betfairlightweight library fast Python wrapper full API market and order streaming
API: https://developer.betfair.com/
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class BetfairAdapter(MarketAdapter):
    def __init__(self, username: str = None, api_key: str = None, use_flumine: bool = True):
        super().__init__(venue_id="betfair", venue_type=VenueType.OTHER)
        self.username = username
        self.api_key = api_key
        self.use_flumine = use_flumine
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(username and api_key),
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.05,  # Betfair commission 2-5% depending on market
            fee_maker_pct=0.05,
            min_order_usd=2.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # Betfair restricted in many countries, UG requires verification
        cc = country_code.upper()
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        if cc == "US":
            return EligibilityStatus.RESTRICTED
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 50, filters: Dict = None) -> List[Market]:
        # Mock - production: use betfairlightweight or flumine
        # pip install betfairlightweight flumine
        # from betfairlightweight import APIClient; client = APIClient(username, api_key)
        # from flumine import Flumine, clients; client = clients.BetfairClient(username, api_key)
        mock_sports = [
            "Man City vs Arsenal - Man City win?",
            "Lakers vs Warriors - Lakers win?",
            "Djokovic vs Nadal - Djokovic win?",
            "Will England win World Cup?",
            "Will Super Bowl go over 50 points?",
        ]
        markets = []
        for i, q in enumerate(mock_sports[:target_count]):
            price = 0.45 + i*0.05
            markets.append(Market(
                id=f"betfair-MOCK-{i}",
                source=MarketSource.POLYMARKET,
                question=q + " - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
                outcomes=["BACK", "LAY"],
                outcome_prices=[price, 1-price],
                tokens=[Token(token_id=f"bf-{i}", outcome="BACK", price=price)],
                volume=100000,
                volume_24h=50000,
                liquidity=50000,
                active=True,
                closed=False,
                event_slug=f"betfair-sport-{i}",
                raw={"venue": "betfair", "type": "exchange", "category": "sports", "lay_available": True, "flumine": self.use_flumine, "framework": "flumine event-based", "data_mode": "mock", "data_source": "betfair_mock_fallback", "is_mock": True, "safety": "MOCK_DATA must be impossible to reach live execution"},
                venue_id="betfair",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="betfair_mock_fallback",
                is_mock=True
            ))
        logger.info(f"Betfair discovered {len(markets)} sports exchange markets (flumine={self.use_flumine}, lay betting available)")
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "market_id": market.id,
            "bid": market.best_price - 0.02,
            "ask": market.best_price + 0.02,
            "spread": 0.04,
            "bid_size": 10000,
            "ask_size": 10000,
            "depth": market.liquidity,
            "venue": "betfair",
            "lay": "Available - bet against outcomes, opens arb and market-making not possible on traditional sportsbooks",
            "flumine": "Risk management, simulation, paper trading, multi-venue",
            "streaming": "Market and order streaming via betfairlightweight"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "betfair", "commission": "2-5%", "lay": True}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        # Sports exchanges ideal for market-making and arbitrage rather than directional
        # Place two-sided quotes and earn spread, or arb price discrepancy between Betfair and Pinnacle
        return {
            "status": "dry_run",
            "message": f"Betfair exchange order {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id} - lay available, market-making and arb rather than directional, flumine framework",
            "venue": "betfair",
            "strategy": "Market-making and arbitrage ideal for small capital, two-sided quotes earn spread"
        }

class BetdaqAdapter(MarketAdapter):
    def __init__(self, username: str = None, api_key: str = None):
        super().__init__(venue_id="betdaq", venue_type=VenueType.OTHER)
        self.username = username
        self.api_key = api_key
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(username),
            supports_portfolio=True,
            fee_taker_pct=0.02,
            fee_maker_pct=0.02,
            min_order_usd=2.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        cc = country_code.upper()
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        if cc == "US":
            return EligibilityStatus.RESTRICTED
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 30, filters: Dict = None) -> List[Market]:
        mock_qs = ["Will Team A win? BETDAQ", "Will Over 2.5 goals?"]
        markets = []
        for i, q in enumerate(mock_qs[:target_count]):
            price = 0.5
            markets.append(Market(
                id=f"betdaq-MOCK-{i}",
                source=MarketSource.POLYMARKET,
                question=q + " - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
                outcomes=["BACK", "LAY"],
                outcome_prices=[price, 1-price],
                tokens=[Token(token_id=f"bd-{i}", outcome="BACK", price=price)],
                volume=20000,
                volume_24h=10000,
                liquidity=15000,
                active=True,
                closed=False,
                event_slug=f"betdaq-{i}",
                raw={"venue": "betdaq", "type": "exchange", "category": "sports", "flumine_support": True, "data_mode": "mock", "data_source": "betdaq_mock_fallback", "is_mock": True, "safety": "MOCK_DATA must be impossible to reach live execution"},
                venue_id="betdaq",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="betdaq_mock_fallback",
                is_mock=True
            ))
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {"spread": 0.04, "bid": market.best_price-0.02, "ask": market.best_price+0.02, "depth": market.liquidity, "venue": "betdaq"}

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "betdaq"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        return {"status": "dry_run", "message": f"BETDAQ order {opportunity.side} ${max_spend_usd} @ {max_price}", "venue": "betdaq"}

class BetConnectAdapter(MarketAdapter):
    def __init__(self, api_key: str = None):
        super().__init__(venue_id="betconnect", venue_type=VenueType.OTHER)
        self.api_key = api_key
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key),
            fee_taker_pct=0.02,
            min_order_usd=2.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        cc = country_code.upper()
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 20, filters: Dict = None) -> List[Market]:
        mock_qs = ["Will Team B win? BetConnect"]
        markets = []
        for i, q in enumerate(mock_qs[:target_count]):
            price = 0.5
            markets.append(Market(
                id=f"betconnect-MOCK-{i}",
                source=MarketSource.POLYMARKET,
                question=q + " - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
                outcomes=["BACK", "LAY"],
                outcome_prices=[price, 1-price],
                tokens=[Token(token_id=f"bc-{i}", outcome="BACK", price=price)],
                volume=15000,
                volume_24h=8000,
                liquidity=10000,
                active=True,
                closed=False,
                event_slug=f"betconnect-{i}",
                raw={"venue": "betconnect", "type": "exchange", "category": "sports", "data_mode": "mock", "data_source": "betconnect_mock_fallback", "is_mock": True, "safety": "MOCK_DATA must be impossible to reach live execution"},
                venue_id="betconnect",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="betconnect_mock_fallback",
                is_mock=True
            ))
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {"spread": 0.04, "depth": market.liquidity, "venue": "betconnect"}

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "betconnect"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        return {"status": "dry_run", "message": f"BetConnect order {opportunity.side} ${max_spend_usd} @ {max_price}", "venue": "betconnect"}
