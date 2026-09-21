
"""
AFX DEX Adapter - perpetual contract DEX wallet-signed requests on-chain settlement
Minimum deposit 10 USDC minimum withdrawal 2 USDC, no API keys EIP-712 signatures Ethereum wallet
Ideal for local agent that already manages wallet
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class AFXAdapter(MarketAdapter):
    def __init__(self, wallet_address: str = None, private_key: str = None):
        super().__init__(venue_id="afx_dex", venue_type=VenueType.FINANCIAL)
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(wallet_address and private_key),
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.0005,
            fee_maker_pct=0.0002,
            min_order_usd=1.0
        )
        self.min_deposit = 10.0
        self.min_withdrawal = 2.0

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # DEX, generally worldwide eligible, no KYC for trading
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 20, filters: Dict = None) -> List[Market]:
        # Mock - production: AFX DEX API with EIP-712 signed requests
        # https://docs.afx.trade/
        symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "AVAX-PERP", "ARB-PERP"]
        markets = []
        for i, sym in enumerate(symbols[:target_count]):
            prob = 0.5 + (i*0.02 - 0.04)
            markets.append(Market(
                id=f"afx-MOCK-{sym}",
                source=MarketSource.POLYMARKET,
                question=f"Will {sym} funding positive? AFX DEX perpetual",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=[Token(token_id=sym, outcome="YES", price=prob)],
                volume=50000,
                volume_24h=20000,
                liquidity=10000,
                active=True,
                closed=False,
                event_slug=sym,
                raw={"venue": "afx_dex", "symbol": sym, "type": "perpetual", "category": "crypto", "auth": "EIP-712 wallet-signed", "min_deposit": 10, "min_withdrawal": 2, "data_mode": "mock", "data_source": "afx_mock_fallback", "is_mock": True, "safety": "MOCK_DATA MUST NEVER REACH LIVE EXECUTION"}
            ,
                venue_id="afx",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="afx_mock_fallback",
                is_mock=True
            ))
        logger.info(f"AFX DEX discovered {len(markets)} perpetual markets (min deposit {self.min_deposit} USDC)")
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "market_id": market.id,
            "symbol": market.raw.get("symbol"),
            "bid": market.best_price - 0.001,
            "ask": market.best_price + 0.001,
            "spread": 0.002,
            "depth": market.liquidity,
            "venue": "afx_dex",
            "auth": "EIP-712 wallet-signed, no API keys",
            "settlement": "on-chain",
            "min_deposit": self.min_deposit,
            "min_withdrawal": self.min_withdrawal
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "afx_dex", "wallet": self.wallet_address, "min_deposit": 10, "min_withdrawal": 2}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd < self.min_deposit and opportunity.market.id not in ["test"]:
            # For existing positions, allow smaller
            pass
        return {
            "status": "dry_run",
            "message": f"AFX DEX wallet-signed EIP-712 order {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id} - no API keys, on-chain settlement",
            "venue": "afx_dex",
            "auth": "EIP-712",
            "min_deposit": self.min_deposit
        }
