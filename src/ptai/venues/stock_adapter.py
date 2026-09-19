"""
Stock/ETF Adapter - for event-driven, momentum, mean reversion strategies
Treats stocks as venue for multi-venue opportunity engine
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource


class StockAdapter(MarketAdapter):
    def __init__(self, broker: str = "mock", api_key: str = None):
        super().__init__(venue_id=f"stock_{broker}", venue_type=VenueType.FINANCIAL)
        self.broker = broker
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key),
            supports_portfolio=True,
            supports_history=True,
            supports_browser_fallback=True,
            fee_taker_pct=0.0,  # Many brokers zero commission, but spread
            fee_maker_pct=0.0,
            min_order_usd=1.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # Stock trading: US brokers often restrict non-US, but paper trading always possible
        country_code = country_code.upper()
        if country_code == "US":
            return EligibilityStatus.ELIGIBLE
        if country_code == "UG":
            # UG can trade via international brokers, but many US brokers require verification
            return EligibilityStatus.REQUIRES_VERIFICATION
        return EligibilityStatus.REQUIRES_VERIFICATION

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        markets: List[Market] = []
        
        # Mock for V3 - in production would use Alpaca, Yahoo Finance, etc.
        import random
        stock_symbols = [
            ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Google"), ("AMZN", "Amazon"), ("META", "Meta"),
            ("TSLA", "Tesla"), ("NVDA", "Nvidia"), ("JPM", "JPMorgan"), ("JNJ", "J&J"), ("V", "Visa"),
            ("PG", "Procter & Gamble"), ("UNH", "UnitedHealth"), ("HD", "Home Depot"), ("MA", "Mastercard"),
            ("DIS", "Disney"), ("PYPL", "PayPal"), ("ADBE", "Adobe"), ("CRM", "Salesforce"), ("NFLX", "Netflix"),
            ("SPY", "S&P 500 ETF"), ("QQQ", "Nasdaq ETF"), ("IWM", "Russell 2000 ETF"), ("DIA", "Dow ETF"),
            ("TLT", "20Y Treasury ETF"), ("GLD", "Gold ETF"), ("USO", "Oil ETF")
        ]
        
        for i, (symbol, name) in enumerate(stock_symbols[:min(target_count, 100)]):
            price = random.uniform(50, 500)
            vol = random.uniform(1000000, 100000000)
            change = random.uniform(-5, 5)
            prob_up = 0.5 + change / 100.0 * 0.3
            prob_up = max(0.15, min(0.85, prob_up))
            
            # Create market as "Will stock close higher?"
            # But also could be event-driven: earnings beat, etc.
            m = Market(
                id=f"STOCK-MOCK-{symbol}",
                source=MarketSource.PREDICTIT,
                question=f"Will {symbol} ({name}) close higher tomorrow? (Stock momentum/event)",
                description=f"Mock stock {symbol} {name} last ${price:.2f} change {change:.1f}% vol ${vol:,.0f}",
                outcomes=["YES", "NO"],
                outcome_prices=[prob_up, 1-prob_up],
                tokens=[
                    Token(token_id=f"STOCK-MOCK-{symbol}_UP", outcome="YES", price=prob_up),
                    Token(token_id=f"STOCK-MOCK-{symbol}_DOWN", outcome="NO", price=1-prob_up)
                ],
                volume=vol,
                volume_24h=vol,
                liquidity=vol*0.05,
                active=True,
                closed=False,
                slug=symbol.lower(),
                event_slug=f"stock-{self.broker}",
                market_type="binary",
                raw={"mock": True, "venue": f"stock_{self.broker}", "symbol": symbol, "name": name, "last_price": price, "change_pct": change}
            )
            markets.append(m)
        
        min_vol = filters.get("min_volume", 100000)
        filtered = [m for m in markets if m.volume_24h >= min_vol]
        logger.info(f"Stock {self.broker} mock discovery: {len(markets)} -> {len(filtered)}")
        return filtered[:target_count]

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # Stocks have NBBO
        spread = 0.0005 if market.liquidity > 10000000 else 0.002  # 5 bps vs 20 bps
        return {
            "market_id": market.id,
            "bid": market.best_price - spread/2,
            "ask": market.best_price + spread/2,
            "spread": spread,
            "spread_pct": spread,
            "depth": market.liquidity
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "orders": [], "venue": f"stock_{self.broker}", "paper": True}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd <= 0:
            return {"status": "rejected", "reason": "Invalid guard params", "venue": f"stock_{self.broker}"}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000", "venue": f"stock_{self.broker}"}
        
        logger.info(f"Stock {self.broker} guard: market={opportunity.market.id} side={opportunity.side} max_spend=${max_spend_usd}")
        
        return {
            "status": "dry_run",
            "venue": f"stock_{self.broker}",
            "message": f"Would place {opportunity.side} ${max_spend_usd} for {opportunity.market.id} on {self.broker}",
            "market_id": opportunity.market.id
        }
