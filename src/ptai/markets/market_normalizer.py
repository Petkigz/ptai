"""
Market Normalizer - normalizes markets from different venues to common format
FIXED: Robust normalization for 19 venues, not just Polymarket/Kalshi
Previously: Only Polymarket and Kalshi normalization
Now: Handles all venues: polymarket, kalshi, manifold, predictit, simmer, cymetica, crypto (binance, whitebit, afx, grvt, pionex), stocks, betfair, betdaq, betconnect, ccxt_unified, veynor, openpx, apify
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime, timezone

from .base import Market, MarketSource, Token


class MarketNormalizer:
    """
    Normalizes markets from any venue to common Market model
    Makes adapter contract, market normalization robust - per user request not to add venues blindly but make contract robust
    """
    
    def normalize_polymarket(self, raw_event: Dict) -> List[Market]:
        markets = []
        try:
            event_slug = raw_event.get("slug", "")
            event_title = raw_event.get("title", "")
            raw_markets = raw_event.get("markets", [])
            for rm in raw_markets:
                try:
                    import json
                    outcomes = rm.get("outcomes", [])
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    outcome_prices = rm.get("outcomePrices", [])
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)
                    outcome_prices = [float(p) for p in outcome_prices]
                    
                    clob_ids = rm.get("clobTokenIds", [])
                    if isinstance(clob_ids, str):
                        clob_ids = json.loads(clob_ids)
                    
                    tokens = []
                    for i, outcome in enumerate(outcomes):
                        token_id = clob_ids[i] if i < len(clob_ids) else f"{rm.get('id')}_{i}"
                        price = outcome_prices[i] if i < len(outcome_prices) else 0.5
                        tokens.append(Token(token_id=token_id, outcome=outcome, price=price))
                    
                    market = Market(
                        id=rm.get("id", "") or rm.get("conditionId", ""),
                        source=MarketSource.POLYMARKET,
                        question=rm.get("question", "") or event_title,
                        description=rm.get("description", "") or raw_event.get("description", ""),
                        outcomes=outcomes,
                        outcome_prices=outcome_prices,
                        tokens=tokens,
                        volume=float(rm.get("volume", 0) or raw_event.get("volume", 0) or 0),
                        volume_24h=float(rm.get("volume24hr", 0) or raw_event.get("volume24hr", 0) or 0),
                        liquidity=float(rm.get("liquidity", 0) or 0),
                        slug=rm.get("slug", ""),
                        event_slug=event_slug,
                        condition_id=rm.get("conditionId", ""),
                        market_type="binary" if len(outcomes) == 2 else "categorical",
                        raw={**rm, "venue": "polymarket", "normalized": True, "category": self._detect_category(rm.get("question", "") or event_title)}
                    )
                    markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to normalize polymarket market {rm.get('id')}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Polymarket normalization failed: {e}")
        
        return markets

    def normalize_kalshi(self, raw_market: Dict) -> Optional[Market]:
        try:
            market_id = raw_market.get("id", "") or raw_market.get("ticker", "")
            question = raw_market.get("title", "") or raw_market.get("subtitle", "")
            yes_price = float(raw_market.get("yes_bid", 50)) / 100.0
            if yes_price > 1:
                yes_price = yes_price / 100
            
            tokens = [
                Token(token_id=f"{market_id}_yes", outcome="YES", price=yes_price),
                Token(token_id=f"{market_id}_no", outcome="NO", price=1-yes_price)
            ]
            
            market = Market(
                id=market_id,
                source=MarketSource.KALSHI,
                question=question,
                description=raw_market.get("subtitle", ""),
                outcomes=["YES", "NO"],
                outcome_prices=[yes_price, 1-yes_price],
                tokens=tokens,
                volume=float(raw_market.get("volume", 0)),
                volume_24h=float(raw_market.get("volume_24h", 0)),
                liquidity=float(raw_market.get("liquidity", 0) or raw_market.get("open_interest", 0)),
                slug=raw_market.get("ticker", ""),
                event_slug=raw_market.get("event_ticker", ""),
                market_type="binary",
                raw={**raw_market, "venue": "kalshi", "normalized": True, "category": self._detect_category(question)}
            )
            return market
        except Exception as e:
            logger.warning(f"Kalshi normalization failed: {e}")
            return None

    def normalize_generic(self, raw_market: Dict, venue_id: str, source: MarketSource = MarketSource.POLYMARKET) -> Optional[Market]:
        """
        Generic normalization for any venue - makes adapter contract robust
        Handles: manifold, predictit, simmer, cymetica, crypto, whitebit, afx, grvt, pionex, betfair, etc.
        """
        try:
            # Extract common fields with fallbacks
            market_id = str(raw_market.get("id") or raw_market.get("ticker") or raw_market.get("symbol") or f"{venue_id}-{hash(str(raw_market))%10000}")
            question = raw_market.get("question") or raw_market.get("title") or raw_market.get("name") or f"{venue_id} market {market_id}"
            
            # Price extraction - handle various formats
            price = 0.5
            if "outcome_prices" in raw_market and raw_market["outcome_prices"]:
                price = float(raw_market["outcome_prices"][0])
            elif "price" in raw_market:
                price = float(raw_market["price"])
            elif "yes_price" in raw_market:
                price = float(raw_market["yes_price"])
            elif "last_price" in raw_market:
                # For crypto, convert price change to prob
                change = float(raw_market.get("change", 0))
                price = max(0.1, min(0.9, 0.5 + change*0.5))
            elif "probability" in raw_market:
                price = float(raw_market["probability"])
            
            price = max(0.01, min(0.99, price))
            
            # Volume/liquidity extraction
            volume = float(raw_market.get("volume", 0) or raw_market.get("quote_volume", 0) or 0)
            volume_24h = float(raw_market.get("volume_24h", 0) or raw_market.get("volume24hr", 0) or raw_market.get("volume", 0) or volume*0.5 or 0)
            liquidity = float(raw_market.get("liquidity", 0) or raw_market.get("open_interest", 0) or raw_market.get("depth", 0) or volume*0.1 or 1000)
            
            # End date
            end_date = None
            for date_field in ["end_date", "endDate", "close_time", "expiration_time", "closeTime"]:
                if raw_market.get(date_field):
                    try:
                        date_str = raw_market[date_field]
                        if isinstance(date_str, str):
                            end_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        break
                    except:
                        continue
            
            # Category detection
            category = raw_market.get("category") or self._detect_category(question)
            
            # Tokens
            token_id = raw_market.get("token_id") or raw_market.get("clob_token_id") or f"{venue_id}-{market_id}-YES"
            tokens = [Token(token_id=str(token_id), outcome="YES", price=price)]
            
            # Event slug
            event_slug = raw_market.get("event_slug") or raw_market.get("event_ticker") or raw_market.get("slug") or f"{venue_id}-event"
            
            market = Market(
                id=market_id,
                source=source,
                question=question[:500],  # cap length
                description=raw_market.get("description", "")[:1000],
                outcomes=["YES", "NO"],
                outcome_prices=[price, 1-price],
                tokens=tokens,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                active=raw_market.get("active", True),
                closed=raw_market.get("closed", False),
                slug=raw_market.get("slug", market_id)[:100],
                event_slug=str(event_slug)[:100],
                market_type=raw_market.get("market_type", "binary"),
                raw={
                    **raw_market,
                    "venue": venue_id,
                    "normalized": True,
                    "category": category,
                    "original_venue": venue_id,
                    "normalization_time": datetime.now(timezone.utc).isoformat()
                }
            )
            return market
        except Exception as e:
            logger.warning(f"Generic normalization failed for {venue_id} {raw_market.get('id')}: {e}")
            return None

    def _detect_category(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana", "bnb", "perp", "futures", "funding", "usdt", "binance", "whitebit", "afx", "grvt", "pionex"]):
            return "crypto"
        if any(k in q for k in ["trump", "biden", "election", "republican", "democrat", "senate", "congress", "kalshi", "predictit"]):
            return "politics"
        if any(k in q for k in ["nfl", "nba", "mlb", "soccer", "football", "team", "game", "championship", "betfair", "betdaq", "man city", "arsenal", "lakers", "djokovic"]):
            return "sports"
        if any(k in q for k in ["fed", "cpi", "inflation", "interest rate", "fomc", "gdp", "nfp", "jobs", "unemployment", "earnings", "s&p", "aapl"]):
            return "economics"
        if any(k in q for k in ["weather", "hurricane", "temperature"]):
            return "weather"
        if any(k in q for k in ["gpt", "agi", "ai", "agent", "llm", "simmer"]):
            return "ai"
        return "general"

    def normalize(self, raw_data: Any, source: MarketSource, venue_id: str = None) -> List[Market]:
        """
        Normalize from any source - robust for 19 venues
        Makes adapter contract robust per user request
        """
        venue_id = venue_id or source.value
        
        if source == MarketSource.POLYMARKET:
            if isinstance(raw_data, dict) and "markets" in raw_data:
                return self.normalize_polymarket(raw_data)
            elif isinstance(raw_data, list):
                all_markets = []
                for event in raw_data:
                    if isinstance(event, dict) and "markets" in event:
                        all_markets.extend(self.normalize_polymarket(event))
                    else:
                        m = self.normalize_generic(event, venue_id, source)
                        if m:
                            all_markets.append(m)
                return all_markets
            else:
                m = self.normalize_generic(raw_data, venue_id, source)
                return [m] if m else []
        elif source == MarketSource.KALSHI:
            if isinstance(raw_data, list):
                markets = []
                for rm in raw_data:
                    m = self.normalize_kalshi(rm)
                    if m:
                        markets.append(m)
                    else:
                        m2 = self.normalize_generic(rm, venue_id, source)
                        if m2:
                            markets.append(m2)
                return markets
            else:
                m = self.normalize_kalshi(raw_data)
                if m:
                    return [m]
                m2 = self.normalize_generic(raw_data, venue_id, source)
                return [m2] if m2 else []
        else:
            # Generic for all other venues: manifold, predictit, simmer, cymetica, crypto, whitebit, afx, grvt, pionex, betfair, etc.
            if isinstance(raw_data, list):
                markets = []
                for rm in raw_data:
                    m = self.normalize_generic(rm, venue_id, source)
                    if m:
                        markets.append(m)
                return markets
            else:
                m = self.normalize_generic(raw_data, venue_id, source)
                return [m] if m else []

    def validate_market(self, market: Market) -> bool:
        """Validate normalized market has required fields"""
        if not market.id:
            return False
        if not market.question:
            return False
        if not (0 < market.best_price < 1):
            return False
        if market.liquidity < 0:
            return False
        if market.volume_24h < 0:
            return False
        return True

    def get_report(self) -> Dict[str, Any]:
        return {
            "normalizer": "Market Normalizer - robust for 19 venues",
            "supported_venues": [
                "polymarket", "kalshi", "manifold", "predictit", "simmer", "cymetica",
                "crypto_binance", "whitebit", "afx_dex", "grvt", "pionex", "stock_mock",
                "betfair", "betdaq", "betconnect", "ccxt_unified", "veynor", "openpx", "apify"
            ],
            "fields_normalized": ["id", "question", "outcomes", "outcome_prices", "tokens", "volume", "volume_24h", "liquidity", "end_date", "active", "closed", "slug", "event_slug", "category", "venue"],
            "category_detection": "politics, sports, crypto, economics, weather, ai, general via keywords",
            "robustness": "Handles various price formats, volume formats, date formats, fallback to generic normalization",
            "validation": "Checks id, question, price 0-1, liquidity, volume non-negative"
        }
