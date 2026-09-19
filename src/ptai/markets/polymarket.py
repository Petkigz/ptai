"""
Polymarket integration - Gamma API for discovery, CLOB for execution
Fully local, uses public APIs, no cloud agent needed
"""
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import aiohttp
from loguru import logger

from .base import Market, Token, MarketSource
from ..config import get_settings

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

class PolymarketClient:
    def __init__(self, gamma_api: str = GAMMA_API, clob_api: str = CLOB_API):
        self.gamma_api = gamma_api.rstrip("/")
        self.clob_api = clob_api.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        settings = get_settings()
        self.min_liquidity = settings.min_liquidity
        self.min_volume_24h = settings.min_volume_24h

    def _parse_outcomes(self, market_data: Dict) -> tuple:
        """Parse outcomes which come as JSON strings"""
        outcomes = market_data.get("outcomes", "[]")
        outcome_prices = market_data.get("outcomePrices", "[]")
        clob_token_ids = market_data.get("clobTokenIds", "[]")

        # Handle if already list or stringified
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except:
                outcomes = [outcomes]
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except:
                outcome_prices = []
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except:
                clob_token_ids = []

        # Convert prices to float
        prices = []
        for p in outcome_prices:
            try:
                prices.append(float(p))
            except:
                prices.append(0.5)

        return outcomes, prices, clob_token_ids

    def _event_to_markets(self, event: Dict) -> List[Market]:
        """Convert Gamma event -> list of Market objects"""
        markets = []
        event_slug = event.get("slug", "")
        event_title = event.get("title", "")

        for m in event.get("markets", []):
            try:
                outcomes, prices, token_ids = self._parse_outcomes(m)

                tokens = []
                for i, outcome in enumerate(outcomes):
                    token_id = token_ids[i] if i < len(token_ids) else f"{m.get('id')}_{i}"
                    price = prices[i] if i < len(prices) else 0.5
                    tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

                # Volume & liquidity can be at event or market level
                volume = float(m.get("volume") or event.get("volume") or 0)
                volume_24h = float(m.get("volume24hr") or event.get("volume24hr") or m.get("volume24h") or event.get("volume24h") or 0)
                liquidity = float(m.get("liquidity") or event.get("liquidity") or 0)

                # End date
                end_date_str = m.get("endDate") or event.get("endDate") or event.get("end_date")
                end_date = None
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    except:
                        pass

                market = Market(
                    id=str(m.get("id") or m.get("conditionId") or ""),
                    source=MarketSource.POLYMARKET,
                    question=m.get("question") or event.get("title") or "",
                    description=m.get("description") or event.get("description") or "",
                    outcomes=outcomes,
                    outcome_prices=prices,
                    tokens=tokens,
                    volume=volume,
                    volume_24h=volume_24h,
                    liquidity=liquidity,
                    end_date=end_date,
                    active=event.get("active", True) and not event.get("closed", False),
                    closed=event.get("closed", False),
                    slug=m.get("slug", ""),
                    event_slug=event_slug,
                    condition_id=m.get("conditionId") or m.get("condition_id") or "",
                    market_type="binary" if len(outcomes) == 2 else "categorical",
                    raw={"event": event, "market": m}
                )
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market {m.get('id')}: {e}")
                continue
        return markets

    def fetch_events(self, limit: int = 100, offset: int = 0, order: str = "volume24hr", active: bool = True, closed: bool = False) -> List[Dict]:
        """Fetch raw events from Gamma API"""
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": False,
            "active": str(active).lower(),
            "closed": str(closed).lower()
        }
        try:
            resp = self.session.get(f"{self.gamma_api}/events", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Gamma API fetch failed offset={offset}: {e}")
            return []

    def scan_markets(self, target_count: int = 750, order_by: str = "volume24hr") -> List[Market]:
        """
        Scan 500-1000 markets as required
        Paginates through events endpoint
        """
        logger.info(f"Scanning Polymarket for {target_count} markets, order_by={order_by}")
        all_markets: List[Market] = []
        offset = 0
        batch = 100
        seen_ids = set()

        # Map order_by aliases
        order_map = {
            "volume24hr": "volume24hr",
            "volume_24hr": "volume24hr",
            "volume": "volume",
            "liquidity": "liquidity",
            "volume24h": "volume24hr"
        }
        order = order_map.get(order_by, "volume24hr")

        while len(all_markets) < target_count:
            events = self.fetch_events(limit=batch, offset=offset, order=order, active=True, closed=False)
            if not events:
                logger.warning(f"No more events at offset {offset}, got {len(all_markets)} markets")
                break

            for ev in events:
                markets = self._event_to_markets(ev)
                for m in markets:
                    if m.id in seen_ids:
                        continue
                    # Apply filters
                    if m.liquidity < self.min_liquidity and self.min_liquidity > 0:
                        continue
                    if m.volume_24h < self.min_volume_24h and self.min_volume_24h > 0:
                        continue
                    if not m.active:
                        continue
                    # Must have valid token
                    if not m.tokens:
                        continue
                    seen_ids.add(m.id)
                    all_markets.append(m)
                    if len(all_markets) >= target_count:
                        break
                if len(all_markets) >= target_count:
                    break

            logger.info(f"Scanned offset {offset}, total markets {len(all_markets)}/{target_count}")
            offset += batch

            # Avoid hammering API
            time.sleep(0.3)

            # Safety: don't loop forever
            if offset > 2000:
                break

        logger.success(f"Scan complete: {len(all_markets)} markets found")
        return all_markets[:target_count]

    async def scan_markets_async(self, target_count: int = 750, order_by: str = "volume24hr") -> List[Market]:
        """Async version for faster scanning"""
        # For simplicity, use sync in async wrapper
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.scan_markets(target_count, order_by))

    def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get orderbook for a token from CLOB"""
        try:
            resp = self.session.get(f"{self.clob_api}/book", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price"""
        try:
            resp = self.session.get(f"{self.clob_api}/midpoint", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception as e:
            logger.debug(f"Midpoint fetch failed {token_id}: {e}")
            return None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        try:
            resp = self.session.get(f"{self.clob_api}/price", params={"token_id": token_id, "side": "buy"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except:
            return None

class PolymarketExecutor:
    """Handles actual order placement via CLOB API"""
    def __init__(self, private_key: Optional[str] = None, funder: Optional[str] = None, chain_id: int = 137, signature_type: int = 1, host: str = CLOB_API):
        self.private_key = private_key
        self.funder = funder
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.host = host
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.private_key:
            logger.warning("No private key - running in DRY RUN / read-only mode")
            return
        try:
            from py_clob_client.client import ClobClient
            self.client = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=self.chain_id,
                signature_type=self.signature_type,
                funder=self.funder
            )
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            logger.success("Polymarket CLOB client initialized")
        except Exception as e:
            logger.error(f"Failed to init CLOB client: {e}")
            self.client = None

    def place_order(self, token_id: str, price: float, size: float, side: str = "BUY", order_type: str = "GTC", dry_run: bool = True) -> Dict[str, Any]:
        """Place limit order"""
        if dry_run or not self.client:
            logger.info(f"[DRY RUN] Would place {side} {size} @ {price} for token {token_id[:20]}...")
            return {"status": "dry_run", "orderID": f"dry_{int(time.time())}", "price": price, "size": size}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            side_const = BUY if side.upper() == "BUY" else SELL
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side_const,
                token_id=token_id
            )
            signed = self.client.create_order(order_args)
            o_type = OrderType.GTC if order_type == "GTC" else OrderType.FOK
            resp = self.client.post_order(signed, o_type)
            logger.success(f"Order placed: {resp}")
            return resp
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_balance(self) -> Dict:
        if not self.client:
            return {"balance": 0, "mock": True}
        try:
            # This requires additional API calls
            return {"balance": "unknown - check via client"}
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return {"balance": 0, "error": str(e)}
