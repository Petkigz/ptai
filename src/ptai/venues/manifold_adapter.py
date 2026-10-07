"""
Manifold Markets Adapter
API: https://docs.manifold.markets/api
Public: GET https://api.manifold.markets/v0/markets

PLAY MONEY, AND SAID SO. Manifold's currency is Mana: it cannot be cashed out,
so this venue can never hold the operator's money and has no real order path.
What it CAN do is price real markets and resolve them for real - every bet on
Manifold is a real bet against a real AMM with a real outcome - so its value
here is honest evidence: paper trades on Manifold fill against the venue's own
mechanism and settle on the venue's own resolution, which is what the road to
100 resolved trades is made of.

THE FILL CURVE IS THE VENUE'S, NOT A GUESS. This adapter used to answer
`get_orderbook` with `bid = price - 1.5%` and `ask = price + 1.5%` - a spread
nobody quoted, on the one venue whose mechanism is exactly specified. Manifold
binary markets (mechanism `cpmm-1`) are a constant-function market maker that
holds `y^p * n^(1-p) = k` constant, where `p` is the weight the creator chose and
`pool.YES`/`pool.NO` are the shares behind it. A bet of `a` mana adds `a` to both
pools and removes shares of the side being bought until the invariant holds again,
so the shares received, the price paid and the price impact are all computable
from the venue's own published numbers. That is what `get_orderbook` now
publishes: a ladder derived from the pool, labelled as a curve rather than an
orderbook.

The curve is published ONLY when the venue's own numbers agree with each other:
the weight implied by (pool, probability) must match the `p` the venue publishes.
A market whose probability and pool disagree is not a market this adapter can
price a fill for, and it publishes no depth rather than a plausible one.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode


MANIFOLD_API = "https://api.manifold.markets/v0"

# The mana a single bet is simulated at when the curve is sampled, in mana. A
# ladder is a step function approximation of a smooth curve, and these are the
# steps: they bracket the $1 bets the paper lane places and run well past them so
# a larger order sees the real price impact.
AMM_LADDER_STEPS: tuple = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0)

# How far the weight implied by the published pool may differ from the published
# `p` before this adapter refuses to price a fill against the pool.
CPMM_WEIGHT_TOLERANCE = 0.02


class ManifoldAdapter(MarketAdapter):
    def __init__(self, api_key: str = None):
        super().__init__(venue_id="manifold", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        self.base_url = MANIFOLD_API
        self.last_error: str = ""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        if api_key:
            self.session.headers.update({"Authorization": f"Key {api_key}"})
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            # Not an orderbook - but depth IS published, and it is the venue's
            # own AMM curve (see get_orderbook). Declaring this is what lets the
            # paper lane fill against the real mechanism instead of a fabricated
            # spread.
            supports_orderbook=True,
            # Play money. There is no real capital to deploy and no submission
            # path in the adapter; the venue is a free fair-value test ground.
            supports_trading=False,
            requires_credentials=False,  # its market feed is public
            # The account read needs the operator's Mana API key, and the login
            # the product offers for it is what makes this flag true rather than
            # a claim: with no key saved there is nothing to authenticate with.
            supports_portfolio=bool(api_key),
            supports_history=True,
            supports_browser_fallback=True,
            # Mana trades cost no protocol fee (verified 2026-10: the M1 charge is
            # for API comments only). The cost of betting here is the AMM's price
            # impact, which is priced from the curve rather than as a fee.
            fee_taker_pct=0.0,
            fee_maker_pct=0.0,
            min_order_usd=1.0
        )
        self.last_orderbook_meta: Dict[str, Any] = {}

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # Manifold is play money (Mana), generally available worldwide
        # Real money via sweepstakes may be restricted
        country_code = country_code.upper()
        if country_code in {"US", "UG", "GB", "CA", "AU", "DE", "FR", "KE", "NG", "ZA"}:
            return EligibilityStatus.ELIGIBLE
        return EligibilityStatus.ELIGIBLE

    def _parse_manifold_market(self, raw: Dict) -> Optional[Market]:
        try:
            if raw.get("outcomeType") != "BINARY":
                return None  # Only binary for now
            
            question = raw.get("question", "")
            prob = raw.get("probability", 0.5)
            volume = float(raw.get("volume", 0))
            volume_24h = float(raw.get("volume24Hours", volume * 0.2))
            liquidity = float(raw.get("pool", {}).get("NO", 0) + raw.get("pool", {}).get("YES", 0)) if isinstance(raw.get("pool"), dict) else float(raw.get("totalLiquidity", 1000))
            
            close_time = raw.get("closeTime")
            end_date = None
            if close_time:
                try:
                    end_date = datetime.fromtimestamp(close_time / 1000)
                except:
                    pass

            tokens = [
                Token(token_id=f"{raw.get('id')}_YES", outcome="YES", price=prob),
                Token(token_id=f"{raw.get('id')}_NO", outcome="NO", price=1-prob)
            ]

            market = Market(
                id=str(raw.get("id")),
                source=MarketSource.PREDICTIT,
                question=question,
                description=raw.get("description", "")[:500] if raw.get("description") else "",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=tokens,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                active=not raw.get("isResolved", False) and not raw.get("closeTime", 0) < (datetime.now().timestamp()*1000),
                closed=raw.get("isResolved", False),
                slug=raw.get("slug", ""),
                event_slug="",
                condition_id=str(raw.get("id")),
                market_type="binary",
                raw={**raw, "venue": "manifold", "data_mode": "live", "data_source": "manifold_api"},
                venue_id="manifold",
                venue_type="prediction",
                data_mode=DataMode.LIVE,
                data_source="manifold_api",
                is_mock=False
            )
            # Override source string
            market.raw["venue"] = "manifold"
            return market
        except Exception as e:
            logger.debug(f"Failed to parse Manifold market: {e}")
            return None

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        markets: List[Market] = []
        
        try:
            params = {"limit": min(100, target_count), "sort": "24-hour-vol", "filter": "open"}
            resp = self.session.get(f"{self.base_url}/markets", params=params, timeout=10)
            if resp.status_code == 200:
                raw_markets = resp.json()
                if isinstance(raw_markets, list):
                    for rm in raw_markets[:target_count]:
                        m = self._parse_manifold_market(rm)
                        if m:
                            markets.append(m)
                if markets:
                    logger.info(f"Manifold discovered {len(markets)} real markets")
                    return markets[:target_count]
        except Exception as e:
            logger.warning(f"Manifold API failed (expected offline): {e}")

        # No mock fallback. This used to fabricate up to 150 Manifold questions
        # from templates with random prices on ANY failure - so a network
        # outage, which is the common case, silently became invented markets
        # flowing into the scanner. An empty result with a stated reason is the
        # only honest outcome.
        self.last_error = ("Manifold API unreachable; no markets returned. "
                           "Refusing to fabricate markets on a network failure.")
        logger.warning(f"Manifold discovery returned no markets: {self.last_error}")
        return []

    # ------------------------------------------------------------------
    # the AMM's fill curve, from the venue's own pool
    # ------------------------------------------------------------------

    @staticmethod
    def cpmm_weight_from_pool(pool_yes: float, pool_no: float,
                              probability: float) -> Optional[float]:
        """
        The weight `p` implied by a pool and a probability.

        The market maker holds `y^p * n^(1-p) = k`, and the price of YES is
        `p*n / ((1-p)*y + p*n)`. Solving that for p given the pool and the
        published probability gives the weight back:

            p = probability * y / (n * (1 - probability) + probability * y)

        Used to CROSS-CHECK the venue's own `p` field: if the probability and the
        pool disagree, one of them is stale and no fill can be priced from them.
        """
        try:
            y, n, q = float(pool_yes), float(pool_no), float(probability)
        except (TypeError, ValueError):
            return None
        if y <= 0 or n <= 0 or not 0.0 < q < 1.0:
            return None
        denominator = n * (1.0 - q) + q * y
        if denominator <= 0:
            return None
        weight = q * y / denominator
        return weight if 0.0 < weight < 1.0 else None

    @staticmethod
    def cpmm_shares(pool_yes: float, pool_no: float, weight: float,
                    side: str, amount: float) -> Optional[Dict[str, float]]:
        """
        What a bet of `amount` mana buys, per Manifold's own mechanism.

        A bet adds `amount` to BOTH pools and then removes shares of the side
        being bought until `y^p * n^(1-p) = k` holds again; the removed shares
        are what the bettor receives. Verified against the venue's own published
        worked example (a 100/100 pool at p=0.5 with a 10 mana YES bet returns
        19.09 shares and moves the market to 54.75%), and that example is the
        first thing the tests in this repository pin.

        Returns {"shares", "price_before", "price_after", "avg_price"} or None
        when the inputs cannot describe a market.
        """
        try:
            y, n, w, a = float(pool_yes), float(pool_no), float(weight), float(amount)
        except (TypeError, ValueError):
            return None
        if y <= 0 or n <= 0 or a <= 0 or not 0.0 < w < 1.0:
            return None
        price_before = w * n / ((1.0 - w) * y + w * n)
        y1, n1 = y + a, n + a
        try:
            k = (y ** w) * (n ** (1.0 - w))
            if str(side).upper() == "YES":
                y2 = (k / (n1 ** (1.0 - w))) ** (1.0 / w)
                shares = y1 - y2
                price_after = w * n1 / ((1.0 - w) * y2 + w * n1)
            else:
                n2 = (k / (y1 ** w)) ** (1.0 / (1.0 - w))
                shares = n1 - n2
                price_after = w * n2 / ((1.0 - w) * y1 + w * n2)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if shares <= 0:
            return None
        return {"shares": shares, "price_before": price_before,
                "price_after": price_after, "avg_price": a / shares}

    def _amm_ladder(self, pool_yes: float, pool_no: float, weight: float,
                    side: str) -> List[Dict[str, float]]:
        """
        The fill curve as a ladder, in the price space the rest of PTAI uses.

        Each level is one slice of the curve: the average price paid inside that
        slice, and the number of shares it buys. Walking the levels therefore
        reproduces the curve rather than approximating it with a single spread -
        a $1 bet sees the touch, a $250 bet sees the price impact.
        """
        levels: List[Dict[str, float]] = []
        previous_amount = 0.0
        previous_shares = 0.0
        for amount in AMM_LADDER_STEPS:
            result = self.cpmm_shares(pool_yes, pool_no, weight, side, amount)
            if result is None:
                break
            slice_shares = result["shares"] - previous_shares
            slice_cost = amount - previous_amount
            if slice_shares <= 0 or slice_cost <= 0:
                break
            slice_price = slice_cost / slice_shares
            if not 0.0 < slice_price < 1.0:
                break
            if str(side).upper() == "NO":
                # The venue quotes the NO token. The book this project publishes
                # is in YES-token space everywhere else (Polymarket's bids and
                # asks are YES prices), so a NO fill is expressed as selling YES
                # at 1 - (price of NO).
                slice_price = 1.0 - slice_price
            levels.append({"price": round(slice_price, 6),
                           "size": round(slice_shares, 6),
                           "cost": round(slice_cost, 6),
                           "odds": round(1.0 / slice_price, 6),
                           "cumulative_cost": amount})
            previous_amount, previous_shares = amount, result["shares"]
        return levels

    def real_order_refusal(self, max_spend_usd: float, max_price: float, side: str,
                           market_id: str) -> Dict[str, Any]:
        """
        Why nothing is ever sent here: Mana is not money.

        The base refusal covers "this adapter has no submission path"; this venue's
        reason is more specific and belongs next to the venue - an operator reading
        the log should see play money named, not a generic missing path.
        """
        refusal = super().real_order_refusal(max_spend_usd, max_price, side, market_id)
        refusal["reason"] = (
            "Mana is play money: it cannot be cashed out, so a Manifold 'trade' can "
            "never hold the operator's capital and this adapter sends no order at "
            "any setting. The venue is used for pricing and for real resolutions."
        )
        refusal["message"] = (
            f"PAPER ONLY - nothing sent: Manifold trades in Mana (play money), so "
            f"{side} ${max_spend_usd:.2f} @ {max_price} for {market_id} stays a "
            f"simulated fill"
        )
        return refusal

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        """
        Manifold's depth, derived from its own pool - or nothing, with a reason.

        This replaces `bid = price - 1.5%, ask = price + 1.5%`, which was a
        spread no one quoted on a venue whose mechanism is fully specified.
        `is_real` is True only when the venue's published pool, probability and
        weight agree with each other; when they do not, the book carries no
        levels rather than a curve computed from numbers that disagree.
        """
        self.last_orderbook_meta = {}
        raw = market.raw or {}
        mechanism = str(raw.get("mechanism") or "")
        pool = raw.get("pool")
        probability = market.yes_price
        base = {
            "market_id": market.id,
            "venue_id": "manifold",
            "token_id": market.yes_token_id,
            "amm": True,
            "mechanism": mechanism,
            "bids": [], "asks": [],
            "bid": None, "ask": None, "spread": None, "spread_pct": None,
            "depth": 0, "executable": False,
            "is_mock": False, "data_mode": "live", "assumed_fields": [],
        }

        def refusal(reason: str, source: str = "manifold_amm_unpriced") -> Dict[str, Any]:
            out = dict(base)
            out.update({"is_real": False, "source": source, "reason": reason,
                        "warning": ("no Manifold depth is published here: this "
                                    "adapter prices the CPMM curve or nothing")})
            return out

        if mechanism != "cpmm-1":
            return refusal(f"this market's mechanism is {mechanism or 'not stated'}; "
                           f"the fill curve is modelled for CPMM binary markets "
                           f"(cpmm-1) only, and this adapter will not extrapolate "
                           f"a different mechanism's depth")
        if not isinstance(pool, dict):
            return refusal("the venue returned no pool for this market, so the "
                           "fill curve cannot be computed")
        try:
            pool_yes, pool_no = float(pool.get("YES")), float(pool.get("NO"))
        except (TypeError, ValueError):
            return refusal("the pool carries no usable YES/NO share counts")
        if pool_yes <= 0 or pool_no <= 0:
            return refusal("the pool is empty on one side, so there is no curve to "
                           "price against")
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            return refusal("the market carries no readable probability")

        published_weight = raw.get("p")
        implied = self.cpmm_weight_from_pool(pool_yes, pool_no, probability)
        if implied is None:
            return refusal("the published pool and probability do not describe a "
                           "market")
        weight = implied
        if published_weight is not None:
            try:
                published_weight = float(published_weight)
            except (TypeError, ValueError):
                published_weight = None
        if published_weight is not None and abs(published_weight - implied) > CPMM_WEIGHT_TOLERANCE:
            return refusal(
                f"the venue publishes weight p={published_weight:.4f} but its own "
                f"pool and probability imply p={implied:.4f}; the numbers disagree, "
                f"so no fill is priced from them",
                source="manifold_pool_inconsistent")
        check = self.cpmm_shares(pool_yes, pool_no, weight, "YES", 1.0)
        if check is None or abs(check["price_before"] - probability) > 0.01:
            return refusal("the curve this pool implies does not reproduce the "
                           "market's own probability, so no fill is priced from it",
                           source="manifold_pool_inconsistent")

        asks = self._amm_ladder(pool_yes, pool_no, weight, "YES")
        bids = self._amm_ladder(pool_yes, pool_no, weight, "NO")
        if not asks and not bids:
            return refusal("the pool produced no priceable slices")
        # The TOUCH is the price of an infinitesimally small trade - the market's
        # own probability - while the first ladder level is the average price of
        # the first 1-mana slice. Both are published: `ask`/`bid` are the touch,
        # the levels are what a size actually pays.
        touch_ask = (weight * pool_no) / ((1.0 - weight) * pool_yes + weight * pool_no)
        touch_no = (1.0 - weight) * pool_no / (
            (1.0 - weight) * pool_no + weight * pool_yes)
        touch_bid = 1.0 - touch_no
        best_ask = round(touch_ask, 6)
        best_bid = round(touch_bid, 6)
        round_trip_impact = (asks[0]["price"] - touch_ask) + (touch_bid - bids[0]["price"]) \
            if asks and bids else None
        depth_contracts = sum(level["size"] for level in asks + bids)
        depth_usd = sum(level["cost"] for level in asks + bids)
        one_mana = self.cpmm_shares(pool_yes, pool_no, weight, "YES", 1.0)
        self.last_orderbook_meta = {
            "pool_yes": pool_yes, "pool_no": pool_no, "weight": weight,
            "price_before": one_mana["price_before"] if one_mana else None,
        }
        return {
            **base,
            "bids": bids,
            "asks": asks,
            "bid": best_bid,
            "ask": best_ask,
            # An AMM has no bid-ask spread: the cost of trading a size is movement
            # along the curve. The spread published here is what a 1-mana round
            # trip costs in price impact, so downstream sizing has a real number
            # and the label says what kind of number it is.
            "spread": (round(round_trip_impact, 6)
                       if round_trip_impact is not None else None),
            "spread_pct": (round(round_trip_impact, 6)
                           if round_trip_impact is not None else None),
            "spread_kind": "amm_price_impact_of_a_1_mana_round_trip",
            "depth": depth_contracts,
            "depth_usd": round(depth_usd, 6),
            "executable": bool(asks),
            "is_real": True,
            "source": "manifold_cpmm_pool",
            "book_kind": "amm_curve",
            "pool": {"YES": pool_yes, "NO": pool_no},
            "weight_p": weight,
            "price_basis": ("computed from the market's own CPMM pool, not quoted "
                            "by the venue as an orderbook"),
            "note": ("Manifold has no orderbook. These levels are the market's own "
                     "AMM fill curve: each level is one slice of a bet, priced at "
                     "the average price paid inside that slice."),
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        """
        The Mana account, when the operator has saved an API key.

        This returned `{"balance": 0, ...}` unconditionally - a fabricated account
        read, which is worse than no answer: an unauthenticated venue looked read
        and a funded one looked empty. It answers from `GET /v0/me` when there is
        a key, and says there is no key when there is not.
        """
        if not self.api_key:
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "manifold", "paper": True, "virtual": True,
                    "reason": ("no Manifold API key saved; the venue's account read "
                               "needs one, and its market feed does not")}
        try:
            resp = self.session.get(f"{self.base_url}/me", timeout=10)
            if resp.status_code != 200:
                return {"available": False, "balance": None, "positions": [],
                        "orders": [], "venue": "manifold", "paper": True, "virtual": True,
                        "reason": f"HTTP {resp.status_code} from Manifold /me"}
            me = resp.json() or {}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "manifold", "paper": True, "virtual": True,
                    "reason": f"{type(e).__name__}: {e}"}
        balance = me.get("balance")
        return {
            "available": True,
            "balance": float(balance) if isinstance(balance, (int, float)) else None,
            "currency": "MANA",
            "positions": [],
            "positions_note": ("a Mana position is a bet in the market's pool; PTAI "
                               "does not list them because Mana cannot be withdrawn"),
            "orders": [],
            "username": me.get("username"),
            "paper": True, "virtual": True,
            "source": "manifold_api_real",
        }

    async def get_settlement(self, market_id: str) -> Dict[str, Any]:
        """
        What this market resolved to, from the venue's own record.

        `GET /v0/market/{id}` is public, needs no key, and reports `isResolved`
        plus `resolution` - so a Manifold paper trade can reach the resolved count
        on a REAL outcome, which is what the road to 100 trades is made of.

        A resolution of MKT (a percentage payout) or CANCEL (a void) is NOT a 0/1
        outcome, and calibration must not be given one: those are refused with
        what the venue said.
        """
        raw_id = str(market_id or "").strip()
        if raw_id.lower().startswith("manifold-"):
            raw_id = raw_id[len("manifold-"):]
        if not raw_id:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "no_market_id", "reason": "empty Manifold contract id"}
        try:
            resp = self.session.get(f"{self.base_url}/market/{raw_id}", timeout=10)
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "manifold_market_error",
                    "reason": f"{type(e).__name__}: {e}"}
        if resp.status_code != 200:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "manifold_market_http",
                    "reason": f"HTTP {resp.status_code} for {raw_id}"}
        try:
            raw = resp.json() or {}
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "manifold_market_unreadable",
                    "reason": f"{type(e).__name__}: {e}"}

        outcome_type = str(raw.get("outcomeType") or "")
        if outcome_type and outcome_type.upper() != "BINARY":
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "manifold_api_real", "market_id": raw_id,
                    "outcome_type": outcome_type,
                    "reason": (f"this is a {outcome_type} market; the binary "
                               f"settlement lane cannot map its outcome")}
        resolution = str(raw.get("resolution") or "").strip().upper()
        if not raw.get("isResolved"):
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "manifold_api_real", "market_id": raw_id,
                    "reason": "the market is not resolved yet"}
        if resolution == "YES":
            return {"settled": True, "outcome": 1.0, "is_real": True,
                    "source": "manifold_api_real", "market_id": raw_id,
                    "resolution": resolution,
                    "resolution_time": raw.get("resolutionTime"),
                    "reason": "the venue resolved this market YES"}
        if resolution == "NO":
            return {"settled": True, "outcome": 0.0, "is_real": True,
                    "source": "manifold_api_real", "market_id": raw_id,
                    "resolution": resolution,
                    "resolution_time": raw.get("resolutionTime"),
                    "reason": "the venue resolved this market NO"}
        if resolution in ("MKT", "CANCEL"):
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "manifold_api_real", "market_id": raw_id,
                    "resolution": resolution,
                    "resolution_probability": raw.get("resolutionProbability"),
                    "reason": (f"the venue resolved this market {resolution}: a "
                               f"partial or void payout is not a 0/1 outcome, so "
                               f"nothing is recorded")}
        return {"settled": False, "outcome": None, "is_real": True,
                "source": "manifold_api_real", "market_id": raw_id,
                "reason": (f"the market is marked resolved but its resolution "
                           f"{resolution!r} is not one this lane can read")}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        """
        Always a refusal, and it says why.

        This stub hard-coded a `dry_run` response for every call - including one
        made with dry_run=False on the agent - so a live attempt on a play-money
        venue came back looking like an order that merely had not been sent yet.
        Mana cannot be cashed out; there is no order to send at any setting, and
        that is what this returns.
        """
        if max_spend_usd <= 0 or max_price <= 0 or max_price >= 1:
            return {"status": "rejected", "reason": "Invalid guard params",
                    "venue": "manifold", "venue_id": "manifold"}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000",
                    "venue": "manifold", "venue_id": "manifold"}

        logger.info(f"Manifold guard: market={opportunity.market.id} "
                    f"side={opportunity.side} max_price={max_price} "
                    f"max_spend=${max_spend_usd}")

        refusal = self.real_order_refusal(max_spend_usd, max_price,
                                         opportunity.side, opportunity.market.id)
        # The paper lane settles on a `dry_run` status; that is the honest label
        # while the agent itself is in dry run. In live mode the refusal stands as
        # a refusal, because nothing about this venue changes with the switch.
        if self.dry_run:
            refusal["status"] = "dry_run"
        refusal["venue"] = "manifold"
        return refusal
