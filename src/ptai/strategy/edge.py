"""
Edge Calculator - sophisticated effective edge, not just fair - market >=0.08

raw_edge
    ↓
fees
    ↓
spread
    ↓
slippage
    ↓
liquidity
    ↓
model uncertainty
    ↓
correlation
    ↓
remaining time
    ↓
effective edge
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
from loguru import logger

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity
from ..markets.orderbook import read_spread


# The >8% hunt criterion, defined ONCE. It is a MISPRICING threshold, so it
# belongs on the raw edge. It used to be applied to the effective edge in three
# separate places (here, the ranking filter, and the EV gate), which charges
# every cost twice - once to push the edge back under 8%, and again in the net
# EV terms that actually know what a cost is. Both sides of one mispricing were
# refused for it: market 0.70, YES fair 0.85, NO fair 0.55 - raw +0.150 either
# way, and both would have cleared net EV (+$0.22 and +$1.07 on a $3 stake).
HUNT_MISPRICING_MIN = 0.08


def hunted_mispricing(opportunity) -> float:
    """
    The mispricing being hunted, before costs, on the side being traded.

    Raw edge is what the >8% rule is about. An opportunity whose raw edge was
    never filled in falls back to the effective edge rather than being refused
    for a field its constructor never set.
    """
    raw = getattr(opportunity, "raw_edge", 0.0) or 0.0
    effective = getattr(opportunity, "effective_edge", 0.0) or 0.0
    return raw if raw else effective


@dataclass
class EffectiveEdge:
    raw_edge: float
    fees: float
    spread: float
    slippage: float
    liquidity_penalty: float
    uncertainty_penalty: float
    correlation_penalty: float
    time_penalty: float
    effective_edge: float
    conservative_fair: float
    market_price: float
    reasoning: str
    should_trade: bool = False
    # What a share actually costs, and what is left after paying it. `market_price`
    # above is the market's own quote (the mid); on a book with a 99.8% spread
    # there is no trade anywhere near it.
    price_paid: float = 0.0
    executable_edge: float = 0.0
    book_is_real: bool = False
    blocked_by: str = ""


def _as_price(value) -> Optional[float]:
    """A price from a book, or None. Never a made-up number."""
    try:
        if value is None:
            return None
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 0.0 or price > 1.0:
        return None
    return price


# A book this wide cannot pay for the fees and still leave the 8% mispricing
# rule anything to act on: crossing it costs a quarter of the position before
# anything else is charged.
MAX_TRADABLE_SPREAD = 0.25


def _best_quote(book: Dict, side: str):
    """
    The best (price, size) on one side of a book, from either shape.

    Production books carry top-level `bid`/`ask` plus depth arrays; a book that
    only carries its depth arrays is the same book. Reading both means a real
    book is never mistaken for an unusable one just because of how it was
    serialised.
    """
    direct = _as_price(book.get("bid" if side == "bid" else "ask"))
    levels = book.get("bids" if side == "bid" else "asks") or []
    if direct is not None and direct > 0:
        size = book.get(f"{side}_size")
        return direct, (float(size) if isinstance(size, (int, float)) else None)
    best = None
    for level in levels[:25]:
        price = size = None
        if isinstance(level, dict):
            price = _as_price(level.get("price"))
            try:
                size = float(level.get("size"))
            except (TypeError, ValueError):
                size = None
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _as_price(level[0])
            try:
                size = float(level[1])
            except (TypeError, ValueError):
                size = None
        if price is None or price <= 0:
            continue
        if best is None or (price > best[0] if side == "bid" else price < best[0]):
            best = (price, size)
    return best if best else (None, None)


class EdgeCalculator:
    def __init__(self, uncertainty_engine=None):
        self.uncertainty_engine = uncertainty_engine

    def calculate(self, market: Market, fair_prob: float, uncertainty: float = 0.1,
                  orderbook: Dict = None, amount_usd: float = 5.0,
                  correlation_penalty: float = 0.0, category_exposure: float = 0.0,
                  side: str = "YES") -> EffectiveEdge:
        """
        Calculate effective edge with all deductions, ON THE SIDE BEING BOUGHT.

        `fair_prob` is always the probability of YES, whichever side is traded.
        That is the whole subtlety this method used to miss: it computed
        `fair_prob - market_price` unconditionally, so a market the agent believed
        was overpriced produced a NEGATIVE edge and was filtered out - even though
        the same belief is a positive edge on the other side.

            YES 0.55 against a 0.70 market:
                raw_edge = 0.55 - 0.70 = -0.15   ... filtered out
                but buying NO at 0.30 with fair value 0.45 is +0.15

        So the agent could only ever trade when it thought the market UNDERPRICED
        the outcome. Half of all mispricings - the half where the market is too
        high - were structurally invisible to it, and the filter that removed them
        looked like ordinary risk control.

        The NO side mirrors exactly: price 1 - p_yes, fair 1 - fair_yes, so
        edge_no = (1 - fair_yes) - (1 - market_yes) = market_yes - fair_yes.
        """
        orderbook = orderbook or {}
        market_price = market.best_price
        side = str(side or "YES").upper()
        if side == "NO":
            # Mirror the market onto the side being traded, so every deduction
            # below (fees, spread, slippage, penalties) is applied to the price
            # actually paid rather than to the other side's price.
            raw_edge = market_price - fair_prob
            market_price = 1.0 - market_price
            fair_prob = 1.0 - fair_prob
        else:
            raw_edge = fair_prob - market_price

        # Fees - Use accurate fee model from markets/fees.py
        # Polymarket formula: Fee = 0.06 × C × p × (1-p), at $0.50 $1.50 per 100 contracts, $3 position fee $0.09 (3%)
        # For $50 bankroll, fee is biggest constraint
        try:
            from ..markets.fees import FeeEngine
            fee_engine = FeeEngine()
            venue_id = getattr(market, 'source', 'polymarket')
            venue_str = venue_id.value if hasattr(venue_id, 'value') else str(venue_id)
            if market.raw.get("venue"):
                venue_str = market.raw["venue"]
            category = "politics" if "politics" in market.question.lower() or "election" in market.question.lower() else "general"
            fee_result = fee_engine.calculate(venue_id=venue_str, amount_usd=amount_usd, price=market_price, category=category)
            fees = fee_result.fee_pct
            # Also include gas for Polymarket
            from ..execution.gas import GasModel
            gas_model = GasModel()
            gas_result = gas_model.calculate_gas(operation="place_order", amount_usd=amount_usd, venue_id=venue_str)
            gas_pct = gas_result.gas_pct_of_position
        except Exception as e:
            # Fallback to old conservative model
            fees = 0.02 if "politics" in market.question.lower() or "election" in market.question.lower() else 0.008
            gas_pct = 0.016  # $0.05 / $3 = 1.6%

        # Spread from orderbook
        spread, _spread_is_real = read_spread(orderbook, 0.02)
        # If market has low liquidity, spread wider
        if market.liquidity < 1000:
            spread = max(spread, 0.04)
        elif market.liquidity < 5000:
            spread = max(spread, 0.025)

        # Slippage based on amount vs liquidity
        slippage = 0.0
        if market.liquidity > 0:
            ratio = amount_usd / market.liquidity
            slippage = min(0.05, ratio * 0.3)
        else:
            slippage = 0.03

        # Liquidity penalty - illiquid markets penalize
        liquidity_penalty = 0.0
        if market.liquidity < 500:
            liquidity_penalty = 0.03
        elif market.liquidity < 2000:
            liquidity_penalty = 0.01

        # Uncertainty penalty - from uncertainty engine
        uncertainty_penalty = uncertainty * 0.5  # 50% of uncertainty as penalty

        # Correlation penalty - if already exposed to correlated markets
        # category_exposure is current exposure to this category (0-1)
        # If already 15% in category, penalize new trades in same category
        if category_exposure > 0.15:
            correlation_penalty = (category_exposure - 0.15) * 0.5

        # Time penalty - very short time left increases risk, very long time increases uncertainty
        time_penalty = 0.0
        if market.end_date:
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                end = market.end_date
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                hours_left = (end - now).total_seconds() / 3600
                if hours_left < 2:
                    time_penalty = 0.02  # very short, risky
                elif hours_left > 24*60:
                    time_penalty = 0.01  # far out, more uncertainty
            except:
                pass

        # Effective edge - include gas for $50 bankroll math
        # From reality check: $3 position $0.05 gas = 1.6% significant
        try:
            gas_deduction = gas_pct
        except:
            gas_deduction = 0.016
        
        total_deductions = fees + gas_deduction + spread + slippage + liquidity_penalty + uncertainty_penalty + correlation_penalty + time_penalty

        # UNITS. `raw_edge` is in PRICE UNITS - dollars per share, where a share
        # pays $1. Every deduction above is a FRACTION OF THE POSITION (fee_pct
        # and gas_pct_of_position are, spread/slippage/penalties are written as
        # if they are). Subtracting one from the other is a unit error, and it
        # is worst on the cheap side, because $1 of position buys 1/price shares:
        #
        #   buying NO at 0.30, fair 0.45, raw +0.150
        #     deductions 0.154 of notional = $0.41 in CASH on a $3 position
        #     (the EV engine, by a different route, says $0.42 - the two models
        #     agree on the money and disagreed only on this conversion)
        #     charged incorrectly as 0.154 of edge  -> effective -0.004, refused
        #     charged as cash, 0.154 * 0.30 = 0.046 -> effective +0.109
        #     and the EV engine's own answer was +$0.1075 per share
        #
        # So the deductions convert to price units by the price actually paid,
        # which is the side-mirrored `market_price` set above. At 0.70 that is a
        # 1.43x correction; at 0.30 it is 3.33x. This is why a NO side that was
        # genuinely profitable, by both the EV engine and plain arithmetic, read
        # as a marginal loss to everything downstream of here.
        paid_price = max(1e-9, market_price)
        total_deductions_price_units = total_deductions * paid_price
        effective_edge = raw_edge - total_deductions_price_units

        # Conservative fair after uncertainty - same conversion, same reason.
        uncertainty_cost_price_units = uncertainty_penalty * paid_price
        if fair_prob > market_price:
            conservative_fair = fair_prob - uncertainty_cost_price_units
        else:
            conservative_fair = fair_prob + uncertainty_cost_price_units
        conservative_fair = max(0.01, min(0.99, conservative_fair))

        # -- The price that can actually be paid --------------------------------
        # `raw_edge` above is measured against the market's own quote, which is
        # the right basis for "is this market mispriced" and the wrong basis for
        # "what does one share cost me". The operator's log has the two apart by
        # 99 points:
        #
        #   market 2774057, real CLOB: bid 0.001 / ask 0.999
        #   the agent's own arithmetic: fair 0.207 vs market 0.007 -> edge +0.200
        #   "Should trade: True", cost to break even printed as 106.7%
        #
        # A yes share cost 0.999 there, and a no share cost 1 - 0.001 = 0.999.
        # There was no trade at any price near the mid, and the trade it was
        # about to place was a straight loss with a positive-looking edge.
        book = orderbook or {}
        bid, bid_size = _best_quote(book, "bid")
        ask, ask_size = _best_quote(book, "ask")
        quoted_sizes = (bid_size, ask_size)
        book_is_real = (bool(book.get("is_real", False)) and bid is not None
                        and ask is not None)

        blocked_by = ""
        if book_is_real:
            if side == "YES":
                price_paid, size_there = ask, quoted_sizes[1]
            else:
                price_paid, size_there = 1.0 - bid, quoted_sizes[0]
            # The spread is inside `price_paid` now - crossing it IS the cost -
            # so it must not be deducted a second time.
            executable_deductions = total_deductions - spread
            executable_edge = fair_prob - price_paid - executable_deductions * price_paid

            if size_there is not None and not size_there:
                blocked_by = (
                    f"nothing to trade against: the {('ask' if side == 'YES' else 'bid')} "
                    f"side of {market.id} is empty")
            spread_measured = ask - bid
            if not blocked_by and spread_measured > MAX_TRADABLE_SPREAD:
                blocked_by = (f"spread {spread_measured:.1%} on {market.id} cannot pay "
                              f"for fees and leave an edge")
            if not blocked_by and executable_edge <= 0:
                blocked_by = (f"no executable edge: fair {fair_prob:.3f} vs the "
                              f"{price_paid:.3f} a share actually costs")
        else:
            price_paid = market_price
            executable_edge = effective_edge
            blocked_by = "orderbook is not real - there is no executable price to trade on"

        # Should trade? The 8% is the same mispricing rule as everywhere else,
        # the costs must not have eaten the edge entirely, and the trade has to
        # be possible at a price that still pays.
        should_trade = raw_edge >= 0.08 and effective_edge > 0 and not blocked_by

        reasoning = (
            f"[{side}] Raw {raw_edge:.3f} (fair {fair_prob:.3f} - mkt {market_price:.3f}) | "
            f"Deductions ({total_deductions*paid_price:.3f} of edge = "
            f"{total_deductions:.3f} of notional x price {paid_price:.3f}): "
            f"fees {fees:.3f} gas {gas_deduction:.3f} spread {spread:.3f} slip {slippage:.3f} liq {liquidity_penalty:.3f} "
            f"unc {uncertainty_penalty:.3f} corr {correlation_penalty:.3f} time {time_penalty:.3f} | "
            f"Effective {effective_edge:.3f} conservative_fair {conservative_fair:.3f} | "
            f"Should trade: {should_trade} (raw>=8%, effective>0) | $50 math: fee {fees*100:.1f}% + gas {gas_deduction*100:.1f}% + spread {spread*100:.1f}% = { (fees+gas_deduction+spread)*100:.1f}% cost must exceed to break even | "
            f"Executable: pay {price_paid:.3f} -> {executable_edge:+.3f} "
            f"{'(book not real)' if not book_is_real else ''}"
            f"{(' REFUSED: ' + blocked_by) if blocked_by else ''}"
        )

        logger.info(f"Edge calc {market.id}: {reasoning}")

        return EffectiveEdge(
            raw_edge=raw_edge,
            fees=fees,
            spread=spread,
            slippage=slippage,
            liquidity_penalty=liquidity_penalty,
            uncertainty_penalty=uncertainty_penalty,
            correlation_penalty=correlation_penalty,
            time_penalty=time_penalty,
            effective_edge=effective_edge,
            conservative_fair=conservative_fair,
            market_price=market_price,
            reasoning=reasoning,
            should_trade=should_trade,
            price_paid=price_paid,
            executable_edge=executable_edge,
            book_is_real=book_is_real,
            blocked_by=blocked_by,
        )

    def calculate_for_opportunity(self, opportunity: VenueOpportunity, orderbook: Dict = None) -> EffectiveEdge:
        """Calculate for VenueOpportunity"""
        # Create mock market from opportunity
        from ..markets.base import Market, MarketSource
        market = opportunity.market
        return self.calculate(
            market=market,
            fair_prob=opportunity.estimated_fair,
            uncertainty=opportunity.uncertainty,
            orderbook=orderbook,
            amount_usd=5.0,
            correlation_penalty=0.0
        )
