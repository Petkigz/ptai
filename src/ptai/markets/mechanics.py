"""
Market mechanics: the rules a venue imposes on an order before it will accept it.

This module exists because of a specific, silent failure. The venue SDK rounds
the price of every order with `round_normal(price, tick_decimals)` - including
rounding UP - and its own `price_valid()` only checks bounds:

    def price_valid(price, tick_size):
        return float(tick_size) <= price <= 1 - float(tick_size)

There is no tick-multiple check anywhere. So an agent that models an edge at
0.567, sizes at 0.567 and submits 0.567 gets an order signed at 0.57, and
nothing anywhere reports that the price changed. On a 0.55 market that is ~0.9%
of price, which an 8% minimum edge can absorb. On a 0.05 longshot where the
tick is 0.01, the same rounding is a 20% move in price, which is larger than
most real edges.

The rule this module enforces: THE PRICE THAT IS MODELLED MUST BE THE PRICE
THAT IS SIGNED. Rounding happens here, at decision time, and the rounded price
is what flows into expected value, Kelly sizing and the order - so the agent
never believes a price the venue will not honour.

Tick sizes and their decimal places are taken from the venue SDK's own
ROUNDING_CONFIG rather than re-derived, because they are the venue's rules and
not ours to guess. The values are asserted against the SDK in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Any, Dict, Optional, Tuple

# The tick sizes the CLOB will accept, as strings, per the venue's own type:
# Literal['0.1', '0.01', '0.005', '0.0025', '0.001', '0.0001']
TICK_SIZES: Tuple[str, ...] = ("0.1", "0.01", "0.005", "0.0025", "0.001", "0.0001")

# Decimal places per tick. The venue's ROUNDING_CONFIG maps each tick to
# RoundConfig(price=N, size=2, amount=M); these are the price Ns.
TICK_DECIMALS: Dict[str, int] = {
    "0.1": 1,
    "0.01": 2,
    "0.005": 3,
    "0.0025": 4,
    "0.001": 3,
    "0.0001": 4,
}

# RoundConfig(size=2) for every tick: order size in shares carries 2 decimals.
SIZE_DECIMALS: int = 2

# The venue will not accept a price at or beyond the bounds; price_valid uses
# these inclusive bounds.
MIN_TICK_SIZE: str = TICK_SIZES[-1]


def tick_decimals(tick_size: Any) -> int:
    """Decimal places for a tick, defaulting to the finest for unknown values."""
    return TICK_DECIMALS.get(normalise_tick(tick_size), TICK_DECIMALS[MIN_TICK_SIZE])


def normalise_tick(tick_size: Any) -> str:
    """
    Coerce a tick size to one of the venue's literal strings.

    The venue returns the tick as a string ("0.01"). Callers may hold it as a
    float. Both are accepted, and an unrecognised tick is returned as the
    finest tick rather than silently rounding prices to a coarser grid than the
    venue allows.
    """
    if tick_size is None:
        return MIN_TICK_SIZE
    try:
        value = Decimal(str(tick_size))
    except Exception:
        return MIN_TICK_SIZE
    for tick in TICK_SIZES:
        if Decimal(tick) == value:
            return tick
    return MIN_TICK_SIZE


def _quantise(value: Decimal, places: int, rounding: str) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    return value.quantize(quantum, rounding=rounding)


def round_price_to_tick(price: Any, tick_size: Any,
                        side: str = "BUY") -> float:
    """
    Snap a price onto the venue's tick grid, in the direction that does not
    silently worsen the trade.

    A BUY limit is the most the agent will pay, so it rounds DOWN: paying less
    than modelled can only help, and never fills at worse than the model
    assumed. A SELL is the least the agent will accept, so it rounds UP.

    The order is clamped inside the venue's bounds, because price_valid() is
    inclusive and a price exactly at the bound is accepted while 0 or 1 is not.
    """
    tick = normalise_tick(tick_size)
    places = tick_decimals(tick)
    # Decimal(str(x)) so a float that is already on the grid (0.5700000000000001)
    # snaps to the grid instead of being seen as a hair above it.
    value = Decimal(str(price))
    is_buy = str(side or "BUY").strip().upper() in ("BUY", "YES", "LONG", "TRUE", "1")
    rounding = ROUND_DOWN if is_buy else ROUND_UP
    snapped = _quantise(value, places, rounding)

    low = Decimal(tick)
    high = Decimal(1) - low
    if snapped < low:
        snapped = low
    if snapped > high:
        snapped = high
        # Clamping to an inclusive bound may leave it off-grid; snap inward.
        snapped = _quantise(snapped, places, rounding)
    return float(snapped)


def round_size_to_step(size: Any, decimals: int = SIZE_DECIMALS) -> float:
    """
    Round an order size in shares down to the venue's step.

    Down, always: rounding size up would commit more capital than the sizing
    decision approved.
    """
    try:
        value = Decimal(str(size))
    except Exception:
        return 0.0
    if value <= 0:
        return 0.0
    return float(_quantise(value, decimals, ROUND_DOWN))


def is_on_tick(price: Any, tick_size: Any) -> bool:
    """
    Whether a price is exactly on the venue's tick grid.

    Exposed so callers can assert the invariant the SDK never checks. Exact
    because it compares Decimals derived from strings, not floats.
    """
    tick = normalise_tick(tick_size)
    places = tick_decimals(tick)
    value = Decimal(str(price))
    return value == _quantise(value, places, ROUND_HALF_UP)


def tick_rounding_cost(price: Any, tick_size: Any, side: str = "BUY") -> float:
    """
    How far the modelled price moved getting onto the grid, as a fraction.

    Positive means the trade got worse than modelled. This is the number the
    agent needs in order to decide whether an edge survives the venue's price
    grid: on a coarse tick it can exceed the entire edge.
    """
    try:
        original = Decimal(str(price))
    except Exception:
        return 0.0
    if original <= 0:
        return 0.0
    rounded = Decimal(str(round_price_to_tick(price, tick_size, side)))
    return float((rounded - original) / original)


@dataclass
class MarketMechanics:
    """
    What the venue requires of an order in one specific market.

    `is_real` distinguishes "the venue told us" from "we assumed". A default
    tick is a guess, and an agent that sizes against a guess is back to
    modelling a price it may not get; so an assumed mechanics object is always
    labelled and always warns.
    """

    tick_size: str = MIN_TICK_SIZE
    neg_risk: bool = False
    min_order_size: float = 0.0        # shares, per the venue's min_order_size
    min_order_notional_usd: float = 1.0  # the operator's own floor
    taker_fee_rate: float = 0.0
    maker_fee_rate: float = 0.0
    seconds_delay: int = 0
    accepting_orders: bool = True
    source: str = "assumed_default"
    is_real: bool = False
    warnings: list = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def tick(self) -> float:
        return float(self.tick_size)

    def round_price(self, price: float, side: str = "BUY") -> float:
        return round_price_to_tick(price, self.tick_size, side)

    def round_size(self, shares: float) -> float:
        return round_size_to_step(shares)

    def shares_for_usd(self, amount_usd: float, price: float,
                       side: str = "BUY") -> float:
        """Shares bought by a USD amount at the price that will be signed."""
        signed_price = self.round_price(price, side) if side.upper() == "BUY" else price
        if signed_price <= 0:
            return 0.0
        return round_size_to_step(amount_usd / signed_price)

    def usd_for_shares(self, shares: float, price: float) -> float:
        return round(shares * price, 6)

    def validate_order(self, price: float, shares: float) -> Tuple[bool, str]:
        """
        Check an order against the venue's rules before it is signed.

        Returns (ok, reason). Everything checked here was previously unchecked:
        the tick grid was never asserted, and the minimum size came from a
        hardcoded local table rather than the venue.
        """
        if not self.accepting_orders:
            return False, "venue is not accepting orders in this market"
        if price <= 0 or price >= 1:
            return False, f"price {price} outside (0, 1)"
        if not is_on_tick(price, self.tick_size):
            return False, (f"price {price} is not a multiple of the tick "
                           f"{self.tick_size}; the venue would silently "
                           f"re-round it")
        low = self.tick
        if not (low <= price <= 1 - low):
            return False, (f"price {price} outside the venue's accepted range "
                           f"[{low}, {1 - low}]")
        if shares <= 0:
            return False, f"size {shares} shares is not positive"
        if self.min_order_size and shares < self.min_order_size:
            return False, (f"size {shares} shares below the venue minimum "
                           f"{self.min_order_size}")
        notional = shares * price
        if self.min_order_notional_usd and notional < self.min_order_notional_usd:
            return False, (f"notional ${notional:.4f} below the ${self.min_order_notional_usd} "
                           f"minimum")
        return True, "order satisfies the venue's mechanics"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_size": self.tick_size,
            "neg_risk": self.neg_risk,
            "min_order_size": self.min_order_size,
            "min_order_notional_usd": self.min_order_notional_usd,
            "taker_fee_rate": self.taker_fee_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "seconds_delay": self.seconds_delay,
            "accepting_orders": self.accepting_orders,
            "source": self.source,
            "is_real": self.is_real,
            "warnings": list(self.warnings),
        }

    @classmethod
    def assumed(cls, note: str = "") -> "MarketMechanics":
        """
        The mechanics to use when the venue could not be asked.

        Deliberately the FINEST tick and no minimum, because assuming a coarse
        tick would round prices further than the venue requires and quietly
        move the model away from the trade.
        """
        return cls(
            tick_size=MIN_TICK_SIZE,
            source="assumed_default",
            is_real=False,
            warnings=[note or ("mechanisms assumed: venue did not report tick "
                               "size, minimum size or neg-risk")],
        )

    @classmethod
    def from_clob_market_info(cls, info: Dict[str, Any]) -> "MarketMechanics":
        """
        Build from the venue's GET /clob-market-info payload.

        Field names follow MarketDetails: min_tick_size, neg_risk,
        min_order_size, maker_base_fee, taker_base_fee, fee_details,
        seconds_delay, accepting_orders.
        """
        info = info or {}
        warnings: list = []

        tick = info.get("min_tick_size") or info.get("tick_size")
        if tick is None:
            warnings.append("venue reported no tick size; using the finest tick")
            tick = MIN_TICK_SIZE
        tick = normalise_tick(tick)

        min_size = info.get("min_order_size")
        try:
            min_size = float(min_size) if min_size is not None else 0.0
        except (TypeError, ValueError):
            min_size = 0.0

        def _rate(*keys) -> float:
            for key in keys:
                value = info.get(key)
                if value is None and isinstance(info.get("fee_details"), dict):
                    value = info["fee_details"].get(key.replace("_base_fee", "_rate"))
                try:
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
            return 0.0

        accepting = info.get("accepting_orders")
        return cls(
            tick_size=tick,
            neg_risk=bool(info.get("neg_risk", False)),
            min_order_size=min_size,
            taker_fee_rate=_rate("taker_base_fee", "taker_fee_rate"),
            maker_fee_rate=_rate("maker_base_fee", "maker_fee_rate"),
            seconds_delay=int(info.get("seconds_delay") or 0),
            accepting_orders=True if accepting is None else bool(accepting),
            source="clob_market_info",
            is_real=True,
            warnings=warnings,
            raw=info,
        )


def mechanics_for_market(token_id: Optional[str], executor,
                         condition_id: Optional[str] = None) -> MarketMechanics:
    """
    Ask the venue for one market's mechanics, falling back to per-token reads
    and then to a labelled assumption.

    Order of preference:
      1. GET /clob-market-info by condition id - one call, every field.
      2. GET /tick-size + GET /neg-risk by token id - two calls, no minimum size.
      3. An assumed mechanics object that says so.
    """
    if executor is None:
        return MarketMechanics.assumed("no venue client available")

    info = None
    if condition_id:
        getter = getattr(executor, "get_clob_market_info", None)
        if getter is not None:
            try:
                info = getter(condition_id)
            except Exception:
                info = None
    if isinstance(info, dict) and info:
        return MarketMechanics.from_clob_market_info(info)

    if not token_id:
        return MarketMechanics.assumed("no token id to ask the venue about")

    tick = neg = None
    try:
        tick = executor.get_tick_size(token_id)
    except Exception:
        tick = None
    try:
        neg = executor.get_neg_risk(token_id)
    except Exception:
        neg = None
    if tick is None and neg is None:
        return MarketMechanics.assumed(
            "venue did not report tick size or neg-risk for this token")

    warnings = ["minimum order size unknown: the venue's per-market minimum "
                "was not read, so only the local floor is enforced"]
    return MarketMechanics(
        tick_size=normalise_tick(tick),
        neg_risk=bool(neg),
        min_order_size=0.0,
        source="tick_size_and_neg_risk",
        is_real=True,
        warnings=warnings,
    )
