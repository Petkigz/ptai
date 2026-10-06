"""
The directional lane: the honest way a crypto venue can be paper-traded.

The probability lane cannot price a crypto exchange and should not pretend to.
`ADA_PERP at 0.748` is $0.748, not a 74.8% chance: there is no Yes share to buy
and no 0/1 settlement, so a forecast probability has nothing to be an edge
against. That refusal is correct and stays (`Market.is_probability_market`,
`TradingAgentV3._screen_score`).

But it left two venues reading live markets every cycle and never doing
anything with them - the "unavailable even in paper mode" the operator reported.
The missing piece was not a client. It was a LANE: an instrument that matches
what a crypto venue actually sells.

What an exchange actually sells is a position: buy at the ask, sell at the bid,
pay the taker fee, and the P&L is the price when you close. So this module
implements exactly that, and nothing invented:

  * entry and exit are the venue's own quoted prices;
  * the stop and the target come from the venue's MEASURED volatility over the
    venue's own candles (close-to-close, scaled to the horizon) - not from a
    number chosen to make a trade look good;
  * the model's job is one directional probability ("will the price be higher at
    the horizon, 0.01-0.99?"), asked on this lane's own prompt - it is never
    compared against a price that is not a probability;
  * the trade opens only if the expected value after BOTH taker fees is at least
    `DIRECTIONAL_MIN_EV_PER_USD` per dollar, and sizes at half-Kelly capped at 6%
    of the directional purse;
  * settlement walks the venue's candles after the open: stop first, then
    target, and when one candle contains both, the STOP is assumed to have been
    hit first. That is the conservative reading and it is stated, not hidden.

MONEY SEPARATION. A directional position is a different instrument from a
probability bet, so it never touches the probability ledger:

  * it is stored in its own table (`directional_positions`),
  * it draws on its own paper purse (`directional_paper_bankroll`), seeded from a
    declared slice of the paper bankroll and never mixed back into it,
  * and therefore it does NOT count toward the resolved-probability record that
    unlocks live trading.

That last point is the one the operator must be able to check, so it is a test:
a closed directional position leaves `trades` and the calibration table empty.
Directional trades are their own evidence, reported in their own panel, and are
never dressed as something that qualifies a probability forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import math
import statistics

from loguru import logger


#: How long a directional position is held before it settles at the venue's own
#: close, in hours. One day: long enough for a daily-close forecast to mean
#: something, short enough that the paper record builds at the operator's pace.
DIRECTIONAL_HORIZON_HOURS = 24

#: The candle interval the volatility and settlement walk use.
DIRECTIONAL_CANDLE_HOURS = 1

#: Stop and target sit at this multiple of the measured horizon volatility, on
#: each side of the entry. One sigma each way is the plain reading: the trade
#: needs the price to reach a move the market's own recent behaviour makes
#: ordinary, not a move picked to flatter the model.
DIRECTIONAL_STOP_SIGMA = 1.0

#: The smallest expected profit per dollar staked, after both taker fees, that
#: justifies opening. A trade that only just clears zero has no room for the
#: volatility estimate being wrong.
DIRECTIONAL_MIN_EV_PER_USD = 0.01

#: Half-Kelly, capped at the same 6% of the purse the probability lane uses.
DIRECTIONAL_MAX_POSITION_PCT = 0.06

#: How much of the paper bankroll the directional purse is seeded with, once.
#: The two instruments keep separate books; this is the size of the directional
#: one, not a claim that directional trading deserves 20% of anything.
DIRECTIONAL_PURSE_SEED_PCT = 0.20

#: A quote wider than this is a venue problem, not an edge.
DIRECTIONAL_MAX_SPREAD_PCT = 0.01

#: Below this many candles there is no measured volatility to trade on.
DIRECTIONAL_MIN_CANDLES = 12

#: The most directional positions that may be open at once, across venues.
DIRECTIONAL_MAX_OPEN = 2

#: The most markets this lane will spend model time on in one cycle.
DIRECTIONAL_MAX_CANDIDATES = 3


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass
class DirectionalQuote:
    """A venue's own two-sided quote for one symbol, in its quoted currency."""

    venue_id: str
    symbol: str
    bid: float
    ask: float
    last: Optional[float] = None
    change_pct: Optional[float] = None
    volume_24h: Optional[float] = None
    source: str = ""
    is_real: bool = False
    reason: str = ""

    @property
    def mid(self) -> Optional[float]:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        mid = self.mid
        if mid and mid > 0:
            return (self.ask - self.bid) / mid
        return None

    @property
    def valid(self) -> bool:
        return (self.is_real and self.bid > 0 and self.ask > self.bid)

    def to_dict(self) -> Dict[str, Any]:
        return {"venue_id": self.venue_id, "symbol": self.symbol,
                "bid": self.bid, "ask": self.ask, "last": self.last,
                "spread_pct": self.spread_pct, "source": self.source,
                "is_real": self.is_real, "reason": self.reason}


def measured_sigma(candles: List[Dict[str, Any]],
                   horizon_hours: float = DIRECTIONAL_HORIZON_HOURS,
                   candle_hours: float = DIRECTIONAL_CANDLE_HOURS) -> Optional[float]:
    """
    The venue's own volatility, as a fraction of price, over the horizon.

    Close-to-close log returns from the venue's candles, scaled by the square
    root of the horizon. Returns None when there is not enough history to
    measure anything - a trade sized on an unmeasured volatility is a guess with
    a number attached to it.
    """
    closes: List[float] = []
    for candle in candles or []:
        if not isinstance(candle, dict):
            return None
        close = _f(candle.get("close"))
        if close is None or close <= 0:
            return None
        closes.append(close)
    if len(closes) < DIRECTIONAL_MIN_CANDLES:
        return None
    returns = [math.log(closes[i] / closes[i - 1])
               for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(returns) < 2:
        return None
    per_candle = statistics.pstdev(returns)
    if per_candle <= 0:
        return None
    scale = math.sqrt(max(1.0, float(horizon_hours) / max(0.5, float(candle_hours))))
    sigma = per_candle * scale
    # A measured volatility of zero on a real market means the candles are not
    # what we think they are; a sigma past 100% means the history is not usable.
    return sigma if 0.0 < sigma < 1.0 else None


@dataclass
class DirectionalPlan:
    """What the lane would do, and exactly why - including when it refuses."""

    venue_id: str
    symbol: str
    side: str = ""
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    sigma: float = 0.0
    model_prob: float = 0.0
    win_per_usd: float = 0.0
    loss_per_usd: float = 0.0
    ev_per_usd: float = 0.0
    kelly_fraction: float = 0.0
    size_usd: float = 0.0
    horizon_hours: float = DIRECTIONAL_HORIZON_HOURS
    fee_pct: float = 0.0
    reasons: List[str] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)

    @property
    def should_trade(self) -> bool:
        return not self.refusals and self.size_usd > 0 and bool(self.side)

    def to_dict(self) -> Dict[str, Any]:
        return {"venue_id": self.venue_id, "symbol": self.symbol, "side": self.side,
                "entry": self.entry, "stop": self.stop, "target": self.target,
                "sigma": self.sigma, "model_prob": self.model_prob,
                "win_per_usd": self.win_per_usd, "loss_per_usd": self.loss_per_usd,
                "ev_per_usd": self.ev_per_usd,
                "kelly_fraction": self.kelly_fraction, "size_usd": self.size_usd,
                "horizon_hours": self.horizon_hours, "fee_pct": self.fee_pct,
                "should_trade": self.should_trade,
                "reasons": list(self.reasons), "refusals": list(self.refusals)}


def plan_directional_trade(quote: DirectionalQuote, sigma: Optional[float],
                           prob_up: Optional[float], purse_usd: float,
                           fee_pct: float,
                           min_ev_per_usd: float = DIRECTIONAL_MIN_EV_PER_USD,
                           max_position_pct: float = DIRECTIONAL_MAX_POSITION_PCT,
                           stop_sigma: float = DIRECTIONAL_STOP_SIGMA,
                           max_spread_pct: float = DIRECTIONAL_MAX_SPREAD_PCT,
                           ) -> DirectionalPlan:
    """
    Decide whether the measured volatility and the model's direction support a
    position, and how large. Pure arithmetic - every input is either measured or
    declared, and every refusal says which one was missing.
    """
    plan = DirectionalPlan(venue_id=quote.venue_id, symbol=quote.symbol,
                           horizon_hours=DIRECTIONAL_HORIZON_HOURS,
                           fee_pct=float(fee_pct))
    if not quote.valid:
        plan.refusals.append(quote.reason or "no two-sided quote from the venue")
        return plan
    spread = quote.spread_pct
    if spread is None or spread > max_spread_pct:
        plan.refusals.append(
            f"the quote's spread is {('unknown' if spread is None else f'{spread:.2%}')} "
            f"against a {max_spread_pct:.0%} limit - the fee to enter and leave would "
            f"eat the trade")
        return plan
    if sigma is None or sigma <= 0:
        plan.refusals.append("no measured volatility from the venue's candles")
        return plan
    prob = _f(prob_up)
    if prob is None or not 0.01 <= prob <= 0.99:
        plan.refusals.append("the model gave no directional probability for this market")
        return plan

    plan.sigma = float(sigma)
    plan.model_prob = float(prob)
    # A LONG pays the ask and a SHORT sells at the bid: the venue's own prices,
    # taken on the side that costs the trader, not the mid.
    if prob >= 0.5:
        plan.side = "LONG"
        plan.entry = quote.ask
        plan.stop = quote.ask * (1.0 - stop_sigma * sigma)
        plan.target = quote.ask * (1.0 + stop_sigma * sigma)
        plan.reasons.append(
            f"the model puts {prob:.0%} on the price being higher at the horizon, "
            f"and the venue's own 1-sigma move over that horizon is {sigma:.2%}")
    else:
        plan.side = "SHORT"
        plan.entry = quote.bid
        plan.stop = quote.bid * (1.0 + stop_sigma * sigma)
        plan.target = quote.bid * (1.0 - stop_sigma * sigma)
        plan.reasons.append(
            f"the model puts {prob:.0%} on the price being higher at the horizon, "
            f"so this is the short side of the venue's own 1-sigma move "
            f"({sigma:.2%})")
    if plan.entry <= 0 or plan.stop <= 0 or plan.target <= 0:
        plan.refusals.append("the venue's quote does not produce a priceable "
                             "entry, stop and target")
        return plan

    round_trip_fee = 2.0 * float(fee_pct)
    if plan.side == "LONG":
        plan.win_per_usd = (plan.target / plan.entry - 1.0) - round_trip_fee
        plan.loss_per_usd = (1.0 - plan.stop / plan.entry) + round_trip_fee
    else:
        plan.win_per_usd = (1.0 - plan.target / plan.entry) - round_trip_fee
        plan.loss_per_usd = (plan.stop / plan.entry - 1.0) + round_trip_fee
    if plan.win_per_usd <= 0 or plan.loss_per_usd <= 0:
        plan.refusals.append("the measured move does not cover the round-trip fee")
        return plan

    # THE WIN PROBABILITY IS THE TRADE'S, NOT THE FORECAST'S. `prob` is the
    # model's probability that the price is HIGHER, so a LONG wins with `prob`
    # and a SHORT wins with `1 - prob`. Using `prob` for both made every short a
    # coin flip priced the wrong way round - it refused shorts that were
    # genuinely favoured and would have sized the ones it did take as though the
    # model disagreed with them.
    p_win = prob if plan.side == "LONG" else (1.0 - prob)
    plan.ev_per_usd = p_win * plan.win_per_usd - (1.0 - p_win) * plan.loss_per_usd
    if plan.ev_per_usd < min_ev_per_usd:
        plan.refusals.append(
            f"expected value after fees is {plan.ev_per_usd:+.2%} per dollar, below "
            f"the {min_ev_per_usd:.0%} this lane requires")
        return plan

    # Kelly for a bet with a win of `win_per_usd` and a loss of `loss_per_usd`
    # per dollar: f* = (p*b - (1-p)) / b with b = win/loss. Half of it, then the
    # operator's 6% cap - the same two limits the probability lane applies.
    b = plan.win_per_usd / plan.loss_per_usd
    kelly = (p_win * b - (1.0 - p_win)) / b
    plan.kelly_fraction = kelly
    fraction = max(0.0, kelly) / 2.0
    fraction = min(fraction, max_position_pct)
    size = max(0.0, float(purse_usd)) * fraction
    if size <= 0:
        plan.refusals.append("half-Kelly on this purse rounds the position to nothing")
        return plan
    plan.size_usd = round(size, 2)
    plan.reasons.append(
        f"{plan.side} wins with probability {p_win:.0%}; EV {plan.ev_per_usd:+.2%} per "
        f"dollar after {2 * fee_pct:.2%} round-trip fees; half-Kelly {fraction:.2%} of "
        f"a ${purse_usd:,.2f} directional purse")
    return plan


def settle_position(position: Dict[str, Any],
                    candles: List[Dict[str, Any]],
                    last_price: Optional[float] = None,
                    now: Optional[datetime] = None,
                    fee_pct: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Walk the venue's candles after the open and close the position.

    Returns {"exit_price", "exit_reason", "pnl_usd", "fees_usd"} or None while the
    position is still open. Stop is checked before target inside every candle, so
    a candle that spans both is settled at the STOP - the conservative reading,
    applied uniformly rather than guessed per trade.
    """
    side = str(position.get("side") or "").upper()
    entry = _f(position.get("entry_price"))
    size = _f(position.get("position_size_usd"))
    stop = _f(position.get("stop_price"))
    target = _f(position.get("target_price"))
    opened_at = position.get("opened_at")
    if side not in ("LONG", "SHORT") or entry is None or entry <= 0:
        return None
    if size is None or size <= 0 or stop is None or target is None:
        return None
    horizon = _f(position.get("horizon_hours")) or DIRECTIONAL_HORIZON_HOURS
    try:
        opened = datetime.fromisoformat(str(opened_at))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    now = now or datetime.now(timezone.utc)

    for candle in candles or []:
        if not isinstance(candle, dict):
            continue
        low, high = _f(candle.get("low")), _f(candle.get("high"))
        if low is None or high is None:
            continue
        if side == "LONG":
            # STOP FIRST: one candle can contain both levels, and the honest
            # assumption is that the losing one came first.
            if low <= stop:
                return _close(entry, stop, size, fee_pct, "stop")
            if high >= target:
                return _close(entry, target, size, fee_pct, "target")
        else:
            if high >= stop:
                return _close(entry, stop, size, fee_pct, "stop")
            if low <= target:
                return _close(entry, target, size, fee_pct, "target")

    hours_open = (now - opened).total_seconds() / 3600.0
    if hours_open < horizon:
        return None
    exit_price = _f(last_price)
    if exit_price is None:
        # The horizon is up and the venue has not quoted: settle at the last
        # candle close we already have, and say that is what happened.
        for candle in reversed(candles or []):
            exit_price = _f((candle or {}).get("close"))
            if exit_price is not None:
                break
    if exit_price is None or exit_price <= 0:
        return None
    return _close(entry, exit_price, size, fee_pct, "horizon")


def _close(entry: float, exit_price: float, size: float, fee_pct: float,
           reason: str) -> Dict[str, Any]:
    """
    The exit facts, and nothing else.

    P&L is NOT computed here: the sign depends on the side (`pnl_for`), and this
    function used to compute it with a long-only formula that a SHORT would have
    taken at face value. One place computes money, and it takes the side.
    """
    return {"exit_price": round(float(exit_price), 10),
            "exit_reason": reason}


def pnl_for(side: str, entry: float, exit_price: float, size: float,
            fee_pct: float) -> Dict[str, float]:
    """Signed P&L for a closed position. SHORT profits when the price falls."""
    round_trip = float(size) * float(fee_pct) * 2.0
    if str(side).upper() == "LONG":
        gross = float(size) * (float(exit_price) / float(entry) - 1.0)
    else:
        gross = float(size) * (float(entry) / float(exit_price) - 1.0) if exit_price > 0 else 0.0
    return {"gross_usd": round(gross, 6),
            "fees_usd": round(round_trip, 6),
            "pnl_usd": round(gross - round_trip, 6)}


# ---------------------------------------------------------------------------
# the model's one question
# ---------------------------------------------------------------------------

DIRECTIONAL_SYSTEM_PROMPT = (
    "You forecast prices for a trading agent. You are given ONE venue's quote and "
    "the volatility MEASURED from that venue's own candles. Answer with your own "
    "probability that the price will be higher at the horizon. This is a price, "
    "not a probability market: there is no Yes share, and 0.50 means you have no "
    "view. Do not invent news, and do not repeat a probability you produced for "
    "another market. Answer only JSON."
)


def directional_prompt(quote: DirectionalQuote, sigma: float,
                       horizon_hours: float = DIRECTIONAL_HORIZON_HOURS) -> str:
    """The prompt the lane asks, with every number in it measured or declared."""
    change = (f"{quote.change_pct:+.2f}%" if quote.change_pct is not None
              else "not supplied")
    volume = (f"{quote.volume_24h:,.0f}" if quote.volume_24h else "not supplied")
    return f"""MARKET: {quote.symbol} on {quote.venue_id}
QUOTE NOW: bid {quote.bid:,.8g} / ask {quote.ask:,.8g}
24h change: {change}
24h quote volume: {volume}
MEASURED VOLATILITY over the next {horizon_hours:.0f}h: {sigma:.2%} (one standard
deviation, computed from this venue's own hourly candles - a move this size or
smaller happens about two times in three)

TASK: your own probability, 0.01-0.99, that {quote.symbol} is HIGHER than
{quote.ask:,.8g} {horizon_hours:.0f} hours from now.
A trade is only opened when your answer beats the fees on BOTH sides of it, so
0.50 and 0.52 are useful answers: they mean nothing is worth doing here.
The values in the schema are placeholders showing the shape, not answers.
Respond ONLY JSON:
{{"prob_up":<your probability>,"confidence":<your confidence 0-1>,"basis":"<base_rate|trend|volatility|news|market>","reasoning":"<one line, name the evidence you used>"}}
"""


class DirectionalForecaster:
    """
    Asks the local model the lane's one question, and names the model that
    answered.

    No heuristic stands in when the model is unavailable: an unmeasured direction
    is exactly the thing this lane exists to avoid, so a cycle with no model
    opens no directional position and says why.
    """

    def __init__(self, router):
        self.router = router
        self.last_model = ""
        self.last_problem = ""
        self.calls = 0

    def is_available(self) -> bool:
        if self.router is None or not hasattr(self.router, "is_available"):
            self.last_problem = "no model router is configured for this process"
            return False
        try:
            available = bool(self.router.is_available())
        except Exception as e:  # noqa: BLE001
            self.last_problem = f"the model check failed ({type(e).__name__}: {e})"
            return False
        if not available:
            self.last_problem = "no local model server answered"
        return available

    def forecast(self, quote: DirectionalQuote, sigma: float,
                 horizon_hours: float = DIRECTIONAL_HORIZON_HOURS) -> Optional[float]:
        """
        The model's probability that the price is higher at the horizon, or None
        with `last_problem` set. A refusal is one fact, never a default.
        """
        self.last_model = ""
        self.last_problem = ""
        if not self.is_available():
            return None
        prompt = directional_prompt(quote, sigma, horizon_hours)
        try:
            response = self.router.chat(prompt=prompt,
                                        system=DIRECTIONAL_SYSTEM_PROMPT)
        except Exception as e:  # noqa: BLE001
            self.last_problem = f"the model call failed ({type(e).__name__}: {e})"
            return None
        self.calls += 1
        if response is None:
            self.last_problem = "the model returned nothing"
            return None
        self.last_model = str(getattr(response, "model", "") or "")
        parsed = getattr(response, "parsed_json", None)
        if not isinstance(parsed, dict):
            from ..llm.provider import extract_json_object
            parsed = extract_json_object(getattr(response, "content", "") or "")
        if not isinstance(parsed, dict):
            self.last_problem = (f"'{self.last_model}' answered, but its answer had "
                                 f"no usable JSON")
            return None
        value = _f(parsed.get("prob_up"))
        if value is None or not 0.01 <= value <= 0.99:
            self.last_problem = (f"'{self.last_model}' answered prob_up="
                                 f"{parsed.get('prob_up')!r}, which is not a "
                                 f"probability in 0.01-0.99")
            return None
        return value


class DirectionalPaperLane:
    """
    Open, hold and settle directional paper positions, with their own purse.

    The purse is `agent_state["directional_paper_bankroll"]`, seeded ONCE from
    `DIRECTIONAL_PURSE_SEED_PCT` of the paper bankroll. It is deliberately not the
    probability paper purse: a spot position and a probability bet are different
    instruments, and mixing their money would make the paper record unreadable in
    exactly the place the operator needs to trust it.
    """

    def __init__(self, storage, fee_pct_for: Optional[Callable[[str], float]] = None,
                 purse_seed_pct: float = DIRECTIONAL_PURSE_SEED_PCT):
        self.storage = storage
        self.fee_pct_for = fee_pct_for or (lambda venue_id: 0.001)
        self.purse_seed_pct = float(purse_seed_pct)

    # -- the purse -------------------------------------------------------
    def purse(self) -> float:
        raw = self.storage.get_state("directional_paper_bankroll")
        if raw is None:
            paper = float(self.storage.get_paper_bankroll())
            seeded = round(paper * self.purse_seed_pct, 2)
            self.storage.set_state("directional_paper_bankroll", str(seeded))
            logger.info(
                f"Directional paper purse seeded with ${seeded:.2f} "
                f"({self.purse_seed_pct:.0%} of the ${paper:.2f} paper bankroll) - "
                f"the two instruments keep separate books")
            return seeded
        value = _f(raw)
        return value if value is not None else 0.0

    def _set_purse(self, value: float) -> None:
        self.storage.set_state("directional_paper_bankroll", str(round(value, 2)))

    # -- positions -------------------------------------------------------
    def open_positions(self) -> List[Dict[str, Any]]:
        return self.storage.get_open_directional_positions()

    def open(self, plan: DirectionalPlan, market_question: str = "",
             opened_at: Optional[datetime] = None) -> Dict[str, Any]:
        """Open a position for a plan that passed every gate, and draw the purse."""
        if not plan.should_trade:
            return {"opened": False, "reason": "; ".join(plan.refusals) or "refused"}
        open_now = self.open_positions()
        if len(open_now) >= DIRECTIONAL_MAX_OPEN:
            return {"opened": False,
                    "reason": (f"{len(open_now)} directional positions are already "
                               f"open, at the {DIRECTIONAL_MAX_OPEN} the lane allows")}
        if any(p.get("symbol") == plan.symbol for p in open_now):
            return {"opened": False,
                    "reason": f"{plan.symbol} already has an open directional position"}
        purse = self.purse()
        if plan.size_usd > purse:
            return {"opened": False,
                    "reason": (f"the directional purse holds ${purse:,.2f} and this "
                               f"position needs ${plan.size_usd:,.2f}")}
        opened = (opened_at or datetime.now(timezone.utc)).isoformat()
        position_id = self.storage.log_directional_position({
            "venue_id": plan.venue_id, "symbol": plan.symbol, "side": plan.side,
            "entry_price": plan.entry, "stop_price": plan.stop,
            "target_price": plan.target, "position_size_usd": plan.size_usd,
            "model_prob": plan.model_prob, "sigma": plan.sigma,
            "ev_per_usd": plan.ev_per_usd, "fee_pct": plan.fee_pct,
            "horizon_hours": plan.horizon_hours, "opened_at": opened,
            "status": "open", "market_question": market_question,
            "notes": "; ".join(plan.reasons),
        })
        self._set_purse(purse - plan.size_usd)
        logger.info(
            f"DIRECTIONAL PAPER {plan.side} {plan.symbol} at {plan.entry:,.6g} "
            f"(stop {plan.stop:,.6g} / target {plan.target:,.6g}) "
            f"${plan.size_usd:,.2f} - model {plan.model_prob:.0%} up, "
            f"EV {plan.ev_per_usd:+.2%} per dollar, no real capital")
        return {"opened": True, "position_id": position_id,
                "size_usd": plan.size_usd, "side": plan.side,
                "symbol": plan.symbol, "entry": plan.entry}

    def settle_open(self, quote_lookup: Callable[[str, str], Optional[DirectionalQuote]],
                    candles_lookup: Callable[[str, str], List[Dict[str, Any]]],
                    now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Settle every position the venue's candles have resolved, and report how
        many were still open and why. Never closes a position on a guess: a
        symbol whose candles cannot be read stays open and says so.
        """
        report = {"checked": 0, "closed": 0, "still_open": 0, "unreadable": 0,
                  "pnl_usd": 0.0, "results": [], "reasons": []}
        for position in self.open_positions():
            report["checked"] += 1
            venue_id = str(position.get("venue_id") or "")
            symbol = str(position.get("symbol") or "")
            try:
                candles = candles_lookup(venue_id, symbol) or []
            except Exception as e:  # noqa: BLE001
                report["unreadable"] += 1
                report["reasons"].append(f"{symbol}: candles unreadable ({type(e).__name__})")
                continue
            if not candles:
                report["unreadable"] += 1
                report["reasons"].append(f"{symbol}: no candles from {venue_id}")
                continue
            quote = None
            try:
                quote = quote_lookup(venue_id, symbol)
            except Exception:  # noqa: BLE001
                quote = None
            last_price = quote.last if quote is not None and quote.last else (
                quote.mid if quote is not None else None)
            try:
                fee_pct = float(self.fee_pct_for(venue_id) or 0.0)
            except Exception:  # noqa: BLE001
                fee_pct = 0.0
            verdict = settle_position(position, candles, last_price=last_price,
                                      now=now, fee_pct=fee_pct)
            if verdict is None:
                report["still_open"] += 1
                continue
            side = str(position.get("side") or "").upper()
            entry = float(position.get("entry_price") or 0.0)
            size = float(position.get("position_size_usd") or 0.0)
            money = pnl_for(side, entry, verdict["exit_price"], size, fee_pct)
            closed = self.storage.close_directional_position(
                int(position["id"]), exit_price=verdict["exit_price"],
                exit_reason=verdict["exit_reason"], pnl_usd=money["pnl_usd"],
                fees_usd=money["fees_usd"])
            if closed:
                self._set_purse(self.purse() + size + money["pnl_usd"])
                report["closed"] += 1
                report["pnl_usd"] = round(report["pnl_usd"] + money["pnl_usd"], 6)
                report["results"].append({
                    "symbol": symbol, "venue_id": venue_id, "side": side,
                    "entry": entry, "exit": verdict["exit_price"],
                    "reason": verdict["exit_reason"], "pnl_usd": money["pnl_usd"],
                })
                logger.info(
                    f"DIRECTIONAL PAPER CLOSED {side} {symbol} at "
                    f"{verdict['exit_price']:,.6g} ({verdict['exit_reason']}): "
                    f"P&L ${money['pnl_usd']:+,.2f} paper")
            else:
                report["still_open"] += 1
        return report

    def record(self, limit: int = 50) -> Dict[str, Any]:
        """The directional lane's own scoreboard, kept apart from the rest."""
        try:
            rows = self.storage.get_directional_positions(limit=limit)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"{type(e).__name__}: {e}"}
        closed = [r for r in rows if str(r.get("status")) == "closed"]
        wins = [r for r in closed if float(r.get("pnl_usd") or 0.0) > 0]
        pnl = sum(float(r.get("pnl_usd") or 0.0) for r in closed)
        return {
            "available": True,
            "purse_usd": self.purse(),
            "open": [r for r in rows if str(r.get("status")) == "open"],
            "closed": closed,
            "closed_count": len(closed),
            "wins": len(wins),
            "win_rate": (len(wins) / len(closed)) if closed else None,
            "pnl_usd": round(pnl, 6),
            "note": ("Directional paper trades are their own instrument and their own "
                     "purse: they are not probability forecasts, they do not feed the "
                     "calibration score, and they do NOT count toward the resolved "
                     "probability trades that unlock live trading."),
        }
