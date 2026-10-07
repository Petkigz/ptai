"""
V3 Strategy Engine - evaluates venue × market × strategy
Core of PTAI V3: genuinely multi-venue, multi-strategy opportunity engine

Instead of "Find a Polymarket trade", it asks:
"Find the best legitimate opportunity available to my capital right now across all venues and strategies"

Reports:
I scanned 1,200 opportunities.
Polymarket: 3 candidates
Kalshi: 5 candidates
Manifold: 1 candidate
Crypto: 8 candidates
Stocks: 2 candidates
After fees/liquidity/uncertainty: 2 actually tradeable
Best opportunity: Venue B with strategy X

Then learning system determines which combinations actually demonstrate edge.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType
from ..venues.registry import VenueRegistry

from .fair_value import FairValueEngine
from .edge import HUNT_MISPRICING_MIN, EdgeCalculator, hunted_mispricing


# The PAPER lane's own bar, deliberately below the live one (8% mispricing on
# the mid AND the conservative estimate clearing the price AND confidence >=
# 60%). The live bar is the operator's rule and it does not move.
#
# This lane exists because the qualification lifecycle needs resolved paper
# trades - a venue earns live capital with 100 of them - and because a round
# that can never place a single trade teaches the operator and the learning
# chains nothing. What it does NOT do is invent an edge or hide a cost: it
# requires a real two-sided book, a model that actually answered, a mispricing
# of at least 5% on the side traded, and at least 2 cents per share left after
# paying the executable price and every cash cost. It is labelled everywhere it
# appears - the log, the round, the console - as paper/exploration.
EXPLORATION_MISPRICING_MIN = 0.05
EXPLORATION_EXECUTABLE_EDGE_MIN = 0.02

# How many PAPER/EXPLORATION trades one cycle may place. One was the ceiling
# while the lane only had to prove it could trade at all; the operator now needs
# the record to build - "i need to get Polymarket paper record to 100 resolved
# trades so live unlocks" - and a round that finds three qualifying markets may
# as well learn from three. Every other part of the lane's bar is unchanged, each
# trade is $1 (or the sizing the paper bankroll justifies), the exposure manager
# still caps what can be open at once, and a market already held is still skipped.
PAPER_TRADES_PER_CYCLE_DEFAULT = 3
PAPER_TRADES_PER_CYCLE_CEILING = 5


def paper_trades_per_cycle_default() -> int:
    """`PTAI_PAPER_TRADES_PER_CYCLE`, default 3, ceiling 5."""
    import os

    raw = os.environ.get("PTAI_PAPER_TRADES_PER_CYCLE", "")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return PAPER_TRADES_PER_CYCLE_DEFAULT
    return max(1, min(PAPER_TRADES_PER_CYCLE_CEILING, value))


def resolution_days(market) -> Optional[float]:
    """
    Days until this market settles, or None when the market does not say.

    CAPITAL x TIME IS PART OF THE RETURN. A 5% edge that resolves tomorrow is not
    the same trade as a 5% edge that resolves in three months: the second ties the
    capital up for a quarter to earn the same money. That is why the fine print of
    the economic framework puts capital x time x execution risk in the
    denominator, and why a market with no end date is not treated as if it
    resolved tomorrow - None means unknown, and unknown sorts last.
    """
    end = getattr(market, "end_date", None)
    if end is None:
        return None
    try:
        if end.tzinfo is None:
            from datetime import timezone as _tz

            end = end.replace(tzinfo=_tz.utc)
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        now = _dt.now(_tz.utc)
        seconds = (end - now).total_seconds()
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0.0, seconds / 86400.0)


def _refusal_text(fv_result) -> str:
    """The gate's own sentence, without the boolean it is prefixed with."""
    text = str(getattr(fv_result, "reasoning", "") or "").split(
        "Decision: ", 1)[-1].split(" | ", 1)[0].strip()
    for prefix in ("False because ", "True because ", "False. ", "True. "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text[:200]


def exploration_eligible(fv_result, executable_edge: float,
                         execution_quality: float,
                         liquidity_score: float) -> bool:
    """Is this refused market a PAPER candidate for the shadow lane?"""
    if fv_result is None or getattr(fv_result, "should_trade", False):
        return False
    # There has to be a real, two-sided, executable book. An estimated book has
    # no price to pay, and a spread too wide to cross is a loss however good the
    # estimate is: both are refusals the lane must not work around.
    if not bool(getattr(fv_result, "book_is_real", False)):
        return False
    if str(getattr(fv_result, "blocked_by", "") or ""):
        return False
    # The best estimate must survive the price and the cash costs. If it does
    # not, there is no trade here at any estimate and the lane must not take it.
    if float(executable_edge or 0.0) < EXPLORATION_EXECUTABLE_EDGE_MIN:
        return False
    if abs(float(getattr(fv_result, "edge", 0.0) or 0.0)) < EXPLORATION_MISPRICING_MIN:
        return False
    if float(liquidity_score or 0.0) < 0.3 or float(execution_quality or 0.0) < 0.3:
        return False
    # A model must have ANSWERED this market. A heuristic rule of thumb is not
    # evidence and the shadow lane exists to produce evidence.
    chain = getattr(getattr(fv_result, "forecast_result", None), "chain", {}) or {}
    if chain.get("llm_raw") is None:
        return False
    return True
from .strategy_selector import StrategyType, StrategySelector
from .arbitrage import ArbitrageEngine
from .event_trading import EventTradingEngine
from .market_making import MarketMakingEngine
from .momentum import MomentumEngine
from .expected_ev import ExpectedNetEVEngine


@dataclass
class VenueScanReport:
    venue_id: str
    venue_type: str
    total_discovered: int
    after_cheap: int
    after_liquidity: int
    candidates: int  # after fast model / strategy evaluation
    tradeable: int   # after effective edge + risk
    top_opportunity: Optional[VenueOpportunity] = None
    avg_edge: float = 0.0
    avg_score: float = 0.0
    # WHAT THE SCAN ACTUALLY DID with this venue's markets, as opposed to what
    # the screen PLANNED. The operator's log said "199 priced on the cheap
    # context only" while the very next line of the same cycle said 99 markets
    # were "not evaluated - their book is an estimate or absent". Prices and
    # plans are different facts and they get different counters.
    evaluated: int = 0          # markets that reached the pricing stage
    skipped_no_book: int = 0    # ...of the iterated ones, how many had no usable book
    beyond_cap: int = 0         # markets the per-venue cap left unpriced
    # The best market this venue priced that still did not qualify to trade, and
    # what refused it. "How close was the round?" is the question an operator
    # asking "how long do I have to run this" needs answered, and a scan that
    # reports only zeros cannot answer it.
    best_near_miss: Optional[Dict[str, Any]] = None
    # Paper candidates this venue produced: refused by the LIVE gates, eligible
    # for the learning lane. They travel on the report rather than in a third
    # return value, so `scan_venue`'s contract (report, opportunities) is
    # unchanged for every existing caller.
    exploration_candidates: List[Any] = field(default_factory=list)


@dataclass
class MultiVenueScanResult:
    total_scanned: int
    total_after_cheap: int
    total_after_liquidity: int
    total_candidates: int
    total_tradeable: int
    venue_reports: List[VenueScanReport] = field(default_factory=list)
    all_opportunities: List[VenueOpportunity] = field(default_factory=list)
    final_selected: List[VenueOpportunity] = field(default_factory=list)
    arbitrage_opportunities: List = field(default_factory=list)
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)
    execution_time: float = 0.0
    reasoning: str = ""
    best_opportunity: Optional[VenueOpportunity] = None
    # Paper/shadow candidates: refused by the LIVE gates, eligible for the
    # learning lane. Kept out of `all_opportunities` so no count the operator
    # reads (candidates, tradeable, best opportunity) moves because of them.
    exploration_candidates: List[VenueOpportunity] = field(default_factory=list)
    # ...and the closest call of the whole round, for the same reason.
    best_near_miss: Optional[Dict[str, Any]] = None


def _best_near_miss(venue_reports) -> Optional[Dict[str, Any]]:
    """The closest call of the round, across every venue that priced one."""
    best = None
    for report in venue_reports or []:
        row = getattr(report, "best_near_miss", None)
        if not row:
            continue
        if best is None or abs(float(row.get("executable_edge") or 0.0)) > abs(
                float(best.get("executable_edge") or 0.0)):
            best = dict(row)
    return best


def _slippage_from_book(orderbook, market_price: float):
    """
    Slippage implied by the book in front of us, or None if there is not one.

    A measured number here is what lets the EV stage tell a measured cost from
    an assumed one. Returning 0.0 for "no book" would be the same lie as the
    old default, one level up: it would present an unmeasured cost as a measured
    zero, and zero slippage is the most favourable assumption available.
    """
    if not isinstance(orderbook, dict) or not orderbook.get("is_real", False):
        return None
    try:
        from ..markets.orderbook import read_depth, execution_quality_from_book

        quality = execution_quality_from_book(orderbook, market_price)
        # Execution quality is 1.0 on a deep, tight book and falls with spread
        # and thin depth. Turned back into a cost so the same measurement drives
        # both the opportunity score and the EV, instead of two different models.
        return max(0.0, min(0.05, (1.0 - float(quality)) * 0.02))
    except Exception as e:
        logger.debug(f"Slippage could not be measured from the book: {e}")
        return None


def _execution_quality_from_book(orderbook, market) -> float:
    """Kept as the name this module's tests import; see the shared implementation."""
    from ..markets.orderbook import execution_quality_from_book
    return execution_quality_from_book(orderbook, market)


class StrategyEngineV3:
    """
    V3 Strategy Engine: venue × market × strategy
    Evaluates all combinations and ranks on common basis
    """
    def __init__(self, 
                 venue_registry: VenueRegistry = None,
                 fair_value_engine: FairValueEngine = None,
                 edge_calculator: EdgeCalculator = None,
                 strategy_selector: StrategySelector = None):
        self.venue_registry = venue_registry
        self.fair_value_engine = fair_value_engine or FairValueEngine()
        self.edge_calculator = edge_calculator or EdgeCalculator()
        self.strategy_selector = strategy_selector or StrategySelector()
        
        # Strategy engines
        self.arbitrage_engine = ArbitrageEngine(min_spread=0.03)
        self.event_engine = EventTradingEngine()
        self.mm_engine = MarketMakingEngine()
        self.momentum_engine = MomentumEngine()
        self.expected_ev_engine = ExpectedNetEVEngine()
        
        # Filters
        self.min_volume_24h = 500
        self.min_liquidity = 100
        self.max_spread = 0.10

    # THE MARKETS A VENUE SCAN ACTUALLY EVALUATES. The screen (stage 1) and the
    # scan (stage 2) each had their own idea of which markets matter: the screen
    # ranked on book quality and depth, the scan took the top 100 by volume and
    # liquidity. So the operator's cycle chose 8 markets for deep analysis and
    # then dropped all 8 at the per-venue cap, spent no model time at all, and
    # printed a model line that blamed the model. One method, used by both, is
    # how the two cannot drift apart again.
    evaluate_limit: int = 100

    def markets_that_will_be_evaluated(self, markets: List[Market]) -> List[Market]:
        """(cheap filters -> liquidity order -> per-venue cap) as the scan does it."""
        return self.liquidity_filter(self.cheap_filters(markets))[:self.evaluate_limit]

    def cheap_filters(self, markets: List[Market]) -> List[Market]:
        """
        The cheap stage: what is worth reading a book for.

        The volume and liquidity floors exist to keep model time off markets
        nobody trades - and they only mean that when the VENUE PUBLISHES the
        figures. PredictIt's API publishes a best price per contract and no
        volume and no depth at all, so comparing its 0.0 against a floor says
        "nobody trades here" about a market whose trading the venue simply does
        not report. Those markets skip the two floors (their books and spreads
        are still checked, and one venue cannot take more than its turn of the
        deep budget), and every other venue is unchanged.
        """
        filtered = []
        for m in markets:
            if bool(getattr(m, "volume_is_published", True)):
                if m.volume_24h < self.min_volume_24h:
                    continue
                if m.liquidity < self.min_liquidity:
                    continue
            if not m.active or m.closed:
                continue
            # Allow extreme prices for mean reversion strategy, but filter for others
            # Keep all for V3, let strategy decide
            filtered.append(m)
        return filtered

    def liquidity_filter(self, markets: List[Market]) -> List[Market]:
        sorted_markets = sorted(markets, key=lambda x: (x.volume_24h, x.liquidity), reverse=True)
        limit = min(200, max(50, len(sorted_markets)))
        return sorted_markets[:limit]

    def evaluate_market_with_all_strategies(self, market: Market, context: Dict = None,
                                            record: Optional[List[Dict[str, Any]]] = None) -> List[VenueOpportunity]:
        """
        Core V3: evaluate single market with ALL strategies
        Returns list of VenueOpportunity, one per strategy that finds edge.

        `record`, when given, collects the pricing facts for this market - the
        numbers the round's "how close was it" line is built from. A scan that
        reports only zeros cannot answer "how long do I have to run this", and
        the answer has to come from the same numbers the gates used.
        """
        context = context or {}
        opportunities: List[VenueOpportunity] = []
        venue_id = getattr(market, 'source', 'unknown')
        venue_id_str = venue_id.value if hasattr(venue_id, 'value') else str(venue_id)
        # Handle mock venues with raw venue field
        if market.raw.get("venue"):
            venue_id_str = market.raw["venue"]
        
        # Strategy 1: Mispricing (fair value vs market) - existing engine
        try:
            _t0 = time.time()
            fv_result = self.fair_value_engine.estimate(market, context=context)
            _elapsed = time.time() - _t0
            # One line per market. The operator's log had five hundred seconds of
            # silence per market and then a paragraph; this says which market,
            # how long it took and what was decided, in the order it happened.
            logger.info(
                f"Priced {market.id} in {_elapsed:.1f}s: "
                f"{'TRADEABLE ' + str(getattr(fv_result, 'side', 'YES')) if fv_result.should_trade else 'no trade'}"
                f" (fair {fv_result.fair_value:.3f} vs market {market.best_price:.3f}, "
                f"edge {fv_result.edge:+.3f}, conf {fv_result.confidence:.2f})"
                + (f" - {fv_result.reasoning.split('Decision: ', 1)[-1].split(' | ')[0]}"
                   if not fv_result.should_trade else ""))
            # ...and HOW that fair value was built, term by term. The operator
            # could see the LLM say 65% and the ensemble say 58% with nothing in
            # between; this is the between.
            _fc = getattr(fv_result, "forecast_result", None)
            if _fc is not None:
                try:
                    logger.info(f"  why {market.id}: {_fc.explain()}")
                except Exception as _e:  # noqa: BLE001
                    logger.debug(f"could not explain {market.id}: {_e}")
            # The opportunity is built on the hunt criterion plus a positive
            # post-cost edge. A 5% POST-COST floor also stood here, so the same
            # profitable NO trade (effective 0.028) was refused before it existed
            # - the mispricing was never evaluated by anything downstream.
            #
            # A market the LIVE gates refused can still be a PAPER candidate:
            # `exploration_eligible` is the lane's own bar (see the constants at
            # the top of this file), and the two are kept apart here so that no
            # count the operator reads - candidates, tradeable, best opportunity
            # - moves because a shadow trade was built.
            _exec_quality = _execution_quality_from_book(
                context.get("orderbook"), market)
            _liquidity_score = min(1.0, market.liquidity / 10000)
            _is_exploration = exploration_eligible(
                fv_result,
                getattr(fv_result, "effective_edge", 0.0),
                _exec_quality, _liquidity_score)
            if record is not None:
                record.append({
                    "market_id": market.id,
                    "question": (market.question or "")[:110],
                    "side": getattr(fv_result, "side", "YES"),
                    "fair": float(getattr(fv_result, "fair_value", 0.0) or 0.0),
                    "market": float(market.best_price or 0.0),
                    "mispricing": float(getattr(fv_result, "edge", 0.0) or 0.0),
                    "executable_edge": float(
                        getattr(fv_result, "effective_edge", 0.0) or 0.0),
                    "conservative_executable_edge": float(
                        getattr(fv_result, "conservative_executable_edge", 0.0) or 0.0),
                    "conservative_fair": float(
                        getattr(fv_result, "conservative_fair", 0.0) or 0.0),
                    "confidence": float(getattr(fv_result, "confidence", 0.0) or 0.0),
                    "uncertainty": float(getattr(fv_result, "uncertainty", 0.0) or 0.0),
                    "model_answered": (
                        (getattr(getattr(fv_result, "forecast_result", None),
                                 "chain", {}) or {}).get("llm_raw") is not None),
                    "should_trade": bool(getattr(fv_result, "should_trade", False)),
                    "refusal": (_refusal_text(fv_result)),
                })
            if ((fv_result.should_trade and fv_result.effective_edge > 0)
                    or _is_exploration):
                opp = VenueOpportunity(
                    market=market,
                    venue_id=venue_id_str,
                    venue_type=VenueType.PREDICTION,
                    # The side the edge was measured on. Re-deriving it from the
                    # fair value here is how the two could disagree: the edge was
                    # computed for neither side in particular, and this line then
                    # picked one.
                    side=getattr(fv_result, "side", None) or (
                        "YES" if fv_result.fair_value > market.best_price else "NO"),
                    market_price=market.best_price,
                    estimated_fair=fv_result.fair_value,
                    raw_edge=fv_result.edge,
                    effective_edge=fv_result.effective_edge,
                    confidence=fv_result.confidence,
                    uncertainty=fv_result.uncertainty,
                    liquidity_score=_liquidity_score,
                    # Measured from the book when there is one. This was the
                    # literal 0.8 for every mispricing opportunity, so a market
                    # with a wide spread and no depth scored the same as a deep
                    # one - and execution_quality feeds the opportunity score, the
                    # EV penalty and the ranking, so the ranking was partly made
                    # of a constant.
                    execution_quality=_exec_quality,
                    category=context.get("category", "mispricing"),
                    # What this trade will actually cost, from the venue that
                    # will charge it and the book it will trade against. None
                    # where neither exists, which the EV stage reports as an
                    # assumption rather than silently costing at a default.
                    fees_pct=context.get("fee_taker_pct"),
                    slippage_pct=_slippage_from_book(context.get("orderbook"),
                                                     market.best_price),
                    order_gas_usd=context.get("order_gas_usd"),
                    sources=fv_result.forecast_result.sources if fv_result.forecast_result else ["fair_value"],
                    reasoning=fv_result.reasoning,
                    bull_case=fv_result.contradiction_report.bull_case if fv_result.contradiction_report else "",
                    bear_case=fv_result.contradiction_report.bear_case if fv_result.contradiction_report else "",
                    resolution_risks=fv_result.resolution_analysis.risks if fv_result.resolution_analysis else [],
                    # A shadow trade is not a trade PTAI may deploy live capital
                    # on, and `should_trade` stays False so nothing downstream
                    # can mistake it for one.
                    should_trade=bool(fv_result.should_trade),
                )
                # The chain travels WITH the opportunity, so the operator's
                # console can show which component said what for the trades that
                # were actually proposed - not only for the ones in the log file.
                _fc = getattr(fv_result, "forecast_result", None)
                _chain = {}
                _components = []
                _explain = ""
                if _fc is not None:
                    _chain = dict(getattr(_fc, "chain", {}) or {})
                    _components = list(getattr(_fc, "components", []) or [])
                    try:
                        _explain = _fc.explain()
                    except Exception:  # noqa: BLE001 - an explanation must not
                        _explain = ""  # stop a priced market
                opp.calculate_common_score()
                # Tag strategy. The chain attached two lines above used to be
                # WIPED here by this assignment - the comment said the chain
                # travels with the opportunity so the console can show which
                # component said what, and this line is why no console ever could.
                opp.raw = {
                    "strategy": ("mispricing (paper/exploration)" if _is_exploration
                                 else "mispricing"),
                    "venue": venue_id_str,
                    "fair_value_chain": _chain,
                    "components": _components,
                    "explain": _explain,
                }
                opp.raw["price_paid"] = float(
                    getattr(fv_result, "price_paid", 0.0) or 0.0)
                if _is_exploration:
                    _refusal = _refusal_text(fv_result)[:160]
                    opp.raw["exploration"] = True
                    opp.raw["exploration_because"] = (
                        "the live gates refused it (" + _refusal + ") - the paper "
                        "lane's bar is a real book, a model that answered, at least "
                        + f"{EXPLORATION_MISPRICING_MIN:.0%} mispricing and at least "
                        + f"{EXPLORATION_EXECUTABLE_EDGE_MIN:.0%} left after costs on "
                        "the price paid")
                    opp.raw["conservative_executable_edge"] = float(
                        getattr(fv_result, "conservative_executable_edge", 0.0) or 0.0)
                opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Mispricing eval failed for {market.id}: {e}")

        # Strategy 2: Event trading
        try:
            event_signal = self.event_engine.evaluate(market, context=context)
            if event_signal.should_trade:
                opp = self.event_engine.to_venue_opportunity(market, event_signal, context=context)
                if opp:
                    opp.raw = {"strategy": "event_trading", "venue": venue_id_str}
                    opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Event trading eval failed for {market.id}: {e}")

        # Strategy 3: Market making (if orderbook available)
        try:
            orderbook = context.get("orderbook", {})
            if orderbook:
                mm_signal = self.mm_engine.evaluate(market, orderbook=orderbook)
                if mm_signal.should_trade:
                    opp = self.mm_engine.to_venue_opportunity(market, mm_signal)
                    if opp:
                        opp.raw = {"strategy": "market_making", "venue": venue_id_str}
                        opportunities.append(opp)
        except Exception as e:
            logger.debug(f"MM eval failed for {market.id}: {e}")

        # Strategy 4 & 5: Momentum and Mean Reversion
        try:
            mom_signals = self.momentum_engine.evaluate(market, context=context)
            mom_opps = self.momentum_engine.to_venue_opportunities(market, mom_signals)
            for opp in mom_opps:
                opp.raw = {"strategy": opp.category, "venue": venue_id_str}
                opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Momentum eval failed for {market.id}: {e}")

        return opportunities

    async def scan_venue(self, venue_id: str, markets: List[Market], context_provider=None,
                         book_lookup=None,
                         must_price: Optional[set] = None) -> VenueScanReport:
        """
        Scan single venue with all strategies.

        `must_price` is the screen's shortlist: the markets this cycle has
        already given model time to because they have a validated book. They are
        priced FIRST, and the rest of the eligible list fills the remaining
        slots. Without it the two stages disagreed in practice: the screen
        chooses on book quality, the scan caps on volume, and the cap was
        applied before the books were known - so on 2026-09-29 19:33 the model
        was asked about ONE market out of 900 while 74 had a validated book.
        """
        start_total = len(markets)
        
        after_cheap = self.cheap_filters(markets)
        after_liquidity = self.liquidity_filter(after_cheap)
        
        # NOT A PROBABILITY, NOT A TRADE HERE. A crypto exchange quotes
        # currency prices; pricing them as Yes probabilities produced 6300%
        # spreads and 'fair 0.100 vs the 0.748 a share actually costs'. Refused
        # at the door, counted, and said out loud - once per venue, not once per
        # market.
        not_probability = [m for m in after_liquidity
                           if not getattr(m, "is_probability_market", True)]
        if not_probability:
            after_liquidity = [m for m in after_liquidity
                               if getattr(m, "is_probability_market", True)]
            logger.info(
                f"{venue_id}: {len(not_probability)} market(s) not evaluated - this "
                f"venue quotes PRICES, not probabilities, so there is no Yes share "
                f"to buy and no edge to compute (they stay visible in the venue "
                f"inventory; they are not priced here)")

        # Evaluate each market with all strategies
        all_opps: List[VenueOpportunity] = []
        exploration_opps: List[VenueOpportunity] = []
        priced_records: List[Dict[str, Any]] = []
        skipped_no_book = 0
        evaluated = 0
        # THE MARKETS GIVEN MODEL TIME ARE THE ONES PRICED FIRST, then the
        # venue's own cap decides the rest. Both stages now target one set.
        _must = {str(x) for x in (must_price or set())}
        if _must:
            _priority = [m for m in after_liquidity if str(m.id) in _must]
            _rest = [m for m in after_liquidity if str(m.id) not in _must]
            to_price = (_priority + _rest)[:self.evaluate_limit]
        else:
            to_price = after_liquidity[:self.evaluate_limit]
        for market in to_price:  # per-venue cap
            # A market whose book is not real cannot produce a cost or an edge.
            # The operator's 2026-09-29 log ran every one of them through the
            # whole stack - resolution analysis, contradiction, forecaster,
            # ensemble, edge engine, strategy engine, six lines each, four
            # hundred markets - and every one of them ended in "REFUSED:
            # orderbook is not real". The count is reported once instead.
            if book_lookup is not None:
                try:
                    _book = book_lookup(market.id)
                except Exception:  # noqa: BLE001 - an unreadable book is not a real one
                    _book = None
                if isinstance(_book, dict) and _book and not (
                        _book.get("is_real") and _book.get("validated")):
                    skipped_no_book += 1
                    continue
            context = {}
            if context_provider:
                try:
                    context = await context_provider.get_context(market)
                except Exception as e:
                    # NEVER silent. This except used to swallow an
                    # AttributeError - the provider had no get_context - and
                    # substitute a fabricated orderbook for every market on every
                    # cycle. The forecast was then built on a placeholder spread
                    # and a depth copied from the market's own liquidity field,
                    # with nothing logged, so the pipeline looked connected while
                    # no news, sentiment, research or real book reached it.
                    logger.error(
                        f"Context provider FAILED for {market.id}: "
                        f"{type(e).__name__}: {e}. Falling back to a degraded "
                        f"context - the forecast for this market is built WITHOUT "
                        f"news, sentiment, research or a real orderbook, and is "
                        f"marked degraded so nothing downstream mistakes it for a "
                        f"researched forecast.")
                    context = {
                        "orderbook": {"spread": 0.02, "is_real": False},
                        "category": "unknown",
                        "degraded": True,
                        "degraded_reason": f"{type(e).__name__}: {e}",
                        "sources": [],
                    }
            else:
                # No provider at all is also a degraded forecast, not a normal
                # one. Same reason: silence here is indistinguishable from a
                # researched market.
                context = {
                    "orderbook": {"spread": 0.02, "is_real": False},
                    "category": "unknown",
                    "degraded": True,
                    "degraded_reason": "no context provider supplied",
                    "sources": [],
                }
            
            evaluated += 1
            opps = self.evaluate_market_with_all_strategies(
                market, context=context, record=priced_records)
            for _opp in opps:
                # PAPER CANDIDATES ARE NOT CANDIDATES. A shadow trade must not
                # move `candidates`, `tradeable`, `avg_edge` or the round's
                # "best opportunity" - all of which the operator reads as what
                # the LIVE gates found.
                if (_opp.raw or {}).get("exploration"):
                    exploration_opps.append(_opp)
                else:
                    all_opps.append(_opp)
        
        if skipped_no_book:
            logger.info(
                f"{venue_id}: {skipped_no_book} market(s) not evaluated - their book "
                f"is an estimate or absent, so there is no cost and no edge to "
                f"compute (they are counted, not priced)")

        # HOW CLOSE WAS IT. The best priced market that did not qualify, with
        # the gate that refused it - a refusal of "no mispricing" and a refusal
        # of "the conservative estimate missed the ask by half a cent" are very
        # different answers to "how long do I have to run this".
        def _near_miss_key(row):
            return abs(float(row.get("executable_edge") or 0.0))
        near_miss = None
        for row in priced_records:
            if row.get("should_trade"):
                continue
            if near_miss is None or _near_miss_key(row) > _near_miss_key(near_miss):
                near_miss = row
        
        # Candidates after strategy evaluation
        candidates = [o for o in all_opps if o.effective_edge >= 0.03]
        # 8% of MISPRICING, not 8% of post-cost edge - see
        # `hunted_mispricing`. Costs are charged in the net EV terms.
        tradeable = [o for o in all_opps
                     if hunted_mispricing(o) >= HUNT_MISPRICING_MIN
                     and o.confidence >= 0.6 and o.should_trade]
        
        # Sort tradeable by score
        tradeable_sorted = sorted(tradeable, key=lambda x: x.score, reverse=True)
        top_opp = tradeable_sorted[0] if tradeable_sorted else (sorted(candidates, key=lambda x: x.score, reverse=True)[0] if candidates else None)
        
        avg_edge = sum(o.effective_edge for o in candidates) / len(candidates) if candidates else 0
        avg_score = sum(o.score for o in candidates) / len(candidates) if candidates else 0
        
        # Determine venue type
        venue_type = "unknown"
        if markets and len(markets) > 0:
            src = getattr(markets[0], 'source', 'unknown')
            if hasattr(src, 'value'):
                venue_type = src.value
            elif markets[0].raw.get("venue"):
                venue_type = markets[0].raw["venue"]
        
        return VenueScanReport(
            venue_id=venue_id,
            venue_type=venue_type,
            total_discovered=start_total,
            after_cheap=len(after_cheap),
            after_liquidity=len(after_liquidity),
            candidates=len(candidates),
            tradeable=len(tradeable),
            top_opportunity=top_opp,
            best_near_miss=near_miss,
            exploration_candidates=exploration_opps,
            avg_edge=avg_edge,
            avg_score=avg_score,
            evaluated=evaluated,
            skipped_no_book=skipped_no_book,
            # everything the scan read but never iterated because of the cap
            beyond_cap=max(0, len(after_liquidity) - len(to_price)),
        ), all_opps

    def price_opportunity(self, opp: VenueOpportunity,
                          amount_usd: float = 3.0):
        """
        Price one opportunity and set its score to the capital-efficiency ratio.

        ONE score, set in one place: NET EV / (CAPITAL x TIME x EXECUTION RISK).

        This used to compute `net_ev_usd * ev_per_dollar * ev_per_risk` here
        while `rank_and_select` computed a different five-term product, so an
        opportunity's rank depended on which stage wrote to `score` last, and
        neither formula let capital-time be more than a nudge.

        Returns the EV result, or None when it could not be computed - an
        opportunity whose EV cannot be priced is not a candidate, and its score
        stays at zero rather than inheriting whatever it had.
        """
        orderbook = (opp.market.raw.get("orderbook", {})
                     if hasattr(opp.market, "raw")
                     and isinstance(opp.market.raw, dict) else {})
        try:
            ev = self.expected_ev_engine.calculate(opp, amount_usd, orderbook)
        except Exception as e:
            logger.debug(f"EV calc failed for {opp.market.id}: {e}")
            opp._expected_ev = None
            return None
        opp._expected_ev = ev
        opp.score = (ev.ev_per_capital_time_risk if ev.net_ev_usd > 0 else 0.0)
        if isinstance(opp.market.raw, dict):
            # Recorded beside the score so the console can show what the decision
            # was made on, not just what it came out as.
            opp.raw["capital_efficiency"] = {
                "net_ev_usd": round(ev.net_ev_usd, 4),
                "stressed_net_ev_usd": round(ev.stressed_net_ev_usd, 4),
                "capital_days_usd": round(ev.capital_days_usd, 4),
                "execution_risk": round(ev.execution_risk, 4),
                "ev_per_capital_time_risk": round(
                    ev.ev_per_capital_time_risk, 6),
                "costs_are_measured": ev.costs_are_measured,
            }
        return ev

    @staticmethod
    def capital_efficiency_key(opp: VenueOpportunity) -> float:
        """
        The ranking key: net EV per dollar-day of locked capital per unit of
        execution risk. Falls back to the opportunity's score, which for a
        priced opportunity IS this number.
        """
        ev = getattr(opp, "_expected_ev", None)
        if ev is not None:
            return float(getattr(ev, "ev_per_capital_time_risk", 0.0) or 0.0)
        return float(getattr(opp, "score", 0.0) or 0.0)

    @classmethod
    def rank_by_capital_efficiency(cls, opportunities):
        """Rank candidates on the economic question, not on headline profit."""
        priced = [o for o in opportunities if o is not None]
        return sorted(priced, key=cls.capital_efficiency_key, reverse=True)

    async def scan_all_venues(self, markets_by_venue: Dict[str, List[Market]], context_provider=None,
                              max_final_trades: int = 3,
                              book_lookup=None, fee_rate_lookup=None,
                              must_price: Optional[set] = None) -> MultiVenueScanResult:
        """
        Main V3 entry: scan all venues, evaluate all strategies, rank on common basis
        Returns report like:
        I scanned 1,200 opportunities.
        Polymarket: 3 candidates
        Kalshi: 5 candidates
        After fees/liquidity/uncertainty: 2 actually tradeable
        Best opportunity: Venue B
        """
        start = time.time()
        
        total_scanned = sum(len(m) for m in markets_by_venue.values())
        venue_reports: List[VenueScanReport] = []
        all_opportunities: List[VenueOpportunity] = []
        exploration_candidates: List[VenueOpportunity] = []
        strategy_breakdown: Dict[str, int] = {}
        
        # Scan each venue
        for venue_id, markets in markets_by_venue.items():
            try:
                report, opps = await self.scan_venue(
                    venue_id, markets, context_provider=context_provider,
                    book_lookup=book_lookup, must_price=must_price)
                venue_reports.append(report)
                all_opportunities.extend(opps)
                # ...and the paper candidates, which travel on the report: they
                # are what the shadow lane may act on and they are NOT
                # opportunities the live gates approved.
                exploration_candidates.extend(
                    getattr(report, "exploration_candidates", None) or [])
                
                # Strategy breakdown
                for opp in opps:
                    strat = opp.raw.get("strategy", "unknown") if hasattr(opp, 'raw') and isinstance(opp.raw, dict) else "unknown"
                    strategy_breakdown[strat] = strategy_breakdown.get(strat, 0) + 1
                    
            except Exception as e:
                logger.error(f"Venue {venue_id} scan failed: {e}")
                venue_reports.append(VenueScanReport(
                    venue_id=venue_id,
                    venue_type="unknown",
                    total_discovered=len(markets),
                    after_cheap=0,
                    after_liquidity=0,
                    candidates=0,
                    tradeable=0
                ))
        
        # Arbitrage across all venues
        all_markets_flat = [m for markets in markets_by_venue.values() for m in markets]
        arbitrage_opps = []
        try:
            # The books this cycle already read, and each venue's own fee
            # schedule: an arbitrage is only an arbitrage at executable prices.
            arbitrage_raw = self.arbitrage_engine.find_arbitrage(
                all_markets_flat[:500],  # Limit for performance
                book_lookup=book_lookup, fee_rate_lookup=fee_rate_lookup)
            arbitrage_opps = arbitrage_raw
            arb_venue_opps = self.arbitrage_engine.to_venue_opportunities(arbitrage_raw)
            all_opportunities.extend(arb_venue_opps)
            strategy_breakdown["arbitrage"] = len(arb_venue_opps)
        except Exception as e:
            logger.warning(f"Arbitrage scan failed: {e}")
        
        # V10 FIX #9: Common ranking using Expected Net EV - economically meaningful
        # Old: edge×prob×liquidity×execution×calibration×time/(fees+slippage+uncertainty+risk) - not $ profit
        # New: Σ prob×payoff − fees − spread − slippage − funding − gas − execution_loss − uncertainty_penalty
        # Then per dollar, per risk, per capital-time
        for opp in all_opportunities:
            self.price_opportunity(opp, amount_usd=3.0)
        
        # Rank on the capital-efficiency ratio: net EV per dollar-day of locked
        # capital per unit of execution risk. This used to sort on net_ev_usd
        # FIRST, which is how a thirty-day trade with a big headline profit
        # outranked an hourly one earning several times as much per day of
        # capital - the comparison the seventh report is about.
        if self.venue_registry:
            ranked = self.venue_registry.rank_opportunities(all_opportunities)
        else:
            ranked = list(all_opportunities)
        ranked = self.rank_by_capital_efficiency(ranked)
        
        # Filter to tradeable - now also requires net EV >0.
        #
        # This is the filter that actually decides `final_selected`, so the same
        # double-count here would have made the EV gate's fix decoration: a NO
        # trade with raw +0.150, net +$1.07 and 2.8% of edge surviving the costs
        # cleared the EV gate and was then dropped on this line. The 8% is the
        # hunt criterion and is measured on the raw mispricing; the net EV check
        # below is where the costs are charged.
        tradeable = []
        for o in ranked:
            if (hunted_mispricing(o) < HUNT_MISPRICING_MIN
                    or o.confidence < 0.6 or not o.should_trade):
                continue
            ev = getattr(o, '_expected_ev', None)
            if ev and ev.net_ev_usd <= 0:
                continue
            tradeable.append(o)
        
        final_selected = tradeable[:max_final_trades]
        
        # The fallback must not name a research-only finding (a two-leg pair, a
        # basket with no execution path) as the cycle's "best opportunity": the
        # operator's log showed an aborted pair advertised as the best trade of
        # the cycle, with edge 1.323 and a $1218.82 EV on a $1.00 stake.
        def _is_research_only(o) -> bool:
            raw = getattr(o, "raw", None)
            return bool(isinstance(raw, dict) and raw.get("research_only"))
        
        best_opp = final_selected[0] if final_selected else next(
            (o for o in ranked if not _is_research_only(o)), None)
        
        elapsed = time.time() - start
        
        # Build reasoning report
        venue_summary = "\n".join([
            f"{r.venue_id}: discovered {r.total_discovered}, candidates {r.candidates}, tradeable at the venue's own bar {r.tradeable}, "
            f"avg_edge {r.avg_edge:.3f}, top_score {r.top_opportunity.score:.3f} {r.top_opportunity.side} edge {r.top_opportunity.effective_edge:.3f} | {r.top_opportunity.market.question[:60]}" 
            if r.top_opportunity else f"{r.venue_id}: discovered {r.total_discovered}, candidates {r.candidates}, tradeable at the venue's own bar {r.tradeable}"
            for r in venue_reports
        ])
        
        strategy_summary = ", ".join([f"{k}: {v}" for k, v in strategy_breakdown.items()])
        
        # The None guard must cover every field, not just the first. This
        # previously read `best_opp.side if best_opp else ''` and then
        # dereferenced best_opp.effective_edge unconditionally, so the log line
        # raised AttributeError exactly when there was nothing to trade -
        # turning "no opportunities found" into a 500 on the endpoint.
        if best_opp is not None:
            _strategy = best_opp.raw.get("strategy") if hasattr(best_opp, "raw") else "unknown"
            best_line = (f"Best opportunity: {best_opp.venue_id} {best_opp.side} "
                         f"edge {best_opp.effective_edge:.3f} score {best_opp.score:.3f} "
                         f"strategy {_strategy} | {best_opp.market.question[:80]}")
        else:
            best_line = "Best opportunity: none - nothing cleared the gates | DO NOTHING"

        reasoning = (
            f"I scanned {total_scanned} opportunities across {len(markets_by_venue)} venues.\n"
            f"{venue_summary}\n"
            f"Strategies evaluated: {strategy_summary}\n"
            f"Arbitrage candidates: {len([a for a in arbitrage_opps if a.should_trade])} tradeable out of {len(arbitrage_opps)}\n"
            f"After fees/liquidity/uncertainty/risk: {len(tradeable)} actually tradeable opportunities (a venue's own count above is before these gates)\n"
            f"{best_line}\n"
            f"Final selected {len(final_selected)} trades (max {max_final_trades}) | Time {elapsed:.1f}s"
        )
        
        logger.info(reasoning)
        
        total_after_cheap = sum(r.after_cheap for r in venue_reports)
        total_after_liq = sum(r.after_liquidity for r in venue_reports)
        total_candidates = sum(r.candidates for r in venue_reports)
        total_tradeable = len(tradeable)
        
        return MultiVenueScanResult(
            total_scanned=total_scanned,
            total_after_cheap=total_after_cheap,
            total_after_liquidity=total_after_liq,
            total_candidates=total_candidates,
            total_tradeable=total_tradeable,
            venue_reports=venue_reports,
            all_opportunities=ranked,
            final_selected=final_selected,
            arbitrage_opportunities=arbitrage_opps,
            strategy_breakdown=strategy_breakdown,
            execution_time=elapsed,
            reasoning=reasoning,
            best_opportunity=best_opp,
            exploration_candidates=exploration_candidates,
            best_near_miss=_best_near_miss(venue_reports),
        )
