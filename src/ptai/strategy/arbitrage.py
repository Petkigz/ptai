"""
V52: a pair of similar questions is not an arbitrage.

The operator's 2026-09-29 log, eleven times:

    ARBITRAGE FOUND: Same event confidence 0.82: 'Will Sarah Knafo win the 2027
      French presidential election?' vs 'Who will win the next French
      presidential election?' | Price 0.009 vs 0.590 spread 0.581 | Cost 0.420
      profit 0.581 (138.4%) adjusted 112.0% | Trade: True

and then, forty seconds later, the same finding was taken as the best
opportunity of the cycle (edge 1.323, netEV $1218.82 on a $1.00 stake) and
pushed into the exploration lane, where execution aborted:

    ERROR ... Adapter for venue_id marketsource.polymarket not found
    ERROR ... ABORT: opportunity venue_id MarketSource.POLYMARKET+MarketSource.
      PREDICTIT_arb does not match adapter polymarket - exact routing required
    ERROR ... Arb lane could not build the legs of ...: 'ArbitrageOpportunity'
      object has no attribute 'fee_adjusted_profit'

Three defects, one chain: the profit was computed from mid prices on two
questions that are merely similar; the opportunity was handed to the
directional execution path under an id built out of enum reprs; and the arb
lane asked for a field that this dataclass never had.

This file now does what the rest of the system learned in V49c: an arbitrage
label requires the venue's own structure and executable prices.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType

# How much of the (1 - cost) payout the pair must keep, AFTER both venues' own
# fees, before it is called tradeable. The old engine called a pair tradeable on
# a 3% margin computed from mid prices with a flat 2% "fee estimate"; the
# operator's log then showed "adjusted 126.3%" printed as though it were money.
MIN_EXECUTABLE_PROFIT_PCT = 0.03


def _real_two_sided_book(book: Any) -> bool:
    """A book that can price a leg: real, validated, two-sided."""
    if not isinstance(book, dict):
        return False
    if not (book.get("is_real") and book.get("validated")):
        return False
    bid, ask = book.get("bid"), book.get("ask")
    return (isinstance(bid, (int, float)) and isinstance(ask, (int, float))
            and 0.0 < bid < ask < 1.0)


def _venue_id(market: Market) -> str:
    """
    The id the venue registry routes on.

    `str(market.source)` is 'MarketSource.POLYMARKET': a repr, not an adapter
    id. That is how a pair opportunity reached execution under the venue id
    'marketsource.polymarket' and was aborted by exact routing.
    """
    raw = getattr(market, "raw", None) or {}
    if isinstance(raw, dict) and raw.get("venue"):
        return str(raw["venue"])
    source = getattr(market, "source", None)
    return str(getattr(source, "value", source) or "unknown")


@dataclass
class ArbitrageOpportunity:
    market_a: Market
    market_b: Market
    venue_a: str
    venue_b: str
    price_a: float
    price_b: float
    spread: float  # price_b - price_a
    estimated_profit_pct: float
    confidence_same_event: float  # 0-1 how confident same event
    reasoning: str
    should_trade: bool
    side_a: str
    side_b: str
    # V52 - what the claim rests on.
    identity_verified: bool = False      # the two questions ARE the same market
    executable: bool = False             # both legs priced off validated books
    blocked_reason: str = ""             # why not, when not
    cost: Optional[float] = None         # executable cost per $1 payout
    indicative_profit_pct: float = 0.0   # mid-price figure: never a trade
    fee_rate_a: Optional[float] = None
    fee_rate_b: Optional[float] = None
    executable_price_a: Optional[float] = None
    executable_price_b: Optional[float] = None

    @property
    def fee_adjusted_profit(self) -> float:
        """
        The arb lane's executor read this attribute; it did not exist, so the
        lane raised AttributeError before it could place a single leg.
        """
        return self.estimated_profit_pct

    @property
    def payout_usd(self) -> float:
        return 1.0


class ArbitrageEngine:
    """
    Detects the SAME market quoted differently on two venues.

    Example: Polymarket Trump 0.61, Kalshi Trump 0.68 = 7% spread, buy low sell
    high IF the two venues resolve the same question identically - which is the
    part that has to be shown, not assumed.
    """

    def __init__(self, min_spread: float = 0.03, min_confidence_same_event: float = 0.7):
        self.min_spread = min_spread
        self.min_confidence = min_confidence_same_event

    def _normalize_question(self, q: str) -> str:
        """Normalize question for comparison"""
        q = q.lower()
        q = re.sub(r'[^a-z0-9 ]', ' ', q)
        q = re.sub(r'\s+', ' ', q).strip()
        # Remove dates, filler words
        q = re.sub(r'\b(will|the|a|an|in|on|by|at|for|2026|2027|2025)\b', '', q)
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def _same_event_score(self, m1: Market, m2: Market) -> float:
        """Score 0-1 how likely same event"""
        # Different venues - don't compare same venue
        if m1.source == m2.source and m1.id == m2.id:
            return 0.0

        # Normalize questions
        q1_norm = self._normalize_question(m1.question)
        q2_norm = self._normalize_question(m2.question)

        if not q1_norm or not q2_norm:
            return 0.0

        # Sequence matcher similarity
        similarity = SequenceMatcher(None, q1_norm, q2_norm).ratio()

        # Bonus if same keywords: Trump, election, etc.
        keywords1 = set(q1_norm.split())
        keywords2 = set(q2_norm.split())
        if len(keywords1) > 0 and len(keywords2) > 0:
            jaccard = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            similarity = max(similarity, jaccard)

        # Check end dates close (within 7 days)
        if m1.end_date and m2.end_date:
            diff_days = abs((m1.end_date - m2.end_date).total_seconds()) / 86400
            if diff_days > 7:
                similarity *= 0.5
            elif diff_days < 1:
                similarity = min(1.0, similarity * 1.1)

        return similarity

    def _same_market_identity(self, m1: Market, m2: Market) -> bool:
        """
        Are these two questions THE SAME MARKET?

        A 0.82 string similarity is not evidence and never was: 'Will Sarah
        Knafo win the 2027 French presidential election?' and 'Who will win the
        next French presidential election?' are different questions with
        different resolutions, and the price gap between them is not a gap in
        the same thing. Only identical normalized wording - and agreeing
        end-dates when both venues carry one - is treated as the same market.
        Anything else is a candidate to research, not an arbitrage to trade.
        """
        if self._normalize_question(m1.question) != self._normalize_question(m2.question):
            return False
        if m1.end_date and m2.end_date:
            diff_days = abs((m1.end_date - m2.end_date).total_seconds()) / 86400
            if diff_days > 7:
                return False
        return True

    def find_arbitrage(self, markets: List[Market],
                       book_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
                       fee_rate_lookup: Optional[Callable[[Market], Any]] = None,
                       ) -> List[ArbitrageOpportunity]:
        """
        Find cross-venue price discrepancies. A pair becomes `should_trade`
        only when its identity is verified AND both legs priced off validated,
        executable books, with each venue's own fee charged.
        """
        opportunities: List[ArbitrageOpportunity] = []

        for i, m1 in enumerate(markets):
            for j in range(i + 1, len(markets)):
                m2 = markets[j]
                if m1.source == m2.source:
                    continue  # Same venue, not arbitrage (could be internal arb but skip)

                same_event_conf = self._same_event_score(m1, m2)
                if same_event_conf < self.min_confidence:
                    continue

                price_a = float(m1.best_price)
                price_b = float(m2.best_price)
                spread = abs(price_b - price_a)
                if spread < self.min_spread:
                    continue

                # The indicative figure, from the quoted prices. Kept for
                # research and NEVER used as a trade's economics.
                low_price = min(price_a, price_b)
                high_price = max(price_a, price_b)
                indicative_cost = low_price + (1 - high_price)
                if indicative_cost <= 0:
                    continue
                indicative_profit = (1.0 - indicative_cost) / indicative_cost

                venue_a = _venue_id(m1)
                venue_b = _venue_id(m2)

                side_a = "YES" if price_a < price_b else "NO"
                side_b = "NO" if side_a == "YES" else "YES"

                identity = self._same_market_identity(m1, m2)
                book_a = self._book_for(m1, book_lookup)
                book_b = self._book_for(m2, book_lookup)
                # The MARKET is what the venue's fee is read for: the same
                # convention the combinatorial engine uses, because the fee
                # comes from that market's own CLOB info, not from a name.
                fee_a = self._fee_for(m1, fee_rate_lookup)
                fee_b = self._fee_for(m2, fee_rate_lookup)

                executable = False
                cost: Optional[float] = None
                profit_pct = 0.0
                exec_price_a = exec_price_b = None
                blocked = ""

                if not identity:
                    blocked = (f"the two questions are similar, not the same market "
                               f"({same_event_conf:.2f}): '{m1.question[:60]}' vs "
                               f"'{m2.question[:60]}' - research only, no arbitrage")
                elif not _real_two_sided_book(book_a):
                    blocked = (f"no validated executable book on {venue_a} "
                               f"(price {price_a:.3f} is a quote, not a cost)")
                elif not _real_two_sided_book(book_b):
                    blocked = (f"no validated executable book on {venue_b} "
                               f"(price {price_b:.3f} is a quote, not a cost)")
                elif fee_a is None or fee_b is None:
                    blocked = (f"the venue's own fee could not be read "
                               f"({venue_a}: {fee_a}, {venue_b}: {fee_b}) - a profit "
                               f"computed on an assumed fee is not a profit")
                else:
                    # What a share actually COSTS on each leg: the cheap side is
                    # bought at its ask, the other side is bought as the NO token
                    # at (1 - its bid).
                    if price_a < price_b:
                        exec_price_a = float(book_a["ask"])
                        exec_price_b = 1.0 - float(book_b["bid"])
                    else:
                        exec_price_a = 1.0 - float(book_a["bid"])
                        exec_price_b = float(book_b["ask"])
                    cost = (exec_price_a * (1.0 + fee_a)
                            + exec_price_b * (1.0 + fee_b))
                    if cost >= 1.0:
                        blocked = (f"executable cost {cost:.4f} per $1 payout leaves "
                                   f"nothing: pay {exec_price_a:.4f} on {venue_a} + "
                                   f"{exec_price_b:.4f} on {venue_b} with their own "
                                   f"fees {fee_a*100:.2f}%/{fee_b*100:.2f}%")
                    else:
                        profit_pct = (1.0 - cost) / cost
                        executable = profit_pct >= MIN_EXECUTABLE_PROFIT_PCT
                        if not executable:
                            blocked = (f"executable cost {cost:.4f} leaves "
                                       f"{profit_pct*100:.2f}%, below the "
                                       f"{MIN_EXECUTABLE_PROFIT_PCT*100:.0f}% floor")

                should_trade = executable and identity

                if should_trade:
                    reasoning = (
                        f"SAME MARKET, EXECUTABLE: confidence {same_event_conf:.2f}: "
                        f"'{m1.question[:60]}' | buy {side_a} at {exec_price_a:.4f} on "
                        f"{venue_a} + {side_b} at {exec_price_b:.4f} on {venue_b} | "
                        f"cost {cost:.4f} per $1 payout, profit "
                        f"{1.0 - cost:.4f} ({profit_pct*100:.1f}% after both venues' own "
                        f"fees) | quoted {price_a:.3f} vs {price_b:.3f} | Trade: True")
                else:
                    reasoning = (
                        f"PAIR CANDIDATE (not tradeable): confidence "
                        f"{same_event_conf:.2f}: '{m1.question[:60]}' vs "
                        f"'{m2.question[:60]}' | quoted {price_a:.3f} vs {price_b:.3f} "
                        f"(indicative {indicative_profit*100:.1f}% on mid prices) | "
                        f"REFUSED: {blocked} | Trade: False")

                opp = ArbitrageOpportunity(
                    market_a=m1,
                    market_b=m2,
                    venue_a=venue_a,
                    venue_b=venue_b,
                    price_a=price_a,
                    price_b=price_b,
                    spread=spread,
                    # Only an executable, fee-charged number may carry the name
                    # that the ranking and the reports sort on.
                    estimated_profit_pct=profit_pct if should_trade else 0.0,
                    confidence_same_event=same_event_conf,
                    reasoning=reasoning,
                    should_trade=should_trade,
                    side_a=side_a,
                    side_b=side_b,
                    identity_verified=identity,
                    executable=executable,
                    blocked_reason=blocked,
                    cost=cost,
                    indicative_profit_pct=indicative_profit,
                    fee_rate_a=fee_a,
                    fee_rate_b=fee_b,
                    executable_price_a=exec_price_a,
                    executable_price_b=exec_price_b,
                )
                opportunities.append(opp)

                if should_trade:
                    # This line is the operator's bug report every time it is
                    # wrong, so it may only print what was checked.
                    logger.info(f"ARBITRAGE FOUND: {reasoning}")
                else:
                    logger.debug(f"Pair candidate refused: {reasoning}")

        opportunities.sort(key=lambda x: x.estimated_profit_pct, reverse=True)
        tradeable = [o for o in opportunities if o.should_trade]
        logger.info(
            f"Arbitrage scan: {len(markets)} markets -> {len(opportunities)} "
            f"candidate(s), {len(tradeable)} tradeable at executable prices with "
            f"verified identity"
            + (f" (best indicative {max(o.indicative_profit_pct for o in opportunities)*100:.1f}% "
               f"on mid prices - not a trade)" if opportunities else ""))
        return opportunities

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _book_for(market: Market,
                  book_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]]
                  ) -> Optional[Dict[str, Any]]:
        if book_lookup is not None:
            try:
                book = book_lookup(market.id)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Book lookup failed for {market.id}: {e}")
                book = None
            if book:
                return book
        raw = getattr(market, "raw", None) or {}
        if isinstance(raw, dict) and isinstance(raw.get("orderbook"), dict):
            return raw["orderbook"]
        return None

    @staticmethod
    def _fee_for(market: Market,
                 fee_rate_lookup: Optional[Callable[[Market], Any]]
                 ) -> Optional[float]:
        """
        The venue's own taker rate for this market, or None when it is not read.

        `fee_rate_lookup` may answer with the rate, or with the
        {"rate": .., "source": ..} record the agent produces - and None means
        "not read", which refuses the pair rather than charging an invented 2%.
        """
        if fee_rate_lookup is None:
            return None
        try:
            got = fee_rate_lookup(market)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Fee lookup failed for {market.id}: {e}")
            return None
        if isinstance(got, dict):
            got = got.get("rate")
        if got is None:
            return None
        try:
            return float(got)
        except (TypeError, ValueError):
            logger.debug(f"Fee rate for {market.id} is not a number: {got!r}")
            return None

    def to_venue_opportunities(self, arb_opps: List[ArbitrageOpportunity]) -> List[VenueOpportunity]:
        """
        Convert arbitrage opps to VenueOpportunity for unified RANKING.

        A pair is not a directional order: it is two legs on two venues, and
        the directional execution path can only ever send one of them. It is
        therefore flagged research-only here and `should_trade=False`, so it is
        counted in the report (and is available to the arbitrage lane, which
        knows how to build both legs) but can never be selected as a trade to
        send to a single adapter.
        """
        venue_opps = []
        for arb in arb_opps:
            if not arb.should_trade:
                continue
            opp = VenueOpportunity(
                market=arb.market_a,
                venue_id=f"{arb.venue_a}+{arb.venue_b}",
                venue_type=VenueType.PREDICTION,
                side=arb.side_a,
                market_price=arb.executable_price_a if arb.executable_price_a else arb.price_a,
                estimated_fair=arb.price_b,  # Fair is other venue price
                raw_edge=arb.spread,
                effective_edge=arb.estimated_profit_pct,
                confidence=arb.confidence_same_event,
                uncertainty=1 - arb.confidence_same_event,
                liquidity_score=min(1.0, min(arb.market_a.liquidity, arb.market_b.liquidity) / 10000),
                execution_quality=0.6,  # Arbitrage execution harder
                category="arbitrage",
                sources=[f"arb_{arb.venue_a}_{arb.venue_b}"],
                reasoning=arb.reasoning,
                bull_case=f"Buy {arb.side_a} at {arb.executable_price_a} on {arb.venue_a}",
                bear_case=f"Buy {arb.side_b} at {arb.executable_price_b} on {arb.venue_b}",
                resolution_risks=[f"Resolution mismatch risk conf {arb.confidence_same_event:.2f}",
                                  "Execution timing risk"],
                should_trade=False,
                raw={
                    "strategy": "arbitrage",
                    "research_only": True,
                    "pair": True,
                    "legs": [
                        {"venue_id": arb.venue_a, "market_id": arb.market_a.id,
                         "side": arb.side_a, "price": arb.executable_price_a},
                        {"venue_id": arb.venue_b, "market_id": arb.market_b.id,
                         "side": arb.side_b, "price": arb.executable_price_b},
                    ],
                    "executed_by": "arbitrage lane (both legs) - never the single-venue path",
                },
            )
            opp.calculate_common_score()
            venue_opps.append(opp)

        return venue_opps
