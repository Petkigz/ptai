"""
Combinatorial Arbitrage - for markets the VENUE says are mutually exclusive.

The arithmetic here is only true under one premise: exactly one of the group's
markets settles YES. Under that premise, sum(prices) < 1 means buy every YES for
less than the $1 one of them must pay, and sum(prices) > 1 means buy every NO
(the basket pays n-1).

The premise is the whole trade. This engine used to ASSUME it - `is_exclusive =
True  # Assume exclusive if same event_slug`, and `is_exhaustive` true whenever a
group held three markets - and then read the sum off `best_price`. The
operator's 2026-09-29 log shows what that produced:

    MECE group elon-musk-of-tweets-...: 10 markets sum YES 0.003 | buy_all_yes
      cost $0.003 payout $1.000 profit $0.997 (33233.3%) | Should trade True
    MECE group bitcoin-above-on-september-29-2026: 10 markets sum YES 4.498 |
      sell_all_yes_buy_all_no ... profit $3.498 (63.6%) | Should trade True

Neither is an arbitrage. The first is a field of candidates the 200-market scan
only partly captured (the sum is missing 99.7% of the field, so of course it is
under 1); the second is ten Bitcoin price strikes, which are not mutually
exclusive at all - several of them win together, so "exactly one YES wins" is
false and the $3.50 is invented. Both were labelled "Exclusive True" because a
slug grouped them, and both were called "Should trade True" because 33,233% and
63.6% both clear a 2% bar.

What changed: a basket is an arbitrage only when the evidence supports it.

  1. EXCLUSIVITY comes from the venue, not from a shared slug. Polymarket marks
     a mutually exclusive basket with `negRisk` on the event (and on each of its
     markets). No mark, no trade: a missing flag is answered as "not verified",
     never as an assumption.
  2. COMPLETENESS - every outcome of the event must be in the basket. The venue
     lists the event's markets; if the scan read 10 of 34, the sum is meaningless
     and the basket is refused with both numbers named.
  3. THE SUM must be inside the band a real basket could sit in. A sum of 0.003
     or 4.498 is evidence of a broken basket, not of free money.
  4. THE PRICE must be executable. Cost is the ASK on every YES leg and
     (1 - best bid) on every NO leg, from a book that validated as real. A
     basket priced off `best_price` is a model, not a trade.
  5. THE FEE must come from the venue. `calculated as 2% x legs x 0.5` was an
     invented constant; it is now the venue's own declared taker rate, and if it
     cannot be read the net is not claimed.

A second honesty fix lives here too: this engine does not place orders, and the
loop does not execute its output. `wired_to_execution` is False on every group so
no reader - log, console or report - can take "Should trade" for "bought".

Example:
- Market group: Who wins election? Trump 0.45, Biden 0.30, Other 0.20 = sum 0.95 <1 => buy all YES for $0.95, guaranteed $1.00 profit $0.05 = 5.26%
- If sum 1.08 >1 => buy all NO, cost (1-0.45)+(1-0.30)+(1-0.20)=0.55+0.70+0.80=2.05 for $2.00 payout? Actually need to think: buying NO is buying opposite.
  For MECE, exactly one YES wins. If sum >1, you can sell YES or buy NO baskets.

Negative Risk: Polymarket lets you convert NO shares across mutually exclusive markets
- If you hold NO on Trump, NO on Biden, NO on Other in same event, you can convert to USDC because at least 2 NOs must win (only 1 YES can win)
- This frees capital and creates synthetic positions

Implementation uses linear programming concept but simplified for PTAI - no need full LP solver for $50, heuristic grouping
"""
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import re
from difflib import SequenceMatcher

from ..markets.base import Market


@dataclass
class CombinatorialGroup:
    group_id: str
    event_slug: str
    markets: List[Market]
    sum_yes: float
    is_exhaustive: bool  # MECE - mutually exclusive collectively exhaustive
    is_exclusive: bool
    arbitrage_type: str  # "buy_all_yes" if sum<1, "sell_all_yes" if sum>1, "none"
    estimated_profit_pct: float
    cost: float
    payout: float
    should_trade: bool
    reasoning: str
    # --- what the evidence actually supports (see the module docstring) -----
    verified: bool = False               # structure + executable prices agree
    verdict_reason: str = ""             # the one sentence behind `verified`
    verification: Dict[str, Any] = field(default_factory=dict)
    blocked_reason: Optional[str] = None  # why it is NOT an arbitrage, in words
    cost_basis: str = "indicative_best_price"   # or "executable_asks"
    net_profit_usd: Optional[float] = None      # None when fees were not read
    fee_rate: Optional[float] = None
    fee_source: Optional[str] = None
    outcomes_declared: Optional[int] = None     # the venue's own outcome count
    outcomes_seen: int = 0
    executability: Dict[str, Any] = field(default_factory=dict)
    wired_to_execution: bool = False            # this engine places NO orders


@dataclass
class NegativeRiskConversion:
    group_id: str
    markets: List[Market]
    no_shares: Dict[str, float]  # market_id -> NO shares
    convertible_value: float
    freed_capital: float
    reasoning: str


class CombinatorialArbitrageEngine:
    """
    Finds combinatorial arbitrage in mutually exclusive markets
    Highest risk-adjusted return for $50 bankroll per user's Top 5
    """
    def __init__(self, min_profit_pct: float = 0.02, max_group_size: int = 10):
        self.min_profit_pct = min_profit_pct
        self.max_group_size = max_group_size

    def _group_by_event_slug(self, markets: List[Market]) -> Dict[str, List[Market]]:
        """Group markets by event_slug - natural MECE groups"""
        groups: Dict[str, List[Market]] = {}
        for m in markets:
            slug = m.event_slug or "unknown"
            if slug not in groups:
                groups[slug] = []
            groups[slug].append(m)
        # Only keep groups with 2+ markets (potential MECE)
        return {k: v for k, v in groups.items() if len(v) >= 2}

    def _detect_exclusive_by_question(self, markets: List[Market]) -> List[List[Market]]:
        """
        CANDIDATE groups from question similarity - not verified baskets.

        A 70%-similar question and a shared word bag is a hint that two markets
        belong to one contest; it is not evidence that exactly one of them wins.
        "Will Bitcoin be above $90k" and "... above $100k" are 90% similar and
        NOT exclusive - both can win. Groups found here are therefore handed to
        the same verification as everything else (`_mece_evidence`), which needs
        the venue's own `negRisk` mark and its complete outcome list before the
        basket is called an arbitrage.
        """
        # Patterns: "Will X win?" with different candidates, "Who will win?" etc.
        # For PTAI, use event_slug as primary, but also detect by similar question prefix
        groups: List[List[Market]] = []
        used = set()
        
        for i, m1 in enumerate(markets):
            if m1.id in used:
                continue
            # Find markets with similar question prefix
            prefix = m1.question[:30].lower()
            group = [m1]
            for j, m2 in enumerate(markets[i+1:], i+1):
                if m2.id in used:
                    continue
                # Check if questions share prefix and different outcomes
                if m1.event_slug and m2.event_slug and m1.event_slug == m2.event_slug:
                    group.append(m2)
                    used.add(m2.id)
                elif SequenceMatcher(None, m1.question[:50].lower(), m2.question[:50].lower()).ratio() > 0.7:
                    # Similar questions, likely same event different outcomes
                    # Check if outcomes are different candidates
                    q1_words = set(m1.question.lower().split())
                    q2_words = set(m2.question.lower().split())
                    # If they share many words but differ in candidate name, likely exclusive
                    if len(q1_words & q2_words) / len(q1_words | q2_words) > 0.5:
                        group.append(m2)
                        used.add(m2.id)
            
            if len(group) >= 2:
                groups.append(group)
                used.add(m1.id)
        
        return groups

    # ------------------------------------------------------------------
    # is this group actually a mutually exclusive, complete basket?
    # ------------------------------------------------------------------

    # A real MECE basket is quoted near 1: the outcomes divide one probability
    # space. This band is loose on purpose - a genuine dislocation can be large -
    # and it is a SANITY check, not the evidence: 0.003 and 4.498 are refused
    # because no complete set of exclusive outcomes looks like that, whatever the
    # slug says.
    SUM_BAND = (0.75, 1.25)

    def _mece_evidence(self, group_markets: List[Market],
                       neg_risk_lookup: Optional[Callable[[Market], Any]] = None,
                       ) -> Dict[str, Any]:
        """
        What the VENUE says about this group, read from the market payloads.

        Two sources are accepted, and both are the VENUE talking rather than this
        module inferring:

          * the `negRisk` mark in the market payload (Gamma sets it on the event
            and on each of its markets), and
          * `neg_risk_lookup(market)`, which is the venue's own answer from
            `getClobMarketInfo` (`MarketMechanics.neg_risk`, read through the
            SDK's `get_neg_risk`) - the fact the order path itself relies on when
            it signs a neg-risk order.

        Everything here is read, nothing is assumed: a market no source can
        answer for makes the group unverified, and the reason says so.
        """
        exclusive_flags: List[Optional[bool]] = []
        flag_sources: List[Optional[str]] = []
        conflicts: List[str] = []
        declared: Optional[int] = None
        declared_from = None

        for m in group_markets:
            raw = m.raw if isinstance(m.raw, dict) else {}
            event = raw.get("event") if isinstance(raw.get("event"), dict) else {}
            market = raw.get("market") if isinstance(raw.get("market"), dict) else {}
            flag = None
            for source in (event, market):
                for key in ("negRisk", "neg_risk", "enableNegRisk"):
                    if key in source:
                        flag = bool(source.get(key))
                        break
                if flag is not None:
                    break
            payload_flag = flag
            source_used = "payload negRisk mark" if payload_flag is not None else None
            # The venue's own market info is asked as well, not only when the
            # payload is silent: two venue sources that DISAGREE mean the data is
            # stale somewhere, and a stale answer must not decide a trade.
            venue_flag = None
            if neg_risk_lookup is not None:
                try:
                    answer = neg_risk_lookup(m)
                except Exception as e:  # noqa: BLE001 - unreadable is "no answer"
                    logger.debug(f"neg-risk answer for {m.id} not readable: "
                                 f"{type(e).__name__}: {e}")
                    answer = None
                if isinstance(answer, dict):
                    answer = answer.get("neg_risk")
                if answer is not None:
                    venue_flag = bool(answer)
            if payload_flag is not None and venue_flag is not None:
                if payload_flag != venue_flag:
                    conflicts.append(m.id)
                    flag = None
                    source_used = None
                else:
                    source_used = "payload negRisk mark"
            elif payload_flag is not None:
                flag = payload_flag
            elif venue_flag is not None:
                flag = venue_flag
                source_used = "the venue's own clob market info"
            exclusive_flags.append(flag)
            flag_sources.append(source_used)

            if declared is None:
                listed = event.get("markets")
                if isinstance(listed, list) and listed:
                    declared = len(listed)
                    declared_from = "the event's own market list"
                else:
                    for key in ("marketCount", "marketsCount", "outcomesCount"):
                        if isinstance(event.get(key), int):
                            declared = int(event[key])
                            declared_from = f"the event's {key}"
                            break

        any_marked = any(f is not None for f in exclusive_flags)
        all_marked_exclusive = bool(exclusive_flags) and all(f is True for f in exclusive_flags)
        if conflicts:
            why = (f"the venue's payload and its own market info disagree about "
                   f"whether this is a mutually exclusive basket "
                   f"({', '.join(conflicts[:3])}"
                   f"{', ...' if len(conflicts) > 3 else ''}), so neither answer "
                   f"can be relied on")
        elif not any_marked:
            why = ("neither the venue's payload nor its own market info says this "
                   "is a mutually exclusive basket, so 'exactly one of them wins' "
                   "is an assumption and not a fact")
        elif not all_marked_exclusive:
            why = ("the venue does not mark every market in this group as part of "
                   "the same mutually exclusive basket")
        else:
            why = None

        sources = {s for s in flag_sources if s}
        return {
            "exclusive": all_marked_exclusive,
            "exclusive_flags": exclusive_flags,
            "exclusive_sources": sorted(sources),
            "exclusive_source": (", ".join(sorted(sources)) if sources else None),
            "exclusive_why": why,
            "exclusive_conflicts": conflicts,
            "declared_outcomes": declared,
            "declared_source": declared_from,
        }

    def _price_basket(self, group_markets: List[Market], arb_type: str,
                      book_lookup: Optional[Callable[[str], Optional[Dict]]],
                      ) -> Dict[str, Any]:
        """
        What the basket costs if it is bought RIGHT NOW, leg by leg.

        A YES leg costs the ask. A NO leg costs 1 - best bid (a NO share is the
        other side of the same book). Both need a book that validated as this
        market's and was not stale or crossed - an estimate must never price an
        arbitrage, which is the same rule the order path follows.
        """
        legs: List[Dict[str, Any]] = []
        missing: List[str] = []
        total = 0.0
        for m in group_markets:
            book = (book_lookup(m.id) if book_lookup else None) or {}
            usable = bool(book.get("is_real")) and bool(book.get("validated"))
            if not usable:
                missing.append(m.id)
                legs.append({"market_id": m.id, "priced": False,
                             "why": (book.get("warning")
                                     or "no validated real book for this leg")})
                continue
            if arb_type == "buy_all_yes":
                price = book.get("executable_price_yes") or book.get("best_ask")
            else:
                price = book.get("executable_price_no")
                if price is None and book.get("best_bid") is not None:
                    price = round(1.0 - float(book["best_bid"]), 4)
            if price is None or not (0.0 < float(price) < 1.0):
                missing.append(m.id)
                legs.append({"market_id": m.id, "priced": False,
                             "why": "the book carried no usable price for this side"})
                continue
            price = float(price)
            total += price
            legs.append({
                "market_id": m.id, "priced": True, "side": ("YES" if arb_type == "buy_all_yes" else "NO"),
                "price": round(price, 4),
                # Total: a book that carried no depth is 0 depth, not a crash -
                # the same rule the pool reader follows. A reader that raises on
                # a thin payload is how a measured book became an estimate.
                "depth_usd": round(float((book.get("ask_size") if arb_type == "buy_all_yes"
                                          else book.get("bid_size")) or 0.0), 2),
                "spread": book.get("spread"),
                "source": book.get("source") or "unknown",
            })
        return {"legs": legs, "missing": missing, "cost": round(total, 6),
                "complete": not missing}

    def _evaluate_group(self, group_markets: List[Market], *,
                        book_lookup: Optional[Callable[[str], Optional[Dict]]],
                        fee_rate_lookup: Optional[Callable[[Market], Any]],
                        neg_risk_lookup: Optional[Callable[[Market], Any]] = None,
                        min_profit_pct: float = 0.02,
                        min_sum_gap: float = 0.03,
                        ) -> Dict[str, Any]:
        """
        One group, evaluated honestly, cheapest checks first.

        The structural checks (venue mark, complete outcome list, size, sum band)
        cost nothing and run before anything touches a book or a fee rate, so a
        cycle with 200 markets and dozens of lookalike slugs does not pay for
        groups that were never baskets.
        """
        n = len(group_markets)
        sum_yes = sum(float(m.best_price) for m in group_markets)
        evidence = self._mece_evidence(group_markets,
                                       neg_risk_lookup=neg_risk_lookup)
        verdict: Dict[str, Any] = {"sum_yes": round(sum_yes, 6), **evidence}

        if sum_yes < 1.0:
            arb_type, payout = "buy_all_yes", 1.0
            indicative_cost = sum_yes
        elif sum_yes > 1.0:
            # Buying all n NOs pays n-1 when exactly one YES wins.
            arb_type, payout = "sell_all_yes_buy_all_no", float(n - 1)
            indicative_cost = n - sum_yes
        else:
            arb_type, payout, indicative_cost = "none", 1.0, sum_yes

        lo, hi = self.SUM_BAND
        in_band = lo <= sum_yes <= hi
        declared = evidence["declared_outcomes"]
        verdict["sum_in_band"] = in_band

        empty = {"legs": [], "missing": [m.id for m in group_markets], "cost": 0.0,
                 "complete": False}

        def _result(verified: bool, blocked: Optional[str], *,
                    cost: float = indicative_cost,
                    cost_basis: str = "indicative_best_price",
                    net_usd: Optional[float] = None,
                    net_pct: Optional[float] = None,
                    pricing: Optional[Dict[str, Any]] = None,
                    fee_rate: Optional[float] = None,
                    fee_source: Optional[str] = None,
                    fees_known: bool = False) -> Dict[str, Any]:
            gross = round(payout - cost, 6)
            if blocked:
                reason = blocked
            elif verified and net_usd is not None and net_pct is not None:
                reason = (f"verified MECE basket ({evidence['exclusive_source']}; all "
                          f"{n} outcome(s) present): {arb_type.replace('_', ' ')} at "
                          f"executable prices, net ${net_usd:.4f} "
                          f"({net_pct * 100:.2f}%) after the venue's fee "
                          f"({fee_source or 'venue rate'})")
            else:
                reason = "no arbitrage in this basket"
            return {
                "arb_type": arb_type, "payout": payout, "cost": cost,
                "cost_basis": cost_basis, "sum_yes": sum_yes,
                "indicative_cost": round(indicative_cost, 6),
                "estimated_profit_pct": (float(net_pct) if net_pct is not None else 0.0),
                "net_profit_usd": net_usd, "gross_profit_usd": gross,
                "fee_rate": fee_rate, "fee_source": fee_source,
                "verified": verified, "blocked_reason": blocked,
                "should_trade": bool(verified and fees_known and net_usd is not None
                                     and net_usd > 0 and net_pct is not None
                                     and net_pct > min_profit_pct
                                     and abs(sum_yes - 1.0) > min_sum_gap),
                "reason": reason, "verification": verdict,
                "executability": pricing or empty,
                "outcomes_declared": declared, "outcomes_seen": n,
                "is_exclusive": bool(evidence["exclusive"]),
                "is_exhaustive": bool(evidence["exclusive"] and declared == n),
            }

        # --- 1. the venue's own mark for a mutually exclusive basket ---------
        if not evidence["exclusive"]:
            return _result(False, evidence["exclusive_why"])
        # --- 2. the event's complete outcome list ---------------------------
        if declared is None:
            return _result(False, (
                "the venue did not list how many outcomes this event has, so a "
                "missing leg cannot be ruled out and the sum means nothing"))
        if int(declared) != n:
            return _result(False, (
                f"the scan read {n} of the event's {declared} outcome(s) - a basket "
                f"with a leg missing cannot be arbitraged, and its sum "
                f"({sum_yes:.3f}) is missing the rest of the field"))
        # --- 3. a basket this engine will even consider ---------------------
        if n > self.max_group_size:
            return _result(False, (
                f"{n} legs is more than the {self.max_group_size}-leg maximum this "
                f"engine will consider, and more than a $50 bankroll can post"))
        # --- 4. the sum band -------------------------------------------------
        if not in_band:
            return _result(False, (
                f"sum YES {sum_yes:.3f} is outside the {lo}-{hi} band a complete "
                f"basket of exclusive outcomes can be quoted at - at least one of "
                f"'exclusive' and 'complete' is false, whatever the slug says"))

        # --- 5. executable prices, leg by leg --------------------------------
        pricing = self._price_basket(group_markets, arb_type, book_lookup)
        if not pricing["complete"]:
            return _result(False, (
                f"{len(pricing['missing'])} of {n} leg(s) have no validated real "
                f"book ({', '.join(pricing['missing'][:3])}"
                f"{', ...' if len(pricing['missing']) > 3 else ''}), so the basket "
                f"cannot be priced at executable prices"), pricing=pricing)
        cost = pricing["cost"]
        gross = payout - cost

        # --- 6. the venue's own fee, per leg ---------------------------------
        fee_rates: List[Optional[float]] = []
        fee_source = None
        for m in group_markets:
            rate = None
            if fee_rate_lookup is not None:
                try:
                    got = fee_rate_lookup(m)
                except Exception as e:  # noqa: BLE001 - unreadable is "not read"
                    logger.debug(f"Fee rate for {m.id} not readable: "
                                 f"{type(e).__name__}: {e}")
                    got = None
                if isinstance(got, dict):
                    rate = got.get("rate")
                    fee_source = fee_source or got.get("source")
                elif isinstance(got, (int, float)):
                    rate = float(got)
            fee_rates.append(float(rate) if rate is not None else None)
        fees_known = bool(fee_rates) and all(r is not None for r in fee_rates)
        verdict["fees_known"] = fees_known
        fee_rate = max(fee_rates) if fees_known else None
        if not fees_known:
            return _result(True, (
                "the venue's own fee rate could not be read for every leg, so the "
                "net of fees is not claimed (the structure is verified; the net is "
                "not)"), cost=cost, cost_basis="executable_asks", pricing=pricing,
                fees_known=False)
        fee_usd = cost * float(fee_rate)
        net_usd = round(gross - fee_usd, 6)
        net_pct = (net_usd / cost) if cost > 0 else None

        if net_usd <= 0:
            return _result(True, (
                f"verified MECE basket, but there is no profit at executable "
                f"prices: cost ${cost:.4f} for ${payout:.4f} payout is "
                f"${net_usd:.4f} net of fees"), cost=cost,
                cost_basis="executable_asks", net_usd=net_usd, net_pct=net_pct,
                pricing=pricing, fee_rate=fee_rate, fee_source=fee_source,
                fees_known=True)
        if abs(sum_yes - 1.0) <= min_sum_gap:
            return _result(True, (
                f"the basket is priced at {sum_yes:.4f}: the gap is inside the "
                f"{min_sum_gap * 100:.1f}% the engine needs to cover fees and "
                f"slippage"), cost=cost, cost_basis="executable_asks",
                net_usd=net_usd, net_pct=net_pct, pricing=pricing,
                fee_rate=fee_rate, fee_source=fee_source, fees_known=True)
        if net_pct <= min_profit_pct:
            return _result(True, (
                f"verified MECE basket, but the executable net is "
                f"{net_pct * 100:.2f}% (${net_usd:.4f}), under the "
                f"{min_profit_pct * 100:.1f}% the engine requires after costs"),
                cost=cost, cost_basis="executable_asks", net_usd=net_usd,
                net_pct=net_pct, pricing=pricing, fee_rate=fee_rate,
                fee_source=fee_source, fees_known=True)

        return _result(True, None, cost=cost, cost_basis="executable_asks",
                       net_usd=net_usd, net_pct=net_pct, pricing=pricing,
                       fee_rate=fee_rate, fee_source=fee_source, fees_known=True)

    def find_combinatorial_arbitrage(
            self, markets: List[Market],
            book_lookup: Optional[Callable[[str], Optional[Dict]]] = None,
            fee_rate_lookup: Optional[Callable[[Market], Any]] = None,
            neg_risk_lookup: Optional[Callable[[Market], Any]] = None,
    ) -> List[CombinatorialGroup]:
        """
        Candidate baskets, each with the evidence that makes it an arbitrage or
        the reason it is not one.

        `book_lookup(market_id) -> validated book`,
        `fee_rate_lookup(market) -> rate or {"rate": .., "source": ..}` and
        `neg_risk_lookup(market) -> bool or {"neg_risk": ..}` come from the caller
        because this engine holds no venue connection. Without them every group is
        reported as UNVERIFIED with that reason - a basket priced off the venue's
        last trade is not a trade, and this says so instead of inventing one.
        """
        opportunities: List[CombinatorialGroup] = []
        slug_groups = self._group_by_event_slug(markets)

        def _build(group_id: str, slug: str, group_markets: List[Market],
                   min_sum_gap: float) -> CombinatorialGroup:
            verdict = self._evaluate_group(
                group_markets, book_lookup=book_lookup,
                fee_rate_lookup=fee_rate_lookup, neg_risk_lookup=neg_risk_lookup,
                min_profit_pct=self.min_profit_pct, min_sum_gap=min_sum_gap)
            reasoning = (
                f"{group_id}: {verdict['outcomes_seen']} market(s) sum YES "
                f"{verdict['sum_yes']:.3f} | {verdict['arb_type']} cost "
                f"${verdict['cost']:.4f} ({verdict['cost_basis']}) payout "
                f"${verdict['payout']:.3f}"
                + (f" net ${verdict['net_profit_usd']:.4f}" if verdict['net_profit_usd'] is not None
                   else " net not measured")
                + f" | {'VERIFIED' if verdict['verified'] else 'NOT AN ARBITRAGE'}: "
                + f"{verdict['reason']} | research only: this engine places no orders"
                + f" | Markets: {', '.join([m.question[:30] for m in group_markets[:3]])}")
            return CombinatorialGroup(
                group_id=group_id, event_slug=slug, markets=group_markets,
                sum_yes=verdict["sum_yes"], is_exhaustive=verdict["is_exhaustive"],
                is_exclusive=verdict["is_exclusive"],
                arbitrage_type=verdict["arb_type"],
                estimated_profit_pct=verdict["estimated_profit_pct"],
                cost=verdict["cost"], payout=verdict["payout"],
                should_trade=verdict["should_trade"], reasoning=reasoning,
                verified=verdict["verified"], verdict_reason=verdict["reason"],
                verification=verdict["verification"],
                blocked_reason=verdict["blocked_reason"],
                cost_basis=verdict["cost_basis"],
                net_profit_usd=verdict["net_profit_usd"],
                fee_rate=verdict["fee_rate"], fee_source=verdict["fee_source"],
                outcomes_declared=verdict["outcomes_declared"],
                outcomes_seen=verdict["outcomes_seen"],
                executability=verdict["executability"],
                wired_to_execution=False)

        # Method 1: group by event_slug. Still only a CANDIDATE grouping - the
        # venue's own marks decide whether it is a basket.
        for slug, group_markets in slug_groups.items():
            # NOT truncated before the sum any more. Keeping the ten most liquid
            # markets of a thirty-market event and then summing their prices is
            # how "sum 0.006, profit $0.994" appeared: the sum was computed over
            # a subset, so of course it was under 1. The full group is evaluated
            # and a basket too large to post is refused, with its size named.
            opportunities.append(_build(slug, slug, group_markets, min_sum_gap=0.03))

        # Method 2: question-similarity groups, for markets the scan gave no
        # event slug. A similar question is NOT evidence of exclusivity, so these
        # go through exactly the same verification and are refused without it.
        similarity_groups = self._detect_exclusive_by_question(markets)
        for group_markets in similarity_groups:
            slug = group_markets[0].event_slug
            if slug in slug_groups:
                continue
            opportunities.append(_build(
                f"similarity_{group_markets[0].id}",
                group_markets[0].event_slug or "similarity",
                group_markets, min_sum_gap=0.05))

        # Verified first, then by the profit that survived, so a fabricated
        # percentage can never outrank a real basket in any reader's view.
        opportunities.sort(
            key=lambda x: (x.verified, x.net_profit_usd if x.net_profit_usd is not None else -1.0),
            reverse=True)

        tradeable = [o for o in opportunities if o.should_trade]
        verified = [o for o in opportunities if o.verified]
        logger.info(
            f"Combinatorial scan: {len(markets)} markets -> {len(slug_groups)} "
            f"slug groups + {len(similarity_groups)} similarity groups -> "
            f"{len(opportunities)} candidate(s), {len(verified)} verified basket(s), "
            f"{len(tradeable)} tradeable at executable prices (research only: this "
            f"engine places no orders)")
        refused = [o for o in opportunities if not o.verified and o.blocked_reason]
        if refused:
            # One line naming the biggest refused candidate and WHY, so the
            # operator can see the fantasy sums are being refused rather than
            # quietly dropped.
            biggest = max(refused, key=lambda o: abs(o.sum_yes - 1.0))
            logger.info(
                f"Combinatorial refused {len(refused)} group(s) as NOT an "
                f"arbitrage; largest gap: {biggest.group_id} sum YES "
                f"{biggest.sum_yes:.3f} - {biggest.blocked_reason}")
        return opportunities

    def find_negative_risk_conversions(self, positions: List[Dict], markets: List[Market]) -> List[NegativeRiskConversion]:
        """
        Find negative risk conversions - convert NO shares across exclusive markets to free capital
        Polymarket feature: if you hold NO on all outcomes in exclusive group, at least n-1 NOs must win
        You can convert n-1 NO shares to USDC immediately
        
        Example: Trump 0.45, Biden 0.30, Other 0.25. If you hold 10 NO Trump, 10 NO Biden, 10 NO Other
        You know at least 2 of those NOs will win (only 1 YES can win), so you can convert 10 NOs to $10 USDC?
        Actually: you have 10 NO each, total 30 NO shares, guaranteed 20 will win (n-1 per share) => $20 payout
        You can convert to free capital.
        
        This unlocks capital and creates synthetic positions.
        """
        conversions: List[NegativeRiskConversion] = []
        
        # Group positions by event_slug
        positions_by_event: Dict[str, List[Dict]] = {}
        for pos in positions:
            slug = pos.get("event_slug", "unknown")
            if slug not in positions_by_event:
                positions_by_event[slug] = []
            positions_by_event[slug].append(pos)
        
        for event_slug, event_positions in positions_by_event.items():
            if len(event_positions) < 2:
                continue
            
            # Check if all positions are NO side
            no_positions = [p for p in event_positions if p.get("side", "").upper() == "NO" or p.get("outcome", "").upper() == "NO"]
            if len(no_positions) < 2:
                continue
            
            # Find corresponding markets
            event_markets = [m for m in markets if m.event_slug == event_slug]
            if len(event_markets) < 2:
                continue
            
            # Calculate convertible value
            # If you hold X NO shares on each of n markets, you can convert X*(n-1) NO shares to USDC? 
            # Simplified: min NO shares across group * (n-1) is guaranteed win
            no_shares = {}
            min_shares = float('inf')
            for pos in no_positions:
                shares = pos.get("shares", pos.get("amount_usd", 0) / 0.5)  # approximate
                market_id = pos.get("market_id", "")
                no_shares[market_id] = shares
                min_shares = min(min_shares, shares)
            
            if min_shares == float('inf') or min_shares <= 0:
                continue
            
            n = len(no_positions)
            convertible = min_shares * (n - 1)  # guaranteed winning NO shares
            freed_capital = convertible * 1.0  # each NO winning = $1
            
            reasoning = (
                f"Negative risk conversion for {event_slug}: {n} NO positions, min shares {min_shares:.2f}, "
                f"convertible {convertible:.2f} shares -> ${freed_capital:.2f} freed capital | "
                f"NO shares {no_shares} | "
                f"This unlocks capital and creates synthetic YES position"
            )
            
            conv = NegativeRiskConversion(
                group_id=event_slug,
                markets=event_markets,
                no_shares=no_shares,
                convertible_value=convertible,
                freed_capital=freed_capital,
                reasoning=reasoning
            )
            conversions.append(conv)
            logger.info(f"NEGATIVE RISK CONVERSION: {reasoning}")
        
        return conversions

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Combinatorial Arbitrage + Negative Risk",
            "method": ("MECE arithmetic: in a basket of mutually exclusive outcomes "
                       "sum YES should be 1; under 1 buy all YES, over 1 buy all NO"),
            "verified_only": ("a basket counts only with the venue's own negRisk "
                              "mark, the event's complete outcome list, a sum inside "
                              "the 0.75-1.25 band, an executable ask/bid price per "
                              "leg and the venue's own fee rate"),
            "negative_risk": "Polymarket feature: convert NO shares across exclusive markets to free capital, at least n-1 NOs must win",
            "profit_example": "Trump 0.45+Biden 0.30+Other 0.20=0.95<1 => buy all YES $0.95 guaranteed $1.00 profit 5.26%",
            "importance": "Highest risk-adjusted return for $50 bankroll - Top 5 to implement first per user",
            "implementation": ("group by event_slug and question similarity, then "
                               "verify each group: negRisk mark, complete outcome "
                               "list, sum band, executable prices, venue fee"),
            "not_executed": ("this engine places no orders; the loop records its "
                             "output as research and nothing downstream trades it"),
            "real_prize": "Arbitrage just needs speed and execution, not directional edge, more realistic for $50 than directional bets"
        }
