
"""
Cross-Venue Arbitrage Engine - Ready-Made Infrastructure
Uses CCXT unified + Veynor + Apify + OpenPX for cross-venue arb

Key opportunity: same event (e.g., Fed cuts rates in June) often trades on both Kalshi and Polymarket at different implied probabilities
If you can buy YES on cheaper venue and NO on other for combined cost below $1.00, lock in risk-free spread at settlement

Challenge: both legs must settle on exactly same event definition, fund transfers between venues slow
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from difflib import SequenceMatcher
import re

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType

@dataclass
class CrossVenueArb:
    event_key: str
    venue_a: str
    venue_b: str
    market_a: Market
    market_b: Market
    price_a: float
    price_b: float
    spread: float
    profit_pct: float
    fee_adjusted_profit: float
    confidence_same_event: float
    settlement_risk: str  # identical, similar, different
    capital_required: float
    should_trade: bool
    reasoning: str

class CrossVenueArbitrageEngine:
    """
    Enhanced cross-venue arb across 18 venues
    Uses CCXT unified layer + Veynor intelligence + Apify scanner + OpenPX sub-ms
    """
    def __init__(self, min_spread: float = 0.03, min_confidence: float = 0.7):
        self.min_spread = min_spread
        self.min_confidence = min_confidence
        self.fees = {
            "polymarket": 0.02,
            "kalshi": 0.07,
            "manifold": 0.0,
            "predictit": 0.10,
            "simmer": 0.01,
            "cymetica": 0.015,
            "binance": 0.001,
            "whitebit": 0.001,
            "afx_dex": 0.0005,
            "grvt": 0.0003,
            "pionex": 0.0005,
            "stock_mock": 0.0005,
            "betfair": 0.05,
            "betdaq": 0.02,
            "betconnect": 0.02,
            "ccxt_unified": 0.001,
            "veynor": 0.0,
            "openpx": 0.01,
            "apify": 0.0,
        }

    def _normalize_question(self, q: str) -> str:
        q = q.lower()
        q = re.sub(r'[^a-z0-9 ]', '', q)
        q = re.sub(r'\s+', ' ', q)
        # Remove filler
        for w in ["will", "the", "a", "an", "by", "on", "in", "at", "for"]:
            q = q.replace(f" {w} ", " ")
        return q.strip()

    def _same_event_score(self, m1: Market, m2: Market) -> float:
        # Use event_slug if same
        if m1.event_slug and m2.event_slug and m1.event_slug == m2.event_slug:
            return 0.95
        # Normalize questions
        q1 = self._normalize_question(m1.question)
        q2 = self._normalize_question(m2.question)
        seq = SequenceMatcher(None, q1, q2).ratio()
        # Jaccard
        words1 = set(q1.split())
        words2 = set(q2.split())
        jaccard = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        # End date proximity if available
        date_bonus = 0
        if m1.end_date and m2.end_date:
            try:
                diff = abs((m1.end_date - m2.end_date).days)
                if diff <= 2:
                    date_bonus = 0.1
                elif diff <= 7:
                    date_bonus = 0.05
            except:
                pass
        # Category bonus
        cat_bonus = 0.05 if m1.raw.get("category") == m2.raw.get("category") else 0
        score = seq*0.5 + jaccard*0.3 + date_bonus + cat_bonus
        return min(1.0, score)

    def _check_settlement_risk(self, m1: Market, m2: Market, confidence: float) -> str:
        # Different venues may resolve same event differently due to ambiguous rules
        # Always verify resolution criteria identical before arbing
        if confidence > 0.9 and m1.event_slug == m2.event_slug:
            return "identical - same event_slug, low settlement risk"
        if confidence > 0.8:
            return "similar - high confidence same event, verify resolution rules"
        if confidence > 0.7:
            return "moderate - check resolution criteria identical"
        return "different - high settlement risk, avoid arb"

    def find_cross_venue_arbitrage(self, markets_by_venue: Dict[str, List[Market]]) -> List[CrossVenueArb]:
        all_markets = []
        for venue_id, markets in markets_by_venue.items():
            for m in markets:
                # Ensure raw venue
                if not m.raw.get("venue"):
                    m.raw["venue"] = venue_id
                all_markets.append((venue_id, m))

        opportunities: List[CrossVenueArb] = []

        # Compare every pair across different venues
        for i in range(len(all_markets)):
            venue_a, m_a = all_markets[i]
            for j in range(i+1, len(all_markets)):
                venue_b, m_b = all_markets[j]
                if venue_a == venue_b:
                    continue  # no arb same venue (use combinatorial instead)
                # Skip intelligence/scanner venues
                if venue_a in ["veynor", "apify", "ccxt_unified"] or venue_b in ["veynor", "apify", "ccxt_unified"]:
                    continue

                confidence = self._same_event_score(m_a, m_b)
                if confidence < self.min_confidence:
                    continue

                price_a = m_a.best_price
                price_b = m_b.best_price
                spread = abs(price_a - price_b)
                if spread < self.min_spread:
                    continue

                # Arb logic: buy YES cheaper, buy NO expensive (1 - expensive)
                # Cost = cheap + (1 - expensive), profit = 1 - cost = spread
                cheap = min(price_a, price_b)
                expensive = max(price_a, price_b)
                cost = cheap + (1 - expensive)
                profit = 1 - cost  # = spread
                profit_pct = profit / cost if cost > 0 else 0

                # Fee adjusted
                fee_a = self.fees.get(venue_a, 0.02)
                fee_b = self.fees.get(venue_b, 0.02)
                fee_total = fee_a + fee_b
                fee_adjusted = profit_pct - fee_total

                settlement_risk = self._check_settlement_risk(m_a, m_b, confidence)

                # Capital required - both legs
                capital_required = 6.0  # $3 per leg for $50 bankroll

                # Should trade if fee-adjusted >2% and confidence >0.8 and settlement identical/similar
                should_trade = (
                    fee_adjusted > 0.02 and
                    confidence > 0.8 and
                    "identical" in settlement_risk or "similar" in settlement_risk
                )
                # Actually need both conditions
                should_trade = fee_adjusted > 0.02 and confidence > 0.8 and ("identical" in settlement_risk or "similar" in settlement_risk)

                reasoning = (
                    f"Cross-venue arb: {venue_a} {price_a:.3f} vs {venue_b} {price_b:.3f} spread {spread*100:.1f}% "
                    f"cost ${cost:.3f} profit ${profit:.3f} ({profit_pct*100:.1f}%) fee_adj {fee_adjusted*100:.1f}% "
                    f"conf {confidence:.2f} settlement {settlement_risk} | "
                    f"Event: {m_a.question[:40]} vs {m_b.question[:40]} | "
                    f"Buy YES cheap {cheap:.3f} ({venue_a if price_a==cheap else venue_b}) + NO expensive {expensive:.3f} => risk-free if same resolution"
                )

                arb = CrossVenueArb(
                    event_key=f"{m_a.event_slug or m_a.id}_{m_b.event_slug or m_b.id}",
                    venue_a=venue_a,
                    venue_b=venue_b,
                    market_a=m_a,
                    market_b=m_b,
                    price_a=price_a,
                    price_b=price_b,
                    spread=spread,
                    profit_pct=profit_pct,
                    fee_adjusted_profit=fee_adjusted,
                    confidence_same_event=confidence,
                    settlement_risk=settlement_risk,
                    capital_required=capital_required,
                    should_trade=should_trade,
                    reasoning=reasoning
                )
                opportunities.append(arb)
                if should_trade:
                    logger.info(f"CROSS-VENUE ARB FOUND: {reasoning}")

        opportunities.sort(key=lambda x: x.fee_adjusted_profit, reverse=True)
        logger.info(f"Cross-venue arb scan: {len(all_markets)} markets across {len(markets_by_venue)} venues -> {len(opportunities)} candidates, {len([o for o in opportunities if o.should_trade])} tradeable")
        return opportunities

    def to_venue_opportunities(self, arbs: List[CrossVenueArb]) -> List[VenueOpportunity]:
        opps = []
        for arb in arbs:
            if not arb.should_trade:
                continue
            # Create two opportunities - one per venue
            # For ranking, create combined opportunity
            market = arb.market_a if arb.price_a < arb.price_b else arb.market_b
            venue_id = arb.venue_a if arb.price_a < arb.price_b else arb.venue_b
            cheap_price = min(arb.price_a, arb.price_b)
            opp = VenueOpportunity(
                market=market,
                venue_id=f"{arb.venue_a}+{arb.venue_b}",
                venue_type=VenueType.PREDICTION,
                side="YES",
                market_price=cheap_price,
                estimated_fair=max(arb.price_a, arb.price_b),
                raw_edge=arb.spread,
                effective_edge=arb.fee_adjusted_profit,
                confidence=arb.confidence_same_event,
                uncertainty=0.1,
                liquidity_score=0.8,
                execution_quality=0.9 if "openpx" in [arb.venue_a, arb.venue_b] else 0.7,
                category="arbitrage",
                sources=[arb.venue_a, arb.venue_b, "ccxt_unified", "veynor"],
                reasoning=arb.reasoning,
                should_trade=arb.should_trade
            )
            opp.calculate_common_score()
            opp.raw = {"strategy": "cross_venue_arbitrage", "venues": [arb.venue_a, arb.venue_b], "settlement_risk": arb.settlement_risk}
            opps.append(opp)
        return opps

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Cross-Venue Arbitrage - CCXT + Veynor + Apify + OpenPX",
            "method": "Same event different price, buy YES cheaper + NO expensive cost < $1 lock risk-free spread at settlement",
            "venues": list(self.fees.keys()),
            "fees": self.fees,
            "challenges": "Both legs must settle exactly same event definition, fund transfers between venues slow, settlement risk different venues resolve differently",
            "infrastructure": {
                "ccxt": "Supports prediction markets alongside crypto, one strategy reads odds across multiple venues same code, handles per-venue quirks",
                "veynor": "Single Python client for Kalshi+Polymarket cross-venue data whale trades smart money arb opportunities, free 100 credits/month",
                "apify": "Paid arb scanners Polymarket Kalshi PredictIt ranking fee-adjusted edge $2 per 1000 matched pairs expensive for $50 but useful if scale",
                "openpx": "Rust client sub-millisecond WebSocket Polymarket+Kalshi typed interfaces highest-performance latency matters"
            },
            "risk_rules": "Correlation across venues: Fed cuts rates on Polymarket and same on Kalshi is one bet not two, aggregate per event not per venue. Settlement risk verify resolution identical. Capital fragmentation $50 across multiple venues tiny positions fixed costs gas withdrawal min order eat larger percentage concentrate 2-3 venues until bankroll grows",
            "example": "Fed cuts rates June: Polymarket 0.61 vs Kalshi 0.68 spread 7% cost 0.93 profit 0.07 profit_pct 7.5% fee_adj 7.5%-9% = -1.5% NO TRADE vs Polymarket 0.55 vs Kalshi 0.70 spread 15% cost 0.85 profit 0.15 profit_pct 17.6% fee_adj 8.6% TRADE"
        }
