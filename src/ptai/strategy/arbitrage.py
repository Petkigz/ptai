"""
Arbitrage Engine - detects same event across venues with price discrepancy
Core to venue × market × strategy: venue C price 0.61, venue D price 0.72 for same event = arbitrage
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import re
from difflib import SequenceMatcher

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType


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


class ArbitrageEngine:
    """
    Detects arbitrage across venues
    Example: Polymarket Trump 0.61, Kalshi Trump 0.68 = 7% spread, buy low sell high if same resolution
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
            try:
                diff_days = abs((m1.end_date - m2.end_date).total_seconds()) / 86400
                if diff_days > 7:
                    similarity *= 0.5
                elif diff_days < 1:
                    similarity = min(1.0, similarity * 1.1)
            except:
                pass
        
        return similarity

    def find_arbitrage(self, markets: List[Market]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities across venues"""
        opportunities: List[ArbitrageOpportunity] = []
        
        # Group by venue for efficiency
        # Compare each market with others from different venues
        for i, m1 in enumerate(markets):
            for j in range(i+1, len(markets)):
                m2 = markets[j]
                if m1.source == m2.source:
                    continue  # Same venue, not arbitrage (could be internal arb but skip)
                
                # Same event confidence
                same_event_conf = self._same_event_score(m1, m2)
                if same_event_conf < self.min_confidence:
                    continue
                
                # Price spread
                price_a = m1.best_price
                price_b = m2.best_price
                spread = abs(price_b - price_a)
                
                if spread < self.min_spread:
                    continue
                
                # Estimate profit: buy low, sell high, minus fees
                # Simplified: if m1=0.61, m2=0.72, buy YES at 0.61 on venue A, buy NO at 0.28 (1-0.72) on venue B
                # Total cost 0.61+0.28=0.89, guaranteed 1.0, profit 0.11 = 11% / 0.89 = 12.3%
                # But need to consider fees, execution risk, resolution mismatch
                low_price = min(price_a, price_b)
                high_price = max(price_a, price_b)
                # Cost = low + (1 - high) = 1 - (high - low) = 1 - spread
                cost = low_price + (1 - high_price)
                profit = 1.0 - cost
                profit_pct = profit / cost if cost > 0 else 0
                
                # Adjust for confidence and fees
                fee_estimate = 0.02  # 2% combined fees
                adjusted_profit_pct = profit_pct * same_event_conf - fee_estimate
                
                # Should trade if adjusted profit > 3% and high confidence
                should_trade = adjusted_profit_pct > 0.03 and same_event_conf > 0.8
                
                # Determine sides
                if price_a < price_b:
                    side_a = "YES"
                    side_b = "NO"
                    venue_a = getattr(m1, 'source', 'unknown')
                    venue_b = getattr(m2, 'source', 'unknown')
                else:
                    side_a = "NO"
                    side_b = "YES"
                    venue_a = getattr(m1, 'source', 'unknown')
                    venue_b = getattr(m2, 'source', 'unknown')
                
                reasoning = (
                    f"Same event confidence {same_event_conf:.2f}: '{m1.question[:60]}' vs '{m2.question[:60]}' | "
                    f"Price {price_a:.3f} vs {price_b:.3f} spread {spread:.3f} | "
                    f"Cost {cost:.3f} profit {profit:.3f} ({profit_pct*100:.1f}%) adjusted {adjusted_profit_pct*100:.1f}% | "
                    f"Trade: {should_trade}"
                )
                
                opp = ArbitrageOpportunity(
                    market_a=m1,
                    market_b=m2,
                    venue_a=str(venue_a),
                    venue_b=str(venue_b),
                    price_a=price_a,
                    price_b=price_b,
                    spread=spread,
                    estimated_profit_pct=adjusted_profit_pct,
                    confidence_same_event=same_event_conf,
                    reasoning=reasoning,
                    should_trade=should_trade,
                    side_a=side_a,
                    side_b=side_b
                )
                opportunities.append(opp)
                
                if should_trade:
                    logger.info(f"ARBITRAGE FOUND: {reasoning}")
        
        # Sort by adjusted profit
        opportunities.sort(key=lambda x: x.estimated_profit_pct, reverse=True)
        logger.info(f"Arbitrage scan: {len(markets)} markets -> {len(opportunities)} candidates, {len([o for o in opportunities if o.should_trade])} tradeable")
        return opportunities

    def to_venue_opportunities(self, arb_opps: List[ArbitrageOpportunity]) -> List[VenueOpportunity]:
        """Convert arbitrage opps to VenueOpportunity for unified ranking"""
        venue_opps = []
        for arb in arb_opps:
            if not arb.should_trade:
                continue
            # Create two opportunities (one per venue) but mark as arbitrage
            # For unified ranking, create single opportunity representing arbitrage
            # Use market_a as primary, but note arbitrage
            from ..markets.base import Market
            # Arbitrage effective edge is profit pct
            opp = VenueOpportunity(
                market=arb.market_a,
                venue_id=f"{arb.venue_a}+{arb.venue_b}_arb",
                venue_type=VenueType.PREDICTION,
                side=arb.side_a,
                market_price=arb.price_a,
                estimated_fair=arb.price_b,  # Fair is other venue price
                raw_edge=arb.spread,
                effective_edge=arb.estimated_profit_pct,
                confidence=arb.confidence_same_event,
                uncertainty=1-arb.confidence_same_event,
                liquidity_score=min(1.0, min(arb.market_a.liquidity, arb.market_b.liquidity) / 10000),
                execution_quality=0.6,  # Arbitrage execution harder
                category="arbitrage",
                sources=[f"arb_{arb.venue_a}_{arb.venue_b}"],
                reasoning=arb.reasoning,
                bull_case=f"Buy {arb.side_a} at {arb.price_a} on {arb.venue_a}",
                bear_case=f"Buy {arb.side_b} at {1-arb.price_b:.3f} on {arb.venue_b}",
                resolution_risks=[f"Resolution mismatch risk conf {arb.confidence_same_event:.2f}", "Execution timing risk"],
                should_trade=arb.should_trade
            )
            opp.calculate_common_score()
            venue_opps.append(opp)
        
        return venue_opps
