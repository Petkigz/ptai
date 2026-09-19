"""
Combinatorial Arbitrage - Linear Programming for mutually exclusive/collectively exhaustive markets
Polymarket negative risk feature lets you convert NO shares across exclusive markets to free capital

If P(A)+P(B)+P(C) < 1, buy all YES
If P(A)+P(B)+P(C) > 1, sell all YES (buy all NO)

Example:
- Market group: Who wins election? Trump 0.45, Biden 0.30, Other 0.20 = sum 0.95 <1 => buy all YES for $0.95, guaranteed $1.00 profit $0.05 = 5.26%
- If sum 1.08 >1 => buy all NO, cost (1-0.45)+(1-0.30)+(1-0.20)=0.55+0.70+0.80=2.05 for $2.00 payout? Actually need to think: buying NO is buying opposite.
  For MECE, exactly one YES wins. If sum >1, you can sell YES or buy NO baskets.

Negative Risk: Polymarket lets you convert NO shares across mutually exclusive markets
- If you hold NO on Trump, NO on Biden, NO on Other in same event, you can convert to USDC because at least 2 NOs must win (only 1 YES can win)
- This frees capital and creates synthetic positions

Implementation uses linear programming concept but simplified for PTAI - no need full LP solver for $50, heuristic grouping
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
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
        """Detect mutually exclusive markets by question patterns"""
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

    def find_combinatorial_arbitrage(self, markets: List[Market]) -> List[CombinatorialGroup]:
        """
        Find MECE arbitrage opportunities
        Uses linear programming concept: sum of YES prices should =1 for exhaustive exclusive
        """
        opportunities: List[CombinatorialGroup] = []
        
        # Method 1: Group by event_slug (most reliable)
        slug_groups = self._group_by_event_slug(markets)
        
        for slug, group_markets in slug_groups.items():
            if len(group_markets) > self.max_group_size:
                # Too large, skip or take subset with highest liquidity
                group_markets = sorted(group_markets, key=lambda x: x.liquidity, reverse=True)[:self.max_group_size]
            
            sum_yes = sum(m.best_price for m in group_markets)
            # For MECE, sum should be 1.0
            # If sum <1, buy all YES: cost = sum, payout =1, profit =1-sum
            # If sum >1, sell all YES (buy all NO): cost = (n - sum), payout = n-1? Actually need to calculate properly
            # Simplified: if sum>1, you can arbitrage by buying NOs: cost = sum of NO prices = n - sum_yes, payout = n-1 (since exactly one YES wins, n-1 NOs win)
            # Profit = (n-1) - (n - sum) = sum -1
            
            n = len(group_markets)
            if sum_yes < 1.0:
                cost = sum_yes
                payout = 1.0
                profit = payout - cost
                profit_pct = profit / cost if cost > 0 else 0
                arb_type = "buy_all_yes"
            elif sum_yes > 1.0:
                # Buying all NOs: cost = sum of (1-price) = n - sum_yes, payout = n-1 (all but one NO win)
                cost = n - sum_yes
                payout = n - 1
                profit = payout - cost
                # Actually profit = (n-1) - (n - sum) = sum -1
                profit_pct = profit / cost if cost > 0 else 0
                arb_type = "sell_all_yes_buy_all_no"
            else:
                cost = sum_yes
                payout = 1.0
                profit = 0
                profit_pct = 0
                arb_type = "none"
            
            # Adjust for fees (2% per trade, n trades)
            fee_estimate = 0.02 * n * 0.5  # half because some may be maker
            adjusted_profit_pct = profit_pct - fee_estimate
            
            should_trade = abs(sum_yes - 1.0) > 0.03 and adjusted_profit_pct > self.min_profit_pct
            
            # Check if group is likely exhaustive (sum close to 1 in normal times)
            # If sum is far from 1 (<0.8 or >1.2) may not be exhaustive, could be missing outcomes
            is_exhaustive = 0.8 <= sum_yes <= 1.2 or len(group_markets) >= 3
            is_exclusive = True  # Assume exclusive if same event_slug
            
            reasoning = (
                f"MECE group {slug}: {n} markets sum YES {sum_yes:.3f} | "
                f"Type {arb_type} cost ${cost:.3f} payout ${payout:.3f} profit ${profit:.3f} ({profit_pct*100:.1f}%) "
                f"adj {adjusted_profit_pct*100:.1f}% after fees | "
                f"Exhaustive {is_exhaustive} Exclusive {is_exclusive} | "
                f"Should trade {should_trade} | "
                f"Markets: {', '.join([m.question[:30] for m in group_markets[:3]])}"
            )
            
            group = CombinatorialGroup(
                group_id=slug,
                event_slug=slug,
                markets=group_markets,
                sum_yes=sum_yes,
                is_exhaustive=is_exhaustive,
                is_exclusive=is_exclusive,
                arbitrage_type=arb_type,
                estimated_profit_pct=adjusted_profit_pct,
                cost=cost,
                payout=payout,
                should_trade=should_trade,
                reasoning=reasoning
            )
            opportunities.append(group)
            
            if should_trade:
                logger.info(f"COMBINATORIAL ARB FOUND: {reasoning}")
        
        # Method 2: Also try question similarity grouping for markets without event_slug
        similarity_groups = self._detect_exclusive_by_question(markets)
        for group_markets in similarity_groups:
            # Skip if already covered by slug groups
            slug = group_markets[0].event_slug
            if slug in slug_groups:
                continue
            
            sum_yes = sum(m.best_price for m in group_markets)
            n = len(group_markets)
            if sum_yes < 1.0:
                cost = sum_yes
                payout = 1.0
                profit = payout - cost
                profit_pct = profit / cost if cost > 0 else 0
                arb_type = "buy_all_yes"
            else:
                cost = n - sum_yes
                payout = n - 1
                profit = payout - cost
                profit_pct = profit / cost if cost > 0 else 0
                arb_type = "sell_all_yes_buy_all_no"
            
            fee_estimate = 0.02 * n * 0.5
            adjusted_profit_pct = profit_pct - fee_estimate
            should_trade = abs(sum_yes - 1.0) > 0.05 and adjusted_profit_pct > self.min_profit_pct
            
            reasoning = (
                f"Similarity group: {n} markets sum YES {sum_yes:.3f} | "
                f"Type {arb_type} cost ${cost:.3f} payout ${payout:.3f} profit {profit_pct*100:.1f}% adj {adjusted_profit_pct*100:.1f}% | "
                f"Should trade {should_trade}"
            )
            
            group = CombinatorialGroup(
                group_id=f"similarity_{group_markets[0].id}",
                event_slug=group_markets[0].event_slug or "similarity",
                markets=group_markets,
                sum_yes=sum_yes,
                is_exhaustive=False,
                is_exclusive=True,
                arbitrage_type=arb_type,
                estimated_profit_pct=adjusted_profit_pct,
                cost=cost,
                payout=payout,
                should_trade=should_trade,
                reasoning=reasoning
            )
            opportunities.append(group)
        
        opportunities.sort(key=lambda x: abs(x.estimated_profit_pct), reverse=True)
        logger.info(f"Combinatorial scan: {len(markets)} markets -> {len(slug_groups)} slug groups + {len(similarity_groups)} similarity groups -> {len(opportunities)} candidates, {len([o for o in opportunities if o.should_trade])} tradeable")
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
            "method": "Linear programming concept: sum YES prices in MECE group should =1. If <1 buy all YES, if >1 buy all NO",
            "negative_risk": "Polymarket feature: convert NO shares across exclusive markets to free capital, at least n-1 NOs must win",
            "profit_example": "Trump 0.45+Biden 0.30+Other 0.20=0.95<1 => buy all YES $0.95 guaranteed $1.00 profit 5.26%",
            "importance": "Highest risk-adjusted return for $50 bankroll - Top 5 to implement first per user",
            "implementation": "Group by event_slug (natural MECE), also similarity grouping, calculate sum YES, profit, fees, should_trade if |sum-1|>3% and profit>2%",
            "real_prize": "Arbitrage just needs speed and execution, not directional edge, more realistic for $50 than directional bets"
        }
