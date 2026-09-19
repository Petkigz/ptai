"""
Event Graph Consistency - Build graph of related markets, flag violations

"Trump wins" implies "Republican wins"
"X by June" implies "X by December"

Flag violations where implied probabilities don't match
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger
import re

from ..markets.base import Market


@dataclass
class ConsistencyRule:
    rule_id: str
    description: str
    implies: str  # e.g. "trump_win" implies "republican_win"
    implied_by: str
    type: str  # "implication", "temporal", "mutual_exclusion"
    strength: float  # 0-1 how strong implication


@dataclass
class ConsistencyViolation:
    market_a: Market
    market_b: Market
    rule: ConsistencyRule
    price_a: float
    price_b: float
    expected_relation: str
    actual_relation: str
    violation_size: float
    edge: float
    should_trade: bool
    reasoning: str


class EventGraphConsistencyEngine:
    """
    Builds event graph and flags consistency violations
    """
    def __init__(self):
        self.rules = self._build_default_rules()

    def _build_default_rules(self) -> List[ConsistencyRule]:
        return [
            ConsistencyRule("trump_implies_gop", "Trump wins implies Republican wins", "trump_win", "republican_win", "implication", 0.95),
            ConsistencyRule("biden_implies_dem", "Biden wins implies Democrat wins", "biden_win", "democrat_win", "implication", 0.95),
            ConsistencyRule("june_implies_dec", "X by June implies X by December", "x_by_june", "x_by_december", "temporal", 0.99),
            ConsistencyRule("btc_100k_implies_90k", "BTC >100k implies BTC >90k", "btc_gt_100k", "btc_gt_90k", "implication", 0.99),
            ConsistencyRule("fed_hike_implies_not_cut", "Fed hike implies not cut", "fed_hike", "not_fed_cut", "mutual_exclusion", 0.90),
            ConsistencyRule("trump_vs_biden_exclusive", "Trump win and Biden win mutually exclusive (same election)", "trump_win", "biden_win", "mutual_exclusion", 0.99),
        ]

    def _extract_event_type(self, market: Market) -> Set[str]:
        """Extract event types from market question"""
        q = market.question.lower()
        types = set()
        if "trump" in q and "win" in q:
            types.add("trump_win")
        if "biden" in q and "win" in q:
            types.add("biden_win")
        if "republican" in q and "win" in q:
            types.add("republican_win")
        if "democrat" in q and "win" in q:
            types.add("democrat_win")
        if "btc" in q or "bitcoin" in q:
            if "100k" in q or "100,000" in q:
                types.add("btc_gt_100k")
            if "90k" in q or "90,000" in q:
                types.add("btc_gt_90k")
        if "june" in q:
            types.add("x_by_june")
        if "december" in q or "dec" in q:
            types.add("x_by_december")
        if "fed" in q and ("hike" in q or "raise" in q):
            types.add("fed_hike")
        if "fed" in q and "cut" in q:
            types.add("fed_cut")
        return types

    def find_violations(self, markets: List[Market]) -> List[ConsistencyViolation]:
        violations: List[ConsistencyViolation] = []
        
        # Build map of event type -> markets
        type_to_markets: Dict[str, List[Market]] = {}
        for m in markets:
            types = self._extract_event_type(m)
            for t in types:
                if t not in type_to_markets:
                    type_to_markets[t] = []
                type_to_markets[t].append(m)
        
        # Check each rule
        for rule in self.rules:
            # Find markets for implies and implied_by
            implies_markets = type_to_markets.get(rule.implies, [])
            implied_by_markets = type_to_markets.get(rule.implied_by, [])
            
            if not implies_markets or not implied_by_markets:
                continue
            
            for m_a in implies_markets:
                for m_b in implied_by_markets:
                    if m_a.id == m_b.id:
                        continue
                    
                    price_a = m_a.best_price
                    price_b = m_b.best_price
                    
                    violation = None
                    edge = 0
                    
                    if rule.type == "implication":
                        # If A implies B, then P(A) <= P(B)
                        # If P(A) > P(B), violation
                        if price_a > price_b + 0.02:  # 2% tolerance
                            violation_size = price_a - price_b
                            edge = violation_size * rule.strength
                            violation = ConsistencyViolation(
                                market_a=m_a,
                                market_b=m_b,
                                rule=rule,
                                price_a=price_a,
                                price_b=price_b,
                                expected_relation=f"P({rule.implies}) <= P({rule.implied_by})",
                                actual_relation=f"P({rule.implies})={price_a:.3f} > P({rule.implied_by})={price_b:.3f}",
                                violation_size=violation_size,
                                edge=edge,
                                should_trade=edge > 0.05,
                                reasoning=f"Implication violation: {rule.description} but {price_a:.3f} > {price_b:.3f} diff {violation_size:.3f} edge {edge*100:.1f}%"
                            )
                    elif rule.type == "temporal":
                        # X by June implies X by December, so P(June) <= P(Dec)
                        if price_a > price_b + 0.02:
                            violation_size = price_a - price_b
                            edge = violation_size * rule.strength
                            violation = ConsistencyViolation(
                                market_a=m_a,
                                market_b=m_b,
                                rule=rule,
                                price_a=price_a,
                                price_b=price_b,
                                expected_relation=f"P({rule.implies}) <= P({rule.implied_by}) temporal",
                                actual_relation=f"P({rule.implies})={price_a:.3f} > P({rule.implied_by})={price_b:.3f}",
                                violation_size=violation_size,
                                edge=edge,
                                should_trade=edge > 0.05,
                                reasoning=f"Temporal violation: {rule.description} but {price_a:.3f} > {price_b:.3f}"
                            )
                    elif rule.type == "mutual_exclusion":
                        # If mutually exclusive, P(A)+P(B) <=1
                        # If sum >1, violation
                        sum_prices = price_a + price_b
                        if sum_prices > 1.02:
                            violation_size = sum_prices - 1.0
                            edge = violation_size * rule.strength * 0.5  # half because need to split
                            violation = ConsistencyViolation(
                                market_a=m_a,
                                market_b=m_b,
                                rule=rule,
                                price_a=price_a,
                                price_b=price_b,
                                expected_relation=f"P({rule.implies})+P({rule.implied_by}) <=1",
                                actual_relation=f"Sum {sum_prices:.3f} >1",
                                violation_size=violation_size,
                                edge=edge,
                                should_trade=edge > 0.05,
                                reasoning=f"Mutual exclusion violation: {rule.description} sum {sum_prices:.3f} >1 diff {violation_size:.3f}"
                            )
                    
                    if violation:
                        violations.append(violation)
                        if violation.should_trade:
                            logger.info(f"CONSISTENCY VIOLATION: {violation.reasoning}")
        
        violations.sort(key=lambda x: x.edge, reverse=True)
        return violations

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Event Graph Consistency",
            "rules": len(self.rules),
            "examples": [
                "Trump wins implies Republican wins - if P(Trump)>P(GOP) violation",
                "X by June implies X by December - temporal",
                "BTC >100k implies BTC >90k",
                "Trump win and Biden win mutually exclusive same election sum <=1"
            ],
            "method": "Build graph of related markets, extract event types from questions, check implication/temporal/mutual_exclusion rules, flag violations where implied probs don't match"
        }
