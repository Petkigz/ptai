"""
Correlation Engine - detects correlated markets
"Will Trump..." "Will Republican..." "Will Senate..." "Will X policy..." may all correlate
"""
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import re
from loguru import logger

from ..markets.base import Market


@dataclass
class CorrelationGroup:
    group_id: str
    markets: List[str]  # market_ids
    reason: str
    correlation_strength: float  # 0-1


class CorrelationEngine:
    """
    Detects correlation between markets.
    If 8 positions are same bet, risk engine needs to understand.
    """
    def __init__(self):
        # Keywords that indicate correlation
        self.correlation_keywords = {
            "trump": ["trump", "donald"],
            "biden": ["biden", "joe"],
            "republican": ["republican", "gop", "trump"],
            "democrat": ["democrat", "democratic", "biden"],
            "senate": ["senate", "congress", "house"],
            "btc": ["btc", "bitcoin", "crypto"],
            "eth": ["eth", "ethereum"],
            "fed": ["fed", "federal reserve", "interest rate", "powell"],
            "election": ["election", "vote", "ballot"],
            "ukraine": ["ukraine", "russia", "putin", "zelensky"],
        }

    def extract_topics(self, market: Market) -> Set[str]:
        """Extract topics from market question"""
        text = market.question.lower()
        topics = set()
        for topic, keywords in self.correlation_keywords.items():
            for kw in keywords:
                if kw in text:
                    topics.add(topic)
                    break
        return topics

    def calculate_correlation(self, market_a: Market, market_b: Market) -> float:
        """Calculate correlation between two markets 0-1"""
        topics_a = self.extract_topics(market_a)
        topics_b = self.extract_topics(market_b)
        
        if not topics_a or not topics_b:
            return 0.0
        
        # Jaccard similarity of topics
        intersection = len(topics_a & topics_b)
        union = len(topics_a | topics_b)
        jaccard = intersection / union if union > 0 else 0
        
        # If same event slug, high correlation
        if market_a.event_slug and market_a.event_slug == market_b.event_slug:
            jaccard = max(jaccard, 0.8)
        
        # If same category and close end dates, moderate correlation
        # Would need category detection
        
        return jaccard

    def group_markets(self, markets: List[Market]) -> List[CorrelationGroup]:
        """Group markets by correlation"""
        groups: List[CorrelationGroup] = []
        used = set()
        
        for i, market_a in enumerate(markets):
            if market_a.id in used:
                continue
            
            group_markets = [market_a.id]
            topics = self.extract_topics(market_a)
            
            for j, market_b in enumerate(markets[i+1:], i+1):
                if market_b.id in used:
                    continue
                corr = self.calculate_correlation(market_a, market_b)
                if corr > 0.5:  # threshold for correlation
                    group_markets.append(market_b.id)
                    used.add(market_b.id)
            
            if len(group_markets) > 1:
                group_id = f"group_{'_'.join(list(topics)[:2])}" if topics else f"group_{i}"
                groups.append(CorrelationGroup(
                    group_id=group_id,
                    markets=group_markets,
                    reason=f"Shared topics: {topics}",
                    correlation_strength=0.6
                ))
                used.add(market_a.id)
        
        return groups

    def get_correlation_group(self, market: Market, existing_positions: List[Dict]) -> str:
        """Get correlation group for new market based on existing positions"""
        topics = self.extract_topics(market)
        if not topics:
            return "uncorrelated"
        
        # Check if any existing position shares topics
        for pos in existing_positions:
            pos_topics = self.extract_topics_from_question(pos.get("question", ""))
            if topics & pos_topics:
                # Shares topic, use that topic as group
                shared = list(topics & pos_topics)[0]
                return shared
        
        # No correlation, use primary topic or uncorrelated
        return list(topics)[0] if topics else "uncorrelated"

    def extract_topics_from_question(self, question: str) -> Set[str]:
        """Extract topics from question string"""
        text = question.lower()
        topics = set()
        for topic, keywords in self.correlation_keywords.items():
            for kw in keywords:
                if kw in text:
                    topics.add(topic)
                    break
        return topics

    def check_portfolio_correlation_risk(self, new_market: Market, existing_positions: List[Dict]) -> Tuple[bool, float, str]:
        """
        Check if adding new market would create too much correlated exposure
        Returns (is_risky, correlation_score, reason)
        """
        new_topics = self.extract_topics(new_market)
        if not new_topics:
            return False, 0.0, "No topics detected - uncorrelated"
        
        # Count how many existing positions share topics
        correlated_count = 0
        correlated_exposure = 0
        for pos in existing_positions:
            pos_topics = self.extract_topics_from_question(pos.get("question", ""))
            if new_topics & pos_topics:
                correlated_count += 1
                correlated_exposure += pos.get("amount_usd", 0)
        
        if correlated_count >= 3:
            return True, 0.8, f"Already have {correlated_count} correlated positions sharing {new_topics}, exposure ${correlated_exposure}"
        elif correlated_count >= 2:
            return False, 0.5, f"Moderate correlation: {correlated_count} existing share {new_topics}"
        
        return False, 0.0, "Low correlation"
