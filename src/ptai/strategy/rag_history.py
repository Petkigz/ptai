"""
RAG over Historical Outcomes - Store past market questions, resolutions, price paths, retrieve similar markets to estimate base rates

"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import math
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone


@dataclass
class HistoricalMarket:
    market_id: str
    question: str
    category: str
    resolution: int  # 1 YES, 0 NO
    final_price: float
    volume: float
    price_path: List[float]  # price over time
    resolved_at: datetime
    similarity_score: float = 0.0


@dataclass
class BaseRateEstimate:
    market_id: str
    base_rate: float
    num_similar: int
    similar_markets: List[HistoricalMarket]
    confidence: float
    reasoning: str


class HistoricalRAG:
    """
    RAG over historical outcomes for base rates
    """
    def __init__(self):
        self.history: List[HistoricalMarket] = []
        self._load_mock_history()

    def _load_mock_history(self):
        """Load mock historical data - in production would be SQLite/Postgres"""
        mock_data = [
            HistoricalMarket("hist_1", "Will Trump win 2020 election?", "politics", 0, 0.35, 100000, [0.5, 0.45, 0.4, 0.35], datetime(2020, 11, 4, tzinfo=timezone.utc)),
            HistoricalMarket("hist_2", "Will Trump win 2016 election?", "politics", 1, 0.35, 80000, [0.3, 0.35, 0.4, 0.35], datetime(2016, 11, 9, tzinfo=timezone.utc)),
            HistoricalMarket("hist_3", "Will BTC be above $50k by Dec 2021?", "crypto", 1, 0.65, 50000, [0.5, 0.6, 0.65, 0.65], datetime(2021, 12, 31, tzinfo=timezone.utc)),
            HistoricalMarket("hist_4", "Will BTC be above $100k by Dec 2024?", "crypto", 0, 0.25, 60000, [0.4, 0.3, 0.25, 0.25], datetime(2024, 12, 31, tzinfo=timezone.utc)),
            HistoricalMarket("hist_5", "Will Fed raise rates in Jan 2023?", "economics", 1, 0.70, 40000, [0.6, 0.65, 0.7, 0.7], datetime(2023, 1, 31, tzinfo=timezone.utc)),
            HistoricalMarket("hist_6", "Will Fed cut rates in 2024?", "economics", 1, 0.60, 35000, [0.5, 0.55, 0.6, 0.6], datetime(2024, 12, 31, tzinfo=timezone.utc)),
        ]
        self.history = mock_data

    def _similarity(self, question1: str, question2: str) -> float:
        """Calculate similarity between questions"""
        q1 = question1.lower()
        q2 = question2.lower()
        # Sequence matcher
        seq_score = SequenceMatcher(None, q1, q2).ratio()
        # Keyword Jaccard
        words1 = set(re.findall(r'\w+', q1))
        words2 = set(re.findall(r'\w+', q2))
        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            return max(seq_score, jaccard)
        return seq_score

    def retrieve_similar(self, question: str, category: str = None, top_k: int = 5) -> List[HistoricalMarket]:
        """Retrieve similar historical markets"""
        scored = []
        for hist in self.history:
            if category and hist.category != category:
                continue
            sim = self._similarity(question, hist.question)
            if sim > 0.3:  # threshold
                hist_copy = HistoricalMarket(
                    market_id=hist.market_id,
                    question=hist.question,
                    category=hist.category,
                    resolution=hist.resolution,
                    final_price=hist.final_price,
                    volume=hist.volume,
                    price_path=hist.price_path,
                    resolved_at=hist.resolved_at,
                    similarity_score=sim
                )
                scored.append(hist_copy)
        
        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:top_k]

    def estimate_base_rate(self, market_id: str, question: str, category: str) -> BaseRateEstimate:
        """Estimate base rate from similar historical markets"""
        similar = self.retrieve_similar(question, category, top_k=10)
        
        if not similar:
            # No similar, use 50% base rate
            return BaseRateEstimate(
                market_id=market_id,
                base_rate=0.5,
                num_similar=0,
                similar_markets=[],
                confidence=0.3,
                reasoning=f"No similar historical markets for '{question[:50]}', using 50% base rate low confidence"
            )
        
        # Weighted average by similarity
        total_weight = sum(m.similarity_score for m in similar)
        if total_weight == 0:
            base_rate = 0.5
        else:
            base_rate = sum(m.resolution * m.similarity_score for m in similar) / total_weight
        
        # Confidence based on number of similar and avg similarity
        avg_sim = total_weight / len(similar) if similar else 0
        confidence = min(0.9, avg_sim * 0.7 + min(0.3, len(similar) / 10 * 0.3))
        
        reasoning = (
            f"Base rate from {len(similar)} similar historical markets: "
            f"{', '.join([f'{m.question[:30]} res {m.resolution} sim {m.similarity_score:.2f}' for m in similar[:3]])} | "
            f"Weighted base rate {base_rate:.3f} confidence {confidence:.2f} avg sim {avg_sim:.2f}"
        )
        
        return BaseRateEstimate(
            market_id=market_id,
            base_rate=base_rate,
            num_similar=len(similar),
            similar_markets=similar,
            confidence=confidence,
            reasoning=reasoning
        )

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "RAG over Historical Outcomes",
            "method": "Store past market questions, resolutions, price paths, retrieve similar to estimate base rates",
            "history_size": len(self.history),
            "categories": list(set(m.category for m in self.history)),
            "example": "Will Trump win 2024? Retrieve 2020, 2016 Trump elections, base rate 50% (1 win 1 loss)",
            "importance": "Base rates crucial for fair value, LLM alone hallucinates without historical anchor"
        }
