"""
Bayesian Updating - Update fair value as news arrives with exponential decay, older news matters less

"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from loguru import logger
import math


@dataclass
class NewsEvent:
    timestamp: datetime
    headline: str
    sentiment: float  # -1 to +1
    credibility: float  # 0-1
    impact: float  # -1 to +1 estimated impact on probability
    source: str


@dataclass
class BayesianState:
    market_id: str
    prior: float  # initial fair value
    posterior: float  # updated fair value
    confidence: float
    news_events: List[NewsEvent]
    last_update: datetime
    reasoning: str


class BayesianUpdater:
    """
    Bayesian updating with exponential decay for news
    """
    def __init__(self, decay_half_life_hours: float = 24.0):
        self.decay_half_life = decay_half_life_hours  # half life for news impact

    def _time_decay(self, event_time: datetime, now: datetime = None) -> float:
        """Exponential decay based on time: older news matters less"""
        now = now or datetime.now(timezone.utc)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        
        hours_ago = (now - event_time).total_seconds() / 3600
        # Decay: weight = 0.5^(hours_ago / half_life)
        decay = 0.5 ** (hours_ago / self.decay_half_life)
        return max(0.05, decay)  # min 5% weight even for old news

    def update(self, prior: float, news_events: List[NewsEvent],
               base_rate: float = 0.5, market_price: float = 0.5) -> BayesianState:
        """
        Bayesian update: posterior ∝ prior * likelihood(news)
        Simplified: posterior = weighted average of prior, news impacts, base rate, market price
        With time decay
        """
        if not news_events:
            return BayesianState(
                market_id="",
                prior=prior,
                posterior=prior,
                confidence=0.5,
                news_events=[],
                last_update=datetime.now(timezone.utc),
                reasoning=f"No news, posterior = prior {prior:.3f}"
            )
        
        now = datetime.now(timezone.utc)
        
        # Calculate weighted news impact with time decay
        total_weight = 0
        weighted_impact = 0
        
        for event in news_events:
            decay = self._time_decay(event.timestamp, now)
            weight = decay * event.credibility
            # Impact -1 to +1 means change in probability
            # e.g. positive news for Trump win +0.1 => increases prob by 0.1*weight
            weighted_impact += event.impact * weight
            total_weight += weight
        
        avg_impact = weighted_impact / total_weight if total_weight > 0 else 0
        
        # Bayesian update: posterior = prior + impact, but also pull towards base rate and market
        # Weight prior 40%, news 40%, base rate 10%, market 10%
        posterior = (
            prior * 0.4 +
            (prior + avg_impact) * 0.4 +  # news adjusted prior
            base_rate * 0.1 +
            market_price * 0.1
        )
        
        posterior = max(0.01, min(0.99, posterior))
        
        # Confidence based on agreement and recency
        # If news agrees (all positive or all negative), high confidence, else low
        # If recent news, higher confidence
        recent_news = [e for e in news_events if (now - e.timestamp).total_seconds() / 3600 < 12]
        recency_confidence = min(1.0, len(recent_news) / 3 * 0.5 + 0.5)
        
        # Agreement: variance of impacts low => high agreement
        if len(news_events) > 1:
            impacts = [e.impact for e in news_events]
            mean_impact = sum(impacts) / len(impacts)
            variance = sum((i - mean_impact) ** 2 for i in impacts) / len(impacts)
            agreement = max(0.1, 1.0 - variance * 2)
        else:
            agreement = 0.7
        
        confidence = (recency_confidence * 0.5 + agreement * 0.5) * 0.8 + 0.2  # 0.2-1.0
        
        reasoning = (
            f"Bayesian update: prior {prior:.3f} + news impact {avg_impact:+.3f} (weighted {total_weight:.2f} from {len(news_events)} events) | "
            f"Base rate {base_rate:.3f} market {market_price:.3f} => posterior {posterior:.3f} | "
            f"Recent news {len(recent_news)} recency conf {recency_confidence:.2f} agreement {agreement:.2f} final conf {confidence:.2f} | "
            f"Decay half-life {self.decay_half_life}h, older news matters less"
        )
        
        return BayesianState(
            market_id="",
            prior=prior,
            posterior=posterior,
            confidence=confidence,
            news_events=news_events,
            last_update=now,
            reasoning=reasoning
        )

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Bayesian Updating",
            "method": "Update fair value as news arrives with exponential decay, older news matters less",
            "decay": f"Half-life {self.decay_half_life}h, weight = 0.5^(hours_ago/half_life), min 5%",
            "update": "Posterior = prior*0.4 + (prior+impact)*0.4 + base_rate*0.1 + market*0.1",
            "confidence": "Based on recency (recent news higher) and agreement (low variance higher)",
            "example": "Trump win prior 50%, positive news +0.1 impact with high credibility and recent => posterior 55-60%"
        }
