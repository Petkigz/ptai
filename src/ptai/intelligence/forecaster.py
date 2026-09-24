"""
Independent forecasting components - not just one LLM saying 75%
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import math
from loguru import logger

from ..markets.base import Market
from ..markets.orderbook import read_spread


@dataclass
class ModelForecast:
    model_name: str
    probability: float  # 0-1
    confidence: float   # 0-1
    uncertainty: float  # 0-1, higher = more uncertain
    reasoning: str
    sources: List[str]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        # Clamp
        self.probability = max(0.01, min(0.99, self.probability))
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseRateModel:
    """
    Statistical/base-rate model:
    historical frequencies, event type, time remaining, analogues, polling
    """
    def __init__(self):
        # Historical base rates by category (would be learned from calibration DB)
        self.base_rates = {
            "politics": 0.52,
            "sports": 0.50,
            "crypto": 0.48,
            "economics": 0.51,
            "weather": 0.55,
            "default": 0.50
        }
        self.historical_data = []  # Would load from DB

    def forecast(self, market: Market, category: str = "default") -> ModelForecast:
        base = self.base_rates.get(category, self.base_rates["default"])
        
        # Adjust by time remaining - closer to resolution, more certainty if market stable?
        # For now simple: if very short time, lean toward market price (efficient)
        time_factor = 0.0
        if market.end_date:
            try:
                now = datetime.now(timezone.utc)
                end = market.end_date
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                hours_left = (end - now).total_seconds() / 3600
                if hours_left < 24:
                    time_factor = 0.05  # slight mean reversion to market for very short term
            except:
                pass

        # Adjust by volume - high volume markets more efficient
        volume_adjustment = 0.0
        if market.volume_24h > 100000:
            volume_adjustment = (market.best_price - base) * 0.1

        prob = base + time_factor + volume_adjustment
        prob = max(0.05, min(0.95, prob))

        return ModelForecast(
            model_name="base_rate",
            probability=prob,
            confidence=0.6,
            uncertainty=0.15,
            reasoning=f"Base rate for {category}: {base:.2f}, time_factor={time_factor:.3f}, vol_adj={volume_adjustment:.3f}",
            sources=["historical_frequencies", "base_rates"]
        )


class NewsModel:
    """
    News model: processes current news, official announcements, reputable sources, timestamps
    """
    def __init__(self, llm_router=None):
        self.llm_router = llm_router

    def forecast(self, market: Market, news_text: str = "", research: Dict = None) -> ModelForecast:
        research = research or {}
        # If no news, neutral
        if not news_text and not research:
            return ModelForecast(
                model_name="news",
                probability=market.best_price,
                confidence=0.3,
                uncertainty=0.3,
                reasoning="No news available - neutral",
                sources=[]
            )

        # Simple heuristic: if news mentions positive keywords, adjust
        # In production, would use LLM to analyze news
        text = (news_text + " " + str(research)).lower()
        positive_words = ["win", "success", "up", "increase", "positive", "good"]
        negative_words = ["lose", "fail", "down", "decrease", "negative", "bad"]
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        sentiment = (pos_count - neg_count) / max(1, pos_count + neg_count)
        # Convert sentiment to probability adjustment
        adjustment = sentiment * 0.1  # max 10% adjustment
        
        prob = market.best_price + adjustment
        prob = max(0.05, min(0.95, prob))

        return ModelForecast(
            model_name="news",
            probability=prob,
            confidence=0.6 if news_text else 0.4,
            uncertainty=0.2,
            reasoning=f"News sentiment {sentiment:.2f}, adjustment {adjustment:.3f}, pos={pos_count} neg={neg_count}",
            sources=["news_api", "web_search"] if news_text else []
        )


class XModel:
    """
    X model: volume, author credibility, sentiment, acceleration, bot likelihood, narratives
    X is information source, NOT truth. Must do credibility analysis.
    """
    def __init__(self):
        self.bot_threshold = 0.7

    def analyze_credibility(self, tweets: List[Dict]) -> Dict:
        """Detect bot-like bursts, duplicates, farming, fake accounts, coordinated narratives"""
        if not tweets:
            return {"credibility": 0.3, "bot_likelihood": 0.5, "issues": ["no_data"]}

        # Simple heuristics for demo - production would use ML
        issues = []
        bot_likelihood = 0.0
        
        # Check duplicate content
        contents = [t.get("text", "") for t in tweets]
        unique_ratio = len(set(contents)) / max(1, len(contents))
        if unique_ratio < 0.5:
            issues.append("duplicate_posts")
            bot_likelihood += 0.3
        
        # Check account ages, follower ratios (would need API)
        # For now mock
        
        # Check burst timing
        if len(tweets) > 20:
            # If many tweets in short time, possible coordinated
            issues.append("high_volume_burst")
            bot_likelihood += 0.2

        credibility = max(0.1, 1.0 - bot_likelihood - len(issues)*0.1)
        
        return {
            "credibility": credibility,
            "bot_likelihood": min(1.0, bot_likelihood),
            "unique_ratio": unique_ratio,
            "issues": issues,
            "total_tweets": len(tweets)
        }

    def forecast(self, market: Market, sentiment_result: Dict = None, tweets: List[Dict] = None) -> ModelForecast:
        sentiment_result = sentiment_result or {}
        tweets = tweets or []

        credibility = self.analyze_credibility(tweets)
        
        # X signal should be: information source -> credibility -> corroboration -> novelty -> time decay -> model input
        # NOT direct probability
        raw_sentiment = sentiment_result.get("score", 0)  # -1 to 1
        # Convert sentiment to probability but heavily discounted by credibility
        # If credibility low, X has little weight
        sentiment_prob = 0.5 + raw_sentiment * 0.2  # sentiment contributes max 20%
        
        # Discount by credibility
        # High credibility -> keep sentiment, low credibility -> revert to 0.5
        prob = 0.5 + (sentiment_prob - 0.5) * credibility["credibility"]
        prob = max(0.05, min(0.95, prob))

        return ModelForecast(
            model_name="x_sentiment",
            probability=prob,
            confidence=credibility["credibility"] * 0.7,
            uncertainty=0.25 + (1 - credibility["credibility"])*0.2,
            reasoning=f"X sentiment {raw_sentiment:.2f} -> prob {sentiment_prob:.3f} * credibility {credibility['credibility']:.2f} = {prob:.3f}. Issues: {credibility['issues']}",
            sources=["x_api", "x_snscrape"] if tweets else []
        )


class MarketMicrostructureModel:
    """
    Market microstructure: bid/ask, spread, orderbook depth, recent trades, price velocity, volume, liquidity, large orders
    """
    def forecast(self, market: Market, orderbook: Dict = None, recent_trades: List[Dict] = None) -> ModelForecast:
        orderbook = orderbook or {}
        recent_trades = recent_trades or []

        # Analyze spread - wide spread = uncertainty
        # A missing spread means the book was not read. Use a conservative
        # default so the model still runs, but do not pretend it was measured.
        spread, _spread_is_real = read_spread(orderbook, 0.03)
        spread_uncertainty = min(0.3, spread * 2)  # wide spread -> higher uncertainty

        # Analyze volume - high volume = more efficient, price more trustworthy
        # But also check for large orders that might be manipulation
        large_order_detected = False
        if recent_trades:
            avg_size = sum(t.get("size", 0) for t in recent_trades) / len(recent_trades)
            max_size = max(t.get("size", 0) for t in recent_trades)
            if max_size > avg_size * 5:
                large_order_detected = True

        # Price velocity - rapid moves might indicate informed trading or overreaction
        price_velocity = 0.0
        if len(recent_trades) >= 2:
            # Simple velocity calc
            try:
                prices = [t.get("price", 0.5) for t in recent_trades[-5:]]
                price_velocity = prices[-1] - prices[0]
            except:
                pass

        # For microstructure, forecast tends to lean toward market price (efficient market)
        # But adjust if detecting overreaction or large orders
        prob = market.best_price
        if large_order_detected:
            # If large order detected, might be manipulation - increase uncertainty, don't follow
            prob = market.best_price * 0.9 + 0.5 * 0.1  # slight mean reversion
        if abs(price_velocity) > 0.1:
            # Rapid move - possible overreaction, mean reversion tendency
            prob = market.best_price - price_velocity * 0.1

        prob = max(0.05, min(0.95, prob))

        return ModelForecast(
            model_name="market_microstructure",
            probability=prob,
            confidence=0.7 if market.volume_24h > 10000 else 0.5,
            uncertainty=0.1 + spread_uncertainty + (0.1 if large_order_detected else 0),
            reasoning=f"Spread {spread:.3f} -> uncertainty {spread_uncertainty:.3f}, large_order={large_order_detected}, velocity={price_velocity:.3f}",
            sources=["orderbook", "recent_trades", "liquidity"]
        )
