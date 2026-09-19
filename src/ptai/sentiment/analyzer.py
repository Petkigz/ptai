"""
Sentiment Analyzer - Builds sentiment score from tweets + news
Fully local using LLM or simple heuristics
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from loguru import logger

from .x_scraper import Tweet
from ..config import get_settings

@dataclass
class SentimentResult:
    score: float  # -1 (very bearish) to +1 (very bullish), 0 neutral
    bullish_pct: float  # 0-1 proportion bullish
    bearish_pct: float
    neutral_pct: float
    summary: str
    key_phrases: List[str]
    tweet_count: int
    confidence: float  # 0-1
    raw_tweets: List[Dict] = None

class SentimentAnalyzer:
    def __init__(self, llm_config=None):
        self.settings = get_settings()
        self.llm_config = llm_config
        # Simple keyword lists for fallback heuristic
        self.bullish_keywords = [
            "bullish", "will happen", "likely", "probably", "yes", "confirm", "announce",
            "growth", "up", "increase", "win", "pass", "approve", "success", "good news",
            "optimistic", "positive", "strong", "momentum"
        ]
        self.bearish_keywords = [
            "bearish", "won't", "unlikely", "no", "fail", "reject", "down", "decrease",
            "lose", "crash", "bad news", "negative", "weak", "concern", "risk", "doubt",
            "delay", "cancel", "pessimistic"
        ]

    def _heuristic_score(self, tweets: List[Tweet]) -> Tuple[float, str]:
        """Simple keyword-based sentiment as fallback"""
        if not tweets:
            return 0.0, "No tweets found, neutral"

        bullish = 0
        bearish = 0
        neutral = 0

        for tweet in tweets:
            text_lower = tweet.text.lower()
            b_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
            be_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)

            if b_count > be_count:
                bullish += 1
            elif be_count > b_count:
                bearish += 1
            else:
                neutral += 1

        total = len(tweets)
        bullish_pct = bullish / total
        bearish_pct = bearish / total

        # Score from -1 to +1
        score = bullish_pct - bearish_pct

        summary = f"{bullish} bullish, {bearish} bearish, {neutral} neutral out of {total}"
        return score, summary

    def _extract_key_phrases(self, tweets: List[Tweet], top_n: int = 5) -> List[str]:
        """Extract most common phrases"""
        if not tweets:
            return []

        # Simple word frequency (exclude stopwords)
        stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "will", "be", "to", "of", "in", "on", "for", "with", "this", "that"}
        words = []
        for t in tweets:
            # Clean text
            clean = re.sub(r'http\S+|@\w+|#\w+', '', t.text.lower())
            clean = re.sub(r'[^\w\s]', '', clean)
            words.extend([w for w in clean.split() if len(w) > 3 and w not in stopwords])

        counter = Counter(words)
        return [w for w, c in counter.most_common(top_n)]

    def analyze_llm(self, tweets: List[Tweet], market_question: str) -> Optional[SentimentResult]:
        """Use local LLM (Ollama) for deeper sentiment analysis"""
        if not tweets:
            return SentimentResult(
                score=0.0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34,
                summary="No data", key_phrases=[], tweet_count=0, confidence=0.0
            )

        try:
            # Try Ollama
            import ollama
            settings = get_settings()
            host = settings.ollama_host
            model = settings.ollama_model

            # Prepare prompt
            tweet_texts = "\n".join([f"- {t.text[:200]}" for t in tweets[:20]])

            prompt = f"""You are a prediction market sentiment analyst. Analyze these tweets about:
MARKET: {market_question}

TWEETS:
{tweet_texts}

Task:
1. Is sentiment overall BULLISH (YES likely), BEARISH (NO likely), or NEUTRAL?
2. Give a score from -1.0 (very bearish, NO) to +1.0 (very bullish, YES), 0 neutral.
3. Provide bullish%, bearish%, neutral% that sum to 1.0
4. Confidence 0-1 based on volume and clarity
5. One sentence summary

Respond ONLY in JSON format:
{{
  "score": 0.2,
  "bullish_pct": 0.5,
  "bearish_pct": 0.2,
  "neutral_pct": 0.3,
  "confidence": 0.7,
  "summary": "Mostly bullish because..."
}}"""

            client = ollama.Client(host=host)
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}
            )
            content = response["message"]["content"]

            # Extract JSON
            import json
            # Find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                bullish_pct = float(data.get("bullish_pct", 0.33))
                bearish_pct = float(data.get("bearish_pct", 0.33))
                neutral_pct = float(data.get("neutral_pct", 0.34))
                confidence = float(data.get("confidence", 0.5))
                summary = data.get("summary", "LLM analysis")

                # Normalize
                total = bullish_pct + bearish_pct + neutral_pct
                if total > 0:
                    bullish_pct /= total
                    bearish_pct /= total
                    neutral_pct /= total

                key_phrases = self._extract_key_phrases(tweets)

                return SentimentResult(
                    score=score,
                    bullish_pct=bullish_pct,
                    bearish_pct=bearish_pct,
                    neutral_pct=neutral_pct,
                    summary=summary,
                    key_phrases=key_phrases,
                    tweet_count=len(tweets),
                    confidence=confidence,
                    raw_tweets=[t.to_dict() for t in tweets[:10]]
                )

        except ImportError:
            logger.warning("ollama not installed, using heuristic")
        except Exception as e:
            logger.warning(f"LLM sentiment failed: {e}, using heuristic")

        return None

    def analyze(self, tweets: List[Tweet], market_question: str) -> SentimentResult:
        """Main analysis - tries LLM first, fallback to heuristic"""
        # Try LLM
        llm_result = self.analyze_llm(tweets, market_question)
        if llm_result:
            logger.info(f"LLM sentiment for '{market_question[:50]}': score={llm_result.score:.2f}, conf={llm_result.confidence:.2f}")
            return llm_result

        # Fallback heuristic
        score, summary = self._heuristic_score(tweets)
        key_phrases = self._extract_key_phrases(tweets)

        total = len(tweets) if tweets else 1
        # Estimate pcts from score
        if score > 0:
            bullish_pct = 0.5 + score/2
            bearish_pct = 0.5 - score/2
            bullish_pct = min(0.9, bullish_pct)
            bearish_pct = max(0.05, bearish_pct)
        elif score < 0:
            bearish_pct = 0.5 + abs(score)/2
            bullish_pct = 0.5 - abs(score)/2
            bearish_pct = min(0.9, bearish_pct)
            bullish_pct = max(0.05, bullish_pct)
        else:
            bullish_pct = bearish_pct = 0.33

        neutral_pct = 1 - bullish_pct - bearish_pct
        confidence = min(0.6, len(tweets)/50)  # More tweets = more confidence, capped at 0.6 for heuristic

        return SentimentResult(
            score=score,
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            neutral_pct=neutral_pct,
            summary=summary,
            key_phrases=key_phrases,
            tweet_count=len(tweets),
            confidence=confidence,
            raw_tweets=[t.to_dict() for t in tweets[:10]]
        )
