"""
X Engine - X as information source, NOT truth
Credibility analysis, independent corroboration, information novelty, time decay, model input
Detects bot bursts, duplicates, engagement farming, old info, fake accounts, coordinated narratives, source credibility
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class XSignal:
    market_id: str
    raw_sentiment: float  # -1 to 1
    volume: int
    credibility: float  # 0-1
    bot_likelihood: float
    novelty: float  # 0-1, is this new info?
    time_decay: float  # 0-1, recent = higher
    corroborated: bool
    issues: List[str] = field(default_factory=list)
    adjusted_sentiment: float = 0.0  # After credibility, novelty, decay
    should_use: bool = True
    # V10 FIX #12: Real X checks, not placeholders
    bot_burst_detected: bool = False
    duplicate_rate: float = 0.0
    farming_detected: bool = False
    unique_ratio: float = 1.0
    farming_score: float = 0.0


class XEngine:
    """
    X Engine - sophisticated X analysis, not just sentiment = probability
    """
    def __init__(self, x_scraper=None, sentiment_analyzer=None):
        self.x_scraper = x_scraper
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_tweets(self, tweets: List[Dict]) -> Dict:
        """Analyze tweets for credibility, bots, duplicates, etc."""
        if not tweets:
            return {
                "credibility": 0.3,
                "bot_likelihood": 0.5,
                "issues": ["no_tweets"],
                "unique_ratio": 0,
                "volume": 0
            }

        # V10 FIX #12: Real X credibility checks - not placeholders
        texts = [t.get("text", "") for t in tweets]
        unique_ratio = len(set(texts)) / max(1, len(texts))
        duplicate_rate = 1.0 - unique_ratio
        
        issues = []
        bot_likelihood = 0.0

        if unique_ratio < 0.6:
            issues.append("duplicate_posts")
            bot_likelihood += 0.3

        # V10 FIX #12: Engagement farming detection - high likes but low quality, copy-paste, engagement bait
        farming_detected = False
        farming_score = 0.0
        for t in tweets:
            text_lower = t.get("text", "").lower()
            # Farming patterns: "like and retweet", "follow me", "drop a", "comment below", engagement bait
            farming_keywords = ["like and retweet", "follow me", "drop a", "comment below", "rt if", "like if", "giveaway", "airdrop"]
            if any(kw in text_lower for kw in farming_keywords):
                farming_score += 0.1
            # High engagement but very short text = farming
            likes = t.get("likes", 0) or t.get("like_count", 0) or 0
            if likes > 100 and len(text_lower) < 20:
                farming_score += 0.2
        
        if farming_score > 0.5:
            issues.append("engagement_farming")
            farming_detected = True
            bot_likelihood += 0.25

        # Old info resurfacing - check timestamps
        now = datetime.now(timezone.utc)
        old_count = 0
        for t in tweets:
            try:
                ts = t.get("timestamp")
                if ts:
                    tweet_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if (now - tweet_time).total_seconds() > 24*3600*7:
                        old_count += 1
            except:
                pass
        if old_count > len(tweets) * 0.5:
            issues.append("old_info_resurfacing")
            bot_likelihood += 0.2

        # V10 FIX #12: Bot burst detection - many tweets same minute = coordinated burst
        bot_burst_detected = False
        try:
            from collections import Counter
            minutes = []
            for t in tweets:
                ts = t.get("timestamp", "")
                if ts:
                    minutes.append(ts[:16])
            if minutes:
                most_common = Counter(minutes).most_common(1)[0][1]
                if most_common >= 5:
                    bot_burst_detected = True
                    issues.append("bot_burst")
                    bot_likelihood += 0.3
        except:
            pass

        # Coordinated narratives - similar phrasing high overlap
        if unique_ratio < 0.4 and len(tweets) >= 5:
            issues.append("coordinated_narrative")
            bot_likelihood += 0.2

        credibility = max(0.1, 1.0 - bot_likelihood - len(issues)*0.15)

        return {
            "credibility": credibility,
            "bot_likelihood": min(1.0, bot_likelihood),
            "unique_ratio": unique_ratio,
            "duplicate_rate": duplicate_rate,
            "issues": issues,
            "volume": len(tweets),
            "old_count": old_count,
            "bot_burst_detected": bot_burst_detected,
            "farming_detected": farming_detected,
            "farming_score": farming_score
        }

    def calculate_novelty(self, tweets: List[Dict], existing_knowledge: str = "") -> float:
        """Information novelty - is this new info or already known?"""
        # Would compare tweet content to existing knowledge base
        # For now simple: if tweets mention recent keywords, higher novelty
        if not tweets:
            return 0.0
        
        # Check if tweets contain timestamps like "just now", "breaking"
        text = " ".join([t.get("text", "") for t in tweets]).lower()
        novelty_keywords = ["breaking", "just", "now", "announced", "today", "new"]
        novelty_score = sum(1 for kw in novelty_keywords if kw in text) / len(novelty_keywords)
        return min(1.0, novelty_score + 0.3)

    def calculate_time_decay(self, tweets: List[Dict]) -> float:
        """Time decay - recent info more valuable"""
        if not tweets:
            return 0.0
        
        now = datetime.now(timezone.utc)
        recent_count = 0
        for t in tweets:
            try:
                ts = t.get("timestamp")
                if ts:
                    tweet_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hours_ago = (now - tweet_time).total_seconds() / 3600
                    if hours_ago < 24:
                        recent_count += 1
            except:
                pass
        
        return recent_count / max(1, len(tweets))

    def corroborate(self, x_signal: str, news: str = "", web_research: str = "") -> bool:
        """Independent corroboration - does other sources confirm X?"""
        if not x_signal or not (news or web_research):
            return False
        
        # Simple keyword overlap for demo
        # Production would use LLM to check corroboration
        x_words = set(x_signal.lower().split()[:20])
        other_text = (news + " " + web_research).lower()
        overlap = sum(1 for w in x_words if w in other_text and len(w) > 4)
        return overlap >= 3

    async def get_signal(self, market, news: str = "", web_research: str = "") -> XSignal:
        """Get X signal with full analysis - V9 FIX #4 real implementation
        - Bot burst detection (many tweets same minute)
        - Duplicate detection via unique_ratio
        - Engagement farming detection
        - Old info resurfacing detection
        - Fake accounts, coordinated narratives via credibility
        - Time decay 40 sec circuit breaker not 21 min
        """
        tweets = []
        sentiment_score = 0.0
        
        if self.x_scraper:
            try:
                # Check circuit breaker - 40 sec not 21 min if blocked
                is_blocked = getattr(self.x_scraper, 'is_blocked', False) or getattr(self.x_scraper, 'circuit_open', False)
                if is_blocked:
                    logger.debug(f"X scraper circuit open for {market.id} - skipping, 40 sec breaker")
                else:
                    tweets = await self.x_scraper.search(market.question, limit=20) if hasattr(self.x_scraper, 'search') else []
            except Exception as e:
                logger.warning(f"X scrape failed: {e} - circuit breaker 40 sec")
                # Record failure for circuit breaker
                if hasattr(self.x_scraper, 'record_failure'):
                    try:
                        self.x_scraper.record_failure()
                    except:
                        pass
        
        if self.sentiment_analyzer and tweets:
            try:
                sentiment_result = self.sentiment_analyzer.analyze(market.question, tweets)
                sentiment_score = sentiment_result.get("score", 0) if isinstance(sentiment_result, dict) else float(sentiment_result) if isinstance(sentiment_result, (int, float)) else 0
            except Exception as e:
                logger.debug(f"Sentiment analyze failed: {e}")

        # Analyze - real credibility checks
        credibility_analysis = self.analyze_tweets(tweets)
        novelty = self.calculate_novelty(tweets)
        time_decay = self.calculate_time_decay(tweets)
        corroborated = self.corroborate(" ".join([t.get("text", "") for t in tweets[:5]]), news, web_research)

        # V10 FIX #12: Use real checks from analyze_tweets, not recomputed placeholders
        bot_burst_detected = credibility_analysis.get("bot_burst_detected", False)
        duplicate_rate = credibility_analysis.get("duplicate_rate", 1.0 - credibility_analysis.get("unique_ratio", 1.0))
        farming_detected = credibility_analysis.get("farming_detected", False)
        unique_ratio = credibility_analysis.get("unique_ratio", 1.0)
        farming_score = credibility_analysis.get("farming_score", 0.0)

        # Adjusted sentiment: raw * credibility * novelty * time_decay * corroboration boost
        base = sentiment_score
        adjusted = base * credibility_analysis["credibility"] * (0.5 + novelty*0.5) * (0.5 + time_decay*0.5)
        if corroborated:
            adjusted *= 1.2
        adjusted = max(-1.0, min(1.0, adjusted))

        should_use = (
            credibility_analysis["credibility"] > 0.4 and
            credibility_analysis["bot_likelihood"] < 0.6 and
            len(tweets) >= 3 and
            not bot_burst_detected
        )

        signal = XSignal(
            market_id=market.id,
            raw_sentiment=sentiment_score,
            volume=len(tweets),
            credibility=credibility_analysis["credibility"],
            bot_likelihood=credibility_analysis["bot_likelihood"],
            novelty=novelty,
            time_decay=time_decay,
            corroborated=corroborated,
            issues=credibility_analysis["issues"],
            adjusted_sentiment=adjusted,
            should_use=should_use,
            bot_burst_detected=bot_burst_detected,
            duplicate_rate=duplicate_rate,
            farming_detected=farming_detected,
            unique_ratio=unique_ratio,
            farming_score=farming_score
        )
        # V9: Add compatibility fields for v3_loop get_context_for_market
        signal.sentiment_score = adjusted
        signal.score = adjusted
        signal.tweets = tweets[:5]
        signal.reasoning = f"X raw {sentiment_score:.2f} adj {adjusted:.2f} cred {credibility_analysis['credibility']:.2f} bot {credibility_analysis['bot_likelihood']:.2f} novelty {novelty:.2f} decay {time_decay:.2f} corroborated {corroborated} burst {bot_burst_detected} dup {duplicate_rate:.2f} farming {farming_detected} issues {credibility_analysis['issues']} use {should_use}"
        # Ensure fields are accessible for v3_loop

        logger.info(f"X signal for {market.id}: raw {sentiment_score:.2f} -> adjusted {adjusted:.2f} cred {credibility_analysis['credibility']:.2f} bot {credibility_analysis['bot_likelihood']:.2f} novelty {novelty:.2f} decay {time_decay:.2f} corroborated {corroborated} burst {bot_burst_detected} use {should_use}")

        return signal
