"""
PTAI Brain - Fair Value Builder
Combines:
- Market price
- X sentiment
- Web research
- LLM reasoning (LM Studio / Ollama / any local)
- Base rates

Goal: Build fair value estimate to flag mispricing >8%
Supports LM Studio as primary (user uses LM Studio)
"""
import re
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from loguru import logger

from ..config import get_settings
from ..markets.base import Market
from ..sentiment import SentimentResult

@dataclass
class FairValueResult:
    market_id: str
    question: str
    market_price: float
    fair_value: float
    edge: float
    confidence: float
    reasoning: str
    sentiment_score: Optional[float] = None
    sentiment_summary: Optional[str] = None
    should_trade: bool = False
    side: str = "YES"  # YES or NO
    raw: Dict = None
    llm_provider: str = "heuristic"

class Brain:
    def __init__(self, llm_config=None, llm_router=None):
        self.settings = get_settings()
        self.llm_config = llm_config or self.settings.to_llm_config()
        # Reuse the caller's router when one is given. Building a second one from
        # settings meant the fallback forecast could run against a different
        # provider, endpoint, model and timeout than the intelligence stack that
        # asked for it.
        self.llm_router = llm_router
        # Initialize LLM Router (supports LM Studio, Ollama, etc)
        if self.llm_router is not None:
            # Supplied by the caller: use it as-is rather than building a second
            # one from settings, which is how two parts of the system ended up
            # talking to different providers.
            # Never assume the supplied object's shape: it only has to be a
            # router, and logging must not be able to break construction.
            try:
                name = self.llm_router.get_provider_name()
            except Exception:
                name = type(self.llm_router).__name__
            logger.info(f"Brain reusing the supplied LLM router ({name})")
            return
        try:
            from ..llm.provider import LLMRouter
            model_name_lower = (self.llm_config.model + " " + getattr(self.settings, 'ollama_model', '') + " " + getattr(self.settings, 'lm_studio_model', '')).lower()
            is_r1_model = "r1" in model_name_lower or "distill" in model_name_lower or "reasoning" in model_name_lower
            max_toks = 500 if is_r1_model else self.llm_config.max_tokens
            if is_r1_model:
                logger.warning(f"Detected R1 reasoning model ({self.llm_config.model}) - setting max_tokens=500 for faster trading (was {self.llm_config.max_tokens}). R1 thinks 1000+ tokens internally, so 6-8 min per market unavoidable. For 10-min cycle, use non-R1 qwen/qwen3-32b or set MAX_DEEP_ANALYZE=5")

            self.llm_router = LLMRouter(
                preferred=self.llm_config.provider,
                ollama_host=self.llm_config.ollama_host,
                lm_studio_host=self.llm_config.lm_studio_host,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=max_toks
            )
            logger.info(f"Brain LLM Router: provider={self.llm_router.get_provider_name()} model={self.llm_config.model} lm_studio={self.llm_config.lm_studio_host} ollama={self.llm_config.ollama_host} max_tokens={max_toks} is_r1={is_r1_model}")
        except Exception as e:
            logger.warning(f"LLM Router init failed: {e}, using heuristic")
            self.llm_router = None

    def _build_prompt(self, market: Market, sentiment: Optional[SentimentResult], research_text: str = "") -> tuple:
        """Build LLM prompt for fair value estimation, returns (system, user)"""
        sentiment_block = ""
        if sentiment:
            sentiment_block = f"""
X/TWITTER SENTIMENT (last 24h):
- Score: {sentiment.score:.2f} (-1 bearish NO, +1 bullish YES)
- Bullish: {sentiment.bullish_pct:.0%}, Bearish: {sentiment.bearish_pct:.0%}, Neutral: {sentiment.neutral_pct:.0%}
- Summary: {sentiment.summary}
- Key phrases: {', '.join(sentiment.key_phrases[:5])}
- Tweet count: {sentiment.tweet_count}
- Confidence: {sentiment.confidence:.2f}
"""

        research_block = ""
        if research_text:
            research_block = f"""
WEB RESEARCH (local browser/terminal):
{research_text[:2000]}
"""

        system_prompt = """You are an expert prediction market superforecaster. Be extremely concise. No chain-of-thought, no <think> tag, just direct JSON. You must earn money or shutdown. Calibrated, base-rate aware. SPEED CRITICAL: Respond in <100 tokens JSON only."""

        user_prompt = f"""MARKET: {market.question[:200]}
YES price: {market.yes_price:.3f} ({market.yes_price:.1%}) NO: {market.no_price:.3f}
Vol24h ${market.volume_24h:,.0f} Liq ${market.liquidity:,.0f} End {market.end_date}
Desc: {market.description[:250]}

{sentiment_block}
{research_block}

TASK: Fair value 0.01-0.99. Edge=fair-market. Trade if |edge|>=8% and conf>=0.60.
BE CONCISE - NO <think> reasoning tags, direct JSON only (<80 tokens).
Respond ONLY JSON:
{{"fair_value":0.65,"edge":0.15,"confidence":0.72,"side":"YES","should_trade":true,"reasoning":"Base 60%... X bullish... Market 50% low because..."}}
Rules: fair 0.01-0.99, side YES if fair>market else NO, calibrated, if unsure fair~market low conf.
"""

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """Call LLM via router (supports LM Studio, Ollama)"""
        if not self.llm_router:
            return None

        try:
            response = self.llm_router.chat(prompt=user_prompt, system=system_prompt)
            if response and response.parsed_json:
                logger.info(f"LLM {response.provider} parsed JSON: fair={response.parsed_json.get('fair_value')} edge={response.parsed_json.get('edge')}")
                return response.parsed_json
            elif response:
                # Try to extract JSON from content
                import re, json
                content = response.content
                # Try to find JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                logger.warning(f"LLM {response.provider} returned no JSON: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _fallback_heuristic(self, market: Market, sentiment: Optional[SentimentResult]) -> FairValueResult:
        """
        What to say about a market when the LLM did not answer.

        This used to invent a fair value from `random.uniform`. That number was
        then labelled `llm_reasoning`, given the largest weight in the ensemble
        (0.30), and handed a confidence that INCREASED with the size of the random
        edge - so noise was not merely admitted, it was rewarded. On a machine
        with no local model, as the startup log reports, the single largest
        component of every forecast was a random draw, and the resulting trades
        were recorded as evidence that the venue and the strategy worked.

        The honest answer to "no model answered" is that the agent has no opinion.
        Fair value is the market's own price, the edge is zero, and it does not
        trade. That costs nothing and it does not manufacture learning data from
        nothing.

        Sentiment is kept, because it IS evidence: when a real sentiment result
        exists, the estimate moves with it and the confidence reflects the
        sentiment's own confidence.
        """
        market_price = market.yes_price

        if sentiment and sentiment.confidence > 0.3:
            sentiment_adjustment = sentiment.score * 0.15 * sentiment.confidence
            fair_value = market_price + sentiment_adjustment
            confidence = 0.4 + sentiment.confidence * 0.3
            source = "sentiment only (no LLM)"
        else:
            # No LLM and no sentiment: no basis for an opinion. Saying so is the
            # correct output, not a guess in its place.
            fair_value = market_price
            confidence = 0.0
            source = "no model and no sentiment"

        fair_value = max(0.05, min(0.95, fair_value))
        edge = fair_value - market_price

        should_trade = (confidence > 0
                        and abs(edge) >= self.settings.min_edge_pct
                        and confidence >= 0.55)
        side = "YES" if edge > 0 else "NO"
        reasoning = (f"{source}: market {market_price:.1%}, sentiment "
                     f"{sentiment.score if sentiment else 0:.2f} -> fair "
                     f"{fair_value:.1%}, edge {edge:.1%}. No LLM opinion was "
                     f"produced, so this is not an LLM forecast.")

        return FairValueResult(
            market_id=market.id,
            question=market.question,
            market_price=market_price,
            fair_value=fair_value,
            edge=edge,
            confidence=confidence,
            reasoning=reasoning,
            sentiment_score=sentiment.score if sentiment else None,
            sentiment_summary=sentiment.summary if sentiment else None,
            should_trade=should_trade,
            side=side,
            raw={"method": "heuristic"},
            llm_provider="heuristic"
        )

    def estimate_fair_value(self, market: Market, sentiment: Optional[SentimentResult] = None, research_text: str = "") -> FairValueResult:
        """Main entry - estimate fair value"""
        system_prompt, user_prompt = self._build_prompt(market, sentiment, research_text)

        llm_result = self._call_llm(system_prompt, user_prompt)

        if llm_result:
            try:
                fair_value = float(llm_result.get("fair_value", market.yes_price))
                fair_value = max(0.01, min(0.99, fair_value))
                market_price = market.yes_price
                edge = fair_value - market_price
                confidence = float(llm_result.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                side = llm_result.get("side", "YES" if edge > 0 else "NO")
                should_trade = bool(llm_result.get("should_trade", False))
                reasoning = llm_result.get("reasoning", "No reasoning")

                if abs(edge) < self.settings.min_edge_pct:
                    should_trade = False
                if confidence < 0.55:
                    should_trade = False

                provider_name = self.llm_router.get_provider_name() if self.llm_router else "unknown"

                result = FairValueResult(
                    market_id=market.id,
                    question=market.question,
                    market_price=market_price,
                    fair_value=fair_value,
                    edge=edge,
                    confidence=confidence,
                    reasoning=reasoning,
                    sentiment_score=sentiment.score if sentiment else None,
                    sentiment_summary=sentiment.summary if sentiment else None,
                    should_trade=should_trade,
                    side=side,
                    raw=llm_result,
                    llm_provider=provider_name
                )
                logger.info(f"Brain [{provider_name}]: {market.question[:60]} | Market {market_price:.1%} Fair {fair_value:.1%} Edge {edge:.1%} Conf {confidence:.2f} Trade? {should_trade}")
                return result

            except Exception as e:
                logger.error(f"Failed to parse LLM result {llm_result}: {e}")

        logger.warning(f"Using fallback heuristic for {market.id}")
        return self._fallback_heuristic(market, sentiment)

    def batch_estimate(self, markets: List[Market], sentiments: Dict[str, SentimentResult], research_dict: Dict[str, str] = None) -> List[FairValueResult]:
        results = []
        research_dict = research_dict or {}
        for market in markets:
            sentiment = sentiments.get(market.question) or sentiments.get(market.id)
            research = research_dict.get(market.id, "")
            res = self.estimate_fair_value(market, sentiment, research)
            results.append(res)
        return results
