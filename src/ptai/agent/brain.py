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
from dataclasses import dataclass, field

from loguru import logger

from ..config import get_settings
from ..markets.base import Market
from ..sentiment import SentimentResult

class AnswerRepetitionDetector:
    """
    Catches a model that answers the same thing to every question.

    Removing the numeric example from the prompt is the cure; this is the
    detector that says whether the cure worked. It keeps the last few answers
    and, before any answer is used, asks: is this value just repeating while the
    markets it is supposedly forecasting are materially different?

    If so the answer is not a forecast. It is treated as no opinion at all -
    confidence zero, no weight in the ensemble, no trade - and the reason is
    written on the component so the operator sees WHY the LLM contributed
    nothing rather than reading a smaller edge and guessing.

    Conservative on purpose: it needs `min_repeats` answers agreeing to within
    `tolerance` while their market prices span at least `market_spread`. A model
    that legitimately agrees with the market price on similar markets is not
    flagged, because those markets are not materially different.
    """

    def __init__(self, window: int = 6, min_repeats: int = 4,
                 tolerance: float = 0.005, market_spread: float = 0.10):
        self.window = window
        self.min_repeats = min_repeats
        self.tolerance = tolerance
        self.market_spread = market_spread
        self.answers: List[Dict[str, float]] = []
        self.flagged: List[Dict[str, Any]] = []

    def observe(self, market_id: str, market_price: float,
                fair_value: float) -> Optional[str]:
        """
        Record one answer. Returns a reason string when it looks anchored.
        """
        self.answers.append({"market_id": market_id,
                             "market_price": float(market_price),
                             "fair_value": float(fair_value)})
        if len(self.answers) > self.window:
            self.answers = self.answers[-self.window:]

        close = [a for a in self.answers
                 if abs(a["fair_value"] - fair_value) <= self.tolerance]
        if len(close) < self.min_repeats:
            return None
        prices = [a["market_price"] for a in close]
        spread = max(prices) - min(prices)
        if spread < self.market_spread:
            # Same answer on markets that ARE alike - that is agreement with the
            # market, not anchoring, and it must not be punished.
            return None
        detail = (f"the model answered {fair_value:.3f} on {len(close)} of its "
                  f"last {len(self.answers)} forecasts while those markets were "
                  f"priced {min(prices):.1%}-{max(prices):.1%} "
                  f"(spread {spread:.1%}). A repeated value across materially "
                  f"different markets is the prompt being echoed, not a "
                  f"forecast, so this answer carries no weight")
        self.flagged.append({"market_id": market_id, "fair_value": fair_value,
                             "repeats": len(close), "market_spread": round(spread, 4),
                             "reason": detail})
        return detail

    def report(self) -> Dict[str, Any]:
        """What the operator/dashboard reads: how often, and on what."""
        return {
            "answers_seen": len(self.answers),
            "flagged_count": len(self.flagged),
            "window": self.window,
            "min_repeats": self.min_repeats,
            "tolerance": self.tolerance,
            "market_spread_required": self.market_spread,
            "last": self.flagged[-1] if self.flagged else None,
            "how": ("an identical fair value across markets that are priced "
                    "differently is treated as no opinion: confidence 0, no "
                    "ensemble weight, no trade"),
        }


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
    # WHICH model produced this answer. `llm_provider` was the provider CLASS
    # ("LMStudioProvider"), and the model id was nowhere on the result - so a
    # trade's record could not say whether a 27B model or a rule of thumb priced
    # it, and neither the log nor the page could name it.
    llm_model: str = ""

@dataclass
class _SentimentView:
    """
    The fields `_build_prompt` reads, from whatever shape the caller had.

    Missing fields get honest neutral defaults rather than fabricated ones: a
    sentiment dict with a score and nothing else must not invent a tweet count
    or a confidence that makes the sentiment look corroborated.
    """
    score: float = 0.0
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    neutral_pct: float = 0.0
    summary: str = ""
    key_phrases: List[str] = field(default_factory=list)
    tweet_count: int = 0
    confidence: float = 0.0


def _as_sentiment(sentiment):
    """
    Accept either the structured object or the dict V3 puts in the context.

    One value, two shapes: `EnsembleForecaster` passes `context["sentiment"]`,
    which V3 builds as a dict, while `_build_prompt` reads attributes. Converting
    at the boundary is the fix; changing one caller would have left the next one
    free to reintroduce it.
    """
    if sentiment is None:
        return None
    if isinstance(sentiment, dict):
        if not sentiment:
            return None
        phrases = sentiment.get("key_phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        def _num(key, default=0.0):
            try:
                return float(sentiment.get(key, default) or default)
            except (TypeError, ValueError):
                return default
        return _SentimentView(
            score=_num("score"),
            bullish_pct=_num("bullish_pct"),
            bearish_pct=_num("bearish_pct"),
            neutral_pct=_num("neutral_pct"),
            summary=str(sentiment.get("summary") or
                       sentiment.get("reasoning") or ""),
            key_phrases=[str(p) for p in phrases][:10],
            tweet_count=int(_num("tweet_count")),
            confidence=_num("confidence", _num("credibility")),
        )
    return sentiment


class Brain:
    def __init__(self, llm_config=None, llm_router=None, answer_repetition=None):
        self.settings = get_settings()
        self.llm_config = llm_config or self.settings.to_llm_config()
        # Reuse the caller's router when one is given. Building a second one from
        # settings meant the fallback forecast could run against a different
        # provider, endpoint, model and timeout than the intelligence stack that
        # asked for it.
        self.llm_router = llm_router
        # Every answer this process produces goes through here. Removing the
        # numeric example from the prompt is the fix for anchoring; this is how
        # we find out whether it worked, per process, from the answers themselves.
        # A caller that already has one (the ensemble) passes it in, because a
        # detector per market can never see the repetition it exists to catch.
        self.answer_repetition = answer_repetition or AnswerRepetitionDetector()
        # Set by _call_llm on every call: which model, and what went wrong when
        # nothing came back.
        self._last_llm_model = ""
        self._last_llm_problem = ""
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
            # DEBUG, not INFO: the ensemble builds a Brain per market, so this
            # printed once per market and buried the one line that matters -
            # the router's LOCAL MODEL line, which names the model.
            logger.debug(f"Brain reusing the supplied LLM router ({name})")
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
                max_tokens=max_toks,
                # Bounded: a slow model must produce NO forecast rather than
                # hold the cycle. See config.llm_timeout_seconds.
                timeout_seconds=getattr(self.settings, "llm_timeout_seconds", 180.0),
                # Whether the model may reason before answering. Off means the
                # prompt may still say "no chain-of-thought"; on means the
                # request asks LM Studio to enable it (PTAI_LLM_THINKING).
                thinking=self.llm_thinking(),
            )
            logger.info(f"Brain LLM Router: provider={self.llm_router.get_provider_name()} model={self.llm_config.model} lm_studio={self.llm_config.lm_studio_host} ollama={self.llm_config.ollama_host} max_tokens={max_toks} is_r1={is_r1_model} thinking={self.llm_thinking()}")
        except Exception as e:
            logger.warning(f"LLM Router init failed: {e}, using heuristic")
            self.llm_router = None

    def _build_prompt(self, market: Market, sentiment: Optional[SentimentResult], research_text: str = "") -> tuple:
        """Build LLM prompt for fair value estimation, returns (system, user)"""
        # V3 hands the ensemble a DICT; this function expects an object with
        # attributes. Both were true at the same time, so on the real V3 path the
        # LLM branch raised AttributeError('dict' object has no attribute
        # 'score') before it ever reached the provider - and the caller's
        # `except` logged it as "LLM forecast failed", which is why the component
        # looked wired and produced nothing.
        sentiment = _as_sentiment(sentiment)
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

        # THINKING, SAID OUT LOUD.
        #
        # The old system prompt forbade chain-of-thought unconditionally. That
        # was a speed decision from a log where one market took ~9 minutes, and
        # it applied even when the operator wanted the model to reason. Now the
        # instruction follows the ONE setting (`llm_thinking`): off = the old
        # speed wording; on = the model is told to reason and only the final
        # JSON answer is read back.
        if self.llm_thinking():
            system_prompt = """You are an expert prediction market superforecaster. Reason through the market carefully first, then answer with direct JSON only. Put your reasoning in your own scratch space and end with the JSON object the user asks for. You must earn money or shutdown. Calibrated, base-rate aware."""
        else:
            system_prompt = """You are an expert prediction market superforecaster. Be extremely concise. No chain-of-thought, no <think> tag, just direct JSON. You must earn money or shutdown. Calibrated, base-rate aware. SPEED CRITICAL: Respond in <100 tokens JSON only."""

        # THE EXAMPLE USED TO CARRY THE ANSWER.
        #
        # It read `{"fair_value":0.65,"edge":0.15,"confidence":0.72,...}` immediately
        # before the model produced its own answer. On the operator's 2026-09-27
        # log, seventeen forecasts across markets priced 15%-60% came back
        # 0.65 / 0.72 / edge 0.15 - the example, verbatim, every time. A model
        # shown one worked example and asked for "the same shape" copies the
        # numbers in the example, so the ensemble's largest-weight component was
        # reporting the prompt back at us and the 0.15 "edge" was a constant.
        #
        # The schema is still given (the parser needs a shape), but the numbers
        # are placeholders the model cannot mistake for an answer, and the
        # instruction says so explicitly.
        user_prompt = f"""MARKET: {market.question[:200]}
YES price: {market.yes_price:.3f} ({market.yes_price:.1%}) NO: {market.no_price:.3f}
Vol24h ${market.volume_24h:,.0f} Liq ${market.liquidity:,.0f} End {market.end_date}
Desc: {market.description[:250]}

{sentiment_block}
{research_block}

TASK: your own probability that YES resolves, 0.01-0.99. Edge = fair_value - YES price.
Do the work yourself: the price, the evidence above, the base rate for this kind of
question. Then answer.
- If the evidence does not move you away from the YES price, answer with the YES
  price and a low confidence. That is a correct answer, not a failure.
- Do not reuse a probability you produced for another market: each market's answer
  has to come from that market's own evidence.
- The values in the schema below are PLACEHOLDERS showing the shape only. They are
  not answers, not a target, and not a default.
BE CONCISE - NO <think> reasoning tags, direct JSON only (<90 tokens).
Respond ONLY JSON, with your own numbers:
{{"fair_value":<your probability>,"edge":<fair minus the YES price>,"confidence":<your confidence>,"side":"YES"|"NO","basis":"<base_rate|news|research|sentiment|market>","should_trade":<true|false>,"reasoning":"<one line, name the evidence you used>"}}
Rules: fair 0.01-0.99; side YES if fair>YES price else NO; if unsure, fair~YES price and low confidence.
"""

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[Dict]:
        """
        Call the local model through the router, and say WHICH model answered.

        `self._last_llm_model` and `self._last_llm_problem` are set here so the
        caller can name the model in its own line and, when no answer came back,
        say why - "Using fallback heuristic" with no reason was how a whole
        cycle of model timeouts could read like a working heuristic system.
        """
        self._last_llm_model = ""
        self._last_llm_problem = ""
        if not self.llm_router:
            self._last_llm_problem = "no model router is configured for this process"
            return None

        try:
            response = self.llm_router.chat(prompt=user_prompt, system=system_prompt)
            if response is not None:
                self._last_llm_model = getattr(response, "model", "") or ""
            if response and response.parsed_json:
                logger.info(
                    f"LLM {response.provider} ['{self._last_llm_model}'] parsed "
                    f"JSON: fair={response.parsed_json.get('fair_value')} "
                    f"edge={response.parsed_json.get('edge')}")
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
                logger.warning(
                    f"LLM {response.provider} ['{self._last_llm_model}'] returned "
                    f"no JSON: {content[:500]}")
                self._last_llm_problem = (
                    f"'{self._last_llm_model}' answered, but its answer had no "
                    f"usable JSON")
            else:
                self._last_llm_problem = self._model_failure_reason()
            return None
        except Exception as e:
            self._last_llm_problem = f"the model call failed ({type(e).__name__}: {e})"
            logger.error(f"LLM call failed: {type(e).__name__}: {e}")
            return None

    def _model_failure_reason(self) -> str:
        """Why no model answer arrived - with the model id and the last error."""
        usage = {}
        try:
            usage = self.llm_router.usage_report()
        except Exception:  # noqa: BLE001 - a router that cannot report still has to be described
            usage = {}
        model = (usage.get("last_model") or self._last_llm_model
                 or (self.llm_router.active_model if hasattr(self.llm_router, "active_model") else "")
                 or "no model")
        error = usage.get("last_error") or ""
        if not getattr(self, "llm_router", None) or not hasattr(self.llm_router, "is_available"):
            return f"'{model}' produced no answer"
        try:
            available = bool(self.llm_router.is_available())
        except Exception:  # noqa: BLE001
            available = False
        if not available:
            # Name WHERE the server is not answering: "no model" plus the
            # endpoint is the fact the operator can act on (start the server),
            # and the verdict sentence is left for the startup line.
            hosts = []
            for attr in ("lm_studio_host", "ollama_host"):
                host = getattr(self.llm_router, attr, "")
                if host:
                    hosts.append(str(host))
            where = (" / ".join(hosts) if hosts
                     else (self.llm_router.active_model_reason
                           if hasattr(self.llm_router, "active_model_reason")
                           else "no endpoint recorded"))
            which = f"'{model}'" if model and model != "no model" else "no model"
            return (f"no local model server is answering at {where}, so {which} "
                    f"was called; every market is priced by the rule-based fallback")
        if error:
            return f"'{model}' gave no answer ({error})"
        return f"'{model}' gave no answer"

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
            # WHY there is no LLM opinion travels with the result. The cycle's
            # model line reads it, so "no answer was recorded" is followed by
            # the router's own words instead of by silence.
            raw={"method": "heuristic",
                 "llm_problem": (self._last_llm_problem
                                  or "the local model gave no usable answer"),
                 "llm_asked": True},
            llm_provider="heuristic"
        )

    def estimate_fair_value(self, market: Market, sentiment: Optional[SentimentResult] = None, research_text: str = "") -> FairValueResult:
        """
        Main entry - estimate fair value.

        `sentiment` may be the structured object or the dict V3 puts in the
        context. It is normalised HERE, once, because every consumer below reads
        attributes: `_build_prompt` formats them and `_fallback_heuristic` reads
        `sentiment.confidence`. Normalising inside `_build_prompt` alone fixed
        the prompt and then failed on the next reader, which is the same
        one-shape-assumption bug moved one step along.
        """
        sentiment = _as_sentiment(sentiment)
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

                # Is this answer a forecast, or the model echoing itself?
                anchored = self.answer_repetition.observe(
                    market.id, market_price, fair_value)
                if anchored:
                    logger.warning(
                        f"Brain [{market.id}]: LLM answer REPEATED across "
                        f"different markets - {anchored}. It is recorded as no "
                        f"opinion: confidence 0, no ensemble weight, no trade.")
                    confidence = 0.0
                    should_trade = False
                    reasoning = f"ANCHORED ANSWER (no weight): {anchored} | {reasoning}"

                provider_name = self.llm_router.get_provider_name() if self.llm_router else "unknown"
                model_name = self._last_llm_model or self.llm_provider_model()

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
                    raw={**llm_result,
                         "llm_problem": "",
                         "llm_asked": True,
                         "anchored": bool(anchored),
                         "anchoring_reason": anchored or "",
                         "basis": llm_result.get("basis", "")},
                    llm_provider=provider_name,
                    llm_model=model_name,
                )
                logger.info(
                    f"Brain [{provider_name}/{model_name or '?'}]: {market.question[:60]} | "
                    f"Market {market_price:.1%} Fair {fair_value:.1%} "
                    f"Edge {edge:+.1%} Conf {confidence:.2f} "
                    f"Trade? {should_trade}"
                    + (f" | basis {llm_result.get('basis')}"
                       if llm_result.get("basis") else "")
                    + (" | ANCHORED - no weight" if anchored else ""))
                return result

            except Exception as e:
                logger.error(f"Failed to parse LLM result {llm_result}: {e}")

        why = self._last_llm_problem or "the local model gave no usable answer"
        logger.warning(f"Using fallback heuristic for {market.id}: {why}")
        return self._fallback_heuristic(market, sentiment)

    def llm_thinking(self) -> bool:
        """Whether the local model is allowed to reason before it answers.

        Read from the LLM config first (the CLI and the console both build it
        from settings) and from the settings object second, so every path that
        reaches the Brain asks the same question and gets the same answer.
        """
        value = getattr(self.llm_config, "thinking", None)
        if value is None:
            value = getattr(self.settings, "llm_thinking", False)
        return bool(value)

    def llm_provider_model(self) -> str:
        """
        The model id this process will call, from the router if it can say.

        The startup lines already name it; this is for a result that has to
        carry it even when the answer did not come back.
        """
        try:
            model = getattr(self.llm_router, "active_model", None)
            if model:
                return str(model)
            provider = getattr(self.llm_router, "provider", None)
            return str(getattr(provider, "model", "") or "")
        except Exception:  # noqa: BLE001 - naming the model must never raise
            return ""

    def anchoring_report(self) -> Dict[str, Any]:
        """How many answers this process refused as repetitions, and why."""
        return self.answer_repetition.report()

    def batch_estimate(self, markets: List[Market], sentiments: Dict[str, SentimentResult], research_dict: Dict[str, str] = None) -> List[FairValueResult]:
        results = []
        research_dict = research_dict or {}
        for market in markets:
            sentiment = sentiments.get(market.question) or sentiments.get(market.id)
            research = research_dict.get(market.id, "")
            res = self.estimate_fair_value(market, sentiment, research)
            results.append(res)
        return results
