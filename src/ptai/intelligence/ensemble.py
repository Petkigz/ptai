"""
Ensemble Forecaster - synthesizes multiple independent models
Base-rate 0.68, News 0.75, X 0.71, Market 0.69, LLM 0.73 -> Ensemble 0.712
Much more defensible than one LLM output.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import math
from loguru import logger

from ..markets.base import Market
from .forecaster import ModelForecast


@dataclass
class ForecastResult:
    market_id: str
    question: str
    market_price: float
    fair_probability: float  # Ensemble probability
    confidence: float
    uncertainty: float
    edge: float  # fair - market
    effective_edge: float = 0.0
    should_trade: bool = False
    reasoning: str = ""
    bull_case: str = ""
    bear_case: str = ""
    unknown: str = ""
    resolution_risks: List[str] = field(default_factory=list)
    model_forecasts: List[ModelForecast] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    calibration_adjusted: float = 0.0
    conservative_fair: float = 0.0  # After uncertainty penalty
    # EVERY component, with the probability it wanted, the confidence it claimed
    # and the weight it actually got. The operator could see the LLM say 65% and
    # the ensemble say 58% and nothing in between: "which model contributed what"
    # had no answer anywhere in the system. This is that answer, per market.
    components: List[Dict[str, Any]] = field(default_factory=list)
    # The chain, step by step, in the order the numbers are produced:
    # market -> llm raw -> ensemble (weighted) -> calibrated -> conservative.
    chain: Dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """
        One line: how the final probability was built, term by term.

        `fair 0.580 <- ensemble 0.580 [llm_reasoning 0.650 c0.80 w0.24 47% |
        news 0.500 c0.00 w0.00 0% ...] -> calibrated -> conservative; market 0.465`
        """
        parts = []
        for c in self.components:
            share = c.get("weight_share")
            share_txt = f"{share:.0%}" if isinstance(share, (int, float)) else "-"
            why = "" if c.get("contributes") else f" ({c.get('note') or 'no weight'})"
            parts.append(
                f"{c['model']} {c['probability']:.3f} c{c['confidence']:.2f} "
                f"w{c['weight_used']:.3f} {share_txt}{why}")
        ch = self.chain or {}
        llm = ch.get("llm_raw")
        head = (f"fair {self.fair_probability:.3f} <- ensemble "
                f"{ch.get('ensemble_raw', self.fair_probability):.3f} "
                f"(llm raw {llm:.3f}) " if llm is not None
                else f"fair {self.fair_probability:.3f} <- ensemble "
                     f"{ch.get('ensemble_raw', self.fair_probability):.3f} ")
        return (head + "[" + " | ".join(parts) + "]"
                + f" -> calibrated {ch.get('calibrated', self.fair_probability):.3f}"
                + f" -> conservative {self.conservative_fair:.3f}"
                + f"; market {self.market_price:.3f}")

    def to_proposal(self) -> Dict[str, Any]:
        """LLM proposes, deterministic code decides"""
        return {
            "market": self.question,
            "market_id": self.market_id,
            "side": "YES" if self.fair_probability > self.market_price else "NO",
            "fair_probability": round(self.fair_probability, 4),
            "market_probability": round(self.market_price, 4),
            "edge": round(self.edge, 4),
            "effective_edge": round(self.effective_edge, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "reasoning": self.reasoning,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "sources": self.sources,
            "resolution_risks": self.resolution_risks,
            "trade": self.should_trade,
            "calibration_adjusted": round(self.calibration_adjusted, 4),
            "conservative_fair": round(self.conservative_fair, 4),
            "components": self.components,
            "chain": self.chain,
            "explain": self.explain(),
        }


def _llm_raw(forecasts) -> Optional[float]:
    """The LLM's own probability, before any ensemble weighting - or None."""
    for f in forecasts:
        if f.model_name == "llm_reasoning":
            return round(float(f.probability), 4)
    return None


class EnsembleForecaster:
    """
    Ensemble of independent forecasting components.
    Each model votes, then weighted average.
    Calibration adjusts final probability.
    """
    def __init__(self, calibration_engine=None, uncertainty_engine=None, llm_router=None):
        # One detector for the whole process, not one per market. A fresh one per
        # market can never see a repetition - which is exactly how the operator's
        # 17 identical 0.65 answers went unnoticed by a system that had the
        # machinery to notice them.
        from ..agent.brain import AnswerRepetitionDetector
        self.answer_repetition = AnswerRepetitionDetector()
        self.calibration_engine = calibration_engine
        self.uncertainty_engine = uncertainty_engine
        self.llm_router = llm_router
        # WHAT THE MODEL WAS ACTUALLY ASKED, this cycle.
        #
        # The cycle's model line used to build its sentence from the deep
        # shortlist - a PLAN written before pricing - and then say "8 market(s)
        # were sent to the model but no answer was recorded" when the router's
        # own counter read 0 calls. Both numbers were in the same log line and
        # they contradicted each other, which is how the operator saw it. A
        # market counts as SENT here, at the call, and nowhere else.
        self.llm_asked = 0          # markets that reached the model step
        self.llm_answered = 0       # ...of those, markets the model answered
        self.llm_problems: list = []  # why the ones that were not answered were not
        # Model weights - can be learned from calibration performance
        self.model_weights = {
            "base_rate": 0.15,
            "news": 0.20,
            "x_sentiment": 0.15,
            "market_microstructure": 0.20,
            "llm_reasoning": 0.30,
            # A non-LLM heuristic answer. Same floor as any unlisted model, so it
            # cannot outvote the models that actually ran.
            "heuristic_reasoning": 0.10,
        }

    # The weight this component gets when the LLM DID NOT answer. Deliberately
    # the floor weight used for unrecognised models, not the 0.30 of
    # llm_reasoning: a heuristic that is not an LLM must not be weighted like one.
    HEURISTIC_MODEL_WEIGHT_NAME = "heuristic_reasoning"

    def reset_llm_accounting(self) -> None:
        """Start a fresh count of what the model was asked, for one cycle."""
        self.llm_asked = 0
        self.llm_answered = 0
        self.llm_problems = []

    def llm_accounting(self) -> Dict[str, Any]:
        """What the model was asked this cycle, and what it did not answer."""
        return {"asked": int(self.llm_asked),
                "answered": int(self.llm_answered),
                "problems": list(self.llm_problems[-3:])}

    def add_llm_forecast(self, market: Market, llm_result: Dict) -> ModelForecast:
        """
        Convert a Brain result to a ModelForecast.

        The provider tag is carried through. It used to be dropped here: Brain
        labels a non-LLM answer llm_provider="heuristic", and this method named
        every result "llm_reasoning" regardless - so a rule of thumb arrived in
        the ensemble wearing the LLM's name and the LLM's 0.30 weight, the largest
        of any component.
        """
        try:
            fair = llm_result.get("fair_value", 0.5) if isinstance(llm_result, dict) else getattr(llm_result, "fair_value", 0.5)
            conf = llm_result.get("confidence", 0.6) if isinstance(llm_result, dict) else getattr(llm_result, "confidence", 0.6)
            reasoning = llm_result.get("reasoning", "") if isinstance(llm_result, dict) else getattr(llm_result, "reasoning", "")
            provider = (llm_result.get("llm_provider", "") if isinstance(llm_result, dict)
                        else getattr(llm_result, "llm_provider", "")) or ""
            # Anything that is not a real model answer is named for what it is,
            # and falls to the default weight rather than the LLM's.
            model_name = (self.HEURISTIC_MODEL_WEIGHT_NAME
                          if str(provider).lower() in ("heuristic", "fallback", "")
                          else "llm_reasoning")
            if model_name == self.HEURISTIC_MODEL_WEIGHT_NAME:
                conf = min(float(conf), 0.4)
                reasoning = f"[NOT AN LLM ANSWER - provider={provider or 'none'}] {reasoning}"

            model_id = (llm_result.get("llm_model", "") if isinstance(llm_result, dict)
                        else getattr(llm_result, "llm_model", "")) or ""
            anchored = bool(llm_result.get("anchored", False)
                            if isinstance(llm_result, dict) else False)
            basis = (llm_result.get("basis", "") if isinstance(llm_result, dict)
                     else "") or ""
            if anchored:
                reasoning = f"[ANCHORED ANSWER - no weight] {reasoning}"
            return ModelForecast(
                model_name=model_name,
                probability=float(fair),
                confidence=float(conf),
                uncertainty=1.0 - float(conf),
                reasoning=reasoning[:500],
                # The actual id, plus the provider, so the trace can print
                # "llm_reasoning (lm_studio: qwen2.5-14b-instruct)".
                sources=[str(provider or "local_llm"),
                         f"{provider}:{model_id}" if model_id else "local_llm"],
                model_id=str(model_id),
                anchored=anchored,
            )
        except Exception as e:
            logger.warning(f"LLM forecast conversion failed: {e}")
            return ModelForecast(
                model_name="llm_reasoning",
                probability=market.best_price,
                confidence=0.3,
                uncertainty=0.4,
                reasoning=f"LLM conversion failed: {e}",
                sources=[]
            )

    def ensemble(self, forecasts: List[ModelForecast], market: Market,
                 category: str = None, llm_skipped: str = "",
                 deep_analysis: bool = True) -> ForecastResult:
        """Weighted ensemble of forecasts"""
        if not forecasts:
            return ForecastResult(
                market_id=market.id,
                question=market.question,
                market_price=market.best_price,
                fair_probability=market.best_price,
                confidence=0.3,
                uncertainty=0.5,
                edge=0.0,
                reasoning="No forecasts available"
            )

        # Weighted average by confidence and model weights
        total_weight = 0
        weighted_prob = 0
        total_conf = 0
        total_uncertainty = 0
        all_sources = []
        all_reasoning = []

        component_rows: List[Dict[str, Any]] = []
        for f in forecasts:
            nominal = self.model_weights.get(f.model_name, 0.1)
            weight = nominal * f.confidence
            # Discount by uncertainty
            weight *= (1 - f.uncertainty * 0.5)
            weighted_prob += f.probability * weight
            total_weight += weight
            total_conf += f.confidence
            total_uncertainty += f.uncertainty
            all_sources.extend(f.sources)
            all_reasoning.append(f"{f.model_name}={f.probability:.3f}(conf {f.confidence:.2f}): {f.reasoning[:100]}")
            # WHY a component got the weight it did, kept with the number. A
            # component with no data has weight 0 - which is the honest outcome -
            # and the trace must say that rather than leave a 0.500 that reads
            # like a neutral opinion.
            note = ""
            if getattr(f, "anchored", False):
                note = ("anchored answer (the model repeated itself across "
                        "different markets): confidence 0, contributes nothing")
            elif f.confidence <= 0:
                note = "no data: confidence 0, contributes nothing"
            elif weight == 0:
                note = "weighted to zero"
            component_rows.append({
                "model": f.model_name,
                # The id that answered, when there is one: the row is what the
                # console's forecast trace prints, and "llm_reasoning" alone
                # named the weight, never the model.
                "model_id": getattr(f, "model_id", "") or "",
                "probability": round(float(f.probability), 4),
                "confidence": round(float(f.confidence), 4),
                "uncertainty": round(float(f.uncertainty), 4),
                "weight_nominal": round(float(nominal), 4),
                "weight_used": round(float(weight), 6),
                "contributes": bool(weight > 0),
                "note": note,
                "anchored": bool(getattr(f, "anchored", False)),
                "basis": getattr(f, "basis", "") or "",
                "reasoning": (f.reasoning or "")[:240],
                "sources": list(f.sources or []),
            })

        if total_weight > 0:
            ensemble_prob = weighted_prob / total_weight
        else:
            # Not one model had data. The average of a set of models that all
            # declined to answer is not a forecast - it used to be whatever
            # midpoint their placeholders averaged to (0.5), which is how a
            # market priced 0.007 could come back with a fair value near 0.10 and
            # an edge worth trading. With no information the market's own price is
            # the estimate; there is no edge to claim.
            ensemble_prob = market.best_price

        if total_weight > 0:
            avg_conf = total_conf / len(forecasts) if forecasts else 0.5
            avg_uncertainty = total_uncertainty / len(forecasts) if forecasts else 0.5
        else:
            avg_conf = 0.0
            avg_uncertainty = 0.5

        # Calibration adjustment
        calibrated_prob = ensemble_prob
        if total_weight <= 0:
            logger.info(
                f"Ensemble {market.id}: no model had data (market {market.best_price:.3f}) "
                f"- reporting the market price and no confidence, not an opinion")
        if self.calibration_engine:
            try:
                # The market's OWN category. This was hardcoded to "default"
                # with a comment saying it would detect the category later -
                # so every sport, election and crypto market was calibrated
                # against one pooled curve, and the per-category learning the
                # calibration engine is built around never happened.
                calibrated_prob = self.calibration_engine.calibrate(
                    probability=ensemble_prob,
                    category=(category
                              or getattr(market, "category", None)
                              or "default"),
                    confidence=avg_conf
                )
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

        # Uncertainty penalty - conservative fair value
        # forecast 71% ±8% -> conservative 63%
        conservative_fair = calibrated_prob
        if self.uncertainty_engine:
            try:
                conservative_fair = self.uncertainty_engine.conservative_estimate(
                    probability=calibrated_prob,
                    uncertainty=avg_uncertainty
                )
            except:
                # Simple fallback: penalize by uncertainty
                conservative_fair = calibrated_prob - avg_uncertainty * 0.5
                conservative_fair = max(0.05, min(0.95, conservative_fair))
        else:
            conservative_fair = calibrated_prob - avg_uncertainty * 0.3
            conservative_fair = max(0.05, min(0.95, conservative_fair))

        edge = calibrated_prob - market.best_price
        conservative_edge = conservative_fair - market.best_price

        # Determine if should trade based on effective edge (conservative)
        # Effective edge will be calculated later by edge calculator with fees etc
        should_trade = abs(conservative_edge) >= 0.08 and avg_conf >= 0.6

        reasoning = " | ".join(all_reasoning)

        for row in component_rows:
            row["weight_share"] = (round(row["weight_used"] / total_weight, 4)
                                   if total_weight > 0 else 0.0)

        return ForecastResult(
            market_id=market.id,
            question=market.question,
            market_price=market.best_price,
            fair_probability=calibrated_prob,
            confidence=avg_conf,
            uncertainty=avg_uncertainty,
            edge=edge,
            effective_edge=conservative_edge,  # temporary, will be refined
            should_trade=should_trade,
            reasoning=reasoning[:2000],
            model_forecasts=forecasts,
            calibration_adjusted=calibrated_prob,
            conservative_fair=conservative_fair,
            components=component_rows,
            chain={
                # The four numbers the operator asked to see together.
                "market_price": round(float(market.best_price), 4),
                "llm_raw": _llm_raw(forecasts),
                "ensemble_raw": round(float(ensemble_prob), 4),
                "calibrated": round(float(calibrated_prob), 4),
                "conservative": round(float(conservative_fair), 4),
                "executable_edge": None,  # filled by FairValueEngine after costs
                "total_weight": round(float(total_weight), 6),
                "components_with_data": [r["model"] for r in component_rows
                                         if r["contributes"]],
                "components_without_data": [r["model"] for r in component_rows
                                            if not r["contributes"]],
                # Why the largest-weight component is absent, when it is.
                "llm_skipped": llm_skipped,
                "deep_analysis": bool(deep_analysis),
                "llm_anchored": bool(
                    next((r for r in component_rows
                          if r["model"] == "llm_reasoning"), {})
                    .get("anchored", False)),
            },
            sources=list(set(all_sources)),
        )

    def forecast_market(self, market: Market, context: Dict = None) -> ForecastResult:
        """Full forecasting pipeline for one market"""
        context = context or {}
        forecasts = []

        # Base rate
        from .forecaster import BaseRateModel, NewsModel, XModel, MarketMicrostructureModel
        base_model = BaseRateModel()
        forecasts.append(base_model.forecast(
            market, category=context.get("category", "default"),
            # Real counted frequencies when the cycle loaded them. None means the
            # model keeps contributing nothing, and says so.
            prior=context.get("base_rate_prior"),
        ))

        # News
        news_model = NewsModel(llm_router=self.llm_router)
        forecasts.append(news_model.forecast(market, news_text=context.get("news", ""), research=context.get("research")))

        # X
        x_model = XModel()
        forecasts.append(x_model.forecast(
            market,
            sentiment_result=context.get("sentiment"),
            tweets=context.get("tweets", []),
            # The status travels with the evidence so the component can say WHY
            # it has none: a blocked scraper, a missing engine, or a failure.
            status=str(context.get("x_status") or ""),
            unavailable_reason=str(context.get("x_unavailable_reason") or ""),
        ))

        # Market microstructure
        mm_model = MarketMicrostructureModel()
        forecasts.append(mm_model.forecast(market, orderbook=context.get("orderbook"), recent_trades=context.get("recent_trades")))

        # LLM reasoning if available.
        #
        # ...but only for a market that was chosen for deep analysis. The cycle
        # screens every market on its measured book first and spends model time on
        # the shortlist: this is where that decision is honoured, and the trace
        # records it so "no LLM component" reads as a decision rather than as a
        # failure.
        deep_ok = context.get("deep_analysis") is not False
        llm_skipped_reason = "" if deep_ok else str(
            context.get("screen_reason") or "not in this cycle's deep shortlist")
        if not deep_ok:
            logger.debug(f"LLM skipped for {market.id}: {llm_skipped_reason}")
        if context.get("llm_result"):
            forecasts.append(self.add_llm_forecast(market, context["llm_result"]))
        elif not deep_ok:
            pass
        elif self.llm_router:
            # Try to get LLM forecast via Brain
            try:
                from ..agent.brain import Brain
                # `Brain()` built its OWN router from settings, which could
                # differ from the one this ensemble was constructed with - so the
                # fallback forecast could run against a different provider,
                # endpoint, model and timeout than the rest of the intelligence
                # stack. Hand it the router already in use.
                brain = Brain(llm_router=self.llm_router,
                              answer_repetition=self.answer_repetition)
                # The research is passed through. V3 goes to the trouble of
                # fetching news, X sentiment and web research, and the LLM call
                # was dropping the web research on the floor - so the component
                # with the largest weight reasoned from the market question and
                # the sentiment alone.
                #
                # `sentiment` is handed over as V3 built it; Brain accepts the
                # dict or the structured object.
                self.llm_asked += 1
                llm_res = brain.estimate_fair_value(
                    market=market,
                    sentiment=context.get("sentiment"),
                    research_text=context.get("research") or "",
                )
                # Did the model produce the estimate, or did the heuristic ?
                # The result says which: an answer carries the model id the
                # server reported, a fallback does not.
                if getattr(llm_res, "llm_model", ""):
                    self.llm_answered += 1
                else:
                    problem = (getattr(llm_res, "raw", {}) or {}).get(
                        "llm_problem") or "no answer was recorded for this market"
                    self.llm_problems.append(str(problem))
                forecasts.append(self.add_llm_forecast(market, {
                    "fair_value": llm_res.fair_value,
                    "confidence": llm_res.confidence,
                    "reasoning": llm_res.reasoning,
                    # Carried through, or the ensemble cannot tell an LLM answer
                    # from a heuristic one.
                    "llm_provider": getattr(llm_res, "llm_provider", ""),
                    # ...and which model answered, so the component row and the
                    # console's trace can name it.
                    "llm_model": getattr(llm_res, "llm_model", ""),
                    # ...and whether the answer was a repeated one, so it can be
                    # shown as such instead of as a confident forecast.
                    "anchored": bool((llm_res.raw or {}).get("anchored", False)),
                    "basis": (llm_res.raw or {}).get("basis", ""),
                }))
            except Exception as e:
                logger.warning(f"LLM forecast failed: {e}")

        return self.ensemble(forecasts, market,
                             category=context.get("category"),
                             llm_skipped=llm_skipped_reason,
                             deep_analysis=deep_ok)
