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
            "conservative_fair": round(self.conservative_fair, 4)
        }


class EnsembleForecaster:
    """
    Ensemble of independent forecasting components.
    Each model votes, then weighted average.
    Calibration adjusts final probability.
    """
    def __init__(self, calibration_engine=None, uncertainty_engine=None, llm_router=None):
        self.calibration_engine = calibration_engine
        self.uncertainty_engine = uncertainty_engine
        self.llm_router = llm_router
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

            return ModelForecast(
                model_name=model_name,
                probability=float(fair),
                confidence=float(conf),
                uncertainty=1.0 - float(conf),
                reasoning=reasoning[:500],
                sources=["lm_studio", "local_llm"]
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
                 category: str = None) -> ForecastResult:
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

        for f in forecasts:
            weight = self.model_weights.get(f.model_name, 0.1) * f.confidence
            # Discount by uncertainty
            weight *= (1 - f.uncertainty * 0.5)
            weighted_prob += f.probability * weight
            total_weight += weight
            total_conf += f.confidence
            total_uncertainty += f.uncertainty
            all_sources.extend(f.sources)
            all_reasoning.append(f"{f.model_name}={f.probability:.3f}(conf {f.confidence:.2f}): {f.reasoning[:100]}")

        if total_weight > 0:
            ensemble_prob = weighted_prob / total_weight
        else:
            ensemble_prob = sum(f.probability for f in forecasts) / len(forecasts)

        avg_conf = total_conf / len(forecasts) if forecasts else 0.5
        avg_uncertainty = total_uncertainty / len(forecasts) if forecasts else 0.5

        # Calibration adjustment
        calibrated_prob = ensemble_prob
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
            sources=list(set(all_sources)),
            calibration_adjusted=calibrated_prob,
            conservative_fair=conservative_fair
        )

    def forecast_market(self, market: Market, context: Dict = None) -> ForecastResult:
        """Full forecasting pipeline for one market"""
        context = context or {}
        forecasts = []

        # Base rate
        from .forecaster import BaseRateModel, NewsModel, XModel, MarketMicrostructureModel
        base_model = BaseRateModel()
        forecasts.append(base_model.forecast(market, category=context.get("category", "default")))

        # News
        news_model = NewsModel(llm_router=self.llm_router)
        forecasts.append(news_model.forecast(market, news_text=context.get("news", ""), research=context.get("research")))

        # X
        x_model = XModel()
        forecasts.append(x_model.forecast(market, sentiment_result=context.get("sentiment"), tweets=context.get("tweets", [])))

        # Market microstructure
        mm_model = MarketMicrostructureModel()
        forecasts.append(mm_model.forecast(market, orderbook=context.get("orderbook"), recent_trades=context.get("recent_trades")))

        # LLM reasoning if available
        if context.get("llm_result"):
            forecasts.append(self.add_llm_forecast(market, context["llm_result"]))
        elif self.llm_router:
            # Try to get LLM forecast via Brain
            try:
                from ..agent.brain import Brain
                # `Brain()` built its OWN router from settings, which could
                # differ from the one this ensemble was constructed with - so the
                # fallback forecast could run against a different provider,
                # endpoint, model and timeout than the rest of the intelligence
                # stack. Hand it the router already in use.
                brain = Brain(llm_router=self.llm_router)
                # The research is passed through. V3 goes to the trouble of
                # fetching news, X sentiment and web research, and the LLM call
                # was dropping the web research on the floor - so the component
                # with the largest weight reasoned from the market question and
                # the sentiment alone.
                #
                # `sentiment` is handed over as V3 built it; Brain accepts the
                # dict or the structured object.
                llm_res = brain.estimate_fair_value(
                    market=market,
                    sentiment=context.get("sentiment"),
                    research_text=context.get("research") or "",
                )
                forecasts.append(self.add_llm_forecast(market, {
                    "fair_value": llm_res.fair_value,
                    "confidence": llm_res.confidence,
                    "reasoning": llm_res.reasoning,
                    # Carried through, or the ensemble cannot tell an LLM answer
                    # from a heuristic one.
                    "llm_provider": getattr(llm_res, "llm_provider", ""),
                }))
            except Exception as e:
                logger.warning(f"LLM forecast failed: {e}")

        return self.ensemble(forecasts, market,
                             category=context.get("category"))
