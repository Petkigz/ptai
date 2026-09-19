"""
Fair Value Engine - synthesizes intelligence into fair value
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market
from ..intelligence.ensemble import EnsembleForecaster, ForecastResult
from ..intelligence.calibration import CalibrationEngine
from ..intelligence.uncertainty import UncertaintyEngine
from ..intelligence.contradiction import ContradictionEngine
from ..intelligence.resolution_analyzer import ResolutionAnalyzer


@dataclass
class FairValueResult:
    market_id: str
    fair_value: float
    confidence: float
    uncertainty: float
    edge: float
    effective_edge: float
    should_trade: bool
    reasoning: str
    forecast_result: Optional[ForecastResult] = None
    resolution_analysis: Any = None
    contradiction_report: Any = None


class FairValueEngine:
    """
    Fair Value Engine - most important part.
    Uses ensemble, calibration, uncertainty, contradiction, resolution analysis.
    """
    def __init__(self, llm_router=None, calibration_engine=None, uncertainty_engine=None):
        self.llm_router = llm_router
        self.calibration_engine = calibration_engine or CalibrationEngine()
        self.uncertainty_engine = uncertainty_engine or UncertaintyEngine()
        self.contradiction_engine = ContradictionEngine(llm_router=llm_router)
        self.resolution_analyzer = ResolutionAnalyzer(llm_router=llm_router)
        self.ensemble_forecaster = EnsembleForecaster(
            calibration_engine=self.calibration_engine,
            uncertainty_engine=self.uncertainty_engine,
            llm_router=llm_router
        )

    def estimate(self, market: Market, context: Dict = None) -> FairValueResult:
        context = context or {}

        # Step 1: Resolution risk check - if ambiguous, no trade
        resolution_analysis = self.resolution_analyzer.analyze(market)
        if not resolution_analysis.should_trade:
            logger.warning(f"Resolution risk blocks trade for {market.id}: {resolution_analysis.risks}")
            return FairValueResult(
                market_id=market.id,
                fair_value=market.best_price,
                confidence=0.3,
                uncertainty=0.3,
                edge=0.0,
                effective_edge=0.0,
                should_trade=False,
                reasoning=f"Blocked by resolution risk: {resolution_analysis.risks}",
                resolution_analysis=resolution_analysis
            )

        # Step 2: Contradiction analysis - bull vs bear
        contradiction_report = self.contradiction_engine.synthesize(
            market=market,
            research_text=context.get("research", ""),
            news=context.get("news", ""),
            tweets=context.get("tweets", [])
        )

        # Step 3: Ensemble forecasting
        # Add contradiction info to context
        context["bull_strength"] = sum(e.strength for e in contradiction_report.supporting_yes)
        context["bear_strength"] = sum(e.strength for e in contradiction_report.supporting_no)
        context["resolution_risks"] = resolution_analysis.risks
        context["spread"] = context.get("orderbook", {}).get("spread", 0.02)

        forecast_result = self.ensemble_forecaster.forecast_market(market, context=context)

        # Adjust confidence by contradiction
        adjusted_confidence = max(0.1, forecast_result.confidence + contradiction_report.confidence_adjustment)
        forecast_result.confidence = adjusted_confidence

        # Step 4: Effective edge calculation
        from .edge import EdgeCalculator
        edge_calc = EdgeCalculator(uncertainty_engine=self.uncertainty_engine)
        effective = edge_calc.calculate(
            market=market,
            fair_prob=forecast_result.fair_probability,
            uncertainty=forecast_result.uncertainty,
            orderbook=context.get("orderbook"),
            amount_usd=context.get("amount_usd", 5.0),
            correlation_penalty=context.get("correlation_penalty", 0.0),
            category_exposure=context.get("category_exposure", 0.0)
        )

        # Step 5: Final decision with uncertainty margin
        should_trade, reason = self.uncertainty_engine.should_trade(
            effective_edge=effective.effective_edge,
            confidence=adjusted_confidence,
            uncertainty=forecast_result.uncertainty,
            resolution_risks=resolution_analysis.risks
        )

        # Override if contradiction report says high risk
        if contradiction_report.confidence_adjustment < -0.15 and effective.effective_edge < 0.12:
            should_trade = False
            reason = f"High conflicting evidence + edge {effective.effective_edge:.3f} < 12%"

        final_reasoning = (
            f"Ensemble fair {forecast_result.fair_probability:.3f} market {market.best_price:.3f} "
            f"raw edge {forecast_result.edge:.3f} effective {effective.effective_edge:.3f} | "
            f"Conf {adjusted_confidence:.2f} unc {forecast_result.uncertainty:.2f} | "
            f"Resolution risk {resolution_analysis.risk_score:.2f} | "
            f"Contradiction net {contradiction_report.net_score:.2f} | "
            f"Decision: {should_trade} because {reason} | "
            f"Edge reasoning: {effective.reasoning}"
        )

        return FairValueResult(
            market_id=market.id,
            fair_value=forecast_result.fair_probability,
            confidence=adjusted_confidence,
            uncertainty=forecast_result.uncertainty,
            edge=forecast_result.edge,
            effective_edge=effective.effective_edge,
            should_trade=should_trade,
            reasoning=final_reasoning,
            forecast_result=forecast_result,
            resolution_analysis=resolution_analysis,
            contradiction_report=contradiction_report
        )
