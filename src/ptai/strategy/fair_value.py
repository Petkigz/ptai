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
    # The outcome side this forecast is traded on, so no caller has to re-derive
    # it and risk disagreeing with the edge that was computed.
    side: str = "YES"
    forecast_result: Optional[ForecastResult] = None
    resolution_analysis: Any = None
    contradiction_report: Any = None


class FairValueEngine:
    """
    Fair Value Engine - most important part.
    Uses ensemble, calibration, uncertainty, contradiction, resolution analysis.
    """
    def __init__(self, llm_router=None, calibration_engine=None, uncertainty_engine=None,
                 web_researcher=None):
        self.llm_router = llm_router
        self.calibration_engine = calibration_engine or CalibrationEngine()
        self.uncertainty_engine = uncertainty_engine or UncertaintyEngine()
        # Without a researcher the contradiction engine can only read the
        # market definition. That used to be papered over by keyword-matching
        # the question for words like "will", which produced 0.6-strength
        # evidence on nearly every market.
        self.contradiction_engine = ContradictionEngine(
            llm_router=llm_router, web_researcher=web_researcher)
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

        # Step 2: Contradiction analysis - bull vs bear.
        # Prefer the research result the caller already fetched, so the two
        # researchers read the retrieved sources rather than re-running search.
        contradiction_report = self.contradiction_engine.synthesize(
            market=market,
            research_text=context.get("research", ""),
            news=context.get("news", ""),
            tweets=context.get("tweets", []),
            research_result=context.get("research_result"),
        )

        # Step 3: Ensemble forecasting
        # Add contradiction info to context
        context["bull_strength"] = sum(e.strength for e in contradiction_report.supporting_yes)
        context["bear_strength"] = sum(e.strength for e in contradiction_report.supporting_no)
        context["resolution_risks"] = resolution_analysis.risks
        # Do not invent a spread. If the orderbook context did not come back
        # with one, `spread` stays None and downstream can see it is unknown
        # rather than pricing a made-up 2%.
        _ob = context.get("orderbook") or {}
        context["spread"] = _ob.get("spread")
        context["spread_is_real"] = bool(_ob.get("is_real", False))
        if context["spread"] is None:
            logger.debug(f"FairValue {market.id}: no measured spread available")

        forecast_result = self.ensemble_forecaster.forecast_market(market, context=context)

        # Adjust confidence by contradiction. The adjustment is applied only
        # when the report is backed by retrieved sources; for an unresearched
        # report there is nothing to contradict with, and applying a fixed
        # penalty every time would be a constant masquerading as analysis.
        if contradiction_report.researched:
            adjusted_confidence = max(
                0.1, forecast_result.confidence + contradiction_report.confidence_adjustment)
        else:
            adjusted_confidence = forecast_result.confidence
        forecast_result.confidence = adjusted_confidence

        # Step 4: Effective edge calculation, ON THE SIDE THE TRADE WOULD BE ON.
        #
        # A fair value BELOW the market is a positive edge on NO. Computing it as
        # fair - market unconditionally made every such market look like a loss,
        # so the agent only ever bought YES and half of all mispricings - the half
        # where the market was too high - were structurally invisible to it.
        #
        # Derived from the forecast that was just computed, so the side the edge
        # is measured for is the same side the trade is placed on. These used to
        # be decided in two different files.
        side = str(context.get("side") or (
            "YES" if forecast_result.fair_probability > market.best_price else "NO")).upper()

        # The raw edge has to be on the side being traded, exactly like the
        # effective edge below it and for the same reason. It was the ensemble's
        # YES-space `fair - market` on an object whose `side` was NO, so a
        # genuinely profitable NO mispricing carried a raw edge of -0.15: the
        # effective edge said +0.106, the raw edge said -0.15, and every
        # downstream check that reads the raw edge saw the wrong sign. The
        # opportunity was built and then refused by the gate that hunts for
        # mispricing, on a market it had just correctly identified as mispriced.
        raw_edge = forecast_result.edge if side == "YES" else -forecast_result.edge

        from .edge import EdgeCalculator
        edge_calc = EdgeCalculator(uncertainty_engine=self.uncertainty_engine)
        effective = edge_calc.calculate(
            market=market,
            fair_prob=forecast_result.fair_probability,
            uncertainty=forecast_result.uncertainty,
            orderbook=context.get("orderbook"),
            amount_usd=context.get("amount_usd", 5.0),
            correlation_penalty=context.get("correlation_penalty", 0.0),
            category_exposure=context.get("category_exposure", 0.0),
            side=side,
        )

        # Step 5: Final decision with uncertainty margin
        should_trade, reason = self.uncertainty_engine.should_trade(
            effective_edge=effective.effective_edge,
            confidence=adjusted_confidence,
            uncertainty=forecast_result.uncertainty,
            resolution_risks=resolution_analysis.risks,
            # The 8% rule is about mispricing; the costs are charged once, here
            # and in the net EV, not twice.
            raw_edge=raw_edge,
        )

        # Override if contradiction report says high risk.
        # Measured on the same mispricing scale as the 12% in the uncertainty
        # gate above, not on the post-cost edge - otherwise contradictory
        # evidence and the costs already deducted both count against the same
        # edge, and the two thresholds that were written as one rule disagree.
        if (contradiction_report.confidence_adjustment < -0.15
                and abs(raw_edge) < 0.12):
            should_trade = False
            reason = (f"High conflicting evidence + mispricing "
                      f"{raw_edge:.3f} < 12%")

        final_reasoning = (
            f"Ensemble fair {forecast_result.fair_probability:.3f} market {market.best_price:.3f} "
            f"raw edge [{side}] {raw_edge:.3f} effective {effective.effective_edge:.3f} | "
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
            edge=raw_edge,  # the mispricing on `side`, not the ensemble's YES edge
            effective_edge=effective.effective_edge,
            side=side,
            should_trade=should_trade,
            reasoning=final_reasoning,
            forecast_result=forecast_result,
            resolution_analysis=resolution_analysis,
            contradiction_report=contradiction_report
        )
