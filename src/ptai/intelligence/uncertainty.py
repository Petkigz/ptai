"""
Uncertainty Engine - prevents LLM from turning uncertainty into fake edge
Market 60%, Model 71% (+11% edge) but uncertainty ±8% -> conservative 63% -> only 3% edge -> NO TRADE
"""
from typing import Dict, List, Tuple
import math
from loguru import logger
from ..markets.orderbook import read_spread


class UncertaintyEngine:
    """
    Calculates uncertainty margin and conservative fair value.
    If confidence interval overlaps market substantially, no trade.
    """
    def __init__(self):
        self.uncertainty_sources = [
            "model_disagreement",      # Ensemble models disagree
            "low_confidence",          # Low average confidence
            "sparse_data",             # Little information
            "high_volatility",         # Market volatile
            "wide_spread",             # Wide bid/ask
            "conflicting_evidence",    # Bull/bear both strong
            "resolution_ambiguity",    # Ambiguous resolution rules
            "time_pressure"            # Short time to resolution
        ]

    def calculate_uncertainty(self, forecasts: List, context: Dict = None) -> float:
        """Calculate total uncertainty from multiple sources"""
        context = context or {}
        uncertainties = []

        # Model disagreement - if models disagree widely, high uncertainty
        if forecasts:
            probs = [f.probability for f in forecasts]
            if len(probs) > 1:
                mean = sum(probs) / len(probs)
                variance = sum((p - mean) ** 2 for p in probs) / len(probs)
                disagreement = math.sqrt(variance)
                uncertainties.append(disagreement * 1.5)  # weight disagreement heavily

        # Low confidence
        if forecasts:
            avg_conf = sum(f.confidence for f in forecasts) / len(forecasts)
            low_conf_uncertainty = (1 - avg_conf) * 0.3
            uncertainties.append(low_conf_uncertainty)

        # Sparse data
        source_count = len(context.get("sources", []))
        if source_count < 3:
            uncertainties.append(0.15)
        elif source_count < 5:
            uncertainties.append(0.05)

        # Wide spread. An UNKNOWN spread is worse than a wide one: we cannot
        # bound the cost of getting in or out, so it must raise uncertainty
        # rather than quietly become a 2% default. This feeds position sizing.
        spread, spread_is_real = read_spread(context, 0.02)
        if spread_is_real:
            uncertainties.append(min(0.2, spread * 2))
        else:
            uncertainties.append(0.2)

        # Conflicting evidence (bull and bear both strong)
        bull_strength = context.get("bull_strength", 0)
        bear_strength = context.get("bear_strength", 0)
        if bull_strength > 0.6 and bear_strength > 0.6:
            uncertainties.append(0.15)  # Both strong = conflicting

        # Resolution ambiguity
        if context.get("resolution_risks"):
            uncertainties.append(len(context["resolution_risks"]) * 0.05)

        # Time pressure - very short time = higher uncertainty if not close to resolution
        hours_left = context.get("hours_to_resolution")
        if hours_left is not None:
            if hours_left < 1:
                uncertainties.append(0.1)  # very short time, risky
            elif hours_left > 24*30:
                uncertainties.append(0.05)  # far out, more uncertainty

        total_uncertainty = sum(uncertainties)
        # Cap at 0.3 (30% uncertainty)
        total_uncertainty = min(0.3, total_uncertainty)
        # Floor at 0.02 (minimum uncertainty)
        total_uncertainty = max(0.02, total_uncertainty)

        logger.debug(f"Uncertainty calc: sources={uncertainties} total={total_uncertainty:.3f}")
        return total_uncertainty

    def conservative_estimate(self, probability: float, uncertainty: float) -> float:
        """
        Conservative fair value: penalize by uncertainty.
        If forecast 71% ±8%, conservative = 71% - 8% = 63% for YES side,
        or if betting NO, would be +8%.
        For edge calculation, we want to be conservative toward market price.
        """
        # For simplicity: subtract uncertainty (be conservative)
        # In production: would use confidence interval
        conservative = probability - uncertainty
        # But don't go beyond 0.5 if original was above 0.5 and market is below?
        # Actually: conservative should move toward 0.5 (less confident)
        # 71% with 8% uncertainty -> 63% (moves toward 0.5)
        # 29% with 8% uncertainty -> 37% (also toward 0.5)
        if probability > 0.5:
            conservative = probability - uncertainty
        else:
            conservative = probability + uncertainty
        
        conservative = max(0.01, min(0.99, conservative))
        return conservative

    def effective_edge(self, market_price: float, fair_prob: float, uncertainty: float, fees: float = 0, spread: float = 0, slippage: float = 0) -> Tuple[float, float, str]:
        """
        Calculate effective edge after all deductions:
        raw_edge -> fees -> spread -> slippage -> liquidity -> uncertainty -> correlation -> time -> effective_edge
        
        Returns (effective_edge, conservative_fair, reasoning)
        """
        raw_edge = fair_prob - market_price
        
        # Conservative fair after uncertainty
        conservative_fair = self.conservative_estimate(fair_prob, uncertainty)
        conservative_edge = conservative_fair - market_price

        # Deduct fees, spread, slippage
        total_costs = fees + spread + slippage
        effective_edge = conservative_edge - total_costs

        # For NO side, edge is negative of YES edge, but we take absolute for comparison
        # Actually edge should be signed: if fair > market, edge positive for YES
        # If fair < market, edge positive for NO (i.e. market overpriced YES = underpriced NO)
        # So effective_edge is signed

        reasoning = (
            f"Raw edge {raw_edge:.3f} (fair {fair_prob:.3f} - market {market_price:.3f}) | "
            f"Uncertainty {uncertainty:.3f} -> conservative fair {conservative_fair:.3f} edge {conservative_edge:.3f} | "
            f"Costs fees {fees:.3f} spread {spread:.3f} slippage {slippage:.3f} total {total_costs:.3f} | "
            f"Effective edge {effective_edge:.3f}"
        )

        logger.info(reasoning)
        return effective_edge, conservative_fair, reasoning

    def should_trade(self, effective_edge: float, confidence: float, uncertainty: float, resolution_risks: List[str] = None) -> Tuple[bool, str]:
        """
        Final decision: should we trade?
        If uncertainty interval overlaps market substantially, no trade.
        """
        resolution_risks = resolution_risks or []

        # If resolution risks exist, no trade regardless of edge
        if resolution_risks:
            return False, f"Resolution risks: {resolution_risks}"

        # If effective edge < 8%, no trade
        if abs(effective_edge) < 0.08:
            return False, f"Effective edge {effective_edge:.3f} < 8% threshold"

        # If confidence < 60%, no trade
        if confidence < 0.6:
            return False, f"Confidence {confidence:.3f} < 60%"

        # If uncertainty > 15%, be cautious - require higher edge
        if uncertainty > 0.15 and abs(effective_edge) < 0.12:
            return False, f"High uncertainty {uncertainty:.3f} requires edge >12%, got {effective_edge:.3f}"

        return True, f"Pass: edge {effective_edge:.3f} conf {confidence:.3f} unc {uncertainty:.3f}"
