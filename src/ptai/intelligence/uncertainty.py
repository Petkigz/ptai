"""
Uncertainty Engine - prevents LLM from turning uncertainty into fake edge
Market 60%, Model 71% (+11% edge) but uncertainty ±8% -> conservative 63% -> only 3% edge -> NO TRADE
"""
from typing import Dict, List, Tuple
import math
from loguru import logger
from ..markets.orderbook import read_spread


def conservative_probability(probability: float, market_price: float,
                             uncertainty: float) -> float:
    """
    The system's ONE uncertainty haircut: shrink the estimate toward the price.

    The estimate loses its edge, and never gains one - the haircut moves it
    toward what the market says and stops at the market. Two defects the old
    rule had, both visible in the operator's 2026-09-29 18:14 log:

      * it moved the estimate toward 0.5 whatever the market said, so a fair
        value of 0.49 against a market of 0.40 - the case the whole mispricing
        hunt is built for - was made LARGER by its own risk haircut: the log
        printed `-> conservative 0.990` beside the edge calculator's own
        `conservative_fair 0.355` in the same cycle. A haircut that increases
        the edge is not a haircut;
      * anchored on 0.5 it can cross the price and report a conservative edge
        on the OTHER side of the market, which is an invented trade.

    The size of the haircut is the uncertainty itself, which is what the
    engine's own documentation has always said it was: "forecast 71% ±8% ->
    conservative 63%".
    """
    try:
        probability = float(probability)
        market_price = float(market_price)
        uncertainty = max(0.0, float(uncertainty))
    except (TypeError, ValueError):
        return probability
    if probability >= market_price:
        conservative = probability - uncertainty
        conservative = max(market_price, conservative)
    else:
        conservative = probability + uncertainty
        conservative = min(market_price, conservative)
    return max(0.01, min(0.99, conservative))


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
        """
        Calculate total uncertainty from multiple sources.

        ONLY the components that answered are measured. A component with no
        data has confidence 0, and it used to vote twice: its placeholder
        probability widened the "model disagreement" term and its zero
        confidence dragged the average down, charging 0.21 of "low confidence"
        for a market whose actual model was 75% confident. Its absence is
        already charged once, honestly, by the sparse-data term below.
        """
        context = context or {}
        uncertainties = []

        # The components with something to say. Everything in this method is
        # measured over these and only these.
        with_data = [f for f in forecasts or []
                     if float(getattr(f, "confidence", 0.0) or 0.0) > 0
                     and not getattr(f, "anchored", False)]

        # Model disagreement - if models disagree widely, high uncertainty
        if with_data:
            probs = [f.probability for f in with_data]
            if len(probs) > 1:
                mean = sum(probs) / len(probs)
                variance = sum((p - mean) ** 2 for p in probs) / len(probs)
                disagreement = math.sqrt(variance)
                uncertainties.append(disagreement * 1.5)  # weight disagreement heavily

        # Low confidence
        if with_data:
            avg_conf = sum(f.confidence for f in with_data) / len(with_data)
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
        Conservative fair value, with no market to shrink toward.

        Kept for callers that only have the probability. The one haircut the
        system actually applies is `conservative_probability` below, which
        needs the price; this delegates with a 0.5 anchor so an old caller
        still gets "71% ±8% -> 63%".
        """
        return conservative_probability(probability, 0.5, uncertainty)

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

    def should_trade(self, effective_edge: float, confidence: float,
                     uncertainty: float, resolution_risks: List[str] = None,
                     raw_edge: float = None) -> Tuple[bool, str]:
        """
        Final decision: should we trade?

        The 8% and 12% bars are on the MISPRICING (raw edge), not on the
        effective edge. This function is the first gate in the chain and it was
        the one that made the rest unreachable: it returned False for any
        opportunity whose POST-COST edge was under 8%, so `FairValueResult
        .should_trade` was False and the strategy engine never even CONSTRUCTED
        the opportunity. Fixing the EV gate further down would have been
        decoration, because there would have been nothing to evaluate.

        Charging costs here and again in the net EV double-counts them, and it
        refuses trades that are profitable after those same costs:

            market 0.70, YES fair 0.85 / NO fair 0.55 - raw +0.150 either way
              YES: net +$0.22 (+7.3% of a $3 stake), effective 0.052 -> refused
              NO : net +$1.07 (+35.8% of a $3 stake), effective 0.028 -> refused

        Costs are still enforced, once, and not as a percentage bar: an edge the
        costs have eaten entirely is not a trade, and the net EV terms decide
        whether what is left is worth doing.

        `raw_edge` is optional so existing callers keep their behaviour; the
        mispricing falls back to the effective edge when it is not supplied.
        """
        resolution_risks = resolution_risks or []
        mispricing = raw_edge if raw_edge else effective_edge

        # If resolution risks exist, no trade regardless of edge
        if resolution_risks:
            return False, f"Resolution risks: {resolution_risks}"

        # If mispricing < 8%, no trade
        if abs(mispricing) < 0.08:
            return False, (f"Mispricing {mispricing:.3f} < 8% threshold "
                           f"(effective {effective_edge:.3f} after costs)")

        # Costs must not consume the whole edge. This is the cost check that
        # replaces the post-cost percentage bar, and it is the one that matters.
        if effective_edge <= 0:
            return False, (f"Costs consumed the edge: effective "
                           f"{effective_edge:.3f} from mispricing {mispricing:.3f}")

        # If confidence < 60%, no trade
        if confidence < 0.6:
            return False, f"Confidence {confidence:.3f} < 60%"

        # If uncertainty > 15%, be cautious - require higher mispricing
        if uncertainty > 0.15 and abs(mispricing) < 0.12:
            return False, (f"High uncertainty {uncertainty:.3f} requires "
                           f"mispricing >12%, got {mispricing:.3f}")

        return True, (f"Pass: mispricing {mispricing:.3f} effective "
                      f"{effective_edge:.3f} conf {confidence:.3f} "
                      f"unc {uncertainty:.3f}")
