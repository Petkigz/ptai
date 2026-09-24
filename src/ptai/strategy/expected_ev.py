"""
Expected Net EV Engine - V10 FIX #9
Opportunity scoring needs stronger statistical normalization

Current: edge × prob × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)
Problem: Not economically meaningful expected return

Wanted:
Expected Net P&L = Σ probability(outcome) × payoff(outcome) − fees − spread − slippage − funding − gas − expected execution loss − model uncertainty penalty

Then compare expected net return per dollar, per unit risk, per unit capital-time
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from ..venues.adapter import VenueOpportunity
from ..markets.orderbook import read_spread
from .edge import HUNT_MISPRICING_MIN, hunted_mispricing

@dataclass
class ExpectedEVResult:
    gross_ev_usd: float  # Expected gross profit
    net_ev_usd: float  # Expected net profit after all costs
    gross_ev_pct: float  # Gross return %
    net_ev_pct: float  # Net return %
    ev_per_dollar: float  # Net EV per dollar invested
    ev_per_risk: float  # Net EV per unit risk
    ev_per_capital_time: float  # Net EV per dollar per hour
    fees_usd: float
    spread_usd: float
    slippage_usd: float
    gas_usd: float
    uncertainty_penalty_usd: float
    execution_loss_usd: float
    funding_usd: float
    reasoning: str
    should_trade: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_ev_usd": round(self.gross_ev_usd, 4),
            "net_ev_usd": round(self.net_ev_usd, 4),
            "gross_ev_pct": round(self.gross_ev_pct, 4),
            "net_ev_pct": round(self.net_ev_pct, 4),
            "ev_per_dollar": round(self.ev_per_dollar, 4),
            "ev_per_risk": round(self.ev_per_risk, 4),
            "ev_per_capital_time": round(self.ev_per_capital_time, 6),
            "fees_usd": round(self.fees_usd, 4),
            "spread_usd": round(self.spread_usd, 4),
            "slippage_usd": round(self.slippage_usd, 4),
            "gas_usd": round(self.gas_usd, 4),
            "uncertainty_penalty_usd": round(self.uncertainty_penalty_usd, 4),
            "execution_loss_usd": round(self.execution_loss_usd, 4),
            "funding_usd": round(self.funding_usd, 4),
            "reasoning": self.reasoning,
            "should_trade": self.should_trade
        }

class ExpectedNetEVEngine:
    """
    V10 FIX #9: Economically meaningful EV calculation
    """
    def __init__(self):
        logger.info("ExpectedNetEVEngine initialized - V10 FIX #9")
    
    def calculate(self, opportunity: VenueOpportunity, amount_usd: float, orderbook: Dict = None) -> ExpectedEVResult:
        """
        Calculate expected net P&L with realistic costs
        
        Formula:
        Expected Net P&L = Σ prob(outcome) × payoff(outcome) − fees − spread − slippage − funding − gas − execution_loss − uncertainty_penalty
        
        For prediction markets (binary):
        - If YES side: payoff if YES wins = (1-market_price)/market_price * amount, if NO wins = -amount
        - Expected gross = prob_fair * payoff_win + (1-prob_fair) * payoff_lose
        - prob_fair = estimated_fair (calibrated)
        
        For crypto/stocks: similar but with price movement
        """
        orderbook = orderbook or {}
        
        market_price = opportunity.market_price
        fair_prob = opportunity.estimated_fair
        edge = opportunity.effective_edge
        confidence = opportunity.confidence
        uncertainty = opportunity.uncertainty
        
        # Fees
        fee_pct = opportunity.fees_pct or 0.02
        fees_usd = amount_usd * fee_pct
        
        # Spread - from orderbook or opportunity
        # read_spread handles an explicit None and reports whether the value
        # was actually measured, rather than defaulting silently.
        _measured_spread, _spread_is_real = read_spread(orderbook, 0.02)
        spread_pct = opportunity.spread_pct or _measured_spread
        spread_usd = amount_usd * spread_pct * 0.5  # half spread cost (buy at ask)
        
        # Slippage - from opportunity or orderbook depth
        slippage_pct = opportunity.slippage_pct or orderbook.get("slippage_estimate", 0.01) or 0.01
        # Real slippage based on amount/liquidity
        liquidity = getattr(opportunity.market, 'liquidity', 10000)
        if liquidity > 0:
            real_slippage_pct = min(0.05, amount_usd / liquidity * 0.5)
            slippage_pct = max(slippage_pct, real_slippage_pct)
        slippage_usd = amount_usd * slippage_pct
        
        # Gas - for on-chain venues
        gas_usd = 0.05 if opportunity.venue_id in ["polymarket", "afx_dex"] else 0.0
        if orderbook.get("venue") in ["polymarket", "afx_dex"]:
            gas_usd = 0.05
        
        # Funding - for perps/futures
        funding_usd = 0.0
        if "PERP" in opportunity.market.id or "futures" in str(opportunity.market.raw.get("type", "")).lower():
            # Funding ~0.01% per 8h, ~0.03% per day
            hours = opportunity.time_to_resolution_hours or 24
            funding_rate_daily = 0.0003
            funding_usd = amount_usd * funding_rate_daily * (hours / 24)
        
        # Execution loss - queue position, rejection, latency
        # If orderbook not real, higher execution loss
        is_real_ob = orderbook.get("is_real", False)
        execution_quality = opportunity.execution_quality or 0.5
        if not is_real_ob:
            execution_loss_pct = 0.02  # 2% extra if estimation only
        else:
            execution_loss_pct = (1.0 - execution_quality) * 0.02
        execution_loss_usd = amount_usd * execution_loss_pct
        
        # Uncertainty penalty - model disagreement, low confidence
        # Higher uncertainty = higher penalty
        uncertainty_penalty_pct = uncertainty * 0.5  # 50% of uncertainty as penalty
        if confidence < 0.7:
            uncertainty_penalty_pct += (0.7 - confidence) * 0.1
        uncertainty_penalty_usd = amount_usd * uncertainty_penalty_pct
        
        # Gross EV calculation
        # For prediction market binary YES:
        # If fair_prob = 0.7, market_price = 0.6, edge = 0.1
        # Bet YES: win prob = fair_prob, win payoff = (1-market_price) = 0.4 per share, but we bet amount_usd at price market_price
        # Number of shares = amount_usd / market_price
        # If YES wins: profit = shares * (1 - market_price) = amount_usd * (1-market_price)/market_price
        # If NO wins: loss = -amount_usd
        # Expected gross = fair_prob * profit_win + (1-fair_prob) * (-amount_usd)
        # = fair_prob * amount_usd * (1-market_price)/market_price - (1-fair_prob) * amount_usd
        # = amount_usd * [fair_prob*(1-market_price)/market_price - (1-fair_prob)]
        # Simplified for small edge: gross EV ≈ amount_usd * edge * confidence
        
        if opportunity.side.upper() == "YES":
            # YES side
            if market_price > 0:
                profit_if_win = amount_usd * (1 - market_price) / market_price
            else:
                profit_if_win = amount_usd
            loss_if_lose = -amount_usd
            gross_ev_usd = fair_prob * profit_if_win + (1 - fair_prob) * loss_if_lose
        else:
            # NO side - similar but for NO
            no_price = 1 - market_price
            if no_price > 0:
                profit_if_win = amount_usd * (1 - no_price) / no_price
            else:
                profit_if_win = amount_usd
            loss_if_lose = -amount_usd
            # For NO, fair prob of NO = 1 - fair_prob(YES)
            fair_prob_no = 1 - fair_prob
            gross_ev_usd = fair_prob_no * profit_if_win + (1 - fair_prob_no) * loss_if_lose
        
        # For financial venues (crypto/stocks), EV is edge * amount * confidence
        if opportunity.venue_type.value == "financial" or "crypto" in opportunity.venue_id or "stock" in opportunity.venue_id:
            gross_ev_usd = amount_usd * edge * confidence
        
        # Net EV after all costs
        total_costs = fees_usd + spread_usd + slippage_usd + gas_usd + funding_usd + execution_loss_usd + uncertainty_penalty_usd
        net_ev_usd = gross_ev_usd - total_costs
        
        gross_ev_pct = gross_ev_usd / amount_usd if amount_usd > 0 else 0
        net_ev_pct = net_ev_usd / amount_usd if amount_usd > 0 else 0
        
        ev_per_dollar = net_ev_pct  # net return per dollar
        ev_per_risk = net_ev_usd / max(0.01, uncertainty * amount_usd)  # per unit risk
        
        # Capital-time: per dollar per hour
        hours = opportunity.time_to_resolution_hours or 24
        ev_per_capital_time = net_ev_usd / max(1, amount_usd * hours) * 24  # per dollar per day
        
        # The >8% hunt criterion is a MISPRICING threshold, so it is measured on
        # the RAW edge. Applying it to the effective edge charges every cost
        # twice: once when the costs push the edge back under 8%, and again in
        # the net EV terms right above, which are the terms that actually know
        # what a cost is. It refused trades the cost-aware gates had already
        # approved, on both sides of the same mispricing:
        #
        #   market 0.70, YES fair 0.85, NO fair 0.55 - raw +0.150 either way
        #     YES: gross +21.4%, net +$0.22 (+7.3% of stake), eff 0.052 -> refused
        #     NO : gross +50.0%, net +$1.07 (+35.8% of stake), eff 0.028 -> refused
        #
        # Both passed net>0, net%>=3% and confidence>=60%. The only failing
        # condition was the effective edge, and the NO case - the one V16's
        # symmetric edge was built to reach - was the more profitable of the two.
        #
        # `raw_edge` is absent on hand-built opportunities, so an unset value
        # falls back to the effective edge rather than refusing a trade for a
        # field the caller never filled in.
        mispricing = hunted_mispricing(opportunity)

        # Should trade if net EV >0 and meets thresholds
        should_trade = (
            net_ev_usd > 0 and
            net_ev_pct >= 0.03 and  # at least 3% net after all costs
            mispricing >= HUNT_MISPRICING_MIN and  # 8% gross, costs charged below
            confidence >= 0.60
        )
        
        reasoning = (
            f"Gross EV ${gross_ev_usd:.2f} ({gross_ev_pct*100:.1f}%) = fair {fair_prob:.2f} vs mkt {market_price:.2f} edge {edge*100:.1f}% conf {confidence:.2f} * ${amount_usd:.2f} | "
            f"Costs: fees ${fees_usd:.2f} ({fee_pct*100:.1f}%) spread ${spread_usd:.2f} ({spread_pct*100:.1f}%) slippage ${slippage_usd:.2f} ({slippage_pct*100:.1f}%) "
            f"gas ${gas_usd:.2f} funding ${funding_usd:.2f} exec_loss ${execution_loss_usd:.2f} unc_penalty ${uncertainty_penalty_usd:.2f} = total ${total_costs:.2f} | "
            f"Net EV ${net_ev_usd:.2f} ({net_ev_pct*100:.1f}%) per $ {ev_per_dollar*100:.1f}% per risk {ev_per_risk:.2f} per $/day {ev_per_capital_time*100:.2f}% | "
            f"Should trade: {should_trade} (net>0, net%>=3%, "
            f"mispricing {mispricing*100:.1f}%>=8% raw, conf>=60% - "
            f"effective edge {edge*100:.1f}% is informational, the costs are "
            f"charged once in net EV)"
        )
        
        return ExpectedEVResult(
            gross_ev_usd=gross_ev_usd,
            net_ev_usd=net_ev_usd,
            gross_ev_pct=gross_ev_pct,
            net_ev_pct=net_ev_pct,
            ev_per_dollar=ev_per_dollar,
            ev_per_risk=ev_per_risk,
            ev_per_capital_time=ev_per_capital_time,
            fees_usd=fees_usd,
            spread_usd=spread_usd,
            slippage_usd=slippage_usd,
            gas_usd=gas_usd,
            uncertainty_penalty_usd=uncertainty_penalty_usd,
            execution_loss_usd=execution_loss_usd,
            funding_usd=funding_usd,
            reasoning=reasoning,
            should_trade=should_trade
        )
    
    def rank_by_net_ev(self, opportunities: list, amount_per_opp: float = 3.0) -> list:
        """Rank opportunities by expected net EV per dollar per risk"""
        scored = []
        for opp in opportunities:
            ev = self.calculate(opp, amount_per_opp)
            # Score = net_ev * confidence * liquidity * execution / (uncertainty + risk)
            # But now economically meaningful
            score = ev.net_ev_usd * opp.confidence * opp.liquidity_score * opp.execution_quality / max(0.01, opp.uncertainty + opp.fees_pct + opp.slippage_pct)
            scored.append((score, ev, opp))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "ExpectedNetEVEngine - V10 FIX #9",
            "principle": "Expected Net P&L = Σ prob×payoff − fees − spread − slippage − funding − gas − execution_loss − uncertainty_penalty",
            "vs_old": "Old: edge×prob×liquidity×execution×calibration×time/(fees+slippage+uncertainty+risk) - not economically meaningful. New: actual expected $ profit after all costs, per dollar, per risk, per capital-time",
            "metrics": [
                "gross_ev_usd - expected profit before costs",
                "net_ev_usd - after fees/spread/slippage/gas/funding/execution/uncertainty",
                "ev_per_dollar - net return % per $ invested",
                "ev_per_risk - net EV per unit uncertainty",
                "ev_per_capital_time - net EV per $ per day (capital efficiency)"
            ],
            "costs_modeled": ["fees", "spread (half)", "slippage (amount/liquidity)", "gas (on-chain)", "funding (perps)", "execution_loss (queue/rejection/latency)", "uncertainty_penalty (model disagreement)"]
        }
