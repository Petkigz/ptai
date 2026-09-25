"""
Expected Net EV Engine - V10 FIX #9
Opportunity scoring needs stronger statistical normalization

Current: edge × prob × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)
Problem: Not economically meaningful expected return

Wanted:
Expected Net P&L = Σ probability(outcome) × payoff(outcome) − fees − spread − slippage − funding − gas − expected execution loss − model uncertainty penalty

Then compare expected net return per dollar, per unit risk, per unit capital-time
"""

from typing import Dict, Any, List, Optional
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
    # WHERE EACH COST CAME FROM, and what the trade looks like if the guesses
    # are wrong.
    #
    # The EV is a prediction built partly from the order book (measured), partly
    # from the venue's own declared schedule (declared), and partly from our
    # defaults and models (assumed). Presenting those as one number let a
    # default 2% fee decide a trade as though it had been observed, and it let a
    # venue that charges nothing be charged it anyway. The provenance travels
    # with the result so the gate, the learning record and the operator can all
    # tell which one they are looking at.
    amount_usd: float = 0.0
    cost_provenance: Dict[str, str] = field(default_factory=dict)
    assumed_costs: List[str] = field(default_factory=list)
    assumed_cost_usd: float = 0.0
    # Net EV with every ASSUMED cost doubled and everything measured left alone.
    # If the trade only clears because of a guess, it does not clear.
    stressed_net_ev_usd: float = 0.0

    # An assumption smaller than this is reported but does not make a decision
    # "unmeasured": a $0.004 gas estimate against a $3 stake is noise, and
    # treating it as load-bearing would train everyone to ignore the flag.
    MATERIAL_ASSUMPTION_PCT = 0.001

    def material_assumptions(self, amount_usd: float) -> List[str]:
        """The assumed costs big enough to have moved the decision."""
        if amount_usd <= 0:
            return list(self.assumed_costs)
        if self.assumed_cost_usd >= amount_usd * self.MATERIAL_ASSUMPTION_PCT:
            return list(self.assumed_costs)
        return []

    @property
    def costs_are_measured(self) -> bool:
        """
        True when nothing MATERIAL in this decision was invented.

        `assumed_costs` still lists every assumption, including immaterial ones;
        this is the flag a consumer can gate on without discarding a good trade
        because a gas estimate of four tenths of a cent was in it.
        """
        return not self.material_assumptions(self.amount_usd)

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
            "cost_provenance": dict(self.cost_provenance),
            "assumed_costs": list(self.assumed_costs),
            "assumed_cost_usd": round(self.assumed_cost_usd, 4),
            "stressed_net_ev_usd": round(self.stressed_net_ev_usd, 4),
            "costs_are_measured": self.costs_are_measured,
            "assumed_cost_pct": (round(self.assumed_cost_usd / self.amount_usd, 5)
                                 if self.amount_usd else 0.0),
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
        
        # ------------------------------------------------------------------
        # Costs, and where each one came from.
        #
        # Three sources, and they are not interchangeable:
        #   MEASURED  - read off the book in front of us
        #   DECLARED  - the venue's own published schedule, carried on the
        #               opportunity by the scan that built it
        #   ASSUMED   - our default, because nobody looked
        #
        # The old code wrote `opportunity.fees_pct or 0.02`, which conflates
        # "no fee" with "not known": a venue declaring 0.0 - Manifold and Kalshi
        # both do - was charged the 2% default, and a venue whose schedule had
        # been read was charged the same 2% if its schedule happened to be zero.
        # Both errors move the EV, and the EV decides the trade.
        # ------------------------------------------------------------------
        provenance: Dict[str, str] = {}
        assumed: List[str] = []
        assumed_cost_usd = 0.0

        # Fees
        if opportunity.fees_pct is None:
            fee_pct = 0.02
            provenance["fees"] = "assumed: no venue fee schedule on the opportunity"
            assumed.append("fees")
            assumed_cost_usd += amount_usd * fee_pct
        else:
            fee_pct = float(opportunity.fees_pct)
            provenance["fees"] = f"declared by the venue: {fee_pct*100:.2f}%"
        fees_usd = amount_usd * fee_pct

        # Spread - from orderbook or opportunity
        # read_spread reports whether the value was measured; that flag used to
        # be assigned to `_spread_is_real` and thrown away, so a default spread
        # travelled into the EV indistinguishable from a read one.
        _measured_spread, _spread_is_real = read_spread(orderbook, 0.02)
        if _spread_is_real:
            spread_pct = _measured_spread
            provenance["spread"] = "measured from the orderbook"
        elif opportunity.spread_pct is not None:
            spread_pct = float(opportunity.spread_pct)
            provenance["spread"] = "declared by the venue: orderbook spread"
        else:
            spread_pct = _measured_spread
            provenance["spread"] = "assumed: no measured spread on the book"
            assumed.append("spread")
            assumed_cost_usd += amount_usd * spread_pct * 0.5
        spread_usd = amount_usd * spread_pct * 0.5  # half spread cost (buy at ask)

        # Slippage - from opportunity, the book's own estimate, or depth
        liquidity = getattr(opportunity.market, 'liquidity', 10000)
        depth_slippage = (min(0.05, amount_usd / liquidity * 0.5)
                          if liquidity and liquidity > 0 else 0.0)
        if opportunity.slippage_pct is not None:
            slippage_pct = float(opportunity.slippage_pct)
            provenance["slippage"] = "declared by the venue: book depth"
        elif orderbook.get("slippage_estimate") is not None:
            try:
                slippage_pct = float(orderbook["slippage_estimate"])
                provenance["slippage"] = "measured from the orderbook"
            except (TypeError, ValueError):
                slippage_pct = max(0.01, depth_slippage)
                provenance["slippage"] = "assumed: unreadable slippage estimate"
                assumed.append("slippage")
                assumed_cost_usd += amount_usd * slippage_pct
        else:
            # Depth against our own size is an ESTIMATE, not a measurement: it
            # assumes the book is uniform. Conservative on purpose, and labelled.
            slippage_pct = max(0.01, depth_slippage)
            provenance["slippage"] = ("assumed: liquidity-based estimate, no book"
                                      if not depth_slippage
                                      else "assumed: amount/liquidity estimate")
            assumed.append("slippage")
            assumed_cost_usd += amount_usd * slippage_pct
        slippage_usd = amount_usd * slippage_pct

        # Gas - from the venue's own model, not a constant.
        #
        # This was `0.05` for Polymarket and AFX and 0 elsewhere. On a $3 order
        # that is 1.7% of the stake charged to a venue whose order flow is
        # relayed off-chain - a fabricated cost that made marginal real edges
        # look negative. The number now comes from the one place that models it,
        # and it says which kind of number it is.
        gas_usd, gas_note = self._gas_usd(opportunity, orderbook, amount_usd)
        provenance["gas"] = gas_note
        if gas_note.startswith("assumed") and gas_usd > 0:
            assumed.append("gas")
            assumed_cost_usd += gas_usd
        
        # Funding - for perps/futures
        funding_usd = 0.0
        if "PERP" in opportunity.market.id or "futures" in str(opportunity.market.raw.get("type", "")).lower():
            # Funding ~0.01% per 8h, ~0.03% per day
            hours = opportunity.time_to_resolution_hours or 24
            funding_rate_daily = 0.0003
            funding_usd = amount_usd * funding_rate_daily * (hours / 24)
        
        # Execution loss - queue position, rejection, latency.
        #
        # With no real book there is nothing to measure execution against, so
        # the 2% is an assumption and is labelled as one rather than being
        # presented as a modelled cost.
        is_real_ob = orderbook.get("is_real", False)
        execution_quality = opportunity.execution_quality or 0.5
        if not is_real_ob:
            execution_loss_pct = 0.02
            provenance["execution_loss"] = "assumed: no real orderbook to measure"
            assumed.append("execution_loss")
            assumed_cost_usd += amount_usd * execution_loss_pct
        else:
            execution_loss_pct = (1.0 - execution_quality) * 0.02
            provenance["execution_loss"] = (
                f"measured from the orderbook: execution quality "
                f"{execution_quality:.2f}")
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

        # How does the trade look if the costs WE GUESSED are twice as bad?
        #
        # Only the assumed ones move: a measured spread is a measurement and
        # doubling it would be inventing pessimism. This is the rule that makes
        # an assumption cost something - a trade whose only path to positive EV
        # runs through a default 2% fee is not a trade PTAI has evidence for.
        stressed_net_ev_usd = (
            gross_ev_usd - total_costs - assumed_cost_usd
            if assumed_cost_usd > 0 else net_ev_usd
        )

        # Should trade if net EV >0 and meets thresholds
        should_trade = (
            net_ev_usd > 0 and
            net_ev_pct >= 0.03 and  # at least 3% net after all costs
            mispricing >= HUNT_MISPRICING_MIN and  # 8% gross, costs charged below
            confidence >= 0.60 and
            # ...and it still clears when the guesses are wrong.
            (stressed_net_ev_usd > 0 or not assumed_cost_usd > 0)
        )
        
        reasoning = (
            f"Gross EV ${gross_ev_usd:.2f} ({gross_ev_pct*100:.1f}%) = fair {fair_prob:.2f} vs mkt {market_price:.2f} edge {edge*100:.1f}% conf {confidence:.2f} * ${amount_usd:.2f} | "
            f"Costs: fees ${fees_usd:.2f} ({fee_pct*100:.1f}%) spread ${spread_usd:.2f} ({spread_pct*100:.1f}%) slippage ${slippage_usd:.2f} ({slippage_pct*100:.1f}%) "
            f"gas ${gas_usd:.2f} funding ${funding_usd:.2f} exec_loss ${execution_loss_usd:.2f} unc_penalty ${uncertainty_penalty_usd:.2f} = total ${total_costs:.2f} | "
            f"Net EV ${net_ev_usd:.2f} ({net_ev_pct*100:.1f}%) per $ {ev_per_dollar*100:.1f}% per risk {ev_per_risk:.2f} per $/day {ev_per_capital_time*100:.2f}% | "
            f"Cost provenance: {', '.join(f'{k}={v}' for k, v in provenance.items())} | "
            f"Assumed ${assumed_cost_usd:.2f} of ${total_costs:.2f} total "
            f"({', '.join(assumed) if assumed else 'nothing assumed'}) - "
            f"net EV if the assumptions are doubled: ${stressed_net_ev_usd:.2f} | "
            f"Should trade: {should_trade} (net>0, net%>=3%, "
            f"mispricing {mispricing*100:.1f}%>=8% raw, conf>=60%, and positive "
            f"under cost stress - effective edge {edge*100:.1f}% is "
            f"informational, the costs are charged once in net EV)"
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
            should_trade=should_trade,
            cost_provenance=provenance,
            assumed_costs=assumed,
            assumed_cost_usd=assumed_cost_usd,
            stressed_net_ev_usd=stressed_net_ev_usd,
            amount_usd=amount_usd,
        )
    
    def _gas_usd(self, opportunity, orderbook: Dict, amount_usd: float):
        """
        Gas for this order, from the model that owns the number.

        Returns (usd, provenance). The previous implementation hardcoded $0.05
        for Polymarket and AFX DEX and nothing anywhere else - a constant with
        no source, charged as 1.7% of a $3 stake to a venue whose orders are
        relayed rather than sent on-chain. `GasModel` already models this per
        venue and per operation; using it means one number, one place, and a
        default that can be argued with instead of a literal nobody can trace.
        """
        venue = str(getattr(opportunity, "venue_id", "") or "")

        # The venue's own mechanism, when it declares one. This is a fact about
        # how orders reach the book, not an estimate of a price.
        declared = getattr(opportunity, "order_gas_usd", None)
        if declared is not None:
            declared = float(declared)
            if declared <= 0:
                return 0.0, (
                    f"not applicable: {venue or 'this venue'} relays orders - "
                    f"the operator pays no gas to place one")
            return declared, f"declared by the venue: ${declared:.4f} per order"

        try:
            from ..execution.gas import GasModel

            result = GasModel().calculate_gas(
                operation="place_order", amount_usd=amount_usd,
                venue_id=venue or "unknown")
            usd = float(getattr(result, "gas_usd", 0.0) or 0.0)
            if usd <= 0:
                return 0.0, (
                    f"not applicable: {venue} settles off-chain "
                    f"({getattr(result, 'reasoning', '')[:80]})")
            # The model's own inputs are defaults (gas price, token price), so
            # this is an estimate and is labelled one even though it is not a
            # literal typed into this function.
            return usd, (f"assumed: GasModel estimate ${usd:.4f} "
                         f"({getattr(result, 'chain', 'unknown')})")
        except Exception as e:
            logger.warning(
                f"Gas model unavailable for {venue or 'unknown venue'}: "
                f"{type(e).__name__}: {e} - charging the on-chain venues at the "
                f"model's old default and labelling it assumed")
            usd = 0.05 if venue in ("polymarket", "afx_dex") else 0.0
            if usd <= 0:
                return 0.0, "not applicable: no gas model and not a known on-chain venue"
            return usd, "assumed: gas model unavailable, legacy default"

    def rank_by_net_ev(self, opportunities: list, amount_per_opp: float = 3.0) -> list:
        """Rank opportunities by expected net EV per dollar per risk"""
        scored = []
        for opp in opportunities:
            ev = self.calculate(opp, amount_per_opp)
            # Score = net_ev * confidence * liquidity * execution / (uncertainty + risk)
            # But now economically meaningful
            # `opp.fees_pct` is None when nobody measured it; treating that as 0
            # in a denominator makes an unmeasured opportunity look CHEAPER, so
            # the same conservative defaults as the EV itself are used.
            _fees = opp._cost_or_default(opp.fees_pct, 0.02)
            _slip = opp._cost_or_default(opp.slippage_pct, 0.01)
            score = ev.net_ev_usd * opp.confidence * opp.liquidity_score * opp.execution_quality / max(0.01, opp.uncertainty + _fees + _slip)
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
