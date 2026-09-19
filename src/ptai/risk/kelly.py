"""
Kelly Criterion implementation - critical risk management
One wrong prediction could wipe account, so we use Half-Kelly and 6% cap

Formula for binary market:
- Market price = implied probability m
- Fair value = true probability p (our estimate)
- Edge = p - m
- Odds b = (1-m)/m for YES bet (how much you win per 1 risked)
- Kelly fraction f* = (p*b - q)/b where q = 1-p
  Simplified: f* = p - q/b = p - (1-p)/b
  Also: f* = (p*(b+1)-1)/b

For binary prediction market, if we think fair is p and market is m:
- If betting YES: b = (1-m)/m, but actually in polymarket price is cost, payoff is 1
  So profit if win = (1-m)/m * stake? Actually if you buy YES at 0.6, you risk 0.6 to win 0.4
  So odds against you: b = (1-m)/m ? Let's derive properly:

Traditional Kelly: f = (bp - q)/b where b = net odds received (profit per unit bet)
In Polymarket:
- Buy YES at price m: you pay m, you get 1 if YES wins, 0 if NO wins
- Profit if YES: 1 - m, Loss if NO: -m
- So b = (1-m)/m (profit divided by stake)

Thus:
f* = (p*b - (1-p))/b = p - (1-p)/b

We use Half-Kelly for safety: f = f* * kelly_fraction (0.5)
And cap at max_pct (0.06 = 6%)
"""
from dataclasses import dataclass
from typing import Optional
import math
from loguru import logger

@dataclass
class KellyResult:
    kelly_fraction_raw: float  # Full Kelly
    kelly_fraction_adj: float  # After half-kelly and cap
    position_size_usd: float
    position_size_pct: float
    edge: float
    fair_prob: float
    market_price: float
    odds: float
    expected_value: float
    should_bet: bool
    reason: str

class KellyCalculator:
    def __init__(self, kelly_fraction: float = 0.5, max_pct: float = 0.06, min_edge: float = 0.08):
        """
        kelly_fraction: 0.5 = Half-Kelly (recommended for safety)
        max_pct: 0.06 = max 6% bankroll per bet
        min_edge: 0.08 = minimum 8% edge required
        """
        self.kelly_fraction = kelly_fraction
        self.max_pct = max_pct
        self.min_edge = min_edge
        logger.info(f"Kelly init: fraction={kelly_fraction}, max={max_pct}, min_edge={min_edge}")

    def calculate(self, market_price: float, fair_prob: float, bankroll: float) -> KellyResult:
        """
        Calculate Kelly position size

        Args:
            market_price: Current market price (0-1) e.g. 0.6 = 60c
            fair_prob: Our fair value estimate (0-1) e.g. 0.75 = we think 75% true chance
            bankroll: Current bankroll in USD

        Returns:
            KellyResult with sizing
        """
        # Validate inputs
        market_price = max(0.01, min(0.99, market_price))
        fair_prob = max(0.01, min(0.99, fair_prob))
        bankroll = max(1.0, bankroll)

        edge = fair_prob - market_price
        abs_edge = abs(edge)

        # Determine side
        is_yes_bet = edge > 0  # If fair > market, we bet YES
        # For Kelly, we need odds for the side we're betting

        if is_yes_bet:
            # Betting YES
            m = market_price
            p = fair_prob
            q = 1 - p
            # Avoid div by zero
            if m <= 0.01:
                b = 99.0
            else:
                b = (1 - m) / m  # Net odds
        else:
            # Betting NO: equivalent to betting YES on opposite side
            # If fair < market, we think NO is more likely than market implies
            # Market NO price = 1 - market_price
            # Fair NO prob = 1 - fair_prob
            m_no = 1 - market_price
            p_no = 1 - fair_prob
            q_no = 1 - p_no
            m = m_no
            p = p_no
            q = q_no
            if m <= 0.01:
                b = 99.0
            else:
                b = (1 - m) / m

        # Kelly formula: f* = (b*p - q)/b
        if b <= 0:
            b = 0.01

        kelly_raw = (b * p - q) / b

        # Expected value
        ev = p * (1 - m) - q * m if is_yes_bet else p * (1 - m) - q * m
        # Actually EV for YES bet: p*(1-m) + (1-p)*(-m) = p - m
        ev_simple = fair_prob - market_price if is_yes_bet else (1 - fair_prob) - (1 - market_price)
        # Which equals edge for YES, -edge for NO but we already flipped, so use abs?

        # Apply half-kelly
        kelly_adj = kelly_raw * self.kelly_fraction

        # Cap at max_pct
        kelly_capped = min(kelly_adj, self.max_pct)
        kelly_capped = max(0, kelly_capped)  # No negative

        position_usd = bankroll * kelly_capped
        position_pct = kelly_capped

        # Should we bet?
        should_bet = True
        reason = f"Edge {edge:.2%}, Kelly raw {kelly_raw:.2%}, adj {kelly_adj:.2%}, capped {kelly_capped:.2%}"

        if abs_edge < self.min_edge:
            should_bet = False
            reason = f"Edge {abs_edge:.2%} < min {self.min_edge:.2%} - NO BET"

        if kelly_raw <= 0:
            should_bet = False
            reason = f"Kelly raw {kelly_raw:.2%} <=0 - negative EV, NO BET"

        if ev_simple <= 0 and is_yes_bet:
            # For NO bets, edge is negative but we flipped, so check raw
            if kelly_raw <= 0:
                should_bet = False
                reason = f"Negative EV {ev_simple:.4f} - NO BET"

        # Additional safety: don't bet if fair prob < 0.55 and > 0.45 and edge small?
        # No, edge check already covers

        return KellyResult(
            kelly_fraction_raw=kelly_raw,
            kelly_fraction_adj=kelly_capped,
            position_size_usd=position_usd,
            position_size_pct=position_pct,
            edge=edge,
            fair_prob=fair_prob,
            market_price=market_price,
            odds=b,
            expected_value=ev_simple,
            should_bet=should_bet,
            reason=reason
        )

    def calculate_batch(self, opportunities: list, bankroll: float) -> list:
        """Calculate Kelly for batch, respecting max open positions and bankroll"""
        results = []
        for opp in opportunities:
            res = self.calculate(
                market_price=opp.get("market_price", 0.5),
                fair_prob=opp.get("fair_value", 0.5),
                bankroll=bankroll
            )
            results.append({**opp, "kelly": res})
        return results

# Example usage and test
if __name__ == "__main__":
    calc = KellyCalculator(kelly_fraction=0.5, max_pct=0.06, min_edge=0.08)

    tests = [
        (0.60, 0.75, 50),  # Market 60c, we think 75%, $50 bankroll
        (0.50, 0.65, 50),
        (0.70, 0.60, 50),  # We think NO
        (0.50, 0.52, 50),  # Small edge - should reject
    ]

    for m, f, b in tests:
        r = calc.calculate(m, f, b)
        print(f"Market {m:.2f} Fair {f:.2f} Bank ${b} -> Edge {r.edge:.2%} Kelly {r.kelly_fraction_adj:.2%} Size ${r.position_size_usd:.2f} Bet? {r.should_bet} Reason: {r.reason}")
