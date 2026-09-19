"""
Gas Cost Model - Polygon gas for Polymarket, critical for $50 bankroll
While Polygon gas is cheap, few cents per transaction is significant % of capital

For $50 bankroll, max position 6% = $3
Gas $0.05 = 1.6% of $3 position
Gas $0.10 = 3.3% of $3 position
Gas $0.20 = 6.6% of $3 position

Must include gas in effective edge calculation
"""
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger


@dataclass
class GasResult:
    venue_id: str
    chain: str
    gas_usd: float
    gas_pct_of_position: float
    operation: str  # approve, trade, cancel, etc
    reasoning: str


class GasModel:
    def __init__(self):
        # Polygon gas prices - cheap but not zero
        # Typical Polygon gas: 30-100 gwei, MATIC ~$0.5
        # Simple tx: ~50k gas, complex CLOB: ~100-200k gas
        self.polygon_gas_price_gwei = 50  # average
        self.matic_price_usd = 0.5
        self.gas_limits = {
            "approve_usdc": 50000,
            "place_order": 150000,
            "cancel_order": 80000,
            "claim_winnings": 100000,
            "transfer": 21000
        }

    def calculate_gas(self, operation: str = "place_order", amount_usd: float = 3.0, venue_id: str = "polymarket") -> GasResult:
        if "polymarket" not in venue_id.lower():
            # Non-Polygon venues: gas 0 or included in fees
            return GasResult(
                venue_id=venue_id,
                chain="off-chain" if "kalshi" in venue_id.lower() or "manifold" in venue_id.lower() else "centralized",
                gas_usd=0.0,
                gas_pct_of_position=0.0,
                operation=operation,
                reasoning=f"Gas 0 for {venue_id} - off-chain or centralized, fees cover it"
            )
        
        # Polygon gas calculation
        # gas_cost_matic = gas_limit * gas_price_gwei * 1e-9
        # gas_cost_usd = gas_cost_matic * matic_price_usd
        gas_limit = self.gas_limits.get(operation, 100000)
        gas_cost_matic = gas_limit * self.polygon_gas_price_gwei * 1e-9
        gas_cost_usd = gas_cost_matic * self.matic_price_usd
        
        # But also need to consider USDC approval once, and possible price fluctuations
        # Conservative: use $0.02-$0.10 range
        # For PTAI, use $0.05 as conservative average per trade
        conservative_gas = max(gas_cost_usd, 0.05)  # min $0.05
        
        gas_pct = conservative_gas / amount_usd if amount_usd > 0 else 0
        
        reasoning = (
            f"Polygon gas: {gas_limit} gas * {self.polygon_gas_price_gwei} gwei = {gas_cost_matic:.6f} MATIC * ${self.matic_price_usd} = ${gas_cost_usd:.4f} "
            f"conservative ${conservative_gas:.4f} | Operation {operation} | "
            f"Amount ${amount_usd:.2f} | Gas {gas_pct*100:.2f}% of position | "
            f"At $50 bankroll $3 position, gas $0.05 = 1.6%, $0.10=3.3% - significant!"
        )
        
        return GasResult(
            venue_id=venue_id,
            chain="polygon",
            gas_usd=conservative_gas,
            gas_pct_of_position=gas_pct,
            operation=operation,
            reasoning=reasoning
        )

    def get_gas_report(self, bankroll: float = 50.0) -> Dict:
        position = bankroll * 0.06
        operations = ["approve_usdc", "place_order", "cancel_order", "claim_winnings"]
        results = {}
        for op in operations:
            results[op] = self.calculate_gas(operation=op, amount_usd=position, venue_id="polymarket")
        
        return {
            "bankroll": bankroll,
            "position_6pct": position,
            "polygon_gas_price_gwei": self.polygon_gas_price_gwei,
            "matic_price_usd": self.matic_price_usd,
            "operations": {
                op: {"gas_usd": r.gas_usd, "gas_pct": r.gas_pct_of_position, "chain": r.chain}
                for op, r in results.items()
            },
            "reality_check": f"Gas cheap but not negligible for $50. $0.05 gas = {0.05/position*100:.1f}% of ${position:.2f} position. 10 trades/day = $0.50 gas/day = {0.50/bankroll*100:.1f}% of bankroll daily just gas. Must include in effective edge.",
            "total_cost_per_trade": f"Fees 1.5-3% + gas 1.6% + spread 2% + slippage 1% = ~6% total cost. Edge must >6% to break even, >8% to trade per PTAI rules."
        }
