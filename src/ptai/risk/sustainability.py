"""
Sustainability Calculator - $50 Challenge Math
Realistic assessment of turning $50 into enough to pay for itself

The $50 bankroll is single biggest constraint. Forces small positions, magnifies fee/gas impact.

To pay for itself ($10-20/month VPS/data), need 20-40% monthly return on $50.
Requires extraordinary, consistent edge or extreme luck.

Most valuable outcome isn't $50, it's infrastructure and knowledge.
If you can make $50 bot profitable after fees, you have system that can scale - that's real prize.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger
import math


@dataclass
class SustainabilityResult:
    bankroll: float
    monthly_cost: float
    required_monthly_return_pct: float
    required_daily_profit: float
    max_position_usd: float
    fee_per_trade: float
    gas_per_trade: float
    total_cost_per_trade_pct: float
    break_even_edge: float
    avg_profit_per_trade_at_10pct: float
    trades_needed_daily: float
    win_rate_needed: float
    is_sustainable: bool
    reality_check: str
    roadmap: str


class SustainabilityCalculator:
    def __init__(self, monthly_cost: float = 15.0, fee_pct: float = 0.02, gas_usd: float = 0.05):
        self.monthly_cost = monthly_cost
        self.fee_pct = fee_pct
        self.gas_usd = gas_usd

    def calculate(self, bankroll: float = 50.0, avg_edge_after_costs: float = 0.05, win_rate: float = 0.55) -> SustainabilityResult:
        """
        Calculate sustainability for $50 challenge
        
        Args:
            bankroll: Current bankroll $50
            avg_edge_after_costs: Average edge after fees/gas/spread/slippage (e.g. 5% = 0.05)
            win_rate: Win rate (e.g. 0.55 = 55%)
        """
        max_position = bankroll * 0.06  # 6% Kelly cap = $3 on $50
        
        fee_per_trade = max_position * self.fee_pct  # $3 * 2% = $0.06
        total_cost_pct = self.fee_pct + (self.gas_usd / max_position) + 0.02 + 0.01  # fee + gas + spread 2% + slippage 1%
        # At $50: fee 2% + gas 1.6% + spread 2% + slippage 1% = 6.6% total cost
        break_even_edge = total_cost_pct  # Must exceed this to break even
        
        # Required returns
        required_monthly_return_pct = self.monthly_cost / bankroll  # $15 / $50 = 30%
        required_daily_profit = (bankroll * required_monthly_return_pct) / 30  # $0.50 daily
        
        # Profit per trade at avg edge
        avg_profit_per_trade = max_position * avg_edge_after_costs  # $3 * 5% = $0.15
        
        trades_needed_daily = required_daily_profit / avg_profit_per_trade if avg_profit_per_trade > 0 else 999
        
        # Win rate needed: need positive EV
        # EV = win_rate * avg_win - (1-win_rate) * avg_loss
        # For binary 50/50 payoff, need win_rate > 50% + edge
        # Simplified: win_rate_needed = 0.5 + break_even_edge/2
        win_rate_needed = 0.5 + break_even_edge
        
        is_sustainable = avg_edge_after_costs > break_even_edge and win_rate > win_rate_needed and trades_needed_daily < 5
        
        reality_check = (
            f"Bankroll ${bankroll:.2f} max position ${max_position:.2f} (6%) | "
            f"Fee ${fee_per_trade:.4f} ({self.fee_pct*100:.1f}%) gas ${self.gas_usd:.4f} ({self.gas_usd/max_position*100:.1f}%) total cost {total_cost_pct*100:.1f}% | "
            f"Break-even edge {break_even_edge*100:.1f}% must exceed to profit | "
            f"Monthly cost ${self.monthly_cost:.2f} requires {required_monthly_return_pct*100:.0f}% monthly return = ${required_daily_profit:.2f} daily | "
            f"At {avg_edge_after_costs*100:.0f}% avg edge after costs, profit ${avg_profit_per_trade:.2f}/trade, need {trades_needed_daily:.1f} winning trades/day | "
            f"Win rate needed >{win_rate_needed*100:.0f}% vs actual {win_rate*100:.0f}% | "
            f"Sustainable: {is_sustainable} | "
            f"Reality: $50 is learning budget, not income. Need extraordinary edge or luck for 20-40% monthly. Most valuable is infrastructure and knowledge that scales."
        )
        
        roadmap = (
            f"Phase 1 Read-Only Week 1-2: Build data ingestion (Polymarket SDK, news RSS), LLM fair-value engine outputs only, log predictions vs market for week, see if remotely accurate | "
            f"Phase 2 Paper Trading Week 3-4: Add risk engine, order simulation, run full loop paper mode, track P/L, fees, slippage, discover if edge real | "
            f"Phase 3 Live Tiny Month 2+: Connect API real money, start $1 positions, validate execution pipeline not make money, gradually increase to 6% Kelly cap if works | "
            f"Phase 4 Scale: If $50 bot profitable after fees, system can scale - that's real prize"
        )
        
        return SustainabilityResult(
            bankroll=bankroll,
            monthly_cost=self.monthly_cost,
            required_monthly_return_pct=required_monthly_return_pct,
            required_daily_profit=required_daily_profit,
            max_position_usd=max_position,
            fee_per_trade=fee_per_trade,
            gas_per_trade=self.gas_usd,
            total_cost_per_trade_pct=total_cost_pct,
            break_even_edge=break_even_edge,
            avg_profit_per_trade_at_10pct=avg_profit_per_trade,
            trades_needed_daily=trades_needed_daily,
            win_rate_needed=win_rate_needed,
            is_sustainable=is_sustainable,
            reality_check=reality_check,
            roadmap=roadmap
        )

    def get_detailed_report(self, bankroll: float = 50.0) -> Dict:
        """Detailed sustainability report for dashboard"""
        from ..markets.fees import FeeEngine
        from ..execution.gas import GasModel
        
        fee_engine = FeeEngine()
        gas_model = GasModel()
        
        fee_report = fee_engine.get_sustainability_report(bankroll=bankroll)
        gas_report = gas_model.get_gas_report(bankroll=bankroll)
        
        scenarios = []
        for edge in [0.03, 0.05, 0.08, 0.10, 0.15]:
            for win_rate in [0.52, 0.55, 0.60]:
                result = self.calculate(bankroll=bankroll, avg_edge_after_costs=edge, win_rate=win_rate)
                scenarios.append({
                    "edge_after_costs": edge,
                    "win_rate": win_rate,
                    "trades_needed_daily": result.trades_needed_daily,
                    "is_sustainable": result.is_sustainable,
                    "daily_profit_needed": result.required_daily_profit,
                    "profit_per_trade": result.avg_profit_per_trade_at_10pct
                })
        
        # Best and worst case
        best_case = self.calculate(bankroll=bankroll, avg_edge_after_costs=0.15, win_rate=0.60)
        realistic_case = self.calculate(bankroll=bankroll, avg_edge_after_costs=0.05, win_rate=0.55)
        worst_case = self.calculate(bankroll=bankroll, avg_edge_after_costs=0.03, win_rate=0.52)
        
        return {
            "bankroll": bankroll,
            "fee_report": fee_report,
            "gas_report": gas_report,
            "scenarios": scenarios,
            "cases": {
                "best_15pct_edge_60wr": {
                    "trades_needed": best_case.trades_needed_daily,
                    "sustainable": best_case.is_sustainable,
                    "reality": best_case.reality_check
                },
                "realistic_5pct_edge_55wr": {
                    "trades_needed": realistic_case.trades_needed_daily,
                    "sustainable": realistic_case.is_sustainable,
                    "reality": realistic_case.reality_check
                },
                "worst_3pct_edge_52wr": {
                    "trades_needed": worst_case.trades_needed_daily,
                    "sustainable": worst_case.is_sustainable,
                    "reality": worst_case.reality_check
                }
            },
            "roadmap": best_case.roadmap,
            "red_flags": [
                "LLM Hallucination: LLM will confidently make things up. Risk engine and hard limits only protection",
                "Overfitting: If you tweak 8% threshold or Kelly fraction until backtest looks good, it will fail live. Keep simple",
                "API Changes: Polymarket API can change, bot will break, plan maintenance",
                "$50 Ceiling: Even perfect edge, $50 will not generate life-changing money quickly. It's learning budget. Be prepared to lose it all",
                "Fee Magnification: $3 position $0.045 fee 1.5% + gas 1.6% + spread 2% = 5.1% cost must exceed to break even",
                "Liquidity: Thin markets high slippage, stick to >$10k volume",
                "Gas: Polygon cheap but $0.05 = 1.6% of $3 position, 10 trades/day $0.50 = 1% bankroll daily gas"
            ],
            "critical_rules": [
                "Never use leverage - Polymarket buying shares, don't margin",
                "Diversify - Don't put all $3 into one market, split 2-3 uncorrelated",
                "Cut losses fast - If position drops 30-40%, close it, thesis wrong",
                "Take profits - If gains 50%, sell half, lock in",
                "Beware thin markets - Low liquidity high slippage, stick >$10k volume",
                "Daily loss limit $5 - Stop trading day if hit",
                "Max open positions 3 - 18% max exposure (3×6%)",
                "Kill switch - Physical/software immediately cancel orders and stop loop"
            ],
            "math": {
                "fee_formula": "Fee = 0.06 × C × p × (1-p), at $0.50 $1.50 per 100 contracts, $3 position 6 contracts fee $0.09 (3%)",
                "kelly_formula": "f = (bp - q)/b, b=(1-m)/m, p=fair prob, q=1-p, Half-Kelly 0.5×, cap 6% $3 on $50, $1 min",
                "position_sizing_example": "Market 60c, fair 75%, edge 15%, b=(1-0.6)/0.6=0.666, Kelly raw (0.666*0.75-0.25)/0.666=37%, Half-Kelly 18.5%, capped 6% → $3 on $50",
                "required_return": "$10-20/month cost on $50 = 20-40% monthly return, extraordinary edge or luck required"
            }
        }
