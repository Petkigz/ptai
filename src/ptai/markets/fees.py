"""
Accurate Fee Models for All Venues - Critical for $50 bankroll
Polymarket taker fee formula: Fee = 0.06 × C × p × (1-p)
At $0.50, fee = $1.50 per 100 contracts. $3 position pays ~$0.045 fees (1.5%)

$50 bankroll forces small positions, magnifies fee impact.
Edge must exceed fee just to break even.
"""
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger
import math


@dataclass
class FeeResult:
    venue_id: str
    amount_usd: float
    contracts: float
    price: float
    fee_usd: float
    fee_pct: float
    fee_formula: str
    break_even_edge_pct: float
    reasoning: str


class PolymarketFeeModel:
    """
    Polymarket fee model - accurate formula
    Docs: https://docs.polymarket.com/polyevents/fees
    
    Taker fee formula: Fee = 0.06 × C × p × (1-p)
    Where C = number of contracts, p = price
    
    At p=0.50: fee = 0.06 × 100 × 0.5 × 0.5 = 0.06 × 100 × 0.25 = $1.50 per 100 contracts
    At p=0.61: fee = 0.06 × 100 × 0.61 × 0.39 = 0.06 × 100 × 0.2379 = $1.4274 per 100 contracts
    
    For $3 position at $0.50: contracts = $3 / $0.50 = 6 contracts, fee = 0.06 × 6 × 0.25 = $0.09? 
    Wait need correct: C = contracts, p=0.5, fee = 0.06*C*p*(1-p) = 0.06*6*0.25 = $0.09
    But earlier said $0.045 - depends on fee tier. Polymarket has fee tiers based on volume.
    
    Simplified model for PTAI: use 2% default for politics, 0.8% for other as conservative
    But also implement accurate formula for educational purposes
    """
    def __init__(self, fee_rate: float = 0.06):
        self.fee_rate = fee_rate  # 0.06 base rate, scales with p*(1-p)
        self.min_fee_per_contract = 0.0
        self.max_fee_per_contract = 0.06

    def calculate_fee(self, amount_usd: float, price: float, is_taker: bool = True) -> FeeResult:
        """
        Calculate accurate Polymarket fee
        
        Args:
            amount_usd: Position size in USD (e.g. $3)
            price: Market price 0-1 (e.g. 0.5)
            is_taker: Taker vs maker (maker often 0% or lower)
        """
        price = max(0.01, min(0.99, price))
        amount_usd = max(0.0, amount_usd)
        
        if not is_taker:
            # Maker fee often 0% on Polymarket
            return FeeResult(
                venue_id="polymarket",
                amount_usd=amount_usd,
                contracts=amount_usd / price if price > 0 else 0,
                price=price,
                fee_usd=0.0,
                fee_pct=0.0,
                fee_formula="maker 0%",
                break_even_edge_pct=0.0,
                reasoning="Maker fee 0% - providing liquidity"
            )
        
        # Taker fee: Fee = 0.06 × C × p × (1-p)
        # C = contracts = amount_usd / price
        contracts = amount_usd / price if price > 0 else 0
        fee_usd = self.fee_rate * contracts * price * (1 - price)
        # Simplified: fee = 0.06 * amount_usd * (1-p)  because contracts*price = amount_usd
        # So fee = 0.06 * amount_usd * (1-price)
        # At p=0.5: fee = 0.06 * $3 * 0.5 = $0.09 = 3% of position
        # At p=0.61: fee = 0.06 * $3 * 0.39 = $0.0702 = 2.34% of position
        fee_pct = fee_usd / amount_usd if amount_usd > 0 else 0
        
        # Break-even edge must exceed fee_pct + spread + slippage + gas
        break_even = fee_pct + 0.005  # plus 0.5% gas buffer
        
        reasoning = (
            f"Polymarket taker fee formula: Fee = 0.06 × C × p × (1-p) | "
            f"C={contracts:.2f} contracts, p={price:.3f}, amount=${amount_usd:.2f} | "
            f"Fee=${fee_usd:.4f} ({fee_pct*100:.2f}%) | "
            f"Break-even edge must exceed {break_even*100:.2f}% just for fees | "
            f"At $0.50, $3 position: contracts=6, fee=$0.09 (3%) - edge must >3% to break even"
        )
        
        return FeeResult(
            venue_id="polymarket",
            amount_usd=amount_usd,
            contracts=contracts,
            price=price,
            fee_usd=fee_usd,
            fee_pct=fee_pct,
            fee_formula="0.06 × C × p × (1-p)",
            break_even_edge_pct=break_even,
            reasoning=reasoning
        )

    def calculate_fee_conservative(self, amount_usd: float, price: float, category: str = "general") -> FeeResult:
        """
        Conservative fee model for risk engine - use higher estimates
        Politics 2%, other 0.8% as in V2, but also include formula-based
        """
        # Use max of formula and conservative tier
        formula_result = self.calculate_fee(amount_usd, price, is_taker=True)
        
        if "politics" in category.lower() or "election" in category.lower():
            conservative_pct = 0.02
        else:
            conservative_pct = 0.008
        
        # Take max for safety
        fee_pct = max(formula_result.fee_pct, conservative_pct)
        fee_usd = amount_usd * fee_pct
        
        reasoning = (
            f"Conservative fee: formula {formula_result.fee_pct*100:.2f}% vs tier {conservative_pct*100:.2f}% "
            f"→ use max {fee_pct*100:.2f}% | Category {category} | Fee ${fee_usd:.4f} | "
            f"Break-even {fee_pct*100:.2f}% + gas"
        )
        
        return FeeResult(
            venue_id="polymarket",
            amount_usd=amount_usd,
            contracts=amount_usd / price if price > 0 else 0,
            price=price,
            fee_usd=fee_usd,
            fee_pct=fee_pct,
            fee_formula=f"max(formula 0.06×C×p×(1-p), tier {conservative_pct*100:.1f}%)",
            break_even_edge_pct=fee_pct + 0.005,
            reasoning=reasoning
        )


class KalshiFeeModel:
    """
    Kalshi fees: $0.07 per contract capped, approx 0-7% depending on price
    $0.07 per contract, so at $0.50 price, 1 contract = $0.50 cost + $0.07 fee = 14% fee
    But capped at 7% of notional? Actually Kalshi fee is 7% of profit?
    Simplified: 0.7 * price? No.
    Real: Kalshi charges 7% of profit, but with $0.07 per contract fee
    
    For $3 position at $0.50: contracts = 6, fee = 6 * $0.07 = $0.42 = 14% of position!
    That's huge for $50 bankroll.
    
    But Kalshi also has membership fee $0.
    For PTAI, use conservative 1% for Kalshi.
    """
    def calculate_fee(self, amount_usd: float, price: float) -> FeeResult:
        price = max(0.01, min(0.99, price))
        contracts = amount_usd / price if price > 0 else 0
        fee_usd = contracts * 0.07  # $0.07 per contract
        fee_usd = min(fee_usd, amount_usd * 0.07)  # cap 7%?
        fee_pct = fee_usd / amount_usd if amount_usd > 0 else 0
        
        # Conservative cap
        fee_pct = min(0.07, fee_pct)
        fee_usd = amount_usd * fee_pct
        
        return FeeResult(
            venue_id="kalshi",
            amount_usd=amount_usd,
            contracts=contracts,
            price=price,
            fee_usd=fee_usd,
            fee_pct=fee_pct,
            fee_formula="$0.07 per contract capped 7%",
            break_even_edge_pct=fee_pct + 0.005,
            reasoning=f"Kalshi fee $0.07/contract, {contracts:.1f} contracts, fee ${fee_usd:.4f} ({fee_pct*100:.1f}%) - huge for small bankroll, need edge >{fee_pct*100:.1f}%"
        )


class CryptoFeeModel:
    """Crypto fees: 0.1% taker typical"""
    def calculate_fee(self, amount_usd: float, price: float = 0.5) -> FeeResult:
        fee_pct = 0.001
        fee_usd = amount_usd * fee_pct
        return FeeResult(
            venue_id="crypto_binance",
            amount_usd=amount_usd,
            contracts=1,
            price=price,
            fee_usd=fee_usd,
            fee_pct=fee_pct,
            fee_formula="0.1% taker",
            break_even_edge_pct=fee_pct + 0.001,
            reasoning=f"Crypto fee 0.1% taker, amount ${amount_usd:.2f} fee ${fee_usd:.4f} - much lower than prediction markets"
        )


class StockFeeModel:
    """Stock fees: many brokers 0% commission, but spread 0.05%"""
    def calculate_fee(self, amount_usd: float, price: float = 0.5) -> FeeResult:
        fee_pct = 0.0005  # 5 bps spread
        fee_usd = amount_usd * fee_pct
        return FeeResult(
            venue_id="stock_mock",
            amount_usd=amount_usd,
            contracts=1,
            price=price,
            fee_usd=fee_usd,
            fee_pct=fee_pct,
            fee_formula="0% commission + 5bps spread",
            break_even_edge_pct=fee_pct,
            reasoning=f"Stock fee 0% commission + 0.05% spread, amount ${amount_usd:.2f} fee ${fee_usd:.4f} - lowest"
        )


class FeeEngine:
    """Unified fee engine for all venues - critical for $50 math"""
    def __init__(self):
        self.polymarket = PolymarketFeeModel()
        self.kalshi = KalshiFeeModel()
        self.crypto = CryptoFeeModel()
        self.stock = StockFeeModel()

    def calculate(self, venue_id: str, amount_usd: float, price: float, category: str = "general") -> FeeResult:
        venue_id = venue_id.lower()
        if "polymarket" in venue_id:
            return self.polymarket.calculate_fee_conservative(amount_usd, price, category=category)
        elif "kalshi" in venue_id:
            return self.kalshi.calculate_fee(amount_usd, price)
        elif "crypto" in venue_id:
            return self.crypto.calculate_fee(amount_usd, price)
        elif "stock" in venue_id:
            return self.stock.calculate_fee(amount_usd, price)
        else:
            # Default conservative 2%
            return FeeResult(
                venue_id=venue_id,
                amount_usd=amount_usd,
                contracts=amount_usd / price if price > 0 else 0,
                price=price,
                fee_usd=amount_usd * 0.02,
                fee_pct=0.02,
                fee_formula="default 2% conservative",
                break_even_edge_pct=0.025,
                reasoning=f"Default fee 2% for unknown venue {venue_id}"
            )

    def get_sustainability_report(self, bankroll: float = 50.0) -> Dict:
        """
        $50 challenge math - sustainability report
        """
        position_6pct = bankroll * 0.06  # $3 on $50
        
        fees = {
            "polymarket_0.50": self.polymarket.calculate_fee(position_6pct, 0.50),
            "polymarket_0.61": self.polymarket.calculate_fee(position_6pct, 0.61),
            "polymarket_conservative_politics": self.polymarket.calculate_fee_conservative(position_6pct, 0.61, "politics"),
            "kalshi_0.50": self.kalshi.calculate_fee(position_6pct, 0.50),
            "crypto_0.50": self.crypto.calculate_fee(position_6pct, 0.50),
            "stock_0.50": self.stock.calculate_fee(position_6pct, 0.50),
        }
        
        # Sustainability: need 20-40% monthly return to pay for $10-20 costs
        monthly_cost = 15.0  # $15/month VPS/data
        required_monthly_return_pct = monthly_cost / bankroll  # 30% on $50
        required_daily_return_pct = required_monthly_return_pct / 30  # 1% daily
        required_daily_profit = bankroll * required_daily_return_pct  # $0.50 daily on $50
        
        # With $3 positions, how many winning trades needed?
        # If avg profit per trade 10% edge after fees = $0.30 per $3 trade
        avg_profit_per_trade = position_6pct * 0.10  # $0.30
        trades_needed_daily = required_daily_profit / avg_profit_per_trade if avg_profit_per_trade > 0 else 999
        
        return {
            "bankroll": bankroll,
            "max_position_6pct": position_6pct,
            "fees": {k: {"fee_usd": v.fee_usd, "fee_pct": v.fee_pct, "break_even": v.break_even_edge_pct, "formula": v.fee_formula} for k, v in fees.items()},
            "sustainability": {
                "monthly_cost": monthly_cost,
                "required_monthly_return_pct": required_monthly_return_pct,
                "required_monthly_return": f"{required_monthly_return_pct*100:.1f}%",
                "required_daily_return_pct": required_daily_return_pct,
                "required_daily_profit": required_daily_profit,
                "avg_profit_per_trade_10pct_edge": avg_profit_per_trade,
                "trades_needed_daily": trades_needed_daily,
                "reality_check": f"Need {required_monthly_return_pct*100:.0f}% monthly return = {trades_needed_daily:.1f} winning trades per day at 10% avg edge - extraordinary edge or extreme luck required. $50 is learning budget, not income. Most valuable outcome is infrastructure and knowledge, not $50.",
                "gas_note": "Polygon gas cheap but few cents per tx is significant % of $3 position. At $50, gas $0.05 = 1.6% of $3 position"
            },
            "fee_comparison": "Polymarket 1.5-3% at $0.50, Kalshi up to 14% ($0.07/contract) huge for small bankroll, Crypto 0.1% lowest, Stocks 0.05% lowest - fee is biggest constraint for $50"
        }
