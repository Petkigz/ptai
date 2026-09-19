"""
Backtest Engine - Test trading strategies on historical Polymarket data
Premium product feature - users can test before live
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import random
from loguru import logger

@dataclass
class BacktestResult:
    strategy_name: str
    start_date: str
    end_date: str
    initial_bankroll: float
    final_bankroll: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_edge: float
    max_drawdown: float
    sharpe: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

class BacktestEngine:
    """
    Premium Backtesting - test strategies on historical data
    For product: users can backtest before risking real money
    """
    def __init__(self, storage=None):
        self.storage = storage
        logger.info("BacktestEngine initialized")
    
    def run(self, strategy_config: Dict[str, Any], historical_markets: List[Dict] = None, days: int = 30) -> BacktestResult:
        """
        Run backtest for N days
        strategy_config: {min_edge: 0.08, max_pos_pct: 0.06, kelly_fraction: 0.5, model: "qwen/qwen3-32b"}
        historical_markets: list of markets with resolved outcomes (if None, mock)
        """
        logger.info(f"Backtest starting: {strategy_config} for {days} days")
        
        initial = strategy_config.get("bankroll", 50.0)
        bankroll = initial
        min_edge = strategy_config.get("min_edge", 0.08)
        max_pos_pct = strategy_config.get("max_pos_pct", 0.06)
        kelly_frac = strategy_config.get("kelly_fraction", 0.5)
        
        trades = []
        equity_curve = [{"date": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), "bankroll": bankroll}]
        winning = 0
        losing = 0
        max_dd = 0
        peak = bankroll
        
        # Mock historical data if not provided
        if not historical_markets:
            # Generate mock 30 days * 10 markets per day
            historical_markets = []
            for day in range(days):
                for i in range(10):
                    # Mock market with random edge and outcome
                    market_price = random.uniform(0.2, 0.8)
                    fair_value = market_price + random.uniform(-0.15, 0.15)
                    edge = fair_value - market_price
                    # Simulate actual outcome: if fair > market, more likely YES wins
                    actual_prob = fair_value
                    actual_outcome = 1 if random.random() < actual_prob else 0
                    
                    historical_markets.append({
                        "day": day,
                        "question": f"Mock market day {day} #{i}",
                        "market_price": market_price,
                        "fair_value": fair_value,
                        "edge": edge,
                        "actual_outcome": actual_outcome,
                        "confidence": random.uniform(0.5, 0.9)
                    })
        
        for market in historical_markets:
            edge = market["edge"]
            if abs(edge) < min_edge:
                continue
            if market["confidence"] < 0.6:
                continue
            
            # Kelly sizing
            # Simplified: f* = edge / (market_price * (1-market_price)) * kelly_frac, capped
            kelly_raw = abs(edge) / (market["market_price"] * (1 - market["market_price"]) + 0.01) * kelly_frac
            kelly_capped = min(kelly_raw, max_pos_pct)
            position_size = bankroll * kelly_capped
            
            if position_size < 1:  # min $1
                continue
            
            # Simulate trade outcome
            # If we bet YES and actual is 1, win; if we bet NO (edge negative) and actual 0, win
            side = "YES" if edge > 0 else "NO"
            won = (side == "YES" and market["actual_outcome"] == 1) or (side == "NO" and market["actual_outcome"] == 0)
            
            if won:
                # Win: profit = size * (1-market_price)/market_price for YES, or market_price/(1-market_price) for NO
                # Simplified: profit = size * |edge| / market_price
                profit = position_size * abs(edge) / market["market_price"] * 2  # mock 2x
                bankroll += profit
                winning += 1
            else:
                bankroll -= position_size
                losing += 1
            
            # Track drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            
            trades.append({
                "question": market["question"],
                "market_price": market["market_price"],
                "fair_value": market["fair_value"],
                "edge": edge,
                "side": side,
                "size": position_size,
                "won": won,
                "bankroll_after": bankroll,
                "actual_outcome": market["actual_outcome"]
            })
            
            # Equity curve daily
            if len(trades) % 10 == 0:
                equity_curve.append({"date": (datetime.now(timezone.utc) - timedelta(days=days - market["day"])).isoformat(), "bankroll": bankroll})
        
        total_pnl = bankroll - initial
        total_pnl_pct = total_pnl / initial if initial > 0 else 0
        win_rate = winning / max(1, winning + losing) * 100
        avg_edge = sum(t["edge"] for t in trades) / max(1, len(trades))
        
        # Sharpe mock
        sharpe = (total_pnl_pct / max(0.01, max_dd)) if max_dd > 0 else total_pnl_pct * 10
        
        result = BacktestResult(
            strategy_name=strategy_config.get("name", "Default Strategy"),
            start_date=(datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
            end_date=datetime.now(timezone.utc).isoformat(),
            initial_bankroll=initial,
            final_bankroll=bankroll,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=len(trades),
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            avg_edge=avg_edge,
            max_drawdown=max_dd,
            sharpe=sharpe,
            trades=trades[-50:],  # last 50
            equity_curve=equity_curve
        )
        
        logger.success(f"Backtest done: {initial} -> {bankroll:.2f} PnL {total_pnl:.2f} {total_pnl_pct:.1%} win rate {win_rate:.1f}% trades {len(trades)}")
        
        return result
