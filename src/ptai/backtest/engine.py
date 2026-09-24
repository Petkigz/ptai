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
    # V9 FIX #6: Mark synthetic vs real
    is_synthetic: bool = False
    data_mode: str = "live"  # live or mock
    warnings: List[str] = field(default_factory=list)
    is_production_grade: bool = False

class BacktestEngine:
    """
    Premium Backtesting - test strategies on historical data
    For product: users can backtest before risking real money
    """
    def __init__(self, storage=None):
        self.storage = storage
        logger.info("BacktestEngine initialized")
    
    def run(self, strategy_config: Dict[str, Any], historical_markets: List[Dict] = None, days: int = 30, allow_synthetic: bool = False,
            dataset: Any = None, seed: int = 0) -> BacktestResult:
        """
        V9 FIX #6: Replace synthetic backtester with real historical data requirement
        Previously generated random synthetic markets if no data - misleading
        Now: requires real historical data, or if allow_synthetic=True clearly marks as synthetic and warns not production-grade

        strategy_config: {min_edge: 0.08, max_pos_pct: 0.06, kelly_fraction: 0.5, model: "qwen/qwen3-32b"}
        historical_markets: list of markets with resolved outcomes (if None, must set allow_synthetic=True to generate mock, but result marked synthetic)
        dataset: a HistoricalDataset from backtest.historical - its rows are used
                 in preference to historical_markets, and its warnings are carried
                 onto the result so assumed costs are never silently dropped.
        seed: seeds the latency-drift simulation. The V10 cost model perturbs the
                 execution price with a random walk, which previously used the
                 module-level RNG - so the same data and the same strategy
                 produced a different PnL on every run, and comparing two
                 configs compared their noise. A fixed seed makes a backtest
                 reproducible; seed=0 is deterministic by default.
        """
        if dataset is not None:
            historical_markets = list(dataset.rows)
        rng = random.Random(seed)
        logger.info(f"Backtest starting: {strategy_config} for {days} days - V9 requires real historical data")
        
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
        is_synthetic = False
        
        # V9 FIX #6: Require real historical data, synthetic only if explicitly allowed and marked
        if not historical_markets:
            if not allow_synthetic:
                logger.error("Backtest requires real historical markets with resolved outcomes - no synthetic data allowed by default V9 FIX")
                raise ValueError(
                    "Backtest requires real historical data: historical_markets must be provided with resolved outcomes. "
                    "Synthetic random markets are misleading and not production-grade. "
                    "If you want synthetic for development, set allow_synthetic=True but result will be marked synthetic and should NOT be used for live trading decisions. "
                    "Real historical data needs: fees, spreads, slippage, partial fills, latency, liquidity, position limits, resolution timing, bankroll constraints"
                )
            # Synthetic path - only if explicitly allowed
            is_synthetic = True
            logger.warning("Backtest using SYNTHETIC random markets - NOT production-grade, misleading results, DO NOT use for live trading V9")
            historical_markets = []
            for day in range(days):
                for i in range(10):
                    market_price = rng.uniform(0.2, 0.8)
                    fair_value = market_price + rng.uniform(-0.15, 0.15)
                    edge = fair_value - market_price
                    actual_prob = fair_value
                    actual_outcome = 1 if rng.random() < actual_prob else 0
                    
                    historical_markets.append({
                        "day": day,
                        "question": f"SYNTHETIC Mock market day {day} #{i} - NOT REAL DATA - V9 WARNING",
                        "market_price": market_price,
                        "fair_value": fair_value,
                        "edge": edge,
                        "actual_outcome": actual_outcome,
                        "confidence": rng.uniform(0.5, 0.9),
                        "data_mode": "mock",
                        "is_synthetic": True
                    })
        
        # V10 FIX #6: High-fidelity backtest with spread/depth/fees/slippage/latency/partial fills
        for market in historical_markets:
            edge = market["edge"]
            if abs(edge) < min_edge:
                continue
            if market["confidence"] < 0.6:
                continue
            
            # V10 FIX #6: Real orderbook data if available, else estimate
            bid = market.get("bid", market["market_price"] - market.get("spread", 0.02)/2)
            ask = market.get("ask", market["market_price"] + market.get("spread", 0.02)/2)
            spread = market.get("spread", ask - bid) if "spread" in market else (ask - bid)
            depth = market.get("depth", market.get("liquidity", 5000))
            fee_pct = market.get("fee_pct", 0.02)
            
            # V10 FIX #6: Kelly sizing with real spread consideration
            # Effective price includes half spread + slippage
            effective_price = ask if edge > 0 else (1-bid)  # buying YES at ask, NO at 1-bid
            kelly_raw = abs(edge) / (effective_price * (1 - effective_price) + 0.01) * kelly_frac
            kelly_capped = min(kelly_raw, max_pos_pct)
            position_size = bankroll * kelly_capped
            
            if position_size < 1:
                continue
            
            # V10 FIX #6: Slippage based on depth - amount / liquidity
            slippage_pct = min(0.05, position_size / max(1, depth) * 0.5)
            slippage_usd = position_size * slippage_pct
            
            # Fees
            fees_usd = position_size * fee_pct
            
            # Gas for on-chain
            gas_usd = 0.05 if market.get("venue_id") in ["polymarket", "afx_dex"] else 0.0
            
            # V10 FIX #6: Latency - market may have moved between signal and execution
            latency_ms = market.get("latency_ms", 200)
            # Simulate price drift during latency: random walk proportional to sqrt(latency)
            import math
            drift_vol = 0.001 * math.sqrt(latency_ms / 100)  # 0.1% vol per 100ms
            price_drift = rng.gauss(0, drift_vol)
            executed_price = market["market_price"] + price_drift
            executed_price = max(0.01, min(0.99, executed_price))
            
            # V10 FIX #6: Partial fills - if size > depth, only partial fills
            fill_ratio = min(1.0, depth / max(1, position_size * 2))  # if depth < 2*size, partial
            if fill_ratio < 1.0:
                # Partial fill - only part of position executes
                filled_size = position_size * fill_ratio
            else:
                filled_size = position_size
            
            # V10 FIX #6: Queue position - not first in queue, may not get filled at best price
            queue_position = market.get("queue_position", 0.5)  # 0=front, 1=back
            queue_penalty = queue_position * spread * 0.5  # worse queue = pay more spread
            executed_price += queue_penalty if edge > 0 else -queue_penalty
            
            # Simulate trade outcome with real costs
            side = "YES" if edge > 0 else "NO"
            won = (side == "YES" and market["actual_outcome"] == 1) or (side == "NO" and market["actual_outcome"] == 0)
            
            if won:
                # Win: profit = filled_size * (1-executed_price)/executed_price - costs
                # For YES: buy at executed_price, sell at 1.0 if wins
                if side == "YES":
                    gross_profit = filled_size * (1 - executed_price) / executed_price if executed_price > 0 else filled_size
                else:
                    no_price = 1 - executed_price
                    gross_profit = filled_size * (1 - no_price) / no_price if no_price > 0 else filled_size
                
                net_profit = gross_profit - fees_usd - slippage_usd - gas_usd
                bankroll += net_profit
                winning += 1
            else:
                # Loss: -filled_size - costs (fees still paid)
                net_loss = filled_size + fees_usd + slippage_usd + gas_usd
                bankroll -= net_loss
                losing += 1
            
            # Track drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            
            # V10 FIX #6: High-fidelity trade record
            trades.append({
                "question": market["question"],
                "market_price": market["market_price"],
                "executed_price": executed_price,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "depth": depth,
                "fair_value": market["fair_value"],
                "edge": edge,
                "side": side,
                "size": position_size,
                "filled_size": filled_size,
                "fill_ratio": fill_ratio,
                "fees_usd": fees_usd,
                "slippage_usd": slippage_usd,
                "gas_usd": gas_usd,
                "latency_ms": latency_ms,
                "price_drift": price_drift,
                "queue_position": queue_position,
                "queue_penalty": queue_penalty,
                "won": won,
                "bankroll_after": bankroll,
                "actual_outcome": market["actual_outcome"],
                "net_pnl": net_profit if won else -net_loss,
                "data_mode": market.get("data_mode", "live"),
                "is_synthetic": market.get("is_synthetic", is_synthetic)
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
        
        warnings = []
        dataset_warnings: List[str] = []
        if dataset is not None:
            dataset_warnings = list(getattr(dataset, "warnings", []) or [])
            if getattr(dataset, "is_baseline", False):
                warnings.append("BASELINE RUN - no strategy signal was supplied, so fair "
                                "value equals the market price and edge is zero. This "
                                "measures costs, not skill, and is not production-grade")
        if is_synthetic:
            warnings.append("SYNTHETIC DATA - NOT production-grade, random markets, no real fees/spreads/slippage/latency/liquidity")
            warnings.append("DO NOT use synthetic backtest for live trading decisions - misleading results")
            warnings.append("Real backtest needs: historical market/orderbook data, fees, spreads, slippage, partial fills, latency, liquidity, position limits, resolution timing, bankroll constraints")
        
        result = BacktestResult(
            strategy_name=strategy_config.get("name", "Default Strategy") + (" [SYNTHETIC - NOT PRODUCTION-GRADE]" if is_synthetic else " [REAL DATA]"),
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
            trades=trades[-50:],
            equity_curve=equity_curve,
            is_synthetic=is_synthetic,
            data_mode="mock" if is_synthetic else "live",
            warnings=warnings + dataset_warnings,
            # A baseline run cannot be production-grade: with no signal there is
            # no strategy being evaluated, so a passing result would prove nothing.
            is_production_grade=(not is_synthetic
                                 and not (dataset is not None and getattr(dataset, "is_baseline", False))
                                 and len(historical_markets) >= 100)
        )
        
        if is_synthetic:
            logger.warning(f"Backtest done SYNTHETIC: {initial} -> {bankroll:.2f} PnL {total_pnl:.2f} {total_pnl_pct:.1%} win rate {win_rate:.1f}% trades {len(trades)} - NOT PRODUCTION-GRADE - BLOCKS LIVE DEPLOYMENT")
        else:
            logger.success(f"Backtest done REAL DATA: {initial} -> {bankroll:.2f} PnL {total_pnl:.2f} {total_pnl_pct:.1%} win rate {win_rate:.1f}% trades {len(trades)} - production-grade: {result.is_production_grade}")
        
        return result

    def can_deploy_live(self, result: BacktestResult) -> tuple[bool, str]:
        """
        V9 FIX #6: Block live deployment if synthetic - real historical data gate
        If no real data, mark result synthetic and block live deployment decisions
        """
        if result.is_synthetic:
            return False, f"SYNTHETIC backtest {result.strategy_name} - MUST NEVER be used for live deployment decisions - needs real historical data with fees/spreads/slippage/latency"
        if not result.is_production_grade:
            return False, f"Not production-grade: need >=100 real markets, got synthetic={result.is_synthetic} data_mode={result.data_mode} trades={result.total_trades} - block live deployment"
        if result.total_trades < 100:
            return False, f"Only {result.total_trades} trades - need 100+ for statistical significance - block live"
        if result.win_rate < 55:
            return False, f"Win rate {result.win_rate:.1f}% <55% - block live"
        return True, f"Production-grade REAL DATA {result.total_trades} trades win {result.win_rate:.1f}% - can consider live with caution"
