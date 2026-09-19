"""
Whale Tracking - Monitor large Polymarket wallets on Polygon, track historical P&L, copy smart ones or fade dumb ones

Polymarket trades are on Polygon, public blockchain, can track wallets
- Large wallets: >$10k volume
- Track P&L over time
- Copy smart whales with delay, or fade dumb ones
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import math


@dataclass
class WhaleWallet:
    address: str
    total_volume: float
    total_trades: int
    pnl_usd: float
    win_rate: float
    avg_position: float
    is_smart: bool  # historically profitable
    is_dumb: bool  # historically unprofitable (fade)
    score: float  # -1 to +1, +1 very smart, -1 very dumb
    recent_trades: List[Dict]
    reasoning: str


@dataclass
class WhaleSignal:
    market_id: str
    whale_address: str
    whale_score: float
    side: str
    amount_usd: float
    market_price: float
    signal_type: str  # "copy_smart", "fade_dumb"
    edge_estimate: float
    should_trade: bool
    reasoning: str


class WhaleTracker:
    """
    Tracks Polymarket whales on Polygon
    """
    def __init__(self, min_whale_volume: float = 10000, smart_threshold: float = 0.6, dumb_threshold: float = 0.4):
        self.min_whale_volume = min_whale_volume
        self.smart_threshold = smart_threshold  # win rate >60% smart
        self.dumb_threshold = dumb_threshold  # win rate <40% dumb

    def analyze_wallet(self, address: str, trades: List[Dict]) -> WhaleWallet:
        """
        Analyze wallet historical P&L
        trades: list of {market_id, side, amount, entry_price, exit_price, pnl, timestamp}
        """
        if not trades:
            return WhaleWallet(
                address=address,
                total_volume=0,
                total_trades=0,
                pnl_usd=0,
                win_rate=0.5,
                avg_position=0,
                is_smart=False,
                is_dumb=False,
                score=0,
                recent_trades=[],
                reasoning="No trades"
            )
        
        total_volume = sum(t.get("amount_usd", 0) for t in trades)
        total_trades = len(trades)
        pnl_usd = sum(t.get("pnl_usd", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.5
        avg_position = total_volume / total_trades if total_trades > 0 else 0
        
        is_smart = win_rate >= self.smart_threshold and pnl_usd > 0 and total_volume >= self.min_whale_volume
        is_dumb = win_rate <= self.dumb_threshold and pnl_usd < 0 and total_volume >= self.min_whale_volume
        
        # Score -1 to +1 based on win rate and PnL
        # Win rate 60%+ and positive PnL => +0.5 to +1
        # Win rate 40%- and negative PnL => -0.5 to -1
        if is_smart:
            score = min(1.0, (win_rate - 0.5) * 2 + min(0.5, pnl_usd / 10000))
        elif is_dumb:
            score = max(-1.0, (win_rate - 0.5) * 2 + max(-0.5, pnl_usd / 10000))
        else:
            score = (win_rate - 0.5) * 1.0  # -0.5 to +0.5 for neutral
        
        reasoning = (
            f"Wallet {address[:10]}... volume ${total_volume:.0f} trades {total_trades} "
            f"PnL ${pnl_usd:.2f} win rate {win_rate*100:.1f}% avg ${avg_position:.2f} | "
            f"Smart {is_smart} (win>{self.smart_threshold*100:.0f}%) Dumb {is_dumb} (win<{self.dumb_threshold*100:.0f}%) score {score:.2f}"
        )
        
        return WhaleWallet(
            address=address,
            total_volume=total_volume,
            total_trades=total_trades,
            pnl_usd=pnl_usd,
            win_rate=win_rate,
            avg_position=avg_position,
            is_smart=is_smart,
            is_dumb=is_dumb,
            score=score,
            recent_trades=trades[-10:],  # last 10
            reasoning=reasoning
        )

    def get_whale_signals(self, market_id: str, market_price: float,
                         whale_trades: List[Dict], whale_wallets: Dict[str, WhaleWallet]) -> List[WhaleSignal]:
        """
        Get signals from whale activity on specific market
        whale_trades: recent trades on this market {whale_address, side, amount, price, timestamp}
        """
        signals: List[WhaleSignal] = []
        
        for trade in whale_trades:
            address = trade.get("whale_address", "")
            wallet = whale_wallets.get(address)
            if not wallet:
                continue
            
            side = trade.get("side", "YES")
            amount = trade.get("amount_usd", 0)
            
            signal_type = None
            edge_estimate = 0
            should_trade = False
            
            if wallet.is_smart:
                # Copy smart whale with delay
                # Smart whale buying YES => we should also buy YES, but with smaller size and delay
                signal_type = "copy_smart"
                edge_estimate = abs(wallet.score) * 0.05  # smart score 0.8 => 4% edge estimate
                should_trade = wallet.score > 0.6 and amount > 100  # only copy large smart trades
            elif wallet.is_dumb:
                # Fade dumb whale
                # Dumb whale buying YES => we sell YES / buy NO
                signal_type = "fade_dumb"
                edge_estimate = abs(wallet.score) * 0.04  # dumb score -0.7 => 2.8% edge for fading
                should_trade = wallet.score < -0.6 and amount > 100
            
            if signal_type:
                reasoning = (
                    f"Whale {address[:10]}... score {wallet.score:.2f} {'smart' if wallet.is_smart else 'dumb'} "
                    f"{side} ${amount} on {market_id} price {market_price:.3f} | "
                    f"Signal {signal_type} edge est {edge_estimate*100:.1f}% should trade {should_trade} | "
                    f"Wallet {wallet.reasoning}"
                )
                
                signal = WhaleSignal(
                    market_id=market_id,
                    whale_address=address,
                    whale_score=wallet.score,
                    side=side,
                    amount_usd=amount,
                    market_price=market_price,
                    signal_type=signal_type,
                    edge_estimate=edge_estimate,
                    should_trade=should_trade,
                    reasoning=reasoning
                )
                signals.append(signal)
                
                if should_trade:
                    logger.info(f"WHALE SIGNAL: {reasoning}")
        
        signals.sort(key=lambda x: abs(x.edge_estimate), reverse=True)
        return signals

    def mock_whale_data(self) -> Tuple[List[WhaleWallet], Dict[str, List[Dict]]]:
        """
        Mock whale data for demo - in production would query Polygon blockchain
        Polymarket trades on Polygon, can use Alchemy/Infura to track large wallets
        """
        # Mock 5 whales
        wallets = [
            WhaleWallet(
                address="0x1234...smart1",
                total_volume=50000,
                total_trades=100,
                pnl_usd=8000,
                win_rate=0.68,
                avg_position=500,
                is_smart=True,
                is_dumb=False,
                score=0.85,
                recent_trades=[],
                reasoning="Smart whale 68% win rate +$8k"
            ),
            WhaleWallet(
                address="0x5678...dumb1",
                total_volume=30000,
                total_trades=80,
                pnl_usd=-5000,
                win_rate=0.35,
                avg_position=375,
                is_smart=False,
                is_dumb=True,
                score=-0.75,
                recent_trades=[],
                reasoning="Dumb whale 35% win rate -$5k fade"
            ),
            WhaleWallet(
                address="0x9abc...neutral",
                total_volume=20000,
                total_trades=50,
                pnl_usd=500,
                win_rate=0.52,
                avg_position=400,
                is_smart=False,
                is_dumb=False,
                score=0.05,
                recent_trades=[],
                reasoning="Neutral whale 52% win rate"
            ),
        ]
        
        # Mock recent trades per market
        market_whale_trades = {
            "market_trump_win": [
                {"whale_address": "0x1234...smart1", "side": "YES", "amount_usd": 500, "price": 0.61, "timestamp": "2026-01-01"},
                {"whale_address": "0x5678...dumb1", "side": "YES", "amount_usd": 300, "price": 0.62, "timestamp": "2026-01-01"},
            ]
        }
        
        wallets_dict = {w.address: w for w in wallets}
        
        return wallets, market_whale_trades, wallets_dict

    def get_report(self) -> Dict[str, Any]:
        return {
            "tracker": "Whale Tracking",
            "method": "Monitor large Polymarket wallets on Polygon, track historical P&L, copy smart or fade dumb",
            "polygon": "Polymarket trades on Polygon, public blockchain, can track via Alchemy/Infura",
            "thresholds": f"Min whale volume ${self.min_whale_volume}, smart win rate >{self.smart_threshold*100:.0f}%, dumb <{self.dumb_threshold*100:.0f}%",
            "signals": "Copy smart whales with delay, fade dumb whales, edge estimate based on whale score",
            "mock": "Mock data for demo, real would query Polygon blockchain for large wallets and track P&L"
        }
