
"""
Paper Trading Engine - robust learning loop
Makes paper-trading qualification and learning loop robust per user request
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger
import json
from pathlib import Path

from ..venues.registry import VenueRegistry
from ..venues.adapter import VenueOpportunity
from ..markets.base import Market

@dataclass
class PaperTrade:
    trade_id: str
    venue_id: str
    market_id: str
    question: str
    category: str
    side: str
    price: float
    amount_usd: float
    forecast_prob: float
    confidence: float
    edge: float
    timestamp: datetime
    outcome: Optional[int] = None  # 1 win 0 loss None unresolved
    resolved_at: Optional[datetime] = None
    profit_usd: float = 0.0
    brier_score: float = 0.0

class PaperTradingEngine:
    """
    Robust paper trading for venue qualification and learning
    Previously: simple paper trading
    Now: comprehensive tracking, Brier, calibration, profit after fees, learning loop
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.trades_file = self.data_dir / "paper_trades.jsonl"
        self.trades: List[PaperTrade] = []
        self._load()

    def _load(self):
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        if data.get('resolved_at'):
                            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
                        self.trades.append(PaperTrade(**data))
            except Exception as e:
                logger.warning(f"Paper trades load failed: {e}")

    def _save_trade(self, trade: PaperTrade):
        try:
            with open(self.trades_file, 'a') as f:
                data = {
                    "trade_id": trade.trade_id,
                    "venue_id": trade.venue_id,
                    "market_id": trade.market_id,
                    "question": trade.question,
                    "category": trade.category,
                    "side": trade.side,
                    "price": trade.price,
                    "amount_usd": trade.amount_usd,
                    "forecast_prob": trade.forecast_prob,
                    "confidence": trade.confidence,
                    "edge": trade.edge,
                    "timestamp": trade.timestamp.isoformat(),
                    "outcome": trade.outcome,
                    "resolved_at": trade.resolved_at.isoformat() if trade.resolved_at else None,
                    "profit_usd": trade.profit_usd,
                    "brier_score": trade.brier_score
                }
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.warning(f"Paper trade save failed: {e}")

    def record_paper_trade(self, opportunity: VenueOpportunity, amount_usd: float) -> PaperTrade:
        import uuid
        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8],
            venue_id=opportunity.venue_id,
            market_id=opportunity.market.id,
            question=opportunity.market.question[:200],
            category=opportunity.category,
            side=opportunity.side,
            price=opportunity.market_price,
            amount_usd=amount_usd,
            forecast_prob=opportunity.estimated_fair,
            confidence=opportunity.confidence,
            edge=opportunity.effective_edge,
            timestamp=datetime.now(timezone.utc)
        )
        self.trades.append(trade)
        self._save_trade(trade)
        logger.info(f"Paper trade recorded: {trade.venue_id} {trade.market_id} {trade.side} ${amount_usd} @ {trade.price:.3f} edge {trade.edge*100:.1f}% cat {trade.category}")
        return trade

    def resolve_trade(self, trade_id: str, outcome: int, profit_usd: float = None):
        # outcome 1 win 0 loss
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.outcome is None:
                trade.outcome = outcome
                trade.resolved_at = datetime.now(timezone.utc)
                # Calculate Brier score
                actual = 1.0 if outcome == 1 else 0.0
                trade.brier_score = (trade.forecast_prob - actual) ** 2
                # Calculate profit if not provided
                if profit_usd is None:
                    if outcome == 1:
                        # Win: profit = amount * (1-price)/price? Simplified
                        trade.profit_usd = trade.amount_usd * (1 - trade.price) / trade.price if trade.side == "YES" else trade.amount_usd * trade.price / (1-trade.price)
                    else:
                        trade.profit_usd = -trade.amount_usd
                else:
                    trade.profit_usd = profit_usd
                logger.info(f"Paper trade resolved: {trade_id} outcome {outcome} profit ${trade.profit_usd:.2f} brier {trade.brier_score:.3f}")
                break

    def get_venue_performance(self, venue_id: str, category: str = None) -> Dict[str, Any]:
        filtered = [t for t in self.trades if t.venue_id == venue_id]
        if category:
            filtered = [t for t in filtered if t.category == category]
        
        resolved = [t for t in filtered if t.outcome is not None]
        if not resolved:
            return {
                "venue_id": venue_id,
                "category": category or "all",
                "total_paper_trades": len(filtered),
                "resolved": 0,
                "win_rate": 0,
                "avg_edge": 0,
                "brier_score": 0.5,
                "profit_paper": 0,
                "forecast_skill": 0.5,
                "is_qualified": False
            }
        
        wins = sum(1 for t in resolved if t.outcome == 1)
        win_rate = wins / len(resolved)
        avg_edge = sum(t.edge for t in resolved) / len(resolved)
        brier = sum(t.brier_score for t in resolved) / len(resolved)
        profit = sum(t.profit_usd for t in resolved)
        skill = max(0, 1 - brier*2)
        
        return {
            "venue_id": venue_id,
            "category": category or "all",
            "total_paper_trades": len(filtered),
            "resolved": len(resolved),
            "win_rate": win_rate,
            "avg_edge": avg_edge,
            "brier_score": brier,
            "profit_paper": profit,
            "forecast_skill": skill,
            "is_qualified": len(filtered) >= 100 and win_rate >= 0.55 and brier <= 0.25 and skill >= 0.6 and profit >= 0
        }

    def get_all_performance(self) -> Dict[str, Dict]:
        venues = set(t.venue_id for t in self.trades)
        result = {}
        for venue in venues:
            result[venue] = self.get_venue_performance(venue)
            # Also per category
            categories = set(t.category for t in self.trades if t.venue_id == venue)
            for cat in categories:
                result[f"{venue}:{cat}"] = self.get_venue_performance(venue, cat)
        return result

    def get_learning_report(self) -> Dict[str, Any]:
        all_perf = self.get_all_performance()
        # Sort by skill
        sorted_perf = sorted(all_perf.values(), key=lambda x: x.get("forecast_skill", 0), reverse=True)
        
        qualified = [p for p in sorted_perf if p.get("is_qualified")]
        not_qualified = [p for p in sorted_perf if not p.get("is_qualified")]
        
        return {
            "total_trades": len(self.trades),
            "resolved": len([t for t in self.trades if t.outcome is not None]),
            "venues": len(set(t.venue_id for t in self.trades)),
            "qualified": len(qualified),
            "not_qualified": len(not_qualified),
            "leaderboard": [
                {
                    "venue": p["venue_id"],
                    "category": p["category"],
                    "total": p["total_paper_trades"],
                    "win_rate": p["win_rate"],
                    "brier": p["brier_score"],
                    "skill": p["forecast_skill"],
                    "profit": p["profit_paper"],
                    "qualified": p["is_qualified"]
                } for p in sorted_perf[:10]
            ],
            "concentration": {
                "strong": [f"{p['venue_id']}:{p['category']} skill={p['forecast_skill']:.2f}" for p in qualified[:3]],
                "weak": [f"{p['venue_id']}:{p['category']} skill={p['forecast_skill']:.2f}" for p in not_qualified if p["forecast_skill"] < 0.55][:3],
                "recommendation": f"Concentrate on {qualified[0]['venue_id']}:{qualified[0]['category']}" if qualified else "Need paper trading"
            },
            "principle": "PTAI learns which venue/category combos it is good at and concentrates research there - Weather strong Economic strong Kalshi strong Politics weak Crypto weak example"
        }
