"""
Trade Outcomes - tracks whether prediction worked, which venue/strategy/model works
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger


@dataclass
class TradeOutcome:
    trade_id: str
    market_id: str
    venue_id: str
    strategy: str
    category: str
    forecast_prob: float
    market_price: float
    edge: float
    side: str
    amount_usd: float
    actual_outcome: Optional[float] = None  # 1 win, 0 loss
    pnl: float = 0.0
    resolved_at: Optional[datetime] = None
    brier_score: float = 0.0
    was_correct: bool = False


class TradeOutcomeTracker:
    """
    Tracks trade outcomes to learn which venues/strategies work.
    """
    def __init__(self, storage=None):
        self.storage = storage
        self.outcomes: List[TradeOutcome] = []

    def record_trade(self, trade_id: str, market_id: str, venue_id: str, strategy: str, category: str,
                     forecast_prob: float, market_price: float, edge: float, side: str, amount_usd: float):
        outcome = TradeOutcome(
            trade_id=trade_id,
            market_id=market_id,
            venue_id=venue_id,
            strategy=strategy,
            category=category,
            forecast_prob=forecast_prob,
            market_price=market_price,
            edge=edge,
            side=side,
            amount_usd=amount_usd
        )
        self.outcomes.append(outcome)
        logger.info(f"Trade outcome recorded {trade_id}: {market_id} {venue_id} {strategy} edge {edge:.3f}")

    def record_resolution(self, trade_id: str, actual_outcome: float, pnl: float):
        for o in self.outcomes:
            if o.trade_id == trade_id:
                o.actual_outcome = actual_outcome
                o.pnl = pnl
                o.resolved_at = datetime.now(timezone.utc)
                # Brier
                o.brier_score = (o.forecast_prob - actual_outcome) ** 2
                # Was correct?
                predicted_yes = o.forecast_prob >= 0.5
                actual_yes = actual_outcome == 1.0
                o.was_correct = predicted_yes == actual_yes
                logger.info(f"Trade resolved {trade_id}: actual {actual_outcome} pnl ${pnl:.2f} correct {o.was_correct} brier {o.brier_score:.3f}")
                break

    def get_venue_performance(self) -> Dict[str, Dict]:
        """Performance by venue"""
        by_venue: Dict[str, List[TradeOutcome]] = {}
        for o in self.outcomes:
            if o.actual_outcome is None:
                continue
            by_venue.setdefault(o.venue_id, []).append(o)
        
        result = {}
        for venue_id, outcomes in by_venue.items():
            wins = sum(1 for o in outcomes if o.was_correct)
            total = len(outcomes)
            profit = sum(o.pnl for o in outcomes)
            avg_brier = sum(o.brier_score for o in outcomes) / total if total else 0.5
            result[venue_id] = {
                "total": total,
                "wins": wins,
                "win_rate": wins / total if total else 0,
                "profit": profit,
                "avg_brier": avg_brier,
                "forecast_skill": max(0, 1 - avg_brier*2),
                "avg_edge": sum(o.edge for o in outcomes) / total if total else 0
            }
        return result

    def get_category_performance(self) -> Dict[str, Dict]:
        """Performance by category"""
        by_cat: Dict[str, List[TradeOutcome]] = {}
        for o in self.outcomes:
            if o.actual_outcome is None:
                continue
            by_cat.setdefault(o.category, []).append(o)
        
        result = {}
        for cat, outcomes in by_cat.items():
            wins = sum(1 for o in outcomes if o.was_correct)
            total = len(outcomes)
            profit = sum(o.pnl for o in outcomes)
            avg_brier = sum(o.brier_score for o in outcomes) / total if total else 0.5
            result[cat] = {
                "total": total,
                "wins": wins,
                "win_rate": wins / total if total else 0,
                "profit": profit,
                "avg_brier": avg_brier,
                "skill": max(0, 1 - avg_brier*2)
            }
        return result

    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Performance by strategy"""
        by_strat: Dict[str, List[TradeOutcome]] = {}
        for o in self.outcomes:
            if o.actual_outcome is None:
                continue
            by_strat.setdefault(o.strategy, []).append(o)
        
        result = {}
        for strat, outcomes in by_strat.items():
            wins = sum(1 for o in outcomes if o.was_correct)
            total = len(outcomes)
            profit = sum(o.pnl for o in outcomes)
            result[strat] = {
                "total": total,
                "win_rate": wins / total if total else 0,
                "profit": profit
            }
        return result

    def should_concentrate_on(self) -> Dict:
        """Where should PTAI concentrate?"""
        venue_perf = self.get_venue_performance()
        cat_perf = self.get_category_performance()
        
        # Sort by skill
        top_venues = sorted(venue_perf.items(), key=lambda x: x[1]["forecast_skill"], reverse=True)[:3]
        top_cats = sorted(cat_perf.items(), key=lambda x: x[1]["skill"], reverse=True)[:3]
        weak_cats = sorted(cat_perf.items(), key=lambda x: x[1]["skill"])[:2]
        
        return {
            "top_venues": [{"venue": v[0], **v[1]} for v in top_venues],
            "top_categories": [{"category": c[0], **c[1]} for c in top_cats],
            "weak_categories": [{"category": c[0], **c[1]} for c in weak_cats],
            "total_trades": len([o for o in self.outcomes if o.actual_outcome is not None]),
            "recommendation": f"Focus on {top_cats[0][0]} in {top_venues[0][0]}" if top_cats and top_venues else "Need more data"
        }
