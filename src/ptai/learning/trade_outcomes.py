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

    Persisted, and tolerant about the fields a caller supplies.

    Two separate problems made this tracker useless in V3:

      1. V3 called `record_trade(market_id=..., venue_id=..., strategy=...,
         edge=..., confidence=..., amount_usd=..., data_mode=..., trust_tier=...)`
         while this method requires trade_id, category, forecast_prob,
         market_price and side. Every call raised TypeError, so no venue or
         strategy outcome was ever recorded and allocation kept re-estimating
         from nothing.
      2. `self.outcomes` was memory-only, so even a successful record vanished
         at the next restart.

    The signature now accepts the caller's fields, fills what it can, and
    REFUSES - loudly - when a field it genuinely needs to compute performance is
    missing, rather than silently recording a zero that would look like a loss.
    """
    def __init__(self, storage=None):
        self.storage = storage
        self.outcomes: List[TradeOutcome] = []
        self.loaded_from_storage = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> int:
        """Restore outcomes from the database. Returns how many were loaded."""
        if not self.storage:
            return 0
        try:
            rows = self.storage.conn.execute(
                "SELECT * FROM trade_outcomes ORDER BY recorded_at"
            ).fetchall()
        except Exception as e:
            logger.error(
                f"TradeOutcomeTracker could not load history: {type(e).__name__}: "
                f"{e}. Allocation would otherwise re-estimate from nothing.")
            return 0

        for row in rows:
            def _dt(value):
                if not value:
                    return None
                try:
                    return datetime.fromisoformat(value)
                except (TypeError, ValueError):
                    return None

            self.outcomes.append(TradeOutcome(
                trade_id=row["trade_id"],
                market_id=row["market_id"],
                venue_id=row["venue_id"] or "",
                strategy=row["strategy"] or "",
                category=row["category"] or "",
                forecast_prob=float(row["forecast_prob"] or 0.0),
                market_price=float(row["market_price"] or 0.0),
                edge=float(row["edge"] or 0.0),
                side=row["side"] or "",
                amount_usd=float(row["amount_usd"] or 0.0),
                actual_outcome=row["actual_outcome"],
                pnl=float(row["pnl"] or 0.0),
                resolved_at=_dt(row["resolved_at"]),
                brier_score=float(row["brier_score"] or 0.0),
                was_correct=bool(row["was_correct"]),
            ))
        resolved = sum(1 for o in self.outcomes if o.actual_outcome is not None)
        if self.outcomes:
            logger.info(
                f"Trade outcomes loaded: {len(self.outcomes)} records, "
                f"{resolved} resolved")
        return len(self.outcomes)

    def _persist(self, outcome: TradeOutcome, *, update: bool = False) -> bool:
        if not self.storage:
            return False
        try:
            if update:
                self.storage.conn.execute(
                    "UPDATE trade_outcomes SET actual_outcome=?, pnl=?, "
                    "resolved_at=?, brier_score=?, was_correct=? "
                    "WHERE trade_id=?",
                    (outcome.actual_outcome, outcome.pnl,
                     outcome.resolved_at.isoformat() if outcome.resolved_at else None,
                     outcome.brier_score, int(outcome.was_correct),
                     outcome.trade_id))
            else:
                self.storage.conn.execute(
                    "INSERT OR REPLACE INTO trade_outcomes (trade_id, market_id, "
                    "venue_id, strategy, category, forecast_prob, market_price, "
                    "edge, side, amount_usd, actual_outcome, pnl, resolved_at, "
                    "brier_score, was_correct, recorded_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (outcome.trade_id, outcome.market_id, outcome.venue_id,
                     outcome.strategy, outcome.category, outcome.forecast_prob,
                     outcome.market_price, outcome.edge, outcome.side,
                     outcome.amount_usd, outcome.actual_outcome, outcome.pnl,
                     outcome.resolved_at.isoformat() if outcome.resolved_at else None,
                     outcome.brier_score, int(outcome.was_correct),
                     datetime.now(timezone.utc).isoformat()))
            self.storage.conn.commit()
            return True
        except Exception as e:
            logger.error(
                f"Trade outcome NOT PERSISTED for {outcome.trade_id}: "
                f"{type(e).__name__}: {e}. This result will not influence "
                f"allocation after a restart.")
            return False

    def record_trade(self, trade_id: str = None, market_id: str = "", venue_id: str = "",
                     strategy: str = "", category: str = "", forecast_prob: float = None,
                     market_price: float = None, edge: float = 0.0, side: str = "",
                     amount_usd: float = 0.0, **extra):
        """
        Record an opened trade.

        `extra` absorbs caller-specific fields (confidence, data_mode,
        trust_tier) that this tracker does not use, so a caller cannot fail the
        whole record because it passed one field too many. That is the bug that
        silenced V3's learning: a TypeError on an unknown keyword discarded the
        entire outcome.

        Required to be meaningful: a trade id to correlate the resolution later,
        and the market/venue/strategy the result will be attributed to. A record
        missing any of those is refused with a reason, because it cannot
        influence allocation even if stored.
        """
        missing = [name for name, value in (
            ("trade_id", trade_id), ("market_id", market_id),
            ("venue_id", venue_id), ("strategy", strategy)) if not value]
        if missing:
            logger.error(
                f"Trade outcome REFUSED for {market_id or 'unknown market'}: "
                f"missing {missing}. An outcome with no {'/'.join(missing)} "
                f"cannot be attributed to a venue or strategy, so it cannot "
                f"inform allocation.")
            return False

        # forecast_prob is what the Brier score and `was_correct` are computed
        # from. Falling back to the market price silently records the market's
        # view as if it were ours and would corrupt model calibration.
        if forecast_prob is None:
            logger.error(
                f"Trade outcome REFUSED for {trade_id}: no forecast_prob, so "
                f"calibration for this trade cannot be computed. Recording the "
                f"market price as a forecast would corrupt every score it feeds.")
            return False

        if market_price is None:
            logger.error(
                f"Trade outcome REFUSED for {trade_id}: no market_price, so the "
                f"edge and P&L attributed to this trade would be invented.")
            return False

        outcome = TradeOutcome(
            trade_id=str(trade_id),
            market_id=market_id,
            venue_id=venue_id,
            strategy=strategy,
            category=category or "default",
            forecast_prob=float(forecast_prob),
            market_price=float(market_price),
            edge=float(edge or 0.0),
            side=side,
            amount_usd=float(amount_usd or 0.0),
        )
        self.outcomes.append(outcome)
        self._persist(outcome)
        logger.info(
            f"Trade outcome recorded {trade_id}: {market_id} {venue_id} "
            f"{strategy} edge {outcome.edge:.3f}")
        return True

    def record_resolution(self, trade_id: str, actual_outcome: float, pnl: float) -> bool:
        for o in self.outcomes:
            if o.trade_id == trade_id:
                if o.actual_outcome is not None:
                    logger.warning(
                        f"Trade {trade_id} already resolved as {o.actual_outcome}; "
                        f"ignoring {actual_outcome}")
                    return False
                o.actual_outcome = actual_outcome
                o.pnl = pnl
                o.resolved_at = datetime.now(timezone.utc)
                o.brier_score = (o.forecast_prob - actual_outcome) ** 2
                predicted_yes = o.forecast_prob >= 0.5
                actual_yes = actual_outcome == 1.0
                o.was_correct = predicted_yes == actual_yes
                self._persist(o, update=True)
                logger.info(
                    f"Trade resolved {trade_id}: actual {actual_outcome} "
                    f"pnl ${pnl:.2f} correct {o.was_correct} brier {o.brier_score:.3f}")
                return True

        logger.warning(
            f"Trade resolution for unknown trade_id {trade_id} - nothing to "
            f"update. If the trade was opened before the last restart, this "
            f"means its outcome was not persisted.")
        return False

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
