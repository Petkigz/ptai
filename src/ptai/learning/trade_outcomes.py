"""
Trade Outcomes - tracks whether prediction worked, which venue/strategy/model works
"""
import math
from typing import Any, Dict, List, Optional


def _optional_float(value) -> Optional[float]:
    """A number, or None when it was not measured. Zero and absent differ."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
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
    # Costs and execution mode. None means NOT MEASURED - which is different from
    # zero, and the qualification gate must not read it as free.
    fees_usd: Optional[float] = None
    slippage_bps: Optional[float] = None
    execution_quality: Optional[float] = None
    data_mode: str = ""


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
                    "brier_score, was_correct, recorded_at, fees_usd, "
                    "slippage_bps, execution_quality, data_mode) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (outcome.trade_id, outcome.market_id, outcome.venue_id,
                     outcome.strategy, outcome.category, outcome.forecast_prob,
                     outcome.market_price, outcome.edge, outcome.side,
                     outcome.amount_usd, outcome.actual_outcome, outcome.pnl,
                     outcome.resolved_at.isoformat() if outcome.resolved_at else None,
                     outcome.brier_score, int(outcome.was_correct),
                     datetime.now(timezone.utc).isoformat(),
                     outcome.fees_usd, outcome.slippage_bps,
                     outcome.execution_quality, outcome.data_mode or None))
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
            # Genuinely optional, and stored as None when absent so the gate can
            # tell "no fee was charged" from "we did not look".
            fees_usd=_optional_float(extra.get("fees_usd")),
            slippage_bps=_optional_float(extra.get("slippage_bps")),
            execution_quality=_optional_float(extra.get("execution_quality")),
            data_mode=str(extra.get("data_mode") or ""),
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


# ----------------------------------------------------------------------
# feeding the qualification gate from what actually happened
# ----------------------------------------------------------------------
#
# The venue qualification gate reads its numbers from two places, and neither of
# them was ever written in production:
#
#   * adapter.performance_stats - a dict initialised to {"total_paper_trades": 0}
#     that no code path ever increments.
#   * data/venue_qualification.json - loaded at startup, and written ONLY by
#     evaluate_qualification, which the trading loop never called.
#
# So the gate's sample size was permanently zero (a venue could never earn its
# way to live no matter how well it traded), while any leftover file on disk -
# written by a test, a demo or an older version - was read back as EVIDENCE. A
# stale entry reading "120 trades, brier 0.18, profit factor 1.5" satisfies three
# of the four statistical gates, which is the wrong way to fail.
#
# This function is the connection that was missing: the numbers come from
# recorded outcomes, in the database, and nowhere else.

def qualification_stats_from_outcomes(storage, venue_id: str) -> Dict[str, Any]:
    """
    The qualification gate's inputs for one venue, from the outcome log.

    Returns zeros (with failing brier/log-loss) when nothing is recorded, so an
    unmeasured venue fails closed rather than inheriting anything.
    """
    empty = {
        "total_paper_trades": 0, "win_rate": 0.0, "avg_edge": 0.0,
        "brier_score": 1.0, "log_loss": 1.0, "calibration_ece": 0.5,
        "forecast_skill": 0.0, "profit_paper": 0.0, "profit_live": 0.0,
        "net_pnl": 0.0, "expected_value": 0.0, "profit_factor": 0.0,
        "drawdown_max": 0.0, "fees_total": 0.0, "slippage_total": 0.0,
        # Unmeasured, and 0.0 rather than the 0.5 threshold: the threshold value
        # used to sit here as a default, so every unmeasured venue passed the
        # execution-quality check by coincidence.
        "execution_quality_avg": 0.0,
        "costs_measured": 0, "slippage_measured": 0,
        "modes": [], "live_trades": 0, "paper_trades": 0,
        "source": "no recorded outcomes",
    }
    if storage is None:
        return empty
    try:
        rows = storage.conn.execute(
            """
            SELECT forecast_prob, edge, actual_outcome, pnl, brier_score,
                   was_correct, amount_usd, fees_usd, slippage_bps,
                   execution_quality, data_mode
            FROM trade_outcomes
            WHERE venue_id = ? AND actual_outcome IS NOT NULL
            ORDER BY COALESCE(resolved_at, recorded_at)
            """,
            (venue_id,),
        ).fetchall()
    except Exception as e:
        logger.error(
            f"Could not read outcomes for {venue_id}: {type(e).__name__}: {e}. "
            f"Reporting no evidence rather than assuming competence.")
        return empty
    if not rows:
        return empty

    n = len(rows)
    wins = sum(1 for r in rows if r["was_correct"])

    def _f(row, key, default=0.0):
        value = row[key]
        return default if value is None else float(value)

    pnls = [_f(r, "pnl") for r in rows]
    edges = [_f(r, "edge") for r in rows]
    probs = [min(max(_f(r, "forecast_prob", 0.5), 1e-6), 1 - 1e-6) for r in rows]
    actuals = [1.0 if _f(r, "actual_outcome") > 0.5 else 0.0 for r in rows]

    brier_values = [r["brier_score"] for r in rows]
    if all(b is not None for b in brier_values):
        brier = sum(float(b) for b in brier_values) / n
    else:
        brier = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / n

    log_loss = sum(
        -(a * math.log(p) + (1 - a) * math.log(1 - p))
        for p, a in zip(probs, actuals)
    ) / n

    # Expected calibration error over ten probability buckets: the average gap
    # between what the agent claimed and what happened. A high Brier score with a
    # low ECE means bad forecasts; a high ECE means mis-calibrated ones, and they
    # need different fixes.
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        bucket = [(p, a) for p, a in zip(probs, actuals) if lo <= p < hi]
        if not bucket:
            continue
        avg_conf = sum(p for p, _ in bucket) / len(bucket)
        avg_actual = sum(a for _, a in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_actual)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    # Fees and slippage that were actually recorded. Counted only where the
    # trade carries them, and reported alongside how many trades were measured -
    # a total over 3 of 150 trades must not read like a total over all of them.
    fee_values = [float(r["fees_usd"]) for r in rows if r["fees_usd"] is not None]
    slip_values = [float(r["slippage_bps"]) for r in rows if r["slippage_bps"] is not None]
    quality_values = [float(r["execution_quality"]) for r in rows
                      if r["execution_quality"] is not None]
    modes = [str(r["data_mode"] or "") for r in rows]

    # Realised P&L split by execution mode. Qualification is supposed to weigh
    # paper and live separately; with no mode recorded they were one pool, so a
    # paper result could carry a venue toward live capital.
    live_rows = [r for r in rows if str(r["data_mode"] or "").lower() in ("live", "real")]
    paper_rows = [r for r in rows if str(r["data_mode"] or "").lower() not in ("live", "real")]
    profit_live = sum(_f(r, "pnl") for r in live_rows)
    profit_paper = sum(_f(r, "pnl") for r in paper_rows)

    # Slippage in USD: basis points of the notional actually traded.
    slippage_usd = sum(
        float(r["slippage_bps"]) / 10000.0 * float(r["amount_usd"] or 0.0)
        for r in rows if r["slippage_bps"] is not None
    )

    # Execution quality, MEASURED from recorded slippage. Unmeasured is 0.0, not
    # the 0.5 threshold: an unmeasured venue must fail the execution-quality
    # check rather than pass it by coincidence.
    if quality_values:
        execution_quality = sum(quality_values) / len(quality_values)
    elif slip_values:
        # 0 bps of slippage is perfect; 500 bps (5%) is worthless.
        execution_quality = max(
            0.0, 1.0 - (sum(slip_values) / len(slip_values)) / 500.0)
    else:
        execution_quality = 0.0

    # Maximum drawdown on the realised equity curve, as a fraction of the peak.
    #
    # Starting equity is the RECORDED INITIAL bankroll, not the current one.
    # Using the current bankroll replayed history from a point in the future: the
    # curve was anchored at today's balance, so the drawdown described a past that
    # never happened, and it moved every time the balance did.
    try:
        start = float(storage.get_performance_summary().get("initial_bankroll")
                      or storage.get_bankroll())
    except Exception:
        start = storage.get_bankroll()
    equity, peak, drawdown = start, start, 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)

    return {
        "total_paper_trades": n,
        "win_rate": wins / n,
        "avg_edge": sum(edges) / n,
        "brier_score": brier,
        "log_loss": log_loss,
        "calibration_ece": ece,
        "forecast_skill": max(0.0, 1 - brier * 2),
        # Every recorded outcome, paper and live. Named profit_paper because that
        # is the key the gate reads; the split is reported alongside.
        "profit_paper": profit_paper,
        "profit_live": profit_live,
        "net_pnl": sum(pnls),
        "expected_value": sum(edges) / n,
        "profit_factor": profit_factor,
        "drawdown_max": drawdown,
        "fees_total": sum(fee_values),
        "slippage_total": slippage_usd,
        "execution_quality_avg": execution_quality,
        # How much of the record the cost figures actually cover, so a partial
        # measurement cannot be mistaken for a complete one.
        "costs_measured": len(fee_values),
        "slippage_measured": len(slip_values),
        "modes": sorted(set(m for m in modes if m)),
        "live_trades": len(live_rows),
        "paper_trades": len(paper_rows),
        "source": f"{n} recorded outcomes",
    }
