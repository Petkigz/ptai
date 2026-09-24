"""
Settlement Engine - closes the execution -> outcome -> calibration chain.

This module did not exist, and its absence is why the agent could not learn.

`CalibrationEngine.record_forecast` had two production callers. Its
`record_resolution` had zero, anywhere in `src/ptai/`. So:

  * the resolved-forecast count stayed at 0 forever,
  * `is_degrading()` is guarded by `resolved >= 50`, so it always returned False,
  * `kill_switch.check_calibration_collapse(...)` therefore could never fire, and
  * `trades.resolved` was written as 0 and never updated, which made
    `get_performance_summary()` compute win rate over an empty set forever.

Every cycle, this engine walks the forecasts and trades that are still open,
asks the venue that hosts each market whether it has settled, and records the
outcome when - and only when - the venue gives a real answer.

The rule throughout: an outcome that cannot be read is NOT an outcome. A
guessed resolution becomes a permanent, wrong calibration point, and the agent
would then adjust its probabilities using data it invented. So every path that
cannot establish a real settlement records nothing and says why.
"""

from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SettledItem:
    """One forecast or trade that reached a determination."""
    market_id: str
    venue_id: str
    kind: str                    # "forecast" | "trade"
    applied: bool
    reason: str
    outcome: Optional[float] = None
    pnl: Optional[float] = None
    source: str = ""


@dataclass
class SettlementReport:
    checked_forecasts: int = 0
    checked_trades: int = 0
    settled: int = 0
    unresolved: int = 0
    unreadable: int = 0
    ambiguous: int = 0
    errors: int = 0
    realised_pnl_usd: float = 0.0
    # Reported SEPARATELY from the real figure above. Both used to be summed
    # into one number, so a report reading "realised +$9.20" could be $8.00 of
    # simulation and $1.20 of real money - and the operator has no way to tell,
    # which makes the one number they care about unreliable.
    paper_pnl_usd: float = 0.0
    live_settled: int = 0
    paper_settled: int = 0
    items: List[SettledItem] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked_forecasts": self.checked_forecasts,
            "checked_trades": self.checked_trades,
            "settled": self.settled,
            "unresolved": self.unresolved,
            "unreadable": self.unreadable,
            "ambiguous": self.ambiguous,
            "errors": self.errors,
            "realised_pnl_usd": round(self.realised_pnl_usd, 2),
            "paper_pnl_usd": round(self.paper_pnl_usd, 2),
            "live_settled": self.live_settled,
            "paper_settled": self.paper_settled,
            "execution_time": round(self.execution_time, 2),
            "items": [
                {"market_id": i.market_id, "venue_id": i.venue_id, "kind": i.kind,
                 "applied": i.applied, "outcome": i.outcome, "pnl": i.pnl,
                 "source": i.source, "reason": i.reason}
                for i in self.items
            ],
        }


def compute_pnl(side: str, entry_price: float, stake_usd: float,
                outcome: float, price_is_token_price: bool = False) -> Optional[float]:
    """
    P&L for a binary position held to settlement.

    `side` is what we bought: YES/1/LONG pays when outcome is 1.0, NO/0/SHORT
    pays when outcome is 0.0.

    `price_is_token_price` says which of the two meanings `entry_price` has, and
    the two are NOT interchangeable:

      * False (the original contract) - `entry_price` is the YES price, so a NO
        buy paid 1 - entry. Callers holding only a YES price use this, and
        Polymarket's NO price is 1 - YES price to within the spread.
      * True - `entry_price` is the price of the token actually bought, which is
        what the execution path fills at and now stores as
        `token_price_at_entry`. Buying NO at 0.30 means paying 0.30 a share.

    Reading one as the other is not a rounding error. Buying NO at 0.30 and
    settling on the YES scale pays out as though the shares cost 0.70, turning a
    +$7.00 win on a $3 stake into +$1.29.

    Returns None when the inputs cannot produce an honest number - a settlement
    P&L computed from a missing price is worse than no P&L, because it moves the
    bankroll.
    """
    if outcome is None:
        return None
    try:
        entry = float(entry_price)
        stake = float(stake_usd)
    except (TypeError, ValueError):
        return None
    if stake <= 0 or not (0.0 < entry < 1.0):
        return None
    if outcome not in (0.0, 1.0):
        return None

    side_norm = str(side or "").strip().upper()
    bought_yes = side_norm in ("YES", "1", "LONG", "BUY", "TRUE")

    if bought_yes:
        price_paid = entry
        won = outcome == 1.0
    else:
        price_paid = entry if price_is_token_price else 1.0 - entry
        won = outcome == 0.0

    if price_paid <= 0:
        return None

    shares = stake / price_paid
    payout = shares * (1.0 if won else 0.0)
    return round(payout - stake, 6)


class SettlementEngine:
    """Walk open forecasts and trades, settle what the venues confirm."""

    def __init__(self, venue_registry=None, storage=None,
                 calibration_engine=None, trade_outcome_tracker=None):
        self.venue_registry = venue_registry
        self.storage = storage
        self.calibration_engine = calibration_engine
        self.trade_outcome_tracker = trade_outcome_tracker
        # Guards against re-asking a venue about a market it cannot answer for.
        # Without this, an adapter that does not implement settlement would be
        # queried on every cycle forever.
        self._unsupported_venues: set = set()

    def _adapter_for(self, venue_id: str):
        if not self.venue_registry:
            return None
        vid = str(venue_id or "").split("+")[0].strip().lower()
        return self.venue_registry.adapters.get(vid)

    async def settle_pending(self, max_markets: int = 50) -> SettlementReport:
        """
        Settle up to `max_markets` markets this cycle.

        Bounded deliberately: settlement is a network round trip per market, and
        a cycle must not block scanning for minutes.

        Two defects lived here and both had the same shape - settlement was
        considered a side effect of the CALIBRATION store rather than a
        responsibility of the position ledger:

          * `if not pending: return report` meant that with no unresolved
            forecasts, NO trade was ever settled. A fresh install, or any run
            where the calibration store is empty or was reset, would leave every
            open position open forever.
          * the loop iterated `pending` forecasts only, so an open trade whose
            market had no forecast entry was invisible to settlement. A position
            could be opened, never closed, and never contribute a P&L - the
            agent would keep accounting for capital it had already lost.

        The set of markets to ask about is now the UNION of markets with a
        pending forecast and markets with an open trade. Whether a forecast
        exists is irrelevant to whether a position needs closing.
        """
        import time
        start = time.time()
        report = SettlementReport()

        pending = []
        if self.calibration_engine is not None:
            pending = self.calibration_engine.pending_forecasts()
        else:
            # Settlement of POSITIONS does not require a calibration engine.
            # Only the forecast half of the work below does.
            logger.warning(
                "SettlementEngine has no calibration engine - forecasts cannot "
                "be resolved, open positions still will be")
        report.checked_forecasts = len(pending)

        # Trades awaiting settlement, keyed by market, so a settlement can close
        # the position as well as the forecast.
        open_trades: Dict[str, Dict] = {}
        if self.storage:
            try:
                for t in self.storage.get_unresolved_trades(limit=max_markets * 4):
                    open_trades.setdefault(str(t.get("market_id")), t)
            except Exception as e:
                logger.error(f"Settlement: could not read open trades: {e}")
        report.checked_trades = len(open_trades)

        # Forecasts first (they carry a venue hint), then trade-only markets.
        pending_by_market: Dict[str, Any] = {}
        market_order: List[str] = []
        for point in pending:
            market_id = str(point.market_id)
            if market_id not in pending_by_market:
                pending_by_market[market_id] = point
                market_order.append(market_id)
        for market_id in open_trades:
            if market_id not in pending_by_market:
                market_order.append(market_id)

        if not market_order:
            report.execution_time = time.time() - start
            logger.info("Settlement: no pending forecasts and no open positions")
            return report

        seen_markets = set()
        for market_id in market_order:
            if len(seen_markets) >= max_markets:
                break
            if market_id in seen_markets:
                continue
            seen_markets.add(market_id)

            point = pending_by_market.get(market_id)
            trade = open_trades.get(market_id)
            has_forecast = point is not None

            venue_id = self._venue_for_market(market_id, open_trades, point)
            if not venue_id:
                report.unreadable += 1
                report.items.append(SettledItem(
                    market_id=market_id, venue_id="",
                    kind="forecast" if has_forecast else "trade",
                    applied=False, source="no_venue",
                    reason="no venue associated with this market, so nothing "
                           "can be asked about its resolution",
                ))
                continue

            if venue_id in self._unsupported_venues:
                report.unreadable += 1
                continue

            adapter = self._adapter_for(venue_id)
            if adapter is None:
                report.unreadable += 1
                report.items.append(SettledItem(
                    market_id=market_id, venue_id=venue_id,
                    kind="forecast" if has_forecast else "trade",
                    applied=False, source="no_adapter",
                    reason=f"no adapter registered for {venue_id}",
                ))
                continue

            try:
                verdict = await adapter.get_settlement(market_id)
            except Exception as e:
                report.errors += 1
                report.items.append(SettledItem(
                    market_id=market_id, venue_id=venue_id,
                    kind="forecast" if has_forecast else "trade",
                    applied=False, source="error",
                    reason=f"{type(e).__name__}: {e}",
                ))
                continue

            if not isinstance(verdict, dict) or not verdict.get("is_real"):
                # The venue could not tell us. Never record an outcome here.
                source = (verdict or {}).get("source", "unknown")
                if source == "unsupported":
                    self._unsupported_venues.add(venue_id)
                    logger.info(
                        f"Settlement: {venue_id} cannot report settlement - "
                        f"skipping it for the rest of this run rather than "
                        f"asking every cycle")
                report.unreadable += 1
                report.items.append(SettledItem(
                    market_id=market_id, venue_id=venue_id,
                    kind="forecast" if has_forecast else "trade",
                    applied=False, source=str(source),
                    reason=(verdict or {}).get("reason", "no real settlement data"),
                ))
                continue

            if not verdict.get("settled"):
                report.unresolved += 1
                continue

            outcome = verdict.get("outcome")
            if outcome is None:
                report.ambiguous += 1
                report.items.append(SettledItem(
                    market_id=market_id, venue_id=venue_id,
                    kind="forecast" if has_forecast else "trade",
                    applied=False, source=str(verdict.get("source", "")),
                    reason=verdict.get("reason", "settled but outcome ambiguous"),
                ))
                continue

            item = SettledItem(
                market_id=market_id, venue_id=venue_id,
                kind="forecast" if has_forecast else "trade",
                applied=False, outcome=float(outcome),
                source=str(verdict.get("source", "")),
                reason=verdict.get("reason", "settled"),
            )
            applied_something = False

            # --- the forecast, if one is pending for this market ---
            #
            # A failure to resolve the forecast no longer `continue`s past the
            # trade. Forecast bookkeeping and position closure are separate
            # jobs, and a calibration problem must not strand a position.
            if has_forecast:
                try:
                    applied = self.calibration_engine.record_resolution(
                        point.forecast_id, float(outcome))
                except Exception as e:
                    report.errors += 1
                    logger.error(
                        f"Settlement: recording resolution for {market_id} failed: "
                        f"{type(e).__name__}: {e}")
                    applied = False
                if applied:
                    applied_something = True
                else:
                    logger.warning(
                        f"Settlement: forecast {point.forecast_id} on "
                        f"{market_id} was not applied")

            # --- the trade, if one is open on this market ---
            if trade is not None:
                # The price ACTUALLY PAID per share, from the token that was
                # bought - not re-derived from the YES price.
                #
                # `market_price` used to hold the fill price after the order
                # filled, and `compute_pnl` was handed it as though it were the
                # YES price, from which it computed 1 - entry for a NO buy. Once
                # execution moved to pricing the NO token, that was applied to a
                # number that was already the NO price:
                #
                #   YES 0.70, NO token bought at 0.30, $3 stake
                #     correct:  $3 / 0.30 - $3 = +$7.00 when NO wins
                #     as read:  $3 / (1 - 0.30) - $3 = +$1.29
                #
                # so a winning NO was paid out as if it had cost 0.70 a share.
                #
                # `token_price_at_entry` is stored now. Rows written before it
                # existed put the FILL price in `market_price`, and the fill is
                # always on the token that was bought (the paper path resolves
                # the same token and reads its book), so the same reading is the
                # correct one for them - it corrects their NO accounting rather
                # than preserving the old under-count.
                # Which account this settlement pays into. Read from the row,
                # the same way `resolve_trade` decides, so the report cannot
                # disagree with what actually moved.
                paper_settlement = (
                    str(trade.get("execution_mode") or "").strip().lower()
                    == "paper"
                    or (not trade.get("execution_mode")
                        and str(trade.get("status") or "").lower() == "paper"))

                entry_price = trade.get("token_price_at_entry")
                price_source = "token_price_at_entry"
                if entry_price is None:
                    entry_price = trade.get("market_price")
                    price_source = "market_price (legacy: YES-scale re-derived)"
                pnl = compute_pnl(
                    side=trade.get("side"),
                    entry_price=entry_price,
                    stake_usd=trade.get("position_size_usd"),
                    outcome=float(outcome),
                    price_is_token_price=True,
                )
                if pnl is None:
                    item.pnl = None
                    logger.warning(
                        f"Settlement: trade {trade.get('id')} on {market_id} "
                        f"cannot be valued (side={trade.get('side')}, "
                        f"price={trade.get('market_price')}, "
                        f"stake={trade.get('position_size_usd')}) - leaving it open")
                else:
                    try:
                        closed = self.storage.resolve_trade(
                            trade_id=trade.get("id"),
                            outcome=float(outcome),
                            pnl=pnl,
                            notes=f"settled via {verdict.get('source')}",
                        )
                    except Exception as e:
                        report.errors += 1
                        logger.error(f"Settlement: resolve_trade failed: {e}")
                        closed = False
                    if closed:
                        item.pnl = pnl
                        item.kind = "trade"
                        applied_something = True
                        logger.info(
                            f"Settlement: trade {trade.get('id')} side="
                            f"{trade.get('side')} entry={entry_price} "
                            f"({price_source}) outcome={outcome} "
                            f"pnl={pnl:+.2f}")
                        if paper_settlement:
                            report.paper_pnl_usd += pnl
                            report.paper_settled += 1
                        else:
                            report.realised_pnl_usd += pnl
                            report.live_settled += 1
                        if self.trade_outcome_tracker is not None:
                            try:
                                self.trade_outcome_tracker.record_resolution(
                                    trade_id=str(trade.get("id")),
                                    actual_outcome=float(outcome),
                                    pnl=pnl,
                                )
                            except Exception as e:
                                # Loud, per the standing rule: a silent failure
                                # to record is worse than a visible one.
                                logger.error(
                                    f"Settlement: trade outcome not recorded for "
                                    f"trade {trade.get('id')}: {type(e).__name__}: {e}")
                    else:
                        logger.warning(
                            f"Settlement: trade {trade.get('id')} on {market_id} "
                            f"was already resolved or could not be closed")

            if applied_something:
                item.applied = True
                report.settled += 1
            report.items.append(item)

        report.execution_time = time.time() - start
        logger.info(
            f"Settlement: {report.settled} settled, {report.unresolved} still open, "
            f"{report.ambiguous} ambiguous, {report.unreadable} unreadable, "
            f"{report.errors} errors, realised ${report.realised_pnl_usd:+.2f} "
            f"paper ${report.paper_pnl_usd:+.2f} "
            f"in {report.execution_time:.1f}s")
        return report

    def _venue_for_market(self, market_id: str,
                          open_trades: Dict[str, Dict],
                          point=None) -> str:
        """
        Which venue hosts this market.

        The trade record knows its venue and is authoritative. Failing that the
        forecast carries the venue it was recorded against. If neither knows,
        this returns "" rather than guessing - trying venues in turn until one
        answers would attach an outcome to whichever venue happened to reply,
        with no provenance at all.
        """
        trade = open_trades.get(market_id)
        if trade and trade.get("venue_id"):
            return str(trade["venue_id"])

        if point is not None and getattr(point, "venue_id", None):
            return str(point.venue_id)

        # Fall back to scanning known adapters for one that recognises the id.
        if self.venue_registry:
            for vid, adapter in self.venue_registry.adapters.items():
                if vid in self._unsupported_venues:
                    continue
                # Polymarket ids are numeric strings; kalshi ids are uppercase
                # tickers. Prefer an exact-prefix match where the id carries it.
                if market_id.startswith(f"{vid}:"):
                    return vid
        return ""

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "SettlementEngine",
            "principle": (
                "An outcome that cannot be read is not an outcome. Every path "
                "that cannot establish a real settlement records nothing and "
                "says why."
            ),
            "closes": "execution -> settlement -> outcome -> calibration -> qualification -> allocation",
            "unsupported_venues_this_run": sorted(self._unsupported_venues),
        }
