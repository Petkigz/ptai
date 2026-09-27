"""
Sports bets: placed, recorded, settled, and learned from - the way trades are.

The sports lane could price a full match card and had NOTHING that recorded a
bet. It scanned fixtures, built `BetOpportunity` objects, printed "3 executable",
and dropped them: no position, no settlement, no P&L, no learning, no panel. The
betting half of PTAI was a calculator sitting next to a trading system, which is
exactly what "the sports side looks lacking compared to the trading side" means.

This module gives it the same lifecycle the prediction path has:

    quote  -> place  -> record  -> settle  -> P&L  -> learn

and it is explicit about the two things it does NOT do:

  * it does not invent a result. A bet settles only when a feed reports the
    fixture FINISHED with a score. No result, no settlement - the bet stays open
    and the console says which feed it is waiting for;
  * it does not claim a bet it cannot settle. Markets whose result cannot be read
    from a score (corners, cards, props - they need per-match statistics, not the
    final score) are REFUSED at placement with that reason on them. Accepting a
    bet whose outcome can never be known is not risk-taking, it is bookkeeping
    that only looks like trading.

What a bet costs and pays:

  * BACK: win -> stake x (decimal odds - 1), less commission on the winnings for
    an exchange; lose -> -stake.
  * LAY: win -> +stake less commission; lose -> -stake x (decimal odds - 1). The
    liability is recorded at placement, because on an exchange that is the money
    actually at risk.
  * PUSH/VOID (a totals line landing exactly on the number): stake returned, the
    bet is closed out and deliberately EXCLUDED from the learning record - a bet
    nobody won teaches nothing about the model.

Every placed bet also writes a `trade_outcomes` row, so it flows into the same
machinery as everything else: qualification, the venue x strategy matrix, the
money guard's loss limits, and the calibration record.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

#: Market types whose outcome is decidable from the final score alone.
SETTLEABLE_MARKETS = {"h2h", "totals", "spreads"}

#: Market families PTAI prices but cannot settle yet, and what they would need.
UNSETTLEABLE_REASONS = {
    "btts": "needs both-teams-to-score, which no configured feed returns",
    "correct_score": "needs an exact-score feed, not a result",
    "corners": "needs per-match corner statistics, which no configured feed returns",
    "cards": "needs per-match card statistics, which no configured feed returns",
    "red_card": "needs per-match card statistics, which no configured feed returns",
    "booking": "needs per-match card statistics, which no configured feed returns",
    "shots": "needs per-match shot statistics, which no configured feed returns",
    "offsides": "needs per-match offside statistics, which no configured feed returns",
    "props": "needs confirmed lineups and player statistics",
    "half": "needs half-time scores from a statistics feed",
    "ht_ft": "needs half-time scores from a statistics feed",
    "race_to": "needs in-play goal timings, which no configured feed returns",
    "team": "needs per-team aggregate models with their own settlement rules",
    "clean_sheet": "needs a per-team result feed, not a fixture result",
    "win_to_nil": "needs a per-team result feed, not a fixture result",
    "total": "needs a line to settle against",
}


def classify_market(market_type: str):
    """
    (kind, line, refusal) for a market key.

    The engine names its markets after their content - "totals_2.5",
    "asian_handicap_-0.5", "btts" - so settleability is decided by reading the
    name against the settlement rules PTAI actually has, not by a hand-kept list
    that would silently fall behind the pricing code.

    Quarter-line Asian handicaps (-0.25, +0.75) are deliberately REFUSED. They
    settle half-win or half-loss, and doing that arithmetic wrong would produce a
    P&L that looks precise and is not - the bet stays unplaced with the reason on
    it instead.
    """
    key = str(market_type or "").strip().lower()
    if key == "h2h":
        return "h2h", None, ""
    if key.startswith("totals"):
        line = _suffix_number(key)
        if line is None:
            return "totals", None, ("totals market with no readable line, so the "
                                    "result could not be decided")
        return "totals", line, ""
    if "handicap" in key:
        line = _suffix_number(key)
        if line is None:
            return "spreads", None, ("handicap market with no readable handicap, "
                                     "so the result could not be decided")
        # A quarter line is a line whose quadruple is an ODD integer: the bet is
        # split across the two neighbouring half-lines and can win or lose half
        # of itself. -0.5 and -1.0 are quarter-line-free; -0.25 and +0.75 are not.
        if round(line * 4) % 2 != 0:
            return "spreads", line, (
                f"quarter-line handicap ({line:+g}) settles half-win or half-loss; "
                f"that arithmetic is not implemented, so this market is not priced "
                f"for real money yet")
        return "spreads", line, ""
    for family, why in UNSETTLEABLE_REASONS.items():
        if key.startswith(family):
            return "", None, f"cannot be settled: {why}"
    return "", None, (f"cannot be settled: no settlement rule for market type "
                      f"{market_type!r} yet")


def _suffix_number(key: str):
    """The first number in a market key ("totals_2.5" -> 2.5), or None."""
    import re  # noqa: WPS433 - one regex, used here and nowhere else

    match = re.search(r"[-+]?\d+(?:\.\d+)?", key)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sports_bets (
    bet_id TEXT PRIMARY KEY,
    event_key TEXT NOT NULL,
    league TEXT,
    sport TEXT,
    market_type TEXT,
    outcome TEXT,
    line REAL,
    side TEXT,
    book TEXT,
    odds REAL,
    stake_usd REAL,
    liability_usd REAL,
    forecast_prob REAL,
    market_prob REAL,
    edge REAL,
    data_mode TEXT,
    execution_mode TEXT,
    status TEXT DEFAULT 'open',
    pnl_usd REAL,
    settle_reason TEXT,
    settlement_source TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sports_bets_status ON sports_bets (status, event_key);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SportsBet:
    bet_id: str
    event_key: str
    league: str
    sport: str
    market_type: str
    outcome: str
    line: Optional[float]
    side: str
    book: str
    odds: float
    stake_usd: float
    liability_usd: float
    forecast_prob: Optional[float]
    market_prob: Optional[float]
    edge: Optional[float]
    data_mode: str
    execution_mode: str
    status: str = "open"
    pnl_usd: Optional[float] = None
    settle_reason: str = ""
    settlement_source: str = ""
    created_at: str = field(default_factory=_now)
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class SportsBook:
    """The sports lane's position book."""

    def __init__(self, storage, tracker=None, data_engine=None,
                 min_stake: float = 1.0, max_bets_per_cycle: int = 1,
                 commission_pct: float = 0.0):
        self.storage = storage
        self.tracker = tracker
        self.data = data_engine
        self.min_stake = float(min_stake)
        self.max_bets_per_cycle = int(max_bets_per_cycle)
        self.commission_pct = float(commission_pct)
        self.last_placement: Dict[str, Any] = {}
        self.last_settlement: Dict[str, Any] = {}
        if self.storage is not None:
            try:
                self.storage.conn.executescript(SCHEMA)
                self.storage.conn.commit()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not create the sports bet ledger: "
                             f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def placement_blockers(self, opp: Any) -> List[str]:
        """
        Why this bet must NOT be recorded. Empty list means it may be.

        Every one of these is a refusal with a reason on it, never a silent drop:
        "it priced 3 bets and placed none" was the state this module replaced, and
        a silent drop is indistinguishable from a bug.
        """
        blockers: List[str] = []
        if getattr(opp, "blockers", None):
            blockers.extend(str(b) for b in opp.blockers)
        _kind, _line, refusal = classify_market(getattr(opp, "market_type", ""))
        if refusal:
            blockers.append(refusal)
        stake = float(getattr(opp, "stake", 0.0) or 0.0)
        if stake < self.min_stake:
            blockers.append(f"stake ${stake:.2f} below the ${self.min_stake:.2f} "
                            f"minimum order size")
        if not float(getattr(opp, "price", 0.0) or 0.0) > 1.0:
            blockers.append("no usable price (decimal odds must be above 1.0)")
        if not getattr(opp, "event_key", ""):
            blockers.append("no fixture key, so the result could never be found")
        return blockers

    def place(self, cycle_result: Dict[str, Any], *,
              execution_mode: str = "paper",
              bankroll: Optional[float] = None,
              guard=None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Record the bets this cycle's scan selected.

        `limit` bounds how many are placed (one on a $50 account): the lane is
        learning, and a full match card of correlated bets on the same fixture is
        one position wearing several names.

        `guard` is the money guard, consulted exactly as the trading path
        consults it - a loss limit that only stopped prediction markets would be
        half a limit.
        """
        limit = self.max_bets_per_cycle if limit is None else int(limit)
        candidates: Sequence[Any] = cycle_result.get("top") or []
        # `top` holds the five best as dicts. The full list is on the engine; the
        # cycle result is what this function is handed, so it works from `top`.
        placed: List[Dict[str, Any]] = []
        refused: List[Dict[str, Any]] = []

        existing = self.open_keys()
        for raw in candidates:
            if len(placed) >= limit:
                break
            blockers = self.placement_blockers_dict(raw)
            if blockers:
                refused.append({"outcome": raw.get("outcome"),
                                "market_type": raw.get("market_type"),
                                "reasons": blockers})
                continue
            key = self._identity(raw)
            if key in existing:
                continue

            stake = float(raw.get("stake") or 0.0)
            if guard is not None:
                try:
                    decision = guard.check(stake, execution_mode, bankroll or 0.0,
                                           venue_id=str(raw.get("book") or ""))
                    if decision.refused:
                        refused.append({"outcome": raw.get("outcome"),
                                        "market_type": raw.get("market_type"),
                                        "reasons": [f"money guard: {decision.reason}"]})
                        continue
                    stake = float(decision.approved_usd or 0.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Money guard could not judge a sports bet "
                                   f"({type(e).__name__}: {e}) - refusing it")
                    refused.append({"outcome": raw.get("outcome"),
                                    "market_type": raw.get("market_type"),
                                    "reasons": [f"money guard failed: {e}"]})
                    continue
            if stake < self.min_stake:
                refused.append({"outcome": raw.get("outcome"),
                                "market_type": raw.get("market_type"),
                                "reasons": [f"stake ${stake:.2f} below the minimum"]})
                continue

            bet = self._bet_from(raw, stake, execution_mode)
            if self._insert(bet):
                placed.append(bet.to_dict())
                existing.add(key)
                self._learn_forward(bet)

        self.last_placement = {
            "placed": len(placed), "refused": len(refused),
            "bets": placed, "refusals": refused,
            "execution_mode": execution_mode,
            "note": ("sports bets are simulated at the REAL quoted price: no "
                     "configured sports book has an order path, and the record "
                     "says which mode each bet is in"),
        }
        if placed:
            logger.info(f"[sports] placed {len(placed)} paper bet(s): "
                        + "; ".join(f"{b['outcome']} @ {b['book']} "
                                    f"${b['stake_usd']:.2f} ({b['odds']:.2f})"
                                    for b in placed))
        if refused:
            logger.info(f"[sports] {len(refused)} candidate(s) not placed: "
                        + " | ".join(
                            f"{r['outcome']}: {r['reasons'][0]}" for r in refused[:3]))
        return self.last_placement

    def _bet_from(self, raw: Dict[str, Any], stake: float,
                  execution_mode: str) -> SportsBet:
        odds = float(raw.get("price") or 0.0)
        _kind, parsed_line, _refusal = classify_market(raw.get("market_type"))
        # The line travels with the bet, taken from the market key at PLACEMENT.
        # Re-reading it at settlement would mean a rename in the pricing code
        # could silently change what an already-placed bet settles against.
        line = raw.get("line") if raw.get("line") is not None else parsed_line
        side = str(raw.get("side") or "back").lower()
        liability = stake * (odds - 1.0) if side == "lay" else stake
        return SportsBet(
            bet_id=f"sports-{uuid.uuid4().hex[:12]}",
            event_key=str(raw.get("event_key") or ""),
            league=str(raw.get("league") or ""),
            sport=str(raw.get("sport") or ""),
            market_type=str(raw.get("market_type") or "").lower(),
            outcome=str(raw.get("outcome") or ""),
            line=line,
            side=side,
            book=str(raw.get("book") or "unknown book"),
            odds=odds,
            stake_usd=round(stake, 4),
            liability_usd=round(liability, 4),
            forecast_prob=raw.get("fair_prob"),
            market_prob=(1.0 / odds) if odds > 1.0 else None,
            edge=raw.get("edge"),
            data_mode=str(raw.get("data_mode") or ""),
            execution_mode=execution_mode,
        )

    @staticmethod
    def placement_blockers_dict(raw: Dict[str, Any]) -> List[str]:
        """`placement_blockers` for a dict (the shape the cycle result carries)."""
        probe = _DictOpp(raw)
        return SportsBook._blockers_for_dict(probe)

    @staticmethod
    def _blockers_for_dict(opp: "_DictOpp") -> List[str]:
        blockers = [str(b) for b in (opp.blockers or [])]
        _kind, _line, refusal = classify_market(opp.market_type)
        if refusal:
            blockers.append(refusal)
        if float(opp.stake or 0.0) <= 0:
            blockers.append("no stake: the sizing layer refused or produced zero")
        if not float(opp.price or 0.0) > 1.0:
            blockers.append("no usable price (decimal odds must be above 1.0)")
        if not opp.event_key:
            blockers.append("no fixture key, so the result could never be found")
        return blockers

    def _identity(self, raw: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return self._identity_parts(str(raw.get("event_key") or ""),
                                    str(raw.get("market_type") or ""),
                                    str(raw.get("outcome") or ""),
                                    str(raw.get("side") or "back"))

    @staticmethod
    def _identity_parts(event_key: str, market_type: str, outcome: str,
                        side: str) -> Tuple[str, str, str, str]:
        return (event_key, market_type.lower(), outcome.strip().lower(),
                side.strip().lower())

    def open_keys(self) -> set:
        return {self._identity_parts(b["event_key"], b["market_type"],
                                     b["outcome"], b["side"])
                for b in self.open_bets()}

    def _insert(self, bet: SportsBet) -> bool:
        try:
            self.storage.conn.execute(
                """INSERT INTO sports_bets
                   (bet_id, event_key, league, sport, market_type, outcome, line,
                    side, book, odds, stake_usd, liability_usd, forecast_prob,
                    market_prob, edge, data_mode, execution_mode, status, pnl_usd,
                    settle_reason, settlement_source, created_at, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',NULL,'','',?,NULL)""",
                (bet.bet_id, bet.event_key, bet.league, bet.sport,
                 bet.market_type, bet.outcome, bet.line, bet.side, bet.book,
                 bet.odds, bet.stake_usd, bet.liability_usd, bet.forecast_prob,
                 bet.market_prob, bet.edge, bet.data_mode, bet.execution_mode,
                 bet.created_at))
            self.storage.conn.commit()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not record sports bet {bet.bet_id}: "
                         f"{type(e).__name__}: {e}")
            return False

    def _learn_forward(self, bet: SportsBet) -> None:
        """
        Write the outcome row the rest of PTAI learns from.

        The prediction path records every trade this way; a sports bet that only
        lived in its own table would be invisible to qualification, to the venue x
        strategy matrix and to the money guard's loss limits.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.record_trade(
                trade_id=bet.bet_id,
                market_id=f"{bet.book}:{bet.event_key}:{bet.market_type}",
                venue_id=str(bet.book or "sports").lower().replace(" ", "_"),
                strategy=f"sports:{bet.market_type}",
                category=bet.league or "sports",
                forecast_prob=(bet.forecast_prob if bet.forecast_prob is not None
                               else bet.market_prob),
                market_price=bet.market_prob,
                edge=bet.edge or 0.0,
                side=("YES" if bet.side == "back" else "NO"),
                amount_usd=bet.stake_usd,
                execution_mode=bet.execution_mode,
                yes_price=bet.market_prob,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Sports bet {bet.bet_id} was placed but not recorded "
                         f"for learning: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def open_bets(self) -> List[Dict[str, Any]]:
        try:
            rows = self.storage.conn.execute(
                "SELECT * FROM sports_bets WHERE status='open' "
                "ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read open sports bets: {e}")
            return []

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            rows = self.storage.conn.execute(
                "SELECT * FROM sports_bets ORDER BY created_at DESC LIMIT ?",
                (int(limit),)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read sports bets: {e}")
            return []

    def summary(self) -> Dict[str, Any]:
        bets = self.recent(limit=500)
        settled = [b for b in bets if b["status"] in ("won", "lost")]
        pnl = sum(float(b["pnl_usd"] or 0.0) for b in bets)
        return {
            "open": sum(1 for b in bets if b["status"] == "open"),
            "settled": len(settled),
            "won": sum(1 for b in settled if b["status"] == "won"),
            "lost": sum(1 for b in settled if b["status"] == "lost"),
            "void": sum(1 for b in bets if b["status"] == "void"),
            "staked_usd": round(sum(float(b["stake_usd"] or 0.0) for b in bets), 2),
            "pnl_usd": round(pnl, 2),
            "settleable_markets": sorted(SETTLEABLE_MARKETS),
            "awaiting_result": sum(1 for b in bets if b["status"] == "open"),
            "last_settlement": self.last_settlement,
        }

    # ------------------------------------------------------------------
    # settlement
    # ------------------------------------------------------------------

    async def settle(self, events: Optional[Sequence[Any]] = None,
                     leagues: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Close every open bet whose fixture has FINISHED with a score.

        `events` may be supplied (the cycle already fetched fixtures); otherwise
        they are fetched for the leagues that have open bets. Nothing settles on a
        missing score: a finished match with no readable score is reported as such
        and left open.
        """
        open_bets = self.open_bets()
        if not open_bets:
            self.last_settlement = {"settled": 0, "open": 0, "reason": "no open bets"}
            return self.last_settlement

        if events is None:
            if self.data is None:
                self.last_settlement = {
                    "settled": 0, "open": len(open_bets),
                    "reason": ("no results feed is attached to the sports book, so "
                               "these bets cannot be settled yet")}
                return self.last_settlement
            leagues = leagues or sorted({b["league"] for b in open_bets if b["league"]})
            try:
                events = await self.data.fetch_events(leagues or ("epl", "nba"))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[sports] results fetch failed: "
                               f"{type(e).__name__}: {e}")
                events = []

        finished = {e.key: e for e in (events or []) if _is_finished(e)}
        settled, waiting, unreadable = [], [], []
        for bet in open_bets:
            event = finished.get(bet["event_key"])
            if event is None:
                waiting.append(bet["bet_id"])
                continue
            if event.home_score is None or event.away_score is None:
                unreadable.append({"bet_id": bet["bet_id"],
                                   "reason": "the fixture is FINISHED but the feed "
                                             "reported no score"})
                continue
            outcome = _settle_bet(bet, event, float(event.home_score),
                                  float(event.away_score))
            if outcome["status"] == "unsettleable":
                # Left OPEN and named. A bet whose result cannot be read is not
                # settled, refunded or guessed - it is reported, so the operator
                # sees the one thing that would need looking at rather than a
                # quietly invented P&L.
                unreadable.append({"bet_id": bet["bet_id"],
                                   "reason": outcome["reason"]})
                continue
            self._resolve(bet, outcome,
                          source=f"score from {event.provider or 'feed'}")
            settled.append({"bet_id": bet["bet_id"], "outcome": bet["outcome"],
                            "stake_usd": bet["stake_usd"],
                            "execution_mode": bet["execution_mode"],
                            "book": bet["book"], "odds": bet["odds"],
                            **outcome})

        if unreadable:
            logger.warning(
                "[sports] " + str(len(unreadable)) + " bet(s) cannot be settled "
                "from the result that came back: "
                + " | ".join(u["reason"] for u in unreadable[:3])
                + " - they stay OPEN and nothing is refunded")
        self.last_settlement = {
            "settled": len(settled), "open": len(waiting) + len(unreadable),
            "awaiting_result": len(waiting), "unreadable": unreadable,
            "bets": settled, "source": "final scores",
            "note": ("a bet settles only on a result a feed reported; nothing is "
                     "inferred from the absence of one"),
        }
        if settled:
            logger.info(f"[sports] settled {len(settled)} bet(s): "
                        + "; ".join(f"{s['outcome']} {s['status']} "
                                    f"${s['pnl']:+.2f}" for s in settled))
        if waiting:
            logger.info(f"[sports] {len(waiting)} bet(s) still open - their "
                        f"fixture has not been reported finished yet")
        return self.last_settlement

    def _resolve(self, bet: Dict[str, Any], outcome: Dict[str, Any],
                 source: str) -> None:
        try:
            self.storage.conn.execute(
                "UPDATE sports_bets SET status=?, pnl_usd=?, settle_reason=?, "
                "settlement_source=?, resolved_at=? WHERE bet_id=?",
                (outcome["status"], outcome["pnl"], outcome["reason"], source,
                 _now(), bet["bet_id"]))
            self.storage.conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not settle sports bet {bet['bet_id']}: "
                         f"{type(e).__name__}: {e}")
            return

        if outcome["status"] == "void":
            # Stake returned: nothing was won or lost, so there is nothing to
            # learn - and recording a 50% outcome would teach the model that a
            # refund was a coin flip.
            logger.info(f"[sports] {bet['bet_id']} voided ({outcome['reason']}); "
                        f"stake returned and EXCLUDED from the learning record")
            return

        if self.tracker is not None:
            try:
                self.tracker.record_resolution(
                    bet["bet_id"], 1.0 if outcome["status"] == "won" else 0.0,
                    float(outcome["pnl"]))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Sports bet {bet['bet_id']} settled but the outcome "
                             f"was not recorded: {type(e).__name__}: {e}")


def _is_finished(event: Any) -> bool:
    status = str(getattr(event, "status", "") or "").upper().replace("STATUS_", "")
    return status.replace("_", "") in ("FINAL", "FULLTIME", "FT", "FINALOT",
                                       "FINALPEN", "COMPLETED", "POST")


class _DictOpp:
    """The attributes `SportsBook._blockers_for_dict` reads, from a dict."""

    def __init__(self, raw: Dict[str, Any]):
        self.blockers = raw.get("blockers") or []
        self.market_type = raw.get("market_type")
        self.stake = raw.get("stake")
        self.price = raw.get("price")
        self.event_key = raw.get("event_key")


def _void(bet: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {"status": "void", "pnl": 0.0, "reason": reason}


def _settle_bet(bet: Dict[str, Any], event: Any, home_score: float,
                away_score: float) -> Dict[str, Any]:
    """
    Decide a bet from the final score. Returns status/pnl/reason.

    PTAI reports what happened; it does not decide what "the result" was. A NO
    side on a h2h market, for instance, means the named outcome did NOT win.
    """
    # The market KEY names the market ("totals_2.5"); settlement works in KINDS
    # ("totals") with an explicit line. Using the key directly here settled
    # nothing: every priced market fell through to "no settlement rule".
    market_kind, parsed_line, refusal = classify_market(bet.get("market_type"))
    if refusal:
        return {"status": "unsettleable", "pnl": 0.0, "reason": refusal}
    line = bet.get("line") if bet.get("line") is not None else parsed_line
    outcome = str(bet["outcome"] or "").strip()
    odds = float(bet["odds"] or 0.0)
    stake = float(bet["stake_usd"] or 0.0)
    side = str(bet["side"] or "back").lower()

    won, reason, kind = _did_the_named_outcome_win(
        market_kind, outcome, line, home_score, away_score,
        getattr(event, "home_team", ""), getattr(event, "away_team", ""))
    if kind == "void":
        return _void(bet, reason)
    if kind != "decided":
        # NOT a refund. An outcome we cannot READ is not a bet that returned the
        # stake: refunding it would invent money the venue never gave back, so the
        # bet stays open and is surfaced as needing attention instead.
        return {"status": "unsettleable", "pnl": 0.0, "reason": reason}

    if side == "lay":
        # A lay wins when the named outcome does NOT happen, and pays the stake
        # less commission; when it does happen, the liability is lost.
        if not won:
            # The stake is kept; commission on an exchange is taken from it by
            # the venue, and the engine's own de-vig already accounts for the
            # price, so nothing is deducted twice here.
            return {"status": "won", "pnl": round(stake, 2),
                    "reason": f"lay on {outcome}: it did not win ({reason})"}
        pnl = -stake * (odds - 1.0)
        return {"status": "lost", "pnl": round(pnl, 2),
                "reason": f"lay on {outcome}: it won ({reason})"}

    if won:
        return {"status": "won", "pnl": round(stake * (odds - 1.0), 2),
                "reason": f"back on {outcome}: won ({reason})"}
    return {"status": "lost", "pnl": round(-stake, 2),
            "reason": f"back on {outcome}: lost ({reason})"}


def _did_the_named_outcome_win(market_type: str, outcome: str,
                               line: Optional[float], home_score: float,
                               away_score: float, home_team: str = "",
                               away_team: str = "") -> Tuple[Optional[bool], str, str]:
    """
    Why the named outcome did or did not happen.

    Returns (True won / False lost / None, why, kind) where kind is
    "decided" (settle it), "void" (the stake comes back - a line landing exactly
    on the number) or "unsettleable" (we could not read the result for this bet,
    so it must STAY OPEN rather than be refunded).
    """
    text = outcome.strip().lower()
    if market_type == "h2h":
        home_won = home_score > away_score
        away_won = away_score > home_score
        draw = home_score == away_score
        if text in ("draw", "tie", "x"):
            return draw, f"{home_score}-{away_score}", "decided"
        side = _which_side(text, home_team, away_team)
        if side == "home":
            return home_won, f"{home_score}-{away_score}", "decided"
        if side == "away":
            return away_won, f"{home_score}-{away_score}", "decided"
        return None, (f"the fixture finished {home_score}-{away_score} but this "
                      f"bet's outcome {outcome!r} could not be matched to either "
                      f"side ({home_team or 'home'} / {away_team or 'away'})"), \
            "unsettleable"

    if market_type == "totals":
        total = home_score + away_score
        if line is None:
            return None, "a totals bet with no line cannot be settled", "unsettleable"
        if abs(total - float(line)) < 1e-9:
            return None, f"the total landed exactly on {line}: stake returned", "void"
        over = "over" in text
        if "over" not in text and "under" not in text:
            return None, (f"totals outcome {outcome!r} is neither Over nor Under, so "
                          f"this bet cannot be read"), "unsettleable"
        return ((total > float(line) if over else total < float(line)),
                f"total {total:g} against a line of {line:g}", "decided")

    if market_type == "spreads":
        if line is None:
            return None, "a spread bet with no handicap cannot be settled", \
                "unsettleable"
        # Convention: the handicap is applied to the side NAMED in `outcome`.
        if "away" in text:
            adjusted = away_score + float(line) - home_score
            margin = f"{home_score}-{away_score}, away {float(line):+g}"
        else:
            adjusted = home_score + float(line) - away_score
            margin = f"{home_score}-{away_score}, home {float(line):+g}"
        if abs(adjusted) < 1e-9:
            return None, f"the handicap landed exactly on the number ({margin})", \
                "void"
        return adjusted > 0, margin, "decided"

    return None, f"no settlement rule for market type {market_type!r}", \
        "unsettleable"


def _which_side(text: str, home_team: str, away_team: str) -> Optional[str]:
    """
    Is this book label the home team, the away team, or neither?

    Compared against the FIXTURE's own team names, so "Arsenal" settles a bet
    labelled "Arsenal" or "Arsenal FC" - and a label that matches neither is
    reported as unmatchable rather than settled against whichever side happens to
    look closest.
    """
    if "home" in text:
        return "home"
    if "away" in text:
        return "away"

    def norm(value: str) -> str:
        cleaned = (value or "").strip().lower()
        for suffix in (" fc", " afc", " cf", " sc", " united"):
            cleaned = cleaned.replace(suffix, "")
        return cleaned.strip()

    label = norm(text)
    if not label:
        return None
    home, away = norm(home_team), norm(away_team)
    if home and (label == home or label in home or home in label):
        return "home"
    if away and (label == away or label in away or away in label):
        return "away"
    return None
