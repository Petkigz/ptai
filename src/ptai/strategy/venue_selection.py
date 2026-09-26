"""
Venue selection - which venue is the agent using, and why.

The question this answers is the one an operator asks first: "the agent looks
for the most profitable venue - so which one is it on, and do I have to fund
them all?"

Two facts decide the design:

  1. MONEY CANNOT MOVE BETWEEN VENUES BY ITSELF. Polymarket holds USDC in a
     Polymarket account; Kalshi holds USD at Kalshi. Moving between them is a
     withdrawal and a deposit, which is the operator's action, takes minutes to
     days, and costs fees. So the agent cannot rebalance a portfolio of venues,
     and any design that assumes it can is fiction.
  2. THEREFORE: ONE VENUE HOLDS THE LIVE CAPITAL. The others are scanned and
     paper-traded for free. The agent's job here is not to spread money - it is
     to put the money where the evidence is best, and to say plainly when the
     evidence is not good enough to justify moving it.

That second point is why this module produces a RECOMMENDATION and a MIGRATION
PLAN rather than an allocation. It does not move anything; it tells the operator
what it would do and what moving would cost.

What a venue is ranked on, in order:

  * net P&L after costs, per resolved trade, with the sample size shown. Not
    forecast skill: a venue can be beautifully calibrated and still lose money
    after fees, and capital follows money.
  * a sample large enough to mean anything. Below MIN_SAMPLE_FOR_EVIDENCE the
    verdict is "not enough evidence", never a score that looks like a result.
  * whether the operator can actually fund it at all, from where they are.

Nothing here can promote a venue to live. That is the qualification gate's job,
and it stays the gate: this module chooses WHERE to put money among venues that
have already earned it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

# Below this many resolved trades a venue has no measurable edge, and any
# ranking is noise with a number attached.
MIN_SAMPLE_FOR_EVIDENCE = 30
# The venue qualification gate needs this many BEFORE live capital is permitted.
LIVE_QUALIFICATION_SAMPLE = 100
# The gates capability_engine enforces, quoted here so the UI can show the
# operator exactly how far along a venue is instead of a mystery wall.
QUALIFICATION_GATES = {
    "sample_size": LIVE_QUALIFICATION_SAMPLE,
    "win_rate": 0.55,
    "brier_score": 0.25,
    "profit_factor": 1.1,
}
# One live venue until the budget can fund a second one above its practical
# minimum. Two half-funded accounts cannot both meet a minimum order size, which
# is worse than one that can trade.
MAX_LIVE_VENUES_SMALL_BUDGET = 1
# A switch has to earn back what it costs. If the per-trade edge cannot cover the
# withdrawal and the re-deposit within one qualification horizon, the move is
# churn - the operator paying the rails to chase noise.
MAX_TRADES_TO_RECOUP_MOVE = LIVE_QUALIFICATION_SAMPLE

ROLE_LIVE = "live"          # holds real capital and is allowed to deploy it
ROLE_PAPER = "paper"        # scanned and simulated, no real capital
ROLE_UNAVAILABLE = "unavailable"  # cannot be funded from here


@dataclass
class VenueAssessment:
    """One venue, with the evidence for and against putting money in it."""

    venue_id: str
    label: str = ""
    # Measured performance, from resolved trades only.
    resolved_trades: int = 0
    net_pnl_usd: float = 0.0
    win_rate: float = 0.0
    avg_brier: float = 0.0
    # The same figures, split by what actually happened to the money. Never
    # summed: a paper win does not prove a live venue, and a live loss must not
    # be hidden by paper gains.
    live_resolved_trades: int = 0
    live_net_pnl_usd: float = 0.0
    paper_resolved_trades: int = 0
    paper_net_pnl_usd: float = 0.0
    # Which of the two the ranking above was actually given: "live", "paper", or
    # "none". Named, because a venue ranked on simulated results while the
    # operator reads it as performance is the confusion this split removes.
    pnl_evidence: str = "none"
    # Current state.
    open_positions: int = 0
    paper_positions: int = 0
    qualified: bool = False
    fundable: bool = False
    funded: bool = False
    authorised_usd: float = 0.0
    available_usd: float = 0.0
    # How costly and how achievable it is to start trading here. Used to break
    # ties when no venue has evidence yet - which is the state every new install
    # starts in, and the one where an alphabetical tie-break would silently
    # recommend a venue the operator cannot even open an account at.
    # Whether the balance was actually READ, as opposed to assumed. "Not
    # funded" and "we could not look" are different problems with different
    # fixes, and a single message for both sends the operator to the wrong one.
    balance_is_real: bool = False
    reported_balance_usd: float = 0.0
    smallest_practical_usd: float = 0.0
    minimum_deposit_usd: float = 0.0
    available_from: str = ""
    residency_required: Optional[str] = None
    # Distance to qualification, so the operator can see the remaining work.
    gates_met: Dict[str, bool] = field(default_factory=dict)
    gates_remaining: List[str] = field(default_factory=list)
    score: Optional[float] = None
    role: str = ROLE_PAPER
    evidence_basis: str = "no resolved trades"
    blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def pnl_per_trade(self) -> float:
        return self.net_pnl_usd / self.resolved_trades if self.resolved_trades else 0.0

    @property
    def has_evidence(self) -> bool:
        return self.resolved_trades >= MIN_SAMPLE_FOR_EVIDENCE

    @property
    def can_hold_live_capital(self) -> bool:
        return self.funded and self.authorised_usd > 0

    @property
    def deployable_live(self) -> bool:
        """Everything must line up: money, permission, and a proven venue."""
        return (self.can_hold_live_capital and self.qualified
                and self.available_usd > 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "label": self.label or self.venue_id,
            "role": self.role,
            "qualified": self.qualified,
            "fundable": self.fundable,
            "funded": self.funded,
            "deployable_live": self.deployable_live,
            "resolved_trades": self.resolved_trades,
            "net_pnl_usd": round(self.net_pnl_usd, 2),
            # Both figures, and which one the ranking used, so the console can
            # show a real P&L next to a simulated one without either pretending
            # to be the other.
            "live_resolved_trades": self.live_resolved_trades,
            "live_net_pnl_usd": round(self.live_net_pnl_usd, 2),
            "paper_resolved_trades": self.paper_resolved_trades,
            "paper_net_pnl_usd": round(self.paper_net_pnl_usd, 2),
            "pnl_evidence": self.pnl_evidence,
            "pnl_per_trade": round(self.pnl_per_trade, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_brier": round(self.avg_brier, 4),
            "open_positions": self.open_positions,
            "paper_positions": self.paper_positions,
            "authorised_usd": round(self.authorised_usd, 2),
            "available_usd": round(self.available_usd, 2),
            "score": None if self.score is None else round(self.score, 4),
            "evidence_basis": self.evidence_basis,
            "has_evidence": self.has_evidence,
            "balance_is_real": self.balance_is_real,
            "reported_balance_usd": round(self.reported_balance_usd, 2),
            "smallest_practical_usd": round(self.smallest_practical_usd, 2),
            "minimum_deposit_usd": round(self.minimum_deposit_usd, 2),
            "available_from": self.available_from,
            "residency_required": self.residency_required,
            "gates_met": self.gates_met,
            "gates_remaining": self.gates_remaining,
            "blockers": self.blockers,
            "notes": self.notes,
        }


@dataclass
class VenueSelection:
    """
    The decision: which venue holds the money, and what to do about it.

    `live_venue` is what the agent is ACTUALLY trading with real capital right
    now. `candidate` is where the evidence says the money should go. They differ
    when a migration is worth doing, and the plan says what that costs.
    """

    live_venue: Optional[str] = None
    candidate: Optional[str] = None
    # Says what is missing about LIVE capital, never that paper is a
    # consolation: paper mode is a complete way of working, and the operator
    # chose it. The old wording ("paper only") read as a shortfall on a screen
    # whose mode was already paper.
    verdict: str = "no live capital yet - running in paper"
    reasons: List[str] = field(default_factory=list)
    assessments: List[VenueAssessment] = field(default_factory=list)
    switch_warranted: bool = False
    switch_plan: Dict[str, Any] = field(default_factory=dict)
    autonomy: Dict[str, Any] = field(default_factory=dict)
    total_budget_usd: float = 0.0
    max_live_venues: int = MAX_LIVE_VENUES_SMALL_BUDGET
    warnings: List[str] = field(default_factory=list)
    # Said out loud, because "which venue is best" is a question about what the
    # ranking actually measures, and the honest answer is money already realised -
    # not forecast skill, not a backtest, and not volume.
    ranking_basis: str = (
        "realised net P&L per resolved trade, with the sample size shown; "
        f"no score is given below {MIN_SAMPLE_FOR_EVIDENCE} resolved trades"
    )
    # The venue the agent is ACTUALLY trading, read from storage. It is not
    # recomputed from the ranking, or the money's home would drift on noise.
    remembered_live_venue: Optional[str] = None

    @property
    def live_venues(self) -> List[VenueAssessment]:
        return [a for a in self.assessments if a.role == ROLE_LIVE]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "live_venue": self.live_venue,
            "candidate": self.candidate,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "live_venues": [a.venue_id for a in self.live_venues],
            "assessments": [a.to_dict() for a in self.assessments],
            "switch_warranted": self.switch_warranted,
            "switch_plan": self.switch_plan,
            "autonomy": self.autonomy,
            "total_budget_usd": round(self.total_budget_usd, 2),
            "max_live_venues": self.max_live_venues,
            "warnings": self.warnings,
            "ranking_basis": self.ranking_basis,
            "remembered_live_venue": self.remembered_live_venue,
        }


class VenueSelector:
    """
    Ranks venues on measured results and picks where the money belongs.

    It reads the trades table directly for the money figures and the qualification
    report for the gate, so every number it prints traces to a row. It never
    invents a venue, never scores one without evidence, and never recommends
    moving money it cannot show would be better off elsewhere.
    """

    def __init__(self, storage=None, funding_routes: Optional[Dict[str, Any]] = None):
        self.storage = storage
        self.funding_routes = funding_routes if funding_routes is not None else {}

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------

    # The state key under which the live venue is remembered. One key, one
    # meaning, so a restart does not change which venue the agent thinks it is on.
    LIVE_VENUE_KEY = "venue_selection.live_venue"

    def remembered_live_venue(self) -> Optional[str]:
        """The venue the agent was last trading with, from storage."""
        if self.storage is None:
            return None
        try:
            value = self.storage.get_state(self.LIVE_VENUE_KEY)
            return value or None
        except Exception as e:
            logger.debug(f"Could not read the remembered live venue: {e}")
            return None

    def remember_live_venue(self, venue_id: Optional[str]) -> None:
        """
        Write down which venue holds the live capital. Survives a restart.

        `None` means "no venue holds it" and CLEARS the record. Treating it as a
        no-op - which is what this did - made the one call that says "stop
        trading live" silently keep the old venue live, and the caller had no way
        to tell.
        """
        if self.storage is None:
            return
        try:
            self.storage.set_state(self.LIVE_VENUE_KEY,
                                   str(venue_id) if venue_id else "")
        except Exception as e:
            # Never silent: a choice that cannot be recorded would be re-made
            # from the ranking next cycle, which is the drift this prevents.
            logger.error(f"Could not record the live venue {venue_id}: {e}")

    def known_venues(self) -> List[str]:
        """
        Every venue that has ever traded, according to the trade log.

        Used when nothing is running: the console can still rank the venues that
        have a history instead of showing an empty table, and a venue that has
        traded is exactly the set that could have evidence.
        """
        try:
            rows = self.storage.conn.execute(
                "SELECT DISTINCT venue_id FROM trades "
                "WHERE venue_id IS NOT NULL AND venue_id != ''").fetchall()
            return sorted(r[0] for r in rows)
        except Exception as e:
            logger.debug(f"Could not list known venues: {e}")
            return []

    def venue_evidence(self) -> Dict[str, Dict[str, Any]]:
        """
        The per-venue figures, made public because they answer the operator's
        question as well as this selector's.

        One source: the ranking here and the "best validated venue" line on the
        operator's screen must not be computed twice with different filters.
        """
        return self._per_venue_stats()

    def _per_venue_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Resolved and open counts per venue, from the trades table.

        Paper and live are counted separately: a paper win does not prove a live
        venue, and a live loss must not be hidden by paper gains.
        """
        if self.storage is None:
            return {}
        try:
            rows = self.storage.conn.execute(
                """
                SELECT venue_id,
                       COUNT(*) AS total,
                       SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS resolved,
                       SUM(CASE WHEN resolved = 1 THEN pnl ELSE 0 END) AS net_pnl,
                       SUM(CASE WHEN resolved = 1 AND pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN resolved = 0 AND status = 'paper' THEN 1 ELSE 0 END) AS open_paper,
                       SUM(CASE WHEN resolved = 0 AND status != 'paper' THEN 1 ELSE 0 END) AS open_live,
                       -- The same three figures, split by EXECUTION mode.
                       --
                       -- The docstring of this method already promised that
                       -- "a paper win does not prove a live venue, and a live
                       -- loss must not be hidden by paper gains", while `net_pnl`
                       -- summed both into one number. The selector then ranks
                       -- venues for REAL capital on a mixture of real and
                       -- simulated results. These are the honest figures, and
                       -- the caller decides which one answers its question.
                       SUM(CASE WHEN resolved = 1
                                AND COALESCE(execution_mode,'live') = 'live'
                                THEN pnl ELSE 0 END) AS live_net_pnl,
                       SUM(CASE WHEN resolved = 1
                                AND COALESCE(execution_mode,'live') = 'paper'
                                THEN pnl ELSE 0 END) AS paper_net_pnl,
                       SUM(CASE WHEN resolved = 1
                                AND COALESCE(execution_mode,'live') = 'live'
                                THEN 1 ELSE 0 END) AS live_resolved,
                       SUM(CASE WHEN resolved = 1
                                AND COALESCE(execution_mode,'live') = 'paper'
                                THEN 1 ELSE 0 END) AS paper_resolved,
                       SUM(CASE WHEN resolved = 1 AND pnl > 0
                                AND COALESCE(execution_mode,'live') = 'live'
                                THEN 1 ELSE 0 END) AS live_wins,
                       SUM(CASE WHEN resolved = 1 AND pnl > 0
                                AND COALESCE(execution_mode,'live') = 'paper'
                                THEN 1 ELSE 0 END) AS paper_wins
                FROM trades
                GROUP BY venue_id
                """
            ).fetchall()
        except Exception as e:
            logger.error(f"Could not read per-venue performance: "
                         f"{type(e).__name__}: {e}")
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            resolved = int(row["resolved"] or 0)
            wins = int(row["wins"] or 0)
            live_resolved = int(row["live_resolved"] or 0)
            paper_resolved = int(row["paper_resolved"] or 0)
            live_wins = int(row["live_wins"] or 0)
            paper_wins = int(row["paper_wins"] or 0)
            out[str(row["venue_id"] or "")] = {
                "total": int(row["total"] or 0),
                "resolved": resolved,
                "net_pnl": float(row["net_pnl"] or 0.0),
                "win_rate": (wins / resolved) if resolved else 0.0,
                "open_paper": int(row["open_paper"] or 0),
                "open_live": int(row["open_live"] or 0),
                # Split by mode. `live_*` is what the real account did;
                # `paper_*` is the simulation, which is evidence toward
                # qualification and NOT evidence of what the venue will do with
                # real money.
                "live_resolved": live_resolved,
                "live_net_pnl": float(row["live_net_pnl"] or 0.0),
                "live_win_rate": (live_wins / live_resolved) if live_resolved else 0.0,
                "paper_resolved": paper_resolved,
                "paper_net_pnl": float(row["paper_net_pnl"] or 0.0),
                "paper_win_rate": (paper_wins / paper_resolved) if paper_resolved else 0.0,
            }
        return out

    # ------------------------------------------------------------------
    # the evidence matrix
    # ------------------------------------------------------------------

    MIN_CELL_SAMPLES = 5

    def evidence_matrix(self) -> Dict[str, Dict[str, Any]]:
        """
        What the agent has actually learned, cell by cell.

        "Venue = good/bad" is the crudest possible reading of the outcome log,
        and it was the only one available: every statistic grouped by
        `venue_id`. The truth the log already contains is finer - a venue can be
        good at a five-minute sports market and hopeless at a thirty-day
        political one, and one strategy's results are not another's. The
        seventh report asks allocation to learn
        VENUE x MARKET TYPE x STRATEGY x EXECUTION STYLE; these are the
        dimensions the record can actually support, so these are the cells.

        Read from `trade_outcomes`, which is the only table that carries the
        RESOLVED result together with what was predicted, what it cost and
        whether the fill was priced against a real book.

        `execution_mode` is a dimension rather than a filter: paper and live are
        never averaged into one cell, because a simulated win is not a live one.
        """
        if self.storage is None:
            return {}
        try:
            rows = self.storage.conn.execute(
                """
                SELECT venue_id, strategy, category,
                       COALESCE(execution_mode, 'unclassified') AS mode,
                       COUNT(*) AS resolved,
                       SUM(pnl) AS net_pnl,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN fill_is_real THEN 1 ELSE 0 END) AS real_evidence,
                       AVG(expected_net_ev_pct) AS expected_ev_pct
                FROM trade_outcomes
                WHERE actual_outcome IS NOT NULL
                GROUP BY venue_id, strategy, category, mode
                """
            ).fetchall()
        except Exception as e:
            logger.error(f"Could not read the evidence matrix: "
                         f"{type(e).__name__}: {e}")
            return {}

        cells: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            resolved = int(row["resolved"] or 0)
            wins = int(row["wins"] or 0)
            key = "|".join(str(row[k] or "") for k in
                           ("venue_id", "strategy", "category", "mode"))
            cells[key] = {
                "key": key,
                "venue_id": str(row["venue_id"] or ""),
                "strategy": str(row["strategy"] or ""),
                "category": str(row["category"] or ""),
                "execution_mode": str(row["mode"] or ""),
                "resolved": resolved,
                "net_pnl": float(row["net_pnl"] or 0.0),
                "net_pnl_per_trade": (float(row["net_pnl"] or 0.0) / resolved
                                      if resolved else 0.0),
                "win_rate": (wins / resolved) if resolved else 0.0,
                "real_evidence": int(row["real_evidence"] or 0),
                "real_evidence_coverage": (
                    int(row["real_evidence"] or 0) / resolved if resolved else 0.0),
                "expected_net_ev_pct": (
                    None if row["expected_ev_pct"] is None
                    else float(row["expected_ev_pct"])),
                "enough_evidence": resolved >= self.MIN_CELL_SAMPLES,
            }
        return cells

    def cell_evidence(self, venue_id: str, strategy: str = "",
                      category: str = "",
                      execution_mode: str = "") -> Dict[str, Any]:
        """
        The most specific slice of the record that has ENOUGH evidence.

        THE ONE RULE: RELAX DIMENSIONAL SPECIFICITY, NEVER EVIDENCE PROVENANCE.

        A narrow question - "how has polymarket x value x sports done LIVE?" -
        can legitimately be answered from a wider slice, because the wider slice
        is the same kind of thing counted more broadly. It can never be answered
        from a DIFFERENT KIND of evidence. Simulated trades are not live trades
        with less detail; they are a different thing, and a live-capital decision
        that quietly leans on 9 paper trades because only 3 live ones exist is
        exactly the failure this matrix was built to prevent.

        So the execution mode, once the caller names it, is FIXED at every level
        of the ladder:

            LIVE  ->  LIVE venue x strategy x market type
                  ->  LIVE venue x strategy
                  ->  LIVE venue
                  ->  no live evidence

        The answer always says which level answered it, and how many trades that
        level is standing on. A two-trade cell that happens to hold the best
        number in the record is not an answer, and neither is a venue total
        presented as if it were about one strategy.
        """
        cells = list(self.evidence_matrix().values())
        mode = str(execution_mode or "")
        requested = {"venue_id": venue_id, "strategy": strategy,
                     "category": category}
        order = ["venue_id", "strategy", "category"]
        # Only the dimensions the caller actually asked about: a venue-only
        # question has one level, not three levels that all mean the same thing.
        # venue_id is the floor and is never dropped - a level below it would be
        # answering about OTHER venues.
        asked = [k for k in order if requested[k]]
        ladder = []
        for keep in range(len(asked), 0, -1):
            keys = asked[:keep]
            wanted = {k: requested[k] for k in keys}
            if mode:
                wanted["execution_mode"] = mode
            ladder.append((keys, wanted))

        best_partial = None
        for keys, wanted in ladder:
            matched = [c for c in cells
                       if all(str(c.get(field) or "") == str(value or "")
                              for field, value in wanted.items() if value)]
            if not matched:
                continue
            resolved = sum(c["resolved"] for c in matched)
            if best_partial is None:
                best_partial = (keys, matched, resolved, wanted)
            if resolved >= self.MIN_CELL_SAMPLES:
                summary = self._summarise_cells(keys, matched, wanted)
                if len(keys) < len(asked):
                    dropped = [k for k in asked if k not in keys]
                    summary["reason"] = (
                        f"no slice narrower than this reached "
                        f"{self.MIN_CELL_SAMPLES} resolved trades, so this is "
                        f"the {summary['level']} aggregate ({resolved} trades); "
                        f"it does not distinguish {', '.join(dropped)}"
                        + (f" (the execution mode is held fixed to {mode})"
                           if mode else ""))
                return summary

        if best_partial is not None:
            keys, matched, resolved, wanted = best_partial
            summary = self._summarise_cells(keys, matched, wanted)
            summary.update({
                "level": "insufficient",
                "reason": (f"no slice of the {mode or 'record'} evidence has "
                           f"{self.MIN_CELL_SAMPLES} resolved trades; the widest "
                           f"match has {resolved} across {len(matched)} cell(s)"
                           + (f" in {mode} mode" if mode else "")),
            })
            return summary

        # Nothing matched at all. Say WHY, because "no evidence" and "no evidence
        # in this mode" call for different responses from the operator - the
        # second one is a paper record that has not earned live capital yet.
        venue_cells = [c for c in cells if c["venue_id"] == venue_id]
        other_modes = sorted({c["execution_mode"] for c in venue_cells}) if venue_cells else []
        return {
            "level": "none", "venue_id": venue_id, "strategy": strategy,
            "category": category, "execution_mode": mode,
            "resolved": 0, "net_pnl": 0.0, "net_pnl_per_trade": 0.0,
            "win_rate": 0.0, "real_evidence": 0,
            "real_evidence_coverage": 0.0, "cells": 0,
            "has_evidence": False, "enough_evidence": False,
            "reason": (
                (f"nothing resolved for this venue in {mode} mode; its record is "
                 f"in {', '.join(other_modes)} - simulated evidence does not "
                 f"answer for {'live' if mode == 'live' else mode} capital")
                if mode and other_modes else
                "nothing resolved for this venue"),
        }

    @staticmethod
    def _summarise_cells(keys: List[str], matched: List[Dict[str, Any]],
                         wanted: Dict[str, str]) -> Dict[str, Any]:
        """
        One answer out of the cells that matched a level of the ladder.

        `keys` are the dimensions this level distinguishes; the execution mode is
        reported separately because it is never relaxed - it is the same at every
        level or the level would not have matched.
        """
        resolved = sum(c["resolved"] for c in matched)
        net_pnl = sum(c["net_pnl"] for c in matched)
        wins = sum(c["win_rate"] * c["resolved"] for c in matched)
        real = sum(c["real_evidence"] for c in matched)
        label = " x ".join("venue" if k == "venue_id" else k for k in keys)
        if wanted.get("execution_mode"):
            label = f"{label} (execution_mode={wanted['execution_mode']})"
        return {
            "level": label,
            "venue_id": wanted.get("venue_id") or matched[0]["venue_id"],
            "strategy": wanted.get("strategy", ""),
            "category": wanted.get("category", ""),
            "execution_mode": wanted.get("execution_mode", ""),
            "resolved": resolved,
            "net_pnl": net_pnl,
            "net_pnl_per_trade": (net_pnl / resolved) if resolved else 0.0,
            "win_rate": (wins / resolved) if resolved else 0.0,
            "real_evidence": real,
            "real_evidence_coverage": (real / resolved) if resolved else 0.0,
            "cells": len(matched),
            "has_evidence": resolved > 0,
            "enough_evidence": resolved >= VenueSelector.MIN_CELL_SAMPLES,
        }


    def best_cells(self, minimum: int = None) -> List[Dict[str, Any]]:
        """
        The cells worth allocating on, best first.

        Ranked on net P&L per trade rather than total: a cell with three trades
        that made $3 each is not a better place for capital than a cell with
        forty that made $0.50 each, and the total says it is.
        """
        bar = self.MIN_CELL_SAMPLES if minimum is None else int(minimum)
        cells = [c for c in self.evidence_matrix().values()
                 if c["resolved"] >= bar]
        return sorted(cells, key=lambda c: c["net_pnl_per_trade"], reverse=True)

    def _brier_by_venue(self, tracker) -> Dict[str, float]:
        """Calibration per venue, when a tracker is available."""
        if tracker is None:
            return {}
        try:
            performance = tracker.get_venue_performance() or {}
        except Exception as e:
            logger.warning(f"Could not read venue calibration: {e}")
            return {}
        return {venue: float(stats.get("avg_brier", 0.5))
                for venue, stats in performance.items()}

    # ------------------------------------------------------------------
    # assessment
    # ------------------------------------------------------------------

    def assess(self, venue_ids: List[str],
               accounts: Optional[List[Dict[str, Any]]] = None,
               qualified_ids: Optional[List[str]] = None,
               tracker=None,
               labels: Optional[Dict[str, str]] = None) -> List[VenueAssessment]:
        """
        One assessment per venue.

        A venue with no recorded trades still gets an assessment - it is simply
        marked "no resolved trades" and cannot be chosen. Dropping it would make
        an untested venue look the same as a venue that is not registered at all.
        """
        stats = self._per_venue_stats()
        brier = self._brier_by_venue(tracker)
        accounts_by_venue = {str(a.get("venue_id")): a for a in (accounts or [])}
        qualified = set(qualified_ids or [])
        labels = labels or {}

        assessments: List[VenueAssessment] = []
        for venue_id in sorted(set(venue_ids) | set(accounts_by_venue)):
            account = accounts_by_venue.get(venue_id) or {}
            row = stats.get(venue_id, {})
            live_resolved = int(row.get("live_resolved", 0) or 0)
            paper_resolved = int(row.get("paper_resolved", 0) or 0)
            # The ranking is given LIVE results when there are any, and the
            # PAPER results otherwise - never their sum. The first venue a fresh
            # install chooses has only paper evidence to go on, and that is
            # legitimate for QUALIFICATION; what would not be legitimate is
            # presenting it as what the venue did with real money.
            if live_resolved:
                ranked_pnl = float(row.get("live_net_pnl", 0.0))
                ranked_win_rate = float(row.get("live_win_rate", 0.0))
                evidence = "live"
            elif paper_resolved:
                ranked_pnl = float(row.get("paper_net_pnl", 0.0))
                ranked_win_rate = float(row.get("paper_win_rate", 0.0))
                evidence = "paper"
            else:
                ranked_pnl = 0.0
                ranked_win_rate = 0.0
                evidence = "none"

            assessed = VenueAssessment(
                venue_id=venue_id,
                label=labels.get(venue_id, account.get("venue_label") or venue_id),
                resolved_trades=int(row.get("resolved", 0)),
                net_pnl_usd=ranked_pnl,
                win_rate=ranked_win_rate,
                live_resolved_trades=live_resolved,
                live_net_pnl_usd=float(row.get("live_net_pnl", 0.0)),
                paper_resolved_trades=paper_resolved,
                paper_net_pnl_usd=float(row.get("paper_net_pnl", 0.0)),
                pnl_evidence=evidence,
                avg_brier=brier.get(venue_id, 0.0),
                open_positions=int(account.get("live_position_count",
                                              row.get("open_live", 0)) or 0),
                paper_positions=int(row.get("open_paper", 0)),
                qualified=venue_id in qualified,
                fundable=venue_id in self.funding_routes,
                funded=bool(account.get("funded")),
                balance_is_real=bool(account.get("balance_is_real")),
                # `deposited_usd` in the capital ledger is the AUTHORISED BUDGET,
                # not the venue's own balance. Labelling it "reported balance"
                # showed the operator a number the venue never reported - and the
                # venue's actual figure was sitting in `reported_balance_usd` all
                # along, one field away.
                reported_balance_usd=float(account.get("reported_balance_usd") or 0.0),
                authorised_usd=float(account.get("budget_usd", 0.0) or 0.0),
                available_usd=float(account.get("available_usd", 0.0) or 0.0),
            )
            self._fill_gates(assessed, row)
            self._fill_entry_cost(assessed)
            assessments.append(assessed)
        return assessments

    def _fill_entry_cost(self, a: VenueAssessment) -> None:
        """How expensive and how possible it is to start here."""
        route = self.funding_routes.get(a.venue_id) or {}
        a.smallest_practical_usd = float(route.get("smallest_practical_usd") or 0.0)
        a.minimum_deposit_usd = float(route.get("minimum_deposit_usd") or 0.0)
        a.available_from = str(route.get("available_from") or "")
        a.residency_required = route.get("residency_required")
        if a.residency_required:
            a.notes.append(f"requires {a.residency_required}")

    def _fill_gates(self, a: VenueAssessment, row: Dict[str, Any]) -> None:
        """Record which qualification gates a venue has met, and which remain."""
        gates = {
            "sample_size": a.resolved_trades >= LIVE_QUALIFICATION_SAMPLE,
            "win_rate": a.win_rate >= QUALIFICATION_GATES["win_rate"],
            "brier_score": (a.avg_brier > 0
                            and a.avg_brier <= QUALIFICATION_GATES["brier_score"]),
        }
        # Profit factor is only meaningful with a sample, and a venue with no
        # losing trades would divide by zero - so it is computed from the
        # recorded rows rather than assumed.
        profit_factor = self._profit_factor(a.venue_id)
        gates["profit_factor"] = (profit_factor is not None
                                  and profit_factor >= QUALIFICATION_GATES["profit_factor"])
        a.gates_met = gates
        a.gates_remaining = [g for g, met in gates.items() if not met]

        if a.qualified:
            a.evidence_basis = (f"qualified: {a.resolved_trades} resolved trades, "
                                f"net ${a.net_pnl_usd:+.2f}")
        elif not a.resolved_trades:
            a.evidence_basis = "no resolved trades: nothing to choose on"
        elif not a.has_evidence:
            a.evidence_basis = (f"only {a.resolved_trades} resolved trades - below "
                                f"the {MIN_SAMPLE_FOR_EVIDENCE} needed to rank on")
        else:
            a.evidence_basis = (f"{a.resolved_trades} resolved trades, "
                                f"${a.pnl_per_trade:+.4f}/trade, "
                                f"win rate {a.win_rate:.0%}")

    def _profit_factor(self, venue_id: str) -> Optional[float]:
        if self.storage is None:
            return None
        try:
            row = self.storage.conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) AS gains, "
                "COALESCE(SUM(CASE WHEN pnl < 0 THEN -pnl ELSE 0 END), 0) AS losses "
                "FROM trades WHERE resolved = 1 AND venue_id = ?", (venue_id,)).fetchone()
        except Exception as e:
            logger.warning(f"profit factor for {venue_id} failed: {e}")
            return None
        losses = float(row["losses"] or 0.0)
        gains = float(row["gains"] or 0.0)
        # No losses yet is not an infinite profit factor; it is an unmeasured one.
        if losses <= 0:
            return None
        return gains / losses

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def select(self, assessments: List[VenueAssessment],
               total_budget_usd: float = 0.0,
               smallest_practical_usd: float = 10.0) -> VenueSelection:
        """
        Choose the live venue, and say whether switching is worth it.

        The rule that shapes everything: a venue becomes live only when it is
        FUNDED, AUTHORISED and QUALIFIED. Until then the agent paper trades it -
        which costs nothing and is how it earns the qualification in the first
        place. So "the most profitable venue" is, on a new install, a venue with
        no evidence yet, and the honest answer is to fund one and let it prove
        itself there.
        """
        selection = VenueSelection(assessments=assessments,
                                   total_budget_usd=float(total_budget_usd or 0.0))

        for a in assessments:
            self._assign_role(a)

        live = [a for a in assessments if a.role == ROLE_LIVE]

        # Which venue is live is REMEMBERED, not recomputed from the ranking.
        #
        # Recomputing it every cycle would let a marginally better per-trade
        # number move the home of the real money without anybody deciding to.
        # The ranking says where money SHOULD go; this says where it IS.
        remembered = self.remembered_live_venue()

        # The incumbent is ordered first, BEFORE the cap is applied. Apply the cap
        # first and it can demote the venue the money is already in, which turns
        # "the cap keeps one live venue" into "the cap relocates the money".
        live.sort(
            key=lambda a: (a.venue_id == remembered, a.qualified,
                           a.has_evidence, a.pnl_per_trade),
            reverse=True,
        )

        # Two live venues on a small budget is two accounts that cannot meet a
        # minimum order. The cap is explicit rather than implied.
        if len(live) > MAX_LIVE_VENUES_SMALL_BUDGET:
            keep = live[:MAX_LIVE_VENUES_SMALL_BUDGET]
            for a in live:
                if a not in keep:
                    a.role = ROLE_PAPER
                    a.notes.append(
                        "a second live venue is not funded on this budget: two "
                        "accounts that each cannot meet a minimum order beats "
                        "nothing, but one that can beats both")
            live = keep

        live_ids = [a.venue_id for a in live]
        if remembered and remembered in live_ids:
            pass  # already first: the incumbent keeps the capital
        elif remembered and remembered not in live_ids:
            # The venue the money is at is no longer ready. Do not quietly move
            # to another one: the money is still there, and moving it is the
            # operator's action.
            stale = next((a for a in assessments
                          if a.venue_id == remembered), None)
            why = ", ".join(stale.blockers) if stale and stale.blockers \
                else "it is no longer funded, authorised and qualified"
            selection.warnings.append(
                f"{remembered} was the live venue and is no longer ready: {why}. "
                f"Nothing is trading live until it is ready again or the capital "
                f"is moved - and moving it is your action, not the agent's.")
            live = []
        elif live and not remembered:
            # First time a venue becomes ready. Adopt it and write the choice
            # down, so every later cycle reads the same answer.
            self.remember_live_venue(live[0].venue_id)
            remembered = live[0].venue_id

        selection.live_venue = live[0].venue_id if live else None
        selection.remembered_live_venue = remembered
        selection.max_live_venues = MAX_LIVE_VENUES_SMALL_BUDGET

        # The candidate: the best venue that could HOLD money.
        #
        # Ranked on money first, and then - only when money cannot separate them,
        # which is the state of every new install - on how cheaply and how
        # feasibly the operator can actually start. Without that second key the
        # choice falls to list order, and a fresh install would recommend
        # whichever venue happens to come first in an enumeration.
        fundable = [a for a in assessments if a.fundable]
        ranked = sorted(
            fundable,
            key=lambda a: (
                a.qualified,
                a.has_evidence,
                a.pnl_per_trade,
                a.net_pnl_usd,
                # Lower entry cost wins a tie.
                -a.smallest_practical_usd,
                -a.minimum_deposit_usd,
                # A venue the operator may not be able to open at all loses.
                a.residency_required is None,
                a.venue_id,
            ),
            reverse=True,
        )
        selection.candidate = ranked[0].venue_id if ranked else None

        self._build_verdict(selection, live, ranked)
        self._build_switch_plan(selection, live, ranked,
                                smallest_practical_usd=smallest_practical_usd)
        selection.autonomy = autonomy_contract()
        return selection

    def _assign_role(self, a: VenueAssessment) -> None:
        if not a.fundable:
            a.role = ROLE_UNAVAILABLE
            a.blockers.append(
                "no way to fund this venue from here: the agent cannot place "
                "real orders on an account it cannot hold capital in")
        elif a.deployable_live:
            a.role = ROLE_LIVE
        else:
            a.role = ROLE_PAPER
            # Both can be true, and each has its own fix, so say both rather
            # than the vaguest one that covers them.
            if a.authorised_usd <= 0:
                a.blockers.append("no budget authorised for this venue")
            if not a.funded:
                a.blockers.append(
                    "the account balance could not be read, so the agent cannot "
                    "confirm there is money to trade with"
                    if not a.balance_is_real else
                    f"the account shows ${a.reported_balance_usd:.2f}, so there "
                    f"is nothing here to trade with yet")
            if not a.qualified:
                remaining = ", ".join(a.gates_remaining) or "qualification"
                a.blockers.append(f"not qualified yet ({remaining})")

    def _build_verdict(self, selection: VenueSelection, live: List[VenueAssessment],
                       ranked: List[VenueAssessment]) -> None:
        r = selection.reasons
        if live:
            venue = live[0]
            selection.verdict = f"live on {venue.label}"
            r.append(
                f"{venue.label} holds the live capital: funded, authorised "
                f"${venue.authorised_usd:.2f}, qualified, and showing "
                f"${venue.pnl_per_trade:+.4f} per resolved trade over "
                f"{venue.resolved_trades} trades.")
            others = [a for a in selection.assessments if a.venue_id != venue.venue_id]
            for a in others:
                if a.role == ROLE_PAPER:
                    r.append(f"{a.label} is scanned and paper-traded only"
                             + (f" ({a.blockers[0]})" if a.blockers else "") + ".")
            return

        selection.verdict = ("no venue holds live capital yet - the agent runs "
                             "in paper, on the same live data, until one does")
        if not ranked:
            r.append("No venue can be funded from here, so there is nothing to "
                     "trade live and nothing to choose between.")
            return
        best = ranked[0]
        if not best.has_evidence:
            r.append(
                f"{best.label} is the venue to fund first, but the choice is "
                f"PROVISIONAL: it has {best.evidence_basis}. Paper trading it "
                f"costs nothing and is what earns the qualification.")
            # Say plainly what the choice was actually made on, so the operator
            # knows this is an entry-cost decision and not a performance one.
            basis = []
            if best.smallest_practical_usd:
                basis.append(f"smallest practical deposit "
                             f"${best.smallest_practical_usd:.0f}")
            if best.minimum_deposit_usd:
                basis.append(f"deposit minimum ${best.minimum_deposit_usd:.0f}")
            if best.residency_required:
                basis.append(f"requires {best.residency_required}")
            elif best.available_from:
                basis.append(f"available {best.available_from}")
            if basis:
                r.append(
                    "The choice is on entry cost and access, not on performance - "
                    "nothing has a track record yet: " + ", ".join(basis) + ".")
            losers = [a for a in ranked[1:] if a.residency_required]
            for a in losers:
                r.append(
                    f"{a.label} cannot be funded by this operator without "
                    f"{a.residency_required}, so it is scanned and paper-traded "
                    f"only.")
            r.append(
                f"Live capital needs {LIVE_QUALIFICATION_SAMPLE} resolved trades "
                f"with win rate >= {QUALIFICATION_GATES['win_rate']:.0%}, Brier "
                f"<= {QUALIFICATION_GATES['brier_score']}, profit factor >= "
                f"{QUALIFICATION_GATES['profit_factor']}. Until then, live is "
                f"refused - by design, not by accident.")
        else:
            r.append(
                f"{best.label} has the best evidence ({best.evidence_basis}) but "
                f"is not ready for live: "
                + ("; ".join(best.blockers) or "qualification"))
        r.append(
            "You fund ONE account. The money cannot be moved between venues by "
            "the agent - a withdrawal and a deposit are yours to make, which is "
            "why the agent recommends rather than reallocates.")

    def _build_switch_plan(self, selection: VenueSelection,
                           live: List[VenueAssessment],
                           ranked: List[VenueAssessment],
                           smallest_practical_usd: float) -> None:
        """
        Should the money move, and what does moving cost?

        Moving capital between venues is not free and not instant, so it has to
        clear a bar: the candidate must have real evidence, beat the incumbent by
        a margin that covers the move, and the budget must be large enough that a
        second account could still meet a minimum order.
        """
        if not ranked:
            return
        candidate = ranked[0]
        incumbent = live[0] if live else None

        plan: Dict[str, Any] = {
            "from": incumbent.venue_id if incumbent else None,
            "to": candidate.venue_id,
            "candidate_label": candidate.label,
            "warranted": False,
            "steps": [],
            "min_budget_to_split_usd": round(smallest_practical_usd * 2, 2),
        }

        if incumbent and incumbent.venue_id == candidate.venue_id:
            plan["warranted"] = False
            plan["reason"] = "the funded venue is already the best-evidenced one"
            selection.switch_plan = plan
            return

        if not candidate.has_evidence:
            plan["warranted"] = False
            plan["reason"] = (
                f"{candidate.label} cannot be compared yet: {candidate.evidence_basis}. "
                f"Switching on no evidence would just be moving the money to a "
                f"venue with less of a track record.")
            selection.switch_plan = plan
            return

        if incumbent is None:
            # Nothing is live yet, so "switching" is simply funding the candidate.
            plan["warranted"] = True
            plan["reason"] = (
                f"No venue is live yet. Fund {candidate.label}; it is the best "
                f"evidence available ({candidate.evidence_basis}).")
            plan["steps"] = [
                f"Deposit ${selection.total_budget_usd:.2f} to {candidate.label}.",
                "Authorise the budget in the console (Capital tab).",
                f"Run paper cycles until {LIVE_QUALIFICATION_SAMPLE} trades "
                f"resolve and the qualification gates pass.",
                "Switch to live. From then on the agent trades without asking.",
            ]
            selection.switch_warranted = True
            selection.switch_plan = plan
            return

        delta = candidate.pnl_per_trade - incumbent.pnl_per_trade
        plan["delta_per_trade_usd"] = round(delta, 4)
        if delta <= 0:
            plan["warranted"] = False
            plan["reason"] = (
                f"{candidate.label} does not beat {incumbent.label} per trade "
                f"(${candidate.pnl_per_trade:+.4f} vs "
                f"${incumbent.pnl_per_trade:+.4f}). Stay where the money is.")
            selection.switch_plan = plan
            return

        # A move costs a withdrawal, a deposit, and the time out of the market.
        # Requiring the edge to recover that is the difference between a decision
        # and churn.
        if selection.total_budget_usd <= 0:
            # A move cost is a percentage of an amount, so with no amount there
            # is no cost - and "free" would make every move look worth doing.
            plan["warranted"] = False
            plan["reason"] = (
                f"No authorised budget is recorded, so the cost of moving the "
                f"money cannot be worked out. {candidate.label} beats "
                f"{incumbent.label} by ${delta:+.4f} per trade, which is worth "
                f"acting on once the amount is known.")
            selection.switch_warranted = False
            selection.switch_plan = plan
            return

        move_cost_usd = self._estimated_move_cost(incumbent.venue_id,
                                                  candidate.venue_id,
                                                  selection.total_budget_usd)
        # The cost is computed, so it must actually be used. Computing it and
        # then switching anyway on any positive delta is how an agent pays $3 to
        # chase a cent.
        trades_to_recoup = move_cost_usd / delta
        plan["estimated_move_cost_usd"] = round(move_cost_usd, 2)
        plan["trades_to_recoup"] = int(trades_to_recoup) + 1
        if trades_to_recoup > MAX_TRADES_TO_RECOUP_MOVE:
            plan["warranted"] = False
            plan["reason"] = (
                f"{candidate.label} beats {incumbent.label} by only "
                f"${delta:+.4f} per trade, and moving costs about "
                f"${move_cost_usd:.2f} - that is "
                f"{plan['trades_to_recoup']} trades to earn the move back. "
                f"Not worth it: stay where the money is and keep measuring.")
            selection.switch_warranted = False
            selection.switch_plan = plan
            return

        plan["warranted"] = True
        plan["reason"] = (
            f"{candidate.label} beats {incumbent.label} by "
            f"${delta:+.4f} per trade on {candidate.resolved_trades} resolved "
            f"trades. Moving costs about ${move_cost_usd:.2f} and some days out "
            f"of the market, earned back in about "
            f"{plan['trades_to_recoup']} trades - worth doing once, not every "
            f"cycle.")
        plan["steps"] = [
            f"Withdraw ${selection.total_budget_usd:.2f} from {incumbent.label} "
            f"(redeem any open positions first - the settlement engine reports "
            f"what is still claimable).",
            f"Deposit it to {candidate.label}.",
            "Authorise the budget in the console, then set the previous venue's "
            "budget to 0 so only one account holds live capital.",
        ]
        selection.switch_warranted = True
        selection.switch_plan = plan

    @staticmethod
    def _estimated_move_cost(from_venue: str, to_venue: str,
                             amount_usd: float) -> float:
        """
        A floor, not a quote: withdrawal costs plus a card on-ramp if that is the
        only way in. The operator sees the number so a marginal edge does not
        trigger a migration that costs more than it earns.
        """
        withdrawal = 0.02 * amount_usd          # bridge/withdrawal friction
        onramp = 0.04 * amount_usd              # worst case: a card on-ramp
        return withdrawal + onramp


# ----------------------------------------------------------------------
# autonomy
# ----------------------------------------------------------------------

def autonomy_contract() -> Dict[str, Any]:
    """
    What the agent does by itself, and the four things only the operator can do.

    This exists because the question "does it trade automatically or does it
    wait for my approval" should have an answer in the product, not in a
    conversation. The answer is: it trades automatically. There is no per-trade
    approval step anywhere in the execution path, and this is the list of exactly
    what it will do without being asked.

    The operator's part is not approving trades. It is the things the agent
    genuinely cannot do for itself: hold the money, log in, and set the limits.
    """
    return {
        "trades_without_approval": True,
        "per_trade_approval": False,
        "cycle_minutes": 10,
        "agent_does_autonomously": [
            "Scans every registered venue for markets and prices them",
            "Builds a fair value, measures the edge, and rejects anything under "
            "the threshold",
            "Sizes the position with Kelly, capped at 6% of capital",
            "Runs the risk chain: exposure, correlation, limits, execution guard",
            "Places the order with the venue - no approval step",
            "Cancels and re-prices resting orders, and reconciles fills",
            "Settles resolved markets and redeems settled winnings",
            "Records the outcome and moves it into the next decision",
            "Repeats every 10 minutes, indefinitely",
            "Refuses to trade when no edge exists, and says so",
        ],
        "operator_only_actions": [
            {"action": "Deposit money at the venue",
             "why": "Money goes to your venue account, not to the agent. It has "
                    "no account of its own.",
             "how_often": "Once, or when you add capital"},
            {"action": "Log in / provide the API key or wallet signer",
             "why": "The agent has no identity of its own at a venue.",
             "how_often": "Once"},
            {"action": "Set the mode and the budget per venue",
             "why": "This is the risk limit. It is the only place the agent's "
                    "permission to spend is defined.",
             "how_often": "Rarely"},
            {"action": "The kill switch",
             "why": "Stops everything immediately. It can also fire by itself on "
                    "drawdown.",
             "how_often": "Only if you want to stop"},
        ],
        "what_it_never_needs": [
            "Approval for an individual trade",
            "Your seed phrase",
            "Access to a wallet holding funds other than the trading budget",
        ],
        "honest_limit": (
            "The agent is autonomous in trading, not in funding. It cannot move "
            "money between venues, cannot deposit for you, and cannot exceed the "
            "budget you set - if it could, the budget would not be a limit."
        ),
    }
