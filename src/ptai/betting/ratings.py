"""
Team ratings PTAI builds itself, from results it can actually read.

The sports models take `strengths` / `ratings` dictionaries. Nothing produced
them, and `run_cycle` called from the trading loop passed neither - so on every
cycle the sports lane had no independent view of a fixture and, correctly, refused
to claim an edge. The list of priced markets was real; the edge was never
PTAI's. That is why the sports side looks thinner than the trading side: on the
trading side the model (LLM + ensemble + book de-vig) runs, here it was silent.

This closes it with the one input a results feed can honestly supply: an Elo-style
rating per team, updated only from FINISHED fixtures with a SCORE, persisted so it
carries across restarts and improves every cycle.

Rules it holds itself to:

  * a rating is EARNED. A team with fewer than `MIN_MATCHES` results is reported
    as UNRATED, and the model is simply not given a number for it - an invented
    starting strength is a forecast with no evidence behind it;
  * only finished matches with both scores update anything. A fixture listed as
    scheduled contributes nothing;
  * the update is standard Elo with a home advantage term, so the number means
    what it says: expected score against an average opponent of the same league;
  * ratings are per league, because a rating is only comparable inside one.

It is a first model, and it says so. It is not claimed to beat the closing price;
the qualification gate is what decides whether it ever gets real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

#: Matches needed before a team gets a rating at all.
MIN_MATCHES = 3
START = 1500.0
K = 20.0
#: Home advantage in Elo points, ~ the standard football figure.
HOME_ADVANTAGE = 65.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS sports_ratings (
    league TEXT NOT NULL,
    team TEXT NOT NULL,
    rating REAL NOT NULL,
    matches INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    goals_for REAL NOT NULL DEFAULT 0,
    goals_against REAL NOT NULL DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (league, team)
);
CREATE TABLE IF NOT EXISTS sports_results_seen (
    event_key TEXT PRIMARY KEY,
    league TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score REAL,
    away_score REAL,
    finished_at TEXT
);
"""


def _expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400.0))


@dataclass
class TeamRating:
    league: str
    team: str
    rating: float
    matches: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0

    @property
    def rated(self) -> bool:
        return self.matches >= MIN_MATCHES

    def to_dict(self) -> Dict[str, Any]:
        return {"league": self.league, "team": self.team,
                "rating": round(self.rating, 1), "matches": self.matches,
                "wins": self.wins, "draws": self.draws, "losses": self.losses,
                "goals_for": self.goals_for, "goals_against": self.goals_against,
                "rated": self.rated}


class RatingsBook:
    """Elo ratings over finished fixtures, persisted."""

    def __init__(self, storage=None):
        self.storage = storage
        self._cache: Dict[str, Dict[str, TeamRating]] = {}
        if self.storage is not None:
            try:
                self.storage.conn.executescript(SCHEMA)
                self.storage.conn.commit()
                self._cache = self._load()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not open the ratings store: "
                             f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Dict[str, TeamRating]]:
        out: Dict[str, Dict[str, TeamRating]] = {}
        try:
            rows = self.storage.conn.execute(
                "SELECT * FROM sports_ratings").fetchall()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read ratings: {e}")
            return out
        for row in rows:
            r = dict(row)
            out.setdefault(r["league"], {})[r["team"]] = TeamRating(
                league=r["league"], team=r["team"], rating=float(r["rating"]),
                matches=int(r["matches"]), wins=int(r["wins"]),
                draws=int(r["draws"]), losses=int(r["losses"]),
                goals_for=float(r["goals_for"]),
                goals_against=float(r["goals_against"]))
        return out

    def _persist(self, rating: TeamRating) -> None:
        if self.storage is None:
            return
        try:
            self.storage.conn.execute(
                """INSERT INTO sports_ratings
                   (league, team, rating, matches, wins, draws, losses, goals_for,
                    goals_against, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(league, team) DO UPDATE SET
                     rating=excluded.rating, matches=excluded.matches,
                     wins=excluded.wins, draws=excluded.draws,
                     losses=excluded.losses, goals_for=excluded.goals_for,
                     goals_against=excluded.goals_against,
                     updated_at=excluded.updated_at""",
                (rating.league, rating.team, rating.rating, rating.matches,
                 rating.wins, rating.draws, rating.losses, rating.goals_for,
                 rating.goals_against, datetime.now(timezone.utc).isoformat()))
            self.storage.conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not persist the rating for {rating.team}: {e}")

    def _seen(self, event_key: str) -> bool:
        if self.storage is None:
            return False
        try:
            row = self.storage.conn.execute(
                "SELECT 1 FROM sports_results_seen WHERE event_key=?",
                (event_key,)).fetchone()
            return row is not None
        except Exception:  # noqa: BLE001
            return False

    def _mark_seen(self, event: Any) -> None:
        if self.storage is None:
            return
        try:
            self.storage.conn.execute(
                "INSERT OR REPLACE INTO sports_results_seen VALUES (?,?,?,?,?,?,?)",
                (event.key, getattr(event, "league", ""),
                 getattr(event, "home_team", ""), getattr(event, "away_team", ""),
                 getattr(event, "home_score", None), getattr(event, "away_score", None),
                 datetime.now(timezone.utc).isoformat()))
            self.storage.conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not mark result seen: {e}")

    # ------------------------------------------------------------------
    def update(self, events: Iterable[Any]) -> Dict[str, Any]:
        """
        Fold finished fixtures into the ratings. Returns what changed.

        A fixture already folded in is skipped by its key, so a cycle that sees
        yesterday's results again does not count them twice - the classic way a
        ratings book quietly drifts.
        """
        updated, skipped, unrated = [], 0, []
        for event in events or []:
            if not _finished_with_scores(event):
                continue
            key = getattr(event, "key", "")
            if not key:
                continue
            if self._seen(key):
                skipped += 1
                continue
            league = str(getattr(event, "league", "") or "")
            home = str(getattr(event, "home_team", "") or "")
            away = str(getattr(event, "away_team", "") or "")
            if not league or not home or not away:
                continue
            hg = float(getattr(event, "home_score"))
            ag = float(getattr(event, "away_score"))
            table = self._cache.setdefault(league, {})
            h = table.setdefault(home, TeamRating(league, home, START, 0))
            a = table.setdefault(away, TeamRating(league, away, START, 0))

            expected_home = _expected(h.rating + HOME_ADVANTAGE, a.rating)
            if hg > ag:
                score_home, result = 1.0, "home"
            elif hg < ag:
                score_home, result = 0.0, "away"
            else:
                score_home, result = 0.5, "draw"

            h.rating += K * (score_home - expected_home)
            a.rating += K * ((1.0 - score_home) - (1.0 - expected_home))
            for team, scored, conceded in ((h, hg, ag), (a, ag, hg)):
                team.matches += 1
                team.goals_for += scored
                team.goals_against += conceded
            if result == "home":
                h.wins += 1
                a.losses += 1
            elif result == "away":
                a.wins += 1
                h.losses += 1
            else:
                h.draws += 1
                a.draws += 1

            self._persist(h)
            self._persist(a)
            self._mark_seen(event)
            updated.append({"league": league, "home": home, "away": away,
                            "score": f"{hg:g}-{ag:g}"})
            for team in (h, a):
                if not team.rated and team.matches == MIN_MATCHES:
                    unrated.append(team.team)

        result = {"updated": len(updated), "already_counted": skipped,
                  "matches": updated[:5]}
        if updated:
            logger.info(f"[sports] ratings updated from {len(updated)} finished "
                        f"fixture(s): " + "; ".join(
                            f"{m['home']} {m['score']} {m['away']}"
                            for m in updated[:3]))
        if skipped:
            logger.info(f"[sports] {skipped} fixture(s) already counted - not "
                        f"counted twice")
        return result

    # ------------------------------------------------------------------
    def ratings_for(self, league: str) -> Dict[str, Dict[str, float]]:
        """
        `{team: {"rating": ..., "matches": ...}}` for the models.

        Unrated teams are INCLUDED but with their true (below-minimum) match
        count, so a caller can decide; `rated_table` is the strict version.
        """
        return {team: {"rating": round(r.rating, 1), "matches": r.matches}
                for team, r in (self._cache.get(league) or {}).items()}

    def rated_table(self, league: str) -> Dict[str, Dict[str, float]]:
        """Only teams with enough results behind them."""
        return {team: {"rating": round(r.rating, 1), "matches": r.matches}
                for team, r in (self._cache.get(league) or {}).items()
                if r.rated}

    def all_ratings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {league: self.rated_table(league) for league in self._cache}

    def engine_inputs(self, events: Iterable[Any]) -> tuple:
        """
        (strengths, ratings) for these fixtures, in the shapes the models take.

        Both are built from the SAME finished results, so the goal model and the
        Elo model cannot disagree about how much is known. A fixture with an
        UNRATED team on either side is left out entirely: handing the model a
        default 1500 would be a forecast with nothing behind it, and the engine's
        own rule is that no independent view means no edge claim.
        """
        strengths: Dict[str, Dict[str, Any]] = {}
        ratings: Dict[str, Dict[str, Any]] = {}
        for event in events or []:
            league = str(getattr(event, "league", "") or "")
            table = self._cache.get(league) or {}
            home = table.get(str(getattr(event, "home_team", "") or ""))
            away = table.get(str(getattr(event, "away_team", "") or ""))
            if not home or not away or not home.rated or not away.rated:
                continue
            key = getattr(event, "key", "")
            if not key:
                continue
            ratings[key] = {
                "home": {"rating": round(home.rating, 1), "games": home.matches},
                "away": {"rating": round(away.rating, 1), "games": away.matches},
            }
            strengths[key] = {
                "home": _strength(home, table), "away": _strength(away, table),
                "players": [],
            }
        return strengths, ratings

    def snapshot(self) -> Dict[str, Any]:
        leagues = {league: len(table) for league, table in self._cache.items()}
        rated = sum(1 for table in self._cache.values()
                    for r in table.values() if r.rated)
        leaders = {}
        for league, table in self._cache.items():
            rated_teams = [r for r in table.values() if r.rated]
            if not rated_teams:
                continue
            best = max(rated_teams, key=lambda r: r.rating)
            leaders[league] = {"team": best.team, "rating": round(best.rating, 1),
                               "matches": best.matches}
        return {
            "leagues": leagues, "teams": sum(leagues.values()), "rated_teams": rated,
            "min_matches": MIN_MATCHES, "leaders": leaders,
            "how": ("Elo from finished fixtures with a score. A team needs "
                    f"{MIN_MATCHES} results before it is given a rating at all, and "
                    "the model is not told a number before then."),
        }


def _finished_with_scores(event: Any) -> bool:
    status = str(getattr(event, "status", "") or "").upper().replace("STATUS_", "")
    if status.replace("_", "") not in ("FINAL", "FULLTIME", "FT", "FINALOT",
                                       "FINALPEN", "COMPLETED", "POST"):
        return False
    return (getattr(event, "home_score", None) is not None
            and getattr(event, "away_score", None) is not None)


def _strength(rating: TeamRating, table: Dict[str, TeamRating]) -> Dict[str, float]:
    """
    Attack/defence ratios against the league's own average, from real results.

    `defence` > 1 means concedes more than average, which is the convention
    `forecast_soccer` documents. Both are 1.0 (league average) for a team with no
    goals recorded, and the games count travels with them so the shrinkage in the
    model has something to work with.
    """
    played = [r for r in table.values() if r.matches > 0]
    if not played:
        return {"attack": 1.0, "defence": 1.0, "games": 0}
    avg_for = sum(r.goals_for for r in played) / sum(r.matches for r in played)
    avg_against = sum(r.goals_against for r in played) / sum(r.matches for r in played)
    avg_for = avg_for or 1.0
    avg_against = avg_against or 1.0
    return {
        "attack": round((rating.goals_for / max(rating.matches, 1)) / avg_for, 3),
        "defence": round((rating.goals_against / max(rating.matches, 1)) / avg_against, 3),
        "games": rating.matches,
    }
