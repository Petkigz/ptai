"""
Betting Engine - orchestrates the sports subsystem end to end.

Pipeline per cycle:

  fetch fixtures (real feeds)
    -> build per-book odds -> de-vig -> consensus + sharp reference
    -> model forecast (Poisson / Elo) blended with the market
    -> value edges, arbs, exchange quotes
    -> gate everything through DataMode + account health + guard
    -> emit opportunities the canonical executor can act on

The gate is not optional decoration. Every bet produced here carries a
`data_mode`, and a synthetic or paper-mode bet cannot reach live capital -
the same rule the trading path enforces. Nothing in this engine will size
money against a price it invented.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from ..markets.base import DataMode, Market
from .arb import (
    BookmakerRiskManager,
    book_vs_exchange_arb,
    dutch_outcome_group,
    hedge_to_green,
    match_events,
    multi_book_arb,
)
from .exchange import (
    BetSide,
    ExchangeAccount,
    ExchangeMarketMaker,
    size_back,
    size_lay,
)
from .models import SportsForecast, SportsModelEngine
from .odds_math import (
    analyse_book,
    closing_line_value,
    devig,
    kelly_fraction,
    is_sharp_process,
)
from .derivative_markets import (
    AccumulatorLeg,
    AccumulatorQuote,
    MatchCardPrices,
    price_accumulator,
    price_anytime_scorer,
    price_first_scorer,
    price_match_card,
    price_player_count,
)
from .high_scoring import (
    LEAGUE_TO_SPORT,
    SPORT_PARAMS,
    resolve_sport,
    HighScoringCard,
    is_high_scoring,
    params_for,
    price_high_scoring_card,
    price_margin_bands,
    price_period_spread,
    price_period_total,
    price_spread_normal,
    price_total_normal,
)
from .market_types import (
    MARKET_CATALOGUE,
    MatchFacts,
    Settlement,
    catalogue_report,
    markets_for_sport,
    payout_multiplier,
    settle_market,
    validate_line,
)
from .sports_data import BookOdds, ConsensusOdds, SportsDataEngine, SportsEvent, sample_book, sample_events


@dataclass
class BetOpportunity:
    """A fully specified, risk-checked bet."""
    opportunity_id: str
    event_key: str
    sport: str
    league: str
    market_type: str            # h2h | spreads | totals
    outcome: str
    side: str                   # back | lay
    price: float
    book: str
    stake: float
    liability: float
    model_prob: float
    fair_prob: float
    edge: float
    edge_pct: float
    conservative_prob: float
    conservative_edge: float
    net_ev_usd: float
    kelly_stake: float
    uncertainty: float
    confidence: float
    data_mode: str
    data_source: str
    commission_pct: float
    is_arb: bool = False
    executable: bool = False
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["created_at"] = self.created_at.isoformat()
        return d


class BettingEngine:
    """
    Owns the sports/betting side of PTAI.

    Explicitly separated from the prediction-market path because the maths
    differs: exchanges have back/lay liability, books have commission on
    winnings, and stake limits are an account-survival problem rather than
    a liquidity one.
    """

    def __init__(self, settings=None, bankroll: float = 50.0,
                 data_engine: Optional[SportsDataEngine] = None,
                 kelly_frac: float = 0.25, max_position_pct: float = 0.06,
                 min_edge_pct: float = 3.0, min_stake: float = 2.0,
                 max_exposure_pct: float = 0.20):
        self.settings = settings
        self.bankroll = bankroll
        self.data = data_engine or SportsDataEngine(settings=settings)
        self.models = SportsModelEngine()
        self.risk = BookmakerRiskManager()
        self.mm = ExchangeMarketMaker(quote_size=max(min_stake, bankroll * 0.05))
        self.kelly_frac = kelly_frac
        self.max_position_pct = max_position_pct
        self.min_edge_pct = min_edge_pct
        self.min_stake = min_stake
        self.max_exposure_pct = max_exposure_pct

        self.exchange_accounts: Dict[str, ExchangeAccount] = {}
        self.clv_history: List[float] = []
        self.last_cycle: Dict[str, Any] = {}
        self.opportunities: List[BetOpportunity] = []
        # Feed-backed market types we compare model vs book on. Everything
        # else in the catalogue is priced from the model and stays
        # non-executable until a feed supplies real odds for it.
        self.feed_market_types: List[str] = ["h2h", "spreads", "totals"]
        self.priced_cards: Dict[str, MatchCardPrices] = {}
        # High-scoring sports (basketball/gridiron/baseball/hockey) are priced
        # with a normal margin model, not the soccer Poisson grid. Poisson
        # ties variance to the mean, which understates an NBA total's spread
        # by ~30% and makes every line look further away than it is.
        self.priced_high_scoring: Dict[str, HighScoringCard] = {}
        self.player_props: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_bankroll(self, bankroll: float) -> None:
        self.bankroll = max(0.0, float(bankroll))
        for acct in self.exchange_accounts.values():
            logger.info(f"[betting] bankroll updated to ${self.bankroll:.2f} "
                        f"(exchange account balances are separate and unchanged)")

    def register_exchange_account(self, venue_id: str, balance: float,
                                 commission_pct: float = 0.05, min_bet: float = 2.0) -> ExchangeAccount:
        acct = ExchangeAccount(venue_id, balance, commission_pct, min_bet)
        self.exchange_accounts[venue_id] = acct
        return acct

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, leagues: Sequence[str] = ("nba", "epl"),
                        data_mode: DataMode = DataMode.LIVE_SHADOW,
                        strengths: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
                        ratings: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
                        account_health_ok: bool = False) -> Dict[str, Any]:
        """
        One betting cycle.

        `data_mode` is supplied by the caller because only the caller knows
        whether this run may touch real money. `account_health_ok` must be
        True - as verified by AccountHealthEngine - before any bet is marked
        executable for live capital.
        """
        strengths = strengths or {}
        ratings = ratings or {}
        t0 = datetime.now(timezone.utc)

        events = await self.data.fetch_events(leagues)
        if not events:
            result = {
                "ok": False, "events": 0, "opportunities": 0, "arbs": 0,
                "data_mode": data_mode.value,
                "blockers": [f"no fixtures from any feed: {self.data.diagnostics[:5]}"],
                "health": self.data.health(),
                "message": "Refusing to proceed - no real fixture data. "
                           "Invented sports prices must never reach the sizing layer.",
            }
            self.last_cycle = result
            logger.warning(f"[betting] cycle aborted: {result['blockers'][0]}")
            return result

        opps: List[BetOpportunity] = []
        arb_found: List[BetOpportunity | Dict] = []
        consensus_count = 0
        model_count = 0
        card_count = 0
        markets_scanned: Dict[str, int] = {}

        for ev in events:
            # Scan EVERY market type the feed carries, not just h2h. Previously
            # spreads and totals were fetched and then filtered out.
            consensus_by_type: Dict[str, ConsensusOdds] = {}
            for mt in self.feed_market_types:
                c = self.data.consensus(ev, mt)
                if c is not None:
                    consensus_by_type[mt] = c
                    markets_scanned[mt] = markets_scanned.get(mt, 0) + 1
                    for label, price in c.best_price.items():
                        self.data.record_line(ev.key, mt, label, price)

            if not consensus_by_type:
                continue
            consensus_count += 1

            h2h = consensus_by_type.get("h2h")
            forecast = self._forecast(ev, h2h, strengths, ratings) if h2h else None
            if forecast is not None:
                model_count += 1

            for mt, consensus in consensus_by_type.items():
                opps.extend(self._build_opportunities(ev, consensus, forecast, data_mode,
                                                      account_health_ok, market_type=mt))
                arb_found.extend(self._scan_arbs(ev, consensus))

            # Price the full card (corners, cards, goals derivatives, props)
            card = self.price_card(ev, strengths)
            if card is not None:
                card_count += 1
                opps.extend(self._build_card_opportunities(
                    ev, card, consensus_by_type, data_mode, account_health_ok))
                # Player props only exist when someone supplies lineups/shares.
                # The engine will not guess who is playing.
                players = (strengths.get(ev.key) or {}).get("players") or []
                if players:
                    self._build_player_prop_opportunities(
                        ev, players, consensus_by_type, data_mode, account_health_ok,
                        prop_lines=(strengths.get(ev.key) or {}).get("prop_lines"),
                        sink=opps)

        self.opportunities = sorted(opps, key=lambda o: o.net_ev_usd, reverse=True)
        result = {
            "ok": True,
            "events": len(events),
            "with_odds": consensus_count,
            "modelled": model_count,
            "cards_priced": card_count,
            "markets_scanned_by_type": markets_scanned,
            "market_types_available": catalogue_report()["total_markets"],
            "opportunities": len(self.opportunities),
            "executable": sum(1 for o in self.opportunities if o.executable),
            "arbs": len(arb_found),
            "arb_examples": arb_found[:5],
            "data_mode": data_mode.value,
            "data_source": "sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
            "bankroll": self.bankroll,
            "top": [o.to_dict() for o in self.opportunities[:5]],
            "health": self.data.health(),
            "sharp_check": self.sharpness(),
            "risk": self.risk.report(),
            "started_at": t0.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self.last_cycle = result
        logger.info(f"[betting] cycle: {len(events)} fixtures, {consensus_count} with odds, "
                    f"{len(self.opportunities)} opportunities "
                    f"({result['executable']} executable), {len(arb_found)} arbs")
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _forecast(self, ev: SportsEvent, consensus: ConsensusOdds,
                  strengths: Dict, ratings: Dict) -> Optional[SportsForecast]:
        key = ev.key
        is_soccer = ev.sport == "soccer" or ev.league in ("epl", "laliga", "seriea",
                                                           "bundesliga", "ucl", "mls")
        label_map = self._label_map(ev, consensus)
        try:
            if is_soccer and key in strengths:
                fc = self.models.forecast_soccer(ev, strengths[key])
            elif key in ratings:
                draw_width = 25.0 if is_soccer else 0.0
                fc = self.models.forecast_elo(ev, ratings[key], draw_width=draw_width)
            else:
                return None
            return self.models.blend_with_market(fc, consensus, label_map)
        except Exception as e:
            logger.warning(f"[betting] model failed for {key}: {e}")
            return None

    @staticmethod
    def _label_map(ev: SportsEvent, consensus: ConsensusOdds) -> Dict[str, str]:
        """Map book outcome labels onto the model's home/draw/away keys."""
        mapping: Dict[str, str] = {}
        home_l = ev.home_team.strip().lower()
        away_l = ev.away_team.strip().lower()
        for label in consensus.best_price.keys():
            l = label.strip().lower()
            if l in (home_l, "home"):
                mapping[label] = "home"
            elif l in (away_l, "away"):
                mapping[label] = "away"
            elif l in ("draw", "tie", "x"):
                mapping[label] = "draw"
            elif home_l and (home_l in l or l in home_l):
                mapping[label] = "home"
            elif away_l and (away_l in l or l in away_l):
                mapping[label] = "away"
        return mapping

    def _build_opportunities(self, ev: SportsEvent, consensus: ConsensusOdds,
                             forecast: Optional[SportsForecast], data_mode: DataMode,
                             account_health_ok: bool,
                             market_type: str = "h2h") -> List[BetOpportunity]:
        out: List[BetOpportunity] = []
        # The model only produces a result distribution; spread/total markets
        # need their own model and are not scored off the h2h forecast.
        if market_type != "h2h":
            return out
        label_map = self._label_map(ev, consensus)
        reverse_map = {v: k for k, v in label_map.items()}

        # Without a model there is no independent view, so no edge claim.
        # Quoting the market's own de-vigged price back at the market is a
        # tautology, not an edge.
        if forecast is None:
            return out

        edges = self.models.value_edges(forecast, consensus, label_map)
        for e in edges:
            label = e["outcome"]
            model_key = label_map.get(label, label)
            model_prob = forecast.probs.get(model_key, 0.0)
            if model_prob <= 0:
                continue

            price = e["price"]
            book = e["book"]
            commission = e["commission_pct"]
            blockers: List[str] = []
            warnings: List[str] = []

            if e["edge_pct"] < self.min_edge_pct:
                blockers.append(f"edge {e['edge_pct']:+.2f}% below {self.min_edge_pct}% threshold")
            if e["conservative_edge"] <= 0:
                blockers.append(
                    f"edge vanishes under uncertainty (conservative prob "
                    f"{e['conservative_prob']:.3f} vs price {price:.3f})")
            if data_mode.is_synthetic:
                blockers.append(f"data_mode {data_mode.value} is synthetic - cannot trade")
            if data_mode.can_deploy_live_capital and not account_health_ok:
                blockers.append("account health not verified - refusing live capital")

            sizing = size_back(self.bankroll, model_prob, price, self.kelly_frac,
                               self.max_position_pct, commission, self.min_stake)
            stake = sizing["stake"]
            if stake <= 0:
                blockers.append(f"sizing: {sizing['reason']}")

            if stake > 0:
                ok, why = self.risk.can_bet(book, stake, self.bankroll, is_arb=False)
                if not ok:
                    blockers.append(f"account risk: {why}")

            kelly_f = kelly_fraction(model_prob, price, 1.0, self.max_position_pct)
            net_ev = stake * e["edge"]

            line_move = self.data.line_movement(ev.key, "h2h", label)
            if line_move is not None:
                if line_move.is_steam:
                    warnings.append(f"steam move: {line_move.steam_reason}")
                    stake = 0.0
                    blockers.append("steam detected - the edge is already gone")
                elif line_move.direction == "down" and e["edge_pct"] > 0:
                    warnings.append(f"line drifting against us ({line_move.move_pct:+.2f}%) - "
                                    f"price may be stale")

            opp = BetOpportunity(
                opportunity_id=f"bet-{ev.event_id}-{label}-{int(datetime.now(timezone.utc).timestamp())}",
                event_key=ev.key, sport=ev.sport, league=ev.league, market_type=market_type,
                outcome=label, side="back", price=price, book=book,
                stake=round(stake, 2), liability=round(stake, 2),
                model_prob=round(model_prob, 4), fair_prob=round(
                    consensus.fair_probs.get(label, model_prob), 4),
                edge=round(e["edge"], 4), edge_pct=round(e["edge_pct"], 3),
                conservative_prob=e["conservative_prob"],
                conservative_edge=e["conservative_edge"],
                net_ev_usd=round(net_ev, 4), kelly_stake=round(self.bankroll * kelly_f, 2),
                uncertainty=forecast.uncertainty, confidence=forecast.confidence,
                data_mode=data_mode.value,
                data_source="sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
                commission_pct=commission, is_arb=False,
                executable=not blockers and stake > 0,
                blockers=blockers, warnings=warnings,
                reasoning=f"{forecast.reasoning} | best price {price:.3f} at {book}, "
                          f"model {model_prob:.3f}, edge {e['edge_pct']:+.2f}%",
            )
            out.append(opp)
        return out

    def _scan_arbs(self, ev: SportsEvent, consensus: ConsensusOdds) -> List[Dict]:
        found: List[Dict] = []

        # 1. multi-book, sized against whatever stake limit each book allows
        limits = {b.book: self.risk.profile(b.book).max_stake_usd for b in consensus.books}
        multi = multi_book_arb(ev, consensus, self.bankroll, self.max_exposure_pct, limits)
        if multi and multi.executable:
            found.append({"type": "multi_book", "event": ev.key, "edge_pct": multi.edge_pct,
                          "profit": multi.guaranteed_profit, "stakes": multi.stakes,
                          "limiting_leg": multi.limiting_leg})

        # 2. book vs exchange
        for b in consensus.books:
            if not (b.is_exchange or "betfair" in b.book.lower() or "exchange" in b.book.lower()):
                continue
            for r in book_vs_exchange_arb(ev, consensus, b, self.bankroll,
                                          self.max_exposure_pct):
                if r.executable:
                    ok_back, why_back = self.risk.can_bet(r.back_book, r.back_stake, self.bankroll)
                    ok_lay, why_lay = self.risk.can_bet(r.exchange_book, r.lay_stake, self.bankroll)
                    if not (ok_back and ok_lay):
                        r.warnings.append(f"account risk: {why_back} {why_lay}")
                        continue
                    found.append({"type": "book_vs_exchange", "event": ev.key,
                                  "outcome": r.outcome, "back": f"{r.back_book} @ {r.back_price}",
                                  "lay": f"{r.exchange_book} @ {r.lay_price}",
                                  "back_stake": r.back_stake, "lay_stake": r.lay_stake,
                                  "profit": r.guaranteed_profit, "return_pct": r.return_pct,
                                  "warnings": r.warnings})
        return found

    # ------------------------------------------------------------------
    # Market making / hedging
    # ------------------------------------------------------------------

    def quote(self, venue_id: str, selection: str, fair_prob: float, book,
              inventory_exposure: float = 0.0):
        return self.mm.quote(selection, fair_prob, book, inventory_exposure)

    def hedge(self, stake: float, odds: float, current_lay: float, commission_pct: float = 0.05):
        return hedge_to_green(stake, odds, current_lay, commission_pct)

    # ------------------------------------------------------------------
    # Full match card - corners, cards, goal derivatives, props
    # ------------------------------------------------------------------

    def price_card(self, ev: SportsEvent, strengths: Optional[Dict] = None) -> Optional[MatchCardPrices]:
        """
        Price every derivable market on one fixture from one consistent model.

        Returns None when there are no attack/defence strengths for the
        fixture - the engine will not invent a model, and without one there
        is nothing to compare a book price against.
        """
        strengths = strengths or {}
        s = strengths.get(ev.key)
        if not s:
            return None

        # Route by sport BEFORE pricing. Feeding an NBA fixture into the
        # soccer scoreline grid produces numbers that look plausible and are
        # wrong, which is worse than refusing.
        if is_high_scoring(ev.sport) or is_high_scoring(ev.league):
            return self._price_high_scoring_card(ev, s)

        home = s.get("home") or {}
        away = s.get("away") or {}
        card = price_match_card(
            event_key=ev.key,
            home_attack=float(home.get("attack", 1.0)), home_defence=float(home.get("defence", 1.0)),
            away_attack=float(away.get("attack", 1.0)), away_defence=float(away.get("defence", 1.0)),
            corners_home=home.get("corners"), corners_away=away.get("corners"),
            cards_home=home.get("cards"), cards_away=away.get("cards"),
            red_rate=float(s.get("red_rate", 0.06)),
            shots_home=home.get("shots"), shots_away=away.get("shots"),
            sot_home=home.get("shots_on_target"), sot_away=away.get("shots_on_target"),
            offsides=s.get("offsides"),
        )
        self.priced_cards[ev.key] = card
        return card

    def _price_high_scoring_card(self, ev: SportsEvent, s: Dict) -> Optional[MatchCardPrices]:
        """
        Price a basketball/gridiron/baseball/hockey fixture with the normal
        margin model.

        Expects strengths of the form
            {'home': {'expected_points': 114.0}, 'away': {'expected_points': 110.0},
             'total_std': 20.5, 'margin_std': 12.0}
        and falls back to league priors for the standard deviations rather
        than for the scores - a score cannot be invented, but a variance can
        be taken from the long-run league figure.
        """
        home = s.get("home") or {}
        away = s.get("away") or {}
        exp_home = home.get("expected_points")
        exp_away = away.get("expected_points")
        if exp_home is None or exp_away is None:
            logger.info(f"[betting] {ev.key}: no expected points supplied for a "
                        f"high-scoring fixture - refusing to invent a score")
            return None

        # ev.sport is often the league code ("nba"), which params_for now
        # resolves; prefer whichever of the two is recognised.
        sport = resolve_sport(ev.sport) if is_high_scoring(ev.sport) else resolve_sport(ev.league)
        params = params_for(sport)
        card = price_high_scoring_card(
            event_key=ev.key, sport=sport,
            expected_home=float(exp_home), expected_away=float(exp_away),
            total_std=s.get("total_std"), margin_std=s.get("margin_std"),
            total_lines=s.get("total_lines") or [
                round(params.typical_total), params.typical_total + 0.5,
                params.typical_total - 0.5, params.typical_total + 5.5,
                params.typical_total - 5.5],
            spread_lines=s.get("spread_lines") or [0.0, -1.5, 1.5, -3.5, 3.5, -6.5, 6.5, -10.5],
            team_total_lines=s.get("team_total_lines"),
            margin_bands=s.get("margin_bands"),
            periods=s.get("periods") or ({"Q1": 0.25, "H1": 0.5} if sport == "basketball"
                                         else {"H1": 0.5} if sport == "football" else None),
        )
        self.priced_high_scoring[ev.key] = card
        return None

    def high_scoring_fair_prices(self, ev_key: str) -> Dict[str, Dict[str, float]]:
        """Fair prices from the normal margin model, keyed like the soccer card."""
        card = self.priced_high_scoring.get(ev_key)
        if card is None:
            return {}
        out: Dict[str, Dict[str, float]] = {
            "moneyline": card.moneyline,
            "margin_bands": card.margin_bands,
        }
        for ln, t in card.totals.items():
            out[f"totals_{ln}"] = {"over": t.over, "under": t.under, "push": t.push}
        for ln, sp in card.spreads.items():
            out[f"spreads_{ln}"] = {"home": sp.home, "away": sp.away, "push": sp.push}
        for name, v in card.team_totals.items():
            out[f"team_total_{name}"] = {"over": v.over, "under": v.under, "push": v.push}
        for name, v in card.periods.items():
            out[f"period_{name}"] = {"over": v["over"], "under": v["under"]}
        return out

    def card_fair_prices(self, ev_key: str) -> Dict[str, Dict[str, float]]:
        """
        Fair price for every market on the catalogue that the model covers.

        These are model prices, NOT bettable odds. They become opportunities
        only when a feed supplies a real book price to compare against.
        """
        card = self.priced_cards.get(ev_key)
        if card is None:
            return {}
        g = card.goals
        out: Dict[str, Dict[str, float]] = {
            "h2h": {"home": g.p_home, "draw": g.p_draw, "away": g.p_away},
            "double_chance": g.double_chance(),
            "draw_no_bet": g.draw_no_bet(),
            "btts": g.btts(),
            "clean_sheet": g.clean_sheet(),
            "win_to_nil": g.win_to_nil(),
            "correct_score": g.correct_score(),
            "totals_2.5": g.totals(2.5),
            "team_goals_home_1.5": g.team_totals("home", 1.5),
            "team_goals_away_1.5": g.team_totals("away", 1.5),
            "asian_handicap_-0.5": g.handicap(-0.5),
            "asian_handicap_-0.25": g.handicap(-0.25),
            "european_handicap_-1.0": g.european_handicap(-1.0),
            "race_to_2_goals": g.race_to(2),
            "half_time_result": {"home": card.half_time.ht_home, "draw": card.half_time.ht_draw,
                                 "away": card.half_time.ht_away},
            "ht_ft": card.half_time.ht_ft,
            "half_time_goals_0.5": card.half_time.ht_totals,
        }
        if card.corners:
            out["corners_total"] = card.corners.totals
            out["corners_handicap_-1.5"] = card.corners.handicap
            out["corners_1x2"] = {"home": card.corners.p_home, "draw": card.corners.p_draw,
                                  "away": card.corners.p_away}
        if card.cards:
            out["cards_total"] = card.cards.totals
            out["cards_1x2"] = {"home": card.cards.p_home, "draw": card.cards.p_draw,
                                "away": card.cards.p_away}
        out["red_card"] = card.red_card
        out["booking_points"] = card.booking_points
        if card.shots:
            out["shots_total"] = card.shots.totals
        if card.shots_on_target:
            out["shots_on_target_total"] = card.shots_on_target.totals
        if card.offsides:
            out["offsides_total"] = card.offsides.totals
        return out

    def _build_card_opportunities(self, ev: SportsEvent, card: MatchCardPrices,
                                  consensus_by_type: Dict[str, ConsensusOdds],
                                  data_mode: DataMode,
                                  account_health_ok: bool) -> List[BetOpportunity]:
        """
        Turn priced card markets into opportunities where a book price exists.

        Honest limitation: the public feeds PTAI reads carry h2h, spread and
        total. Corners/cards/props need a book that quotes them (Betfair
        exchange does). Where no book price exists the market is priced but
        NOT marked executable, with the reason recorded - quoting a fair
        price with nothing to bet against would be inventing an opportunity.
        """
        out: List[BetOpportunity] = []
        fair = self.card_fair_prices(ev.key)
        # High-scoring sports use the normal margin model and keep their own
        # priced-card dict, so merge their fair prices in.
        fair.update(self.high_scoring_fair_prices(ev.key))
        books_by_type = {mt: c for mt, c in consensus_by_type.items()}

        for market_key, prices in fair.items():
            book = books_by_type.get(self._feed_type_for(market_key))
            for outcome, prob in prices.items():
                if not isinstance(prob, (int, float)) or prob <= 0:
                    continue
                blockers: List[str] = []
                warnings: List[str] = []
                price = 0.0
                book_name = ""

                if book is not None and outcome in book.best_price:
                    price = book.best_price[outcome]
                    book_name = book.best_book.get(outcome, "")
                else:
                    blockers.append(
                        f"no book price for {market_key}/{outcome} - fair price "
                        f"{prob:.3f} computed but nothing to bet against")

                if data_mode.is_synthetic:
                    blockers.append(f"data_mode {data_mode.value} is synthetic - cannot trade")
                if data_mode.can_deploy_live_capital and not account_health_ok:
                    blockers.append("account health not verified - refusing live capital")

                edge = 0.0
                stake = 0.0
                if price > 1.0:
                    edge = prob * price - 1.0
                    if edge * 100.0 < self.min_edge_pct:
                        blockers.append(f"edge {edge*100.0:+.2f}% below {self.min_edge_pct}% threshold")
                    sizing = size_back(self.bankroll, prob, price, self.kelly_frac,
                                       self.max_position_pct, 0.0, self.min_stake)
                    stake = sizing["stake"]
                    if stake <= 0:
                        blockers.append(f"sizing: {sizing['reason']}")

                out.append(BetOpportunity(
                    opportunity_id=f"bet-{ev.event_id}-{market_key}-{outcome}",
                    event_key=ev.key, sport=ev.sport, league=ev.league,
                    market_type=market_key, outcome=str(outcome), side="back",
                    price=round(price, 4), book=book_name,
                    stake=round(stake, 2), liability=round(stake, 2),
                    model_prob=round(float(prob), 4), fair_prob=round(float(prob), 4),
                    edge=round(edge, 4), edge_pct=round(edge * 100.0, 3),
                    conservative_prob=round(float(prob), 4), conservative_edge=round(edge, 4),
                    net_ev_usd=round(stake * edge, 4),
                    kelly_stake=round(stake, 2),
                    uncertainty=0.0, confidence=0.0,
                    data_mode=data_mode.value,
                    data_source="sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
                    commission_pct=0.0, is_arb=False,
                    executable=not blockers and stake > 0,
                    blockers=blockers, warnings=warnings,
                    reasoning=f"{market_key}/{outcome}: model {prob:.3f} vs "
                              f"{'book '+str(round(price,3)) if price>1.0 else 'no book price'}",
                ))
        return out

    # Priced card market -> the feed market type that would quote it.
    # Explicit rather than derived from the key, because "asian_handicap_-0.5"
    # and "totals_2.5" do not decompose into a feed type by splitting on "_".
    _CARD_TO_FEED: Dict[str, str] = {
        "h2h": "h2h",
        "double_chance": "double_chance",
        "draw_no_bet": "draw_no_bet",
        "btts": "btts",
        "clean_sheet": "clean_sheet",
        "win_to_nil": "win_to_nil",
        "correct_score": "correct_score",
        "totals_2.5": "totals",
        "team_goals_home_1.5": "team_goals",
        "team_goals_away_1.5": "team_goals",
        "asian_handicap_-0.5": "spreads",
        "asian_handicap_-0.25": "spreads",
        "european_handicap_-1.0": "european_handicap",
        "race_to_2_goals": "race_to_goals",
        "half_time_result": "half_time_result",
        "ht_ft": "ht_ft",
        "half_time_goals_0.5": "half_time_goals",
        "corners_total": "corners_total",
        "corners_handicap_-1.5": "corners_handicap",
        "corners_1x2": "corners_1x2",
        "cards_total": "cards_total",
        "cards_1x2": "cards_1x2",
        "red_card": "red_card",
        "booking_points": "booking_points",
        "shots_total": "shots_total",
        "shots_on_target_total": "shots_on_target_total",
        "offsides_total": "offsides_total",
    }

    @classmethod
    def _feed_type_for(cls, card_market_key: str) -> Optional[str]:
        if card_market_key in cls._CARD_TO_FEED:
            return cls._CARD_TO_FEED[card_market_key]
        # High-scoring keys carry their line: "totals_225.0", "spreads_-3.5".
        # Strip the line suffix so they map onto the feed's market type.
        for prefix in ("totals_", "spreads_", "team_total_", "period_"):
            if card_market_key.startswith(prefix):
                return prefix.rstrip("_")
        # player props are keyed "player_goals:<name>" - strip the player
        base = card_market_key.split(":", 1)[0]
        return cls._CARD_TO_FEED.get(base, base)

    def price_player_props(self, ev: SportsEvent, players: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Price player props for one fixture.

        players = [{'name': 'Haaland', 'side': 'home', 'share': 0.42,
                    'minutes_share': 0.95, 'position': 'ST'}, ...]

        `share` is the player's historical share of that team's goals/shots,
        so the prop moves with the team model rather than being treated in
        isolation. Returns {} when no player data is supplied - the engine
        does not guess who is playing.

        Every prop carries the void warning from the catalogue: books differ
        on what happens when a player does not start, and that is the single
        biggest settlement risk in props.
        """
        card = self.priced_cards.get(ev.key)
        if card is None or not players:
            return {}

        team_lambda = {
            "home": card.goals.home_lambda,
            "away": card.goals.away_lambda,
        }
        opp_lambda = {
            "home": card.goals.away_lambda,
            "away": card.goals.home_lambda,
        }
        shot_lambda = {}
        if card.shots:
            shot_lambda = {"home": card.shots.home_lambda, "away": card.shots.away_lambda}
        sot_lambda = {}
        if card.shots_on_target:
            sot_lambda = {"home": card.shots_on_target.home_lambda,
                          "away": card.shots_on_target.away_lambda}

        out: Dict[str, Dict[str, float]] = {}
        for pl in players:
            name = pl.get("name")
            side = pl.get("side")
            if not name or side not in ("home", "away"):
                continue
            share = float(pl.get("share", 0.0))
            minutes_share = float(pl.get("minutes_share", 1.0))

            props: Dict[str, float] = {"name": name, "side": side}
            tl = team_lambda[side]

            anytime = price_anytime_scorer(tl, share, minutes_share)
            props["anytime_scorer_yes"] = anytime["yes"]
            props["expected_goals"] = anytime["expected_goals"]

            first = price_first_scorer(tl, share, opp_lambda[side], minutes_share)
            props["first_scorer_yes"] = first["yes"]

            if shot_lambda:
                props["expected_shots"] = price_player_count(
                    shot_lambda[side], share * float(pl.get("shot_share_mult", 1.0)))["expected"]
            if sot_lambda:
                props["expected_shots_on_target"] = price_player_count(
                    sot_lambda[side], share * float(pl.get("sot_share_mult", 1.0)))["expected"]

            # A carded-player prop is a share of the team's card expectation.
            cards_lambda = (card.cards.home_lambda if side == "home"
                            else card.cards.away_lambda) if card.cards else 0.0
            props["expected_cards"] = round(
                cards_lambda * float(pl.get("card_share", 0.0)) * minutes_share, 4)
            props["void_warning"] = ("books differ on non-starters - verify this book's "
                                     "player-prop void rule before trading")

            out[f"{name}"] = props
            self.player_props[f"{ev.key}:{name}"] = props
        return out

    def player_prop_fair_prices(self, ev: SportsEvent, players: List[Dict],
                                lines: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Player props expressed as market keys the catalogue recognises, so
        they flow through the same opportunity and settlement path.
        """
        props = self.price_player_props(ev, players)
        if not props:
            return {}
        lines = lines or {}
        card = self.priced_cards.get(ev.key)
        out: Dict[str, Dict[str, float]] = {}
        for name, p in props.items():
            side = p["side"]
            team_goals = (card.goals.home_lambda if side == "home"
                          else card.goals.away_lambda) if card else 1.0
            share = p["expected_goals"] / team_goals if team_goals > 0 else 0.0

            line = lines.get(name)
            if line is not None:
                pc = price_player_count(team_goals, share, line=line)
                out[f"player_goals:{name}"] = {"over": pc["over"], "under": pc["under"]}
            a = price_anytime_scorer(team_goals, share, 1.0)
            out[f"anytime_scorer:{name}"] = {"yes": a["yes"], "no": a["no"]}
        return out

    def _build_player_prop_opportunities(self, ev: SportsEvent, players: List[Dict],
                                         consensus_by_type: Dict[str, ConsensusOdds],
                                         data_mode: DataMode, account_health_ok: bool,
                                         prop_lines: Optional[Dict[str, float]] = None,
                                         sink: Optional[List[BetOpportunity]] = None) -> List[BetOpportunity]:
        """
        Player props as opportunities, through the same gates as everything else.

        Props carry an extra blocker class the other markets do not: the
        void-if-not-starting rule differs between books, so a prop is never
        marked executable without an explicit confirmation that this book's
        rule has been checked.
        """
        out: List[BetOpportunity] = []
        fair = self.player_prop_fair_prices(ev, players, prop_lines)

        for market_key, prices in fair.items():
            feed_type = self._feed_type_for(market_key)
            book = consensus_by_type.get(feed_type)
            for outcome, prob in prices.items():
                if not isinstance(prob, (int, float)) or prob <= 0:
                    continue
                blockers = ["player prop: confirm this book's void-if-not-starting rule"]
                warnings: List[str] = []
                price = 0.0
                book_name = ""
                stake = 0.0
                edge = 0.0

                if book is not None and outcome in book.best_price:
                    price = book.best_price[outcome]
                    book_name = book.best_book.get(outcome, "")
                    edge = prob * price - 1.0
                    if edge * 100.0 < self.min_edge_pct:
                        blockers.append(f"edge {edge*100.0:+.2f}% below {self.min_edge_pct}% threshold")
                    sizing = size_back(self.bankroll, prob, price, self.kelly_frac,
                                       self.max_position_pct, 0.0, self.min_stake)
                    stake = sizing["stake"]
                    if stake <= 0:
                        blockers.append(f"sizing: {sizing['reason']}")
                else:
                    blockers.append(f"no book price for {market_key}/{outcome} - fair "
                                    f"{prob:.3f} computed but nothing to bet against")

                if data_mode.is_synthetic:
                    blockers.append(f"data_mode {data_mode.value} is synthetic - cannot trade")
                if data_mode.can_deploy_live_capital and not account_health_ok:
                    blockers.append("account health not verified - refusing live capital")

                opp = BetOpportunity(
                    opportunity_id=f"prop-{ev.event_id}-{market_key}-{outcome}",
                    event_key=ev.key, sport=ev.sport, league=ev.league,
                    market_type=market_key.split(":", 1)[0], outcome=str(outcome),
                    side="back", price=round(price, 4), book=book_name,
                    stake=round(stake, 2), liability=round(stake, 2),
                    model_prob=round(float(prob), 4), fair_prob=round(float(prob), 4),
                    edge=round(edge, 4), edge_pct=round(edge * 100.0, 3),
                    conservative_prob=round(float(prob), 4), conservative_edge=round(edge, 4),
                    net_ev_usd=round(stake * edge, 4), kelly_stake=round(stake, 2),
                    uncertainty=0.0, confidence=0.0, data_mode=data_mode.value,
                    data_source="sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
                    commission_pct=0.0, is_arb=False,
                    executable=not blockers and stake > 0,
                    blockers=blockers, warnings=warnings,
                    reasoning=f"{market_key}/{outcome}: model {prob:.3f} vs "
                              f"{'book ' + str(round(price, 3)) if price > 1.0 else 'no book price'}",
                )
                out.append(opp)

        if sink is not None:
            sink.extend(out)
        return out

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------

    def build_accumulator(self, legs: List[Dict]) -> AccumulatorQuote:
        """
        Price a parlay with an explicit correlation haircut.

        legs = [{'market_key','outcome','price','model_prob','event_key','line'}, ...]
        """
        acc_legs = [AccumulatorLeg(**l) for l in legs]
        return price_accumulator(acc_legs)

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_bet(self, market_key: str, facts: MatchFacts, outcome: str,
                   stake: float, decimal_odds: float,
                   line: Optional[float] = None) -> Dict[str, Any]:
        """
        Settle one bet through the catalogue so the engine, paper trading and
        the backtester cannot drift apart on what 'won' means.
        """
        ok, why = validate_line(market_key, line)
        if not ok:
            return {"settled": False, "reason": why, "pnl": 0.0}
        result = settle_market(market_key, facts, outcome, line)
        multiplier = payout_multiplier(result, decimal_odds)
        returned = stake * multiplier
        return {
            "settled": result not in (Settlement.UNSETTLEABLE,),
            "result": result.value,
            "multiplier": round(multiplier, 4),
            "returned": round(returned, 2),
            "pnl": round(returned - stake, 2),
            "stake": round(stake, 2),
        }

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_closing_line(self, entry_odds: float, closing_odds: float) -> float:
        clv = closing_line_value(entry_odds, closing_odds)
        self.clv_history.append(clv)
        return clv

    def sharpness(self) -> Dict[str, Any]:
        """
        Judge the process on CLV rather than on PnL.

        Beating the close is the only short-sample evidence that a betting
        process has real edge. Realised P&L over 20 bets is mostly noise.
        """
        is_sharp, mean_clv, n = is_sharp_process(self.clv_history, min_samples=20)
        return {
            "n_settled_with_clv": n,
            "mean_clv_pct": mean_clv,
            "beats_the_close": is_sharp,
            "verdict": ("process shows real edge - beats the close consistently"
                        if is_sharp else
                        f"insufficient or negative CLV ({n} samples, mean {mean_clv}%) - "
                        f"do not scale up on PnL alone"),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        return {
            "layer": "betting / sports exchange",
            "bankroll": self.bankroll,
            "data_providers": self.data.health(),
            "models": self.models.get_report(),
            "account_risk": self.risk.report(),
            "exchange_accounts": {k: v.report() for k, v in self.exchange_accounts.items()},
            "sharpness": self.sharpness(),
            "market_catalogue": catalogue_report(),
            "market_types_scanned_from_feed": self.feed_market_types,
            "cards_priced": len(self.priced_cards),
            "opportunities": len(self.opportunities),
            "executable": sum(1 for o in self.opportunities if o.executable),
            "gates": [
                "data_mode.is_synthetic -> blocked",
                "data_mode.can_deploy_live_capital requires verified account health",
                "edge below threshold -> blocked",
                "edge vanishes under uncertainty -> blocked",
                "stake below exchange/book minimum -> blocked",
                "bookmaker account-risk policy -> blocked",
                "steam move -> blocked (edge already gone)",
            ],
            "no_fabrication": "no odds are invented; empty feed => no opportunities",
        }

    def to_markets(self, data_mode: DataMode = DataMode.LIVE_SHADOW) -> List[Market]:
        """Expose fixtures as Market objects so the unified scanner can rank them."""
        out: List[Market] = []
        for ev in self.data.events_cache:
            consensus = self.data.consensus(ev, "h2h")
            if consensus is None:
                continue
            m = self.data.to_market(consensus, data_mode)
            if m:
                out.append(m)
        return out
