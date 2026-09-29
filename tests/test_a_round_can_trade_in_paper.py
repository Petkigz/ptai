"""
A round must be ABLE to place a paper trade - and say what it made or lost.

The operator, 2026-09-29: "paper mode need to start genearting income and doing
trades because in all the lkogs ive seen so mfar ive not seen 1 trade ... give
the results in the ui, the profits or losses made in a round".

This is that acceptance, as a test: a REAL `TradingAgentV3.run_cycle` over a
stub venue with a real two-sided book and a model that has a real disagreement
with the price.

What it pins, end to end:

  * a call the live gates refuse but the model's own estimate would still pay
    for becomes a PAPER/EXPLORATION trade - recorded, priced, and in the round;
  * the round reports it: `trades`, `positions_opened`, `staked_usd`, and a
    `net_usd` that is the difference between the account's opening and closing
    value (the spread paid is the first honest P&L a paper round can show);
  * a model that says nothing (no disagreement with the market) trades NOTHING
    - the paper lane is a lane with a bar, not a licence;
  * the same market is not bought again while the position is open.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.agent.v3_loop import TradingAgentV3
from src.ptai.llm.provider import LLMResponse
from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.strategy.strategy_engine import resolution_days
from src.ptai.venues.adapter import EligibilityStatus, MarketAdapter, VenueType

N = 12
# What the stub venue's book looks like: worth 0.40 in the middle, 0.42 to buy a
# YES share. A model that answers 0.55 is 15 points away from the mid and 13 from
# the price it would pay - a real disagreement, below the live bar once the risk
# haircut is applied.
MARKET_MID = 0.40
MARKET_ASK = 0.42


def _market(i: int) -> Market:
    # Real end dates, six hours apart: the paper lane prefers markets that will
    # actually settle, and the operator's 100-trade record only grows when they
    # do. pm0 settles first, then pm1, and so on down the venue.
    return Market(id=f"pm{i}", source=MarketSource.POLYMARKET,
                  question=f"Will event {i} happen this year?",
                  outcome_prices=[MARKET_MID, 1 - MARKET_MID], volume_24h=120_000,
                  liquidity=25_000, venue_id="polymarket",
                  data_mode=DataMode.LIVE,
                  end_date=datetime.now(timezone.utc) + timedelta(hours=6 + 6 * i),
                  raw={"venue_id": "polymarket", "token_id": f"t{i}",
                       "condition_id": f"c{i}"})


def _book(i: int) -> dict:
    return {"market_id": f"pm{i}", "venue_id": "polymarket",
            "bids": [{"price": MARKET_MID, "size": 900}],
            "asks": [{"price": MARKET_ASK, "size": 900}],
            "bid": MARKET_MID, "ask": MARKET_ASK,
            "spread": MARKET_ASK - MARKET_MID, "depth": 40000.0,
            "is_real": True, "is_mock": False, "validated": True,
            "executable": True, "executable_price": MARKET_ASK, "source": "clob"}


class _StubVenue(MarketAdapter):
    """The real adapter base, so every attribute the execution path reads exists."""

    def __init__(self) -> None:
        super().__init__("polymarket", VenueType.PREDICTION, dry_run=True)
        self.capabilities.supports_trading = False
        self.capabilities.fee_taker_pct = 0.0
        self.capabilities.order_gas_usd = 0.0
        self.capabilities.implementation_status = "live"
        self.orders = []

    def check_eligibility(self, country_code):
        return EligibilityStatus.ELIGIBLE

    def is_probability_market(self, market):
        return True

    async def discover_markets(self, target_count=200):
        return [_market(i) for i in range(N)]

    async def get_orderbook(self, market):
        return _book(int(str(market.id).replace("pm", "")))

    async def get_balance(self):
        return {"available": 0.0, "total": 0.0}

    async def get_portfolio(self):
        return {"positions": [], "venue_only_positions": [], "cash": None,
                "account_state_incomplete": True}

    def get_fee_rate(self, market):
        return 0.0

    async def place_order(self, opportunity, max_spend_usd, max_price):
        """The real adapter's dry-run contract: walk the book, report the fill."""
        self.orders.append((opportunity.market.id, max_spend_usd, max_price))
        book = await self.get_orderbook(opportunity.market)
        side = str(getattr(opportunity, "side", "YES")).upper()
        levels = book["asks"] if side == "YES" else book["bids"]
        remaining, filled_usd, shares = float(max_spend_usd), 0.0, 0.0
        for level in levels:
            price = float(level["price"])
            if side == "YES" and price > float(max_price) + 1e-9:
                break
            if side == "NO" and price < 1.0 - float(max_price) - 1e-9:
                break
            take = min(remaining, float(level["size"]) * price)
            filled_usd += take
            shares += take / price
            remaining -= take
            if remaining <= 1e-9:
                break
        if filled_usd <= 0:
            return {"status": "paper", "simulated_filled_usd": 0.0,
                    "filled_price": 0.0,
                    "reason": "the book could not fill it at this price"}
        return {"status": "paper",
                "simulated_filled_usd": round(filled_usd, 4),
                "filled_price": round(filled_usd / shares, 4),
                "shares": round(shares, 6)}


class _StubRouter:
    """
    A local model that ANSWERS, with the number the test chooses.

    A forecast only counts as model-backed if the model answered, so the stub
    returns a parsed JSON block exactly as the provider does - not a heuristic.
    """

    def __init__(self, fair: float, confidence: float = 0.75) -> None:
        self.fair = float(fair)
        self.confidence = float(confidence)
        self.active_model = "stub-model"
        self.active_model_reason = "the test's stand-in model"
        self.provider = None
        self.reset_usage()

    def reset_usage(self) -> None:
        self.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                      "models": {}, "last_model": "", "last_error": ""}

    def usage_report(self) -> dict:
        return dict(self.usage)

    def is_available(self) -> bool:
        return True

    def get_provider_name(self) -> str:
        return "LMStudioProvider"

    def describe(self) -> str:
        return f"{self.active_model} (the test's stand-in) - {self.active_model_reason}"

    def chat(self, prompt: str, system: str = ""):
        self.usage["calls"] += 1
        payload = {"fair_value": self.fair, "confidence": self.confidence,
                   "side": "YES", "should_trade": True,
                   "reasoning": "the test's stand-in answer"}
        self.usage["answered"] += 1
        self.usage["models"]["stub-model"] = self.usage["models"].get("stub-model", 0) + 1
        return LLMResponse(content=json.dumps(payload), model="stub-model",
                           provider="lm_studio", parsed_json=payload)


@pytest.fixture()
def paper_round(tmp_path, monkeypatch):
    """A real agent, a stub venue, a stub model - one round, no network."""
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    monkeypatch.delenv("PTAI_LLM_THINKING", raising=False)

    def run(fair: float, cycles: int = 1):
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        venue = _StubVenue()
        agent.venue_registry.adapters = {"polymarket": venue}
        agent.venue_registry._routed_venues = set()
        router = _StubRouter(fair)
        agent.llm_router = router
        agent.ensemble_forecaster.llm_router = router
        if getattr(agent, "brain", None) is not None:
            agent.brain.llm_router = router
        if getattr(agent, "fair_value_engine", None) is not None:
            agent.fair_value_engine.llm_router = router
        results = []
        for _ in range(cycles):
            results.append(asyncio.run(agent.run_cycle(target_per_venue=N,
                                                       max_trades=3)))
        return agent, venue, results

    return run


def _trade_rows(round_dict):
    return list(round_dict.get("trades") or [])


# ---------------------------------------------------------------------------
# 1. a real disagreement the live gates refuse is still a PAPER trade
# ---------------------------------------------------------------------------

class TestThePaperLaneTrades:
    def test_the_round_places_a_paper_trade_and_records_it(self, paper_round):
        agent, venue, results = paper_round(0.55)
        result = results[0]
        round_dict = result["round"]

        execution = result.get("execution") or []
        recorded = [e for e in execution if e.get("position_recorded")]
        assert recorded, (
            "a market where the model said 0.55 against a 0.42 ask produced no "
            "paper trade at all - this is the operator's 'I have not seen 1 "
            "trade'")
        assert recorded[0]["execution_mode"] == "paper"
        assert recorded[0]["fill"]["filled_price"] == pytest.approx(MARKET_ASK)
        assert recorded[0]["fill"]["filled_usd"] == pytest.approx(1.0, abs=0.01)
        # The venue was asked, and the adapter refused real money itself.
        assert venue.orders, "the paper lane never reached the venue adapter"

    def test_the_round_carries_the_trade_and_a_profit_or_loss(self, paper_round,
                                                              monkeypatch):
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "1")
        _agent, _venue, results = paper_round(0.55)
        round_dict = results[0]["round"]
        rows = _trade_rows(round_dict)
        assert len(rows) == 1
        row = rows[0]
        assert row["market_id"].startswith("pm")
        assert row["venue"] == "polymarket"
        assert row["side"] in ("YES", "NO")
        assert row["execution_mode"] == "paper"
        assert row["amount_usd"] == pytest.approx(1.0, abs=0.01)
        assert row["price"] == pytest.approx(MARKET_ASK)
        assert row["exploration"] is True
        assert "paper/exploration" in row["strategy"]
        # The round's own P&L: the account is worth what it holds, marked at the
        # bid it could sell into, so paying the 2c spread shows up immediately.
        assert round_dict["positions_opened"] == 1
        assert round_dict["staked_usd"] == pytest.approx(1.0, abs=0.01)
        # `net_usd` is the unrounded difference; equity_start/end are printed to
        # the cent and `net_usd` to a hundredth of a cent, so the two agree to
        # within that printing rather than exactly.
        assert round_dict["net_usd"] == pytest.approx(
            round_dict["equity_end"] - round_dict["equity_start"], abs=0.01)
        assert round_dict["net_usd"] == pytest.approx(-0.0476, abs=0.002)
        assert round_dict["verdict"] == "down"
        assert round_dict["realised_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert round_dict["unrealised_pnl"] == pytest.approx(-0.05, abs=0.01)

    def test_three_trades_in_one_round_are_counted_once_each(self, paper_round,
                                                             monkeypatch):
        """
        With the ceiling at three the round's P&L is still ONE number about the
        account - not the sum of three separate stories - and each $1 stake is
        counted exactly once.
        """
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "3")
        _agent, _venue, results = paper_round(0.55)
        round_dict = results[0]["round"]
        rows = _trade_rows(round_dict)
        assert len(rows) == 3
        assert round_dict["positions_opened"] == 3
        assert round_dict["staked_usd"] == pytest.approx(3.0, abs=0.05)
        assert round_dict["net_usd"] == pytest.approx(
            round_dict["equity_end"] - round_dict["equity_start"], abs=0.01)
        # Every stake paid the 2c spread and is marked at the bid: 3 x -0.05.
        assert round_dict["unrealised_pnl"] == pytest.approx(-0.15, abs=0.02)
        assert round_dict["net_usd"] < 0
        assert len({r["market_id"] for r in rows}) == 3

    def test_the_closest_call_is_reported_even_when_nothing_qualifies(
            self, paper_round):
        _agent, _venue, results = paper_round(0.55)
        closest = results[0].get("closest_call") or {}
        assert closest.get("market_id"), (
            "the round must say which market came closest and by how much - it "
            "is the only honest answer to 'how long do I have to run this'")
        assert closest["model_answered"] is True
        assert closest["mispricing"] > 0
        assert closest["refusal"]


class _FunnelVenue(_StubVenue):
    """
    200 markets, and only the lower-volume half has a usable book.

    This is the shape of the operator's 2026-09-29 19:33 cycle: the volume
    leaders were the multi-outcome markets whose CLOB books the venue refused
    (69 rejections, 98 with no book at all), while the mid-volume markets priced
    fine.
    """

    BOOKLESS = 100

    async def discover_markets(self, target_count=200):
        out = []
        for i in range(200):
            market = _market(i)
            # volume descends with i, so pm0..pm99 are the volume leaders
            market.volume_24h = float(200_000 - i * 900)
            out.append(market)
        return out

    async def get_orderbook(self, market):
        index = int(str(market.id).replace("pm", ""))
        if index < self.BOOKLESS:
            return {"market_id": market.id, "venue_id": "polymarket",
                    "bids": [], "asks": [], "is_real": False,
                    "validated": False, "source": "clob",
                    "validation": {"identity": "no usable levels"}}
        return _book(index)


# ---------------------------------------------------------------------------
# 3. model time goes to the markets the venue can actually price
# ---------------------------------------------------------------------------

class TestTheModelTimeGoesToPriceableMarkets:
    def _run(self, tmp_path, monkeypatch, fair: float = 0.55):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
        monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
        monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.venue_registry.adapters = {"polymarket": _FunnelVenue()}
        agent.venue_registry._routed_venues = set()
        router = _StubRouter(fair)
        agent.llm_router = router
        agent.ensemble_forecaster.llm_router = router
        if getattr(agent, "brain", None) is not None:
            agent.brain.llm_router = router
        if getattr(agent, "fair_value_engine", None) is not None:
            agent.fair_value_engine.llm_router = router
        result = asyncio.run(agent.run_cycle(target_per_venue=200, max_trades=3))
        return agent, result

    def test_a_bookless_volume_leader_does_not_eat_a_pricing_slot(
            self, tmp_path, monkeypatch):
        agent, _result = self._run(tmp_path, monkeypatch)
        screen = agent._screen or {}
        # The screen read 200, and the four kinds of outcome add up to 200:
        # chosen 8, below the scan's cap 0, refused here 100 (no usable book),
        # priceable but not shortlisted 92.
        assert screen.get("considered") == 200
        assert screen.get("refused_at_screen") == 100
        assert screen.get("dropped_by_scan") == 0
        assert screen.get("unaccounted") == 0, (
            "every market the screen read must be accounted for - the operator "
            "reads these numbers to answer 'how long do I have to run this'")

    def test_the_shortlist_uses_the_deep_budget_and_the_model_is_called(
            self, tmp_path, monkeypatch):
        agent, result = self._run(tmp_path, monkeypatch)
        status = result.get("local_model") or {}
        assert agent._screen.get("shortlist"), "nothing was shortlisted"
        assert len(agent._screen["shortlist"]) == agent.deep_analysis_limit, (
            "the deep budget is 8 and 100 markets had a validated book; a "
            "shortlist of 1-2 is the funnel defect that produced no trades")
        assert status.get("asked") >= len(agent._screen["shortlist"]), (
            "the markets given model time must be the markets handed to the "
            "model")
        assert status.get("answered_by_model") >= 1
        assert status.get("deep_priced") >= len(agent._screen["shortlist"])

    def test_the_round_trades_because_the_model_was_asked(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "1")
        _agent, result = self._run(tmp_path, monkeypatch)
        round_dict = result["round"]
        assert round_dict["positions_opened"] == 1
        assert len(round_dict["trades"]) == 1
        assert round_dict["net_usd"] < 0  # the spread paid, marked


# ---------------------------------------------------------------------------
# 2. the lane has a bar: no disagreement, no trade
# ---------------------------------------------------------------------------

class TestThePaperLaneIsNotALicence:
    def test_a_model_that_agrees_with_the_market_trades_nothing(self, paper_round):
        _agent, venue, results = paper_round(0.40)
        round_dict = results[0]["round"]
        assert _trade_rows(round_dict) == []
        assert round_dict["positions_opened"] == 0
        assert round_dict["net_usd"] == pytest.approx(0.0, abs=1e-9)
        assert venue.orders == [], "nothing to buy - the venue must not be asked"

    def test_the_same_market_is_not_bought_again_while_it_is_open(self, paper_round):
        agent, _venue, results = paper_round(0.55, cycles=2)
        first = _trade_rows(results[0]["round"])
        second = _trade_rows(results[1]["round"])
        assert first, "the first round should have traded"
        assert second, "the second round should have traded a DIFFERENT market"
        assert first[0]["market_id"] != second[0]["market_id"], (
            "re-buying the same market every round stacks one bet into every "
            "slot; it is not a new learning opportunity")
        held = agent._held_market_ids("paper")
        assert first[0]["market_id"] in held
        assert second[0]["market_id"] in held


class TestTheLaneCanBuildARecord:
    """
    The record the operator is waiting on: up to N paper trades a cycle, each on
    a DIFFERENT market, soonest-resolving first.

    One trade per round was the ceiling while the lane only had to prove it could
    trade. Live capital needs 100 resolved trades, and a round that finds several
    qualifying markets may as well learn from several.
    """

    def test_one_cycle_places_several_distinct_paper_trades(
            self, paper_round, monkeypatch):
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "3")
        _agent, _venue, results = paper_round(0.55)
        round_dict = results[0]["round"]
        rows = _trade_rows(round_dict)
        assert len(rows) == 3, (
            f"3 candidates cleared the lane's bar but {len(rows)} trade(s) were "
            f"placed - the record cannot build at one a round")
        ids = [r["market_id"] for r in rows]
        assert len(set(ids)) == 3, "the same market must not be bought twice"
        # One round, one bankroll change: the stakes are counted once.
        assert round_dict["positions_opened"] == 3

    def test_the_setting_still_bounds_it(self, paper_round, monkeypatch):
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "1")
        _agent, _venue, results = paper_round(0.55)
        rows = _trade_rows(results[0]["round"])
        assert len(rows) == 1, (
            "PTAI_PAPER_TRADES_PER_CYCLE=1 must still mean one per cycle")

    def test_a_market_that_settles_sooner_is_taken_first(self, paper_round,
                                                        monkeypatch):
        """
        Same bar, same $1 - the one that resolves first teaches first.

        Every stub market has an end date six hours apart, and all of them clear
        the lane's bar, so with the ceiling at one the trade must be the market
        that settles soonest: pm0.
        """
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "1")
        _agent, _venue, results = paper_round(0.55)
        rows = _trade_rows(results[0]["round"])
        assert len(rows) == 1
        assert rows[0]["market_id"] == "pm0", (
            "the soonest-resolving candidate must be taken first, or the record "
            "waits on the calendar instead of the model")
        assert resolution_days(_market(0)) < resolution_days(_market(5))
