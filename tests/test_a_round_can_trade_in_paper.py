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

import pytest

from src.ptai.agent.v3_loop import TradingAgentV3
from src.ptai.llm.provider import LLMResponse
from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.venues.adapter import EligibilityStatus, MarketAdapter, VenueType

N = 12
# What the stub venue's book looks like: worth 0.40 in the middle, 0.42 to buy a
# YES share. A model that answers 0.55 is 15 points away from the mid and 13 from
# the price it would pay - a real disagreement, below the live bar once the risk
# haircut is applied.
MARKET_MID = 0.40
MARKET_ASK = 0.42


def _market(i: int) -> Market:
    return Market(id=f"pm{i}", source=MarketSource.POLYMARKET,
                  question=f"Will event {i} happen this year?",
                  outcome_prices=[MARKET_MID, 1 - MARKET_MID], volume_24h=120_000,
                  liquidity=25_000, venue_id="polymarket",
                  data_mode=DataMode.LIVE,
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

    def test_the_round_carries_the_trade_and_a_profit_or_loss(self, paper_round):
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
