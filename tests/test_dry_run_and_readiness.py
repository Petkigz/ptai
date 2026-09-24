"""
Dry run, and the account readiness ladder.

Two bugs motivated this suite of the "the code says one thing and does another"
family, both confirmed by execution:

1. `dry_run` had no effect on execution. The CLI's `--dry-run` defaults to True
   and the agent accepted it, but nothing downstream of the CLI ever read it:
   `ExecutionGuard` passed any non-mock data mode ("LIVE/PAPER OK"), the v3_loop
   logged "allowing paper execution only" and then executed identically, and
   `PolymarketAdapter.place_order` branched only on whether credentials existed.
   A user running the default `ptai run` with live Polymarket credentials
   configured would have placed REAL orders believing they were in a dry run.

2. Live Polymarket execution had never worked. `polymarket_adapter` imported
   `PolymarketExecutor` from `execution/polymarket_executor.py`, which defines
   only `ExecutionOrchestrator`; the ImportError was caught by a bare `except`
   and returned as `{"status": "error"}`, which reads like a transient failure.

3. Account health proved nothing. The portfolio "check" read a capability flag,
   `has_funds = True  # simplified` assumed funds from credentials, whitebit and
   crypto_binance were hardcoded `healthy = True`, and the report advertised
   "Can place orders" / "Can cancel orders" checks that did not exist.

These tests assert behaviour, by calling the code.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueOpportunity,
    VenueType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _market(tokens=True) -> Market:
    return Market(
        id="M-1",
        source=MarketSource.POLYMARKET,
        question="Will X happen?",
        tokens=(
            [Token(token_id="tok-yes", outcome="YES", price=0.5),
             Token(token_id="tok-no", outcome="NO", price=0.5)]
            if tokens else []
        ),
    )


def _opportunity(side="YES", tokens=True) -> VenueOpportunity:
    return VenueOpportunity(
        market=_market(tokens=tokens),
        venue_id="polymarket",
        venue_type=VenueType.PREDICTION,
        side=side,
        market_price=0.5,
        estimated_fair=0.65,
        raw_edge=0.15,
    )


def _credentialed_polymarket(dry_run: bool):
    """A Polymarket adapter holding credentials - the dangerous configuration."""
    from src.ptai.venues.polymarket_adapter import PolymarketAdapter

    adapter = PolymarketAdapter(private_key="0x" + "ab" * 32,
                                funder="0x" + "cd" * 20,
                                dry_run=dry_run)
    adapter.capabilities.supports_trading = True
    return adapter


# --------------------------------------------------------------------------
# 1. the adapter is the last gate and it defaults safe
# --------------------------------------------------------------------------

class TestAdapterDryRunIsTheGate:
    def test_base_adapter_defaults_to_dry_run(self):
        params = inspect.signature(MarketAdapter.__init__).parameters
        assert "dry_run" in params
        assert params["dry_run"].default is True, (
            "the last gate before real money must default to safe"
        )

    def test_can_place_real_orders_requires_both_conditions(self):
        adapter = _credentialed_polymarket(dry_run=True)
        assert not adapter.can_place_real_orders, (
            "credentials alone must not permit real orders"
        )

        adapter.dry_run = False
        assert adapter.can_place_real_orders

        adapter.capabilities.supports_trading = False
        assert not adapter.can_place_real_orders, (
            "a venue with no trading support must never be armed"
        )

    def test_place_order_refuses_real_submission_in_dry_run(self):
        """
        The regression: with real credentials and dry_run=True, place_order must
        return a simulated result and must not reach the CLOB client.
        """
        adapter = _credentialed_polymarket(dry_run=True)

        reached = {"called": False}

        class _Spy:
            def __init__(self, *a, **k):
                pass

            def place_order(self, *a, **k):
                reached["called"] = True
                return {"status": "matched", "orderID": "REAL-ORDER"}

        import src.ptai.markets.polymarket as pm
        original = pm.PolymarketExecutor
        pm.PolymarketExecutor = _Spy
        try:
            result = asyncio.run(adapter.place_order(_opportunity(), 2.0, 0.5))
        finally:
            pm.PolymarketExecutor = original

        assert not reached["called"], (
            "place_order reached the real CLOB client during a dry run"
        )
        assert result["status"] == "dry_run"
        assert result.get("simulated") is True

    def test_place_order_in_dry_run_reports_why(self):
        adapter = _credentialed_polymarket(dry_run=True)
        result = asyncio.run(adapter.place_order(_opportunity(), 2.0, 0.5))
        assert "dry run" in result["message"].lower()
        assert "would place" in result["message"].lower()
        assert "dry_run=true" in result["reason"].lower()

    def test_live_adapter_without_a_client_does_not_claim_success(self):
        """
        Armed, credentialed, but py_clob_client is absent -> the venue's own
        executor reports dry_run. It must never be reported as a live fill.
        """
        adapter = _credentialed_polymarket(dry_run=False)
        result = asyncio.run(adapter.place_order(_opportunity(), 2.0, 0.5))
        assert result["status"] in ("dry_run", "error", "rejected"), (
            f"got {result['status']!r} with no CLOB client available"
        )
        assert result["status"] != "matched"

    def test_agent_propagates_dry_run_to_every_adapter(self):
        """
        Without this the flag stops at the agent, which is exactly how a dry run
        could have submitted real orders.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3

        dry = TradingAgentV3(dry_run=True)
        for vid, adapter in dry.venue_registry.adapters.items():
            if hasattr(adapter, "dry_run"):
                assert adapter.dry_run is True, f"{vid} was not defused"

        live = TradingAgentV3(dry_run=False)
        assert any(
            getattr(a, "dry_run", None) is False
            for a in live.venue_registry.adapters.values()
        ), "a live agent must actually arm its adapters"


class TestPolymarketExecutionIsWired:
    """The import was wrong and the bare except hid it."""

    def test_executor_import_resolves(self):
        """
        `PolymarketAdapter.place_order` used to import PolymarketExecutor from
        execution.polymarket_executor, where it does not exist.
        """
        import src.ptai.execution.polymarket_executor as wrong_module
        with pytest.raises(ImportError):
            from src.ptai.execution.polymarket_executor import PolymarketExecutor  # noqa: F401

        from src.ptai.markets.polymarket import PolymarketExecutor
        assert callable(PolymarketExecutor)

    def test_no_bare_except_hides_the_import(self):
        """
        The wiring bug survived because an `except Exception` turned it into a
        runtime error status. Assert the import is not inside a swallowing
        handler in the place_order path.
        """
        import src.ptai.venues.polymarket_adapter as mod

        source = inspect.getsource(mod.PolymarketAdapter.place_order)
        assert "from ..markets.polymarket import PolymarketExecutor" in source, (
            "place_order must import the executor from markets.polymarket, "
            "where the class actually is"
        )
        assert "execution.polymarket_executor import PolymarketExecutor" not in source

    def test_place_order_signature_matches_what_we_call(self):
        """
        place_order is synchronous and takes (token_id, price, size, side,
        order_type, dry_run). The old call passed market/side/max_price/
        amount_usd to a method named execute() that does not exist.
        """
        from src.ptai.markets.polymarket import PolymarketExecutor

        params = inspect.signature(PolymarketExecutor.place_order).parameters
        for expected in ("token_id", "price", "size", "side", "order_type", "dry_run"):
            assert expected in params, f"place_order has no {expected!r} parameter"
        assert not inspect.iscoroutinefunction(PolymarketExecutor.place_order), (
            "place_order is sync; awaiting it would raise TypeError"
        )
        assert not hasattr(PolymarketExecutor, "execute"), (
            "if execute() now exists, revisit the call site"
        )

    def test_token_id_is_resolved_for_the_side(self):
        adapter = _credentialed_polymarket(dry_run=True)
        assert adapter._resolve_token_id(_opportunity(side="YES")) == "tok-yes"
        assert adapter._resolve_token_id(_opportunity(side="NO")) == "tok-no"

    def test_missing_token_id_is_rejected_not_guessed(self):
        adapter = _credentialed_polymarket(dry_run=False)
        result = asyncio.run(
            adapter.place_order(_opportunity(tokens=False), 2.0, 0.5)
        )
        assert result["status"] == "rejected"
        assert "token_id" in result["reason"]


# --------------------------------------------------------------------------
# 2. the readiness ladder
# --------------------------------------------------------------------------

class TestReadinessLadder:
    def test_levels_are_ordered(self):
        from src.ptai.execution.account_health import (
            TradeReadiness,
            readiness_rank,
        )

        ranks = [
            readiness_rank(TradeReadiness.NOT_CONFIGURED),
            readiness_rank(TradeReadiness.CONFIGURED),
            readiness_rank(TradeReadiness.AUTHENTICATED),
            readiness_rank(TradeReadiness.FUNDED),
            readiness_rank(TradeReadiness.TRADE_PERMITTED),
        ]
        assert ranks == sorted(ranks) == [0, 1, 2, 3, 4]

    def test_only_trade_permitted_is_ready(self):
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine()
        adapter = _credentialed_polymarket(dry_run=True)

        class _Registry:
            adapters = {"polymarket": adapter}

        engine.venue_registry = _Registry()

        # No credentials at all.
        adapter.private_key = None
        adapter.funder = None
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.NOT_CONFIGURED
        assert result.ready_to_trade is False
        assert result.healthy is False
        assert "credentials_missing" in result.blockers

    def test_credentials_alone_do_not_reach_ready(self):
        """
        The core claim: configured is not ready.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine()
        adapter = _credentialed_polymarket(dry_run=True)

        class _Registry:
            adapters = {"polymarket": adapter}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))

        assert result.readiness != TradeReadiness.TRADE_PERMITTED
        assert result.ready_to_trade is False
        assert result.readiness == TradeReadiness.CONFIGURED, (
            f"expected CONFIGURED, got {result.readiness}: {result.reason}"
        )
        assert "auth_unverified" in result.blockers, (
            f"expected the unproven-auth blocker, got {result.blockers}"
        )

    def test_no_venue_is_ever_hardcoded_healthy(self):
        """
        whitebit and crypto_binance used to be `healthy = True` with no
        credentials at all. Sweep every venue the registry can build and assert
        none of them is ready without proof.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.execution.account_health import TradeReadiness

        agent = TradingAgentV3(dry_run=True)
        results = asyncio.run(agent.account_health_engine.check_all_venues())
        assert results, "the registry should produce at least one venue"

        for vid, health in results.items():
            assert health.readiness != TradeReadiness.TRADE_PERMITTED, (
                f"{vid} reports TRADE_PERMITTED with no credentials configured"
            )
            assert health.ready_to_trade is False, f"{vid} claims ready_to_trade"
            assert health.healthy is False, (
                f"{vid} claims healthy with no verified account"
            )

    def test_report_separates_claimed_from_implemented_checks(self):
        """
        The old report listed "Can place orders" and "Can cancel orders" among
        its checks. Neither existed.
        """
        from src.ptai.execution.account_health import AccountHealthEngine

        report = AccountHealthEngine().get_report()
        claimed = " ".join(report["checks_that_require_evidence"]).lower()

        for phantom in ("place orders", "cancel orders", "withdraw", "deposit",
                        "rate limit"):
            assert phantom not in claimed, (
                f"report still claims a {phantom!r} check"
            )
        assert report["ladder"] == [
            "not_configured", "configured", "authenticated", "funded",
            "trade_permitted",
        ]

    def test_unimplemented_venue_reports_not_configured(self):
        """A stub adapter must not be reported as a configured account."""
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine()

        class _Stub(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="stub", venue_type=VenueType.PREDICTION)
                self.capabilities = AdapterCapability(
                    supports_trading=True,
                    implementation_status="unimplemented",
                    implementation_note="no client written",
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.UNKNOWN

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        class _Registry:
            adapters = {"stub": _Stub()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("stub"))
        assert result.readiness == TradeReadiness.NOT_CONFIGURED
        assert "not_implemented" in result.blockers
        assert result.ready_to_trade is False


class TestBalanceProvenance:
    """
    Balance follows the same rule as spread: a number is only usable if the
    producer said where it came from.
    """

    def test_stub_balance_is_not_read_as_zero(self):
        from src.ptai.execution.account_health import read_balance

        # This is literally what whitebit / crypto_adapter / kalshi return.
        balance, is_real, provenance = read_balance(
            {"balance": 0, "positions": [], "venue": "whitebit"}
        )
        assert balance is None, (
            "a stub balance of 0 must be unknown, not 'the account is empty'"
        )
        assert is_real is False
        assert provenance == "no_provenance"

    def test_sourced_balance_is_read(self):
        from src.ptai.execution.account_health import read_balance

        balance, is_real, provenance = read_balance(
            {"available": True, "balance": 42.5, "source": "betfair_account_api"}
        )
        assert balance == 42.5
        assert is_real is True
        assert provenance == "betfair_account_api"

    def test_explicit_zero_from_a_real_source_is_information(self):
        from src.ptai.execution.account_health import read_balance

        balance, is_real, _ = read_balance(
            {"is_real": True, "balance": 0.0, "source": "storage"}
        )
        assert is_real is True
        assert balance == 0.0, (
            "a real zero means the account is empty - that is information, "
            "unlike an unsourced stub zero"
        )

    def test_self_declared_fallback_is_not_real(self):
        from src.ptai.execution.account_health import read_balance

        balance, is_real, provenance = read_balance(
            {"balance": 50.0, "is_real": False, "note": "Fallback placeholder"}
        )
        assert is_real is False
        assert provenance == "declared_not_real"

    def test_unavailable_account_is_not_real(self):
        from src.ptai.execution.account_health import read_balance

        _, is_real, provenance = read_balance(
            {"available": False, "balance": None, "reason": "not configured"}
        )
        assert is_real is False
        assert provenance == "account_unavailable"

    def test_non_numeric_balance_is_not_real(self):
        from src.ptai.execution.account_health import read_balance

        _, is_real, _ = read_balance({"is_real": True, "balance": "lots"})
        assert is_real is False

    def test_unreadable_balance_stops_before_ready(self):
        """
        An adapter that authenticates but will not give a sourced balance must
        land on AUTHENTICATED, below FUNDED.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine()

        class _Sourced(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True, min_order_usd=1.0
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                # The venue answered (venue-side provenance), but gave us no
                # balance number. That proves auth without proving funds.
                return {"source": "clob_real", "positions": []}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        class _Registry:
            adapters = {"polymarket": _Sourced()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.AUTHENTICATED
        assert result.ready_to_trade is False
        assert "balance_unreadable" in result.blockers

    def test_funded_but_unprobed_stops_at_funded(self):
        """
        Auth proven and money proven is still not permission proven. No adapter
        implements an order probe, so FUNDED is the top rung anyone can reach,
        and it is deliberately not 'ready'.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine(bankroll=50.0)

        class _Funded(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True, min_order_usd=1.0
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {"available": True, "balance": 120.0, "source": "clob_real"}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        class _Registry:
            adapters = {"polymarket": _Funded()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))

        assert result.readiness == TradeReadiness.FUNDED, (
            f"expected FUNDED, got {result.readiness} ({result.reason})"
        )
        assert result.ready_to_trade is False
        assert "order_permission_unproven" in result.blockers
        assert "NOT verified" in result.reason

    def test_order_probe_is_what_unlocks_ready(self):
        """
        The ladder must actually be climbable: declare the probe, implement it,
        and the venue reaches TRADE_PERMITTED. Otherwise this is just a wall.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine(bankroll=50.0)

        class _Probed(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True,
                    supports_order_probe=True, min_order_usd=1.0,
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {"available": True, "balance": 120.0, "source": "clob_real"}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

            async def probe_order_permission(self):
                return True

        class _Registry:
            adapters = {"polymarket": _Probed()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.TRADE_PERMITTED
        assert result.ready_to_trade is True
        assert result.healthy is True

    def test_probe_declared_but_not_implemented_is_not_ready(self):
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine(bankroll=50.0)

        class _Liar(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True,
                    supports_order_probe=True, min_order_usd=1.0,
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {"available": True, "balance": 120.0, "source": "clob_real"}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        class _Registry:
            adapters = {"polymarket": _Liar()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness != TradeReadiness.TRADE_PERMITTED
        assert result.ready_to_trade is False

    def test_order_probe_raising_is_not_ready(self):
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine(bankroll=50.0)

        class _Broken(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True,
                    supports_order_probe=True, min_order_usd=1.0,
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {"available": True, "balance": 120.0, "source": "clob_real"}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

            async def probe_order_permission(self):
                raise RuntimeError("venue refused")

        class _Registry:
            adapters = {"polymarket": _Broken()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness != TradeReadiness.TRADE_PERMITTED
        assert "order_permission_unproven" in result.blockers

    def test_underfunded_is_not_funded(self):
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        engine = AccountHealthEngine(bankroll=50.0)

        class _Broke(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="polymarket", venue_type=VenueType.PREDICTION)
                self.private_key = "0x" + "ab" * 32
                self.funder = "0x" + "cd" * 20
                self.capabilities = AdapterCapability(
                    supports_trading=True, supports_portfolio=True, min_order_usd=10.0
                )

            def check_eligibility(self, country_code="UG"):
                return EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=100, **kwargs):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {"available": True, "balance": 2.0, "source": "clob_real"}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        class _Registry:
            adapters = {"polymarket": _Broke()}

        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.AUTHENTICATED
        assert "insufficient_funds" in result.blockers


class TestV3LoopEnforcesReadiness:
    """
    The guard in v3_loop read `account_health.healthy`, which was hardcoded True
    for two venues, and even when it printed "allowing paper execution only" it
    went on to execute identically. Now that `healthy` means ready_to_trade and
    is False for every unproven venue, the paper/live distinction has to be real
    rather than a log line.
    """

    def test_every_unproven_venue_is_not_healthy(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(dry_run=True)
        health = asyncio.run(agent.account_health_engine.check_all_venues())
        assert all(not h.healthy for h in health.values()), (
            "a venue reported healthy without a verified account, which is the "
            "condition the execution guard treats as permission to trade"
        )

    def test_dry_run_agent_cannot_reach_real_submission(self):
        """
        End to end over the real registry: with dry_run=True, no adapter in the
        agent can place a real order, whatever its credentials say.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(dry_run=True)
        for vid, adapter in agent.venue_registry.adapters.items():
            assert not adapter.can_place_real_orders, (
                f"{vid} can place real orders during a dry run"
            )


class TestLocalStorageIsNotVenueAuth:
    """
    Found while writing the ladder, by executing it: PolymarketAdapter.
    get_portfolio() reads the LOCAL database first and returns
    {"is_real": True, "source": "storage"}. That is a truthful statement about
    our own storage and no evidence at all that Polymarket authenticated us.
    The first version of this probe accepted it and certified the auth rung from
    a local file read - the same class of mistake as the capability-flag check
    it replaced.
    """

    def test_venue_side_sources_are_recognised(self):
        from src.ptai.execution.account_health import is_venue_side_provenance

        for good in ("betfair_account_api", "clob_real", "kalshi_api_real",
                     "whitebit_api_real", "betfair_exchange_live"):
            assert is_venue_side_provenance(good), f"{good} should count"

    def test_local_and_fallback_sources_are_rejected(self):
        from src.ptai.execution.account_health import is_venue_side_provenance

        for bad in ("storage", "storage+onchain_attempted", "local", "database",
                    "cache", "fallback_placeholder", "error_fallback",
                    "enhanced_estimation", "whitebit_mock_fallback",
                    "no_provenance", ""):
            assert not is_venue_side_provenance(bad), (
                f"{bad!r} must not count as venue-side authentication"
            )

    def test_a_venue_token_mixed_with_storage_is_rejected(self):
        """
        Compound sources are joined with "+". If any part is a local/fallback
        token, the whole response is not proof.
        """
        from src.ptai.execution.account_health import is_venue_side_provenance

        assert not is_venue_side_provenance("clob_real+storage")
        assert not is_venue_side_provenance("storage+clob_real")

    def test_polymarket_own_portfolio_is_not_treated_as_auth(self):
        """
        Execute the real adapter's get_portfolio and confirm the engine refuses
        to call it authentication.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        adapter = _credentialed_polymarket(dry_run=True)

        class _Registry:
            adapters = {"polymarket": adapter}

        engine = AccountHealthEngine()
        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))

        assert result.readiness == TradeReadiness.CONFIGURED, (
            f"a local-storage portfolio reached {result.readiness}: {result.reason}"
        )
        assert "auth_unverified" in result.blockers
        assert result.ready_to_trade is False

    def test_sourced_balance_without_a_number_keeps_its_provenance(self):
        """
        The venue answering "no balance available" is still an authentication
        event; dropping the provenance would lose that.
        """
        from src.ptai.execution.account_health import read_balance

        balance, is_real, provenance = read_balance(
            {"source": "clob_real", "positions": []}
        )
        assert balance is None
        assert is_real is False
        assert provenance == "clob_real", (
            "the declared provenance must survive a missing balance field"
        )
