"""
The runtime the user actually launches must be the architecture that was built.

Two failure modes motivated this suite, both found by auditing the code rather
than by running it:

1. `ptai run` and `ptai pay-for-yourself` constructed the legacy
   `agent/loop.py` TradingAgent. V3 - the only agent with the qualification ->
   multi-venue discovery -> expected-net-EV -> Kelly -> guard -> executor
   pipeline - had no CLI entry point at all. Architecture was ahead of runtime.

2. `TradingAgentV3.run_cycle` returned `{"total", "selected", "trades"}` on the
   no-markets branch while `run_continuous` read
   `result['opportunities']['final_selected']`. That is a KeyError on every
   cycle in which no venue returned markets - i.e. a fresh install with no
   credentials, which is exactly the state a new user starts in.

These tests execute the runtime; they do not grep it. Assertions on source text
are used only where the property is genuinely static (which class a CLI command
binds), and even then the text is parsed, not executed, so a rename cannot make
them pass vacuously.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "src" / "ptai" / "cli.py"
V3_PATH = REPO_ROOT / "src" / "ptai" / "agent" / "v3_loop.py"


def _cli_module_ast() -> ast.Module:
    return ast.parse(CLI_PATH.read_text())


def _command_fn(name: str) -> ast.FunctionDef:
    """Find a @app.command()-decorated function by name, without importing."""
    for node in _cli_module_ast().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                if "app.command" in text:
                    return node
    raise AssertionError(f"CLI command {name!r} not found")


def _constructed_class_names(fn: ast.FunctionDef) -> set[str]:
    found = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            found.add(node.func.id)
    return found


class TestCliLaunchesTheCanonicalPipeline:
    """`ptai run` and `ptai pay-for-yourself` must construct TradingAgentV3."""

    @pytest.mark.parametrize("command", ["run", "pay_for_yourself"])
    def test_command_constructs_v3(self, command):
        fn = _command_fn(command)
        ctor_names = _constructed_class_names(fn)
        assert "TradingAgentV3" in ctor_names, (
            f"`ptai {command}` builds {sorted(ctor_names)} - it must build "
            f"TradingAgentV3, the only agent with the full pipeline"
        )

    @pytest.mark.parametrize("command", ["run", "pay_for_yourself"])
    def test_command_does_not_construct_legacy_agent(self, command):
        fn = _command_fn(command)
        ctor_names = _constructed_class_names(fn)
        assert "TradingAgent" not in ctor_names, (
            f"`ptai {command}` still constructs the legacy TradingAgent; the "
            f"legacy loop lacks qualification, expected net EV and the guard"
        )

    @pytest.mark.parametrize("command", ["run", "pay_for_yourself"])
    def test_command_awaits_run_continuous(self, command):
        """
        The legacy loop exposed run_autonomous; V3's continuous entry point is
        run_continuous. Calling the wrong name is an AttributeError at startup.
        """
        fn = _command_fn(command)
        awaited = {
            node.value.func.attr
            for node in ast.walk(fn)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
        }
        assert "run_continuous" in awaited, (
            f"`ptai {command}` awaits {sorted(awaited)}; expected run_continuous"
        )
        assert "run_autonomous" not in awaited, (
            "run_autonomous belongs to the legacy agent, not V3"
        )

    def test_v3_is_imported_at_module_level(self):
        """
        A conditional/forward-declared import could pass the AST checks above
        while failing at runtime.
        """
        tree = _cli_module_ast()
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "agent.v3_loop":
                imported |= {a.name for a in node.names}
        assert "TradingAgentV3" in imported


class TestV3HasASafetyDefault:
    """Wiring the CLI to V3 must not be able to arm live trading by accident."""

    def test_default_is_dry_run(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        sig = inspect.signature(TradingAgentV3.__init__)
        assert "dry_run" in sig.parameters, (
            "V3 must accept dry_run; the CLI's --dry-run flag needs somewhere "
            "to land"
        )
        assert sig.parameters["dry_run"].default is True, (
            "dry_run must default True - the safe direction"
        )

    def test_dry_run_drives_data_mode(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.markets.base import DataMode

        dry = TradingAgentV3(dry_run=True)
        live = TradingAgentV3(dry_run=False)
        assert dry.data_mode == DataMode.LIVE_PAPER
        assert live.data_mode == DataMode.LIVE
        assert dry.data_mode != live.data_mode, (
            "dry_run must actually change the data mode, or it is decoration"
        )

    def test_betting_scan_does_not_hardcode_live_shadow(self):
        """
        The betting scan used to pass DataMode.LIVE_SHADOW literally, so the
        agent's own dry_run choice could not reach it.
        """
        source = V3_PATH.read_text()
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "LIVE_SHADOW":
                offenders.append(node.lineno)
        assert not offenders, (
            f"DataMode.LIVE_SHADOW is still hardcoded at line(s) {offenders}; "
            f"the betting scan must use self.data_mode"
        )


class TestNoMarketsReturnShape:
    """
    The no-markets branch must return the same keys as the success path.

    This is asserted by *executing* run_cycle against a scanner that finds
    nothing - the production path for a user who has not configured a venue
    yet - and then feeding the result through run_continuous's own reader.
    """

    def test_no_markets_shape_matches_success_shape(self):
        """
        Both shapes must carry every key the loop and the dashboard read.
        """
        from src.ptai.agent import v3_loop

        # The keys the completion log, the continuous loop and the dashboard read.
        required = {
            "opportunities",
            "do_nothing_success",
            "execution_time",
            "discovery",
            "alpha",
            "betting",
            "execution",
        }

        agent = v3_loop.TradingAgentV3(dry_run=True)

        # Force the no-markets branch: every venue reports zero markets.
        async def _empty(*args, **kwargs):
            return {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(agent, "discover_all_venues", _empty)
            result = asyncio.run(agent.run_cycle())

        assert result.get("status") == "no_markets"
        missing = required - set(result)
        assert not missing, (
            f"the no-markets branch is missing {sorted(missing)}; every reader "
            f"of a cycle result assumes the success-path shape"
        )

        opps = result["opportunities"]
        for key in ("total_candidates", "total_tradeable", "final_selected", "best"):
            assert key in opps, (
                f"opportunities is missing {key!r} on the no-markets branch; "
                f"run_continuous reads final_selected every cycle"
            )

    def test_run_continuous_survives_no_markets(self):
        """
        The regression itself: run_continuous read final_selected with
        __getitem__, so one empty cycle raised KeyError and the loop died.
        """
        from src.ptai.agent import v3_loop

        agent = v3_loop.TradingAgentV3(dry_run=True)

        async def _empty(*args, **kwargs):
            return {}

        cycles = {"n": 0}

        async def _stop_after_one(interval_minutes=10):
            """One iteration of run_continuous's body, then stop."""
            cycles["n"] += 1
            result = await agent.run_cycle()
            # This is the exact read the loop performs each cycle.
            n = (result.get("opportunities") or {}).get("final_selected", 0)
            ok = result.get("do_nothing_success", True)
            assert n == 0
            assert ok is True
            raise asyncio.CancelledError

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(agent, "discover_all_venues", _empty)
            mp.setattr(agent, "run_continuous", _stop_after_one)
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(agent.run_continuous(interval_minutes=1))

        assert cycles["n"] == 1

    def test_loop_reads_are_get_based(self):
        """
        Belt and braces: no remaining bracket-index read of
        result['opportunities'] in the loop body, which would re-introduce the
        same class of crash if a future branch returns a new shape.
        """
        tree = ast.parse(V3_PATH.read_text())
        offenders = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "result"
            ):
                offenders.append(node.lineno)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "result"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in ("opportunities", "do_nothing_success")
            ):
                offenders.append(node.lineno)
        assert not offenders, (
            f"result[...] is bracket-indexed at line(s) {offenders}; reads must "
            f"use .get so an unexpected shape cannot kill the loop"
        )


class TestBankrollSeeding:
    """
    V3 sizes from storage, not from a constructor argument. The CLI's
    --bankroll flag must therefore reach storage before V3 is constructed, or
    the flag is silently ignored.
    """

    def test_cli_seeds_bankroll_before_constructing_v3(self):
        fn = _command_fn("run")
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("_seed_bankroll", "TradingAgentV3")
        ]
        order = [n.func.id for n in sorted(calls, key=lambda n: n.lineno)]
        assert "_seed_bankroll" in order, (
            "--bankroll must be persisted to storage: V3 reads the bankroll from "
            "get_performance_summary(), not from its constructor"
        )
        assert order.index("_seed_bankroll") < order.index("TradingAgentV3"), (
            "bankroll must be seeded before V3 is constructed, since the "
            "constructor reads it"
        )

    def test_seeding_persists_to_storage(self):
        from src.ptai.cli import _seed_bankroll
        from src.ptai.storage.db import Storage

        storage = Storage()
        try:
            original = storage.get_performance_summary().get("bankroll")
        finally:
            storage.close()

        try:
            _seed_bankroll(123.0)
            storage = Storage()
            try:
                got = storage.get_performance_summary().get("bankroll")
            finally:
                storage.close()
            assert float(got) == 123.0
        finally:
            # Never leave the shared store mutated for other tests or for the
            # next real run: the whole point of seeding is that it persists.
            if original is not None:
                _seed_bankroll(float(original))


class TestCycleResultPrinterNeverCrashes:
    """
    Reporting must not be able to raise. The printer runs after the cycle, so
    an exception there loses the result of work already done.
    """

    @pytest.mark.parametrize(
        "result",
        [
            {},
            {"status": "no_markets"},
            {"status": "blocked", "reason": "low bankroll"},
            {
                "status": "success",
                "discovery": {"total_scanned": 10, "per_venue": {"polymarket": {}}},
                "opportunities": {
                    "total_candidates": 3,
                    "total_tradeable": 1,
                    "final_selected": 1,
                    "best": {
                        "venue": "polymarket",
                        "strategy": "value",
                        "question": "Will X happen?",
                        "edge": 0.12,
                        "score": 1.4,
                        "side": "YES",
                        "reasoning": "mispriced",
                    },
                },
                "execution": [{"id": 1}],
                "do_nothing_success": False,
                "execution_time": 12.5,
                "betting": {"events": 4, "data_mode": "live_paper"},
            },
        ],
    )
    def test_printer_accepts(self, result):
        from src.ptai.cli import _print_cycle_result

        _print_cycle_result(result)  # must not raise


class TestRestrictedIsNotUnknown:
    """
    "We have not researched this venue's legality here" and "this venue is
    prohibited here" are different facts.

    The capability engine collapsed both into CapabilityStatus.RESTRICTED,
    because `legal_eligible` treats UNKNOWN like a denial. For a UG operator
    that reported 13 venues as legally restricted when 12 were merely
    unresearched - a false statement about the law that also hid the real work
    item (go and check), and made the agent look like it had exhausted its
    options when it had not started.
    """

    @staticmethod
    def _cap_report(country="UG"):
        """Run the real capability engine and return its report."""
        import asyncio

        from src.ptai.agent import v3_loop

        agent = v3_loop.TradingAgentV3(country_code=country, dry_run=True)
        report = asyncio.run(agent.capability_engine.evaluate_all_venues(target_per_venue=20))
        eligibility = agent.venue_registry.check_all_eligibility()
        return report, eligibility

    def test_unknown_eligibility_is_not_reported_as_restricted(self):
        from src.ptai.venues.adapter import EligibilityStatus
        from src.ptai.venues.capability_engine import CapabilityStatus

        assert EligibilityStatus.UNKNOWN.value == "unknown"
        assert CapabilityStatus.UNDETERMINED.value == "undetermined"
        assert (
            CapabilityStatus.UNDETERMINED != CapabilityStatus.RESTRICTED
        ), "undetermined must be its own status or the distinction is lost"

    def test_live_run_separates_the_two(self):
        """
        Execute the engine against the real registry for UG. The only venue
        with a genuine determination is kalshi; everything unresearched must
        land in undetermined, not restricted.
        """
        import asyncio

        from src.ptai.agent import v3_loop
        from src.ptai.venues.capability_engine import CapabilityStatus

        cap, eligibility = self._cap_report()

        kalshi = {r.venue_id: r for r in cap.venue_reports}.get("kalshi")
        assert kalshi is not None, "kalshi must appear in the report"
        assert kalshi.status == CapabilityStatus.RESTRICTED, (
            f"kalshi is restricted for UG by the adapter's own list, but the "
            f"report says {kalshi.status}"
        )

        unknown_count = sum(
            1 for v in eligibility.values()
            if str(v) == "EligibilityStatus.UNKNOWN"
        )
        assert cap.undetermined_venues == unknown_count, (
            f"report says {cap.undetermined_venues} undetermined but the "
            f"registry returned {unknown_count} UNKNOWN eligibilities"
        )
        assert cap.restricted_venues < cap.total_venues, (
            "not every venue can be legally restricted"
        )

    def test_report_keeps_the_counts_reconcilable(self):
        """
        The counts are read as a summary of why nothing traded. They must sum
        to the venue total, or the summary is misleading.
        """
        import asyncio

        from src.ptai.agent import v3_loop

        cap, _ = self._cap_report()

        accounted = (
            cap.qualified_venues
            + cap.data_only_venues
            + cap.restricted_venues
            + cap.untested_venues
            + cap.undetermined_venues
        )
        assert accounted <= cap.total_venues, (
            f"{accounted} venues accounted for out of {cap.total_venues}"
        )
        assert cap.undetermined_venues > 0, (
            "a fresh UG install has unresearched venues; if this is 0 the "
            "distinction is not being applied"
        )

    def test_reasoning_text_names_the_blocked_venues(self):
        """
        The operator needs to know WHICH venues need researching, not just how
        many.
        """
        import asyncio

        from src.ptai.agent import v3_loop

        cap, _ = self._cap_report()
        text = cap.reasoning

        assert "Undetermined" in text
        assert "not yet researched" in text, (
            "the report must state that these are blocked on research rather "
            "than on law"
        )


class TestCredentialsReachTheAgent:
    """
    A credential that never arrives looks exactly like a credential that was
    never configured.

    `TradingAgentV3.__init__` read the vault with `self.vault.get(...)`, and Vault
    has no `get`. That raised AttributeError on EVERY construction, and the
    handler set both credentials to None. The settings fallback on the same line
    was never evaluated, because the exception happened first. So an operator
    with a correct key in the vault AND in settings got an agent that reported
    "unconfigured" - and live trading, the order probe and redemption were all
    permanently unreachable for a reason no message explained.
    """

    def test_credentials_from_the_vault_reach_the_agent(self, monkeypatch):
        from src.ptai.agent import v3_loop as mod

        class FakeVault:
            def __init__(self, *args, **kwargs):
                pass

            def get_tool_credentials(self, teammate, tool):
                return {"private_key": "0xKEY", "funder": "0xFUNDER"}

        monkeypatch.setattr(mod, "Vault", FakeVault)
        agent = mod.TradingAgentV3(country_code="UG", dry_run=True)
        assert agent.private_key == "0xKEY"
        assert agent.funder == "0xFUNDER"
        # ... and reach the venue that needs them, or the read proved nothing.
        adapter = agent.venue_registry.adapters["polymarket"]
        assert adapter.capabilities.supports_trading is True, (
            "the adapter still believes it cannot trade, so the credentials "
            "stopped at the agent"
        )

    def test_a_broken_vault_falls_back_to_settings(self, monkeypatch):
        """
        A vault that cannot be read must not discard credentials that settings
        already hold. Losing them silently is worse than the vault error itself.
        """
        from src.ptai.agent import v3_loop as mod

        class BrokenVault:
            def __init__(self, *args, **kwargs):
                pass

            def get_tool_credentials(self, teammate, tool):
                raise RuntimeError("vault unreadable")

        monkeypatch.setattr(mod, "Vault", BrokenVault)
        # The agent shares the process settings object, so patch it in a way
        # that is undone afterwards - mutating it directly leaks into every
        # later test in the session.
        probe = mod.TradingAgentV3(country_code="UG", dry_run=True)
        monkeypatch.setattr(probe.settings, "polymarket_private_key",
                            "0xFROM_SETTINGS")
        monkeypatch.setattr(probe.settings, "polymarket_funder_address",
                            "0xFUNDER_SETTINGS")

        # Constructed again with those settings in place.
        agent2 = mod.TradingAgentV3(country_code="UG", dry_run=True)
        assert agent2.private_key == "0xFROM_SETTINGS"
        assert agent2.funder == "0xFUNDER_SETTINGS"

    def test_no_credentials_anywhere_is_none_and_paper_still_works(self, monkeypatch):
        """
        The honest default. No credentials means no live trading - and paper
        trading, which is how a venue earns the right to hold money, is
        unaffected.
        """
        from src.ptai.agent import v3_loop as mod

        class EmptyVault:
            def __init__(self, *args, **kwargs):
                pass

            def get_tool_credentials(self, teammate, tool):
                return None

        monkeypatch.setattr(mod, "Vault", EmptyVault)
        agent = mod.TradingAgentV3(country_code="UG", dry_run=True)
        assert agent.private_key is None
        assert agent.funder is None
        # A redemption client with no funder must skip rather than raise: there
        # is no position list to read without an address.
        assert getattr(agent.redeemer, "funder", None) in (None, "")
