"""
THE LOG SAYS IT ONCE, AND SAYS IT IN THE RIGHT PLACE.

V55's log was read as a bug report, and a third of it was the system talking to
itself. Every defect below is a line the operator saw, or a line he had to hunt
for because the noise buried it:

  * `Storage initialized at data\\ptai.db` and `Portfolio from LOCAL STATE:
    balance $50.00 ...` repeated on every console poll, because the dashboard
    built its own `Storage(db_path="./data/ptai.db")` per request and the
    portfolio line had no memory of what it had already said;
  * `Routing market ... venue_id ... -> exact adapter ...` printed for every
    market of every cycle (DEBUG is on, on this machine), one line per market
    for 899 markets;
  * `ERROR ... Could not read Polymarket credentials ...` in a PAPER run, where
    credentials are not needed at all - an error line for a working system;
  * `Mechanics for predictit-... unreadable: AttributeError: 'PredictItAdapter'
    object has no attribute 'get_mechanics'` - the agent asked a venue for a
    reader it does not have, per market, per cycle, and swallowed its own
    mistake at DEBUG.

The fix for each is the same shape: one reader of the environment, one line per
fact, and a level that matches what actually happened.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.ptai.storage.db import Storage, default_data_dir


class _Capture:
    def __init__(self):
        from loguru import logger
        self.lines = []
        self._sink = logger.add(
            lambda m: self.lines.append((m.record["level"].name,
                                         m.record["message"])),
            level="DEBUG")

    def close(self):
        from loguru import logger
        logger.remove(self._sink)

    def messages(self, level=None):
        return [msg for lvl, msg in self.lines if level is None or lvl == level]


# ----------------------------------------------------------------------
# one reader of the environment
# ----------------------------------------------------------------------

class TestOneFamilyOfPaths:
    def test_storage_honours_ptai_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "custom.db"))
        assert Storage().db_path == tmp_path / "custom.db"

    def test_the_data_dir_is_the_database_folder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "custom.db"))
        monkeypatch.delenv("PTAI_DATA_DIR", raising=False)
        assert Path(default_data_dir()) == tmp_path

    def test_the_vault_lives_beside_the_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "custom.db"))
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        from src.ptai.vault.vault import Vault
        assert Vault().vault_path == tmp_path / "vault.json"

    def test_an_explicit_vault_path_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        from src.ptai.vault.vault import Vault
        explicit = tmp_path / "somewhere-else" / "v.json"
        assert Vault(str(explicit)).vault_path == explicit

    @pytest.mark.parametrize("relative", [
        "src/ptai/dashboard.py",
        "src/ptai/cli.py",
        "src/ptai/agent/loop.py",
        "src/ptai/agent/premium_loop.py",
        "src/ptai/agent/v2_loop.py",
        "src/ptai/venues/polymarket_adapter.py",
    ])
    def test_no_module_hardcodes_the_database_path(self, relative):
        source = (Path(__file__).resolve().parents[1] / relative).read_text()
        assert 'db_path="./data/ptai.db"' not in source


class TestTheStorageLineIsNotAPoll:
    def test_the_first_open_names_the_path_and_repeats_stay_quiet(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "once.db"))
        # a fresh process view, so the once-per-path memory starts empty
        from src.ptai.storage import db as db_module
        monkeypatch.setattr(db_module, "_LOGGED_DB_PATHS", set())
        capture = _Capture()
        try:
            Storage()
            Storage()
        finally:
            capture.close()
        infos = [m for m in capture.messages("INFO") if "Storage initialized" in m]
        assert len(infos) == 1
        assert str(tmp_path / "once.db") in infos[0]


class TestThePortfolioLineIsNotAPoll:
    def test_an_unchanged_local_state_is_not_repeated(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        source = inspect.getsource(PolymarketAdapter)
        assert "_local_state_logged" in source, (
            "the LOCAL STATE portfolio line must remember what it already said")

    def test_only_a_change_prints_it_again(self):
        from src.ptai.venues import polymarket_adapter as module
        source = inspect.getsource(module)
        assert "signature = (round(float(bankroll), 2)" in source


# ----------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------

class TestRoutingIsSaidOncePerVenue:
    def test_the_first_market_prints_and_the_rest_do_not(self):
        from src.ptai.venues.registry import VenueRegistry
        from src.ptai.markets.base import Market, MarketSource, DataMode

        class _Adapter:
            venue_id = "polymarket"

        class _Registry(VenueRegistry):
            def __init__(self):
                self.adapters = {"polymarket": _Adapter()}

        registry = _Registry()
        capture = _Capture()
        try:
            for i in range(20):
                market = Market(id=f"m{i}", source=MarketSource.POLYMARKET,
                                question="q", venue_id="polymarket",
                                data_mode=DataMode.LIVE)
                registry.get_adapter_for_market(market)
        finally:
            capture.close()
        routing = [m for m in capture.messages() if "Routing" in m]
        assert len(routing) == 1
        assert "polymarket" in routing[0]


# ----------------------------------------------------------------------
# credentials in paper mode
# ----------------------------------------------------------------------

class TestCredentialsAreNotAnErrorInPaperMode:
    def test_the_level_matches_the_mode(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.__init__)
        assert 'if getattr(self, "dry_run", True):' in source
        assert "PAPER mode, which needs no credentials" in source

    def test_a_paper_agent_does_not_log_an_error_about_credentials(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "paper.db"))
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("POLYMARKET_FUNDER_ADDRESS", raising=False)
        capture = _Capture()
        try:
            from src.ptai.agent.v3_loop import TradingAgentV3
            TradingAgentV3(country_code="UG", dry_run=True)
        finally:
            capture.close()
        errors = [m for m in capture.messages("ERROR")
                  if "credentials" in m.lower()]
        assert errors == [], ("an error line for a paper run reads like a "
                              "broken agent; paper needs no credentials")

    def test_the_information_is_still_there(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "paper2.db"))
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("POLYMARKET_FUNDER_ADDRESS", raising=False)
        capture = _Capture()
        try:
            from src.ptai.agent.v3_loop import TradingAgentV3
            TradingAgentV3(country_code="UG", dry_run=True)
        finally:
            capture.close()
        said = [m for m in capture.messages()
                if "credentials are not configured" in m]
        assert said, "the fact must still be stated, at a level that fits"


# ----------------------------------------------------------------------
# a venue is asked only for what it has
# ----------------------------------------------------------------------

class TestAVenueIsAskedOnlyForWhatItHas:
    class _PredictItLike:
        venue_id = "predictit"

        class capabilities:
            fee_taker_pct = 0.02

    class _Registry:
        def __init__(self, adapter):
            self._adapter = adapter

        def get_adapter_for_market(self, market):
            return self._adapter

    def _facts(self, adapter):
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.markets.base import Market, MarketSource, DataMode

        class _Self:
            venue_registry = TestAVenueIsAskedOnlyForWhatItHas._Registry(adapter)
            _arb_facts_cache = {}

        self_ = _Self()
        market = Market(id="predictit-8544-33624", source=MarketSource.PREDICTIT,
                        question="Will it happen?", venue_id="predictit",
                        outcome_prices=[0.3], data_mode=DataMode.LIVE,
                        raw={"venue_id": "predictit"})
        return TradingAgentV3._arb_venue_facts(self_, market)

    def test_an_adapter_without_the_reader_is_not_asked(self):
        capture = _Capture()
        try:
            facts = self._facts(self._PredictItLike())
        finally:
            capture.close()
        assert facts["mechanics_source"], "the fact must be recorded, not hunted"
        assert "no per-market mechanics reader" in facts["mechanics_source"]
        problems = [m for m in capture.messages()
                    if "AttributeError" in m or "unreadable" in m]
        assert problems == [], "a venue without that reader is not an error"

    def test_the_declared_fee_is_still_used(self):
        facts = self._facts(self._PredictItLike())
        assert facts["fee"]["rate"] == 0.02
        assert facts["fee"]["source"] == "predictit declared capability fee_taker_pct"

    def test_neg_risk_is_still_unknown_rather_than_false(self):
        facts = self._facts(self._PredictItLike())
        assert facts["neg_risk"] is None, ("an unknown answer must not read as "
                                           "'this market is not exclusive'")

    def test_the_reader_is_checked_with_getattr(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._arb_venue_facts)
        assert 'getattr(adapter, "get_mechanics", None)' in source


class TestAMissingReaderIsSaidOncePerVenue:
    class _PredictItLike:
        venue_id = "predictit"

        class capabilities:
            fee_taker_pct = 0.02

    class _Registry:
        def __init__(self, adapter):
            self._adapter = adapter

        def get_adapter_for_market(self, market):
            return self._adapter

    def _facts_for(self, adapter, market_id="predictit-1"):
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.markets.base import Market, MarketSource, DataMode

        class _Self:
            venue_registry = TestAMissingReaderIsSaidOncePerVenue._Registry(adapter)
            _arb_facts_cache = {}

        self_ = _Self()
        market = Market(id=market_id, source=MarketSource.PREDICTIT,
                        question="Will it happen?", venue_id="predictit",
                        outcome_prices=[0.3], data_mode=DataMode.LIVE,
                        raw={"venue_id": "predictit"})
        return TradingAgentV3._arb_venue_facts(self_, market), self_

    def test_the_sentence_appears_once_for_the_venue(self):
        adapter = self._PredictItLike()
        capture = _Capture()
        try:
            self._facts_for(adapter, "predictit-1")
            for i in range(20):
                self._facts_for(adapter, f"predictit-{i + 2}")
        finally:
            capture.close()
        noted = [m for m in capture.messages()
                 if "publishes no per-market mechanics" in m]
        assert len(noted) == 1, noted

    def test_it_is_information_not_an_error(self):
        adapter = self._PredictItLike()
        capture = _Capture()
        try:
            self._facts_for(adapter)
        finally:
            capture.close()
        levels = [lvl for lvl, msg in capture.lines
                  if "publishes no per-market mechanics" in msg]
        assert levels == ["INFO"]

    def test_the_failure_is_the_AttributeError_that_used_to_be_swallowed(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._arb_venue_facts)
        assert 'hasattr(adapter, "get_mechanics")' not in source
        assert "adapter._mechanics_absent_noted" in source
