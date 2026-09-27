"""
Logins the agent actually uses, and venue switches that actually bind.

Two operator complaints, both about control rather than capability:

  "my logins need to be saved somewhere so I don't always have to log in when the
   system is running automatically" - there was a vault and nothing could put
   anything into it from the product, Betfair read `os.getenv` in the middle of
   the trading loop, Kalshi was constructed with no arguments at all, and the
   sports-data keys the engine asked for in its own error message did not exist on
   `Settings` and so could only ever be empty;

  "no way to add more venues and have the system use them" - nineteen adapters
   were scanned whether or not the operator wanted them, with no switch.

What is pinned here:

  * a saved login WINS over the environment and is what the adapters receive;
  * `.env` keeps working, and each field says which source is in force;
  * a secret is encrypted because the SCHEMA says it is secret, not because its
    name happens to contain "key" - the old rule wrote Betfair passwords and
    Kalshi api_secrets to disk in plaintext;
  * a required field left blank is refused rather than stored as "configured";
  * an already-running adapter is refreshed, so saving a login mid-run takes
    effect on the next cycle instead of at a restart nobody mentioned;
  * a switched-OFF venue is not scanned and not traded - while a switched-ON
    venue still has to pass every safety check, because the switch only ever
    subtracts.
"""

from __future__ import annotations

import json

import pytest

from ptai.storage.db import Storage
from ptai.venues import credentials as store
from ptai.venues import preferences as prefs

#: A key of the shape `validate_private_key` accepts.
PK = "0x" + "ab" * 32
FUNDER = "0x" + "cd" * 20


@pytest.fixture()
def data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def storage(tmp_path):
    s = Storage(db_path=str(tmp_path / "ptai.db"))
    yield s
    s.close()


# --------------------------------------------------------------------------
# saving and reading
# --------------------------------------------------------------------------

def test_a_login_saved_in_the_console_is_what_the_agent_reads(data_dir):
    result = store.save("betfair", {"username": "op", "password": "hunter2",
                                    "app_key": "KEY123"}, data_dir)
    assert result["ok"], result
    assert set(result["saved"]) == {"username", "password", "app_key"}

    assert store.resolve("betfair", data_dir) == {
        "username": "op", "password": "hunter2", "app_key": "KEY123"}

    described = store.describe("betfair", data_dir)
    assert described["configured"] is True
    by_name = {f["name"]: f for f in described["fields"]}
    assert by_name["password"]["source"] == "saved"
    assert "hunter2" not in json.dumps(described), "a secret was returned in clear"


def test_the_environment_still_works_and_says_so(data_dir, monkeypatch):
    monkeypatch.setenv("BETFAIR_USERNAME", "envuser")
    monkeypatch.setenv("BETFAIR_PASSWORD", "envpass")
    monkeypatch.setenv("BETFAIR_APP_KEY", "envkey")
    described = store.describe("betfair", data_dir)
    by_name = {f["name"]: f for f in described["fields"]}
    assert by_name["username"]["source"] == "environment"
    assert described["configured"] is True


def test_a_saved_login_wins_over_the_environment(data_dir, monkeypatch):
    monkeypatch.setenv("BETFAIR_APP_KEY", "from-env")
    store.save("betfair", {"username": "op", "password": "pw",
                           "app_key": "from-console"}, data_dir)
    assert store.resolve("betfair", data_dir)["app_key"] == "from-console"
    by_name = {f["name"]: f for f in store.describe("betfair", data_dir)["fields"]}
    assert by_name["app_key"]["source"] == "saved"


def test_secrets_are_encrypted_because_the_schema_says_so(data_dir):
    store.save("betfair", {"username": "op", "password": "hunter2",
                           "app_key": "KEY123"}, data_dir)
    raw = open(store.vault_path(data_dir)).read()
    # The password is a SECRET even though its name says nothing about keys -
    # the old name-based rule left it in the file in clear text.
    assert "hunter2" not in raw, "the Betfair password was written unencrypted"
    assert "KEY123" not in raw, "the Betfair app key was written unencrypted"
    assert "op" in raw, "a non-secret field was encrypted for no reason"


def test_kalshi_api_secret_is_encrypted_too(data_dir):
    store.save("kalshi", {"api_key": "k", "api_secret": "s3cret",
                          "member_id": "m"}, data_dir)
    raw = open(store.vault_path(data_dir)).read()
    assert "s3cret" not in raw
    assert store.resolve("kalshi", data_dir)["api_secret"] == "s3cret"


def test_a_required_field_left_blank_is_refused(data_dir):
    result = store.save("betfair", {"username": "op", "password": "",
                                    "app_key": "KEY"}, data_dir)
    assert result["ok"] is False
    assert "still needed" in result["error"]
    assert store.resolve("betfair", data_dir) == {}, (
        "a half-filled login was stored as configured")


def test_a_bad_private_key_is_caught_while_typing(data_dir):
    result = store.save("polymarket", {"private_key": "not-a-key",
                                       "funder": FUNDER}, data_dir)
    assert result["ok"] is False
    assert any("private_key" in p for p in result["problems"])


def test_saving_one_field_does_not_wipe_the_others(data_dir):
    store.save("betfair", {"username": "op", "password": "pw",
                           "app_key": "KEY"}, data_dir)
    store.save("betfair", {"username": "op2", "password": "", "app_key": ""},
               data_dir)
    values = store.resolve("betfair", data_dir)
    assert values["username"] == "op2"
    assert values["password"] == "pw", "an untouched secret was deleted"
    assert values["app_key"] == "KEY"


def test_the_polymarket_login_keeps_the_old_vault_name_working(data_dir):
    # Credentials saved before this module existed live under
    # "polymarket_clob"; the operator must not have to type their key twice.
    from ptai.vault.vault import Vault

    Vault(vault_path=store.vault_path(data_dir)).store_tool_credentials(
        "Trader", "polymarket_clob",
        {"private_key": PK, "funder": FUNDER},
        secret_fields=["private_key"])
    assert store.resolve("polymarket", data_dir)["funder"] == FUNDER


def test_forgetting_a_login_removes_it(data_dir):
    store.save("betfair", {"username": "op", "password": "pw", "app_key": "K"},
               data_dir)
    assert store.forget("betfair", data_dir)["ok"]
    assert store.resolve("betfair", data_dir) == {}


def test_an_unreadable_vault_is_reported_not_raised(data_dir):
    with open(store.vault_path(data_dir), "w") as fh:
        fh.write("{ this is not json")
    described = store.describe("betfair", data_dir)
    assert described["configured"] is False


# --------------------------------------------------------------------------
# handing them to the agent
# --------------------------------------------------------------------------

def test_apply_to_settings_fills_what_the_adapters_read(data_dir):
    class Settings:
        the_odds_api_key = ""
        football_data_token = ""
        x_bearer_token = None

    store.save("the_odds_api", {"api_key": "odds-key"}, data_dir)
    state = store.apply_to_settings(Settings, data_dir)
    assert state["the_odds_api"] is True
    assert Settings.the_odds_api_key == "odds-key"


def test_refresh_adapters_reaches_a_running_agent(data_dir):
    class Adapter:
        def __init__(self):
            self.username = ""
            self.password = ""
            self.app_key = ""
            self.client = None

    class Registry:
        def __init__(self, adapter):
            self.adapter = adapter

        def get_adapter_for_venue_id(self, venue_id):
            return self.adapter if venue_id == "betfair" else None

    store.save("betfair", {"username": "op", "password": "pw", "app_key": "K"},
               data_dir)
    adapter = Adapter()
    changed = store.refresh_adapters(Registry(adapter), data_dir=data_dir)
    assert adapter.username == "op" and adapter.app_key == "K"
    assert "betfair" in changed


def test_refresh_updates_the_betfair_client_as_well_as_the_adapter(data_dir):
    class Client:
        def __init__(self):
            self.username = ""
            self.password = ""
            self.app_key = ""

    class Adapter:
        def __init__(self):
            self.username = ""
            self.password = ""
            self.app_key = ""
            self.client = Client()

    class Registry:
        def __init__(self, adapter):
            self.adapter = adapter

        def get_adapter_for_venue_id(self, venue_id):
            return self.adapter if venue_id == "betfair" else None

    store.save("betfair", {"username": "op", "password": "pw", "app_key": "K"},
               data_dir)
    adapter = Adapter()
    store.refresh_adapters(Registry(adapter), data_dir=data_dir)
    assert adapter.client.username == "op", (
        "the adapter was updated but its client would still log in with the old "
        "credentials")


def test_kalshi_gets_the_saved_key(data_dir):
    class Kalshi:
        def __init__(self):
            self.api_key = None
            self.api_secret = None
            self.member_id = None

    class Registry:
        def __init__(self, adapter):
            self.adapter = adapter

        def get_adapter_for_venue_id(self, venue_id):
            return self.adapter if venue_id == "kalshi" else None

    store.save("kalshi", {"api_key": "k", "api_secret": "s"}, data_dir)
    adapter = Kalshi()
    store.refresh_adapters(Registry(adapter), data_dir=data_dir)
    assert adapter.api_key == "k" and adapter.api_secret == "s"


def test_the_agent_applies_saved_logins_when_it_starts():
    # Source-level: the agent must read the store BEFORE it constructs adapters,
    # or a saved login reaches a process that has already built everything.
    from pathlib import Path

    source = Path("src/ptai/agent/v3_loop.py").read_text()
    flat = " ".join(source.split())
    assert "credential_store.apply_to_settings" in flat
    init = flat.split("self.betting_engine = BettingEngine(")[0]
    assert "apply_to_settings" in init, (
        "saved logins are applied after the adapters are built")
    assert "refresh_adapters" in flat, (
        "the agent never refreshes a login saved while it is running")


# --------------------------------------------------------------------------
# venue switches
# --------------------------------------------------------------------------

def test_every_venue_is_on_until_the_operator_says_otherwise(storage):
    assert prefs.is_enabled(storage, "polymarket") is True
    assert prefs.load(storage)["disabled"] == []


def test_switching_a_venue_off_persists(storage):
    assert prefs.set_enabled(storage, "crypto_binance", False)["ok"]
    assert prefs.is_enabled(storage, "crypto_binance") is False
    assert "crypto_binance" in prefs.load(storage)["disabled"]


def test_switching_it_back_on_clears_the_off(storage):
    prefs.set_enabled(storage, "kalshi", False)
    prefs.set_enabled(storage, "kalshi", True)
    assert prefs.is_enabled(storage, "kalshi") is True
    assert "kalshi" not in prefs.load(storage)["disabled"]


def test_an_explicit_off_wins_over_a_stale_on(storage):
    prefs.save(storage, disabled=[], enabled=["kalshi"])
    prefs.set_enabled(storage, "kalshi", False)
    # ...and even if the stale "on" entry survives, off is obeyed.
    current = prefs.load(storage)
    prefs.save(storage, disabled=current["disabled"],
               enabled=current["enabled"] + ["kalshi"])
    assert prefs.is_enabled(storage, "kalshi") is False


def test_filter_adapters_says_which_it_skipped(storage):
    class A:
        venue_id = "polymarket"

    class B:
        venue_id = "crypto_binance"

    prefs.set_enabled(storage, "crypto_binance", False)
    allowed, skipped = prefs.filter_adapters(storage, [A(), B()])
    assert [a.venue_id for a in allowed] == ["polymarket"]
    assert skipped == ["crypto_binance"]


def test_the_switch_is_read_from_storage_so_it_survives_a_restart(storage):
    prefs.set_enabled(storage, "veynor", False)
    reopened = Storage(db_path=str(storage.db_path))
    try:
        assert prefs.is_enabled(reopened, "veynor") is False
    finally:
        reopened.close()


def test_a_disabled_venue_is_not_scanned_and_the_reason_is_the_operators_choice():
    from pathlib import Path

    source = Path("src/ptai/agent/v3_loop.py").read_text()
    flat = " ".join(source.split())
    assert "venue_preferences.filter_adapters" in flat
    assert "switched OFF by you" in flat, (
        "an off venue must read as the operator's choice, not as a failure")
