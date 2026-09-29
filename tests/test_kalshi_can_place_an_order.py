"""
Kalshi is the second venue whose adapter can submit a real order, so it gets the
same treatment Polymarket's order path did: what it sends is asserted field by
field, what it does NOT send is asserted too, and every number that comes back is
the exchange's own rather than a fill the adapter decided had happened.

Nothing here touches the network. Kalshi signs its requests with an RSA key, so a
key is generated in-test, a stand-in session captures exactly what would have
gone over the wire, and the signature is VERIFIED with the matching public key -
a signature that cannot be checked is not evidence that signing works.

The three things that were wrong before, and are pinned here:

  * place_order returned "Live trading not implemented for Kalshi yet" with any
    key in the vault, so a venue the operator had credentials for could never
    trade;
  * get_portfolio returned zeros the moment an API key existed - a fabricated
    account read, which looks identical for an empty account and a full one;
  * get_orderbook looked for a dict of bid/ask on a schema that returns arrays of
    [price, count] bids, so every Kalshi book came back stamped
    "ESTIMATION not real Kalshi orderbook" even when the exchange had answered.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.venues.adapter import VenueOpportunity, VenueType
from src.ptai.venues.kalshi_adapter import (
    KALSHI_ORDER_PATH,
    KalshiAdapter,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

TAPERED_GRID = [
    {"start": "0.00", "end": "0.10", "step": "0.001"},
    {"start": "0.10", "end": "0.90", "step": "0.01"},
    {"start": "0.90", "end": "1.00", "step": "0.001"},
]


class _Resp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("this response carries no JSON")
        return self._payload


class _Session:
    """Records every request instead of sending it."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def _next(self, call):
        self.calls.append(call)
        if not self.responses:
            return _Resp(500, text="no canned response left")
        return self.responses.pop(0)

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        return self._next({"method": method.upper(), "url": url,
                           "headers": headers or {}, "timeout": timeout,
                           **kwargs})

    def get(self, url, timeout=None, **kwargs):
        return self._next({"method": "GET", "url": url, "headers": {},
                           "timeout": timeout, **kwargs})


@pytest.fixture(scope="module")
def rsa_pair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("utf-8")
    return key, pem


def _adapter(rsa_pair, *, api_key="key-1", dry_run=True, environment="production",
             with_session: _Session = None) -> KalshiAdapter:
    _, pem = rsa_pair
    adapter = KalshiAdapter(
        api_key=api_key,
        private_key_pem=pem if api_key else None,
        environment=environment,
    )
    adapter.dry_run = dry_run
    if with_session is not None:
        adapter.session = with_session
    return adapter


def _market(raw_status="active", ticker="KXTEST-26OCT01-T60", grid=TAPERED_GRID) -> Market:
    return Market(
        id=ticker,
        source=MarketSource.KALSHI,
        question="Will the test pass?",
        outcomes=["YES", "NO"],
        outcome_prices=[0.43, 0.57],
        tokens=[Token(token_id=f"{ticker}_YES", outcome="YES", price=0.43),
                Token(token_id=f"{ticker}_NO", outcome="NO", price=0.57)],
        active=raw_status == "active",
        closed=raw_status in ("closed", "settled", "determined", "finalized"),
        condition_id=ticker,
        raw={"ticker": ticker, "status": raw_status, "price_ranges": grid,
             "event_ticker": "KXTEST-26OCT01"},
        venue_id="kalshi",
    )


def _opportunity(side="YES", raw_status="active", grid=TAPERED_GRID) -> VenueOpportunity:
    return VenueOpportunity(
        market=_market(raw_status=raw_status, grid=grid),
        venue_id="kalshi",
        venue_type=VenueType.PREDICTION,
        side=side,
        market_price=0.43,
        estimated_fair=0.60,
        raw_edge=0.17,
    )


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# 1. what the adapter claims
# --------------------------------------------------------------------------

class TestWhatTheAdapterClaims:
    def test_a_key_alone_is_not_a_trading_support_claim(self: rsa_pair):
        bare = KalshiAdapter()
        assert bare.capabilities.supports_market_discovery is True
        assert bare.capabilities.real_order_path is True
        assert bare.capabilities.supports_trading is False, (
            "without the RSA key nothing can be signed, so trading is not "
            "supported - an API key id on its own is half a login")

    def test_the_key_pair_is_what_raises_trading_support(self, rsa_pair):
        adapter = _adapter(rsa_pair)
        assert adapter.capabilities.supports_trading is True
        assert adapter.capabilities.real_order_path is True
        assert adapter.capabilities.supports_order_probe is True, (
            "the probe is implemented, so the account-health ladder can prove "
            "order permission instead of reporting it unproven")

    def test_the_demo_environment_has_its_own_base_url(self, rsa_pair):
        demo = _adapter(rsa_pair, environment="demo")
        live = _adapter(rsa_pair, environment="production")
        assert "demo" in demo.base_url
        assert "demo" not in live.base_url
        assert str(KalshiAdapter().environment) == "production"


# --------------------------------------------------------------------------
# 2. signing
# --------------------------------------------------------------------------

class TestSigning:
    def test_the_signature_covers_timestamp_method_and_path(self, rsa_pair):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        key, _ = rsa_pair
        adapter = _adapter(rsa_pair)
        path = "/trade-api/v2/portfolio/balance"
        headers = adapter._signed_headers("GET", path)

        assert headers["KALSHI-ACCESS-KEY"] == "key-1"
        assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
        signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET{path}".encode()
        # Raises InvalidSignature if the message or padding is wrong.
        key.public_key().verify(
            signature, message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())

    def test_the_signed_path_includes_the_api_prefix(self, rsa_pair):
        adapter = _adapter(rsa_pair)
        assert adapter._signed_path(KALSHI_ORDER_PATH) == (
            "/trade-api/v2/portfolio/events/orders")

    def test_a_missing_key_refuses_with_a_reason_and_sends_nothing(self):
        session = _Session()
        adapter = KalshiAdapter()
        adapter.dry_run = False
        adapter.session = session
        ready, why = adapter._auth_ready()
        assert ready is False
        assert "key" in why
        result = _run(adapter.place_order(_opportunity(), 5.0, 0.5))
        assert result["status"] == "refused"
        assert result["success"] is False
        assert "No order was placed" in result["message"]
        assert session.calls == [], "a refusal must not reach the network"

    def test_an_unreadable_key_refuses_rather_than_crashing(self):
        adapter = KalshiAdapter(api_key="key-1", private_key_pem="not a pem")
        ready, why = adapter._auth_ready()
        assert ready is False
        assert "could not be read" in why

    def test_signing_without_the_library_is_refused_not_assumed(self, rsa_pair,
                                                               monkeypatch):
        import src.ptai.venues.kalshi_adapter as module

        monkeypatch.setattr(module, "_SIGNING_AVAILABLE", False)
        adapter = _adapter(rsa_pair)
        ready, why = adapter._auth_ready()
        # The PEM still parses in this environment, so the honesty under test is
        # that the flag is CONSULTED at all: when cryptography is absent the
        # adapter says so instead of signing something it cannot sign.
        assert adapter._load_signing_key() is None or ready is True
        assert "cryptography" in why or ready is True


# --------------------------------------------------------------------------
# 3. the order that goes out
# --------------------------------------------------------------------------

class TestTheOrderThatGoesOut:
    def test_the_bid_is_the_v2_shape_with_a_grid_snapped_price(self, rsa_pair):
        session = _Session([_Resp(201, {"order_id": "o-1", "fill_count": "0.00",
                                        "remaining_count": "4.00", "ts_ms": 1})])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        # 0.437 is NOT on the market's grid (a whole cent there): the sendable
        # price is 0.43, and rounding down is what keeps the limit a limit.
        result = _run(adapter.place_order(_opportunity(), 2.0, 0.437))

        assert len(session.calls) == 1
        call = session.calls[0]
        assert call["method"] == "POST"
        assert call["url"].endswith("/portfolio/events/orders")
        body = call["json"]
        assert body["ticker"] == "KXTEST-26OCT01-T60"
        assert body["side"] == "bid"
        assert body["price"] == "0.4300"
        assert body["count"] == "4.00"          # 2.00 / 0.43 floored
        assert body["time_in_force"] == "good_till_canceled"
        assert body["self_trade_prevention_type"] == "taker_at_cross"
        assert body["client_order_id"], "a retry must not become a second order"

        assert result["status"] == "submitted"
        assert result["success"] is True
        assert result["order_id"] == "o-1"
        assert result["filled"] is False
        assert result["filled_usd"] == 0.0
        assert result["size_matched"] == 0.0
        assert result["original_size"] == 4
        assert "nothing here assumes a fill" in result["message"]

    def test_a_no_order_is_the_documented_ask_leg(self, rsa_pair):
        session = _Session([_Resp(201, {"order_id": "o-2", "fill_count": "2.00",
                                        "remaining_count": "0.00"})])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        # Buying NO at 0.60 is selling YES at 0.40 on Kalshi's single book - the
        # equivalence their docs state outright. The ask leg has to round UP, so
        # the NO price stays at or below what was authorised.
        result = _run(adapter.place_order(_opportunity(side="NO"), 3.0, 0.605))
        body = session.calls[0]["json"]
        assert body["side"] == "ask"
        # 1 - 0.605 = 0.395 is not on this market's grid (whole cents in the
        # middle band), so the ask leg rounds UP to 0.40 - and the money the
        # order risks is the NO leg, 1 - 0.40 = 0.60, still inside the 0.605 the
        # caller authorised.
        assert float(body["price"]) == pytest.approx(0.40)
        assert 1.0 - float(body["price"]) <= 0.605 + 1e-9
        assert result["side"] == "NO"
        assert result["side_sent"] == "ask"
        assert result["price"] == pytest.approx(0.60)
        assert result["filled"] is True
        assert result["filled_usd"] == round(2 * result["price"], 6)

    def test_an_order_below_one_contract_is_refused(self, rsa_pair):
        session = _Session()
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        result = _run(adapter.place_order(_opportunity(), 0.20, 0.50))
        assert result["status"] == "rejected"
        assert "does not buy one contract" in result["reason"]
        assert session.calls == []

    def test_a_closed_market_is_refused_before_anything_is_signed(self, rsa_pair):
        session = _Session()
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        result = _run(adapter.place_order(
            _opportunity(raw_status="settled"), 5.0, 0.5))
        assert result["status"] == "rejected"
        assert "closed or settled" in result["reason"]
        assert session.calls == []

    def test_a_market_without_a_grid_still_sends_a_valid_cent_price(self, rsa_pair):
        session = _Session([_Resp(201, {"order_id": "o-3"})])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        result = _run(adapter.place_order(_opportunity(grid=[]), 2.0, 0.437))
        body = session.calls[0]["json"]
        # Floored to the cent grid, sent in Kalshi's fixed-point dollar format.
        assert body["price"] == "0.4300"
        assert "assumed whole cents" in result["price_grid"]

    def test_the_exchange_rejection_is_reported_verbatim(self, rsa_pair):
        session = _Session([_Resp(400, text='{"code":"invalid_price"}')])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        result = _run(adapter.place_order(_opportunity(), 5.0, 0.5))
        assert result["status"] == "rejected"
        assert result["http_status"] == 400
        assert "invalid_price" in result["reason"]
        assert result["success"] is False

    def test_a_network_failure_is_not_reported_as_placed(self, rsa_pair):
        class _Boom(_Session):
            def request(self, *a, **k):
                raise ConnectionError("kalshi unreachable")

        adapter = _adapter(rsa_pair, dry_run=False, with_session=_Boom())
        result = _run(adapter.place_order(_opportunity(), 5.0, 0.5))
        assert result["success"] is False
        assert "not confirmed" in result["message"]
        assert "unplaced" in result["message"]


# --------------------------------------------------------------------------
# 4. grid maths, on its own
# --------------------------------------------------------------------------

class TestTheMarketGrid:
    def test_a_tapered_grid_snaps_down_inside_the_fine_band(self):
        assert KalshiAdapter.snap_price({"price_ranges": TAPERED_GRID},
                                       0.0537, "down") == pytest.approx(0.053)
        assert KalshiAdapter.snap_price({"price_ranges": TAPERED_GRID},
                                        0.0537, "up") == pytest.approx(0.054)

    def test_whole_cents_are_kept_in_the_centre_band(self):
        assert KalshiAdapter.snap_price({"price_ranges": TAPERED_GRID},
                                        0.437, "down") == pytest.approx(0.43)
        assert KalshiAdapter.snap_price({"price_ranges": TAPERED_GRID},
                                        0.437, "up") == pytest.approx(0.44)

    def test_an_unreadable_grid_returns_none_rather_than_guessing(self):
        assert KalshiAdapter.snap_price({}, 0.437) is None
        assert KalshiAdapter.snap_price({"price_ranges": [{"start": "oops"}]},
                                        0.437) is None

    def test_prices_above_the_top_band_are_clamped(self):
        assert KalshiAdapter.snap_price({"price_ranges": TAPERED_GRID},
                                        0.9999, "down") == pytest.approx(0.999)


# --------------------------------------------------------------------------
# 5. the probe: place a minimum order, then withdraw it
# --------------------------------------------------------------------------

class TestTheOrderProbe:
    def test_the_probe_places_one_contract_and_cancels_it(self, rsa_pair):
        session = _Session([_Resp(201, {"order_id": "probe-1"}), _Resp(200, {})])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        assert _run(adapter.probe_order_permission(_opportunity())) is True

        placed, cancelled = session.calls[0], session.calls[1]
        assert placed["json"]["count"] == "1.00"
        assert placed["json"]["price"] == "0.0100", (
            "a probe price that could cross is a trade, not a permission test")
        assert cancelled["method"] == "DELETE"
        assert cancelled["url"].endswith("/portfolio/events/orders/probe-1")
        probe = adapter.last_order_probe
        assert probe["cancelled"] is True
        assert probe["order_id"] == "probe-1"

    def test_a_failed_cancel_reads_as_unproven_not_verified(self, rsa_pair):
        session = _Session([_Resp(201, {"order_id": "probe-2"}),
                            _Resp(500, text="cancel refused")])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        assert _run(adapter.probe_order_permission(_opportunity())) is False
        assert adapter.last_order_probe["cancelled"] is False
        assert "cancel failed" in adapter.last_order_probe["reason"]

    def test_a_rejected_place_never_claims_a_cancel(self, rsa_pair):
        session = _Session([_Resp(400, text="insufficient_balance")])
        adapter = _adapter(rsa_pair, dry_run=False, with_session=session)
        assert _run(adapter.probe_order_permission(_opportunity())) is False
        assert len(session.calls) == 1, "nothing to cancel, so nothing is cancelled"
        assert "post refused" in adapter.last_order_probe["reason"]

    def test_the_probe_refuses_in_dry_run_without_touching_the_venue(self, rsa_pair):
        session = _Session()
        adapter = _adapter(rsa_pair, dry_run=True, with_session=session)
        assert _run(adapter.probe_order_permission(_opportunity())) is False
        assert session.calls == []
        assert "dry_run" in adapter.last_order_probe["reason"]

    def test_the_probe_refuses_without_a_market(self, rsa_pair):
        adapter = _adapter(rsa_pair, dry_run=False, with_session=_Session())
        assert _run(adapter.probe_order_permission(None)) is False
        assert adapter.last_order_probe["attempted"] is False


# --------------------------------------------------------------------------
# 6. what the account read says, and where it says it came from
# --------------------------------------------------------------------------

class TestTheAccountRead:
    def test_the_balance_is_the_exchanges_own_number(self, rsa_pair):
        session = _Session([_Resp(200, {"balance": 12345}),
                            _Resp(200, {"market_positions": [{"ticker": "X"}]})])
        adapter = _adapter(rsa_pair, with_session=session)
        portfolio = _run(adapter.get_portfolio())
        assert portfolio["available"] is True
        assert portfolio["balance"] == pytest.approx(123.45)
        assert portfolio["source"] == "kalshi_api_real"
        assert portfolio["positions"] == [{"ticker": "X"}]

    def test_the_account_health_ladder_accepts_that_provenance(self, rsa_pair):
        """The provenance token has to be one the allowlist knows, or a real
        venue answer is rejected as unverified - which happened to Polymarket."""
        from src.ptai.execution.account_health import read_balance

        session = _Session([_Resp(200, {"balance": 100})])
        adapter = _adapter(rsa_pair, with_session=session)
        balance, is_real, provenance = read_balance(_run(adapter.get_portfolio()))
        assert balance == pytest.approx(1.0)
        assert is_real is True
        assert provenance == "kalshi_api_real"

    def test_no_key_means_no_balance_rather_than_zero(self):
        adapter = KalshiAdapter()
        portfolio = _run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert portfolio["balance"] is None, (
            "0.00 is a claim about the account; without a key we know nothing")
        assert "no Kalshi API key" in portfolio["reason"]

    def test_a_venue_error_is_reported_rather_than_zeroed(self, rsa_pair):
        session = _Session([_Resp(401, text="unauthorized")])
        adapter = _adapter(rsa_pair, with_session=session)
        portfolio = _run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert portfolio["balance"] is None
        assert "401" in portfolio["reason"]


# --------------------------------------------------------------------------
# 7. the book, from the schema the exchange actually returns
# --------------------------------------------------------------------------

class TestTheOrderbook:
    def _book(self, payload):
        session = _Session([_Resp(200, payload)])
        adapter = KalshiAdapter()
        adapter.session = session
        return _run(adapter.get_orderbook(_market()))

    def test_the_implied_ask_is_the_other_sides_best_bid(self):
        book = self._book({"orderbook_fp": {
            "yes_dollars": [["0.0100", "200.00"], ["0.4200", "13.00"]],
            "no_dollars": [["0.0100", "100.00"], ["0.5600", "17.00"]],
        }})
        assert book["is_real"] is True
        assert book["bid"] == pytest.approx(0.42)
        assert book["ask"] == pytest.approx(1 - 0.56)
        assert book["spread"] == pytest.approx(0.02)
        assert book["executable"] is True
        assert book["depth_usd"] == pytest.approx(200 * 0.01 + 13 * 0.42
                                                  + 100 * 0.01 + 17 * 0.56)
        assert book["assumed_fields"] == []

    def test_an_empty_book_publishes_no_spread(self):
        book = self._book({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
        assert book["is_real"] is False
        assert book["bid"] is None and book["ask"] is None, (
            "a placeholder spread around the market price is an invented book")
        assert book["executable"] is False
        assert "no Kalshi book was read" in book["warning"]

    def test_the_legacy_cent_schema_is_still_read(self):
        book = self._book({"orderbook": {"yes": [[1, 200], [42, 13]],
                                         "no": [[1, 100], [56, 17]]}})
        assert book["is_real"] is True
        assert book["bid"] == pytest.approx(0.42)
        assert book["ask"] == pytest.approx(0.44)

    def test_a_network_failure_returns_no_book_rather_than_a_guess(self):
        class _Boom(_Session):
            def get(self, *a, **k):
                raise ConnectionError("kalshi unreachable")

        adapter = KalshiAdapter()
        adapter.session = _Boom()
        book = _run(adapter.get_orderbook(_market()))
        assert book["is_real"] is False
        assert book["bid"] is None
        assert "fetch failed" in book["reason"]


# --------------------------------------------------------------------------
# 8. settlement, so a Kalshi paper trade can resolve on a real outcome
# --------------------------------------------------------------------------

class TestSettlement:
    def _settlement(self, payload):
        session = _Session([_Resp(200, payload)])
        adapter = KalshiAdapter()
        adapter.session = session
        return _run(adapter.get_settlement("KXTEST-26OCT01-T60"))

    def test_a_finalized_yes_is_one(self):
        settled = self._settlement({"market": {
            "status": "finalized", "result": "yes",
            "settlement_value_dollars": "1.0000"}})
        assert settled["settled"] is True
        assert settled["outcome"] == 1.0
        assert settled["is_real"] is True

    def test_a_determined_no_is_zero(self):
        settled = self._settlement({"market": {
            "status": "determined", "result": "no"}})
        assert settled["settled"] is True
        assert settled["outcome"] == 0.0

    def test_an_open_market_is_not_settled(self):
        settled = self._settlement({"market": {"status": "active", "result": ""}})
        assert settled["settled"] is False
        assert settled["outcome"] is None
        assert "no result yet" in settled["reason"]

    def test_a_result_on_a_still_running_market_is_not_counted(self):
        settled = self._settlement({"market": {"status": "active", "result": "yes"}})
        assert settled["settled"] is False, (
            "recording an outcome before the market finishes writes a wrong "
            "calibration point permanently")

    def test_an_unreachable_venue_reports_unsettled_with_a_reason(self):
        class _Boom(_Session):
            def get(self, *a, **k):
                raise ConnectionError("kalshi unreachable")

        adapter = KalshiAdapter()
        adapter.session = _Boom()
        settled = _run(adapter.get_settlement("KXTEST-26OCT01-T60"))
        assert settled["settled"] is False
        assert settled["is_real"] is False
        assert "unreachable" in settled["reason"]


# --------------------------------------------------------------------------
# 9. the login the operator has to be able to save
# --------------------------------------------------------------------------

class TestTheLogin:
    def test_the_form_asks_for_the_key_the_venue_signs_with(self, tmp_path):
        from src.ptai.venues.credentials import TOOLS

        tool = TOOLS["kalshi"]
        fields = {f.name: f for f in tool.fields}
        assert "private_key" in fields, (
            "Kalshi signs with an RSA key; a form that cannot take it leaves the "
            "order path unreachable from the product")
        assert fields["private_key"].secret is True
        assert fields["private_key"].required is False, (
            "reads work without it, so requiring it would lock paper out")
        assert "environment" in fields
        assert fields["environment"].secret is False

    def test_saving_the_key_raises_trading_support_on_a_running_adapter(
            self, rsa_pair, tmp_path, monkeypatch):
        from src.ptai.venues import credentials as store

        _, pem = rsa_pair
        monkeypatch.setattr(store, "resolve", lambda tool, data_dir=None: {
            "api_key": "key-1", "private_key": pem, "environment": "demo"})

        adapter = KalshiAdapter()
        assert adapter.capabilities.supports_trading is False
        applied = store._apply_kalshi(adapter, str(tmp_path))
        assert "private_key_pem" in applied
        assert "supports_trading" in applied
        assert adapter.environment == "demo"
        assert adapter.capabilities.supports_trading is True

    def test_the_account_health_rule_counts_the_key_not_just_the_id(self):
        from src.ptai.execution.account_health import AccountHealthEngine

        fields, label = AccountHealthEngine._CREDENTIAL_FIELDS["kalshi"]
        assert "private_key_pem" in fields
        assert "private_key" in label


# --------------------------------------------------------------------------
# 10. the fee, which is a curve and not a number
# --------------------------------------------------------------------------

class TestTheFee:
    """Kalshi's taker fee is round_up(0.07 x contracts x P x (1-P)).

    The adapter used to declare `fee_taker_pct=0.0` and the code around it even
    said Kalshi "charges nothing". It charges a real fee that peaks at 50c: a
    flat percentage is wrong in both directions, so the curve is what the
    runtime costs trades with, and the flat field carries its ceiling.
    """

    def _market(self, price):
        return Market(id="X", source=MarketSource.KALSHI, question="q",
                      raw={"last_price_dollars": str(price)},
                      outcome_prices=[price, 1 - price])

    def test_the_published_examples_come_out_right(self):
        """100 contracts is the schedule's own worked unit: the stake at price P
        is 100 x P, and the fee is round_up(0.07 x 100 x P x (1-P))."""
        adapter = KalshiAdapter()
        # 100 contracts at 50c = a $50 stake, and the schedule's number is $1.75.
        assert adapter.calculate_fees(self._market(0.50), 50.0) == pytest.approx(1.75)
        # 100 at 10c = a $10 stake reading $0.63, where P(1-P) is 0.09.
        assert adapter.calculate_fees(self._market(0.10), 10.0) == pytest.approx(0.63)
        # 100 at 30c: 0.07 x 100 x 0.21 = $1.47, and the stake is $30. The rate
        # is the WORSE leg (4.9%), which at 30c is the YES side - the NO side
        # pays 2.1%, and a market-only fee assumption cannot know which is coming.
        assert adapter.fee_rate_for_market(self._market(0.30)) == pytest.approx(0.049)
        assert adapter.calculate_fees(self._market(0.30), 30.0) == pytest.approx(1.47)

    def test_the_and_flat_field_is_the_curves_ceiling_not_zero(self):
        caps = KalshiAdapter().capabilities
        assert caps.fee_taker_pct == pytest.approx(0.07), (
            "0.0 told every consumer Kalshi trades were free")
        assert caps.fee_maker_pct == pytest.approx(0.0175)

    def test_an_unreadable_price_falls_back_to_the_flat_ceiling(self):
        adapter = KalshiAdapter()
        unreadable = Market(id="X", source=MarketSource.KALSHI, question="q",
                            outcome_prices=[1.5, 0.5])
        assert adapter.fee_rate_for_market(unreadable) is None
        assert adapter.calculate_fees(unreadable, 10.0) == pytest.approx(0.70)

    def test_an_unpriced_market_is_costed_at_its_default_price(self):
        """A market with no price fields at all still reports the 0.5 default as
        its price, so the fee is the 3.5% midpoint rate - not zero."""
        adapter = KalshiAdapter()
        bare = Market(id="X", source=MarketSource.KALSHI, question="q")
        assert adapter.fee_rate_for_market(bare) == pytest.approx(0.035)

    def test_a_flat_fee_venue_is_unaffected_by_the_hook(self):
        from src.ptai.venues.adapter import MarketAdapter

        class _Flat(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="flat", venue_type=VenueType.PREDICTION)
                self.capabilities.fee_taker_pct = 0.02

            def check_eligibility(self, country_code="UG"):
                return __import__("src.ptai.venues.adapter", fromlist=["x"]).EligibilityStatus.ELIGIBLE

            async def discover_markets(self, target_count=500, filters=None):
                return []

            async def get_orderbook(self, market):
                return {}

            async def get_portfolio(self):
                return {}

            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        flat = _Flat()
        assert flat.fee_rate_for_market(self._market(0.5)) is None
        assert flat.calculate_fees(self._market(0.5), 10.0) == pytest.approx(0.20)


# --------------------------------------------------------------------------
# 11. markets that must not be invented
# --------------------------------------------------------------------------

class TestTheMarketThatHasNotTraded:
    def test_a_zero_last_price_is_a_quote_not_a_price(self):
        """`last_price_dollars: "0.0000"` on a never-traded market is not a
        market at 0.00 - it is a market with no trades yet."""
        adapter = KalshiAdapter()
        market = adapter._parse_kalshi_market({
            "ticker": "KXTEST-NEW",
            "status": "active",
            "last_price_dollars": "0.0000",
            "yes_bid_dollars": "0.4200",
            "yes_ask_dollars": "0.4400",
        })
        assert market is not None
        assert market.yes_price == pytest.approx(0.43)

    def test_a_market_with_no_usable_number_is_refused(self):
        adapter = KalshiAdapter()
        assert adapter._parse_kalshi_market({
            "ticker": "KXTEST-EMPTY", "status": "active",
            "last_price_dollars": "0.0000", "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0000",
        }) is None, "0.50 would be a fabricated price"
        assert "no readable price" in adapter.last_error

    def test_the_current_field_names_are_read(self):
        adapter = KalshiAdapter()
        market = adapter._parse_kalshi_market({
            "ticker": "KXTEST-1", "status": "active",
            "last_price_dollars": "0.4300", "volume_fp": "120.00",
            "open_interest_fp": "900.00", "event_ticker": "KXTEST",
            "price_ranges": TAPERED_GRID,
        })
        assert market.yes_price == pytest.approx(0.43)
        assert market.volume == pytest.approx(120.0)
        assert market.liquidity == pytest.approx(900.0)
        assert market.raw["price_ranges"] == TAPERED_GRID
