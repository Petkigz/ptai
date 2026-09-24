"""
Tests for the historical data provider - the input the V9 backtest gate requires.

The provider exists because the gate refused synthetic data without anything
supplying real data, which left the backtester unreachable. These tests cover
the parse-and-validate rules, the entry-price/settlement distinction, the
honest baseline, and the refusal to invent data when the API is down.

All fetching is injected, so the real code path runs without a network.
"""
import json

import pytest

from src.ptai.backtest import HistoricalDataProvider, ResolvedMarket


def _row(**over):
    """A closed Gamma row: outcomes/outcomePrices arrive as JSON strings."""
    row = {
        "id": "1",
        "question": "Will X happen?",
        "closed": True,
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["1","0"]',
        "lastTradePrice": "0.62",
        "volume": "500000",
        "liquidity": "120000",
        "endDate": "2026-01-01T00:00:00Z",
        "category": "crypto",
    }
    row.update(over)
    return row


def _provider(rows):
    return HistoricalDataProvider(fetch=lambda url, params: rows,
                                  cache_dir="/tmp/ptai_hist_cache")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parses_a_normal_closed_row(self):
        m = _provider([_row()]).fetch_resolved_markets(use_cache=False)
        assert len(m) == 1
        assert m[0].question == "Will X happen?"
        assert m[0].entry_price == 0.62
        assert m[0].outcome == 1
        assert m[0].settlement_price == 1.0
        assert m[0].volume_usd == 500_000

    def test_outcome_strings_are_decoded(self):
        """Gamma encodes outcomes as a JSON string, not an array."""
        m = _provider([_row()]).fetch_resolved_markets(use_cache=False)
        assert m[0].settlement_price == 1.0

    def test_plain_lists_also_accepted(self):
        """If a feed ever sends real arrays, that must still work."""
        m = _provider([_row(outcomes=["Yes", "No"], outcomePrices=[0, 1],
                            lastTradePrice=0.4)])
        got = m.fetch_resolved_markets(use_cache=False)
        assert got[0].outcome == 0

    def test_no_row_is_dropped_on_open_market(self):
        assert _provider([_row(closed=False)]).fetch_resolved_markets(use_cache=False) == []

    def test_unsettled_price_is_not_an_outcome(self):
        """
        A market closing at 0.50 voided or has not settled. Treating it as an
        outcome would fabricate a result.
        """
        row = _row(outcomePrices='["0.5","0.5"]', lastTradePrice="0.5")
        assert _provider([row]).fetch_resolved_markets(use_cache=False) == []

    def test_mismatched_outcome_arrays_rejected(self):
        row = _row(outcomes='["Yes","No"]', outcomePrices='["1"]')
        assert _provider([row]).fetch_resolved_markets(use_cache=False) == []

    def test_malformed_json_rejected_not_raised(self):
        assert _provider([_row(outcomes="not json")]).fetch_resolved_markets(use_cache=False) == []

    def test_no_tradeable_price_rejected(self):
        """
        Without a last trade or a book there is no price to backtest against.
        Falling back to the settlement price would mean filling at 1.00.
        """
        row = _row(lastTradePrice=None)
        row.pop("lastTradePrice")
        assert _provider([row]).fetch_resolved_markets(use_cache=False) == []

    def test_falls_back_to_final_book_midpoint(self):
        row = _row(bestBid="0.55", bestAsk="0.59")
        row.pop("lastTradePrice")
        m = _provider([row]).fetch_resolved_markets(use_cache=False)
        assert m[0].entry_price == pytest.approx(0.57, abs=1e-9)

    def test_end_date_parsed_into_resolved_at(self):
        m = _provider([_row()]).fetch_resolved_markets(use_cache=False)
        assert m[0].resolved_at is not None
        assert m[0].resolved_at.year == 2026

    def test_bad_end_date_does_not_drop_the_market(self):
        m = _provider([_row(endDate="nonsense")]).fetch_resolved_markets(use_cache=False)
        assert len(m) == 1
        assert m[0].resolved_at is None


# ---------------------------------------------------------------------------
# Entry price vs settlement
# ---------------------------------------------------------------------------

class TestEntryPriceIsNotSettlement:
    def test_settlement_of_one_is_not_usable_as_an_entry(self):
        """
        The first version of this module used outcomePrices as the entry price.
        A resolved market settles at exactly 1 or 0, so every market failed
        validation - and "buy YES at $1.00" is not a trade anyway.
        """
        m = ResolvedMarket(market_id="1", question="q", entry_price=1.0,
                           outcome=1, volume_usd=900_000, liquidity_usd=90_000,
                           settlement_price=1.0)
        assert m.is_usable is False

    def test_zero_entry_not_usable(self):
        m = ResolvedMarket(market_id="1", question="q", entry_price=0.0,
                           outcome=0, volume_usd=900_000, liquidity_usd=90_000,
                           settlement_price=0.0)
        assert m.is_usable is False

    def test_normal_entry_is_usable(self):
        m = ResolvedMarket(market_id="1", question="q", entry_price=0.62,
                           outcome=1, volume_usd=900_000, liquidity_usd=90_000,
                           settlement_price=1.0)
        assert m.is_usable is True

    def test_unsettled_settlement_blocks_use(self):
        m = ResolvedMarket(market_id="1", question="q", entry_price=0.5,
                           outcome=1, volume_usd=900_000, liquidity_usd=90_000,
                           settlement_price=0.5)
        assert m.is_usable is False

    def test_thin_volume_blocks_use(self):
        m = ResolvedMarket(market_id="1", question="q", entry_price=0.5,
                           outcome=1, volume_usd=10, liquidity_usd=10,
                           settlement_price=1.0)
        assert m.is_usable is False


# ---------------------------------------------------------------------------
# Volume and liquidity floors
# ---------------------------------------------------------------------------

class TestFloors:
    def test_low_volume_market_excluded_from_dataset(self):
        rows = [_row(id="1", volume="50", liquidity="10"),
                _row(id="2", volume="500000", liquidity="120000")]
        d = _provider(rows).build_dataset(markets=_provider(rows).fetch_resolved_markets(use_cache=False))
        assert len(d.rows) == 1
        assert d.rejected == 1

    def test_thin_liquidity_excluded(self):
        rows = [_row(id="1", volume="500000", liquidity="100")]
        p = _provider(rows)
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False))
        assert d.rows == []
        assert any("liquidity floor" in w for w in d.warnings)

    def test_floors_are_configurable(self):
        rows = [_row(id="1", volume="5000", liquidity="2000")]
        p = HistoricalDataProvider(fetch=lambda u, pa: rows, cache_dir="/tmp/h",
                                   min_volume=1000, min_liquidity=500)
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False))
        assert len(d.rows) == 1

    def test_report_states_the_floors(self):
        rep = _provider([]).get_report()
        assert rep["floors"]["min_volume_usd"] == 1000.0
        assert "min_liquidity_usd" in rep["floors"]


# ---------------------------------------------------------------------------
# Signal modes
# ---------------------------------------------------------------------------

class TestSignalModes:
    def _book(self):
        return [_row(id="1", lastTradePrice="0.62", outcomePrices='["1","0"]'),
                _row(id="2", lastTradePrice="0.41", outcomePrices='["0","1"]')]

    def test_no_signal_is_a_baseline_with_zero_edge(self):
        """
        Without a signal the honest result is zero edge, not an invented one.
        """
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False))
        assert d.signal_mode == "none"
        assert d.is_baseline is True
        assert all(r["edge"] == 0.0 for r in d.rows)
        assert any("no signal supplied" in w for w in d.warnings)

    def test_recorded_fair_values_produce_real_edges(self):
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            recorded_fair_values={"1": 0.78, "2": 0.30})
        assert d.signal_mode == "recorded"
        assert d.is_baseline is False
        edges = {r["market_id"]: r["edge"] for r in d.rows}
        assert edges["1"] == pytest.approx(0.16, abs=1e-9)
        assert edges["2"] == pytest.approx(-0.11, abs=1e-9)

    def test_recorded_mode_skips_markets_without_a_fair_value(self):
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            recorded_fair_values={"1": 0.78})
        assert len(d.rows) == 1
        assert d.rejected == 1

    def test_callable_signal_used(self):
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            signal=lambda m: 0.9)
        assert d.signal_mode == "callable"
        assert d.is_baseline is False
        assert all(r["fair_value"] == 0.9 for r in d.rows)

    def test_callable_returning_none_rejects_the_market(self):
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            signal=lambda m: None)
        assert d.rows == []
        assert d.rejected == 2

    def test_callable_returning_an_impossible_price_is_rejected(self):
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            signal=lambda m: 1.4)
        assert d.rows == []

    def test_callable_that_raises_is_reported_not_fatal(self):
        def boom(m):
            raise RuntimeError("signal blew up")
        p = _provider(self._book())
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False), signal=boom)
        assert d.rows == []
        assert any("RuntimeError" in w for w in d.warnings)


# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------

class TestRowShape:
    def _one(self):
        p = _provider([_row()])
        return p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                               recorded_fair_values={"1": 0.78}).rows[0]

    def test_row_carries_everything_the_engine_reads(self):
        row = self._one()
        for key in ("day", "question", "market_price", "fair_value", "edge",
                    "actual_outcome", "confidence", "bid", "ask", "spread",
                    "depth", "fee_pct", "venue_id"):
            assert key in row, key

    def test_row_is_marked_not_synthetic(self):
        row = self._one()
        assert row["is_synthetic"] is False
        assert row["data_mode"] == "historical"

    def test_bid_ask_straddle_the_entry_price(self):
        row = self._one()
        assert row["bid"] < row["market_price"] < row["ask"]
        assert row["spread"] == pytest.approx(row["ask"] - row["bid"], abs=1e-9)

    def test_assumed_cost_fields_are_declared_not_hidden(self):
        """
        Historical spread and depth are not published. The row must say so
        rather than presenting an assumed cost as a measured one.
        """
        row = self._one()
        assert "spread" in row["cost_fields_assumed"]

    def test_dataset_warns_that_costs_are_assumed(self):
        p = _provider([_row()])
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            recorded_fair_values={"1": 0.78})
        assert any("assumed" in w for w in d.warnings)

    def test_rows_are_ordered_oldest_first(self):
        """The equity curve must run forward in time."""
        rows = [_row(id="1", endDate="2026-05-01T00:00:00Z"),
                _row(id="2", endDate="2026-01-01T00:00:00Z"),
                _row(id="3", endDate="2026-03-01T00:00:00Z")]
        p = _provider(rows)
        d = p.build_dataset(markets=p.fetch_resolved_markets(use_cache=False),
                            recorded_fair_values={"1": 0.8, "2": 0.8, "3": 0.8})
        assert [r["market_id"] for r in d.rows] == ["2", "3", "1"]
        assert [r["day"] for r in d.rows] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Failure and caching
# ---------------------------------------------------------------------------

class TestFailure:
    def test_unreachable_api_yields_no_rows(self):
        """
        It must not fall back to generating markets - that is exactly what the
        backtest gate exists to prevent.
        """
        p = HistoricalDataProvider(fetch=lambda u, pa: None, cache_dir="/tmp/h_none")
        d = p.build_dataset()
        assert d.ok is False
        assert d.rows == []

    def test_error_recorded_on_the_provider(self):
        def boom(url, params):
            raise OSError("network down")
        p = HistoricalDataProvider(fetch=boom, cache_dir="/tmp/h_err")
        assert p.fetch_resolved_markets(use_cache=False) == []
        assert "network down" in p.last_error

    def test_non_list_response_is_an_error_not_a_crash(self):
        p = HistoricalDataProvider(fetch=lambda u, pa: {"error": "rate limited"},
                                   cache_dir="/tmp/h_dict")
        assert p.fetch_resolved_markets(use_cache=False) == []
        assert p.last_error

    def test_report_exposes_the_last_error(self):
        p = HistoricalDataProvider(fetch=lambda u, pa: None, cache_dir="/tmp/h_rep")
        p.fetch_resolved_markets(use_cache=False)
        assert p.get_report()["last_error"]


class TestCache:
    def test_second_fetch_served_from_cache(self, tmp_path):
        calls = {"n": 0}

        def counting(url, params):
            calls["n"] += 1
            return [_row()]

        p = HistoricalDataProvider(fetch=counting, cache_dir=str(tmp_path))
        first = p.fetch_resolved_markets(use_cache=True)
        second = p.fetch_resolved_markets(use_cache=True)
        assert len(first) == 1 and len(second) == 1
        assert calls["n"] == 1, "cache should have served the second call"

    def test_cache_can_be_bypassed(self, tmp_path):
        calls = {"n": 0}

        def counting(url, params):
            calls["n"] += 1
            return [_row()]

        p = HistoricalDataProvider(fetch=counting, cache_dir=str(tmp_path))
        p.fetch_resolved_markets(use_cache=True)
        p.fetch_resolved_markets(use_cache=False)
        assert calls["n"] == 2

    def test_corrupt_cache_falls_back_to_fetch(self, tmp_path):
        p = HistoricalDataProvider(fetch=lambda u, pa: [_row()], cache_dir=str(tmp_path))
        p.fetch_resolved_markets(use_cache=True)
        for f in tmp_path.iterdir():
            f.write_text("{not json")
        p2 = HistoricalDataProvider(fetch=lambda u, pa: [_row()], cache_dir=str(tmp_path))
        assert len(p2.fetch_resolved_markets(use_cache=True)) == 1


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_is_honest_about_what_it_cannot_recover(self):
        rep = _provider([]).get_report()
        assert "not_recoverable" in rep
        assert "spread" in rep["not_recoverable"].lower()

    def test_report_documents_all_three_signal_modes(self):
        modes = _provider([]).get_report()["signal_modes"]
        assert set(modes) == {"recorded", "callable", "none"}

    def test_report_names_the_source(self):
        assert "polymarket" in _provider([]).get_report()["source"]

    def test_report_does_not_claim_mock_data(self):
        rep = json.loads(json.dumps(_provider([]).get_report()))
        assert "mock" not in rep
