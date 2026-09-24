"""
Cost provenance: telling a measured spread from an assumed one.

The spread and depth drive every risk decision - edge, slippage, expected EV,
market making - and in ten places the code defaulted to a bare `0.02` with no
way to tell it apart from a real reading.

`OrderbookAnalyzer.analyze` was worse than a default: it read

    bid = raw.get("bid", market.best_price - 0.01)
    ask = raw.get("ask", market.best_price + 0.01)
    spread = ask - bid if bid and ask else raw.get("spread", 0.02)

Because the defaults always supplied both sides, the `else` was unreachable and
the spread was ALWAYS exactly 0.02 when no book was passed - presented as a
computed value. Every downstream consumer priced a made-up number as though the
book had been read.

These tests pin the distinction.
"""
import pytest

from src.ptai.markets.base import Market, MarketSource
from src.ptai.markets.orderbook import (OrderbookAnalyzer, OrderbookSnapshot,
                                        read_spread)


def make_market(price=0.5, liquidity=10000.0, volume=20000.0):
    return Market(id="m1", source=MarketSource.POLYMARKET, question="Will X happen?",
                  outcomes=["YES", "NO"],
                  outcome_prices=[price, round(1 - price, 4)],
                  liquidity=liquidity, volume_24h=volume)


BOOK = {"bid": 0.47, "ask": 0.53, "bid_size": 900, "ask_size": 400, "depth": 5000}


@pytest.fixture
def analyzer():
    return OrderbookAnalyzer()


# ---------------------------------------------------------------------------
# Orderbook provenance
# ---------------------------------------------------------------------------

class TestOrderbookProvenance:
    def test_real_book_is_marked_as_measured(self, analyzer):
        s = analyzer.analyze(make_market(), raw_orderbook=BOOK)
        assert s.spread_source == "orderbook"
        assert s.is_assumed is False
        assert s.assumed_fields == []

    def test_real_spread_is_the_actual_difference(self, analyzer):
        s = analyzer.analyze(make_market(), raw_orderbook=BOOK)
        assert s.spread == pytest.approx(0.06)

    def test_no_book_is_marked_as_assumed(self, analyzer):
        s = analyzer.analyze(make_market())
        assert s.spread_source == "assumed_default"
        assert s.is_assumed is True

    def test_no_book_lists_every_assumed_field(self, analyzer):
        s = analyzer.analyze(make_market())
        for field in ("bid", "ask", "spread", "bid_size", "ask_size", "depth"):
            assert field in s.assumed_fields, field

    def test_the_assumed_spread_is_not_presented_as_computed(self, analyzer):
        """
        The core bug: with no book the old code always produced 0.02 via
        `ask - bid`, so it looked computed. It must now be flagged.
        """
        s = analyzer.analyze(make_market())
        assert s.spread == pytest.approx(0.02)
        assert s.spread_source == "assumed_default"
        assert s.is_assumed is True

    def test_side_defaults_stay_self_consistent(self, analyzer):
        """bid/ask derived from the mid must bracket it."""
        s = analyzer.analyze(make_market(price=0.60))
        assert s.bid < s.mid_price < s.ask
        assert s.mid_price == pytest.approx(0.60)

    def test_reported_spread_without_sides_is_its_own_source(self, analyzer):
        s = analyzer.analyze(make_market(), raw_orderbook={"spread": 0.04})
        assert s.spread == pytest.approx(0.04)
        assert s.spread_source == "reported_spread"
        # bid/ask still had to be assumed
        assert "bid/ask" in s.assumed_fields

    def test_crossed_book_is_not_accepted_as_a_measurement(self, analyzer):
        s = analyzer.analyze(make_market(), raw_orderbook={"bid": 0.55, "ask": 0.45})
        assert s.is_assumed is True
        assert s.spread_source == "assumed_default"

    def test_partial_book_reports_only_what_was_missing(self, analyzer):
        s = analyzer.analyze(make_market(), raw_orderbook={"bid": 0.48, "ask": 0.52})
        assert s.is_assumed is False
        assert "depth" in s.assumed_fields
        assert "bid" not in s.assumed_fields

    def test_a_snapshot_defaults_to_assumed(self):
        """
        Defaulting to assumed is the safe direction: an unlabelled snapshot is
        treated as untrustworthy rather than as data.
        """
        s = OrderbookSnapshot(
            market_id="m", bid=0.4, ask=0.6, spread=0.2, spread_pct=0.4,
            bid_size=1, ask_size=1, depth=2, mid_price=0.5, imbalance=0.0,
            large_orders=[], recent_trades=[], price_velocity=0.0,
            volume_24h=1, liquidity=1)
        assert s.is_assumed is True
        assert s.spread_source == "assumed_default"
        assert s.assumed_fields == []


# ---------------------------------------------------------------------------
# Slippage provenance
# ---------------------------------------------------------------------------

class TestSlippageProvenance:
    def _model(self):
        from src.ptai.execution.slippage import SlippageModel
        return SlippageModel()

    def test_no_book_estimate_is_not_executable(self):
        """
        With no ladder the estimate is modelled from liquidity, so it is an
        estimate and not a fill price.
        """
        r = self._model().estimate_slippage(
            "m1", "YES", 500.0, {"liquidity": 10000, "depth": 5000}, 0.5)
        assert r.book_source == "assumed"
        assert r.is_executable_estimate is False

    def test_ladder_walk_is_marked_executable(self):
        r = self._model().estimate_slippage(
            "m1", "YES", 500.0,
            {"asks": [(0.51, 1000), (0.53, 2000)], "bids": [(0.49, 1000)],
             "spread": 0.02, "depth": 3000}, 0.5)
        assert r.book_source == "ladder"
        assert r.is_executable_estimate is True

    def test_reported_spread_only_is_not_executable(self):
        r = self._model().estimate_slippage(
            "m1", "YES", 500.0, {"spread": 0.03, "liquidity": 8000}, 0.5)
        assert r.book_source == "reported"
        assert r.is_executable_estimate is False

    def test_estimate_defaults_to_not_executable(self):
        from src.ptai.execution.slippage import SlippageEstimate
        est = SlippageEstimate(
            market_id="m", side="YES", amount_usd=1.0, market_price=0.5,
            estimated_fill_price=0.5, slippage_pct=0.0, slippage_usd=0.0,
            liquidity_available=0.0, spread=0.0, depth=0.0,
            should_trade=False, reasoning="")
        assert est.is_executable_estimate is False

    def test_book_source_values_are_distinct(self):
        m = self._model()
        sources = {
            m.estimate_slippage("m", "YES", 100.0, {"liquidity": 5000}, 0.5).book_source,
            m.estimate_slippage("m", "YES", 100.0,
                                {"asks": [(0.5, 1000)], "bids": [(0.4, 1000)]}, 0.5).book_source,
            m.estimate_slippage("m", "YES", 100.0, {"spread": 0.03}, 0.5).book_source,
        }
        assert sources == {"assumed", "ladder", "reported"}


# ---------------------------------------------------------------------------
# Consumers can now refuse an assumed spread
# ---------------------------------------------------------------------------

class TestConsumersCanDistinguish:
    def test_flag_is_reachable_from_the_agent_context_shape(self, analyzer):
        """
        Both agent loops put a snapshot's fields into context. is_assumed must
        travel with them, otherwise the loop cannot gate on it.
        """
        s = analyzer.analyze(make_market())
        assert s.is_assumed is True
        s2 = analyzer.analyze(make_market(), raw_orderbook=BOOK)
        assert s2.is_assumed is False

    def test_market_making_can_ask_whether_the_book_was_real(self, analyzer):
        """
        Market making was quoting against an assumed 2% spread even though
        OrderbookAnalyzer was computing one. Both now agree about provenance.
        """
        from src.ptai.strategy.liquidity_rewards import LiquidityRewardsEngine
        engine = LiquidityRewardsEngine(bankroll=50.0)

        market = make_market()
        book = engine.read_book(market)
        assert book.has_book is False

        market.raw["orderbook"] = BOOK
        book2 = engine.read_book(market)
        assert book2.has_book is True
        assert book2.spread == pytest.approx(0.06)
        assert book2.source == "orderbook"


# ---------------------------------------------------------------------------
# read_spread: the shared reader
# ---------------------------------------------------------------------------

class TestReadSpread:
    """
    `orderbook.get("spread", default)` returns None when the key EXISTS with a
    None value, so a caller reporting "no spread measured" crashed every
    consumer using the default-argument idiom. Conversely the idiom silently
    substitutes a default when the key is missing, which is how a fabricated 2%
    spread travelled through the risk stack.
    """

    def test_measured_spread_is_reported_as_real(self):
        spread, is_real = read_spread({"spread": 0.031, "is_real": True})
        assert spread == pytest.approx(0.031)
        assert is_real is True

    def test_explicit_none_falls_back_and_is_not_real(self):
        spread, is_real = read_spread({"spread": None, "is_real": False})
        assert spread == pytest.approx(0.02)
        assert is_real is False

    def test_missing_key_falls_back_and_is_not_real(self):
        spread, is_real = read_spread({})
        assert spread == pytest.approx(0.02)
        assert is_real is False

    def test_none_orderbook_is_handled(self):
        assert read_spread(None) == (0.02, False)

    def test_assumed_marker_overrides_the_number(self):
        """A book that says it is assumed is not trusted even if it has a value."""
        spread, is_real = read_spread({"spread": 0.05, "spread_source": "assumed_default"})
        assert is_real is False

    def test_zero_spread_is_preserved_not_defaulted(self):
        """A genuinely tight book has a small spread; don't mistake it for absent."""
        spread, is_real = read_spread({"spread": 0.0, "is_real": True})
        assert spread == 0.0
        assert is_real is True

    def test_garbage_is_rejected(self):
        assert read_spread({"spread": "wide"}) == (0.02, False)

    def test_negative_spread_is_rejected(self):
        assert read_spread({"spread": -0.5}) == (0.02, False)

    def test_custom_default_respected(self):
        assert read_spread({}, 0.01)[0] == pytest.approx(0.01)


class TestUnknownSpreadRaisesUncertainty:
    """
    An unknown spread is worse than a wide one: the cost of getting in or out
    cannot be bounded. It must raise uncertainty, not quietly become 2%.
    """

    def _uncertainty(self, context):
        from src.ptai.intelligence.uncertainty import UncertaintyEngine
        return UncertaintyEngine().calculate_uncertainty([], context=context)

    def test_unknown_spread_is_not_treated_as_tight(self):
        unknown = self._uncertainty({"spread": None, "is_real": False})
        tight = self._uncertainty({"spread": 0.01, "is_real": True})
        assert unknown > tight

    def test_unknown_is_at_least_as_bad_as_the_widest_measured(self):
        unknown = self._uncertainty({"spread": None, "is_real": False})
        widest = self._uncertainty({"spread": 0.10, "is_real": True})
        assert unknown >= widest

    def test_an_assumed_book_is_not_treated_as_measured(self):
        assumed = self._uncertainty({"spread": 0.01, "spread_source": "assumed_default"})
        measured = self._uncertainty({"spread": 0.01, "is_real": True})
        assert assumed > measured


class TestNoSilentSpreadDefaultsRemain:
    def test_no_consumer_uses_the_get_default_idiom(self):
        """
        A bare `.get("spread", default)` cannot distinguish absent from None and
        silently substitutes a number. All consumers must go through read_spread.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ptai"
        # Only a NUMERIC LITERAL default silently substitutes a measurement.
        # A fallback to None, or to another expression, is legitimate plumbing.
        pattern = re.compile(r'\.get\(\s*"spread"\s*,\s*[0-9]')
        allowed = {
            # The analyzer reads the raw dict itself and labels provenance.
            "markets/orderbook.py",
            # Reads a betting line handicap, not an orderbook spread.
            "betting/in_play.py",
        }
        offenders = []
        for path in root.rglob("*.py"):
            rel = str(path.relative_to(root))
            if rel in allowed:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line) and not line.strip().startswith("#"):
                    offenders.append(f"{rel}:{i}")
        assert offenders == [], (
            "these still silently default a spread; use read_spread: " + ", ".join(offenders))
