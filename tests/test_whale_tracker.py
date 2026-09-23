"""
Whale tracking: real wallet activity, and an edge that is actually captureable.

Two fabrications lived here. `mock_whale_data()` returned three handwritten
wallets - `0x1234...smart1` is not even a valid hex address - with invented
P&L, and the alpha scan consumed them as real. Then `get_whale_signals`
computed

    edge_estimate = abs(wallet.score) * 0.05   # smart score 0.8 => 4% edge

which invents an edge from a win rate. A wallet's history does not imply a
percentage edge on a specific market; the only edge a copy trade can capture
is the gap between what the whale paid and what the market costs now.
"""
import pytest

from src.ptai.markets.whale_tracker import (
    RESOLVED_LOSS, RESOLVED_WIN, UNRESOLVED, WhaleFeed, WhaleSignal, WhaleTracker,
    WhaleWallet, _is_address, normalise_trade,
)

SMART = "0x" + "a1" * 20
DUMB = "0x" + "b2" * 20
THIN = "0x" + "c3" * 20


def trade_row(address, price=0.55, size=1000.0, pnl=None, side="BUY",
              market="0xmarket1", title="Will X happen?", ts=1700000000):
    """One row in the shape Polymarket's /activity endpoint returns."""
    row = {"proxyWallet": address, "conditionId": market, "title": title,
           "side": side, "price": price, "size": size,
           "usdcSize": round(price * size, 2), "timestamp": ts, "outcome": "Yes"}
    if pnl is not None:
        row["pnl"] = pnl
    return row


def wallet_history(address, wins, losses, open_positions=0, size=1000.0, price=0.55):
    """A resolved history with a given win/loss split."""
    rows = [trade_row(address, price=price, size=size, pnl=+50.0) for _ in range(wins)]
    rows += [trade_row(address, price=price, size=size, pnl=-40.0) for _ in range(losses)]
    rows += [trade_row(address, price=price, size=size) for _ in range(open_positions)]
    return rows


def tracker_with(activity=None, per_wallet=None):
    """A tracker whose HTTP layer is replaced by canned payloads."""
    def http_get(url, params):
        if params and params.get("user"):
            return (per_wallet or {}).get(params["user"], [])
        return activity if activity is not None else []
    return WhaleTracker(http_get=http_get)


# ═══════════════════════════════════════════════════════════════════════════
# The fabrications are gone
# ═══════════════════════════════════════════════════════════════════════════

def test_mock_whale_data_is_gone():
    assert not hasattr(WhaleTracker(), "mock_whale_data")


def test_unreachable_feed_returns_no_wallets_and_says_why():
    tracker = WhaleTracker(http_get=lambda url, params: None)
    feed = tracker.load_whales()
    assert feed.wallets == []
    assert not feed.ok
    assert feed.last_error
    assert feed.is_synthetic is False


def test_empty_feed_returns_no_wallets():
    feed = tracker_with(activity=[]).load_whales()
    assert feed.wallets == []
    assert not feed.ok


def test_report_no_longer_advertises_mock_data():
    """The old report had a field named 'mock' admitting the data was invented."""
    report = WhaleTracker().get_report()
    assert "mock" not in report
    assert "removed_fabrication" in report
    assert "data-api" in report["source"]


# ═══════════════════════════════════════════════════════════════════════════
# Address and row validation
# ═══════════════════════════════════════════════════════════════════════════

def test_address_validation_rejects_the_old_fake_addresses():
    """The fabricated wallets used addresses like 0x1234...smart1."""
    assert _is_address(SMART)
    assert _is_address("0x" + "0" * 40)
    assert not _is_address("0x1234...smart1")
    assert not _is_address("0x1234")
    assert not _is_address("not-an-address")
    assert not _is_address("0x" + "z" * 40)
    assert not _is_address("")
    assert not _is_address(None)


def test_normalise_trade_rejects_unusable_rows():
    assert normalise_trade({}) is None
    assert normalise_trade({"proxyWallet": "0xbad"}) is None
    # no price
    assert normalise_trade({"proxyWallet": SMART, "size": 100}) is None
    # price outside (0, 1)
    assert normalise_trade({"proxyWallet": SMART, "price": 1.5, "size": 100}) is None
    assert normalise_trade({"proxyWallet": SMART, "price": 0.5}) is None  # no size


def test_normalise_trade_computes_usd_from_size_when_absent():
    t = normalise_trade({"proxyWallet": SMART, "price": 0.4, "size": 250.0,
                         "conditionId": "0xm", "side": "BUY"})
    assert t["amount_usd"] == pytest.approx(100.0)


def test_normalise_trade_prefers_the_reported_usdc_size():
    t = normalise_trade(trade_row(SMART, price=0.5, size=1000.0))
    assert t["amount_usd"] == pytest.approx(500.0)


def test_unresolved_trade_is_not_a_loss():
    """
    A row with no pnl field has no outcome yet. Treating it as a loss - which
    is what summing pnl and counting non-wins does - drags every active
    wallet's win rate toward zero and makes good wallets look mediocre.
    """
    assert normalise_trade(trade_row(SMART))["outcome"] == UNRESOLVED
    assert normalise_trade(trade_row(SMART, pnl=10.0))["outcome"] == RESOLVED_WIN
    assert normalise_trade(trade_row(SMART, pnl=-10.0))["outcome"] == RESOLVED_LOSS
    assert normalise_trade(trade_row(SMART, pnl=0.0))["outcome"] == UNRESOLVED


# ═══════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════

def test_discover_whales_ranks_by_notional_not_trade_count():
    """One large position is a whale; fifty dust trades are not."""
    rows = [trade_row(SMART, price=0.5, size=100000.0)]
    rows += [trade_row(DUMB, price=0.5, size=20.0) for _ in range(10)]
    found = tracker_with(activity=rows).discover_whales(min_trades=1)
    assert list(found)[0] == SMART
    assert found[SMART] > found[DUMB]


def test_discover_whales_skips_invalid_addresses():
    rows = [trade_row(SMART), {"proxyWallet": "0xnope", "price": 0.5, "size": 100}]
    found = tracker_with(activity=rows).discover_whales(min_trades=1)
    assert list(found) == [SMART]


def test_discover_whales_enforces_a_minimum_trade_count():
    rows = [trade_row(SMART), trade_row(SMART), trade_row(SMART), trade_row(DUMB)]
    found = tracker_with(activity=rows).discover_whales(min_trades=3)
    assert list(found) == [SMART]


# ═══════════════════════════════════════════════════════════════════════════
# Wallet scoring
# ═══════════════════════════════════════════════════════════════════════════

def test_analyze_wallet_with_no_trades_is_neutral_not_smart():
    w = WhaleTracker().analyze_wallet(SMART, [])
    assert w.total_trades == 0
    assert w.is_smart is False and w.is_dumb is False
    assert w.score == 0.0
    assert "no trades" in w.reasoning


def test_win_rate_counts_only_resolved_trades():
    """10 wins, 10 losses, 30 still open -> 50%, not 20%."""
    trades = [normalise_trade(r) for r in wallet_history(SMART, 10, 10, open_positions=30)]
    w = WhaleTracker(min_resolved_trades=20).analyze_wallet(SMART, trades)
    assert w.resolved_trades == 20
    assert w.win_rate == pytest.approx(0.5)


def test_sample_size_gate_blocks_a_lucky_newcomer():
    """A 100% win rate over three trades is noise and must not be 'smart'."""
    trades = [normalise_trade(r) for r in wallet_history(SMART, 3, 0)]
    w = WhaleTracker(min_resolved_trades=20).analyze_wallet(SMART, trades)
    assert w.win_rate == 1.0
    assert w.sample_size_ok is False
    assert w.is_smart is False
    assert "too small" in w.reasoning


def test_a_proven_winner_is_smart_and_a_proven_loser_is_dumb():
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    smart = tracker.analyze_wallet(
        SMART, [normalise_trade(r) for r in wallet_history(SMART, 40, 10, size=100.0)])
    dumb = tracker.analyze_wallet(
        DUMB, [normalise_trade(r) for r in wallet_history(DUMB, 10, 40, size=100.0)])
    # 80% over 50 resolved shrinks to ~0.47 under the small-sample prior,
    # which is why is_smart - not the raw score - is the qualification gate
    assert smart.is_smart and smart.score > 0.4
    assert dumb.is_dumb and dumb.score < -0.4
    assert smart.resolved_trades == 50 == dumb.resolved_trades


def test_small_sample_is_shrunk_toward_neutral():
    """
    A 70% record over 20 trades must score below a 55% record over 500,
    otherwise luck outranks evidence.
    """
    tracker = WhaleTracker(min_whale_volume=100, min_resolved_trades=20)
    lucky = tracker.analyze_wallet(
        SMART, [normalise_trade(r) for r in wallet_history(SMART, 14, 6, size=10.0)])
    proven = tracker.analyze_wallet(
        DUMB, [normalise_trade(r) for r in wallet_history(DUMB, 275, 225, size=10.0)])
    assert lucky.win_rate > proven.win_rate
    assert lucky.score < proven.score


def test_score_is_bounded():
    tracker = WhaleTracker(min_whale_volume=1, min_resolved_trades=1)
    w = tracker.analyze_wallet(SMART, [normalise_trade(r) for r in
                                       wallet_history(SMART, 5000, 0, size=100000.0)])
    assert -1.0 <= w.score <= 1.0
    assert w.is_smart


def test_volume_gate_blocks_a_small_wallet_from_being_called_smart():
    """Skill without size is not a whale worth copying."""
    trades = [normalise_trade(r) for r in wallet_history(SMART, 40, 5, size=1.0)]
    w = WhaleTracker(min_whale_volume=10000, min_resolved_trades=20).analyze_wallet(SMART, trades)
    assert w.win_rate > 0.8
    assert w.is_smart is False
    assert w.total_volume < 10000


def test_analyze_wallet_records_its_data_source():
    trades = [normalise_trade(r) for r in wallet_history(SMART, 5, 5)]
    assert WhaleTracker().analyze_wallet(SMART, trades).data_source == "polymarket_data_api"


# ═══════════════════════════════════════════════════════════════════════════
# Signals - the edge must be real
# ═══════════════════════════════════════════════════════════════════════════

def _smart_wallet(tracker, address=SMART):
    return tracker.analyze_wallet(
        address, [normalise_trade(r) for r in wallet_history(address, 40, 10, size=100.0)])


def test_edge_is_the_gap_between_whale_entry_and_current_price():
    """
    The whole point. A wallet's win rate does not imply a percentage edge;
    the only thing a copy trade can capture is the price difference.
    """
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    wallet = _smart_wallet(tracker)
    trade = normalise_trade(trade_row(SMART, price=0.50, size=2000.0))
    signals = tracker.get_whale_signals("m1", market_price=0.45,
                                        whale_trades=[trade],
                                        whale_wallets={SMART: wallet})
    assert len(signals) == 1
    # whale bought at 0.50, market is now 0.45 -> copying is 5c cheaper
    assert signals[0].edge_estimate == pytest.approx(0.05, abs=1e-6)
    assert signals[0].should_trade is True


def test_a_copy_after_the_move_has_no_edge_left():
    """
    Following a whale means paying after the move. If the price has already
    risen above their entry there is nothing to capture, and the signal must
    say so rather than reporting the old score-derived edge.
    """
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    wallet = _smart_wallet(tracker)
    trade = normalise_trade(trade_row(SMART, price=0.50, size=2000.0))
    signals = tracker.get_whale_signals("m1", market_price=0.62,
                                        whale_trades=[trade],
                                        whale_wallets={SMART: wallet})
    assert signals[0].edge_estimate < 0
    assert signals[0].should_trade is False
    assert "no edge left" in signals[0].reasoning


def test_edge_does_not_scale_with_the_wallet_score():
    """
    Regression guard on `abs(score) * 0.05`. Two wallets with very different
    scores facing the same prices must see the same edge.
    """
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    strong = tracker.analyze_wallet(
        SMART, [normalise_trade(r) for r in wallet_history(SMART, 95, 5, size=100.0)])
    # 24% win rate with negative P&L - qualifies as dumb, so it signals too
    weak = tracker.analyze_wallet(
        DUMB, [normalise_trade(r) for r in wallet_history(DUMB, 12, 38, size=100.0)])
    assert weak.is_dumb, weak.reasoning
    # same entry and market price for both, so any difference in edge would
    # have to come from the score - which is exactly the old fabrication
    trade_smart = normalise_trade(trade_row(SMART, price=0.50, size=2000.0))
    trade_dumb = normalise_trade(trade_row(DUMB, price=0.50, size=2000.0))
    a = tracker.get_whale_signals("m1", 0.45, [trade_smart], {SMART: strong})
    b = tracker.get_whale_signals("m1", 0.45, [trade_dumb], {DUMB: weak})
    assert a and b
    assert a[0].signal_type == "copy_smart" and b[0].signal_type == "fade_dumb"
    assert a[0].whale_score > 0 > b[0].whale_score
    # opposite directions, so the captureable edge is equal in magnitude but
    # opposite in sign - and in neither case a multiple of the score
    assert a[0].edge_estimate == pytest.approx(-b[0].edge_estimate, abs=1e-9)
    assert abs(a[0].edge_estimate) == pytest.approx(0.05, abs=1e-9)


def test_thin_records_never_produce_a_signal():
    """A wallet with too few resolved trades cannot be copied or faded."""
    tracker = WhaleTracker(min_whale_volume=100, min_resolved_trades=20)
    thin = tracker.analyze_wallet(
        THIN, [normalise_trade(r) for r in wallet_history(THIN, 3, 0, size=5000.0)])
    trade = normalise_trade(trade_row(THIN, price=0.50, size=5000.0))
    assert tracker.get_whale_signals("m1", 0.45, [trade], {THIN: thin}) == []


def test_unknown_wallet_produces_no_signal():
    tracker = WhaleTracker()
    trade = normalise_trade(trade_row(SMART, price=0.5, size=2000.0))
    assert tracker.get_whale_signals("m1", 0.45, [trade], {}) == []


def test_dust_trades_are_ignored():
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    wallet = _smart_wallet(tracker)
    trade = normalise_trade(trade_row(SMART, price=0.50, size=10.0))  # $5
    assert tracker.get_whale_signals("m1", 0.45, [trade], {SMART: wallet}) == []


def test_fade_signal_profits_when_the_price_has_risen():
    """Fading a dumb whale means taking the other side at today's price."""
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    dumb = tracker.analyze_wallet(
        DUMB, [normalise_trade(r) for r in wallet_history(DUMB, 5, 45, size=100.0)])
    trade = normalise_trade(trade_row(DUMB, price=0.50, size=2000.0))
    signals = tracker.get_whale_signals("m1", 0.60, [trade], {DUMB: dumb})
    assert signals[0].signal_type == "fade_dumb"
    assert signals[0].edge_estimate == pytest.approx(0.10, abs=1e-6)
    assert signals[0].should_trade is True


def test_signals_are_sorted_by_edge():
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    wallet = _smart_wallet(tracker)
    trades = [normalise_trade(trade_row(SMART, price=0.40, size=2000.0)),
              normalise_trade(trade_row(SMART, price=0.55, size=2000.0)),
              normalise_trade(trade_row(SMART, price=0.50, size=2000.0))]
    signals = tracker.get_whale_signals("m1", 0.45, trades, {SMART: wallet})
    edges = [s.edge_estimate for s in signals]
    assert edges == sorted(edges, reverse=True)


def test_signal_carries_provenance():
    tracker = WhaleTracker(min_whale_volume=1000, min_resolved_trades=20)
    wallet = _smart_wallet(tracker)
    trade = normalise_trade(trade_row(SMART, price=0.50, size=2000.0))
    s = tracker.get_whale_signals("m1", 0.45, [trade], {SMART: wallet})[0]
    assert s.provenance == "polymarket_data_api"
    assert s.is_synthetic is False
    assert s.whale_entry_price == 0.50


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end against a canned feed
# ═══════════════════════════════════════════════════════════════════════════

def test_load_whales_end_to_end_from_a_real_shaped_feed():
    activity = [trade_row(SMART, price=0.5, size=20000.0) for _ in range(5)]
    history = wallet_history(SMART, 40, 10, size=100.0)
    tracker = tracker_with(activity=activity, per_wallet={SMART: history})
    feed = tracker.load_whales()

    assert feed.ok
    assert len(feed.wallets) == 1
    assert feed.wallets[0].is_smart
    assert feed.wallets_by_address[SMART].resolved_trades == 50
    assert feed.source == "polymarket_data_api"
    assert feed.fetched_at
    assert any(t["whale_address"] == SMART for t in feed.market_trades["0xmarket1"])
    assert tracker.fetch_counts["activity"] >= 1
    assert tracker.fetch_counts["wallet"] == 1


def test_load_whales_rejects_an_invalid_wallet_address_from_the_feed():
    tracker = WhaleTracker(http_get=lambda url, params: None)
    assert tracker.fetch_wallet_activity("0x1234...smart1") == []
    assert "not a valid wallet address" in tracker.last_error


def test_feed_dataclass_reports_ok_only_with_wallets():
    assert WhaleFeed().ok is False
    assert WhaleFeed(wallets=[WhaleWallet(
        address=SMART, total_volume=1, total_trades=1, pnl_usd=0, win_rate=0.5,
        avg_position=0, is_smart=False, is_dumb=False, score=0)]).ok is True
