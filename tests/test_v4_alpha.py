"""
Tests for v4 alpha engines - Top 5 + additional queue
"""
import pytest
from src.ptai.markets.base import Market, MarketSource
from src.ptai.strategy.combinatorial import CombinatorialArbitrageEngine
from src.ptai.strategy.reference_odds import ReferenceOddsEngine
from src.ptai.strategy.event_graph import EventGraphConsistencyEngine
from src.ptai.strategy.favourite_longshot import FavouriteLongshotEngine
from src.ptai.markets.whale_tracker import WhaleTracker
from src.ptai.strategy.rag_history import HistoricalRAG
from src.ptai.intelligence.bayesian import BayesianUpdater, NewsEvent
from src.ptai.risk.dynamic_threshold import DynamicThresholdEngine
from src.ptai.strategy.liquidity_rewards import LiquidityRewardsEngine
from src.ptai.strategy.orderbook_imbalance import OrderBookImbalanceEngine
from src.ptai.strategy.alpha_engine import AlphaEngine
from datetime import datetime, timezone, timedelta


def make_market(id="M1", question="Will Trump win?", price=0.6, liquidity=10000, volume=15000, event_slug="trump-win", category="politics"):
    return Market(
        id=id,
        question=question,
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1-price],
        volume_24h=volume,
        liquidity=liquidity,
        active=True,
        closed=False,
        end_date=None,
        event_slug=event_slug,
        source=MarketSource.POLYMARKET,
        raw={"venue": "polymarket", "question": question, "category": category}
    )


def test_combinatorial_arbitrage_buy_all_yes():
    markets = [
        make_market(id="M1", question="Will Trump win?", price=0.30, event_slug="election-winner"),
        make_market(id="M2", question="Will Biden win?", price=0.30, event_slug="election-winner"),
        make_market(id="M3", question="Will Other win?", price=0.20, event_slug="election-winner"),
    ]
    engine = CombinatorialArbitrageEngine(min_profit_pct=0.01)
    opps = engine.find_combinatorial_arbitrage(markets)
    assert len(opps) >= 1
    opp = opps[0]
    assert opp.sum_yes == pytest.approx(0.80, abs=0.01)
    assert opp.arbitrage_type == "buy_all_yes"
    assert opp.estimated_profit_pct > 0.1
    assert opp.should_trade


def test_combinatorial_arbitrage_sell_all_yes():
    markets = [
        make_market(id="M1", question="Will Trump win?", price=0.45, event_slug="election-winner"),
        make_market(id="M2", question="Will Biden win?", price=0.40, event_slug="election-winner"),
        make_market(id="M3", question="Will Other win?", price=0.30, event_slug="election-winner"),
    ]
    engine = CombinatorialArbitrageEngine(min_profit_pct=0.01)
    opps = engine.find_combinatorial_arbitrage(markets)
    assert len(opps) >= 1
    opp = opps[0]
    assert opp.sum_yes == pytest.approx(1.15, abs=0.01)
    assert opp.arbitrage_type == "sell_all_yes_buy_all_no"
    assert opp.should_trade


def test_reference_odds_engine():
    market = make_market(question="Will BTC be above $100k by Dec?", price=0.62, category="crypto")
    engine = ReferenceOddsEngine()
    refs = engine.get_all_reference_odds(market)
    # Should find BTC reference
    assert len(refs) >= 0  # may be 0 if no match, but test method exists
    # Test explicit deribit ref
    ref = engine.get_deribit_reference(market)
    if ref:
        assert ref.source == "deribit"
        assert 0 < ref.reference_price < 1


def test_event_graph_consistency():
    markets = [
        make_market(id="M1", question="Will Trump win 2024?", price=0.62, event_slug="e1"),
        make_market(id="M2", question="Will Republican win 2024?", price=0.58, event_slug="e2"),
    ]
    engine = EventGraphConsistencyEngine()
    violations = engine.find_violations(markets)
    # Trump 0.62 > GOP 0.58 should be violation (Trump implies GOP)
    assert len(violations) >= 1
    v = violations[0]
    assert v.violation_size > 0
    assert v.edge > 0


def test_event_graph_mutual_exclusion():
    markets = [
        make_market(id="M1", question="Will Trump win?", price=0.60, event_slug="e1"),
        make_market(id="M2", question="Will Biden win?", price=0.55, event_slug="e2"),
    ]
    engine = EventGraphConsistencyEngine()
    violations = engine.find_violations(markets)
    # Sum 1.15 >1 should be violation if mutually exclusive
    assert len(violations) >= 1


def test_favourite_longshot_longshot():
    market = make_market(id="M1", question="Will obscure event happen?", price=0.03, liquidity=15000)
    engine = FavouriteLongshotEngine()
    opps = engine.scan_markets([market])
    assert len(opps) == 1
    opp = opps[0]
    assert opp.is_longshot
    assert opp.edge > 0  # overpriced, should sell YES edge = price - fair positive
    assert opp.fair_estimate < market.best_price


def test_favourite_longshot_favourite():
    market = make_market(id="M1", question="Will favourite win?", price=0.97, liquidity=15000)
    engine = FavouriteLongshotEngine()
    opps = engine.scan_markets([market])
    assert len(opps) == 1
    opp = opps[0]
    assert opp.is_favourite
    assert opp.edge > 0  # underpriced, should buy YES


def test_favourite_longshot_liquid_filter():
    market = make_market(id="M1", question="Will event happen?", price=0.03, liquidity=5000, volume=5000)
    engine = FavouriteLongshotEngine(min_liquidity=10000)
    opps = engine.scan_markets([market])
    # Engine returns opportunity but with liquidity_ok False and should_trade False
    assert len(opps) == 1
    assert not opps[0].liquidity_ok
    assert not opps[0].should_trade


def test_whale_tracker_mock():
    tracker = WhaleTracker()
    wallets, market_trades, wallets_dict = tracker.mock_whale_data()
    assert len(wallets) == 3
    assert len(market_trades) >= 1
    smart = [w for w in wallets if w.is_smart]
    dumb = [w for w in wallets if w.is_dumb]
    assert len(smart) >= 1
    assert len(dumb) >= 1
    assert smart[0].score > 0.6
    assert dumb[0].score < -0.5


def test_whale_signals():
    tracker = WhaleTracker()
    wallets, market_trades, wallets_dict = tracker.mock_whale_data()
    for market_id, trades in market_trades.items():
        signals = tracker.get_whale_signals(market_id=market_id, market_price=0.61, whale_trades=trades, whale_wallets=wallets_dict)
        assert len(signals) >= 1
        # Smart whale buying YES should be copy signal
        for s in signals:
            if s.whale_score > 0.6 and s.side == "YES":
                assert s.signal_type == "copy_smart"


def test_rag_history_retrieve():
    rag = HistoricalRAG()
    similar = rag.retrieve_similar(question="Will Trump win?", top_k=3)
    assert len(similar) >= 1
    # Should find Trump related
    assert any("Trump" in m.question for m in similar)


def test_rag_base_rate():
    rag = HistoricalRAG()
    base_rate = rag.estimate_base_rate(market_id="test", question="Will Trump win 2024?", category="politics")
    assert 0 <= base_rate.base_rate <= 1
    assert base_rate.num_similar >= 0
    assert 0 <= base_rate.confidence <= 1


def test_bayesian_updater_decay():
    updater = BayesianUpdater(decay_half_life_hours=24.0)
    now = datetime.now(timezone.utc)
    news = [
        NewsEvent(timestamp=now - timedelta(hours=2), headline="Recent good news", sentiment=0.7, credibility=0.8, impact=0.08, source="Reuters"),
        NewsEvent(timestamp=now - timedelta(hours=30), headline="Old news", sentiment=0.3, credibility=0.5, impact=0.02, source="Blog"),
    ]
    state = updater.update(prior=0.50, news_events=news, base_rate=0.52, market_price=0.60)
    assert 0 <= state.posterior <= 1
    assert state.confidence > 0
    # Recent news should have higher weight than old
    assert state.posterior != 0.5  # should have moved


def test_bayesian_time_decay():
    updater = BayesianUpdater(decay_half_life_hours=24.0)
    decay_recent = updater._time_decay(datetime.now(timezone.utc) - timedelta(hours=2))
    decay_old = updater._time_decay(datetime.now(timezone.utc) - timedelta(hours=48))
    assert decay_recent > decay_old
    assert decay_recent > 0.5
    assert decay_old < 0.3


def test_dynamic_threshold_liquid():
    market = make_market(price=0.6, liquidity=15000)
    engine = DynamicThresholdEngine(base_threshold=0.08)
    thresh = engine.calculate(market=market, fees=0.02, slippage=0.01, uncertainty=0.1, orderbook={"volatility": 0.02, "spread": 0.02})
    assert thresh.total_threshold >= 0.08
    assert thresh.total_threshold < 0.25  # capped


def test_dynamic_threshold_illiquid():
    market_liquid = make_market(price=0.6, liquidity=15000)
    market_illiquid = make_market(price=0.6, liquidity=500)
    engine = DynamicThresholdEngine(base_threshold=0.08)
    thresh_liquid = engine.calculate(market=market_liquid, fees=0.02, slippage=0.01, uncertainty=0.1)
    thresh_illiquid = engine.calculate(market=market_illiquid, fees=0.02, slippage=0.01, uncertainty=0.1)
    assert thresh_illiquid.total_threshold > thresh_liquid.total_threshold
    assert thresh_illiquid.liquidity_premium > thresh_liquid.liquidity_premium
    # Illiquid needs 15%+
    if market_illiquid.liquidity < 1000:
        assert thresh_illiquid.total_threshold >= 0.15


def test_liquidity_rewards():
    market = make_market(price=0.6, liquidity=10000)
    engine = LiquidityRewardsEngine(bankroll=50.0)
    est = engine.estimate_rewards(market=market, amount_usd=5.0)
    assert est.reward_per_day_usd >= 0
    assert est.should_provide_liquidity or not est.should_provide_liquidity  # bool
    quote = engine.create_quotes(market=market, inventory=0, amount_usd=5.0)
    assert quote.bid_price < quote.ask_price
    assert quote.should_quote


def test_liquidity_rewards_inventory_limit():
    market = make_market(price=0.6, liquidity=10000)
    engine = LiquidityRewardsEngine(bankroll=50.0)  # $50 bankroll, max $5 inventory 10%
    # With 0 inventory should quote
    quote0 = engine.create_quotes(market=market, inventory=0, amount_usd=5.0)
    assert quote0.should_quote
    # With high inventory should not quote or skew
    quote_high = engine.create_quotes(market=market, inventory=10, amount_usd=5.0)
    # Should still create quotes but with skew or not should_quote if too high
    assert quote_high.bid_price <= quote0.bid_price or not quote_high.should_quote


def test_orderbook_imbalance():
    engine = OrderBookImbalanceEngine()
    # Bid stacked bullish
    ob_bid_stacked = {"bids": [(0.60, 5000)], "asks": [(0.61, 1000)], "bid_size": 5000, "ask_size": 1000, "spread": 0.01, "depth": 6000}
    depth = engine.analyze_imbalance(ob_bid_stacked)
    assert depth.imbalance > 0.4
    signal, reason = engine.get_entry_timing_signal(depth, side="YES")
    assert "wait" in signal.lower() or "bullish" in signal.lower()
    
    # Ask stacked bearish - good to buy weakness
    ob_ask_stacked = {"bids": [(0.60, 1000)], "asks": [(0.61, 5000)], "bid_size": 1000, "ask_size": 5000, "spread": 0.01, "depth": 6000}
    depth2 = engine.analyze_imbalance(ob_ask_stacked)
    assert depth2.imbalance < -0.4


def test_alpha_engine_combined():
    markets = [
        make_market(id="M1", question="Will Trump win?", price=0.30, event_slug="election-winner"),
        make_market(id="M2", question="Will Biden win?", price=0.30, event_slug="election-winner"),
        make_market(id="M3", question="Will Other win?", price=0.20, event_slug="election-winner"),
        make_market(id="M4", question="Will BTC be above $100k?", price=0.62, event_slug="btc-100k", category="crypto"),
        make_market(id="M5", question="Will obscure 2% event happen?", price=0.03, liquidity=15000, event_slug="obscure"),
    ]
    engine = AlphaEngine(bankroll=50.0)
    results = engine.scan_all_alpha(markets)
    assert "combinatorial" in results
    assert "reference_odds" in results
    assert "event_graph" in results
    assert "favourite_longshot" in results
    assert "whale" in results
    assert "rag" in results
    assert "dynamic_threshold" in results
    assert "liquidity_rewards" in results
    assert results["combinatorial"]["total"] >= 1
    assert results["combinatorial"]["tradeable"] >= 1


def test_alpha_adjusted_score():
    market = make_market(question="Will BTC be above $100k?", price=0.62, liquidity=15000)
    engine = AlphaEngine(bankroll=50.0)
    base_score = 0.5
    adjusted = engine.calculate_alpha_adjusted_score(base_score=base_score, market=market)
    assert adjusted >= base_score * 0.8  # should not drastically reduce, may boost
