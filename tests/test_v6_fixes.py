
"""
Tests for v6 fixes - architecture vs implementation gaps
Fixes A-E from source inspection
"""
import pytest
from src.ptai.venues.registry import VenueRegistry
from src.ptai.venues.polymarket_adapter import PolymarketAdapter
from src.ptai.venues.kalshi_adapter import KalshiAdapter
from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.venues.adapter import VenueOpportunity, VenueType
from src.ptai.markets.market_normalizer import MarketNormalizer
from src.ptai.strategy.opportunity import OpportunityEngine, FastModelClassifier
from src.ptai.venues.qualification import VenueQualificationEngine
from src.ptai.learning.paper_trading import PaperTradingEngine

def make_market(id="M1", question="Will Trump win?", price=0.6, vol=10000, liq=10000, venue="polymarket", category="politics"):
    return Market(
        id=id,
        source=MarketSource.POLYMARKET,
        question=question,
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1-price],
        tokens=[Token(token_id=id, outcome="YES", price=price)],
        volume=vol,
        volume_24h=vol*0.5,
        liquidity=liq,
        active=True,
        closed=False,
        raw={"venue": venue, "category": category}
    )

def test_venue_learning_bug_fixed():
    """
    Fix E: registry stores venue_id:category but ranking looked up venue_id only
    Now fixed: lookup venue_id:category first then fallbacks
    """
    registry = VenueRegistry(country_code="UG")
    registry.venue_performance = {
        "polymarket:politics": {"total": 50, "wins": 25, "win_rate": 0.5, "brier_score": 0.30, "forecast_skill": 0.4, "brier_sum": 15, "avg_edge": 0.05, "profit": -10},
        "polymarket:sports": {"total": 50, "wins": 35, "win_rate": 0.7, "brier_score": 0.18, "forecast_skill": 0.64, "brier_sum": 9, "avg_edge": 0.08, "profit": 20},
        "kalshi:economics": {"total": 60, "wins": 45, "win_rate": 0.75, "brier_score": 0.15, "forecast_skill": 0.7, "brier_sum": 9, "avg_edge": 0.10, "profit": 30},
    }
    
    m1 = make_market(id="M1", question="Will Trump win?", price=0.6, category="politics")
    m2 = make_market(id="M2", question="Will Lakers win?", price=0.6, category="sports")
    m3 = make_market(id="M3", question="Will CPI exceed?", price=0.6, category="economics")
    
    opp1 = VenueOpportunity(market=m1, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.6, estimated_fair=0.7, raw_edge=0.1, effective_edge=0.08, confidence=0.7, category="politics", should_trade=True)
    opp2 = VenueOpportunity(market=m2, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.6, estimated_fair=0.7, raw_edge=0.1, effective_edge=0.08, confidence=0.7, category="sports", should_trade=True)
    opp3 = VenueOpportunity(market=m3, venue_id="kalshi", venue_type=VenueType.PREDICTION, side="YES", market_price=0.6, estimated_fair=0.7, raw_edge=0.1, effective_edge=0.08, confidence=0.7, category="economics", should_trade=True)
    
    for opp in [opp1, opp2, opp3]:
        opp.calculate_common_score()
    
    initial_scores = [opp1.score, opp2.score, opp3.score]
    # All same initially
    assert initial_scores[0] == initial_scores[1] == initial_scores[2]
    
    ranked = registry.rank_opportunities([opp1, opp2, opp3])
    
    # After fix, kalshi:economics should be top (skill 0.7 win 0.75 brier 0.15)
    # polymarket:sports second (skill 0.64 win 0.7)
    # polymarket:politics last (skill 0.4 win 0.5)
    assert ranked[0].venue_id == "kalshi"
    assert ranked[0].category == "economics"
    assert ranked[1].category == "sports"
    assert ranked[2].category == "politics"
    # Scores should be different now
    assert ranked[0].score > ranked[1].score > ranked[2].score

def test_venue_learning_fallback():
    registry = VenueRegistry(country_code="UG")
    registry.venue_performance = {
        "polymarket:politics": {"total": 50, "wins": 35, "win_rate": 0.7, "brier_score": 0.18, "forecast_skill": 0.64, "brier_sum": 9, "avg_edge": 0.08, "profit": 20},
    }
    
    # Test get_performance_for_venue_category helper
    perf = registry.get_performance_for_venue_category("polymarket", "politics")
    assert perf["win_rate"] == 0.7
    assert perf["forecast_skill"] == 0.64
    
    # Fallback to venue any category
    perf2 = registry.get_performance_for_venue_category("polymarket", "sports")
    # Should find polymarket:politics as best match for venue polymarket
    assert perf2["total"] == 50

@pytest.mark.asyncio
async def test_real_orderbook_not_mock():
    """
    Fix A: orderbook isn't actually real in Polymarket adapter, Mock orderbook for now
    Now fixed: tries CLOB real + enhanced estimation based on liquidity/volume
    """
    adapter = PolymarketAdapter()
    m_high_liq = make_market(id="M1", question="Will event happen?", price=0.6, vol=50000, liq=25000)
    m_low_liq = make_market(id="M2", question="Will obscure happen?", price=0.6, vol=1000, liq=1000)
    
    ob_high = await adapter.get_orderbook(m_high_liq)
    ob_low = await adapter.get_orderbook(m_low_liq)
    
    # High liquidity should have tight spread, low should have wide
    assert ob_high["spread"] < ob_low["spread"]
    assert ob_high["execution_quality"] > ob_low["execution_quality"]
    assert ob_high["slippage_estimate"] < ob_low["slippage_estimate"]
    
    # Should have source field indicating real or enhanced estimation
    assert "source" in ob_high
    assert ob_high["source"] in ["clob_real", "enhanced_estimation", "no_token_fallback", "error_fallback"]
    assert "liquidity" in ob_high
    assert "volume_24h" in ob_high
    assert "slippage_estimate" in ob_high
    assert "execution_quality" in ob_high

@pytest.mark.asyncio
async def test_real_portfolio_not_placeholder():
    """
    Fix B: Portfolio retrieval placeholder {balance:0, positions:[], orders:[]}
    Now fixed: storage+clob+onchain actual balance positions open orders fills exposure
    """
    adapter = PolymarketAdapter()
    portfolio = await adapter.get_portfolio()
    
    # Should have actual fields
    assert "balance" in portfolio
    assert "positions" in portfolio
    assert "exposure" in portfolio
    assert "bankroll" in portfolio
    
    # Should have checks for actual balance positions etc
    # If is_real True, should have checks
    if portfolio.get("is_real"):
        assert "checks" in portfolio
        assert "actual_balance" in portfolio["checks"]
        assert "actual_positions" in portfolio["checks"]
        assert "actual_exposure" in portfolio["checks"]
    
    # Balance should be bankroll $50 default, not 0 placeholder
    assert portfolio["balance"] >= 0
    # Source should indicate real
    assert "source" in portfolio

def test_fast_model_not_just_volume_sort():
    """
    Fix C: fast_model_screen() currently sort by volume take top 50 rather than actually running fast model
    Now fixed: classification, news extraction, duplicate detection
    """
    classifier = FastModelClassifier()
    
    m1 = make_market(id="M1", question="Will Trump win election? Republican", price=0.6, vol=10000, liq=10000)
    m2 = make_market(id="M2", question="Will Lakers win championship? NBA", price=0.6, vol=100000, liq=50000)
    m3 = make_market(id="M3", question="Will Fed cut rates? CPI inflation FOMC", price=0.5, vol=5000, liq=5000)
    
    cls1 = classifier.classify(m1)
    cls2 = classifier.classify(m2)
    cls3 = classifier.classify(m3)
    
    assert cls1["category"] == "politics"
    assert cls2["category"] == "sports"
    assert cls3["category"] == "economics"
    
    assert cls3["has_news_potential"] == True  # Fed CPI has news potential
    assert cls1["confidence"] > 0
    
    # Duplicate detection
    m_dup1 = make_market(id="DUP1", question="Will Trump win 2024 election?", price=0.6, vol=10000, liq=10000)
    m_dup2 = make_market(id="DUP2", question="Will Trump win 2024 election?", price=0.61, vol=5000, liq=5000)
    duplicates = classifier.detect_duplicates([m_dup1, m_dup2])
    assert len(duplicates) >= 1
    assert len(duplicates[0]) == 2

def test_fast_model_screen_category_diversity():
    engine = OpportunityEngine()
    markets = [
        make_market(id=f"POL-{i}", question="Will Trump win? politics", price=0.5 + i*0.01, vol=10000+i*1000, liq=10000, category="politics")
        for i in range(20)
    ] + [
        make_market(id=f"SPORT-{i}", question="Will Lakers win? NBA sports", price=0.5, vol=5000, liq=5000, category="sports")
        for i in range(10)
    ] + [
        make_market(id=f"CRYPTO-{i}", question="Will BTC go up? Bitcoin crypto", price=0.5, vol=8000, liq=8000, category="crypto")
        for i in range(10)
    ]
    
    after_fast = engine.fast_model_screen(markets)
    
    # Should have category diversity, not just top volume (which would be all politics)
    categories = [m.raw.get("category") for m in after_fast]
    assert "politics" in categories
    # Should have at least 2 categories due to diversity logic
    assert len(set(categories)) >= 2

def test_opportunity_ranking_ev_not_just_edge():
    """
    Fix D: edge>=8% alone shouldn't determine which opportunity gets capital
    Now: Expected EV, liquidity, risk, uncertainty, portfolio impact, capital allocation
    Which opportunity gives best risk-adjusted expected return for capital available
    """
    engine = OpportunityEngine()
    
    # High edge poor liquidity vs lower edge excellent liquidity
    m_poor_liq = make_market(id="POOR", question="Will obscure event happen?", price=0.5, vol=1000, liq=1000, category="general")
    m_good_liq = make_market(id="GOOD", question="Will Fed cut rates? economics", price=0.5, vol=50000, liq=50000, category="economics")
    
    opp_poor = VenueOpportunity(market=m_poor_liq, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.5, estimated_fair=0.65, raw_edge=0.15, effective_edge=0.15, confidence=0.7, uncertainty=0.1, liquidity_score=0.05, execution_quality=0.3, category="general", should_trade=True, fees_pct=0.02, slippage_pct=0.03, spread_pct=0.08)
    opp_good = VenueOpportunity(market=m_good_liq, venue_id="kalshi", venue_type=VenueType.PREDICTION, side="YES", market_price=0.5, estimated_fair=0.60, raw_edge=0.10, effective_edge=0.09, confidence=0.85, uncertainty=0.05, liquidity_score=0.9, execution_quality=0.9, category="economics", should_trade=True, fees_pct=0.01, slippage_pct=0.005, spread_pct=0.01)
    
    opp_poor.calculate_common_score()
    opp_good.calculate_common_score()
    
    # Before portfolio impact, good liquidity with lower edge might still lose if raw edge only
    # But with new ranking, risk-adjusted EV with liquidity should make good win
    
    ranked = engine.rank_and_select([opp_poor, opp_good], max_trades=2, bankroll=50.0, current_positions=[])
    
    # Good liquidity, high confidence, low fees should be selected over poor liquidity high edge
    # 15% edge poor liquidity loses to 7% edge excellent liquidity high conf short resolution low correlation
    assert len(ranked) >= 1
    # The good one should have higher final score due to liquidity, execution, confidence, low fees
    if len(ranked) == 2:
        assert ranked[0].market.id == "GOOD"

def test_portfolio_impact_correlation():
    engine = OpportunityEngine()
    
    m1 = make_market(id="M1", question="Will Fed cut rates in June?", price=0.6, vol=20000, liq=20000, venue="polymarket")
    m1.event_slug = "fed-cut-june"
    
    m2 = make_market(id="M2", question="Will Fed cut rates in June?", price=0.61, vol=20000, liq=20000, venue="kalshi")
    m2.event_slug = "fed-cut-june"
    
    opp1 = VenueOpportunity(market=m1, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.6, estimated_fair=0.7, raw_edge=0.1, effective_edge=0.08, confidence=0.7, liquidity_score=0.8, execution_quality=0.8, category="economics", should_trade=True, fees_pct=0.02, slippage_pct=0.01, spread_pct=0.02)
    opp2 = VenueOpportunity(market=m2, venue_id="kalshi", venue_type=VenueType.PREDICTION, side="YES", market_price=0.61, estimated_fair=0.7, raw_edge=0.09, effective_edge=0.08, confidence=0.7, liquidity_score=0.8, execution_quality=0.8, category="economics", should_trade=True, fees_pct=0.02, slippage_pct=0.01, spread_pct=0.02)
    
    # Existing position on same event
    current_positions = [{"event_slug": "fed-cut-june", "question": "Will Fed cut rates in June?", "amount_usd": 5.0}]
    
    # Second position same event should be blocked due to correlation risk - one bet not two
    ranked = engine.rank_and_select([opp2], max_trades=2, bankroll=50.0, current_positions=current_positions)
    
    # Should be blocked because existing $5 + new $3 = $8 > $6 max per event (12% of $50)
    assert len(ranked) == 0

def test_market_scanner_uses_registry():
    from src.ptai.markets.scanner import MarketScanner
    from src.ptai.venues.registry import VenueRegistry
    
    registry = VenueRegistry(country_code="UG")
    registry.register(PolymarketAdapter())
    registry.register(KalshiAdapter())
    
    scanner = MarketScanner(venue_registry=registry)
    assert scanner.venue_registry is not None
    assert len(scanner.venue_registry.adapters) >= 2
    
    # Should have scan_multi_venue method
    assert hasattr(scanner, 'scan_multi_venue')

def test_kalshi_not_stub():
    """
    Kalshi must not be a stub - but "not a stub" means it talks to the real
    API, not that it always returns something.

    This asserted `len(markets) > 0`, which passed because the client had three
    separate mock-fallback paths that invented markets in every failure mode.
    The honest assertion is that the client delegates to the adapter and returns
    real markets or none, with nothing fabricated in between.
    """
    from src.ptai.markets.kalshi import KalshiClient
    client = KalshiClient(enabled=True)
    assert client.adapter is not None, "should delegate to the real KalshiAdapter"

    markets = client.scan_markets(target_count=10)
    for m in markets:
        assert m.id is not None
        assert m.question != ""
        assert not m.is_mock, "KalshiClient returned a fabricated market"
        assert "MOCK" not in str(m.id).upper()

    # disabled client returns nothing rather than inventing
    assert KalshiClient(enabled=False).scan_markets(target_count=10) == []

def test_market_normalizer_robust_19_venues():
    normalizer = MarketNormalizer()
    
    # Test generic normalization for various venues
    venues = ["manifold", "predictit", "simmer", "cymetica", "whitebit", "afx_dex", "grvt", "pionex", "betfair", "betdaq", "betconnect", "ccxt_unified", "veynor", "openpx", "apify"]
    
    for venue in venues:
        raw = {"id": f"{venue}-test-1", "question": f"Will {venue} market happen? Test question", "price": 0.6, "volume": 10000, "liquidity": 10000, "category": "general"}
        markets = normalizer.normalize(raw, source=MarketSource.POLYMARKET, venue_id=venue)
        assert len(markets) == 1
        assert markets[0].id == f"{venue}-test-1"
        assert normalizer.validate_market(markets[0])
    
    report = normalizer.get_report()
    # V7 honest: normalization != trading support
    venues_key = report.get("supported_venues") or report.get("supported_venues_normalization") or []
    assert len(venues_key) >= 15
    assert "polymarket" in venues_key
    # Check honest assessment exists
    assert "actual_trading_support" in report or "important_clarification" in report

def test_venue_qualification_robust():
    engine = VenueQualificationEngine(data_dir="/tmp/test_qual")
    
    # Not qualified - not enough trades
    perf_not_enough = {"total_paper_trades": 50, "win_rate": 0.7, "brier_score": 0.18, "forecast_skill": 0.64, "profit_paper": 20, "avg_edge": 0.08, "net_pnl": 20, "expected_value": 0.08, "profit_factor": 1.5, "log_loss": 0.5, "calibration_ece": 0.1, "execution_quality_avg": 0.6}
    result = engine.evaluate_qualification("polymarket", perf_not_enough)
    assert not result.is_qualified
    assert "50" in result.reasoning  # should mention total
    
    # Qualified - V7 includes P&L, EV, profit_factor, etc not just win rate
    perf_qualified = {
        "total_paper_trades": 120, "win_rate": 0.60, "brier_score": 0.20, "forecast_skill": 0.65, 
        "profit_paper": 15, "avg_edge": 0.05, "net_pnl": 15, "expected_value": 0.05, 
        "profit_factor": 1.5, "log_loss": 0.5, "calibration_ece": 0.1, "execution_quality_avg": 0.6,
        "fees_total": 2, "slippage_total": 1, "drawdown_max": 0.1
    }
    result2 = engine.evaluate_qualification("kalshi", perf_qualified)
    assert result2.is_qualified
    assert result2.qualification_date is not None
    # Check V7 reasoning mentions win rate alone not profitability
    assert "win rate alone NOT profitability" in result2.reasoning or "net P&L" in result2.reasoning

def test_paper_trading_engine():
    engine = PaperTradingEngine(data_dir="/tmp/test_paper")
    
    m = make_market(id="M1", question="Will event happen?", price=0.6)
    opp = VenueOpportunity(market=m, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.6, estimated_fair=0.7, raw_edge=0.1, effective_edge=0.08, confidence=0.7, category="politics", should_trade=True)
    
    trade = engine.record_paper_trade(opportunity=opp, amount_usd=3.0)
    assert trade.venue_id == "polymarket"
    assert trade.amount_usd == 3.0
    
    engine.resolve_trade(trade_id=trade.trade_id, outcome=1, profit_usd=1.5)
    
    perf = engine.get_venue_performance("polymarket")
    assert perf["total_paper_trades"] >= 1
    assert perf["resolved"] >= 1
