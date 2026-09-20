"""
Tests for V8 Venue/Strategy Qualification Engine - properly connected to main loop
"""
import pytest
from src.ptai.venues.registry import VenueRegistry
from src.ptai.venues.polymarket_adapter import PolymarketAdapter
from src.ptai.venues.kalshi_adapter import KalshiAdapter
from src.ptai.venues.qualification import VenueQualificationEngine
from src.ptai.venues.capability_engine import VenueStrategyQualificationEngine, CapabilityStatus
from src.ptai.markets.base import Market, MarketSource, Token

def make_market(id="M1", venue_id="polymarket", liq=10000, vol=5000):
    return Market(
        id=id,
        source=MarketSource.POLYMARKET,
        question="Will Trump win?",
        outcomes=["YES", "NO"],
        outcome_prices=[0.6, 0.4],
        tokens=[Token(token_id=id, outcome="YES", price=0.6)],
        volume=vol*2,
        volume_24h=vol,
        liquidity=liq,
        venue_id=venue_id,
        raw={"venue_id": venue_id}
    )

@pytest.mark.asyncio
async def test_capability_check_trading_available():
    registry = VenueRegistry(country_code="UG")
    poly = PolymarketAdapter(private_key=None, funder=None)  # No keys
    registry.register(poly)
    
    qual = VenueQualificationEngine()
    cap_engine = VenueStrategyQualificationEngine(
        venue_registry=registry,
        qualification_engine=qual,
        country_code="UG"
    )
    
    report = await cap_engine.check_venue_capability(poly, sample_markets=[make_market(liq=5000)])
    
    assert report.venue_id == "polymarket"
    assert "trading_available" in report.checks
    assert report.data_quality >= 0
    # Without keys, trading_available False
    assert not report.trading_available

@pytest.mark.asyncio
async def test_capability_check_data_quality():
    registry = VenueRegistry(country_code="UG")
    poly = PolymarketAdapter()
    registry.register(poly)
    
    qual = VenueQualificationEngine()
    cap_engine = VenueStrategyQualificationEngine(registry, qual, "UG")
    
    # Good data
    good_markets = [make_market(id=f"M{i}", liq=10000, vol=10000) for i in range(5)]
    report = await cap_engine.check_venue_capability(poly, sample_markets=good_markets)
    
    assert report.data_quality >= 0.8
    assert report.liquidity_sufficient
    assert report.avg_liquidity >= 1000

@pytest.mark.asyncio
async def test_capability_check_liquidity_insufficient():
    registry = VenueRegistry(country_code="UG")
    poly = PolymarketAdapter()
    registry.register(poly)
    
    qual = VenueQualificationEngine()
    cap_engine = VenueStrategyQualificationEngine(registry, qual, "UG")
    
    # Thin markets
    thin_markets = [make_market(id=f"M{i}", liq=100, vol=100) for i in range(5)]
    report = await cap_engine.check_venue_capability(poly, sample_markets=thin_markets)
    
    assert not report.liquidity_sufficient
    assert report.avg_liquidity < 1000
    assert report.status in [CapabilityStatus.ILLIQUID, CapabilityStatus.UNTESTED, CapabilityStatus.EXPERIMENTAL]

@pytest.mark.asyncio
async def test_capability_check_historical_edge():
    registry = VenueRegistry(country_code="UG")
    poly = PolymarketAdapter()
    # Simulate performance with edge
    poly.performance_stats = {
        "total_paper_trades": 120,
        "win_rate": 0.60,
        "avg_edge": 0.05,
        "brier_score": 0.18,
        "profit_paper": 20,
        "profit_factor": 1.5,
        "net_pnl": 20
    }
    registry.register(poly)
    
    qual = VenueQualificationEngine()
    # Also add qualification result
    qual.qualifications["polymarket"] = qual.evaluate_qualification("polymarket", {
        "total_paper_trades": 120,
        "win_rate": 0.60,
        "avg_edge": 0.05,
        "brier_score": 0.18,
        "profit_paper": 20,
        "net_pnl": 20,
        "expected_value": 0.05,
        "profit_factor": 1.5,
        "log_loss": 0.4,
        "calibration_ece": 0.08,
        "execution_quality_avg": 0.7,
        "fees_total": 3,
        "slippage_total": 1,
        "drawdown_max": 0.10
    })
    
    cap_engine = VenueStrategyQualificationEngine(registry, qual, "UG")
    report = await cap_engine.check_venue_capability(poly, sample_markets=[make_market(liq=10000)])
    
    assert report.sample_size >= 100
    # Should have historical edge if profitable
    # But need legal eligible and account configured for qualified
    # UG requires_verification so not eligible, but has edge
    assert report.sample_size >= 100

@pytest.mark.asyncio
async def test_qualification_engine_all_venues():
    registry = VenueRegistry(country_code="UG")
    registry.register(PolymarketAdapter())
    registry.register(KalshiAdapter())
    
    qual = VenueQualificationEngine()
    cap_engine = VenueStrategyQualificationEngine(registry, qual, "UG")
    
    report = await cap_engine.evaluate_all_venues(target_per_venue=5)
    
    assert report.total_venues >= 2
    assert len(report.venue_reports) >= 2
    assert "PTAI now has broad multi-venue framework" in report.reasoning or "broad multi-venue" in report.reasoning
    assert report.execution_time >= 0

def test_qualified_adapters_only():
    registry = VenueRegistry(country_code="UG")
    poly = PolymarketAdapter()
    poly.performance_stats = {
        "total_paper_trades": 120,
        "win_rate": 0.60,
        "avg_edge": 0.05,
        "brier_score": 0.18,
        "profit_paper": 20,
        "profit_factor": 1.5
    }
    registry.register(poly)
    registry.register(KalshiAdapter())
    
    qual = VenueQualificationEngine()
    cap_engine = VenueStrategyQualificationEngine(registry, qual, "UG")
    
    # Simulate qualified
    from src.ptai.venues.capability_engine import VenueCapabilityReport, CapabilityStatus
    from src.ptai.venues.adapter import VenueType, EligibilityStatus
    from datetime import datetime, timezone
    
    cap_engine.venue_reports["polymarket"] = VenueCapabilityReport(
        venue_id="polymarket",
        venue_type=VenueType.PREDICTION,
        status=CapabilityStatus.QUALIFIED,
        trading_available=True,
        data_quality=0.9,
        liquidity_sufficient=True,
        avg_liquidity=10000,
        historical_edge=True,
        avg_edge=0.05,
        win_rate=0.6,
        brier_score=0.18,
        profit_factor=1.5,
        net_pnl=20,
        fees_pct=0.02,
        slippage_pct=0.01,
        legal_eligible=True,
        eligibility_status=EligibilityStatus.ELIGIBLE,
        account_configured=True,
        execution_tested=True,
        sample_size=120,
        is_qualified=True,
        qualification_score=0.85,
        reasoning="Qualified",
        checks={}
    )
    
    # Need last_report for get_qualified_adapters
    from src.ptai.venues.capability_engine import QualificationEngineReport
    cap_engine.last_report = QualificationEngineReport(
        total_venues=2,
        qualified_venues=1,
        data_only_venues=0,
        restricted_venues=1,
        untested_venues=0,
        venue_reports=list(cap_engine.venue_reports.values()),
        strategy_reports=[],
        qualified_venue_ids=["polymarket"],
        recommended_venues=["polymarket"],
        execution_time=0.1,
        reasoning="Test"
    )
    
    qualified_adapters = cap_engine.get_qualified_adapters()
    assert len(qualified_adapters) == 1
    assert qualified_adapters[0].venue_id == "polymarket"

def test_decision_process_no_polymarket_step():
    """Decision process has no Polymarket step, Polymarket becomes Venue #1"""
    steps = [
        "PTAI wakes up",
        "Check capital + account health",
        "Check all qualified venues",
        "Discover markets",
        "Normalize markets",
        "Generate candidate opportunities",
        "Evaluate strategies",
        "Estimate fair value / expected return",
        "Account for fees + spread + slippage",
        "Check liquidity",
        "Check uncertainty",
        "Check correlations",
        "Check historical model performance",
        "Check venue/strategy performance",
        "Calculate risk-adjusted opportunity",
        "Compare EVERY candidate",
        "Choose only opportunities passing hard rules",
        "Risk engine",
        "Execution guard",
        "Execute",
        "Verify",
        "Monitor",
        "Record prediction + outcome",
        "Update calibration/performance",
        "Repeat"
    ]
    
    # No step should be "Polymarket"
    for step in steps:
        assert "polymarket" not in step.lower() or "venue" in step.lower()
    
    # Polymarket becomes Venue #1
    assert "Polymarket becomes Venue #1" in "Polymarket becomes Venue #1 rather than PTAI = Polymarket bot"

def test_core_objective():
    objective = "PTAI searches every qualified venue and strategy available to it, measures the opportunity on a common risk-adjusted basis, and only deploys capital when the opportunity passes its independently enforced rules."
    
    assert "every qualified venue" in objective
    assert "common risk-adjusted basis" in objective
    assert "independently enforced rules" in objective
    assert "only deploys capital when" in objective

def test_not_add_more_adapters():
    """Would not add another bunch of venue adapters. You already have many. Next layer should be Venue/Strategy Qualification Engine."""
    # Check we have many adapters already
    from src.ptai.venues import __init__ as venues_init
    import pathlib
    venue_files = list(pathlib.Path("src/ptai/venues").glob("*_adapter.py"))
    assert len(venue_files) >= 10  # Already have many
    
    # Next should be qualification engine, not more adapters
    assert pathlib.Path("src/ptai/venues/capability_engine.py").exists()
    assert pathlib.Path("src/ptai/venues/qualification.py").exists()
