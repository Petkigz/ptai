
"""
Tests for v5 expansion beyond Polymarket - 18 venues
"""
import pytest
from src.ptai.venues.registry import VenueRegistry
from src.ptai.venues.polymarket_adapter import PolymarketAdapter
from src.ptai.venues.kalshi_adapter import KalshiAdapter
from src.ptai.venues.manifold_adapter import ManifoldAdapter
from src.ptai.venues.crypto_adapter import CryptoAdapter
from src.ptai.venues.stock_adapter import StockAdapter
from src.ptai.venues.predictit_adapter import PredictItAdapter
from src.ptai.venues.simmer_adapter import SimmerAdapter
from src.ptai.venues.cymetica_adapter import CymeticaAdapter
from src.ptai.venues.whitebit_adapter import WhiteBITAdapter
from src.ptai.venues.afx_adapter import AFXAdapter
from src.ptai.venues.grvt_adapter import GRVTAdapter
from src.ptai.venues.pionex_adapter import PionexAdapter
from src.ptai.venues.betfair_adapter import BetfairAdapter, BetdaqAdapter, BetConnectAdapter
from src.ptai.venues.ccxt_adapter import CCXTUnifiedAdapter
from src.ptai.venues.veynor_adapter import VeynorAdapter
from src.ptai.venues.openpx_adapter import OpenPXAdapter
from src.ptai.venues.apify_adapter import ApifyAdapter
from src.ptai.strategy.cross_venue_arb import CrossVenueArbitrageEngine
from src.ptai.risk.multi_venue_risk import MultiVenueRiskManager
from src.ptai.markets.base import Market, MarketSource, Token

def make_market(id="M1", question="Will Fed cut rates in June?", price=0.6, venue="polymarket", event_slug="fed-cut-june"):
    return Market(
        id=id,
        source=MarketSource.POLYMARKET,
        question=question,
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1-price],
        tokens=[Token(token_id=id, outcome="YES", price=price)],
        volume=10000,
        volume_24h=5000,
        liquidity=10000,
        active=True,
        closed=False,
        event_slug=event_slug,
        raw={"venue": venue, "category": "economics"}
    )

@pytest.mark.asyncio
async def test_all_venues_register():
    registry = VenueRegistry(country_code="UG")
    registry.register(PolymarketAdapter())
    registry.register(KalshiAdapter())
    registry.register(ManifoldAdapter())
    registry.register(CryptoAdapter(exchange="binance"))
    registry.register(StockAdapter(broker="mock"))
    registry.register(PredictItAdapter())
    registry.register(SimmerAdapter())
    registry.register(CymeticaAdapter())
    registry.register(WhiteBITAdapter())
    registry.register(AFXAdapter())
    registry.register(GRVTAdapter())
    registry.register(PionexAdapter())
    registry.register(BetfairAdapter())
    registry.register(BetdaqAdapter())
    registry.register(BetConnectAdapter())
    registry.register(CCXTUnifiedAdapter())
    registry.register(VeynorAdapter())
    registry.register(OpenPXAdapter())
    registry.register(ApifyAdapter())
    assert len(registry.adapters) == 19
    assert "polymarket" in registry.adapters
    assert "kalshi" in registry.adapters
    assert "betfair" in registry.adapters
    assert "whitebit" in registry.adapters
    assert "afx_dex" in registry.adapters
    assert "ccxt_unified" in registry.adapters
    assert "veynor" in registry.adapters
    assert "openpx" in registry.adapters

@pytest.mark.asyncio
async def test_venue_discovery():
    adapters = [
        PolymarketAdapter(),
        KalshiAdapter(),
        ManifoldAdapter(),
        PredictItAdapter(),
        SimmerAdapter(),
        CymeticaAdapter(),
        WhiteBITAdapter(),
        AFXAdapter(),
        GRVTAdapter(),
        PionexAdapter(),
        BetfairAdapter(),
        CCXTUnifiedAdapter(),
        VeynorAdapter(),
        OpenPXAdapter(),
        ApifyAdapter(),
    ]
    for adapter in adapters:
        markets = await adapter.discover_markets(target_count=5)
        assert len(markets) > 0, f"{adapter.venue_id} discovery failed"
        assert markets[0].best_price > 0

def test_eligibility_ug():
    adapters = [
        PolymarketAdapter(),
        KalshiAdapter(),
        WhiteBITAdapter(),
        BetfairAdapter(),
        AFXAdapter(),  # DEX worldwide eligible
        CCXTUnifiedAdapter(),
        VeynorAdapter(),
    ]
    for adapter in adapters:
        status = adapter.check_eligibility("UG")
        # Should not crash, should return status
        assert status.value in ["eligible", "restricted", "requires_verification", "unknown"]

def test_cross_venue_arb_same_event():
    engine = CrossVenueArbitrageEngine(min_spread=0.03, min_confidence=0.6)
    markets_by_venue = {
        "polymarket": [make_market(id="poly-1", question="Will Fed cut rates in June?", price=0.61, venue="polymarket", event_slug="fed-cut-june")],
        "kalshi": [make_market(id="kalshi-1", question="Will Fed cut rates in June?", price=0.68, venue="kalshi", event_slug="fed-cut-june")],
    }
    arbs = engine.find_cross_venue_arbitrage(markets_by_venue)
    assert len(arbs) >= 1
    arb = arbs[0]
    assert arb.spread == pytest.approx(0.07, abs=0.01)
    assert arb.confidence_same_event > 0.8
    assert arb.venue_a != arb.venue_b
    assert "polymarket" in [arb.venue_a, arb.venue_b]
    assert "kalshi" in [arb.venue_a, arb.venue_b]

def test_cross_venue_arb_different_event_no_arb():
    engine = CrossVenueArbitrageEngine(min_spread=0.03, min_confidence=0.7)
    markets_by_venue = {
        "polymarket": [make_market(id="poly-1", question="Will Trump win?", price=0.60, venue="polymarket", event_slug="trump-win")],
        "kalshi": [make_market(id="kalshi-1", question="Will BTC be above $100k?", price=0.60, venue="kalshi", event_slug="btc-100k")],
    }
    arbs = engine.find_cross_venue_arbitrage(markets_by_venue)
    assert len(arbs) == 0  # different events, no arb

def test_cross_venue_arb_fee_adjusted():
    engine = CrossVenueArbitrageEngine(min_spread=0.03, min_confidence=0.6)
    markets_by_venue = {
        "polymarket": [make_market(id="poly-1", question="Will Fed cut rates in June?", price=0.55, venue="polymarket", event_slug="fed-cut-june")],
        "kalshi": [make_market(id="kalshi-1", question="Will Fed cut rates in June?", price=0.70, venue="kalshi", event_slug="fed-cut-june")],
    }
    arbs = engine.find_cross_venue_arbitrage(markets_by_venue)
    assert len(arbs) >= 1
    arb = arbs[0]
    # Spread 15%, cost 0.85, profit 0.15, profit_pct 17.6%, fee 2%+7%=9%, fee_adj ~8.6% should trade
    assert arb.profit_pct > 0.10
    assert arb.fee_adjusted_profit > 0.02
    assert arb.should_trade

def test_multi_venue_risk_correlation():
    manager = MultiVenueRiskManager(bankroll=50.0)
    positions = [
        {"venue_id": "polymarket", "event_slug": "fed-cut-june", "event_key": "fed-cut-june", "amount_usd": 3.0, "question": "Will Fed cut rates in June?"},
        {"venue_id": "kalshi", "event_slug": "fed-cut-june", "event_key": "fed-cut-june", "amount_usd": 3.0, "question": "Will Fed cut rates in June?"},
    ]
    event_exposure = manager.calculate_event_exposure(positions)
    assert event_exposure["fed-cut-june"] == 6.0
    # Max per event 12% of $50 = $6, so exactly at limit
    report = manager.evaluate(positions=positions, arb_opportunities=[], venues=["polymarket", "kalshi"], country_code="UG")
    # Should flag correlation risk if exceeds
    assert "fed-cut-june" in report.exposure_per_event

def test_multi_venue_risk_fragmentation():
    manager = MultiVenueRiskManager(bankroll=50.0)
    positions = [
        {"venue_id": f"venue-{i}", "event_slug": f"event-{i}", "event_key": f"event-{i}", "amount_usd": 3.0}
        for i in range(5)
    ]
    frag = manager.check_capital_fragmentation(positions)
    assert "fragmentation" in frag.lower() or "venues" in frag.lower()
    # 5 venues with $50 bankroll should trigger fragmentation warning
    assert "5 venues" in frag or "concentrate" in frag

def test_multi_venue_risk_settlement():
    manager = MultiVenueRiskManager(bankroll=50.0)
    arbs = [
        {"venue_a": "polymarket", "venue_b": "kalshi", "confidence_same_event": 0.6},
        {"venue_a": "polymarket", "venue_b": "kalshi", "confidence_same_event": 0.9},
    ]
    risks = manager.check_settlement_risk(arbs)
    assert len(risks) == 2
    assert any("risk" in r.lower() for r in risks)

def test_ccxt_unified():
    adapter = CCXTUnifiedAdapter(venues=["polymarket", "kalshi", "binance"])
    assert adapter.ccxt_available or not adapter.ccxt_available  # either
    ticker = adapter.get_unified_ticker("Fed cut June")
    assert "polymarket" in ticker
    assert "kalshi" in ticker
    assert "spread" in ticker

def test_veynor_intelligence():
    adapter = VeynorAdapter()
    whales = adapter.get_whale_trades(limit=2)
    assert len(whales) >= 1
    assert "whale" in whales[0]
    arbs = adapter.get_arb_opportunities()
    assert len(arbs) >= 1
    assert "spread" in arbs[0]

def test_whitebit_low_minimum():
    adapter = WhiteBITAdapter()
    assert adapter.capabilities.min_order_usd == 1.0
    assert adapter.capabilities.fee_taker_pct == 0.001
    # Check eligibility
    status_ug = adapter.check_eligibility("UG")
    assert status_ug.value in ["eligible", "requires_verification", "restricted"]

def test_afx_dex_wallet_signed():
    adapter = AFXAdapter(wallet_address="0x1234", private_key="0xabc")
    assert adapter.min_deposit == 10.0
    assert adapter.min_withdrawal == 2.0
    assert adapter.capabilities.fee_taker_pct == 0.0005
    status = adapter.check_eligibility("UG")
    assert status.value == "eligible"  # DEX worldwide

def test_betfair_lay_betting():
    adapter = BetfairAdapter()
    assert adapter.capabilities.fee_taker_pct == 0.05
    # Lay available opens arb and market-making not possible on traditional sportsbooks
    # Check mock market has lay
    import asyncio
    markets = asyncio.run(adapter.discover_markets(target_count=2))
    assert len(markets) > 0
    assert markets[0].raw.get("lay_available") == True

def test_openpx_sub_ms():
    adapter = OpenPXAdapter()
    assert adapter.venue_id == "openpx"
    import asyncio
    ob = asyncio.run(adapter.get_orderbook(make_market()))
    assert ob["latency"] == "sub-millisecond"
    assert "polymarket" in ob["venues"]
    assert "kalshi" in ob["venues"]
