
"""
Tests for the multi-venue expansion.

These tests used to assert that eighteen venues discovered markets. Most of
those venues had no client at all - they returned hardcoded rows - so the tests
were asserting the fabrication. `test_veynor_intelligence` checked that a
whale-trade list contained the word "whale"; `test_openpx_sub_ms` checked that a
fabricated orderbook reported "sub-millisecond" latency.

The expansion is still real: eighteen venues are registered and reachable
through one interface. What is now tested is which of them actually work, and
that the ones which do not say so instead of inventing markets.
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
    assert "whitebit" in registry.adapters
    assert "afx_dex" in registry.adapters
    assert "ccxt_unified" in registry.adapters
    assert "veynor" in registry.adapters
    assert "openpx" in registry.adapters
    # The legacy Betfair stub no longer claims venue_id "betfair" - that
    # belongs to the real exchange adapter, and registering the stub under that
    # id shadowed it.
    assert "betfair_legacy_stub" in registry.adapters
    assert "betfair" not in registry.adapters

@pytest.mark.asyncio
async def test_venue_discovery_returns_no_fabricated_markets():
    """
    Every adapter is called and checked for fabrication.

    This previously asserted `len(markets) > 0` for all fifteen, which could
    only pass because fourteen of them returned invented rows. With the network
    blocked the honest count is zero, and the assertion that matters is that
    nothing fabricated comes back.
    """
    adapters = [
        PolymarketAdapter(), KalshiAdapter(), ManifoldAdapter(),
        PredictItAdapter(), SimmerAdapter(), CymeticaAdapter(), WhiteBITAdapter(),
        AFXAdapter(), GRVTAdapter(), PionexAdapter(), BetfairAdapter(),
        CCXTUnifiedAdapter(), VeynorAdapter(), OpenPXAdapter(), ApifyAdapter(),
    ]
    for adapter in adapters:
        markets = await adapter.discover_markets(target_count=5)
        for m in markets:
            assert not m.is_mock, f"{adapter.venue_id} returned a mock market"
            assert "MOCK" not in str(m.id).upper(), \
                f"{adapter.venue_id} returned fabricated id {m.id}"
            assert "MOCK" not in (m.question or "").upper(), \
                f"{adapter.venue_id} fabricated a question"


@pytest.mark.asyncio
async def test_unimplemented_venues_declare_themselves():
    """A venue with no client must say so rather than returning markets."""
    from src.ptai.venues.adapter import STATUS_UNIMPLEMENTED
    for adapter in [SimmerAdapter(), CymeticaAdapter(), AFXAdapter(), GRVTAdapter(),
                    PionexAdapter(), CCXTUnifiedAdapter(), VeynorAdapter(),
                    OpenPXAdapter(), BetdaqAdapter(), BetConnectAdapter()]:
        assert adapter.capabilities.implementation_status == STATUS_UNIMPLEMENTED, \
            adapter.venue_id
        assert adapter.capabilities.supports_market_discovery is False, adapter.venue_id
        assert await adapter.discover_markets() == [], adapter.venue_id
        # UNKNOWN, not ELIGIBLE: an adapter with no client cannot be traded on.
        assert adapter.check_eligibility("UG").value == "unknown", adapter.venue_id


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

def test_ccxt_unified_is_declared_unimplemented():
    """
    This asserted `get_unified_ticker` returned cross-venue prices.

    ccxt is not installed and the method no longer exists - it returned invented
    tickers. Polymarket and Kalshi have real adapters, so this layer is
    redundant rather than merely unbuilt.
    """
    adapter = CCXTUnifiedAdapter(venues=["polymarket", "kalshi", "binance"])
    assert adapter.venue_id == "ccxt_unified"
    assert adapter.capabilities.implementation_status == "unimplemented"
    assert adapter.capabilities.supports_market_discovery is False


def test_veynor_is_declared_unimplemented():
    """
    This asserted a whale-trade list contained the word "whale".

    There is no Veynor client. Whale signals now come from
    markets/whale_tracker.py, which reads public Polymarket activity.
    """
    adapter = VeynorAdapter()
    assert adapter.venue_id == "veynor"
    assert adapter.capabilities.implementation_status == "unimplemented"
    assert not hasattr(adapter, "get_whale_trades")


def test_whitebit_low_minimum():
    adapter = WhiteBITAdapter()
    assert adapter.capabilities.min_order_usd == 1.0
    assert adapter.capabilities.fee_taker_pct == 0.001
    # Check eligibility
    status_ug = adapter.check_eligibility("UG")
    assert status_ug.value in ["eligible", "requires_verification", "restricted"]

def test_afx_dex_is_declared_unimplemented():
    """
    This asserted `min_deposit == 10.0` and a 0.05% fee.

    Neither was real - there is no wallet-signed request client, no EIP-712
    signing and no deposit flow. The constants described a venue, not code.
    """
    adapter = AFXAdapter(wallet_address="0x1234", private_key="0xabc")
    assert adapter.venue_id == "afx_dex"
    assert adapter.capabilities.implementation_status == "unimplemented"
    assert not hasattr(adapter, "min_deposit")


def test_betfair_lay_is_tested_against_the_real_adapter():
    """
    Lay betting is real, but not on this stub.

    The legacy adapter asserted a 5% fee and returned invented markets carrying
    `lay_available: True`. Lay support lives in the exchange adapter, which
    carries both sides of the ladder from real Betfair data.
    """
    legacy = BetfairAdapter()
    assert legacy.capabilities.implementation_status == "unimplemented"
    assert legacy.venue_id != "betfair", \
        "the stub must not share the venue_id of the working adapter"

    from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter, BetfairClient
    real = BetfairExchangeAdapter()
    assert real.venue_id == "betfair"
    assert real.capabilities.implementation_status == "live"
    # Betfair commission is charged on net winnings, not as a taker fee.
    assert real.capabilities.supports_orderbook is True
    assert real.capabilities.supports_trading is False, \
        "order placement is deliberately not implemented"


def test_openpx_is_declared_unimplemented():
    """
    This asserted a fabricated orderbook reported "sub-millisecond" latency.

    No Rust binding exists, and a latency figure for a client that was never
    written is not a capability.
    """
    adapter = OpenPXAdapter()
    assert adapter.venue_id == "openpx"
    assert adapter.capabilities.implementation_status == "unimplemented"

    import asyncio
    ob = asyncio.run(adapter.get_orderbook(make_market()))
    assert ob["available"] is False
    assert "reason" in ob
    assert "latency" not in ob


def test_registry_capability_report_separates_live_from_declared():
    """The UI needs to distinguish reachable venues from merely-registered ones."""
    registry = VenueRegistry(country_code="UG")
    registry.register(PolymarketAdapter())
    registry.register(SimmerAdapter())
    from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
    registry.register(BetfairExchangeAdapter())

    report = registry.capability_report()
    assert report["total_registered"] == 3
    assert "polymarket" in report["live"]
    assert "betfair" in report["live"]
    assert "simmer" in report["unimplemented"]
    assert "simmer" not in report["live"]
    assert report["tradeable"] == [], "nothing here places orders"
    # every venue reports what is missing
    for v in report["venues"]:
        if v["implementation_status"] == "unimplemented":
            assert v["note"], v["venue_id"]
