"""
Alpha score adjustment: signals must be compared against the trade DIRECTION.

`calculate_alpha_adjusted_score` existed for the whole life of the project and
was never called from the live path - it ran only in tests. So none of the
alpha signals ever moved a ranking.

Worse, the logic inside it was direction-blind. All three multipliers tested
magnitudes:

    if max_edge_ref > 0.05:            score *= 1.2   # reference odds
    if base_rate.confidence > 0.6:     score *= 1.1   # RAG base rate
    if price < 0.05 or price > 0.95:   score *= 1.15  # favourite-longshot

so a reference price contradicting the trade by 5% boosted it 20%, a base rate
pointing the other way boosted it, and the favourite-longshot rule boosted
buying an overpriced longshot - the exact trade the bias says to fade. Three
signals meant to be confirmation were confirming their own opposite.
"""
import pytest

from src.ptai.markets.base import Market, MarketSource
from src.ptai.markets.whale_tracker import WhaleSignal
from src.ptai.strategy.alpha_engine import AlphaAdjustment, AlphaEngine


def make_market(question="Will a minor candidate win the election?", price=0.5,
                liquidity=50000.0, mid="m1"):
    return Market(id=mid, source=MarketSource.POLYMARKET, question=question,
                  outcomes=["Yes", "No"], outcome_prices=[price, round(1 - price, 4)],
                  liquidity=liquidity)


@pytest.fixture
def engine():
    return AlphaEngine()


def whale(signal_type="copy_smart", side="YES", edge=0.04, should_trade=True):
    return WhaleSignal(market_id="m1", whale_address="0x" + "a" * 40, whale_score=0.8,
                       side=side, amount_usd=5000.0, market_price=0.5,
                       whale_entry_price=0.46, signal_type=signal_type,
                       edge_estimate=edge, should_trade=should_trade, reasoning="")


# ═══════════════════════════════════════════════════════════════════════════
# Direction
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("side,expected", [
    ("YES", 1), ("BUY", 1), ("LONG", 1), ("BACK", 1), ("OVER", 1), ("HOME", 1),
    ("NO", -1), ("SELL", -1), ("SHORT", -1), ("LAY", -1), ("UNDER", -1), ("AWAY", -1),
    ("yes", 1), ("no", -1),
])
def test_trade_direction_mapping(engine, side, expected):
    assert engine._trade_direction(side) == expected


def test_an_unknown_side_applies_nothing_and_says_so(engine):
    """Guessing a direction would apply signals to the wrong side of a trade."""
    adj = engine.calculate_alpha_adjustment(make_market(), side="MAYBE")
    assert adj.multiplier == 1.0
    assert adj.applied == []
    assert any("unknown trade side" in e for e in adj.errors)


def test_direction_is_recorded_for_audit(engine):
    adj = engine.calculate_alpha_adjustment(make_market(), side="NO")
    assert adj.detail["side"] == "NO" and adj.detail["direction"] == -1


# ═══════════════════════════════════════════════════════════════════════════
# Favourite-longshot - the clearest direction bug
# ═══════════════════════════════════════════════════════════════════════════

def test_fading_an_overpriced_longshot_is_boosted(engine):
    """Longshots are OVERpriced, so side NO at 3% is with the bias."""
    adj = engine.calculate_alpha_adjustment(make_market(price=0.03), side="NO")
    assert adj.multiplier > 1.0
    assert any("with the bias" in a for a in adj.applied)


def test_buying_an_overpriced_longshot_is_penalised(engine):
    """
    The exact inversion. The old rule boosted this trade by 15% because the
    price was extreme, regardless of which side was being bought.
    """
    adj = engine.calculate_alpha_adjustment(make_market(price=0.03), side="YES")
    assert adj.multiplier < 1.0
    assert any("into the bias" in a for a in adj.applied)


def test_buying_an_underpriced_favourite_is_boosted(engine):
    """Favourites are UNDERpriced, so side YES at 97% is with the bias."""
    adj = engine.calculate_alpha_adjustment(make_market(price=0.97), side="YES")
    assert adj.multiplier > 1.0


def test_fading_an_underpriced_favourite_is_penalised(engine):
    adj = engine.calculate_alpha_adjustment(make_market(price=0.97), side="NO")
    assert adj.multiplier < 1.0


def test_a_mid_price_gets_no_tail_adjustment(engine):
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert not any("bias" in a for a in adj.applied)


def test_the_tail_rule_only_applies_to_liquid_markets(engine):
    """Fading an illiquid tail cannot be executed at the quoted price."""
    adj = engine.calculate_alpha_adjustment(make_market(price=0.03, liquidity=500.0),
                                            side="NO")
    assert not any("bias" in a for a in adj.applied)


def test_tail_adjustment_is_symmetric_between_the_two_tails(engine):
    low = engine.calculate_alpha_adjustment(make_market(price=0.03), side="YES")
    high = engine.calculate_alpha_adjustment(make_market(price=0.97), side="NO")
    assert low.multiplier == pytest.approx(high.multiplier)


# ═══════════════════════════════════════════════════════════════════════════
# Whale signals
# ═══════════════════════════════════════════════════════════════════════════

def test_a_copy_signal_aligns_with_the_whales_side(engine):
    adj = engine.calculate_alpha_adjustment(make_market(), side="YES",
                                            whale_signals=[whale("copy_smart", "YES")])
    assert adj.multiplier > 1.0
    assert adj.detail["whale_aligned"] == 1


def test_a_copy_signal_does_not_align_with_the_opposite_trade(engine):
    adj = engine.calculate_alpha_adjustment(make_market(), side="NO",
                                            whale_signals=[whale("copy_smart", "YES")])
    assert adj.detail["whale_aligned"] == 0
    assert not any("whale" in a for a in adj.applied)


def test_a_fade_signal_aligns_with_the_opposite_of_the_whales_side(engine):
    """
    'Fade this dumb whale's YES' means take NO. Treating it as agreement with
    a YES trade would be exactly backwards.
    """
    adj = engine.calculate_alpha_adjustment(make_market(), side="NO",
                                            whale_signals=[whale("fade_dumb", "YES")])
    assert adj.detail["whale_aligned"] == 1
    adj2 = engine.calculate_alpha_adjustment(make_market(), side="YES",
                                             whale_signals=[whale("fade_dumb", "YES")])
    assert adj2.detail["whale_aligned"] == 0


def test_a_whale_signal_that_should_not_trade_is_ignored(engine):
    adj = engine.calculate_alpha_adjustment(make_market(), side="YES",
                                            whale_signals=[whale(should_trade=False)])
    assert adj.detail["whale_aligned"] == 0


def test_whale_signals_can_come_from_context(engine):
    ctx = {"whale_signals": [whale("copy_smart", "YES")]}
    adj = engine.calculate_alpha_adjustment(make_market(), side="YES", context=ctx)
    assert adj.detail["whale_aligned"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Reference odds
# ═══════════════════════════════════════════════════════════════════════════

def test_a_confirming_reference_boosts(engine, monkeypatch):
    """A reference above the market price confirms a YES trade."""
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="kalshi", market_id="m1", reference_price=0.70,
                        polymarket_price=0.50, edge=0.20, confidence=0.8,
                        reasoning="", should_trade=True, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.multiplier > 1.0
    assert adj.detail["references_confirming"] == 1


def test_a_contradicting_reference_penalises(engine, monkeypatch):
    """
    The core bug. A reference 20 points BELOW the market price contradicts a
    YES trade; the old `abs(edge) > 0.05` test boosted it by 20% instead.
    """
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="kalshi", market_id="m1", reference_price=0.30,
                        polymarket_price=0.50, edge=-0.20, confidence=0.8,
                        reasoning="", should_trade=True, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.multiplier < 1.0
    assert adj.detail["references_contradicting"] == 1
    assert any("contradict" in a for a in adj.applied)


def test_the_same_reference_confirms_the_opposite_side(engine, monkeypatch):
    """Direction, not magnitude, decides confirmation."""
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="kalshi", market_id="m1", reference_price=0.30,
                        polymarket_price=0.50, edge=-0.20, confidence=0.8,
                        reasoning="", should_trade=True, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    assert engine.calculate_alpha_adjustment(make_market(price=0.50),
                                             side="NO").multiplier > 1.0
    assert engine.calculate_alpha_adjustment(make_market(price=0.50),
                                             side="YES").multiplier < 1.0


def test_a_small_reference_disagreement_changes_nothing(engine, monkeypatch):
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="kalshi", market_id="m1", reference_price=0.52,
                        polymarket_price=0.50, edge=0.02, confidence=0.8,
                        reasoning="", should_trade=True, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.detail["references_confirming"] == 0
    assert adj.detail["references_contradicting"] == 0


def test_a_reference_that_is_not_tradeable_is_not_treated_as_confirmation(engine, monkeypatch):
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="manifold", market_id="m1", reference_price=0.70,
                        polymarket_price=0.50, edge=0.20, confidence=0.2,
                        reasoning="", should_trade=False, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.detail["references_confirming"] == 0


def test_unreachable_reference_sources_are_not_an_error(engine):
    """No anchor is a normal outcome, not a failure - it must not be reported as one."""
    adj = engine.calculate_alpha_adjustment(make_market(
        question="Will the Lakers win the NBA finals?"), side="YES")
    assert adj.detail["references"] == 0
    assert not any("reference_odds" in e for e in adj.errors)


# ═══════════════════════════════════════════════════════════════════════════
# RAG base rate
# ═══════════════════════════════════════════════════════════════════════════

class _BaseRate:
    def __init__(self, base_rate, confidence=0.9, n=12):
        self.base_rate, self.confidence, self.num_similar = base_rate, confidence, n


def test_an_agreeing_base_rate_boosts(engine, monkeypatch):
    monkeypatch.setattr(engine.rag, "estimate_base_rate",
                        lambda **kw: _BaseRate(0.75))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.multiplier > 1.0
    assert any("RAG" in a and "agrees" in a for a in adj.applied)


def test_a_disagreeing_base_rate_penalises(engine, monkeypatch):
    """
    The old rule boosted on confidence alone, so a base rate of 0.25 against a
    YES trade at 0.50 still increased the score.
    """
    monkeypatch.setattr(engine.rag, "estimate_base_rate",
                        lambda **kw: _BaseRate(0.25))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert adj.multiplier < 1.0
    assert any("disagrees" in a for a in adj.applied)


def test_a_weak_base_rate_is_ignored(engine, monkeypatch):
    monkeypatch.setattr(engine.rag, "estimate_base_rate",
                        lambda **kw: _BaseRate(0.90, confidence=0.2))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert not any("RAG" in a for a in adj.applied)


def test_a_base_rate_from_no_history_is_ignored(engine, monkeypatch):
    """An empty history returning 0.5 is not evidence."""
    monkeypatch.setattr(engine.rag, "estimate_base_rate",
                        lambda **kw: _BaseRate(0.5, confidence=0.9, n=0))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.50), side="YES")
    assert not any("RAG" in a for a in adj.applied)


def test_the_category_is_taken_from_the_market_not_hardcoded(engine, monkeypatch):
    """It used to pass category="unknown" for every market."""
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return _BaseRate(0.5, confidence=0.1, n=0)

    monkeypatch.setattr(engine.rag, "estimate_base_rate", spy)
    market = make_market()
    market.raw["category"] = "politics"
    engine.calculate_alpha_adjustment(market, side="YES")
    assert seen["category"] == "politics"


def test_the_category_falls_back_to_context(engine, monkeypatch):
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return _BaseRate(0.5, confidence=0.1, n=0)

    monkeypatch.setattr(engine.rag, "estimate_base_rate", spy)
    engine.calculate_alpha_adjustment(make_market(), side="YES",
                                      context={"category": "sports"})
    assert seen["category"] == "sports"


# ═══════════════════════════════════════════════════════════════════════════
# Errors are recorded, not swallowed
# ═══════════════════════════════════════════════════════════════════════════

def test_a_failing_signal_is_recorded_not_swallowed(engine, monkeypatch):
    """The old code wrapped every block in `except: pass`."""
    def boom(**kw):
        raise ValueError("rag exploded")

    monkeypatch.setattr(engine.rag, "estimate_base_rate", boom)
    adj = engine.calculate_alpha_adjustment(make_market(), side="YES")
    assert any("rag exploded" in e for e in adj.errors)


def test_one_failing_signal_does_not_stop_the_others(engine, monkeypatch):
    def boom(**kw):
        raise ValueError("rag exploded")

    monkeypatch.setattr(engine.rag, "estimate_base_rate", boom)
    adj = engine.calculate_alpha_adjustment(make_market(price=0.03), side="NO",
                                            whale_signals=[whale("copy_smart", "NO")])
    assert adj.errors                      # rag failed
    assert adj.multiplier > 1.0            # tail + whale still applied


# ═══════════════════════════════════════════════════════════════════════════
# Composition, bounds and back-compat
# ═══════════════════════════════════════════════════════════════════════════

def test_signals_compose_multiplicatively(engine, monkeypatch):
    """Each signal multiplies; a tail price cannot also agree with a RAG rate."""
    # tail (x1.15) + whale (x1.10), RAG held neutral by a matching base rate
    monkeypatch.setattr(engine.rag, "estimate_base_rate", lambda **kw: _BaseRate(0.97))
    tail_and_whale = engine.calculate_alpha_adjustment(
        make_market(price=0.97), side="YES", whale_signals=[whale("copy_smart", "YES")])
    assert tail_and_whale.multiplier == pytest.approx(1.15 * 1.10, abs=1e-3)
    assert len(tail_and_whale.applied) == 2

    # RAG agreement + whale (x1.10) at a mid price where no tail rule applies
    monkeypatch.setattr(engine.rag, "estimate_base_rate", lambda **kw: _BaseRate(0.90))
    rag_and_whale = engine.calculate_alpha_adjustment(
        make_market(price=0.50), side="YES", whale_signals=[whale("copy_smart", "YES")])
    assert len(rag_and_whale.applied) == 2
    assert rag_and_whale.multiplier > 1.10
    assert rag_and_whale.multiplier == pytest.approx(1.10 * 1.09, abs=1e-3)


def test_a_contradiction_can_outweigh_a_boost(engine, monkeypatch):
    from src.ptai.strategy.reference_odds import ReferenceOdds
    ref = ReferenceOdds(source="kalshi", market_id="m1", reference_price=0.05,
                        polymarket_price=0.50, edge=-0.45, confidence=0.9,
                        reasoning="", should_trade=True, category="general")
    monkeypatch.setattr(engine.reference_odds, "get_all_reference_odds",
                        lambda m, k=None: [ref])
    monkeypatch.setattr(engine.rag, "estimate_base_rate", lambda **kw: _BaseRate(0.90))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.97), side="YES")
    assert adj.multiplier < 1.0


def test_adjustment_flags(engine):
    assert AlphaAdjustment(1.2).boosted and not AlphaAdjustment(1.2).penalised
    assert AlphaAdjustment(0.8).penalised and not AlphaAdjustment(0.8).boosted
    assert not AlphaAdjustment(1.0).boosted and not AlphaAdjustment(1.0).penalised


def test_the_multiplier_is_bounded(engine, monkeypatch):
    """No combination of signals should be able to multiply a score wildly."""
    monkeypatch.setattr(engine.rag, "estimate_base_rate", lambda **kw: _BaseRate(0.99))
    adj = engine.calculate_alpha_adjustment(make_market(price=0.99), side="YES",
                                            whale_signals=[whale("copy_smart", "YES")])
    assert 0.5 <= adj.multiplier <= 2.0


def test_backward_compatible_wrapper_still_works(engine):
    """Existing callers pass (base_score, market) and expect a float."""
    assert engine.calculate_alpha_adjusted_score(1.0, make_market(price=0.50)) == pytest.approx(1.0)
    assert engine.calculate_alpha_adjusted_score(2.0, make_market(price=0.03),
                                                 context={"side": "NO"}) > 2.0


def test_the_wrapper_reads_the_side_from_context(engine):
    yes = engine.calculate_alpha_adjusted_score(1.0, make_market(price=0.03),
                                                context={"side": "YES"})
    no = engine.calculate_alpha_adjusted_score(1.0, make_market(price=0.03),
                                               context={"side": "NO"})
    assert no > yes


# ═══════════════════════════════════════════════════════════════════════════
# It is actually wired into the live scoring path
# ═══════════════════════════════════════════════════════════════════════════

def test_the_opportunity_engine_holds_an_alpha_engine():
    """
    The function was dead code: defined, tested, and never called by anything
    in the live path. This is the regression guard on that.
    """
    from src.ptai.strategy.opportunity import OpportunityEngine
    assert OpportunityEngine().alpha_engine is not None


def test_the_scoring_path_calls_the_adjustment():
    """The call site must exist in the selection code, not only in tests."""
    import inspect
    from src.ptai.strategy import opportunity
    source = inspect.getsource(opportunity)
    assert "calculate_alpha_adjustment" in source
    assert 'opp.raw["alpha_adjustment"]' in source


def test_the_adjustment_is_recorded_on_the_opportunity():
    """A boosted score with no record of why cannot be audited."""
    import inspect
    from src.ptai.strategy import opportunity
    src = inspect.getsource(opportunity.OpportunityEngine)
    assert "alpha_adjustment" in src
    assert "multiplier" in src
