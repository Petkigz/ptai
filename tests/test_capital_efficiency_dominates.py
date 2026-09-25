"""
NET EV / (CAPITAL x TIME x EXECUTION RISK) is the decision variable.

The seventh report's fifth item. What was there instead:

  * `final_score` was a product of five terms - net_ev x ev_per_dollar x
    liquidity x execution_quality x time_efficiency x ev_per_risk / uncertainty -
    and capital-time was a `*= (1 + ev_per_capital_time)` bonus on the end of it.
    A term that ranges from one hour to one month cannot be expressed as a
    multiplier between 1 and something small;
  * three of the terms were the same things counted again (net EV twice,
    execution quality twice, uncertainty twice);
  * and it could not affect WHICH candidates were considered: the list arrived
    ordered by the score each opportunity carried before its EV existed, and
    the loop stopped at `max_trades`, so the ratio only ever reordered the
    trades the old ordering had already let through.

The fixtures below are two real trades, priced off real books, both of which
clear every hard gate. One earns four times as much money and locks the capital
up for thirty days; the other earns a fifth as much and is free in two hours.
"""

from __future__ import annotations

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.strategy.expected_ev import ExpectedNetEVEngine
from src.ptai.strategy.opportunity import OpportunityEngine
from src.ptai.venues.adapter import VenueOpportunity, VenueType

STAKE = 3.0


def _book(price: float, size: float = 5000.0):
    return {
        "bids": [{"price": str(round(price - 0.01, 4)), "size": str(size)}],
        "asks": [{"price": str(price), "size": str(size)}],
        "spread": 0.01, "is_real": True, "source": "clob",
    }


def _opp(market_id: str, price: float, fair: float, hours: float,
         execution_quality: float = 0.95) -> VenueOpportunity:
    market = Market(
        id=market_id, source=MarketSource.POLYMARKET,
        question=f"Will {market_id} happen?", outcomes=["YES", "NO"],
        outcome_prices=[price, 1 - price],
        tokens=[Token(token_id=market_id, outcome="YES", price=price)],
        volume=200_000.0, volume_24h=100_000.0, liquidity=50_000.0,
        raw={"orderbook": _book(price)},
    )
    return VenueOpportunity(
        market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION,
        side="YES", market_price=price, estimated_fair=fair,
        raw_edge=round(fair - price, 4), effective_edge=round(fair - price, 4),
        confidence=0.8, uncertainty=0.1, liquidity_score=0.9,
        execution_quality=execution_quality, category="politics",
        should_trade=True, time_to_resolution_hours=hours,
        fees_pct=0.02, spread_pct=0.01, slippage_pct=None, order_gas_usd=0.0,
    )


def _big_slow():
    """20c of edge, thirty days of locked capital, ordinary execution."""
    return _opp("SLOW", price=0.60, fair=0.80, hours=720.0, execution_quality=0.5)


def _small_fast():
    """9c of edge, two hours, excellent execution."""
    return _opp("FAST", price=0.62, fair=0.71, hours=2.0, execution_quality=0.95)


def _ev(opp, amount: float = STAKE):
    return ExpectedNetEVEngine().calculate(
        opp, amount, opp.market.raw["orderbook"])


# ----------------------------------------------------------------------
# the measure itself
# ----------------------------------------------------------------------

def test_both_fixtures_clear_every_hard_gate():
    """The comparison is between two tradeable opportunities, not two ideas."""
    for opp in (_big_slow(), _small_fast()):
        result = _ev(opp)
        assert result.should_trade, f"{opp.market.id} was refused: {result.reasoning}"
        assert result.net_ev_usd > 0


def test_the_ratio_is_net_ev_over_capital_days_and_risk():
    opp = _big_slow()
    result = _ev(opp)
    assert result.capital_days_usd == pytest.approx(
        STAKE * opp.time_to_resolution_hours / 24.0)
    # 30 days of $3: 90 capital-days, and the measure says so.
    assert result.capital_days_usd == pytest.approx(90.0)
    # The numerator is the NET EV - or the STRESSED one when a cost had to be
    # assumed, because a trade that looks efficient only on a guessed cost is
    # not efficient. Both cases are asserted, so this cannot drift into
    # whichever number happens to be true of the fixture today.
    numerator = (result.stressed_net_ev_usd if result.assumed_cost_usd > 0
                 else result.net_ev_usd)
    assert result.ev_per_capital_time_risk == pytest.approx(
        numerator / (result.capital_days_usd * result.execution_risk))


def test_a_guessed_cost_ranks_the_stressed_number():
    """An assumption must not buy a better ranking than it can deliver."""
    declared = _opp("MEASURED", price=0.60, fair=0.80, hours=720.0,
                    execution_quality=0.5)
    engine = ExpectedNetEVEngine()
    measured = engine.calculate(declared, STAKE, _book(0.60, size=50000.0)
                                | {"slippage_estimate": 0.002})
    assumed = _ev(_big_slow())
    assert assumed.assumed_cost_usd > 0, "the fixture must rely on an assumption"
    assert assumed.stressed_net_ev_usd < assumed.net_ev_usd
    assert assumed.ev_per_capital_time_risk < (
        assumed.net_ev_usd / (assumed.capital_days_usd * assumed.execution_risk)), (
        "the ranking used the unstressed EV")


def test_the_losing_candidate_still_carries_its_ratio():
    """So the console can answer why it lost, not only what won."""
    slow, fast = _big_slow(), _small_fast()
    OpportunityEngine().rank_and_select(
        [slow, fast], max_trades=1, bankroll=50.0, current_positions=[])
    assert slow.raw["capital_efficiency"]["ev_per_capital_time_risk"] > 0
    assert fast.raw["capital_efficiency"]["ev_per_capital_time_risk"] > \
        slow.raw["capital_efficiency"]["ev_per_capital_time_risk"]


def test_a_short_lock_up_beats_a_bigger_slow_earning():
    """
    The whole point: four times the profit, a fifth of the efficiency.

    $0.715 earned by locking $3 up for a month is 0.1 cents per dollar-day.
    $0.177 earned in two hours is worth the capital being free for the rest of
    the month.
    """
    big, small = _ev(_big_slow()), _ev(_small_fast())
    assert big.net_ev_usd > small.net_ev_usd, (
        "the fixture is meant to give the slow trade the larger profit")
    assert small.ev_per_capital_time_risk > big.ev_per_capital_time_risk * 100, (
        f"fast {small.ev_per_capital_time_risk:.4f} vs slow "
        f"{big.ev_per_capital_time_risk:.4f}: the ratio is not seeing time")


@pytest.mark.parametrize("quality,expected_direction", [(0.5, "lower"), (0.99, "higher")])
def test_poor_execution_risk_lowers_the_ratio(quality, expected_direction):
    """Execution risk is in the denominator, not decoration."""
    good = _ev(_opp("EXEC", price=0.62, fair=0.71, hours=2.0,
                    execution_quality=0.95))
    other = _ev(_opp("EXEC", price=0.62, fair=0.71, hours=2.0,
                     execution_quality=quality))
    if expected_direction == "lower":
        assert other.ev_per_capital_time_risk < good.ev_per_capital_time_risk
        assert other.execution_risk > good.execution_risk
    else:
        assert other.ev_per_capital_time_risk >= good.ev_per_capital_time_risk


def test_the_ratio_is_carried_in_the_result_dict():
    payload = _ev(_big_slow()).to_dict()
    for key in ("ev_per_capital_time_risk", "capital_days_usd", "execution_risk"):
        assert key in payload
    assert "capital efficiency" in _ev(_big_slow()).reasoning


# ----------------------------------------------------------------------
# the decision
# ----------------------------------------------------------------------

def test_the_selector_picks_the_efficient_trade_not_the_richest_one():
    slow, fast = _big_slow(), _small_fast()
    selected = OpportunityEngine().rank_and_select(
        [slow, fast], max_trades=1, bankroll=50.0, current_positions=[])
    assert [o.market.id for o in selected] == ["FAST"], (
        "the bigger, slower trade was chosen: the ratio is not ordering the "
        "candidates before the max_trades cut")
    # The winner is the less profitable trade, by the measure that decides.
    assert _ev(fast).net_ev_usd < _ev(slow).net_ev_usd
    assert fast.score > 0 and fast.raw["capital_efficiency"]["net_ev_usd"] > 0


def test_the_ordering_reaches_candidates_past_the_cut():
    """
    With two slots both are taken, and the order they are RETURNED in is the
    deployment order - the efficient one first.
    """
    selected = OpportunityEngine().rank_and_select(
        [_big_slow(), _small_fast()], max_trades=2, bankroll=50.0,
        current_positions=[])
    assert [o.market.id for o in selected] == ["FAST", "SLOW"]


def test_the_score_is_the_dominated_product():
    """The ratio times the bounded discounts - nothing else, and no double count."""
    fast = _small_fast()
    selected = OpportunityEngine().rank_and_select(
        [fast], max_trades=1, bankroll=50.0, current_positions=[])
    assert selected, "the fixture did not survive the gates"
    result = _ev(fast)
    assert fast.score == pytest.approx(
        result.ev_per_capital_time_risk * fast.liquidity_score)


def test_the_console_says_what_the_decision_was_made_on():
    from src.ptai.operator_view import describe_snapshot

    selected = OpportunityEngine().rank_and_select(
        [_big_slow(), _small_fast()], max_trades=1, bankroll=50.0,
        current_positions=[])
    lines = describe_snapshot({
        "last_cycle": {
            "available": True, "at": "2026-09-25T00:00:00+00:00",
            "verdict": "DEPLOYED", "markets_scanned": 10, "venues_searched": 1,
            "orders": {"positions_recorded": 1},
            "decided": {
                "capital_efficiency": selected[0].raw["capital_efficiency"]},
        },
    })
    assert any("capital efficiency" in line for line in lines), lines
    assert any("this is the number it ranks on" in line for line in lines), lines


def test_the_scan_records_the_ratio_on_every_opportunity_it_judged():
    """
    The whole scan, not just the selection stage.

    The scan used to compute a SECOND score in a different shape
    (`net_ev_usd * ev_per_dollar * ev_per_risk`) and then re-rank on
    `net_ev_usd` first, so a trade's rank depended on which stage wrote last and
    capital-time was a tiebreak either way. One definition now, and it is
    written on every candidate - including the ones that lost, because "why was
    this not taken" is only answerable if the number that ranked it was
    recorded before the decision.
    """
    import asyncio

    from src.ptai.markets.base import MarketSource
    from src.ptai.strategy.strategy_engine import StrategyEngineV3
    from tests.test_v3 import make_market

    book = _book(0.60)
    markets = [make_market(id=f"SCAN-{i}", question=f"Will scan {i} happen? Trump",
                           price=0.60, vol=50_000, liq=20_000,
                           source=MarketSource.POLYMARKET)
               for i in range(2)]
    for market in markets:
        market.raw["orderbook"] = book

    result = asyncio.run(StrategyEngineV3().scan_all_venues(
        markets_by_venue={"polymarket": markets}, max_final_trades=2))

    judged = list(result.all_opportunities)
    assert judged, "the scan evaluated nothing"
    assert result.total_candidates == sum(r.candidates for r in result.venue_reports)
    for opp in judged:
        block = (opp.raw or {}).get("capital_efficiency")
        assert block is not None, f"{opp.market.id} was judged without recording why"
        assert "ev_per_capital_time_risk" in block
        assert opp._expected_ev is not None, "the scan did not price what it judged"


def test_the_scan_score_is_the_ratio_and_the_ranking_uses_it():
    """
    The scan's own two steps, driven directly.

    Driven through the scan above this could not be proven: every opportunity in
    an offline fixture has negative EV, so both the old formula and the new one
    leave the score at zero and the assertion passes either way. Given a
    profitable trade, the two formulas disagree - which is the whole point.
    """
    from src.ptai.strategy.strategy_engine import StrategyEngineV3

    engine = StrategyEngineV3()
    fast = _small_fast()
    slow = _big_slow()

    fast_ev = engine.price_opportunity(fast, amount_usd=STAKE)
    slow_ev = engine.price_opportunity(slow, amount_usd=STAKE)
    assert fast_ev.net_ev_usd > 0 and slow_ev.net_ev_usd > 0

    # The score IS the ratio. The formula it replaced
    # (net_ev_usd x ev_per_dollar x ev_per_risk) gives a different number, and a
    # different ORDER on these two fixtures.
    assert fast.score == pytest.approx(fast_ev.ev_per_capital_time_risk)
    assert slow.score == pytest.approx(slow_ev.ev_per_capital_time_risk)
    old_formula = lambda ev: ev.net_ev_usd * ev.ev_per_dollar * ev.ev_per_risk
    assert old_formula(fast_ev) > 0
    assert not fast.score == pytest.approx(old_formula(fast_ev)), (
        "the scan is still scoring on the old product")

    # And the ranking is the ratio's order. Sorting on net_ev_usd first - which
    # is what the scan did - puts the slow trade on top.
    assert engine.rank_by_capital_efficiency([slow, fast])[0] is fast
    by_net_ev = sorted([slow, fast],
                       key=lambda o: o._expected_ev.net_ev_usd, reverse=True)
    assert by_net_ev[0] is slow, "the fixtures do not disagree on the two orders"
