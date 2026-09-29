"""
Paper mode has to be able to place a trade, and a round has to say what it made.

The operator, 2026-09-29: "paper mode need to start genearting income and doing
trades because in all the lkogs ive seen so mfar ive not seen 1 trade ... give
the results in the ui, the profits or losses made in a round. now i doont know
how long am supposed to run it to produce results so tell me".

A live cycle over a stub venue whose book really was worth crossing - the model
answered 0.55 against a book asking 0.42 - ended in DO NOTHING, and the log said
why in its own numbers:

    fair 0.490 <- ensemble 0.490 (llm raw 0.550) [base_rate 0.500 c0.00 w0.000
    (no data) | news ... (no data) | x_sentiment ... (no data) |
    market_microstructure 0.400 c0.70 w0.130 | llm_reasoning 0.550 c0.75 w0.197]
    -> calibrated 0.490 -> conservative 0.990; market 0.400
    Priced pm2 ... (fair 0.490 vs market 0.400, edge +0.090, conf 0.29)
    Edge calc pm2: Deductions (... unc 0.339 ...) | Effective -0.072
    REFUSED: no executable edge: fair 0.490 vs the 0.420 a share actually costs

Three arithmetic faults made every trade impossible, and this file pins all of
them the way the operator reads them:

  1. CONFIDENCE AND UNCERTAINTY WERE AVERAGED OVER COMPONENTS THAT SAID
     NOTHING. A silent component already has weight 0 - it cannot move the
     estimate - but it was still counted in `conf 0.29` (three of five
     components had no data) and in the 0.71 uncertainty that demanded a 12%
     mispricing before anything could be considered.
  2. UNCERTAINTY WAS CHARGED AS CASH, TWICE, ON TOP OF THE HAIRCUT IT ALREADY
     APPLIED to the estimate: `unc 0.339` is 84.75% of the share price charged
     as though a venue collected it, and it is what turned a 15-point model
     disagreement into `Effective -0.072`.
  3. THE "CONSERVATIVE" FAIR VALUE WAS NOT CONSERVATIVE: it moved the estimate
     toward 0.5 whatever the market said, so 0.49 against a 0.40 market became
     0.990 - printed in the same cycle as the edge calculator's own 0.355.

And one structural fact: the paper/exploration lane - the lane that exists to
produce the resolved trades a venue earns live capital with - was fed from
`final_selected`, which requires the LIVE gates to pass first, so on a fresh
install it could never fire.
"""

from __future__ import annotations

import pytest

from src.ptai.intelligence.ensemble import EnsembleForecaster, ModelForecast
from src.ptai.intelligence.uncertainty import conservative_probability
from src.ptai.markets.base import Market, MarketSource
from src.ptai.strategy.edge import EdgeCalculator
from src.ptai.strategy.strategy_engine import (
    EXPLORATION_EXECUTABLE_EDGE_MIN,
    EXPLORATION_MISPRICING_MIN,
    exploration_eligible,
)


def _market(price: float = 0.40, liquidity: float = 25_000.0) -> Market:
    m = Market(id="pm0", source=MarketSource.POLYMARKET,
               question="Will event 0 happen this year?",
               outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
               volume=200_000.0, volume_24h=120_000.0, liquidity=liquidity,
               active=True, closed=False, raw={"venue": "polymarket"})
    return m


def _book(ask: float = 0.42, bid: float = 0.40, size: float = 900.0):
    return {"bids": [{"price": bid, "size": size}],
            "asks": [{"price": ask, "size": size}],
            "bid": bid, "ask": ask, "spread": ask - bid, "depth": 40000.0,
            "is_real": True, "is_mock": False, "validated": True,
            "executable": True, "executable_price": ask, "source": "clob"}


class _Fv:
    """The facts the paper lane reads off a priced market."""

    def __init__(self, *, should_trade=False, edge=0.09, effective_edge=0.05,
                 blocked_by="", book_is_real=True, llm_answered=True,
                 reasoning="Decision: False because High uncertainty"):
        self.should_trade = should_trade
        self.edge = edge
        self.effective_edge = effective_edge
        self.blocked_by = blocked_by
        self.book_is_real = book_is_real
        self.reasoning = reasoning
        chain = {"llm_raw": 0.55} if llm_answered else {}
        self.forecast_result = type("F", (), {"chain": chain})()


# ---------------------------------------------------------------------------
# 1. the confidence and the uncertainty belong to the components that answered
# ---------------------------------------------------------------------------

class TestSilentComponentsDoNotVote:
    def _forecasts(self):
        return [
            ModelForecast(model_name="base_rate", probability=0.50,
                          confidence=0.0, uncertainty=1.0, reasoning="no data", sources=[]),
            ModelForecast(model_name="news", probability=0.40,
                          confidence=0.0, uncertainty=1.0, reasoning="no data", sources=[]),
            ModelForecast(model_name="x_sentiment", probability=0.40,
                          confidence=0.0, uncertainty=1.0, reasoning="no data", sources=[]),
            ModelForecast(model_name="market_microstructure", probability=0.40,
                          confidence=0.70, uncertainty=0.30, reasoning="book", sources=[]),
            ModelForecast(model_name="llm_reasoning", probability=0.55,
                          confidence=0.75, uncertainty=0.25, reasoning="model", sources=[]),
        ]

    def test_confidence_is_the_confidence_of_the_components_with_data(self):
        result = EnsembleForecaster(llm_router=None).ensemble(
            self._forecasts(), _market(), category="default")
        # The operator's log said conf 0.29 - the sum divided by FIVE, three of
        # which had no data. It is (0.70 + 0.75) / 2.
        assert result.confidence == pytest.approx(0.725, abs=1e-9)
        assert result.confidence >= 0.6, (
            "the 60% floor has to be reachable for a market only the book and "
            "the model had anything to say about")

    def test_uncertainty_is_the_uncertainty_of_the_components_with_data(self):
        result = EnsembleForecaster(llm_router=None).ensemble(
            self._forecasts(), _market(), category="default")
        assert result.uncertainty == pytest.approx(0.275, abs=1e-9)

    def test_the_trace_says_what_was_averaged_and_over_how_many(self):
        result = EnsembleForecaster(llm_router=None).ensemble(
            self._forecasts(), _market(), category="default")
        basis = result.chain.get("confidence_basis", "")
        assert "2 component(s) with data" in basis
        assert "3 had none" in basis
        assert "confidence 0.725" in result.explain()

    def test_the_estimate_itself_is_unchanged_by_a_silent_component(self):
        # A silent component had weight 0 before this change and weight 0 after
        # it: only the confidence and the uncertainty it was polluting moved.
        # 0.4935 = (0.40 x 0.119 + 0.55 x 0.197) / 0.316, the book and the model.
        result = EnsembleForecaster(llm_router=None).ensemble(
            self._forecasts(), _market(), category="default")
        assert result.fair_probability == pytest.approx(0.4935, abs=0.002)
        assert result.fair_probability == pytest.approx(
            result.chain["ensemble_raw"], abs=1e-4)


# ---------------------------------------------------------------------------
# 2. the conservative estimate is conservative, and it is one number
# ---------------------------------------------------------------------------

class TestTheConservativeEstimateMovesTowardTheMarket:
    def test_it_never_becomes_a_bigger_estimate_than_the_best_one(self):
        # The log's own defect: 0.49 against a 0.40 market became "0.990".
        value = conservative_probability(0.490, 0.400, 0.71)
        assert value == pytest.approx(0.400, abs=1e-9)
        assert value <= 0.490

    def test_it_stops_at_the_price_rather_than_crossing_it(self):
        # Crossing the price would report a conservative edge on the OTHER side
        # of the market - an invented trade. Clamped, the worst it can do is
        # take the edge away.
        assert conservative_probability(0.35, 0.40, 0.50) == pytest.approx(0.40)
        assert conservative_probability(0.45, 0.40, 0.50) == pytest.approx(0.40)

    def test_the_documented_example_still_holds(self):
        # "forecast 71% +-8% -> conservative 63%"
        assert conservative_probability(0.71, 0.60, 0.08) == pytest.approx(0.63)

    def test_a_market_above_the_estimate_shrinks_the_other_way(self):
        # fair 0.30 against a 0.60 market: the NO side. The haircut moves the
        # estimate UP toward the price, which shrinks the NO edge.
        assert conservative_probability(0.30, 0.60, 0.10) == pytest.approx(0.40)

    def test_the_ensemble_and_the_edge_calculator_agree_on_the_number(self):
        # The operator's log carried TWO conservative values for one market in
        # one cycle - `-> conservative 0.990` from the ensemble and
        # `conservative_fair 0.355` from the edge calculator. One haircut, one
        # number: a single model at 0.55 against a 0.40 market with 0.25 of
        # uncertainty gives 0.55 - 0.25 = 0.30 in both.
        result = EnsembleForecaster(llm_router=None).ensemble(
            [ModelForecast(model_name="llm_reasoning", probability=0.55,
                           confidence=0.75, uncertainty=0.25, reasoning="m", sources=[])],
            _market(price=0.20), category="default")
        edge = EdgeCalculator().calculate(
            market=_market(price=0.20), fair_prob=0.55, uncertainty=0.25,
            orderbook=_book(ask=0.22, bid=0.20), amount_usd=1.0, side="YES")
        assert result.conservative_fair == pytest.approx(0.30, abs=1e-9)
        assert edge.conservative_fair == pytest.approx(0.30, abs=1e-9)
        # ...and when the haircut would cross the price it stops AT the price, in
        # both, rather than reporting an edge on the other side of the market.
        result_hi = EnsembleForecaster(llm_router=None).ensemble(
            [ModelForecast(model_name="llm_reasoning", probability=0.55,
                           confidence=0.75, uncertainty=0.25, reasoning="m", sources=[])],
            _market(), category="default")
        edge_hi = EdgeCalculator().calculate(
            market=_market(), fair_prob=0.55, uncertainty=0.25,
            orderbook=_book(), amount_usd=1.0, side="YES")
        assert result_hi.conservative_fair == pytest.approx(0.40, abs=1e-9)
        assert edge_hi.conservative_fair == pytest.approx(0.40, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. uncertainty is charged once - as the haircut, never as cash
# ---------------------------------------------------------------------------

class TestUncertaintyIsNotACashCost:
    def test_the_logs_own_case_leaves_a_positive_edge(self):
        # fair 0.4900 against a book bidding 0.40 / asking 0.42, uncertainty
        # 0.275: the log's `Effective -0.072` came from charging 0.339 of the
        # notional as an uncertainty "cost" on top of the haircut. Without that
        # double charge the trade is +0.034 a share after the costs the
        # calculator assumes when the venue has not declared its own (fees 3.6%
        # + gas 5.0% + spread 2.0% of the notional), and +0.055 with the paper
        # venue's declared zero fee and no gas.
        edge = EdgeCalculator().calculate(
            market=_market(), fair_prob=0.490, uncertainty=0.275,
            orderbook=_book(), amount_usd=1.0, side="YES")
        assert edge.executable_edge == pytest.approx(0.034, abs=0.005)
        assert edge.executable_edge > 0, "the operator's own case is a trade"
        # The uncertainty is still REPORTED - and it is not in the cash sum.
        assert edge.uncertainty_penalty == pytest.approx(0.1375, abs=1e-9)
        cash = (edge.fees + edge.spread + edge.slippage
                + edge.liquidity_penalty + edge.correlation_penalty
                + edge.time_penalty)
        assert cash == pytest.approx(0.056, abs=0.005), (
            "fees + spread are the cash here; the 0.275 of uncertainty is not "
            "(gas is charged on top and is only in the reasoning line)")
        assert "gas 0.050" in edge.reasoning
        assert "NOT cash" in edge.reasoning

    def test_the_conservative_estimate_is_what_the_live_gate_uses(self):
        edge = EdgeCalculator().calculate(
            market=_market(), fair_prob=0.490, uncertainty=0.275,
            orderbook=_book(), amount_usd=1.0, side="YES")
        assert edge.conservative_executable_edge < 0
        assert edge.blocked_by == ""          # the price is payable
        assert not edge.should_trade          # but not at the conservative view
        assert "conservative estimate" in edge.conservative_blocked_by


# ---------------------------------------------------------------------------
# 4. the paper lane can act on a market the live gates refused
# ---------------------------------------------------------------------------

class TestThePaperLaneHasItsOwnBar:
    def test_a_real_edge_the_live_gates_refused_is_a_paper_candidate(self):
        assert exploration_eligible(_Fv(), executable_edge=0.055,
                                    execution_quality=0.8,
                                    liquidity_score=0.9) is True

    def test_a_market_the_live_gates_approved_is_not_a_paper_candidate(self):
        assert exploration_eligible(_Fv(should_trade=True), executable_edge=0.055,
                                    execution_quality=0.8,
                                    liquidity_score=0.9) is False

    def test_a_thin_edge_is_not_a_paper_candidate_either(self):
        assert exploration_eligible(
            _Fv(), executable_edge=EXPLORATION_EXECUTABLE_EDGE_MIN / 2,
            execution_quality=0.8, liquidity_score=0.9) is False

    def test_a_small_mispricing_is_not_a_paper_candidate(self):
        assert exploration_eligible(
            _Fv(edge=EXPLORATION_MISPRICING_MIN / 2), executable_edge=0.055,
            execution_quality=0.8, liquidity_score=0.9) is False

    def test_a_market_with_no_model_answer_is_not_a_paper_candidate(self):
        assert exploration_eligible(_Fv(llm_answered=False),
                                    executable_edge=0.055,
                                    execution_quality=0.8,
                                    liquidity_score=0.9) is False

    def test_no_real_book_is_not_a_paper_candidate(self):
        assert exploration_eligible(_Fv(book_is_real=False, blocked_by="not real"),
                                    executable_edge=0.055, execution_quality=0.8,
                                    liquidity_score=0.9) is False

    def test_an_unexecutable_book_is_not_a_paper_candidate(self):
        assert exploration_eligible(_Fv(blocked_by="spread 99% cannot pay"),
                                    executable_edge=0.055, execution_quality=0.8,
                                    liquidity_score=0.9) is False

    def test_a_thin_book_is_not_a_paper_candidate(self):
        assert exploration_eligible(_Fv(), executable_edge=0.055,
                                    execution_quality=0.1,
                                    liquidity_score=0.9) is False


# ---------------------------------------------------------------------------
# 5. "how long do I have to run it" has a line in the log
# ---------------------------------------------------------------------------

class TestTheRoundSaysHowCloseItCame:
    def _agent(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent._cycle_closest_call = {
            "market_id": "pm0", "side": "YES", "fair": 0.490, "market": 0.400,
            "mispricing": 0.090, "executable_edge": 0.051,
            "conservative_executable_edge": -0.039,
            "model_answered": True,
            "refusal": "High uncertainty 0.195 requires mispricing >12%, got 0.090",
        }
        return agent

    def test_the_line_names_the_market_the_numbers_and_the_bar(self):
        line = self._agent()._closest_call_line()
        assert "pm0" in line and "YES" in line
        assert "0.490" in line and "0.400" in line
        assert "+0.051" in line and "-0.039" in line
        assert "8% mispricing" in line
        assert "High uncertainty" in line

    def test_a_round_that_priced_nothing_says_that(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent._cycle_closest_call = {}
        assert "nothing was priced" in agent._closest_call_line()

    def test_the_round_record_carries_its_trades_and_the_closest_call(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.execution.round import RoundReport
        src = inspect.getsource(TradingAgentV3._score_round)
        assert "report.trades = trades" in src
        assert "report.closest_call" in src
        report = RoundReport()
        assert report.to_dict()["trades"] == []
        assert report.to_dict()["closest_call"] == {}
