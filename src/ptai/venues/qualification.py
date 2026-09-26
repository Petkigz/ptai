"""
Venue Qualification - paper-trading qualification robust
FIXED V7: User correctly identified win rate alone is not profitability
Example: 90% wins +$0.01, 10% losses -$1.00 would have fantastic win rate and still lose money

Now includes:
- net P&L
- expected value
- fees
- slippage
- drawdown
- profit factor
- calibration
- Brier/log loss
- sample size
- execution quality
Not just win rate
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..learning.evidence import FORECAST_BEHIND_PRICE
from loguru import logger
from datetime import datetime, timezone
import json
from pathlib import Path

@dataclass
class QualificationResult:
    venue_id: str
    win_rate: float
    avg_edge: float
    brier_score: float
    profit_paper: float
    profit_live: float
    forecast_skill: float
    is_qualified: bool
    qualification_date: Optional[datetime]
    requirements: Dict[str, Any]
    reasoning: str
    # FIXED V7: Additional metrics beyond win rate
    net_pnl: float = 0.0
    # All resolved evidence, and the two populations separately.
    # `total_paper_trades` holds the PAPER count; the sample-size gate uses the
    # total.
    total_resolved_trades: int = 0
    total_paper_trades: int = 0
    live_trades: int = 0
    expected_value: float = 0.0
    fees_total: float = 0.0
    slippage_total: float = 0.0
    drawdown_max: float = 0.0
    profit_factor: float = 0.0
    calibration_ece: float = 0.0
    log_loss: float = 0.0
    execution_quality_avg: float = 0.0
    sample_size: int = 0
    # Where the expected-value figure came from, and how much of the record it
    # covers. The reasoning line carries the same words, but a field is what a
    # console can read: a substituted mean edge must not be displayed like a
    # measured expected net EV.
    ev_source: str = ""
    real_evidence_coverage: float = 0.0
    ev_bias: Optional[float] = None
    executable_value: Optional[float] = None
    executable_value_coverage: float = 0.0
    fill_price_vs_modelled: Optional[float] = None
    ev_coverage: float = 0.0
    # Every bar, pass or fail, by name. The reasoning line says the same thing
    # in prose, but an operator - or the console - asking WHICH bar refused a
    # venue should not have to parse a sentence, and a test asserting "the fill
    # price check refused this" must not be able to pass on a different check.
    checks: Dict[str, bool] = field(default_factory=dict)
    # The forecast measured against the PRICE, which is the only benchmark that
    # can say whether there is an edge at all. `forecast_skill` above is a
    # rescaled Brier score and never saw a price; these are the comparison.
    market_skill: Optional[float] = None
    # The mean improvement in Brier units per trade: how much better the
    # forecast was than the price, before any interval. Positive means better.
    market_improvement: Optional[float] = None
    market_skill_verdict: str = "unmeasured"
    market_skill_ci_low: Optional[float] = None
    market_skill_ci_high: Optional[float] = None
    market_skill_p_value: Optional[float] = None
    market_skill_samples: int = 0
    market_skill_reason: str = ""
    # The two verdicts themselves, as booleans. `checks` holds every bar, but a
    # dict of them is not written to disk - so a screen reading a loaded record
    # got `checks.get("beats_the_price")` as None for every venue and displayed
    # "has not beaten the price" for a venue that had. These are persisted.
    beats_the_price: Optional[bool] = None
    not_drifting: Optional[bool] = None
    recent_market_skill: Optional[float] = None
    recent_market_skill_verdict: str = "unmeasured"
    recent_market_skill_reason: str = ""
    # When this judgment was made and on how much evidence. A verdict is about a
    # record, and a record grows; the count is what lets a reader tell a fresh
    # judgment from one written before the last hundred trades happened.
    rows_at_evaluation: int = 0
    # The count at the moment the venue last PASSED. Staleness is measured from
    # here, not from the last evaluation: re-running the same bars over the same
    # record is not new evidence, and a judgment that has not been re-earned
    # since 150 more trades arrived should be visible as exactly that.
    qualified_on_trades: int = 0

class VenueQualificationEngine:
    """
    Robust paper-trading qualification
    FIXED V7: Beyond win rate - includes P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier, sample, execution quality
    User: Venue qualification should eventually consider net P&L, expected value, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality not just win rate
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.qualification_file = self.data_dir / "venue_qualification.json"
        self.qualifications: Dict[str, QualificationResult] = {}
        self._load()

        # FIXED V7: Comprehensive requirements beyond win rate
        self.requirements = {
            "min_trades": 100,
            "min_win_rate": 0.55,
            "max_brier": 0.25,
            "max_log_loss": 0.6,
            "max_ece": 0.15,
            "min_forecast_skill": 0.6,
            "min_profit_paper": 0.0,  # must be profitable after fees
            "min_net_pnl": 0.0,  # net P&L after fees/slippage
            "min_expected_value": 0.01,  # EV >1% per trade
            "min_avg_edge": 0.03,  # 3% avg edge
            "min_calibration": 0.5,
            "max_drawdown": 0.20,  # max 20% drawdown
            "min_profit_factor": 1.1,  # gross profit / gross loss >1.1
            "min_execution_quality": 0.5,  # avg execution quality
            # ...MEASURED on at least half the trades being judged. An average
            # taken from one measurement out of 150 is not a measurement of the
            # venue, and the totals above it were already reported with their
            # coverage for exactly this reason - now the gate requires it.
            "min_cost_coverage": 0.5,
            # The EV bar, too, has to cover the sample it judges. `min_ev` may be
            # cleared by one recorded value out of 150 trades, and "EV > 1% per
            # trade" then describes a venue on the strength of a single entry
            # nobody repeated. Same rule as the measured costs above.
            "min_ev_coverage": 0.5,
            # THE QUALITY OF THE EVIDENCE, not one more bar to clear.
            #
            # At least half of the sample must be real: a live fill, or a
            # simulated fill walked down a ladder that existed. A venue
            # "observed" 150 times against an assumed spread has been observed
            # zero times, and the simulation's own arithmetic is what would have
            # qualified it.
            "min_real_evidence_coverage": 0.5,
            # THE EXECUTABLE EV, which is what "profitable" actually means.
            #
            # `min_ev` judges the mean PREDICTION. A prediction is a model of
            # execution, computed at the price the book showed when the
            # opportunity was found - and a venue whose orders fill worse than
            # that on every trade passes `min_ev` while losing the difference.
            # This check judges the same EV recomputed at the price each order
            # actually paid and the fees it actually incurred. Same bar as the
            # prediction claims to clear, because that is the claim being tested.
            "min_executable_ev": 0.01,
            "min_executable_ev_coverage": 0.5,
            # ...and WHERE THE FILLS LANDED against the prices the decisions
            # were made at. Positive means paid more than modelled.
            #
            # This is the realized execution performance, in price units, with
            # no risk penalty and no forecast luck in it: either the venue fills
            # at the prices the agent models, or it does not. A venue that fills
            # 2% above the price its EV was computed at has lost 2% of the stake
            # on every trade, which on a 1-3% net-EV bar is the whole edge.
            # Fails closed when unmeasured.
            "max_fill_price_penalty": 0.01,
            "min_sample_size": 100,
            "max_fees_pct": 0.05,  # fees <5% of profit
            # ------------------------------------------------------------------
            # THE NULL. Every bar above is absolute, and an absolute accuracy
            # score cannot say whether the agent beat the thing it is trading
            # against. In a market priced at 0.50, forecasting 0.50 forever
            # scores a Brier of 0.25 and CLEARS `max_brier`. The market price at
            # entry is the only benchmark that answers the actual question, so
            # the gate now requires the forecasts to beat it - with a bootstrap
            # interval that clears zero - and the entries taken to clear the
            # odds they paid, tested against the prices rather than a coin.
            #
            # Borrowed from the sibling project's deployment protocol
            # (`avt-bot`): model vs null, CI on the difference, p-value on the
            # entries, and NO_SIGNAL as a perfectly good outcome.
            # ------------------------------------------------------------------
            # Two-sided 95% on the paired improvement, so the whole interval
            # must be above zero to claim the price was beaten.
            "min_market_skill_ci_low": 0.0,
            # The entries' right tail: 5% chance of luck, the same bar the
            # sibling project uses on its approved entries.
            "max_market_skill_p_value": 0.05,
            # Same coverage rule as the other measurements: half the sample.
            "min_market_skill_coverage": 0.5,
            # ...and the RECENT record, asked separately. An edge that has died
            # is a reason to stop, not a reason to average it with the months it
            # worked. Newest trades only.
            "recent_window_trades": 60,
            # A recent window entirely behind the price - the interval's upper
            # bound below zero - means the venue is drifting and must not hold
            # live capital, however good its history was. 30 paired trades is the
            # same floor the whole-record test uses.
            "min_recent_pairs": 30,
            # A qualification is a judgment about a record, and the record ages.
            # Once this many settled trades have accumulated since the judgment,
            # it is stale: live capital needs a fresh evaluation, not a
            # re-reading of a file written a hundred trades ago.
            "max_trades_since_evaluation": 150,
        }

    def _load(self):
        if self.qualification_file.exists():
            try:
                with open(self.qualification_file, 'r') as f:
                    data = json.load(f)
                    for venue_id, qual_data in data.items():
                        if qual_data.get("qualification_date"):
                            qual_data["qualification_date"] = datetime.fromisoformat(qual_data["qualification_date"])
                        # Handle old format without new fields
                        for field in ["net_pnl", "expected_value", "fees_total", "slippage_total", "drawdown_max", "profit_factor", "calibration_ece", "log_loss", "execution_quality_avg", "sample_size", "ev_coverage", "real_evidence_coverage", "executable_value_coverage", "market_skill", "market_skill_ci_low", "market_skill_ci_high", "market_skill_p_value", "recent_market_skill", "rows_at_evaluation", "qualified_on_trades", "market_skill_samples", "market_improvement"]:
                            if field not in qual_data:
                                qual_data[field] = 0.0
                        # Strings keep their own defaults, or a loaded record
                        # would claim a verdict of 0.0.
                        for field in ["market_skill_verdict", "market_skill_reason", "recent_market_skill_verdict", "recent_market_skill_reason"]:
                            if field not in qual_data:
                                qual_data[field] = ("unmeasured" if field.endswith("verdict") else "")
                        # Booleans default to unknown, not to a number: a record
                        # written before these existed must read "never measured"
                        # and not "measured and failed".
                        for field in ["beats_the_price", "not_drifting"]:
                            if field not in qual_data:
                                qual_data[field] = None
                        self.qualifications[venue_id] = QualificationResult(**qual_data)
            except Exception as e:
                logger.warning(f"Qualification load failed: {e}")

    def _save(self):
        try:
            data = {}
            for venue_id, qual in self.qualifications.items():
                data[venue_id] = {
                    "venue_id": qual.venue_id,
                    "total_resolved_trades": qual.total_resolved_trades,
                    "total_paper_trades": qual.total_paper_trades,
                    "live_trades": qual.live_trades,
                    "win_rate": qual.win_rate,
                    "avg_edge": qual.avg_edge,
                    "brier_score": qual.brier_score,
                    "profit_paper": qual.profit_paper,
                    "profit_live": qual.profit_live,
                    "forecast_skill": qual.forecast_skill,
                    "is_qualified": qual.is_qualified,
                    "qualification_date": qual.qualification_date.isoformat() if qual.qualification_date else None,
                    "requirements": qual.requirements,
                    "reasoning": qual.reasoning,
                    "net_pnl": qual.net_pnl,
                    "expected_value": qual.expected_value,
                    "fees_total": qual.fees_total,
                    "slippage_total": qual.slippage_total,
                    "drawdown_max": qual.drawdown_max,
                    "profit_factor": qual.profit_factor,
                    "calibration_ece": qual.calibration_ece,
                    "log_loss": qual.log_loss,
                    "execution_quality_avg": qual.execution_quality_avg,
                    "sample_size": qual.sample_size,
                    # Written out, or the file the console reads would lose the
                    # one field that says whether the EV it displays was
                    # measured or substituted.
                    "ev_source": qual.ev_source,
                    "ev_coverage": qual.ev_coverage,
                    "real_evidence_coverage": qual.real_evidence_coverage,
                    "ev_bias": qual.ev_bias,
                    "executable_value": qual.executable_value,
                    "executable_value_coverage": qual.executable_value_coverage,
                    "fill_price_vs_modelled": qual.fill_price_vs_modelled,
                    # The comparison against the price, and the freshness
                    # stamps. Written out or the file the console and the venue
                    # selector read would lose the only evidence that says
                    # whether the edge was ever measured at all.
                    "market_skill": qual.market_skill,
                    "market_improvement": qual.market_improvement,
                    "market_skill_verdict": qual.market_skill_verdict,
                    "market_skill_ci_low": qual.market_skill_ci_low,
                    "market_skill_ci_high": qual.market_skill_ci_high,
                    "market_skill_p_value": qual.market_skill_p_value,
                    "market_skill_samples": qual.market_skill_samples,
                    "market_skill_reason": qual.market_skill_reason,
                    "beats_the_price": qual.beats_the_price,
                    "not_drifting": qual.not_drifting,
                    "recent_market_skill": qual.recent_market_skill,
                    "recent_market_skill_verdict": qual.recent_market_skill_verdict,
                    "recent_market_skill_reason": qual.recent_market_skill_reason,
                    "rows_at_evaluation": qual.rows_at_evaluation,
                    "qualified_on_trades": qual.qualified_on_trades,
                }
            with open(self.qualification_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Qualification save failed: {e}")

    def evaluate_qualification(self, venue_id: str, performance_stats: Dict[str, Any]) -> QualificationResult:
        # The SAMPLE SIZE is all resolved evidence; the PROFIT gate reads the
        # paper curve. Both are deliberate, and both are now named for what they
        # are - this line used to read `total_paper_trades`, which held every
        # outcome including live ones, so the name said paper while the number
        # was the whole record.
        total = performance_stats.get(
            "total_resolved_trades",
            performance_stats.get("total_paper_trades", 0))
        win_rate = performance_stats.get("win_rate", 0)
        avg_edge = performance_stats.get("avg_edge", 0)
        brier = performance_stats.get("brier_score", 1.0)
        log_loss = performance_stats.get("log_loss", 1.0)
        ece = performance_stats.get("calibration_ece", 0.5)
        profit_paper = performance_stats.get("profit_paper", 0)
        profit_live = performance_stats.get("profit_live", 0)
        net_pnl = performance_stats.get("net_pnl", profit_paper)
        # The RECORDED expected net EV at entry - what the agent predicted it
        # would make - not the average edge.
        #
        # This read `performance_stats["expected_value"]` with a fallback to
        # `avg_edge`, and `expected_value` was itself `sum(edges)/n`: the mean of
        # the recorded EFFECTIVE EDGES, a probability-scale number. The check
        # below compares it with `min_expected_value = 0.01` and calls it
        # "EV > 1% per trade". Mean edge and expected monetary value are not the
        # same quantity, and a venue could clear the EV bar on its edge while
        # every trade lost money to fees.
        #
        # None when nothing was measured, and None fails the gate rather than
        # defaulting to zero: "never measured" must not pass, and must not read
        # as "measurably zero" either.
        ev_samples = int(performance_stats.get("expected_value_samples") or 0)
        ev_coverage = float(performance_stats.get("expected_value_coverage") or 0.0)
        expected_value = performance_stats.get("expected_value")
        if ev_samples > 0:
            # The real thing: the expected net EV each trade was predicted to
            # have, recorded before it was taken.
            expected_value = float(expected_value or 0.0)
            ev_source = (f"recorded expected net EV over {ev_samples} trade(s), "
                         f"{ev_coverage*100:.0f}% coverage")
        elif expected_value is None:
            # Older rows carry no expected EV. Falling back to the mean edge is
            # the old behaviour and is allowed, but only when there is nothing
            # measured - and it is labelled, so a reader can tell which number
            # they were given.
            expected_value = avg_edge
            ev_source = "avg_edge fallback (no recorded expected net EV)"
        else:
            # Supplied by the caller rather than read from the outcome log. It is
            # used, and labelled as unverified - claiming it was "recorded over 0
            # trades" would be a false statement about where the number came from.
            expected_value = float(expected_value)
            ev_source = ("caller-supplied expected value, not verified against "
                         "per-trade entries")
        fees_total = performance_stats.get("fees_total", 0)
        slippage_total = performance_stats.get("slippage_total", 0)
        drawdown_max = performance_stats.get("drawdown_max", 0)
        profit_factor = performance_stats.get("profit_factor", 0)
        execution_quality = performance_stats.get("execution_quality_avg", 0.5)
        cost_coverage = float(performance_stats.get("cost_coverage") or 0.0)
        real_evidence_coverage = float(
            performance_stats.get("real_evidence_coverage") or 0.0)
        real_evidence_samples = int(
            performance_stats.get("real_evidence_samples") or 0)
        ev_bias = performance_stats.get("ev_bias")
        ev_bias_samples = int(performance_stats.get("ev_bias_samples") or 0)
        executable_value = performance_stats.get("executable_value")
        executable_samples = int(
            performance_stats.get("executable_value_samples") or 0)
        executable_coverage = float(
            performance_stats.get("executable_value_coverage") or 0.0)
        fill_price_vs_modelled = performance_stats.get("fill_price_vs_modelled")
        price_paid_samples = int(performance_stats.get("price_paid_samples") or 0)
        quality_coverage = float(
            performance_stats.get("execution_quality_coverage") or 0.0)
        skill = performance_stats.get("forecast_skill", 0.5)
        # The comparison against the price, as the outcome log computed it.
        market_skill_value = performance_stats.get("market_skill")
        market_improvement = performance_stats.get("market_improvement")
        market_verdict = str(
            performance_stats.get("market_skill_verdict") or "unmeasured")
        market_ci_low = performance_stats.get("market_skill_ci_low")
        market_ci_high = performance_stats.get("market_skill_ci_high")
        market_p_value = performance_stats.get("market_skill_p_value")
        market_samples = int(performance_stats.get("market_skill_samples") or 0)
        market_coverage = float(
            performance_stats.get("market_skill_coverage") or 0.0)
        market_beats = bool(performance_stats.get("market_skill_beats_price"))
        market_entries_ok = bool(
            performance_stats.get("market_skill_entries_clear_odds"))
        market_reason = str(performance_stats.get("market_skill_reason") or "")
        recent_verdict = str(
            performance_stats.get("recent_market_skill_verdict") or "unmeasured")
        recent_skill_value = performance_stats.get("recent_market_skill")
        recent_reason = str(
            performance_stats.get("recent_market_skill_reason") or "")

        # FIXED V7: Check all requirements beyond win rate
        checks = {
            "min_trades": total >= self.requirements["min_trades"],
            "min_win_rate": win_rate >= self.requirements["min_win_rate"],
            "max_brier": brier <= self.requirements["max_brier"],
            "max_log_loss": log_loss <= self.requirements["max_log_loss"],
            "max_ece": ece <= self.requirements["max_ece"],
            "min_skill": skill >= self.requirements["min_forecast_skill"],
            "min_profit": profit_paper >= self.requirements["min_profit_paper"],
            "min_net_pnl": net_pnl >= self.requirements["min_net_pnl"],
            "min_ev": (expected_value is not None and
                       expected_value >= self.requirements["min_expected_value"]),
            # A value is not evidence. This is the check that says so: the mean
            # recorded expected net EV has to come from at least
            # `min_ev_coverage` of the resolved trades being judged.
            #
            # `ev_samples == 0` fails closed. There are three ways to reach this
            # branch with no recorded EVs - legacy rows, a caller supplying a
            # number, and a venue whose trades were never scored - and none of
            # them is a measurement of the venue. The value is still REPORTED
            # (with its source named) so the console shows what it was given
            # rather than a blank, but it cannot qualify anything.
            "min_ev_coverage": (ev_samples > 0 and
                                ev_coverage >= self.requirements["min_ev_coverage"]),
            "min_edge": avg_edge >= self.requirements["min_avg_edge"],
            "max_drawdown": drawdown_max <= self.requirements["max_drawdown"],
            "min_profit_factor": profit_factor >= self.requirements["min_profit_factor"],
            # Fails closed with no real evidence at all: an unmeasured venue is
            # not a venue whose evidence happened to be simulated, it is a venue
            # nobody has looked at.
            "min_real_evidence": (
                real_evidence_samples > 0
                and real_evidence_coverage
                >= self.requirements["min_real_evidence_coverage"]),
            # The EV model's own error, judged against the bar it is claiming to
            # clear. If the venue's trades were predicted at +1% net EV and
            # realised more than a point below that, the prediction is not a
            # measurement of anything and must not be the number the gate trusts.
            "ev_bias_within_bar": (
                ev_bias is not None and ev_bias_samples > 0
                and ev_bias >= -self.requirements["min_expected_value"]),
            # The EV the fills actually had, on its own coverage. Fails closed
            # with none: a venue whose trades were never repriced at their fills
            # has not shown that its edge survives execution.
            "min_executable_ev": (
                executable_samples > 0
                and executable_coverage >= self.requirements["min_executable_ev_coverage"]
                and executable_value is not None
                and float(executable_value) >= self.requirements["min_executable_ev"]),
            # The fills must land where the agent modelled them. Fails closed
            # when no fill was ever compared with the price its decision was made
            # at - an unmeasured execution is not a good one.
            "fills_at_modelled_price": (
                fill_price_vs_modelled is not None and price_paid_samples > 0
                and float(fill_price_vs_modelled)
                <= self.requirements["max_fill_price_penalty"]),
            "min_execution": (execution_quality >= self.requirements["min_execution_quality"]
                              and min(cost_coverage, quality_coverage)
                              >= self.requirements["min_cost_coverage"]),
            # THE NULL. Fail closed at every step: no paired evidence fails, an
            # interval that still contains zero fails, entries that did not clear
            # the odds they paid fail. `forecast_beats_price` alone is not
            # enough - a forecast can be better calibrated than the market while
            # every trade loses money by paying too much for the side.
            "beats_the_price": (
                market_samples >= self.requirements["min_trades"] // 2
                and market_coverage >= self.requirements["min_market_skill_coverage"]
                and market_ci_low is not None
                and float(market_ci_low) > self.requirements["min_market_skill_ci_low"]
                and market_p_value is not None
                and float(market_p_value) < self.requirements["max_market_skill_p_value"]
                and market_beats and market_entries_ok),
            # The recent record, on its own. `forecast_behind_price` over the
            # newest trades is drift: the edge is gone and the all-time average
            # is a memory of it. Compared against the imported constant, not a
            # literal - this check read `"behind_market"` while the module
            # reported `"forecast_behind_price"`, so it passed everything.
            "not_drifting": (recent_verdict != FORECAST_BEHIND_PRICE),
        }

        is_qualified = all(checks.values())

        # Detailed reasoning showing why win rate alone insufficient
        reasoning = (
            f"Qualification V7 for {venue_id}: "
            f"total {total} >= {self.requirements['min_trades']}? {checks['min_trades']} | "
            f"win_rate {win_rate:.2f} >= {self.requirements['min_win_rate']}? {checks['min_win_rate']} BUT win rate alone NOT profitability - example 90% wins +$0.01 10% losses -$1.00 fantastic win rate still lose money | "
            f"net_pnl ${net_pnl:.2f} >= ${self.requirements['min_net_pnl']}? {checks['min_net_pnl']} | "
            f"expected_value {expected_value*100:.2f}% >= {self.requirements['min_expected_value']*100:.1f}%? {checks['min_ev']} ({ev_source}) on >= {self.requirements['min_ev_coverage']*100:.0f}% of trades? {checks['min_ev_coverage']} | "
            f"evidence from a real book {real_evidence_coverage*100:.0f}% >= "
            f"{self.requirements['min_real_evidence_coverage']*100:.0f}% "
            f"({real_evidence_samples}/{total})? {checks['min_real_evidence']} | "
            f"EV bias "
            f"{('%+.2f%%' % (ev_bias * 100)) if ev_bias is not None else 'unmeasured'} "
            f">= -{self.requirements['min_expected_value']*100:.1f}% over "
            f"{ev_bias_samples} predicted/realised pair(s)? {checks['ev_bias_within_bar']} | "
            f"EXECUTABLE net EV "
            f"{('%+.2f%%' % (float(executable_value)*100)) if executable_value is not None else 'unmeasured'} "
            f">= {self.requirements['min_executable_ev']*100:.1f}% on >= "
            f"{self.requirements['min_executable_ev_coverage']*100:.0f}% of trades? "
            f"{checks['min_executable_ev']} (over {executable_samples}/{total}, "
            f"{executable_coverage*100:.0f}% coverage) | "
            f"fills at the modelled price: paid "
            f"{('%+.2f%%' % (float(fill_price_vs_modelled)*100)) if fill_price_vs_modelled is not None else 'unmeasured'} "
            f"against it, <= {self.requirements['max_fill_price_penalty']*100:.1f}% over "
            f"{price_paid_samples} priced fill(s)? {checks['fills_at_modelled_price']} | "
            f"profit_factor {profit_factor:.2f} >= {self.requirements['min_profit_factor']}? {checks['min_profit_factor']} | "
            f"brier {brier:.3f} <= {self.requirements['max_brier']}? {checks['max_brier']} | "
            f"log_loss {log_loss:.3f} <= {self.requirements['max_log_loss']}? {checks['max_log_loss']} | "
            f"ece {ece:.3f} <= {self.requirements['max_ece']}? {checks['max_ece']} | "
            f"skill {skill:.2f} >= {self.requirements['min_forecast_skill']}? {checks['min_skill']} | "
            f"profit ${profit_paper:.2f} >= ${self.requirements['min_profit_paper']}? {checks['min_profit']} | "
            f"avg_edge {avg_edge*100:.1f}% >= {self.requirements['min_avg_edge']*100:.1f}%? {checks['min_edge']} | "
            f"drawdown {drawdown_max*100:.1f}% <= {self.requirements['max_drawdown']*100:.0f}%? {checks['max_drawdown']} | "
            f"exec_quality {execution_quality:.2f} >= {self.requirements['min_execution_quality']} "
            f"on >= {self.requirements['min_cost_coverage']*100:.0f}% of trades "
            f"(fees measured {cost_coverage*100:.0f}%, quality measured "
            f"{quality_coverage*100:.0f}%)? {checks['min_execution']} | "
            f"fees ${fees_total:.2f} slippage ${slippage_total:.2f} | "
            f"AGAINST THE PRICE: {market_reason or 'unmeasured'} "
            f"[{checks['beats_the_price']}] | "
            f"recent {self.requirements['recent_window_trades']}-trade window: "
            f"{recent_reason or 'unmeasured'} "
            f"({'drifting' if not checks['not_drifting'] else 'not drifting'}) | "
            f"Qualified {is_qualified} | "
            f"FIXED: now considers net P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality not just win rate"
        )

        result = QualificationResult(
            venue_id=venue_id,
            total_resolved_trades=total,
            total_paper_trades=int(performance_stats.get("paper_trades") or 0),
            live_trades=int(performance_stats.get("live_trades") or 0),
            win_rate=win_rate,
            avg_edge=avg_edge,
            brier_score=brier,
            profit_paper=profit_paper,
            profit_live=profit_live,
            forecast_skill=skill,
            is_qualified=is_qualified,
            qualification_date=datetime.now(timezone.utc) if is_qualified else None,
            requirements=self.requirements,
            reasoning=reasoning,
            checks=dict(checks),
            net_pnl=net_pnl,
            expected_value=expected_value,
            fees_total=fees_total,
            slippage_total=slippage_total,
            drawdown_max=drawdown_max,
            profit_factor=profit_factor,
            calibration_ece=ece,
            log_loss=log_loss,
            execution_quality_avg=execution_quality,
            sample_size=total,
            ev_source=ev_source,
            ev_coverage=ev_coverage,
            real_evidence_coverage=real_evidence_coverage,
            ev_bias=ev_bias,
            executable_value=executable_value,
            executable_value_coverage=executable_coverage,
            fill_price_vs_modelled=fill_price_vs_modelled,
            market_skill=market_skill_value,
            market_improvement=(float(market_improvement)
                                if market_improvement is not None else None),
            market_skill_verdict=market_verdict,
            market_skill_ci_low=(float(market_ci_low)
                                 if market_ci_low is not None else None),
            market_skill_ci_high=(float(market_ci_high)
                                  if market_ci_high is not None else None),
            market_skill_p_value=(float(market_p_value)
                                  if market_p_value is not None else None),
            market_skill_samples=market_samples,
            market_skill_reason=market_reason,
            beats_the_price=checks["beats_the_price"],
            not_drifting=checks["not_drifting"],
            recent_market_skill=recent_skill_value,
            recent_market_skill_verdict=recent_verdict,
            recent_market_skill_reason=recent_reason,
            rows_at_evaluation=total,
        )

        # Stamp WHEN this venue last passed, and on how much evidence. A
        # qualification is a judgment about a record; measuring staleness from
        # the last evaluation would always read "fresh", because the loop
        # re-evaluates every cycle - and a stamp that can only ever say "fresh"
        # is not a check.
        previous = self.qualifications.get(venue_id)
        if is_qualified:
            result.qualified_on_trades = total
        elif previous is not None:
            result.qualified_on_trades = previous.qualified_on_trades

        self.qualifications[venue_id] = result
        self._save()

        if is_qualified:
            logger.success(f"Venue {venue_id} QUALIFIED for live trading: {reasoning}")
        else:
            logger.info(f"Venue {venue_id} NOT qualified: {reasoning}")

        return result

    def is_qualified(self, venue_id: str, settled_rows: int = None) -> bool:
        """
        May this venue hold live capital, as of now?

        With `settled_rows` - how many outcomes the venue has TODAY, which only
        the caller with the database can count - the answer also requires the
        judgment not to be stale: a verdict written before the last
        `max_trades_since_evaluation` trades is a verdict about a different
        record. Without it, the stored verdict is returned as stored.

        Nothing in the trading loop calls this: the loop re-evaluates every
        venue every cycle through `evaluate_qualification`, and the bars it
        applies are the same ones. This is the question a reader asks of a
        FILE - the console, an operator, a report - and a file can be old.
        """
        qual = self.qualifications.get(venue_id)
        if not qual:
            return False
        if not qual.is_qualified:
            return False
        if settled_rows is None:
            return True
        return not self.staleness(venue_id, settled_rows)["stale"]

    def staleness(self, venue_id: str, settled_rows: int) -> Dict[str, Any]:
        """
        How much of this venue's record the current judgment does NOT cover.

        Not a statistical test - a statement about the evidence behind a
        verdict. A venue that passed on 100 trades and now has 400 has been
        judged on a quarter of what it has done, and the honest reading is
        "re-judge it", not "it passed once".
        """
        qual = self.qualifications.get(venue_id)
        limit = int(self.requirements.get("max_trades_since_evaluation", 150))
        if qual is None:
            return {"stale": True, "new_trades": None, "limit": limit,
                    "reason": f"{venue_id} has never been evaluated"}
        basis = int(qual.qualified_on_trades or qual.rows_at_evaluation or 0)
        try:
            current = int(settled_rows)
        except (TypeError, ValueError):
            return {"stale": True, "new_trades": None, "limit": limit,
                    "reason": "the venue's settled count could not be read"}
        new_trades = max(0, current - basis)
        stale = new_trades > limit
        return {
            "stale": stale,
            "new_trades": new_trades,
            "limit": limit,
            "judged_on": basis,
            "settled_now": current,
            "reason": (
                f"{new_trades} settled trade(s) since it was judged on "
                f"{basis} - more than the {limit} this judgment covers"
                if stale else
                f"judged on {basis} trades, {new_trades} since (limit {limit})"),
        }

    def get_qualification_report(self) -> Dict[str, Any]:
        qualified = [v for v in self.qualifications.values() if v.is_qualified]
        not_qualified = [v for v in self.qualifications.values() if not v.is_qualified]
        
        return {
            "total_venues": len(self.qualifications),
            "qualified": len(qualified),
            "not_qualified": len(not_qualified),
            "qualified_venues": [q.venue_id for q in qualified],
            "not_qualified_venues": [q.venue_id for q in not_qualified],
            "requirements": self.requirements,
            "requirements_explanation": {
                "min_trades": "100 paper trades - sample size",
                "min_win_rate": "55%+ win rate - BUT not profitability alone",
                "min_net_pnl": "Net P&L after fees/slippage must be positive - fixes 90% wins +$0.01 10% losses -$1.00 still lose money example",
                "min_expected_value": "EV >1% per trade - expected value",
                "min_profit_factor": "Gross profit / gross loss >1.1 - profit factor",
                "max_brier": "Brier <=0.25 - calibration",
                "max_log_loss": "Log loss <=0.6",
                "max_ece": "ECE <=0.15 - calibration error",
                "max_drawdown": "Max 20% drawdown",
                "min_execution_quality": "Avg execution quality 0.5+"
            },
            "details": {
                q.venue_id: {
                    "total": q.total_paper_trades,
                    "win_rate": q.win_rate,
                    "net_pnl": q.net_pnl,
                    "ev": q.expected_value,
                    "profit_factor": q.profit_factor,
                    "brier": q.brier_score,
                    "log_loss": q.log_loss,
                    "ece": q.calibration_ece,
                    "skill": q.forecast_skill,
                    "profit": q.profit_paper,
                    "drawdown": q.drawdown_max,
                    "fees": q.fees_total,
                    "slippage": q.slippage_total,
                    "exec_quality": q.execution_quality_avg,
                    "qualified": q.is_qualified,
                    "reasoning": q.reasoning[:500]
                } for q in self.qualifications.values()
            },
            "principle": "Each venue must prove positive EV through paper trading before real capital, don't assume profitable prove via paper trading/backtesting then cautiously allocate. FIXED V7: win rate alone NOT profitability - now considers net P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality"
        }

    def should_concentrate_on(self) -> Dict[str, Any]:
        qualified = [v for v in self.qualifications.values() if v.is_qualified]
        qualified_sorted = sorted(qualified, key=lambda x: x.forecast_skill, reverse=True)
        
        if not qualified_sorted:
            return {"message": "No qualified venues yet - need paper trading 100+ trades per venue", "recommendation": "Start with Polymarket paper trading"}
        
        return {
            "strong_venues": [f"{q.venue_id} skill={q.forecast_skill:.2f} win={q.win_rate:.2f} brier={q.brier_score:.3f} pnl=${q.net_pnl:.2f} pf={q.profit_factor:.2f}" for q in qualified_sorted[:3]],
            "recommendation": f"Concentrate on {qualified_sorted[0].venue_id} - demonstrated skill {qualified_sorted[0].forecast_skill:.2f} pnl ${qualified_sorted[0].net_pnl:.2f}",
            "principle": "PTAI learns which venue/category combos it is good at and concentrates research there"
        }
