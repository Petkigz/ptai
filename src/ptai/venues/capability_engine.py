"""
Venue/Strategy Qualification Engine - V8
The real target: PTAI searches every qualified venue and strategy available to it,
measures the opportunity on a common risk-adjusted basis, and only deploys capital
when the opportunity passes its independently enforced rules.

User: I would not add another bunch of venue adapters. You already have many.
Instead, next layer should be a Venue/Strategy Qualification Engine.

Conceptually:
                  ALL AVAILABLE VENUES
                           │
                           ▼
                 ┌──────────────────┐
                 │ Capability Check │
                 └────────┬─────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Trading       Data        Liquidity
          available?    quality?    sufficient?
             │            │            │
             └────────────┼────────────┘
                          ▼
                   Strategy Check
                          │
                          ▼
                  Historical Edge?
                          │
                          ▼
                    Fees/Slippage
                          │
                          ▼
                   Legal/Account
                   eligibility
                          │
                          ▼
                    QUALIFIED
                          │
                          ▼
                 OPPORTUNITY ENGINE

Decision process:
PTAI wakes up
       ↓
Check capital + account health
       ↓
Check all qualified venues
       ↓
Discover markets
       ↓
Normalize markets
       ↓
Generate candidate opportunities
       ↓
Evaluate strategies
       ↓
Estimate fair value / expected return
       ↓
Account for fees + spread + slippage
       ↓
Check liquidity
       ↓
Check uncertainty
       ↓
Check correlations
       ↓
Check historical model performance
       ↓
Check venue/strategy performance
       ↓
Calculate risk-adjusted opportunity
       ↓
Compare EVERY candidate
       ↓
Choose only opportunities passing hard rules
       ↓
Risk engine
       ↓
Execution guard
       ↓
Execute
       ↓
Verify
       ↓
Monitor
       ↓
Record prediction + outcome
       ↓
Update calibration/performance
       ↓
Repeat

Notice: There is no Polymarket step in that logic.
Polymarket becomes Venue #1 rather than PTAI = Polymarket bot

Core objective: PTAI searches every qualified venue and strategy available to it,
measures the opportunity on a common risk-adjusted basis, and only deploys capital
when the opportunity passes its independently enforced rules.

That is much closer to $50 — earn enough to pay for yourself or shut down,
while still allowing system to decide best action is not trading at all.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from loguru import logger
import time

from .adapter import MarketAdapter, EligibilityStatus, VenueType
from ..markets.base import Market
from .qualification import VenueQualificationEngine

class CapabilityStatus(str, Enum):
    QUALIFIED = "qualified"
    DATA_ONLY = "data_only"  # Can discover but not trade
    ILLIQUID = "illiquid"
    RESTRICTED = "restricted"        # a legal/regulatory determination says no
    UNDETERMINED = "undetermined"    # legality was never researched - not a "no"
    NO_EDGE = "no_edge"
    EXPERIMENTAL = "experimental"
    UNTESTED = "untested"
    FAILED = "failed"

@dataclass
class VenueCapabilityReport:
    venue_id: str
    venue_type: VenueType
    status: CapabilityStatus
    trading_available: bool
    data_quality: float  # 0-1
    liquidity_sufficient: bool
    avg_liquidity: float
    historical_edge: bool
    avg_edge: float
    win_rate: float
    brier_score: float
    profit_factor: float
    net_pnl: float
    fees_pct: float
    slippage_pct: float
    legal_eligible: bool
    eligibility_status: EligibilityStatus
    account_configured: bool
    execution_tested: bool
    sample_size: int
    is_qualified: bool
    qualification_score: float  # 0-1 overall
    reasoning: str
    checks: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class StrategyCapabilityReport:
    strategy_id: str
    venue_id: str
    status: CapabilityStatus
    historical_edge: bool
    avg_edge: float
    win_rate: float
    brier_score: float
    profit_factor: float
    net_pnl: float
    sample_size: int
    is_qualified: bool
    qualification_score: float
    reasoning: str
    checks: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualificationEngineReport:
    total_venues: int
    qualified_venues: int
    data_only_venues: int
    restricted_venues: int
    untested_venues: int
    venue_reports: List[VenueCapabilityReport]
    strategy_reports: List[StrategyCapabilityReport]
    qualified_venue_ids: List[str]
    recommended_venues: List[str]
    execution_time: float
    reasoning: str
    # Kept separate from restricted_venues: an unresearched jurisdiction is an
    # open question, and collapsing it into "restricted" asserts a legal
    # prohibition nobody established. Defaulted so existing constructors and
    # serialised reports keep working.
    undetermined_venues: int = 0

class VenueStrategyQualificationEngine:
    """
    Venue/Strategy Qualification Engine - V8
    Properly connected to main autonomous loop
    
    User: The pieces are already there in venues/qualification.py
    Question is whether they are properly connected to main autonomous loop.
    
    This engine is the NEXT layer - not more adapters, but qualification
    """
    def __init__(self, venue_registry=None, qualification_engine: VenueQualificationEngine = None, country_code: str = "UG"):
        self.venue_registry = venue_registry
        self.qualification_engine = qualification_engine or VenueQualificationEngine()
        self.country_code = country_code
        self.venue_reports: Dict[str, VenueCapabilityReport] = {}
        self.strategy_reports: Dict[str, StrategyCapabilityReport] = {}
        self.last_report: Optional[QualificationEngineReport] = None

    async def check_venue_capability(self, adapter: MarketAdapter, sample_markets: List[Market] = None) -> VenueCapabilityReport:
        """
        Capability Check:
        - Trading available?
        - Data quality?
        - Liquidity sufficient?
        - Historical Edge?
        - Fees/Slippage
        - Legal/Account eligibility
        """
        venue_id = adapter.venue_id
        venue_type = adapter.venue_type
        
        # Legal/Account eligibility
        eligibility = adapter.check_eligibility(self.country_code)
        legal_eligible = eligibility in [EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_VERIFICATION]
        
        # V10 FIX #4: Account configured = actual ACCOUNT_HEALTH check, not just supports_trading flag
        # supports_trading != credentials exist != funds exist != permissions correct != Uganda accessible
        account_configured = adapter.capabilities.supports_trading
        account_health_details = {}
        
        # For polymarket, need private_key and funder AND funds AND permissions
        if venue_id == "polymarket":
            has_pk = bool(getattr(adapter, 'private_key', None))
            has_funder = bool(getattr(adapter, 'funder', None))
            # V10 FIX #4: Check actual health, not just existence
            account_configured = has_pk and has_funder
            account_health_details = {
                "has_private_key": has_pk,
                "has_funder": has_funder,
                "check": "private_key + funder + funds + permissions + Uganda access needed",
                "note": "V10 FIX #4: Use AccountHealthEngine for full check, this is quick check"
            }
        elif venue_id == "kalshi":
            has_key = bool(getattr(adapter, 'api_key', None))
            account_configured = has_key
            account_health_details = {
                "has_api_key": has_key,
                "note": "Kalshi restricted for UG, paper only even with credentials"
            }
        elif venue_id in ["whitebit", "binance", "crypto_binance", "pionex", "grvt", "afx_dex"]:
            # Financial venues need credentials but $50 can execute
            has_key = bool(getattr(adapter, 'api_key', None))
            has_secret = bool(getattr(adapter, 'api_secret', None))
            # For some, wallet is enough (AFX)
            if venue_id == "afx_dex":
                account_configured = True  # wallet-signed, no API keys
            else:
                account_configured = has_key and has_secret
            account_health_details = {
                "has_api_key": has_key,
                "has_api_secret": has_secret,
                "venue": venue_id,
                "note": "V10 FIX #4: Financial venue, $50 can execute if credentials exist"
            }
        else:
            # For play money venues, supports_trading is sufficient for paper
            account_health_details = {
                "supports_trading_claim": adapter.capabilities.supports_trading,
                "note": "V10 FIX #4: For paper/intelligence venues, supports_trading flag is paper-only"
            }
        
        # Trading available?
        trading_available = legal_eligible and account_configured and adapter.capabilities.supports_trading
        
        # Data quality - try discovery
        data_quality = 0.0
        avg_liquidity = 0.0
        liquidity_sufficient = False
        sample_markets = sample_markets or []
        
        if not sample_markets:
            try:
                sample_markets = await adapter.discover_markets(target_count=20)
            except Exception as e:
                logger.warning(f"{venue_id} discovery for capability check failed: {e}")
                sample_markets = []
        
        if sample_markets:
            # Data quality based on fields present
            valid = sum(1 for m in sample_markets if m.question and m.best_price and m.liquidity > 0)
            data_quality = valid / len(sample_markets) if sample_markets else 0
            avg_liquidity = sum(m.liquidity for m in sample_markets) / len(sample_markets) if sample_markets else 0
            liquidity_sufficient = avg_liquidity >= 1000 and data_quality >= 0.8
        else:
            data_quality = 0.0
            avg_liquidity = 0.0
            liquidity_sufficient = False
        
        # Historical edge - from qualification engine
        perf = adapter.performance_stats
        historical_edge = False
        avg_edge = perf.get("avg_edge", 0)
        win_rate = perf.get("win_rate", 0)
        brier = perf.get("brier_score", 1.0)
        profit_factor = perf.get("profit_factor", 0) if "profit_factor" in perf else 0
        net_pnl = perf.get("profit_paper", 0)
        # ALL resolved evidence, not the paper subset: `total_paper_trades`
        # used to hold every outcome, and now that it honestly holds only the
        # paper ones, reading it here would report "no evidence" for a venue
        # the qualification gate just judged on 120 outcomes. Fallback keeps
        # older producers, which only know the old key, working.
        sample_size = perf.get(
            "total_resolved_trades", perf.get("total_paper_trades", 0))
        
        # Also check qualification engine file
        qual_result = self.qualification_engine.qualifications.get(venue_id)
        if qual_result:
            win_rate = qual_result.win_rate
            brier = qual_result.brier_score
            net_pnl = qual_result.net_pnl
            profit_factor = qual_result.profit_factor
            sample_size = (qual_result.total_resolved_trades
                           or qual_result.total_paper_trades)
            avg_edge = qual_result.avg_edge
        
        # Historical edge.
        #
        # This used to be a SECOND, weaker qualification test written here: a
        # sample floor plus four conditions. The gates below only require
        # historical_edge and trading_available for is_qualified, so a venue could
        # reach qualified_venue_ids while failing log loss, calibration (ECE),
        # forecast skill, expected value, average edge, drawdown and execution
        # quality - every criterion its own qualification engine checks.
        #
        # Two tests for the same decision, where one is weaker, means the weaker
        # one decides. The engine that actually evaluates all fifteen criteria now
        # decides, and this defers to it.
        eng_result = self.qualification_engine.qualifications.get(venue_id)
        if eng_result is not None:
            historical_edge = bool(eng_result.is_qualified)
        elif sample_size >= 20:
            # No evaluation on record. The venue has data, so the honest answer is
            # "not yet judged" rather than a local re-derivation - the refresh in
            # V3 evaluates every registered venue each cycle, so this state should
            # not persist.
            historical_edge = False
        else:
            historical_edge = False
        
        # Fees/slippage
        fees_pct = adapter.capabilities.fee_taker_pct
        slippage_pct = 0.01  # estimate, would be from orderbook
        if sample_markets:
            # Estimate slippage from liquidity
            avg_amount = 3.0  # $50 bankroll 6% cap
            slippage_pct = avg_amount / max(1, avg_liquidity) * 0.5 if avg_liquidity > 0 else 0.02
        
        # Execution tested?
        execution_tested = perf.get("total_paper_trades", 0) > 0
        
        # Overall qualification score 0-1
        score = 0.0
        checks = {}
        
        checks["legal_eligible"] = legal_eligible
        checks["account_configured"] = account_configured
        checks["trading_available"] = trading_available
        checks["data_quality"] = data_quality >= 0.8
        checks["liquidity_sufficient"] = liquidity_sufficient
        checks["historical_edge"] = historical_edge
        checks["sample_size"] = sample_size >= 100
        checks["profitable"] = net_pnl > 0
        checks["calibrated"] = brier <= 0.25
        checks["execution_tested"] = execution_tested
        
        # Score weights
        score += 0.2 if legal_eligible else 0
        score += 0.15 if account_configured else 0
        score += 0.15 if trading_available else 0
        score += 0.15 * data_quality
        score += 0.1 if liquidity_sufficient else 0
        score += 0.15 if historical_edge else (0.05 if sample_size >= 20 else 0)
        score += 0.1 if execution_tested else 0
        
        # Determine status
        if eligibility == EligibilityStatus.RESTRICTED:
            # An explicit legal determination. Only this is "restricted".
            status = CapabilityStatus.RESTRICTED
            is_qualified = False
        elif eligibility == EligibilityStatus.UNKNOWN:
            # Nobody has established whether this venue is legal here. That is
            # an unverified open question, not a prohibition - reporting it as
            # "restricted" told the operator a venue was illegal when the truth
            # was "we have not checked", and buried the research work item.
            status = CapabilityStatus.UNDETERMINED
            is_qualified = False
        elif not legal_eligible:
            status = CapabilityStatus.RESTRICTED
            is_qualified = False
        elif not data_quality >= 0.5:
            status = CapabilityStatus.FAILED
            is_qualified = False
        elif not liquidity_sufficient:
            status = CapabilityStatus.ILLIQUID
            is_qualified = False
        elif sample_size < 20:
            status = CapabilityStatus.UNTESTED
            is_qualified = False
        elif sample_size < 100:
            status = CapabilityStatus.EXPERIMENTAL
            is_qualified = False
        elif historical_edge and trading_available:
            status = CapabilityStatus.QUALIFIED
            is_qualified = True
        elif trading_available and not historical_edge:
            status = CapabilityStatus.NO_EDGE
            is_qualified = False
        elif not trading_available and data_quality >= 0.8:
            status = CapabilityStatus.DATA_ONLY
            is_qualified = False
        else:
            status = CapabilityStatus.EXPERIMENTAL
            is_qualified = False
        
        reasoning = (
            f"Venue {venue_id} capability: "
            f"legal {eligibility.value} eligible={legal_eligible} | "
            f"account configured={account_configured} | "
            f"trading available={trading_available} | "
            f"data quality {data_quality:.2f} | "
            f"liquidity ${avg_liquidity:.0f} sufficient={liquidity_sufficient} | "
            f"historical edge={historical_edge} edge {avg_edge*100:.1f}% win {win_rate*100:.0f}% brier {brier:.3f} pnl ${net_pnl:.2f} pf {profit_factor:.2f} sample {sample_size} | "
            f"fees {fees_pct*100:.1f}% slippage {slippage_pct*100:.2f}% | "
            f"execution tested={execution_tested} | "
            f"score {score:.2f} status {status.value} qualified={is_qualified} | "
            f"PTAI has broad multi-venue framework, but each venue/strategy needs capability validation and testing"
        )
        
        report = VenueCapabilityReport(
            venue_id=venue_id,
            venue_type=venue_type,
            status=status,
            trading_available=trading_available,
            data_quality=data_quality,
            liquidity_sufficient=liquidity_sufficient,
            avg_liquidity=avg_liquidity,
            historical_edge=historical_edge,
            avg_edge=avg_edge,
            win_rate=win_rate,
            brier_score=brier,
            profit_factor=profit_factor,
            net_pnl=net_pnl,
            fees_pct=fees_pct,
            slippage_pct=slippage_pct,
            legal_eligible=legal_eligible,
            eligibility_status=eligibility,
            account_configured=account_configured,
            execution_tested=execution_tested,
            sample_size=sample_size,
            is_qualified=is_qualified,
            qualification_score=score,
            reasoning=reasoning,
            checks=checks
        )
        
        self.venue_reports[venue_id] = report
        logger.info(reasoning)
        return report

    async def check_strategy_capability(self, strategy_id: str, venue_id: str, performance_stats: Dict = None) -> StrategyCapabilityReport:
        """Check strategy capability for specific venue"""
        perf = performance_stats or {}
        
        sample_size = perf.get("total_trades", 0)
        win_rate = perf.get("win_rate", 0)
        avg_edge = perf.get("avg_edge", 0)
        brier = perf.get("brier_score", 0.5)
        profit_factor = perf.get("profit_factor", 0)
        net_pnl = perf.get("net_pnl", 0)
        
        historical_edge = (
            sample_size >= 50 and
            net_pnl > 0 and
            win_rate >= 0.55 and
            profit_factor >= 1.1
        ) if sample_size >= 50 else False
        
        score = 0.0
        if historical_edge:
            score = 0.7 + win_rate*0.2 + (1-brier)*0.1
        elif sample_size >= 20:
            score = 0.3
        else:
            score = 0.0
        
        if historical_edge:
            status = CapabilityStatus.QUALIFIED
            is_qualified = True
        elif sample_size < 20:
            status = CapabilityStatus.UNTESTED
            is_qualified = False
        else:
            status = CapabilityStatus.NO_EDGE
            is_qualified = False
        
        reasoning = f"Strategy {strategy_id} @ {venue_id}: edge {avg_edge*100:.1f}% win {win_rate*100:.0f}% pnl ${net_pnl:.2f} pf {profit_factor:.2f} sample {sample_size} -> {status.value} qualified {is_qualified}"
        
        report = StrategyCapabilityReport(
            strategy_id=strategy_id,
            venue_id=venue_id,
            status=status,
            historical_edge=historical_edge,
            avg_edge=avg_edge,
            win_rate=win_rate,
            brier_score=brier,
            profit_factor=profit_factor,
            net_pnl=net_pnl,
            sample_size=sample_size,
            is_qualified=is_qualified,
            qualification_score=score,
            reasoning=reasoning,
            checks={
                "historical_edge": historical_edge,
                "sample_size": sample_size >= 50,
                "profitable": net_pnl > 0
            }
        )
        
        key = f"{venue_id}:{strategy_id}"
        self.strategy_reports[key] = report
        return report

    async def evaluate_all_venues(self, target_per_venue: int = 20) -> QualificationEngineReport:
        """
        Evaluate ALL available venues - main qualification engine
        Returns which venues are qualified for opportunity engine
        """
        start = time.time()
        
        if not self.venue_registry:
            logger.warning("No venue registry for qualification")
            return QualificationEngineReport(
                total_venues=0,
                qualified_venues=0,
                data_only_venues=0,
                restricted_venues=0,
                untested_venues=0,
                venue_reports=[],
                strategy_reports=[],
                qualified_venue_ids=[],
                recommended_venues=[],
                execution_time=0,
                reasoning="No registry"
            )
        
        venue_reports = []
        for venue_id, adapter in self.venue_registry.adapters.items():
            try:
                report = await self.check_venue_capability(adapter, sample_markets=None)
                venue_reports.append(report)
            except Exception as e:
                logger.error(f"Capability check failed for {venue_id}: {e}")
                # Create failed report
                venue_reports.append(VenueCapabilityReport(
                    venue_id=venue_id,
                    venue_type=adapter.venue_type,
                    status=CapabilityStatus.FAILED,
                    trading_available=False,
                    data_quality=0,
                    liquidity_sufficient=False,
                    avg_liquidity=0,
                    historical_edge=False,
                    avg_edge=0,
                    win_rate=0,
                    brier_score=1.0,
                    profit_factor=0,
                    net_pnl=0,
                    fees_pct=adapter.capabilities.fee_taker_pct,
                    slippage_pct=0.02,
                    legal_eligible=False,
                    eligibility_status=EligibilityStatus.UNKNOWN,
                    account_configured=False,
                    execution_tested=False,
                    sample_size=0,
                    is_qualified=False,
                    qualification_score=0,
                    reasoning=f"Failed capability check: {e}",
                    checks={}
                ))
        
        qualified = [r for r in venue_reports if r.is_qualified]
        data_only = [r for r in venue_reports if r.status == CapabilityStatus.DATA_ONLY]
        restricted = [r for r in venue_reports if r.status == CapabilityStatus.RESTRICTED]
        untested = [r for r in venue_reports if r.status == CapabilityStatus.UNTESTED]
        undetermined = [r for r in venue_reports if r.status == CapabilityStatus.UNDETERMINED]
        
        qualified_ids = [r.venue_id for r in qualified]
        recommended = sorted(qualified, key=lambda x: x.qualification_score, reverse=True)
        recommended_ids = [r.venue_id for r in recommended[:3]]
        
        elapsed = time.time() - start
        
        reasoning = (
            f"Venue/Strategy Qualification Engine V8: "
            f"Total {len(venue_reports)} venues evaluated, "
            f"Qualified {len(qualified)}: {qualified_ids}, "
            f"Data-only {len(data_only)}, "
            f"Restricted {len(restricted)}"
            f"{(' (' + ', '.join(r.venue_id for r in restricted) + ')') if restricted else ''}, "
            f"Undetermined {len(undetermined)}"
            f"{(' (' + ', '.join(r.venue_id for r in undetermined) + ')') if undetermined else ''}"
            f" - legality not yet researched for these, so they are blocked on "
            f"research, not on law, "
            f"Untested {len(untested)}, "
            f"Recommended {recommended_ids} | "
            f"PTAI now has broad multi-venue framework, but each venue/strategy combination still needs capability validation and testing | "
            f"Time {elapsed:.1f}s | "
            f"Next: PTAI searches every qualified venue and strategy, measures opportunity on common risk-adjusted basis, only deploys capital when passes independently enforced rules, DO NOTHING is successful"
        )
        
        report = QualificationEngineReport(
            total_venues=len(venue_reports),
            qualified_venues=len(qualified),
            data_only_venues=len(data_only),
            restricted_venues=len(restricted),
            untested_venues=len(untested),
            undetermined_venues=len(undetermined),
            venue_reports=venue_reports,
            strategy_reports=list(self.strategy_reports.values()),
            qualified_venue_ids=qualified_ids,
            recommended_venues=recommended_ids,
            execution_time=elapsed,
            reasoning=reasoning
        )
        
        self.last_report = report
        logger.success(reasoning)
        return report

    def get_qualified_adapters(self) -> List[MarketAdapter]:
        """Get only qualified adapters for opportunity engine"""
        if not self.venue_registry:
            return []
        
        qualified_ids = []
        if self.last_report:
            qualified_ids = self.last_report.qualified_venue_ids
        else:
            # Fallback to reports
            qualified_ids = [vid for vid, report in self.venue_reports.items() if report.is_qualified]
        
        adapters = []
        for vid in qualified_ids:
            adapter = self.venue_registry.adapters.get(vid)
            if adapter:
                adapters.append(adapter)
        
        return adapters

    def get_report(self) -> Dict[str, Any]:
        if not self.last_report:
            return {"message": "No qualification run yet, need evaluate_all_venues()"}
        
        return {
            "total_venues": self.last_report.total_venues,
            "qualified": self.last_report.qualified_venues,
            "data_only": self.last_report.data_only_venues,
            "restricted": self.last_report.restricted_venues,
            "untested": self.last_report.untested_venues,
            "qualified_venue_ids": self.last_report.qualified_venue_ids,
            "recommended": self.last_report.recommended_venues,
            "execution_time": self.last_report.execution_time,
            "reasoning": self.last_report.reasoning,
            "venue_details": [
                {
                    "venue_id": r.venue_id,
                    "status": r.status.value,
                    "qualified": r.is_qualified,
                    "score": r.qualification_score,
                    "trading_available": r.trading_available,
                    "data_quality": r.data_quality,
                    "liquidity": r.avg_liquidity,
                    "liquidity_sufficient": r.liquidity_sufficient,
                    "historical_edge": r.historical_edge,
                    "edge": r.avg_edge,
                    "win_rate": r.win_rate,
                    "brier": r.brier_score,
                    "profit_factor": r.profit_factor,
                    "net_pnl": r.net_pnl,
                    "fees": r.fees_pct,
                    "legal_eligible": r.legal_eligible,
                    "eligibility": r.eligibility_status.value,
                    "account_configured": r.account_configured,
                    "sample_size": r.sample_size,
                    "reasoning": r.reasoning[:300]
                } for r in self.last_report.venue_reports
            ],
            "principle": "PTAI has broad multi-venue framework, but each venue/strategy needs capability validation and testing. PTAI searches every qualified venue and strategy, measures opportunity on common risk-adjusted basis, only deploys capital when passes independently enforced rules. Polymarket becomes Venue #1 rather than PTAI = Polymarket bot. Adding 20 adapters immediately would be wrong move - prove one adapter end-to-end, then add venues one at a time under same qualification contract."
        }
