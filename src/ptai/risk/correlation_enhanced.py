"""
Correlation-Aware Risk Caps - Enhanced
- Correlation clusters: aggregate exposure to same event, cap per cluster
- Fractional Kelly: 1/4 Kelly not full Kelly, cap at 6% bankroll but lower for low confidence
- Scenario stress tests: simulate all correlated positions losing at once, if wipes account cut size
- Time stop, liquidity stop, daily/weekly loss limits, kill switch

Top 5 to implement first per user - prevents one event from wiping you out
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger
import math

from ..markets.base import Market
from .correlation import CorrelationEngine, CorrelationGroup
from ..markets.orderbook import read_spread


@dataclass
class ExposureCap:
    cluster_id: str
    cluster_name: str
    current_exposure_usd: float
    current_exposure_pct: float
    max_exposure_usd: float
    max_exposure_pct: float
    num_positions: int
    max_positions: int
    can_add: bool
    remaining_capacity_usd: float
    reasoning: str


@dataclass
class StressTestResult:
    scenario: str
    affected_clusters: List[str]
    total_exposure_usd: float
    loss_if_all_lose_usd: float
    loss_if_all_lose_pct: float
    would_wipe: bool
    bankroll_after: float
    should_cut: bool
    reasoning: str


@dataclass
class FractionalKellyResult:
    market_id: str
    full_kelly_pct: float
    quarter_kelly_pct: float
    confidence_adjusted_pct: float
    capped_pct: float
    amount_usd: float
    reasoning: str


class CorrelationAwareRiskManager:
    """
    Enhanced correlation risk manager with exposure caps and stress tests
    """
    def __init__(self, bankroll: float = 50.0, max_cluster_pct: float = 0.12, max_cluster_positions: int = 2):
        self.bankroll = bankroll
        self.max_cluster_pct = max_cluster_pct  # Max 12% per cluster (2 positions * 6%)
        self.max_cluster_positions = max_cluster_positions
        self.correlation_engine = CorrelationEngine()

    def calculate_cluster_exposure(self, positions: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate exposure per correlation cluster
        positions: list of {market_id, question, amount_usd, side, etc}
        """
        clusters: Dict[str, Dict] = {}
        
        for pos in positions:
            # Get cluster for position
            question = pos.get("question", "")
            topics = self.correlation_engine.extract_topics_from_question(question)
            cluster_id = list(topics)[0] if topics else "uncorrelated"
            
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "exposure_usd": 0,
                    "positions": [],
                    "market_ids": []
                }
            
            clusters[cluster_id]["exposure_usd"] += pos.get("amount_usd", 0)
            clusters[cluster_id]["positions"].append(pos)
            clusters[cluster_id]["market_ids"].append(pos.get("market_id", ""))
        
        # Calculate pct
        for cluster_id, data in clusters.items():
            data["exposure_pct"] = data["exposure_usd"] / self.bankroll if self.bankroll > 0 else 0
            data["num_positions"] = len(data["positions"])
        
        return clusters

    def check_exposure_caps(self, new_market: Market, amount_usd: float,
                           existing_positions: List[Dict]) -> ExposureCap:
        """
        Check if adding new market would exceed cluster exposure caps
        """
        clusters = self.calculate_cluster_exposure(existing_positions)
        
        # Find cluster for new market
        new_topics = self.correlation_engine.extract_topics(new_market)
        new_cluster_id = list(new_topics)[0] if new_topics else "uncorrelated"
        cluster_name = new_cluster_id
        
        current_exposure = clusters.get(new_cluster_id, {}).get("exposure_usd", 0)
        current_pct = clusters.get(new_cluster_id, {}).get("exposure_pct", 0)
        num_positions = clusters.get(new_cluster_id, {}).get("num_positions", 0)
        
        max_exposure_usd = self.bankroll * self.max_cluster_pct
        max_exposure_pct = self.max_cluster_pct
        
        new_total_exposure = current_exposure + amount_usd
        new_total_pct = new_total_exposure / self.bankroll if self.bankroll > 0 else 0
        new_num_positions = num_positions + 1
        
        can_add = True
        reasons = []
        
        if new_total_pct > max_exposure_pct:
            can_add = False
            reasons.append(f"Cluster {new_cluster_id} exposure {new_total_pct*100:.1f}% > max {max_exposure_pct*100:.1f}%")
        
        if new_num_positions > self.max_cluster_positions:
            can_add = False
            reasons.append(f"Cluster {new_cluster_id} positions {new_num_positions} > max {self.max_cluster_positions}")
        
        if not reasons:
            reasons.append(f"OK: cluster {new_cluster_id} {new_total_pct*100:.1f}% exposure {new_num_positions} positions")
        
        remaining = max(0, max_exposure_usd - current_exposure)
        
        reasoning = (
            f"Cluster {cluster_name} ({new_cluster_id}): current ${current_exposure:.2f} ({current_pct*100:.1f}%) {num_positions} positions | "
            f"Adding ${amount_usd:.2f} => total ${new_total_exposure:.2f} ({new_total_pct*100:.1f}%) {new_num_positions} positions | "
            f"Max ${max_exposure_usd:.2f} ({max_exposure_pct*100:.1f}%) max positions {self.max_cluster_positions} | "
            f"Can add {can_add} remaining ${remaining:.2f} | "
            f"Reasons: {'; '.join(reasons)}"
        )
        
        return ExposureCap(
            cluster_id=new_cluster_id,
            cluster_name=cluster_name,
            current_exposure_usd=current_exposure,
            current_exposure_pct=current_pct,
            max_exposure_usd=max_exposure_usd,
            max_exposure_pct=max_exposure_pct,
            num_positions=num_positions,
            max_positions=self.max_cluster_positions,
            can_add=can_add,
            remaining_capacity_usd=remaining,
            reasoning=reasoning
        )

    def fractional_kelly(self, market_price: float, fair_prob: float,
                        confidence: float, bankroll: float = None) -> FractionalKellyResult:
        """
        Fractional Kelly: 1/4 Kelly not full Kelly, cap at 6% but lower for low confidence
        
        Full Kelly: f = (bp - q)/b where b = (1-m)/m, p = fair prob, q = 1-p, m = market price
        Example: market 0.60, fair 0.75, edge 15%, b=(1-0.6)/0.6=0.666, Kelly raw (0.666*0.75-0.25)/0.666=37%, Half-Kelly 18.5%, Quarter-Kelly 9.25% capped 6% => $3 on $50
        
        But for $50 bankroll, cap at 6% $3, but lower for low confidence
        """
        bankroll = bankroll or self.bankroll
        market_price = max(0.01, min(0.99, market_price))
        fair_prob = max(0.01, min(0.99, fair_prob))
        
        # Calculate b = odds
        b = (1 - market_price) / market_price if market_price > 0 else 1
        
        p = fair_prob
        q = 1 - p
        
        # Full Kelly
        full_kelly = (b * p - q) / b if b > 0 else 0
        full_kelly = max(0, full_kelly)  # no negative
        
        # Fractional: 1/4 Kelly
        quarter_kelly = full_kelly * 0.25
        
        # Confidence adjustment: low confidence => lower size
        # Confidence 0.5 => 50% of quarter Kelly, 0.8 => 80%, 1.0 => 100%
        confidence_adjusted = quarter_kelly * confidence
        
        # Cap at 6% bankroll, but also $1 min for $50
        capped_pct = min(0.06, confidence_adjusted)
        # For low confidence <0.6, cap even lower at 3%
        if confidence < 0.6:
            capped_pct = min(0.03, capped_pct)
        # For very low confidence <0.4, cap at 1.5%
        if confidence < 0.4:
            capped_pct = min(0.015, capped_pct)
        
        amount_usd = bankroll * capped_pct
        # $1 min position for $50 bankroll
        if amount_usd < 1.0 and capped_pct > 0:
            amount_usd = 1.0
            capped_pct = amount_usd / bankroll if bankroll > 0 else 0
        
        reasoning = (
            f"Market {market_price:.3f} fair {fair_prob:.3f} edge {(fair_prob-market_price)*100:.1f}% | "
            f"b={(1-market_price)/market_price:.3f} p={p:.3f} q={q:.3f} | "
            f"Full Kelly {full_kelly*100:.1f}% Quarter Kelly {quarter_kelly*100:.1f}% | "
            f"Confidence {confidence:.2f} => confidence adjusted {confidence_adjusted*100:.1f}% | "
            f"Capped {capped_pct*100:.1f}% (${amount_usd:.2f} on ${bankroll}) | "
            f"Cap 6% $3 on $50 but lower for low confidence: <0.6 cap 3%, <0.4 cap 1.5%, min $1"
        )
        
        return FractionalKellyResult(
            market_id="",
            full_kelly_pct=full_kelly,
            quarter_kelly_pct=quarter_kelly,
            confidence_adjusted_pct=confidence_adjusted,
            capped_pct=capped_pct,
            amount_usd=amount_usd,
            reasoning=reasoning
        )

    def stress_test_correlated_loss(self, positions: List[Dict], bankroll: float = None) -> List[StressTestResult]:
        """
        Scenario stress tests: simulate all correlated positions losing at once
        If that wipes account, cut size
        """
        bankroll = bankroll or self.bankroll
        clusters = self.calculate_cluster_exposure(positions)
        
        results: List[StressTestResult] = []
        
        # Test each cluster losing
        for cluster_id, data in clusters.items():
            exposure = data["exposure_usd"]
            loss = exposure  # assume 100% loss if all lose
            loss_pct = loss / bankroll if bankroll > 0 else 0
            bankroll_after = bankroll - loss
            would_wipe = bankroll_after < bankroll * 0.5  # loses >50% bankroll
            should_cut = loss_pct > 0.15  # >15% bankroll in one cluster is risky
            
            reasoning = (
                f"Stress test cluster {cluster_id}: {data['num_positions']} positions ${exposure:.2f} ({exposure/bankroll*100:.1f}% bankroll) | "
                f"If all lose: loss ${loss:.2f} ({loss_pct*100:.1f}%) bankroll ${bankroll:.2f} -> ${bankroll_after:.2f} | "
                f"Would wipe >50% {would_wipe} should cut >15% {should_cut} | "
                f"Markets: {', '.join(data['market_ids'][:3])}"
            )
            
            results.append(StressTestResult(
                scenario=f"All {cluster_id} positions lose",
                affected_clusters=[cluster_id],
                total_exposure_usd=exposure,
                loss_if_all_lose_usd=loss,
                loss_if_all_lose_pct=loss_pct,
                would_wipe=would_wipe,
                bankroll_after=bankroll_after,
                should_cut=should_cut,
                reasoning=reasoning
            ))
        
        # Test worst case: all positions lose
        total_exposure = sum(data["exposure_usd"] for data in clusters.values())
        total_loss_pct = total_exposure / bankroll if bankroll > 0 else 0
        results.append(StressTestResult(
            scenario="All positions lose (black swan)",
            affected_clusters=list(clusters.keys()),
            total_exposure_usd=total_exposure,
            loss_if_all_lose_usd=total_exposure,
            loss_if_all_lose_pct=total_loss_pct,
            would_wipe=total_loss_pct > 0.5,
            bankroll_after=bankroll - total_exposure,
            should_cut=total_loss_pct > 0.25,
            reasoning=f"Black swan: all {len(positions)} positions lose ${total_exposure:.2f} ({total_loss_pct*100:.1f}% bankroll) -> ${bankroll-total_exposure:.2f} remaining"
        ))
        
        return results

    def check_time_stop(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """
        Time stop: exit if edge disappears or resolution near, don't hold to expiry just because
        """
        entry_price = position.get("entry_price", 0.5)
        fair_at_entry = position.get("fair_price", entry_price)
        current_fair = position.get("current_fair", fair_at_entry)
        
        # Edge disappeared?
        original_edge = abs(fair_at_entry - entry_price)
        current_edge = abs(current_fair - current_price)
        
        if current_edge < original_edge * 0.3:
            return True, f"Time stop: edge disappeared {original_edge*100:.1f}% -> {current_edge*100:.1f}% <30% original, exit"
        
        # Resolution near?
        # If end_date within 24h and position not in profit, exit to avoid gambling
        end_date = position.get("end_date")
        if end_date:
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                hours_left = (end_date - now).total_seconds() / 3600
                if hours_left < 24:
                    pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                    if pnl < 0.1:  # not in significant profit
                        return True, f"Time stop: resolution in {hours_left:.1f}h <24h and PnL {pnl*100:.1f}% <10%, exit to avoid gambling"
            except:
                pass
        
        return False, "No time stop"

    def check_liquidity_stop(self, market_id: str, current_orderbook: Dict, entry_orderbook: Dict) -> Tuple[bool, str]:
        """
        Liquidity stop: exit if order book thins and slippage exceeds edge
        """
        current_liq = current_orderbook.get("liquidity", current_orderbook.get("depth", 0))
        entry_liq = entry_orderbook.get("liquidity", entry_orderbook.get("depth", 0))
        
        if current_liq < 1000:
            return True, f"Liquidity stop: current liquidity ${current_liq} < $1000, exit"
        
        if entry_liq > 0 and current_liq < entry_liq * 0.3:
            return True, f"Liquidity stop: liquidity thinned {entry_liq} -> {current_liq} <30% entry, exit"
        
        # Check slippage
        current_spread, _ = read_spread(current_orderbook, 0.02)
        if current_spread > 0.05:
            return True, f"Liquidity stop: spread {current_spread*100:.1f}% >5%, slippage exceeds edge, exit"
        
        return False, "Liquidity OK"

    def get_report(self) -> Dict[str, Any]:
        return {
            "manager": "Correlation-Aware Risk Caps",
            "importance": "Top 5 to implement first - prevents one event from wiping you out",
            "principles": [
                "Correlation clusters: aggregate exposure to same event, cap per cluster 12% (2 positions *6%)",
                "Fractional Kelly: 1/4 Kelly not full Kelly, cap 6% bankroll but lower for low confidence <0.6 cap 3% <0.4 cap 1.5% min $1",
                "Scenario stress tests: simulate all correlated positions losing at once, if wipes account cut size",
                "Time stop: exit if edge disappears <30% original or resolution <24h and not in profit",
                "Liquidity stop: exit if order book thins <30% entry or liquidity <$1000 or spread >5%",
                "Daily/weekly loss limits: hit limit → stop trading day, hard kill switch max drawdown",
                "Avoid: <$10k liquidity, <24h to resolution, ambiguous rules, categories you don't understand"
            ],
            "kelly_example": "Market 60c fair 75% edge 15% b=0.666 Kelly raw 37% Half 18.5% Quarter 9.25% capped 6% → $3 on $50",
            "stress_test": "If 5 election markets same bet, that's one big bet, cap per cluster"
        }
