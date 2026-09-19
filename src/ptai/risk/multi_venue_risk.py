
"""
Multi-Market Risk Rules - Expanding to multiple venues introduces new risks
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class MultiVenueRiskReport:
    total_venues: int
    total_exposure_usd: float
    exposure_per_venue: Dict[str, float]
    exposure_per_event: Dict[str, float]  # aggregate per event not per venue
    correlation_risk: str
    settlement_risk: List[str]
    capital_fragmentation: str
    should_trade: bool
    reasoning: str

class MultiVenueRiskManager:
    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll
        self.max_exposure_per_event = bankroll * 0.12  # 12% per event across venues
        self.max_exposure_per_venue = bankroll * 0.30  # 30% per venue
        self.max_total_exposure = bankroll * 0.50  # 50% total
        self.max_venues = 3  # concentrate 2-3 venues until bankroll grows

    def calculate_event_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        # Aggregate exposure per event, not per venue - Fed cuts rates on Polymarket and same on Kalshi is one bet not two
        event_exposure: Dict[str, float] = {}
        for pos in positions:
            event_key = pos.get("event_key") or pos.get("event_slug") or pos.get("question", "")[:30]
            # Normalize event key - same event across venues should have same key
            # Use question similarity grouping in production
            amount = pos.get("amount_usd", 0)
            event_exposure[event_key] = event_exposure.get(event_key, 0) + amount
        return event_exposure

    def calculate_venue_exposure(self, positions: List[Dict]) -> Dict[str, float]:
        venue_exposure: Dict[str, float] = {}
        for pos in positions:
            venue = pos.get("venue_id", "unknown")
            amount = pos.get("amount_usd", 0)
            venue_exposure[venue] = venue_exposure.get(venue, 0) + amount
        return venue_exposure

    def check_correlation_across_venues(self, positions: List[Dict]) -> str:
        event_exposure = self.calculate_event_exposure(positions)
        for event, exposure in event_exposure.items():
            if exposure > self.max_exposure_per_event:
                return f"Correlation risk: event {event} exposure ${exposure:.2f} > max ${self.max_exposure_per_event:.2f} (12% bankroll) - same event across venues is one bet not two"
        return "Correlation OK - per event exposure within 12% cap"

    def check_settlement_risk(self, arb_opportunities: List[Dict]) -> List[str]:
        risks = []
        for arb in arb_opportunities:
            venue_a = arb.get("venue_a")
            venue_b = arb.get("venue_b")
            confidence = arb.get("confidence_same_event", 0)
            if confidence < 0.8:
                risks.append(f"Settlement risk {venue_a} vs {venue_b}: confidence {confidence:.2f} <0.8, different venues may resolve same event differently due to ambiguous rules, verify resolution criteria identical before arbing")
            else:
                risks.append(f"Settlement OK {venue_a} vs {venue_b}: confidence {confidence:.2f}, resolution likely identical but still verify")
        return risks

    def check_capital_fragmentation(self, positions: List[Dict]) -> str:
        num_venues = len(set(p.get("venue_id") for p in positions))
        total_exposure = sum(p.get("amount_usd", 0) for p in positions)
        if num_venues > self.max_venues and self.bankroll < 200:
            return f"Capital fragmentation: {num_venues} venues with ${self.bankroll} bankroll means each position tiny, fixed costs gas withdrawal fees min order sizes eat larger percentage, concentrate on 2-3 venues until bankroll grows, total exposure ${total_exposure:.2f}"
        return f"Capital OK: {num_venues} venues, total exposure ${total_exposure:.2f} within {self.max_total_exposure:.2f} max, bankroll ${self.bankroll}"

    def check_regulatory(self, venues: List[str], country_code: str = "UG") -> str:
        cc = country_code.upper()
        issues = []
        if "kalshi" in venues and cc == "UG":
            issues.append("Kalshi CFTC-regulated US, UG restricted")
        if "betfair" in venues and cc == "US":
            issues.append("Betfair restricted US")
        if "whitebit" in venues and cc == "US":
            issues.append("WhiteBIT restricted US")
        if issues:
            return f"Regulatory exposure: {', '.join(issues)} - operating across Kalshi CFTC-regulated, Polymarket on-chain, sports exchanges geographic restrictions may create compliance complications depending on jurisdiction {cc}"
        return f"Regulatory OK for {cc}: {', '.join(venues)}"

    def evaluate(self, positions: List[Dict], arb_opportunities: List[Dict], venues: List[str], country_code: str = "UG") -> MultiVenueRiskReport:
        venue_exposure = self.calculate_venue_exposure(positions)
        event_exposure = self.calculate_event_exposure(positions)
        total_exposure = sum(venue_exposure.values())

        correlation_risk = self.check_correlation_across_venues(positions)
        settlement_risks = self.check_settlement_risk(arb_opportunities)
        fragmentation = self.check_capital_fragmentation(positions)
        regulatory = self.check_regulatory(venues, country_code)

        should_trade = (
            total_exposure < self.max_total_exposure and
            all(exp < self.max_exposure_per_event for exp in event_exposure.values()) and
            all(exp < self.max_exposure_per_venue for exp in venue_exposure.values())
        )

        reasoning = (
            f"Multi-venue risk: {len(venues)} venues, total exposure ${total_exposure:.2f} max ${self.max_total_exposure:.2f} | "
            f"Per venue: {venue_exposure} max per venue ${self.max_exposure_per_venue:.2f} | "
            f"Per event: {event_exposure} max per event ${self.max_exposure_per_event:.2f} aggregate per event not per venue | "
            f"Correlation: {correlation_risk} | "
            f"Fragmentation: {fragmentation} | "
            f"Regulatory: {regulatory} | "
            f"Settlement risks: {len(settlement_risks)} checked | "
            f"Should trade {should_trade} | "
            f"Principle: arbitrage and market-making beat directional on small capital, expanding venues gives more arb pairs and liquidity but multiplies integration and risk-management work, add one venue at a time prove on paper then go live tiny"
        )

        return MultiVenueRiskReport(
            total_venues=len(venues),
            total_exposure_usd=total_exposure,
            exposure_per_venue=venue_exposure,
            exposure_per_event=event_exposure,
            correlation_risk=correlation_risk,
            settlement_risk=settlement_risks,
            capital_fragmentation=fragmentation,
            should_trade=should_trade,
            reasoning=reasoning
        )

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Multi-Venue Risk Management",
            "rules": {
                "correlation": "If you hold Fed cuts rates on Polymarket and same on Kalshi, that is one bet not two, aggregate exposure per event not per venue, max 12% per event",
                "settlement": "Different venues may resolve same event differently due to ambiguous rules, always verify resolution criteria identical before arbing",
                "fragmentation": "Splitting $50 across multiple venues means each position tiny, fixed costs gas withdrawal fees min order sizes eat larger percentage, concentrate on 2-3 venues until bankroll grows",
                "regulatory": "Kalshi CFTC-regulated, Polymarket on-chain, sports exchanges geographic restrictions, operating across all three may create compliance complications depending on jurisdiction",
                "operational": "Each venue has own API auth model rate limits failure modes, start with one additional venue prove pipeline works then add next",
                "max_per_event": f"${self.max_exposure_per_event:.2f} (12% of ${self.bankroll})",
                "max_per_venue": f"${self.max_exposure_per_venue:.2f} (30%)",
                "max_total": f"${self.max_total_exposure:.2f} (50%)",
                "max_venues_small_bankroll": f"{self.max_venues} venues until bankroll >$200"
            },
            "recommended_sequence": [
                "1. Add Kalshi for cross-venue prediction market arbitrage - highest-probability lowest-risk addition",
                "2. Add CCXT as unified data layer so agent can read odds across Polymarket Kalshi and any crypto venue with one interface",
                "3. Add one crypto derivatives venue WhiteBIT or AFX DEX only for hedging or arbitrage legs not directional bets",
                "4. Add Betfair + flumine if you want to pursue sports market-making which has different liquidity cycles and can fill gaps when prediction markets quiet"
            ],
            "principle": "Arbitrage and market-making beat directional betting on small capital. Expanding venues gives more arbitrage pairs and more liquidity but multiplies integration and risk-management work. Add one venue at a time prove on paper then go live tiny"
        }
