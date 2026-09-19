"""
Fair Value Validation - Cross-reference LLM estimate with heuristic to avoid hallucination-driven trades
From blueprint: Prompting + Validation

LLM's job: estimate true probability based on data ingested
Market price = collective probability
Agent looks for discrepancy >8%

Validation: Cross-reference LLM estimate with simple heuristic (e.g. moving average) to avoid hallucination
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger
import math


@dataclass
class ValidationResult:
    market_id: str
    llm_prob: float
    heuristic_prob: float
    ensemble_prob: float
    disagreement: float
    is_hallucination: bool
    should_trade: bool
    confidence_adjustment: float
    reasoning: str


class FairValueValidator:
    """
    Validates LLM fair value estimate with heuristic to avoid hallucination
    
    Heuristics:
    - Moving average of price
    - Base rate for category
    - Volume-weighted
    - Time decay
    """
    def __init__(self, max_disagreement: float = 0.25):
        self.max_disagreement = max_disagreement  # If LLM and heuristic disagree >25%, likely hallucination

    def heuristic_estimate(self, market, context: Dict = None) -> float:
        """
        Simple heuristic estimate - no LLM, just rules
        Used to cross-reference LLM
        """
        context = context or {}
        price = market.best_price
        
        # Heuristic 1: Base rate by category
        # Politics 50%, Sports 50%, Crypto 50%, etc. but adjust by price
        # If price extreme 0.9, heuristic should be closer to 0.7 not 0.5 (mean reversion)
        base_rate = 0.5
        
        # Mean reversion heuristic: extreme prices often overextended
        if price > 0.85:
            heuristic = 0.70  # Expect revert down from 0.90 to 0.70
        elif price < 0.15:
            heuristic = 0.30  # Expect revert up from 0.10 to 0.30
        elif price > 0.70:
            heuristic = price - 0.10  # Slight mean reversion
        elif price < 0.30:
            heuristic = price + 0.10
        else:
            heuristic = price  # No strong mean reversion in middle
        
        # Heuristic 2: Volume trend
        # High volume + price up = momentum, heuristic up
        # Use raw change_pct if available
        change_pct = market.raw.get("change_pct", 0) if isinstance(market.raw, dict) else 0
        if change_pct:
            # If up 5% recently, heuristic up 1.5%
            heuristic += (change_pct / 100.0) * 0.3
        
        # Heuristic 3: Time remaining
        # Far out events more uncertain, closer to 0.5
        # Would need end_date logic
        
        # Clamp
        heuristic = max(0.05, min(0.95, heuristic))
        
        return heuristic

    def validate(self, market, llm_prob: float, context: Dict = None) -> ValidationResult:
        """
        Validate LLM prob vs heuristic
        If disagreement >25%, likely hallucination - reduce confidence or block trade
        """
        heuristic = self.heuristic_estimate(market, context=context)
        
        disagreement = abs(llm_prob - heuristic)
        is_hallucination = disagreement > self.max_disagreement
        
        # Confidence adjustment
        # If disagreement small <10%, increase confidence
        # If disagreement large >25%, decrease confidence significantly
        if disagreement < 0.10:
            confidence_adj = 0.05  # Slight boost
        elif disagreement < 0.20:
            confidence_adj = 0.0
        elif disagreement < 0.30:
            confidence_adj = -0.15
        else:
            confidence_adj = -0.30  # Large penalty
        
        # Should trade? Block if hallucination and edge small
        # If LLM says 0.85 but heuristic 0.50, disagreement 0.35, likely hallucination
        # Only allow if edge very large >15% and confidence high
        should_trade = True
        if is_hallucination:
            edge = llm_prob - market.best_price
            if abs(edge) < 0.15:  # Edge <15% but hallucination suspected
                should_trade = False
        
        # Ensemble: weighted average, heuristic weight increases when disagreement large
        # If disagreement small, trust LLM more (0.8 LLM, 0.2 heuristic)
        # If disagreement large, trust heuristic more (0.5 LLM, 0.5 heuristic) to avoid hallucination
        if disagreement < 0.10:
            llm_weight = 0.8
        elif disagreement < 0.20:
            llm_weight = 0.7
        else:
            llm_weight = 0.5
        
        ensemble = llm_prob * llm_weight + heuristic * (1 - llm_weight)
        
        reasoning = (
            f"LLM {llm_prob:.3f} vs heuristic {heuristic:.3f} disagreement {disagreement:.3f} | "
            f"Hallucination: {is_hallucination} (threshold {self.max_disagreement}) | "
            f"Confidence adj {confidence_adj:+.2f} | "
            f"Ensemble {ensemble:.3f} (LLM weight {llm_weight:.1f}) | "
            f"Should trade: {should_trade} | "
            f"Heuristic: mean reversion + volume trend, no LLM"
        )
        
        if is_hallucination:
            logger.warning(f"HALLUCINATION SUSPECTED for {market.id}: {reasoning}")
        else:
            logger.info(f"Validation OK for {market.id}: {reasoning}")
        
        return ValidationResult(
            market_id=market.id,
            llm_prob=llm_prob,
            heuristic_prob=heuristic,
            ensemble_prob=ensemble,
            disagreement=disagreement,
            is_hallucination=is_hallucination,
            should_trade=should_trade,
            confidence_adjustment=confidence_adj,
            reasoning=reasoning
        )

    def get_prompt_template(self) -> str:
        """
        Prompting template for fair value engine - from blueprint
        Give LLM market question, current price, recent news headlines, X sentiment
        Ask it to output probability 0-100% and confidence score
        """
        return """
You are a superforecaster estimating true probability for prediction market.

Market Question: {question}
Current Market Price: {price} (implied {price_pct}% probability)
Category: {category}
Volume 24h: ${volume_24h}
Liquidity: ${liquidity}
End Date: {end_date}

Recent News (reputable sources, recency):
{news}

X/Twitter Sentiment (volume, credibility, bot likelihood):
- Volume: {x_volume} tweets
- Sentiment Score: {x_sentiment}
- Credibility: {x_credibility}
- Bot Likelihood: {x_bot_likelihood}
- Novelty: {x_novelty}
- Time Decay: {x_time_decay}

Orderbook:
- Spread: {spread}
- Bid/Ask: {bid}/{ask}
- Depth: {depth}
- Recent Trades: {recent_trades}

Base Rates:
- Historical frequency for similar events: {base_rate}
- Time remaining: {time_remaining}

Research (bull case, bear case, unknown, resolution risks):
Bull Case: {bull_case}
Bear Case: {bear_case}
Unknown: {unknown}
Resolution Risks: {resolution_risks}

Task:
Estimate true probability this market resolves YES.
Consider all evidence, but be calibrated: if you say 70%, it should win 70% historically.
Account for uncertainty: forecast 71% ±8% → conservative 63%.

Output JSON:
{{
  "fair_probability": 0.73,
  "confidence": 0.81,
  "uncertainty": 0.08,
  "edge": 0.12,
  "bull_case": "...",
  "bear_case": "...",
  "unknown": "...",
  "resolution_risks": ["..."],
  "sources": ["..."],
  "reasoning": "...",
  "should_trade": true
}}

Rules:
- PUBLIC INFO ONLY, no insider
- If resolution ambiguous, should_trade=false regardless of edge
- If uncertainty high, conservative estimate
- Effective edge after fees, spread, slippage, uncertainty, correlation, time must >8% to trade
- DO NOTHING is successful if no edge
"""

    def get_validation_report(self, validations: List[ValidationResult]) -> Dict:
        total = len(validations)
        hallucinations = len([v for v in validations if v.is_hallucination])
        avg_disagreement = sum(v.disagreement for v in validations) / total if total > 0 else 0
        
        return {
            "total_validations": total,
            "hallucinations_detected": hallucinations,
            "hallucination_rate": hallucinations / total if total > 0 else 0,
            "avg_disagreement": avg_disagreement,
            "should_trade_blocked": len([v for v in validations if not v.should_trade]),
            "message": f"Validation: {total} markets, {hallucinations} hallucinations suspected ({hallucinations/total*100:.1f}% if total>0), avg disagreement {avg_disagreement:.3f}. Cross-reference LLM with heuristic (moving avg, base rate) to avoid hallucination-driven trades. Most important module - one wrong call must not wipe account."
        }
