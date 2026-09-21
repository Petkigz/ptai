"""
Mock market data for offline testing / demo
"""
from typing import List
import random
from datetime import datetime, timedelta

from .base import Market, Token, MarketSource, DataMode

MOCK_QUESTIONS = [
    "Will Bitcoin hit $100k by end of month?",
    "Will Fed cut rates in December?",
    "Will Tesla stock close above $300 tomorrow?",
    "Will Trump win the 2024 election?",
    "Will Ethereum hit $4000 this week?",
    "Will US enter recession in 2025?",
    "Will OpenAI release GPT-5 in 2024?",
    "Will SpaceX launch Starship successfully?",
    "Will inflation be below 3% next month?",
    "Will Apple announce new iPhone in September?",
    "Will Bitcoin ETF see $1B inflow this week?",
    "Will Ukraine ceasefire happen in 2024?",
    "Will AI regulation pass in US this year?",
    "Will Meta stock hit $500?",
    "Will Gold hit $2500?",
    "Will US unemployment stay below 4%?",
    "Will Google release Gemini 2.0?",
    "Will Netflix subscriber growth beat estimates?",
    "Will Oil hit $100 per barrel?",
    "Will S&P 500 hit all-time high this month?"
]

def generate_mock_markets(count: int = 100) -> List[Market]:
    markets = []
    for i in range(count):
        question = random.choice(MOCK_QUESTIONS) + f" #{i}"
        yes_price = random.uniform(0.15, 0.85)
        no_price = 1 - yes_price
        volume_24h = random.uniform(5000, 500000)
        liquidity = random.uniform(1000, 100000)
        yes_token = Token(token_id=f"mock_yes_{i}_{random.randint(100000,999999)}", outcome="YES", price=yes_price)
        no_token = Token(token_id=f"mock_no_{i}_{random.randint(100000,999999)}", outcome="NO", price=no_price)

        market = Market(
            id=f"MOCK-mock_{i}",
            source=MarketSource.POLYMARKET,
            question=question + " - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
            description=f"Mock market for testing: {question} - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
            outcomes=["YES", "NO"],
            outcome_prices=[yes_price, no_price],
            tokens=[yes_token, no_token],
            volume=volume_24h * random.uniform(1, 5),
            volume_24h=volume_24h,
            liquidity=liquidity,
            end_date=datetime.now() + timedelta(days=random.randint(1, 90)),
            active=True,
            closed=False,
            slug=f"mock-market-{i}",
            event_slug=f"mock-event-{i}",
            condition_id=f"0xmock{i}",
            market_type="binary",
            raw={"mock": True, "venue": "mock", "data_mode": "mock", "data_source": "mock_fallback", "is_mock": True, "safety": "MOCK_DATA must be impossible to reach live execution"},
            venue_id="mock",
            venue_type="prediction",
            data_mode=DataMode.MOCK,
            data_source="mock_fallback",
            is_mock=True
        )
        markets.append(market)
    return markets
