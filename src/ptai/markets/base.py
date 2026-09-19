"""
Base market models - provider agnostic
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MarketSource(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    PREDICTIT = "predictit"

@dataclass
class Token:
    token_id: str
    outcome: str  # YES / NO or specific outcome
    price: float  # 0-1

@dataclass
class Market:
    id: str
    source: MarketSource
    question: str
    description: str = ""
    outcomes: List[str] = field(default_factory=list)
    outcome_prices: List[float] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    slug: str = ""
    event_slug: str = ""
    condition_id: str = ""
    market_type: str = "binary"  # binary, categorical
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_price(self) -> float:
        """Get YES price if binary"""
        if self.outcome_prices:
            return self.outcome_prices[0]
        if self.tokens:
            # Find YES token
            for t in self.tokens:
                if t.outcome.upper() == "YES":
                    return t.price
            return self.tokens[0].price
        return 0.5

    @property
    def yes_token_id(self) -> Optional[str]:
        for t in self.tokens:
            if t.outcome.upper() == "YES":
                return t.token_id
        return self.tokens[0].token_id if self.tokens else None

    @property
    def no_token_id(self) -> Optional[str]:
        for t in self.tokens:
            if t.outcome.upper() == "NO":
                return t.token_id
        return self.tokens[1].token_id if len(self.tokens) > 1 else None

    @property
    def yes_price(self) -> float:
        for t in self.tokens:
            if t.outcome.upper() == "YES":
                return t.price
        return self.best_price

    @property
    def no_price(self) -> float:
        for t in self.tokens:
            if t.outcome.upper() == "NO":
                return t.price
        return 1 - self.best_price

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source.value,
            "question": self.question,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume": self.volume,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "slug": self.slug,
            "event_slug": self.event_slug,
            "active": self.active
        }
