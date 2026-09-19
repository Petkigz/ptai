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
    # FIXED: explicit venue identity immutable through pipeline
    # Previously only source enum, now venue_id str is explicit and immutable
    # venue_id must be adapter's real venue ID, never enum, never first eligible
    venue_id: str = ""  # explicit venue identity: polymarket, kalshi, manifold, etc - immutable
    venue_type: str = "prediction"  # prediction, financial, other

    def __post_init__(self):
        # Ensure venue_id is always explicit and immutable
        # If not provided, derive from source but keep as string not enum
        if not self.venue_id:
            if isinstance(self.source, str):
                self.venue_id = self.source
            elif hasattr(self.source, 'value'):
                self.venue_id = self.source.value
            else:
                self.venue_id = str(self.source)
        # Always store as string, never enum
        if hasattr(self.venue_id, 'value'):
            self.venue_id = self.venue_id.value
        self.venue_id = str(self.venue_id).lower()
        
        # Store venue_id also in raw for audit trail
        if "venue_id" not in self.raw:
            self.raw["venue_id"] = self.venue_id
        if "source" not in self.raw:
            self.raw["source"] = self.source.value if hasattr(self.source, 'value') else str(self.source)

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
