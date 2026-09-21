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

class DataMode(str, Enum):
    """
    CRITICAL SAFETY V9: Hard separation of data types
    LIVE = real API data, executable for real money (if qualified)
    PAPER = real API data but paper trading mode, learning only, no live capital unless qualified
    MOCK = synthetic/fake markets for development/testing, MUST NEVER reach live execution
    """
    LIVE = "live"
    PAPER = "paper"
    MOCK = "mock"

    @property
    def is_executable(self) -> bool:
        return self in (DataMode.LIVE, DataMode.PAPER)

    @property
    def can_deploy_live_capital(self) -> bool:
        return self == DataMode.LIVE

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
    venue_id: str = ""  # explicit venue identity: polymarket, kalshi, manifold, etc - immutable
    venue_type: str = "prediction"  # prediction, financial, other
    # FIXED V9: Hard LIVE/PAPER/MOCK separation - MOCK must never reach live execution
    data_mode: DataMode = DataMode.LIVE
    is_mock: bool = False
    data_source: str = ""

    def __post_init__(self):
        # Ensure venue_id is always explicit and immutable
        if not self.venue_id:
            if isinstance(self.source, str):
                self.venue_id = self.source
            elif hasattr(self.source, 'value'):
                self.venue_id = self.source.value
            else:
                self.venue_id = str(self.source)
        if hasattr(self.venue_id, 'value'):
            self.venue_id = self.venue_id.value
        self.venue_id = str(self.venue_id).lower()

        # V9: Ensure data_mode is enum and is_mock sync
        if isinstance(self.data_mode, str):
            try:
                self.data_mode = DataMode(self.data_mode.lower())
            except:
                self.data_mode = DataMode.LIVE if not self.is_mock else DataMode.MOCK
        if self.is_mock and self.data_mode == DataMode.LIVE:
            self.data_mode = DataMode.MOCK
        if self.data_mode == DataMode.MOCK:
            self.is_mock = True
        if not self.data_source:
            self.data_source = "mock_fallback" if self.data_mode == DataMode.MOCK else f"{self.venue_id}_api"
        
        if "venue_id" not in self.raw:
            self.raw["venue_id"] = self.venue_id
        if "source" not in self.raw:
            self.raw["source"] = self.source.value if hasattr(self.source, 'value') else str(self.source)
        if "data_mode" not in self.raw:
            self.raw["data_mode"] = self.data_mode.value if hasattr(self.data_mode, 'value') else str(self.data_mode)
        if "data_source" not in self.raw:
            self.raw["data_source"] = self.data_source
        if "is_mock" not in self.raw:
            self.raw["is_mock"] = self.is_mock

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
