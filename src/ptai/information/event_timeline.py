"""
Event Timeline - builds timeline of events for market
"""
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TimelineEvent:
    timestamp: datetime
    event: str
    source: str
    impact: float  # -1 to 1, how much it moves probability
    credibility: float


@dataclass
class EventTimelineResult:
    market_id: str
    events: List[TimelineEvent] = field(default_factory=list)
    summary: str = ""


class EventTimeline:
    def __init__(self):
        pass

    def build(self, market, news: List[Dict] = None, tweets: List[Dict] = None) -> EventTimelineResult:
        """Build timeline of events"""
        events = []
        
        # Would parse news and tweets into timeline
        # For now mock
        result = EventTimelineResult(market_id=market.id)
        result.events = events
        result.summary = f"Timeline for {market.question}: {len(events)} events"
        return result
