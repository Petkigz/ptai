"""
Routine - Saved workflow that Bot can run on its own next time
Show a Bot how it's done once, it saves as routine
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

@dataclass
class RoutineStep:
    id: str
    action: str  # click, type, navigate, wait, extract, api_call, etc
    target: str  # selector, url, etc
    value: Optional[str] = None  # typed text, etc
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class Routine:
    id: str
    name: str
    description: str
    steps: List[RoutineStep] = field(default_factory=list)
    created_by: str = ""  # user or bot that recorded
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run: Optional[str] = None
    run_count: int = 0
    success_count: int = 0
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    
    def add_step(self, action: str, target: str, value: str = None, description: str = ""):
        step = RoutineStep(id=str(uuid.uuid4())[:8], action=action, target=target, value=value, description=description)
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [{"id": s.id, "action": s.action, "target": s.target, "value": s.value, "description": s.description, "timestamp": s.timestamp} for s in self.steps],
            "created_by": self.created_by,
            "created_at": self.created_at,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "tags": self.tags
        }
