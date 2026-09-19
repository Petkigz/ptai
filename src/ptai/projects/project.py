"""
Project - Bot takes project from start to end, keeps context, gets smarter, asks approval when needed
Premium product: project management
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

@dataclass
class ProjectTask:
    id: str
    title: str
    description: str
    assigned_to: Optional[str] = None  # bot id
    status: str = "pending"  # pending, running, waiting_approval, completed, failed
    result: Optional[Dict] = None
    needs_approval: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

@dataclass
class Project:
    id: str
    name: str
    description: str
    goal: str
    status: str = "active"  # active, completed, paused, waiting_approval
    tasks: List[ProjectTask] = field(default_factory=list)
    assigned_bots: List[str] = field(default_factory=list)  # bot ids
    context: Dict[str, Any] = field(default_factory=dict)  # Keeps context on how you work
    created_by: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    progress: float = 0.0  # 0-1
    
    def add_task(self, title: str, description: str, assigned_to: str = None, needs_approval: bool = False) -> ProjectTask:
        task = ProjectTask(id=str(uuid.uuid4())[:8], title=title, description=description, assigned_to=assigned_to, needs_approval=needs_approval)
        self.tasks.append(task)
        self._update_progress()
        return task
    
    def _update_progress(self):
        if not self.tasks:
            self.progress = 0.0
            return
        completed = len([t for t in self.tasks if t.status == "completed"])
        self.progress = completed / len(self.tasks)
        if self.progress == 1.0:
            self.status = "completed"
            self.completed_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "status": self.status,
            "progress": self.progress,
            "tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks if t.status == "completed"]),
            "assigned_bots": self.assigned_bots,
            "created_at": self.created_at
        }
