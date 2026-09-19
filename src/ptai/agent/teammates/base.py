"""
PTAI Teammate Base - AI Teammate that can be given real work
Each teammate can sign in to tools, use them like you do, come back with finished work
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from loguru import logger

@dataclass
class Task:
    id: str
    type: str
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

@dataclass
class TeammateStatus:
    name: str
    role: str
    is_busy: bool
    current_task: Optional[str]
    tasks_completed: int
    tasks_failed: int
    avg_duration: float
    last_active: datetime
    tools_signed_in: List[str]

class BaseTeammate(ABC):
    """
    Base for AI Teammates - each can:
    - Sign in to tools (browser, API, terminal)
    - Use tools just like you do
    - Come back with finished work
    - Learn from past work
    """
    def __init__(self, name: str, role: str, vault=None, memory=None, llm_router=None):
        self.name = name
        self.role = role
        self.vault = vault
        self.memory = memory
        self.llm_router = llm_router
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_duration = 0.0
        self.current_task: Optional[Task] = None
        self.last_active = datetime.now(timezone.utc)
        self.tools_signed_in: List[str] = []
        logger.info(f"Teammate {name} ({role}) initialized")
    
    @abstractmethod
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Execute a task assigned to this teammate"""
        pass
    
    async def assign(self, task: Task) -> Dict[str, Any]:
        """Assign work to teammate, track, execute, return finished work"""
        task.assigned_to = self.name
        task.status = "running"
        self.current_task = task
        self.last_active = datetime.now(timezone.utc)
        start = time.time()
        
        logger.info(f"[{self.name}] Assigned task {task.id}: {task.description[:100]}")
        
        try:
            result = await self.execute(task)
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            task.duration_seconds = time.time() - start
            
            self.tasks_completed += 1
            self.total_duration += task.duration_seconds
            
            # Save to memory for learning
            if self.memory:
                try:
                    self.memory.remember_task(task)
                except Exception as e:
                    logger.debug(f"Memory save failed: {e}")
            
            logger.success(f"[{self.name}] Completed {task.id} in {task.duration_seconds:.1f}s")
            return result
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            task.completed_at = datetime.now(timezone.utc)
            task.duration_seconds = time.time() - start
            self.tasks_failed += 1
            logger.error(f"[{self.name}] Failed {task.id}: {e}")
            raise
        finally:
            self.current_task = None
    
    def get_status(self) -> TeammateStatus:
        avg = self.total_duration / max(1, self.tasks_completed)
        return TeammateStatus(
            name=self.name,
            role=self.role,
            is_busy=self.current_task is not None,
            current_task=self.current_task.description if self.current_task else None,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            avg_duration=avg,
            last_active=self.last_active,
            tools_signed_in=self.tools_signed_in
        )
    
    def sign_in_to_tool(self, tool_name: str, credentials: Dict = None) -> bool:
        """Sign in to a tool like you do - browser, API, etc"""
        try:
            # If vault exists, store credentials securely
            if self.vault and credentials:
                self.vault.store_tool_credentials(self.name, tool_name, credentials)
            
            if tool_name not in self.tools_signed_in:
                self.tools_signed_in.append(tool_name)
            
            logger.info(f"[{self.name}] Signed in to {tool_name}")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to sign in to {tool_name}: {e}")
            return False
    
    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools_signed_in
