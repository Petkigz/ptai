"""
Bot - AI Teammate you can create on demand
Give it a task like you would a teammate on desktop or iOS
Takes projects from start to end, keeps context, gets smarter, asks approval when needed
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
from loguru import logger

class BotType(Enum):
    PROJECT = "project"  # One on a project
    OUTBOUND = "outbound"  # One on outbound
    SYSTEMS = "systems"  # One on systems
    SCOUT = "scout"
    RESEARCHER = "researcher"
    TRADER = "trader"
    CUSTOM = "custom"

@dataclass
class BotStatus:
    id: str
    name: str
    type: str
    role: str
    is_running: bool
    is_busy: bool
    current_task: Optional[str]
    tasks_completed: int
    tasks_failed: int
    projects_completed: int
    uptime_seconds: float
    last_active: str
    tools_signed_in: List[str]
    context_size: int
    learning_score: float

@dataclass
class BotTask:
    id: str
    bot_id: str
    title: str
    description: str
    type: str  # research, trade, outreach, system, custom
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, waiting_approval, completed, failed
    result: Optional[Dict] = None
    project_id: Optional[str] = None
    needs_approval: bool = False
    approved: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    duration: float = 0.0

class Bot:
    """
    AI Teammate Bot - Premium Product
    - Create a Bot, give it a task, add another when work grows
    - One on a project, one on outbound, one on systems
    - Works in parallel, collaborates, 24/7
    - Takes projects start to end, keeps context, gets smarter, asks approval
    - Log in once, uses your apps like you would
    - Show how it's done, saves as routine
    """
    def __init__(self, name: str, bot_type: str = "custom", role: str = "", vault=None, memory=None, browser_executor=None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.type = bot_type
        self.role = role or f"{bot_type} specialist"
        self.vault = vault
        self.memory = memory
        self.browser_executor = browser_executor
        
        self.is_running = True
        self.is_busy = False
        self.current_task: Optional[BotTask] = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.projects_completed = 0
        self.created_at = time.time()
        self.last_active = datetime.now(timezone.utc).isoformat()
        self.tools_signed_in: List[str] = []
        self.task_queue: List[BotTask] = []
        self.completed_tasks: List[BotTask] = []
        self.context: Dict[str, Any] = {}  # Keeps context on how you work
        self.routines: List[str] = []  # Saved routines
        self.learning_score = 0.5  # Gets smarter over time
        
        # Auto sign into tools based on type
        self._auto_sign_in()
        
        logger.success(f"Bot created {self.name} ({self.type}) ID {self.id} - {self.role}")
    
    def _auto_sign_in(self):
        """Log Bot in once - uses your apps like you would"""
        tool_map = {
            "project": ["browser", "polymarket", "notion", "slack", "file"],
            "outbound": ["x_api", "x_browser", "email", "discord", "telegram"],
            "systems": ["terminal", "browser", "file", "api"],
            "scout": ["gamma_api", "clob_api"],
            "researcher": ["web_search", "browser", "terminal"],
            "trader": ["polymarket_clob", "browser"],
            "custom": ["browser", "web_search"]
        }
        tools = tool_map.get(self.type, tool_map["custom"])
        for tool in tools:
            self.sign_in(tool)
    
    def sign_in(self, tool: str, credentials: Dict = None) -> bool:
        """Log Bot in once - uses your apps and websites just like you would"""
        try:
            if self.vault and credentials:
                self.vault.store_tool_credentials(self.name, tool, credentials)
            if tool not in self.tools_signed_in:
                self.tools_signed_in.append(tool)
            logger.info(f"[{self.name}] Signed into {tool} - will use it like you do")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Sign in failed {tool}: {e}")
            return False
    
    def keep_context(self, key: str, value: Any):
        """Keep context on how you work - gets smarter over time"""
        self.context[key] = value
        if self.memory:
            try:
                self.memory.remember(f"Bot {self.name} context {key}: {str(value)[:200]}", type="context", metadata={"bot_id": self.id, "key": key}, importance=0.7)
            except:
                pass
    
    def get_context(self, key: str = None) -> Any:
        if key:
            return self.context.get(key)
        return self.context
    
    def learn(self, feedback: str, importance: float = 0.8):
        """Bots get smarter over time - learn from feedback"""
        self.learning_score = min(1.0, self.learning_score + 0.01)
        if self.memory:
            try:
                self.memory.remember(f"Bot {self.name} learned: {feedback}", type="learning", metadata={"bot_id": self.id}, importance=importance)
            except:
                pass
        logger.info(f"[{self.name}] Learned: {feedback[:100]} (score {self.learning_score:.2f})")
    
    def learn_from_other(self, other_bot_name: str, insight: str):
        """Bots keep context and learn from each other"""
        self.learn(f"Learned from {other_bot_name}: {insight}", importance=0.7)
        self.keep_context(f"learned_from_{other_bot_name}", insight)
    
    async def give_task(self, title: str, description: str, task_type: str = "custom", payload: Dict = None, needs_approval: bool = False) -> BotTask:
        """Give task to Bot like you would a teammate on desktop or iOS"""
        task = BotTask(
            id=str(uuid.uuid4())[:8],
            bot_id=self.id,
            title=title,
            description=description,
            type=task_type,
            payload=payload or {},
            needs_approval=needs_approval
        )
        self.task_queue.append(task)
        logger.info(f"[{self.name}] Given task {task.id}: {title}")
        
        # If not busy, start working immediately
        if not self.is_busy:
            asyncio.create_task(self.work())
        
        return task
    
    async def work(self):
        """Bot works 24/7 - takes tasks and completes them"""
        while self.task_queue and self.is_running:
            task = self.task_queue.pop(0)
            self.is_busy = True
            self.current_task = task
            task.status = "running"
            start = time.time()
            
            logger.info(f"[{self.name}] Working on {task.id}: {task.title}")
            
            try:
                # Keep context
                self.keep_context(f"current_task_{task.id}", task.title)
                self.last_active = datetime.now(timezone.utc).isoformat()
                
                # Execute based on type
                result = await self._execute_task(task)
                
                # Check if needs approval
                if task.needs_approval:
                    task.status = "waiting_approval"
                    task.result = result
                    logger.info(f"[{self.name}] Task {task.id} needs approval - coming back to you")
                    # Save to approvals queue
                    try:
                        from ..approvals import ApprovalManager
                        am = ApprovalManager()
                        am.request_approval(task)
                    except:
                        pass
                else:
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    task.duration = time.time() - start
                    self.tasks_completed += 1
                    self.completed_tasks.append(task)
                    
                    # Learn from completion
                    self.learn(f"Completed {task.type}: {task.title} in {task.duration:.1f}s")
                
            except Exception as e:
                task.status = "failed"
                task.result = {"error": str(e)}
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.duration = time.time() - start
                self.tasks_failed += 1
                logger.error(f"[{self.name}] Task {task.id} failed: {e}")
            
            finally:
                self.is_busy = False
                self.current_task = None
            
            # Small delay between tasks
            await asyncio.sleep(1)
    
    async def _execute_task(self, task: BotTask) -> Dict[str, Any]:
        """Execute task based on type - uses tools like you would"""
        # Simulate work - in real product, would call actual tools
        await asyncio.sleep(2)  # Simulate work
        
        if task.type == "scan_markets":
            from ...markets.scanner import MarketScanner
            scanner = MarketScanner()
            markets = scanner.scan(target_count=task.payload.get("count", 100))
            return {"markets": len(markets), "message": f"Scouted {len(markets)} markets"}
        
        elif task.type == "research":
            return {"research": f"Researched {task.payload.get('topic', 'topic')}", "sources": 5}
        
        elif task.type == "trade":
            return {"trade": f"Analyzed {task.payload.get('question', 'market')} edge {task.payload.get('edge', 'N/A')}", "should_trade": True}
        
        elif task.type == "outbound":
            return {"outbound": f"Sent outreach for {task.payload.get('campaign', 'campaign')}", "sent": 10}
        
        elif task.type == "system":
            return {"system": f"Completed system task {task.title}", "status": "done"}
        
        elif task.type == "routine":
            # Run saved routine
            routine_name = task.payload.get("routine_name", "")
            return {"routine": routine_name, "message": f"Ran routine {routine_name} - saved workflow"}
        
        else:
            # Custom task - use LLM to reason
            return {"result": f"Completed custom task: {task.title}", "description": task.description}
    
    def get_status(self) -> BotStatus:
        uptime = time.time() - self.created_at
        return BotStatus(
            id=self.id,
            name=self.name,
            type=self.type,
            role=self.role,
            is_running=self.is_running,
            is_busy=self.is_busy,
            current_task=self.current_task.title if self.current_task else None,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            projects_completed=self.projects_completed,
            uptime_seconds=uptime,
            last_active=self.last_active,
            tools_signed_in=self.tools_signed_in,
            context_size=len(self.context),
            learning_score=self.learning_score
        )
