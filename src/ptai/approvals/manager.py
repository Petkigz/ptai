"""
Approval Manager - Bots come back when your approval is needed
Premium product: approval workflow for AI teammates
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from loguru import logger

@dataclass
class ApprovalRequest:
    id: str
    bot_id: str
    bot_name: str
    task_id: str
    task_title: str
    description: str
    payload: Dict[str, Any]
    result: Dict[str, Any]
    status: str = "pending"  # pending, approved, rejected
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None

class ApprovalManager:
    def __init__(self, db_path: str = "./data/approvals.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.approvals: Dict[str, ApprovalRequest] = {}
        self.load()
    
    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    for a_dict in data:
                        approval = ApprovalRequest(**a_dict)
                        self.approvals[approval.id] = approval
                logger.info(f"ApprovalManager loaded {len(self.approvals)} approvals")
            except Exception as e:
                logger.warning(f"Load approvals failed: {e}")
    
    def save(self):
        try:
            data = [vars(a) for a in self.approvals.values()]
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save approvals failed: {e}")
    
    def request_approval(self, task) -> ApprovalRequest:
        """Bot requests approval - comes back when your approval is needed"""
        # task can be BotTask or ProjectTask
        bot_id = getattr(task, 'bot_id', getattr(task, 'assigned_to', 'unknown'))
        bot_name = getattr(task, 'bot_name', bot_id)
        task_id = getattr(task, 'id', str(uuid.uuid4())[:8])
        task_title = getattr(task, 'title', getattr(task, 'description', 'Task')[:50])
        
        approval = ApprovalRequest(
            id=str(uuid.uuid4())[:8],
            bot_id=bot_id,
            bot_name=bot_name,
            task_id=task_id,
            task_title=task_title,
            description=getattr(task, 'description', ''),
            payload=getattr(task, 'payload', {}),
            result=getattr(task, 'result', {}) or {}
        )
        self.approvals[approval.id] = approval
        self.save()
        logger.info(f"Approval requested {approval.id} by bot {bot_name} for task {task_title} - waiting for user approval")
        return approval
    
    def approve(self, approval_id: str) -> bool:
        if approval_id in self.approvals:
            approval = self.approvals[approval_id]
            approval.status = "approved"
            approval.resolved_at = datetime.now(timezone.utc).isoformat()
            self.save()
            logger.success(f"Approved {approval_id} - bot {approval.bot_name} can continue")
            return True
        return False
    
    def reject(self, approval_id: str) -> bool:
        if approval_id in self.approvals:
            approval = self.approvals[approval_id]
            approval.status = "rejected"
            approval.resolved_at = datetime.now(timezone.utc).isoformat()
            self.save()
            logger.info(f"Rejected {approval_id}")
            return True
        return False
    
    def list_pending(self) -> List[ApprovalRequest]:
        return [a for a in self.approvals.values() if a.status == "pending"]
    
    def list_all(self) -> List[ApprovalRequest]:
        return list(self.approvals.values())
