"""Tests for approvals - bots come back when approval needed"""
import pytest
from pathlib import Path
import tempfile
from src.ptai.approvals.manager import ApprovalManager, ApprovalRequest
from src.ptai.bots.bot import BotTask

class TestApprovalRequest:
    def test_create_request(self):
        req = ApprovalRequest(
            id="appr1",
            bot_id="bot1",
            bot_name="Project Lead",
            task_id="task1",
            task_title="Execute Trade",
            description="Needs approval",
            payload={},
            result={"trade": "data"}
        )
        assert req.bot_name == "Project Lead"
        assert req.status == "pending"

class TestApprovalManager:
    def test_request_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            task = BotTask(
                id="task1",
                bot_id="bot1",
                title="Execute Trade",
                description="Trade BTC",
                type="trade",
                needs_approval=True,
                result={"edge": 0.15}
            )
            task.bot_name = "Trader"
            
            approval = ApprovalRequest(
                id="appr1",
                bot_id="bot1",
                bot_name="Trader",
                task_id="task1",
                task_title="Execute Trade",
                description="Trade BTC",
                payload={},
                result={"edge": 0.15}
            )
            am.approvals[approval.id] = approval
            am.save()
            
            assert len(am.approvals) == 1
    
    def test_approve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            approval = ApprovalRequest(
                id="appr1",
                bot_id="bot1",
                bot_name="Trader",
                task_id="task1",
                task_title="Test",
                description="Desc",
                payload={},
                result={}
            )
            am.approvals[approval.id] = approval
            am.save()
            
            result = am.approve("appr1")
            assert result is True
            assert am.approvals["appr1"].status == "approved"
    
    def test_reject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            approval = ApprovalRequest(
                id="appr1",
                bot_id="bot1",
                bot_name="Trader",
                task_id="task1",
                task_title="Test",
                description="Desc",
                payload={},
                result={}
            )
            am.approvals[approval.id] = approval
            
            result = am.reject("appr1")
            assert result is True
            assert am.approvals["appr1"].status == "rejected"
    
    def test_list_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            am.approvals["appr1"] = ApprovalRequest(
                id="appr1", bot_id="b1", bot_name="Bot1", task_id="t1",
                task_title="Task1", description="Desc", payload={}, result={}, status="pending"
            )
            am.approvals["appr2"] = ApprovalRequest(
                id="appr2", bot_id="b2", bot_name="Bot2", task_id="t2",
                task_title="Task2", description="Desc", payload={}, result={}, status="approved"
            )
            
            pending = am.list_pending()
            assert len(pending) == 1
            assert pending[0].id == "appr1"
    
    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            am.approvals["appr1"] = ApprovalRequest(
                id="appr1", bot_id="b1", bot_name="Bot1", task_id="t1",
                task_title="Task1", description="Desc", payload={}, result={}
            )
            am.approvals["appr2"] = ApprovalRequest(
                id="appr2", bot_id="b2", bot_name="Bot2", task_id="t2",
                task_title="Task2", description="Desc", payload={}, result={}
            )
            
            all_appr = am.list_all()
            assert len(all_appr) == 2
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "approvals.json"
            am1 = ApprovalManager(db_path=str(db_path))
            am1.approvals["appr1"] = ApprovalRequest(
                id="appr1", bot_id="b1", bot_name="Bot1", task_id="t1",
                task_title="Task1", description="Desc", payload={}, result={"edge": 0.1}
            )
            am1.save()
            
            am2 = ApprovalManager(db_path=str(db_path))
            assert len(am2.approvals) == 1
            assert am2.approvals["appr1"].task_title == "Task1"
    
    def test_approve_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            result = am.approve("nonexistent")
            assert result is False
    
    def test_full_flow_with_bot_task(self):
        """Test full flow: bot task needs approval -> approval manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ApprovalManager(db_path=str(Path(tmpdir) / "approvals.json"))
            
            # Simulate bot task that needs approval
            task = BotTask(
                id="task123",
                bot_id="bot_trader",
                title="Execute BTC Trade",
                description="Buy YES at 60c, fair 75%",
                type="trade",
                needs_approval=True,
                result={"market": "BTC up", "edge": 0.15, "size": 3.0}
            )
            task.bot_name = "Trader"
            
            # Request approval
            approval = am.request_approval(task)
            assert approval.task_title == "Execute BTC Trade"
            assert len(am.list_pending()) == 1
            
            # Approve
            am.approve(approval.id)
            assert len(am.list_pending()) == 0
            assert am.approvals[approval.id].status == "approved"
