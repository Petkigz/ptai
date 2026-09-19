"""Tests for routines - show bot how it's done, saves and runs on own"""
import pytest
from pathlib import Path
import tempfile
from src.ptai.routines.routine import Routine, RoutineStep
from src.ptai.routines.manager import RoutineManager
from src.ptai.routines.recorder import RoutineRecorder

class TestRoutineStep:
    def test_create_step(self):
        step = RoutineStep(id="step1", action="click", target="button", value=None, description="Click button")
        assert step.action == "click"
        assert step.target == "button"

class TestRoutine:
    def test_create_routine(self):
        routine = Routine(id="rout1", name="Test Routine", description="Test desc")
        assert routine.name == "Test Routine"
        assert len(routine.steps) == 0
    
    def test_add_step(self):
        routine = Routine(id="rout1", name="Test", description="Desc")
        step = routine.add_step(action="navigate", target="https://polymarket.com", description="Go to site")
        assert len(routine.steps) == 1
        assert step.action == "navigate"
        assert step.target == "https://polymarket.com"
    
    def test_to_dict(self):
        routine = Routine(id="rout1", name="Test", description="Desc")
        routine.add_step(action="click", target="button", description="Click")
        d = routine.to_dict()
        assert d["id"] == "rout1"
        assert d["name"] == "Test"
        assert len(d["steps"]) == 1
        assert "id" in d["steps"][0]

class TestRoutineManager:
    def test_save_and_load_routine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "routines.json"
            rm = RoutineManager(db_path=str(db_path))
            routine = Routine(id="test123", name="Login", description="Login routine")
            routine.add_step(action="navigate", target="https://polymarket.com", description="Go")
            routine.add_step(action="click", target="button:has-text('Login')", description="Click login")
            rm.save_routine(routine)
            
            assert db_path.exists()
            assert len(rm.routines) == 1
            
            # Load new manager
            rm2 = RoutineManager(db_path=str(db_path))
            assert len(rm2.routines) == 1
            loaded = rm2.get_routine("test123")
            assert loaded.name == "Login"
            assert len(loaded.steps) == 2
    
    def test_get_routine_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rm = RoutineManager(db_path=str(Path(tmpdir) / "routines.json"))
            routine = Routine(id="test", name="UniqueName", description="Desc")
            rm.save_routine(routine)
            found = rm.get_routine_by_name("UniqueName")
            assert found is not None
            assert found.id == "test"
    
    def test_list_routines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rm = RoutineManager(db_path=str(Path(tmpdir) / "routines.json"))
            rm.save_routine(Routine(id="r1", name="Rout1", description="Desc"))
            rm.save_routine(Routine(id="r2", name="Rout2", description="Desc"))
            routines = rm.list_routines()
            assert len(routines) == 2
    
    def test_delete_routine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rm = RoutineManager(db_path=str(Path(tmpdir) / "routines.json"))
            routine = Routine(id="todelete", name="DeleteMe", description="Desc")
            rm.save_routine(routine)
            assert len(rm.routines) == 1
            result = rm.delete_routine("todelete")
            assert result is True
            assert len(rm.routines) == 0
    
    def test_load_old_format_without_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "routines.json"
            # Write old format without step ids
            import json
            old_data = [{
                "id": "old1",
                "name": "Old Routine",
                "description": "Old desc",
                "steps": [
                    {"action": "navigate", "target": "https://example.com", "value": None, "description": "Go"},
                ],
                "created_by": "user",
                "created_at": "2024-01-01",
                "run_count": 0,
                "success_count": 0,
                "tags": []
            }]
            with open(db_path, "w") as f:
                json.dump(old_data, f)
            
            rm = RoutineManager(db_path=str(db_path))
            assert len(rm.routines) == 1
            routine = rm.get_routine("old1")
            assert len(routine.steps) == 1
            assert routine.steps[0].id is not None  # Should generate id
    
    @pytest.mark.asyncio
    async def test_run_routine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rm = RoutineManager(db_path=str(Path(tmpdir) / "routines.json"))
            routine = Routine(id="run1", name="TestRun", description="Desc")
            routine.add_step(action="navigate", target="https://polymarket.com", description="Go")
            routine.add_step(action="click", target="button", description="Click")
            rm.save_routine(routine)
            
            result = await rm.run_routine("run1")
            assert result["success"] is True
            assert result["routine"] == "TestRun"
            assert len(result["results"]) == 2
            
            # Check run count incremented
            updated = rm.get_routine("run1")
            assert updated.run_count == 1
            assert updated.success_count == 1

class TestRoutineRecorder:
    def test_start_recording(self):
        recorder = RoutineRecorder()
        routine = recorder.start_recording(name="Test Recording", description="Desc")
        assert recorder.is_recording is True
        assert routine.name == "Test Recording"
        assert recorder.current_routine.id == routine.id
    
    def test_record_step(self):
        recorder = RoutineRecorder()
        recorder.start_recording(name="Test", description="Desc")
        step = recorder.record_step(action="navigate", target="https://example.com", description="Go to example")
        assert step.action == "navigate"
        assert len(recorder.recorded_steps) == 1
    
    def test_stop_recording(self):
        recorder = RoutineRecorder()
        recorder.start_recording(name="Test", description="Desc")
        recorder.record_step(action="click", target="button", description="Click")
        recorder.record_step(action="type", target="input", value="test@example.com", description="Type email")
        
        # Mock manager to avoid writing to default path
        import unittest.mock as mock
        with mock.patch('src.ptai.routines.manager.RoutineManager') as MockManager:
            mock_instance = MockManager.return_value
            routine = recorder.stop_recording()
        
        assert routine is not None
        assert len(routine.steps) == 2
        assert recorder.is_recording is False
    
    def test_stop_without_steps(self):
        recorder = RoutineRecorder()
        recorder.start_recording(name="Empty", description="Desc")
        import unittest.mock as mock
        with mock.patch('src.ptai.routines.manager.RoutineManager'):
            routine = recorder.stop_recording()
        assert routine is None  # No steps recorded
    
    def test_stop_not_recording(self):
        recorder = RoutineRecorder()
        routine = recorder.stop_recording()
        assert routine is None
    
    def test_recording_status(self):
        recorder = RoutineRecorder()
        status = recorder.get_recording_status()
        assert status["is_recording"] is False
        
        recorder.start_recording(name="Test", description="Desc")
        status = recorder.get_recording_status()
        assert status["is_recording"] is True
        assert status["current_routine"] == "Test"
    
    def test_record_without_start(self):
        recorder = RoutineRecorder()
        step = recorder.record_step(action="click", target="button", description="Click")
        assert step is None
