"""Tests for projects - bots take projects start to end"""
import pytest
from pathlib import Path
import tempfile
import json
from src.ptai.projects.project import Project, ProjectTask
from src.ptai.projects.manager import ProjectManager

class TestProjectTask:
    def test_create_task(self):
        task = ProjectTask(id="test123", title="Research", description="Research task")
        assert task.title == "Research"
        assert task.status == "pending"
        assert task.needs_approval is False
    
    def test_task_with_approval(self):
        task = ProjectTask(id="test", title="Execute", description="Exec", needs_approval=True)
        assert task.needs_approval is True

class TestProject:
    def test_create_project(self):
        project = Project(id="proj1", name="Test Project", description="Desc", goal="Earn $500")
        assert project.name == "Test Project"
        assert project.goal == "Earn $500"
        assert project.status == "active"
        assert project.progress == 0.0
    
    def test_add_task(self):
        project = Project(id="proj1", name="Test", description="Desc", goal="Goal")
        task = project.add_task(title="Research", description="Research desc")
        assert len(project.tasks) == 1
        assert task.title == "Research"
    
    def test_update_progress(self):
        project = Project(id="proj1", name="Test", description="Desc", goal="Goal")
        project.add_task(title="Task1", description="Desc1")
        project.add_task(title="Task2", description="Desc2")
        assert project.progress == 0.0
        
        project.tasks[0].status = "completed"
        project._update_progress()
        assert project.progress == 0.5
        
        project.tasks[1].status = "completed"
        project._update_progress()
        assert project.progress == 1.0
        assert project.status == "completed"
    
    def test_to_dict(self):
        project = Project(id="proj1", name="Test", description="Desc", goal="Goal")
        project.add_task(title="Task1", description="Desc")
        d = project.to_dict()
        assert d["id"] == "proj1"
        assert d["name"] == "Test"
        assert d["tasks"] == 1

class TestProjectManager:
    def test_create_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "projects.json"
            pm = ProjectManager(db_path=str(db_path))
            project = pm.create_project(name="Test Project", description="Desc", goal="Goal", assigned_bots=["Bot1"])
            assert project.name == "Test Project"
            assert len(pm.projects) == 1
            assert db_path.exists()
    
    def test_get_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(db_path=str(Path(tmpdir) / "projects.json"))
            project = pm.create_project(name="Test", description="Desc", goal="Goal")
            retrieved = pm.get_project(project.id)
            assert retrieved.id == project.id
    
    def test_list_projects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(db_path=str(Path(tmpdir) / "projects.json"))
            pm.create_project(name="Proj1", description="Desc", goal="Goal1")
            pm.create_project(name="Proj2", description="Desc", goal="Goal2")
            projects = pm.list_projects()
            assert len(projects) == 2
    
    def test_add_task_to_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(db_path=str(Path(tmpdir) / "projects.json"))
            project = pm.create_project(name="Test", description="Desc", goal="Goal")
            task = pm.add_task_to_project(project.id, "Research", "Research desc", assigned_to="Bot1")
            assert task is not None
            assert task.title == "Research"
            assert len(project.tasks) == 1
    
    def test_update_task_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(db_path=str(Path(tmpdir) / "projects.json"))
            project = pm.create_project(name="Test", description="Desc", goal="Goal")
            task = pm.add_task_to_project(project.id, "Task1", "Desc")
            result = pm.update_task_status(project.id, task.id, "completed")
            assert result is True
            assert task.status == "completed"
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "projects.json"
            pm1 = ProjectManager(db_path=str(db_path))
            pm1.create_project(name="Persistent", description="Desc", goal="Goal")
            
            # New manager should load same data
            pm2 = ProjectManager(db_path=str(db_path))
            assert len(pm2.projects) == 1
            assert list(pm2.projects.values())[0].name == "Persistent"
    
    def test_project_with_bots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(db_path=str(Path(tmpdir) / "projects.json"))
            project = pm.create_project(
                name="Q1 Trading",
                description="Earn $500",
                goal="Find 10 mispriced",
                assigned_bots=["Project Lead", "Outbound", "Systems"]
            )
            assert len(project.assigned_bots) == 3
            assert "Project Lead" in project.assigned_bots
