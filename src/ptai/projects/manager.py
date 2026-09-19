"""
Project Manager - Manages projects that bots take from start to end
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid
from loguru import logger
from .project import Project, ProjectTask

class ProjectManager:
    def __init__(self, db_path: str = "./data/projects.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.projects: Dict[str, Project] = {}
        self.load()
    
    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    for p_dict in data:
                        tasks = [ProjectTask(**t) for t in p_dict.get("tasks", [])]
                        project = Project(
                            id=p_dict["id"],
                            name=p_dict["name"],
                            description=p_dict["description"],
                            goal=p_dict.get("goal", ""),
                            status=p_dict.get("status", "active"),
                            tasks=tasks,
                            assigned_bots=p_dict.get("assigned_bots", []),
                            context=p_dict.get("context", {}),
                            created_by=p_dict.get("created_by", "user"),
                            created_at=p_dict.get("created_at", ""),
                            completed_at=p_dict.get("completed_at"),
                            progress=p_dict.get("progress", 0.0)
                        )
                        self.projects[project.id] = project
                logger.info(f"ProjectManager loaded {len(self.projects)} projects")
            except Exception as e:
                logger.warning(f"Load projects failed: {e}")
    
    def save(self):
        try:
            data = []
            for project in self.projects.values():
                p_dict = {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "goal": project.goal,
                    "status": project.status,
                    "tasks": [{"id": t.id, "title": t.title, "description": t.description, "assigned_to": t.assigned_to, "status": t.status, "needs_approval": t.needs_approval, "created_at": t.created_at, "completed_at": t.completed_at} for t in project.tasks],
                    "assigned_bots": project.assigned_bots,
                    "context": project.context,
                    "created_by": project.created_by,
                    "created_at": project.created_at,
                    "completed_at": project.completed_at,
                    "progress": project.progress
                }
                data.append(p_dict)
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save projects failed: {e}")
    
    def create_project(self, name: str, description: str, goal: str, assigned_bots: List[str] = None) -> Project:
        project = Project(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            goal=goal,
            assigned_bots=assigned_bots or []
        )
        self.projects[project.id] = project
        self.save()
        logger.success(f"Created project {name} ID {project.id} goal: {goal} bots: {assigned_bots}")
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[Project]:
        return list(self.projects.values())
    
    def add_task_to_project(self, project_id: str, title: str, description: str, assigned_to: str = None, needs_approval: bool = False) -> Optional[ProjectTask]:
        project = self.get_project(project_id)
        if not project:
            return None
        task = project.add_task(title=title, description=description, assigned_to=assigned_to, needs_approval=needs_approval)
        self.save()
        return task
    
    def update_task_status(self, project_id: str, task_id: str, status: str, result: Dict = None):
        project = self.get_project(project_id)
        if not project:
            return False
        for task in project.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                if status == "completed":
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                project._update_progress()
                self.save()
                return True
        return False
