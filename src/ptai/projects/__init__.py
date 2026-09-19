"""
PTAI Projects - Bots take projects from start to end, keep context, get smarter
Premium feature: project management for AI teammates
"""
from .project import Project, ProjectTask
from .manager import ProjectManager

__all__ = ["Project", "ProjectTask", "ProjectManager"]
