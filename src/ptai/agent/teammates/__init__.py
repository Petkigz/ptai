"""
PTAI Teammates - AI Teammates you can give real work to
Bots can sign in to your tools, use them just like you do, come back with finished work
Premium product level
"""
from .base import BaseTeammate, Task, TeammateStatus
from .scout import ScoutTeammate
from .sentiment_analyst import SentimentAnalystTeammate
from .researcher import ResearcherTeammate
from .quant import QuantTeammate
from .risk_officer import RiskOfficerTeammate
from .trader import TraderTeammate
from .coach import CoachTeammate

__all__ = [
    "BaseTeammate", "Task", "TeammateStatus",
    "ScoutTeammate", "SentimentAnalystTeammate", "ResearcherTeammate",
    "QuantTeammate", "RiskOfficerTeammate", "TraderTeammate", "CoachTeammate"
]
