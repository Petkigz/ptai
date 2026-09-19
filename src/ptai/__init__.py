"""
PTAI - Polymarket Trading AI
Local Autonomous Trading Agent
"""
__version__ = "1.0.0"
__author__ = "PTAI"

from .config import get_settings
from .agent.loop import TradingAgent

__all__ = ["TradingAgent", "get_settings"]
