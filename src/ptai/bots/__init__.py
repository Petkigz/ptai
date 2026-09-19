"""
PTAI Bots - Create a Bot, give it a task, add another when work grows
One on a project, one on outbound, one on systems
AI teammates work in parallel, collaborate, keep working 24/7
"""
from .bot import Bot, BotStatus
from .manager import BotManager

__all__ = ["Bot", "BotStatus", "BotManager"]
