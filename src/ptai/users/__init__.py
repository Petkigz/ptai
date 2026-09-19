"""
PTAI Users - Multi-user support for product
Premium feature: each user has own bankroll, wallet, browser profiles, memory
"""
from .manager import UserManager, User

__all__ = ["UserManager", "User"]
