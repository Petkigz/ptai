"""
Bot Manager - Create a Bot, give it a task, add another when work grows
One on a project, one on outbound, one on systems
AI teammates work in parallel, collaborate where it makes sense, keep working 24/7
"""
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
from .bot import Bot, BotType

class BotManager:
    """
    Manages dynamic AI Teammates - premium product
    - Create bots on demand: project, outbound, systems
    - Bots work in parallel, collaborate, 24/7
    - Give tasks like teammate on desktop/iOS
    - Projects from start to end, keep context, get smarter, approval needed
    """
    def __init__(self, vault=None, memory=None):
        self.vault = vault
        self.memory = memory
        self.bots: Dict[str, Bot] = {}
        logger.info("BotManager initialized - ready to create bots on demand")
    
    def create_bot(self, name: str, bot_type: str = "custom", role: str = "") -> Bot:
        """Create a Bot, give it a task, add another when work grows"""
        bot = Bot(name=name, bot_type=bot_type, role=role, vault=self.vault, memory=self.memory)
        self.bots[bot.id] = bot
        
        # Bots learn from each other - share context
        for other_id, other_bot in self.bots.items():
            if other_id != bot.id:
                # Share recent insights
                try:
                    bot.learn_from_other(other_bot.name, f"Bot {other_bot.name} exists, type {other_bot.type}, {other_bot.tasks_completed} tasks done")
                    other_bot.learn_from_other(bot.name, f"New teammate {bot.name} joined, type {bot_type}")
                except:
                    pass
        
        logger.success(f"Created bot {name} ({bot_type}) ID {bot.id} - total {len(self.bots)} bots")
        
        if self.memory:
            try:
                self.memory.remember(f"Created bot {name} type {bot_type} role {role} - team now {len(self.bots)} bots", type="bot_creation", importance=0.7)
            except:
                pass
        
        return bot
    
    def get_bot(self, bot_id: str) -> Optional[Bot]:
        return self.bots.get(bot_id)
    
    def get_bot_by_name(self, name: str) -> Optional[Bot]:
        for bot in self.bots.values():
            if bot.name.lower() == name.lower():
                return bot
        return None
    
    def list_bots(self) -> List[Bot]:
        return list(self.bots.values())
    
    def remove_bot(self, bot_id: str) -> bool:
        if bot_id in self.bots:
            bot = self.bots[bot_id]
            bot.is_running = False
            del self.bots[bot_id]
            logger.info(f"Removed bot {bot_id} - {len(self.bots)} remaining")
            return True
        return False
    
    async def give_task_to_bot(self, bot_id: str, title: str, description: str, task_type: str = "custom", payload: Dict = None, needs_approval: bool = False):
        """Give task to Bot like you would a teammate"""
        bot = self.get_bot(bot_id)
        if not bot:
            raise ValueError(f"Bot {bot_id} not found")
        return await bot.give_task(title=title, description=description, task_type=task_type, payload=payload, needs_approval=needs_approval)
    
    async def broadcast_task(self, title: str, description: str, task_type: str = "custom", payload: Dict = None):
        """Give same task to all bots - they collaborate where it makes sense"""
        results = []
        for bot in self.bots.values():
            task = await bot.give_task(title=title, description=description, task_type=task_type, payload=payload)
            results.append(task)
        logger.info(f"Broadcast task '{title}' to {len(results)} bots - they work in parallel and collaborate")
        return results
    
    def get_all_status(self) -> Dict[str, Any]:
        status = {}
        for bot_id, bot in self.bots.items():
            status[bot_id] = bot.get_status()
        return status
    
    async def run_parallel(self):
        """Bots work in parallel, 24/7"""
        # Start work for all bots that have queued tasks
        tasks = []
        for bot in self.bots.values():
            if bot.task_queue and not bot.is_busy:
                tasks.append(bot.work())
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def create_default_team(self):
        """Create default team: one on project, one on outbound, one on systems"""
        project_bot = self.create_bot(name="Project Lead", bot_type="project", role="Takes projects from start to end, keeps context, gets smarter")
        outbound_bot = self.create_bot(name="Outbound", bot_type="outbound", role="Handles outbound, X/Twitter, Discord, Telegram outreach")
        systems_bot = self.create_bot(name="Systems", bot_type="systems", role="Manages systems, browser, terminal, files, automation")
        
        # Also add trading specialized bots
        scout = self.create_bot(name="Scout", bot_type="scout", role="Scans 500-1000 markets every 10 min")
        researcher = self.create_bot(name="Researcher", bot_type="researcher", role="Deep research via web, browser, terminal")
        trader = self.create_bot(name="Trader", bot_type="trader", role="Executes trades in own browser")
        
        logger.success(f"Default team created: Project, Outbound, Systems, Scout, Researcher, Trader - {len(self.bots)} bots working in parallel 24/7")
        return [project_bot, outbound_bot, systems_bot, scout, researcher, trader]
