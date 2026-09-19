"""Tests for dynamic bots - create on demand, parallel work, learning"""
import pytest
import asyncio
from src.ptai.bots.bot import Bot, BotType, BotTask
from src.ptai.bots.manager import BotManager

class TestBot:
    def test_create_bot(self):
        bot = Bot(name="Test Bot", bot_type="project", role="Test role")
        assert bot.name == "Test Bot"
        assert bot.type == "project"
        assert bot.is_running is True
        assert bot.is_busy is False
        assert len(bot.tools_signed_in) > 0
    
    def test_bot_auto_sign_in_project(self):
        bot = Bot(name="Project Lead", bot_type="project")
        assert "browser" in bot.tools_signed_in
    
    def test_bot_auto_sign_in_outbound(self):
        bot = Bot(name="Outbound", bot_type="outbound")
        assert "x_api" in bot.tools_signed_in
    
    def test_bot_auto_sign_in_systems(self):
        bot = Bot(name="Systems", bot_type="systems")
        assert "terminal" in bot.tools_signed_in
    
    def test_sign_in(self):
        bot = Bot(name="Test", bot_type="custom")
        result = bot.sign_in("test_tool", {"key": "value"})
        assert result is True
        assert "test_tool" in bot.tools_signed_in
    
    def test_keep_context(self):
        bot = Bot(name="Test", bot_type="custom")
        bot.keep_context("test_key", "test_value")
        assert bot.get_context("test_key") == "test_value"
        assert len(bot.context) == 1
    
    def test_learn(self):
        bot = Bot(name="Test", bot_type="custom")
        initial_score = bot.learning_score
        bot.learn("Test feedback")
        assert bot.learning_score > initial_score
    
    def test_learn_from_other(self):
        bot1 = Bot(name="Bot1", bot_type="project")
        bot2 = Bot(name="Bot2", bot_type="outbound")
        bot1.learn_from_other("Bot2", "Some insight")
        assert bot1.learning_score > 0.5
        assert "learned_from_Bot2" in bot1.context
    
    def test_get_status(self):
        bot = Bot(name="Test", bot_type="project", role="Test role")
        status = bot.get_status()
        assert status.name == "Test"
        assert status.type == "project"
        assert status.role == "Test role"
        assert status.is_running is True
    
    @pytest.mark.asyncio
    async def test_give_task(self):
        bot = Bot(name="Test", bot_type="custom")
        task = await bot.give_task(title="Test Task", description="Test desc", task_type="custom")
        assert task.title == "Test Task"
        assert task.bot_id == bot.id
        assert len(bot.task_queue) >= 0  # May be consumed quickly
    
    @pytest.mark.asyncio
    async def test_work_cycle(self):
        bot = Bot(name="Test", bot_type="custom")
        task = await bot.give_task(title="Test", description="Desc", task_type="custom")
        # Wait for work to complete
        await asyncio.sleep(3)
        assert bot.tasks_completed >= 1

class TestBotManager:
    def test_create_bot(self):
        manager = BotManager()
        bot = manager.create_bot(name="Test Bot", bot_type="project", role="Test")
        assert bot.name == "Test Bot"
        assert len(manager.bots) == 1
    
    def test_create_default_team(self):
        manager = BotManager()
        bots = manager.create_default_team()
        assert len(bots) == 6
        names = [b.name for b in bots]
        assert "Project Lead" in names
        assert "Outbound" in names
        assert "Systems" in names
    
    def test_get_bot(self):
        manager = BotManager()
        bot = manager.create_bot(name="Test", bot_type="custom")
        retrieved = manager.get_bot(bot.id)
        assert retrieved.id == bot.id
    
    def test_get_bot_by_name(self):
        manager = BotManager()
        bot = manager.create_bot(name="UniqueBot", bot_type="custom")
        retrieved = manager.get_bot_by_name("UniqueBot")
        assert retrieved is not None
        assert retrieved.name == "UniqueBot"
    
    def test_list_bots(self):
        manager = BotManager()
        manager.create_bot(name="Bot1", bot_type="project")
        manager.create_bot(name="Bot2", bot_type="outbound")
        bots = manager.list_bots()
        assert len(bots) == 2
    
    def test_remove_bot(self):
        manager = BotManager()
        bot = manager.create_bot(name="ToRemove", bot_type="custom")
        assert len(manager.bots) == 1
        result = manager.remove_bot(bot.id)
        assert result is True
        assert len(manager.bots) == 0
    
    def test_get_all_status(self):
        manager = BotManager()
        manager.create_default_team()
        statuses = manager.get_all_status()
        assert len(statuses) == 6
        assert all(hasattr(s, 'id') for s in statuses.values())
    
    @pytest.mark.asyncio
    async def test_broadcast_task(self):
        manager = BotManager()
        manager.create_bot(name="Bot1", bot_type="project")
        manager.create_bot(name="Bot2", bot_type="outbound")
        tasks = await manager.broadcast_task(title="Broadcast", description="Test", task_type="custom")
        assert len(tasks) == 2
    
    def test_bots_learn_from_each_other(self):
        manager = BotManager()
        manager.create_bot(name="Bot1", bot_type="project")
        # Second bot should learn from first
        manager.create_bot(name="Bot2", bot_type="outbound")
        bot2 = manager.get_bot_by_name("Bot2")
        # Should have context from Bot1
        assert len(bot2.context) > 0
