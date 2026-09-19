"""Tests for AI teammates"""
import tempfile
from pathlib import Path
from src.ptai.agent.teammates.scout import ScoutTeammate
from src.ptai.agent.teammates.researcher import ResearcherTeammate
from src.ptai.agent.teammates.quant import QuantTeammate
from src.ptai.agent.teammates.risk_officer import RiskOfficerTeammate
from src.ptai.agent.teammates.trader import TraderTeammate
from src.ptai.agent.teammates.coach import CoachTeammate
from src.ptai.agent.teammates.coordinator import TeamCoordinator
from src.ptai.vault import Vault
from src.ptai.memory import Memory
from src.ptai.storage.db import Storage
from src.ptai.agent.brain import Brain
from src.ptai.risk import KellyCalculator, RiskManager
from src.ptai.execution.monitor import PositionMonitor
from src.ptai.agent.notifier import Notifier
from src.ptai.config import get_settings

class TestTeammates:
    def test_scout_init(self):
        scout = ScoutTeammate()
        assert scout.name == "Scout"
        assert scout.role != ""
    
    def test_researcher_init(self):
        researcher = ResearcherTeammate()
        assert researcher.name == "Researcher"
    
    def test_quant_init(self):
        quant = QuantTeammate()
        assert quant.name == "Quant"
    
    def test_risk_officer_init(self):
        officer = RiskOfficerTeammate()
        assert "Risk" in officer.name or "Risk" in officer.role or officer.name == "Risk Officer" or "RiskOfficer" in officer.name
    
    def test_trader_init(self):
        trader = TraderTeammate()
        assert trader.name == "Trader"
    
    def test_coach_init(self):
        coach = CoachTeammate()
        assert coach.name == "Coach"
    
    def test_coordinator_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = get_settings()
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            brain = Brain()
            kelly = KellyCalculator()
            risk_manager = RiskManager(storage=storage, kelly_calculator=kelly)
            monitor = PositionMonitor(storage=storage)
            notifier = Notifier(enabled=False)
            
            coordinator = TeamCoordinator(
                vault=vault, memory=memory, storage=storage,
                brain=brain, kelly=kelly, risk_manager=risk_manager,
                monitor=monitor, notifier=notifier
            )
            assert len(coordinator.teammates) == 7
            assert "scout" in coordinator.teammates
            
            storage.close()
            memory.close()
    
    def test_team_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = get_settings()
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            vault = Vault(vault_path=str(Path(tmpdir) / "vault.json"))
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            brain = Brain()
            kelly = KellyCalculator()
            risk_manager = RiskManager(storage=storage, kelly_calculator=kelly)
            monitor = PositionMonitor(storage=storage)
            notifier = Notifier(enabled=False)
            
            coordinator = TeamCoordinator(
                vault=vault, memory=memory, storage=storage,
                brain=brain, kelly=kelly, risk_manager=risk_manager,
                monitor=monitor, notifier=notifier
            )
            status = coordinator.get_team_status()
            assert len(status) == 7
            for name, s in status.items():
                assert hasattr(s, 'name')
                assert hasattr(s, 'is_busy')
            
            storage.close()
            memory.close()
