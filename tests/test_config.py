"""Tests for config"""
import os
from src.ptai.config import get_settings, Settings

class TestConfig:
    def test_get_settings(self):
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'bankroll')
        assert hasattr(settings, 'dry_run')
    
    def test_default_bankroll(self):
        settings = get_settings()
        assert settings.bankroll >= 10
    
    def test_risk_config(self):
        settings = get_settings()
        risk = settings.to_risk_config()
        assert risk.max_position_pct <= 0.2
        assert risk.min_edge_pct >= 0.01
    
    def test_llm_config(self):
        settings = get_settings()
        llm = settings.to_llm_config()
        assert llm is not None
    
    def test_bankroll_env_override(self):
        # Test env var handling
        os.environ["BANKROLL"] = "100"
        # Need to reset singleton
        import src.ptai.config as config_module
        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.bankroll == 100.0
        # Reset
        os.environ.pop("BANKROLL", None)
        config_module._settings = None
