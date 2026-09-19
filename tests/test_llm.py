"""Tests for LLM provider"""
from src.ptai.llm.provider import LLMRouter, LMStudioProvider, OllamaProvider
from unittest.mock import Mock, patch

class TestLLMProviders:
    def test_lm_studio_init(self):
        provider = LMStudioProvider(host="http://localhost:1234", model="local-model")
        assert provider.host == "http://localhost:1234"
    
    def test_ollama_init(self):
        provider = OllamaProvider(host="http://localhost:11434", model="llama3.1:8b")
        assert provider.host == "http://localhost:11434"
    
    def test_router_init(self):
        router = LLMRouter(preferred="auto", ollama_host="http://localhost:11434", lm_studio_host="http://localhost:1234", model="local-model")
        assert router is not None
    
    def test_router_fallback(self):
        # When no LLM available, should fallback to heuristic
        router = LLMRouter(preferred="auto", ollama_host="http://localhost:11434", lm_studio_host="http://localhost:1234", model="local-model")
        # Should not crash even if no LLM
        assert router.get_provider_name() != ""
    
    def test_lm_studio_not_available(self):
        provider = LMStudioProvider(host="http://localhost:5999", model="test")
        # Should return False when not running
        assert provider.is_available() is False

class TestBrain:
    def test_brain_init(self):
        from src.ptai.agent.brain import Brain
        brain = Brain()
        assert brain is not None
    
    def test_brain_fair_value_heuristic(self):
        from src.ptai.agent.brain import Brain
        from src.ptai.markets.base import Market, MarketSource
        brain = Brain()
        market = Market(
            id="test1",
            source=MarketSource.POLYMARKET,
            question="Will BTC go up?",
            outcome_prices=[0.6, 0.4],
            outcomes=["YES", "NO"]
        )
        # Test heuristic fallback when no LLM - actual method is estimate_fair_value
        result = brain.estimate_fair_value(market=market, sentiment=None)
        assert result is not None
        assert hasattr(result, 'fair_value') or isinstance(result, dict)
        # Fair value should be between 0 and 1
        fv = result.fair_value if hasattr(result, 'fair_value') else result.get('fair_value', 0.5)
        assert 0 <= fv <= 1
