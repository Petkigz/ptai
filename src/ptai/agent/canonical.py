"""
PTAI Canonical Production Engine - SINGLE ENTRY POINT

Consolidates to one canonical production engine path as per V9 requirements.

- TradingAgentV3.run_cycle is the ONLY production engine path
- No duplicate loops in dashboard or elsewhere (dashboard only calls run_cycle via API wrapper)
- loop.py, v2_loop.py are deprecated legacy, kept for reference but not used in production
- v3_loop.py is canonical
- This module enforces single entry point and documents it

Usage:
    from ptai.agent.canonical import get_production_engine, TradingAgentV3
    engine = get_production_engine(country_code="UG")
    result = await engine.run_cycle(target_per_venue=200, max_trades=3)

Dashboard:
    /api/v3/run-cycle -> TradingAgentV3.run_cycle (canonical)
    /api/v3/opportunities -> TradingAgentV3.strategy_engine_v3.scan_all_venues (same engine)
    /api/v2/* -> deprecated, redirects to v3 in future
    /api/* -> deprecated legacy

Safety:
    - MOCK_DATA blocked at 3 layers: Market.__post_init__ syncs is_mock<->data_mode, execution_guard, v3_loop pre-execution check
    - Exact routing only: venue_id -> Registry.get_adapter_for_venue_id -> exact market -> exact orderbook, ABORT if missing, never fallback
    - DataMode enum: LIVE, PAPER, MOCK on Market and Opportunity
"""

from .v3_loop import TradingAgentV3

# Canonical engine is V3
CanonicalEngine = TradingAgentV3

def get_production_engine(country_code: str = "UG") -> TradingAgentV3:
    """
    Returns canonical production engine - single entry point
    TradingAgentV3.run_cycle is the ONLY production path
    """
    return TradingAgentV3(country_code=country_code)

# Deprecated aliases - point to canonical to avoid duplicate loops
def get_legacy_engine(country_code: str = "UG"):
    """
    Legacy loop.py and v2_loop.py are deprecated
    They now return canonical engine to ensure single path
    """
    import warnings
    warnings.warn("Legacy engine deprecated, using canonical TradingAgentV3", DeprecationWarning)
    return get_production_engine(country_code)

__all__ = ["CanonicalEngine", "TradingAgentV3", "get_production_engine", "get_legacy_engine"]
