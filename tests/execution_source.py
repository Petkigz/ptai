"""
The source of PTAI's execution path, in one place, for tests that read it.

Some guards are source-level on purpose: they check that a value exists AT THE
CALL SITE, where a runtime test cannot see it (a fill's book provenance, for
one). Those guards used to read `run_cycle` alone, because the recording was
inline in it. It is now a method - `_record_execution` - called by BOTH the
single-opportunity path and the arbitrage lane, so a guard that reads only
`run_cycle` would still pass while the logic it exists to protect was deleted
from the method that now holds it.

`execution_path_source()` returns every method that can record a fill, so the
guards keep their meaning wherever the code is moved next.
"""
import inspect

from src.ptai.agent.v3_loop import TradingAgentV3


def execution_path_source() -> str:
    """Source of every method that can turn an execution into a position."""
    methods = [
        TradingAgentV3.run_cycle,
        TradingAgentV3._record_execution,
        TradingAgentV3._execute_arbitrage_lane,
        TradingAgentV3._arbitrage_blocked,
    ]
    return "\n".join(inspect.getsource(m) for m in methods)


def loop_file_source() -> str:
    """The whole module, for guards that scan the file itself."""
    return inspect.getsource(inspect.getmodule(TradingAgentV3))
