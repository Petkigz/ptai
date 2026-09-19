"""
Trader Teammate - Executes in its own browser, signs into Polymarket, Kalshi, etc
Signs into CLOB API, browser, uses execution orchestrator
"""
from typing import Dict, Any, List
from loguru import logger
from .base import BaseTeammate, Task

class TraderTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, storage=None, browser_executor=None, execution_orchestrator=None, notifier=None):
        super().__init__(name="Trader", role="Executor - Executes in own browser, Polymarket + other sites", vault=vault, memory=memory)
        self.storage = storage
        self.browser_executor = browser_executor
        self.execution_orchestrator = execution_orchestrator
        self.notifier = notifier
        self.sign_in_to_tool("polymarket_clob")
        self.sign_in_to_tool("browser")
        self.sign_in_to_tool("kalshi")
        self.sign_in_to_tool("generic_browser")
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "execute_trades":
            sized_opportunities = task.payload.get("sized_opportunities", [])
            
            if not sized_opportunities:
                logger.info("[Trader] No opportunities to execute")
                return {"executed": [], "count": 0, "message": "No opportunities"}
            
            logger.info(f"[Trader] Executing {len(sized_opportunities)} trades")
            
            # Init browser if needed
            if not self.execution_orchestrator:
                try:
                    from ...execution import BrowserExecutor, ExecutionOrchestrator
                    from ...execution.browser import BrowserConfig
                    from ...config import get_settings
                    settings = get_settings()
                    
                    if not self.browser_executor:
                        browser_config = BrowserConfig(
                            headless=settings.browser_headless,
                            persistent_dir=settings.browser_persistent_dir,
                            stealth=settings.browser_stealth,
                            timeout=settings.browser_timeout
                        )
                        self.browser_executor = BrowserExecutor(config=browser_config)
                        await self.browser_executor.start()
                    
                    from ...markets.polymarket import PolymarketExecutor
                    api_exec = PolymarketExecutor(
                        private_key=settings.polymarket_private_key,
                        funder=settings.polymarket_funder_address
                    )
                    self.execution_orchestrator = ExecutionOrchestrator(
                        api_executor=api_exec,
                        browser_executor=self.browser_executor
                    )
                    logger.success("[Trader] Browser + API executor initialized")
                except Exception as e:
                    logger.warning(f"[Trader] Browser init failed: {e}, API only")
            
            results = []
            for opp in sized_opportunities:
                try:
                    if not self.execution_orchestrator:
                        # Mock execution if no orchestrator
                        result = {"status": "dry_run", "message": "No executor, dry run"}
                    else:
                        result = await self.execution_orchestrator.execute(opp)
                    
                    trade_id = None
                    if self.storage:
                        try:
                            trade_id = self.storage.log_trade({
                                "market_id": opp["market_id"],
                                "market_question": opp["question"],
                                "event_slug": opp["event_slug"],
                                "outcome": opp["side"],
                                "side": opp["side"],
                                "market_price": opp["market_price"],
                                "fair_value": opp["fair_value"],
                                "edge": opp["edge"],
                                "kelly_fraction": opp["kelly_result"].kelly_fraction_adj,
                                "position_size_usd": opp["position_size_usd"],
                                "position_size_pct": opp["position_size_pct"],
                                "confidence": opp["confidence"],
                                "status": "executed" if result.get("status") in ["executed_api", "executed_browser"] else result.get("status"),
                                "notes": f"{opp['reasoning'][:200]} | LLM:{opp.get('llm_provider','?')} | {result}"
                            })
                        except Exception as e:
                            logger.debug(f"Log trade failed: {e}")
                    
                    if self.notifier:
                        try:
                            self.notifier.trade_executed(opp["question"], opp["side"], opp["position_size_usd"], result.get("status", "unknown"))
                        except:
                            pass
                    
                    results.append({"opportunity": opp, "execution": result, "trade_id": trade_id})
                    logger.success(f"[Trader] Executed {opp['question'][:50]} -> {result.get('status')}")
                
                except Exception as e:
                    logger.error(f"[Trader] Execution failed {opp['question'][:50]}: {e}")
                    results.append({"opportunity": opp, "execution": {"status": "failed", "error": str(e)}})
            
            executed_count = len([r for r in results if "executed" in r.get("execution", {}).get("status", "")])
            
            return {
                "executed": results,
                "count": len(results),
                "executed_count": executed_count,
                "message": f"Executed {executed_count}/{len(results)} trades"
            }
        
        else:
            raise ValueError(f"Trader doesn't know {task.type}")
