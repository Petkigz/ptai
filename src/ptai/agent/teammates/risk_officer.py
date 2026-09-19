"""
Risk Officer Teammate - Kelly sizing, risk checks, prevents wipeout
Signs into storage, risk manager, position monitor
"""
from typing import Dict, Any, List
from loguru import logger
from .base import BaseTeammate, Task

class RiskOfficerTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, storage=None, kelly=None, risk_manager=None, monitor=None):
        super().__init__(name="RiskOfficer", role="Risk Manager - Kelly 6% Cap, Prevents Wipeout", vault=vault, memory=memory)
        self.storage = storage
        self.kelly = kelly
        self.risk_manager = risk_manager
        self.monitor = monitor
        self.sign_in_to_tool("risk_manager")
        self.sign_in_to_tool("kelly_calculator")
        self.sign_in_to_tool("position_monitor")
        
        if not kelly or not risk_manager:
            try:
                from ...risk import KellyCalculator, RiskManager
                from ...config import get_settings
                settings = get_settings()
                if not kelly:
                    self.kelly = KellyCalculator(kelly_fraction=settings.kelly_fraction, max_pct=settings.max_position_pct, min_edge=settings.min_edge_pct)
                if not risk_manager and storage:
                    self.risk_manager = RiskManager(storage=storage, kelly_calculator=self.kelly)
            except Exception as e:
                logger.warning(f"RiskOfficer init failed: {e}")
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "size_positions":
            opportunities = task.payload.get("opportunities", [])
            bankroll = task.payload.get("bankroll", 50.0)
            if self.storage:
                bankroll = self.storage.get_bankroll()
            
            logger.info(f"[RiskOfficer] Sizing {len(opportunities)} opportunities, bankroll=${bankroll}")
            
            sized = []
            for opp in opportunities:
                try:
                    risk_check = self.risk_manager.check_all(
                        market_price=opp["market_price"],
                        fair_value=opp["fair_value"],
                        market_id=opp["market_id"],
                        confidence=opp["confidence"]
                    )
                    
                    if not risk_check.allowed:
                        logger.info(f"[RiskOfficer] Rejected {opp['question'][:50]}: {risk_check.reason}")
                        continue
                    
                    kelly_res = risk_check.kelly_result
                    side = "BUY"
                    token_id = opp["yes_token_id"]
                    if opp["side"] == "NO" or opp["edge"] < 0:
                        token_id = opp["no_token_id"]
                    
                    sized_opp = {
                        **opp,
                        "kelly_result": kelly_res,
                        "position_size_usd": risk_check.adjusted_size_usd,
                        "position_size_pct": kelly_res.position_size_pct,
                        "side": side,
                        "token_id": token_id,
                        "risk_reason": risk_check.reason
                    }
                    sized.append(sized_opp)
                    logger.info(f"[RiskOfficer] Sized {opp['question'][:50]} | Edge {opp['edge']:.1%} | Size ${risk_check.adjusted_size_usd:.2f}")
                
                except Exception as e:
                    logger.error(f"[RiskOfficer] Sizing failed {opp['question'][:50]}: {e}")
                    continue
            
            # Sort by edge*confidence, limit by max open
            sized.sort(key=lambda x: abs(x["edge"]) * x["confidence"], reverse=True)
            if self.storage:
                from ...config import get_settings
                settings = get_settings()
                max_new = settings.max_open_positions - self.storage.count_open_positions()
                sized = sized[:max_new]
            
            return {
                "sized": sized,
                "count": len(sized),
                "message": f"Sized {len(sized)} positions, bankroll ${bankroll}"
            }
        
        elif task.type == "check_risk":
            if self.monitor:
                report = self.monitor.get_risk_report()
                return {"risk_report": report, "is_over_exposed": report["is_over_exposed"]}
            else:
                return {"risk_report": {}, "is_over_exposed": False}
        
        elif task.type == "check_self_preservation":
            if self.storage:
                from ...config import get_settings
                settings = get_settings()
                sp = self.storage.check_self_preservation(daily_cost=settings.daily_cost_to_cover, max_unprofitable_days=settings.shutdown_if_unprofitable_days)
                return {"self_preservation": sp, "should_shutdown": sp["should_shutdown"]}
            else:
                return {"self_preservation": {}, "should_shutdown": False}
        
        else:
            raise ValueError(f"RiskOfficer doesn't know {task.type}")
