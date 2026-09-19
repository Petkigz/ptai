"""
Scout Teammate - Scans 500-1000 markets, filters, assigns work to others
Signs into Gamma API, CLOB, can use browser for discovery
"""
from typing import Dict, Any, List
from loguru import logger
from .base import BaseTeammate, Task
from ...markets.scanner import MarketScanner
from ...markets.base import Market

class ScoutTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, llm_router=None, target_count=500):
        super().__init__(name="Scout", role="Market Scanner - Hunts for opportunities", vault=vault, memory=memory, llm_router=llm_router)
        self.scanner = MarketScanner()
        self.target_count = target_count
        self.sign_in_to_tool("gamma_api")
        self.sign_in_to_tool("clob_api")
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "scan_markets":
            count = task.payload.get("count", self.target_count)
            order_by = task.payload.get("order_by", "volume24hr")
            
            logger.info(f"[Scout] Scanning {count} markets ordered by {order_by}")
            markets = self.scanner.scan(target_count=count, order_by=order_by)
            stats = self.scanner.quick_stats(markets)
            
            # Filter and prioritize
            top_by_volume = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:100]
            
            result = {
                "markets": markets,
                "count": len(markets),
                "stats": stats,
                "top_100": top_by_volume,
                "message": f"Scout found {len(markets)} markets, avg vol 24h ${stats.get('avg_volume_24h',0):,.0f}"
            }
            
            # Remember for learning
            if self.memory:
                self.memory.remember(f"Scout scanned {len(markets)} markets, top volume ${stats.get('avg_volume_24h',0):,.0f}")
            
            return result
        
        elif task.type == "filter_markets":
            markets = task.payload.get("markets", [])
            min_liq = task.payload.get("min_liquidity", 1000)
            min_vol = task.payload.get("min_volume_24h", 5000)
            
            filtered = [m for m in markets if m.liquidity >= min_liq and m.volume_24h >= min_vol and m.active]
            
            return {
                "filtered": filtered,
                "original_count": len(markets),
                "filtered_count": len(filtered),
                "message": f"Filtered {len(markets)} -> {len(filtered)} markets"
            }
        
        else:
            raise ValueError(f"Scout doesn't know task type {task.type}")
