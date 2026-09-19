"""
Researcher Teammate - Uses computer, browser, terminal to research markets
Signs into web_search, browser, terminal - ReAct loop 45s per market
"""
from typing import Dict, Any
from loguru import logger
from .base import BaseTeammate, Task

class ResearcherTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, llm_router=None, browser_executor=None, tool_registry=None):
        super().__init__(name="Researcher", role="Deep Researcher - Uses computer, browser, terminal", vault=vault, memory=memory, llm_router=llm_router)
        self.browser_executor = browser_executor
        self.tool_registry = tool_registry
        self.sign_in_to_tool("web_search")
        self.sign_in_to_tool("browser")
        self.sign_in_to_tool("terminal")
        
        # Lazy import to avoid circular
        try:
            from ..tools import ToolRegistry
            from ..researcher import Researcher as ResearcherAgent
            if not tool_registry:
                self.tool_registry = ToolRegistry(browser_executor=browser_executor)
            self.researcher_agent = ResearcherAgent(tool_registry=self.tool_registry, browser_executor=browser_executor)
        except Exception as e:
            logger.warning(f"Researcher tools init failed: {e}")
            self.researcher_agent = None
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "research_batch":
            markets = task.payload.get("markets", [])
            sentiments = task.payload.get("sentiments", {})
            max_markets = task.payload.get("max_markets", 20)
            
            logger.info(f"[Researcher] Researching {min(len(markets), max_markets)} markets via web_search, browser, terminal")
            
            if not self.researcher_agent:
                return {"research": {}, "count": 0, "message": "Researcher tools not available"}
            
            try:
                research_dict = self.researcher_agent.research_batch(markets, sentiments, max_markets=max_markets)
                return {
                    "research": research_dict,
                    "count": len(research_dict),
                    "message": f"Research done for {len(research_dict)} markets"
                }
            except Exception as e:
                logger.warning(f"Research batch failed: {e}")
                return {"research": {}, "count": 0, "error": str(e)}
        
        elif task.type == "research_single":
            market = task.payload.get("market")
            sentiment = task.payload.get("sentiment")
            if not self.researcher_agent:
                return {"research": "", "error": "No researcher"}
            try:
                result = self.researcher_agent.research_market(market, sentiment)
                return {"research": result, "market_id": market.id if market else "unknown"}
            except Exception as e:
                return {"research": "", "error": str(e)}
        
        else:
            raise ValueError(f"Researcher doesn't know {task.type}")
