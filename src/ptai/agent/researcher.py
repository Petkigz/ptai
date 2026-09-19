"""
Researcher - Gives agent ability to research on its own using computer, browser, terminal
ReAct loop: Reason + Act for up to 45 seconds per market
Local only, no cloud
"""
import time
from typing import Dict, List, Optional
from loguru import logger

from ..config import get_settings
from .tools import ToolRegistry
from ..markets.base import Market

class Researcher:
    """
    Autonomous researcher that can use:
    - web_search (DuckDuckGo)
    - browser (Playwright)
    - terminal (shell)
    - file (local FS)
    To build context for fair value
    """
    def __init__(self, tool_registry: Optional[ToolRegistry] = None, browser_executor=None):
        self.settings = get_settings()
        self.tool_registry = tool_registry or ToolRegistry(browser_executor=browser_executor)
        self.max_time_per_market = 45  # seconds, as per spec
        logger.info("Researcher initialized with tools: web_search, browser, terminal, file")

    def research(self, market: Market, sentiment_summary: str = "") -> str:
        """
        Research a market for up to 45 seconds
        Returns research text to feed into Brain
        """
        start = time.time()
        question = market.question
        research_parts = []

        logger.info(f"Researching: {question[:80]} (max {self.max_time_per_market}s)")

        # 1. Web search (fast, local)
        try:
            # Clean query
            query = question.replace("Will ", "").replace("?", "")[:80]
            web_result = self.tool_registry.execute("web_search", query=query, max_results=5)
            if web_result and "failed" not in web_result.lower():
                research_parts.append(f"WEB SEARCH for '{query}':\n{web_result[:1500]}")
                logger.info(f"Web search got {len(web_result)} chars")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        if time.time() - start > self.max_time_per_market:
            return "\n\n".join(research_parts)

        # 2. Terminal research - check local files, news, etc
        try:
            # Example: check if we have any local knowledge base
            # For now, just log that terminal tool is available
            # In future, could run: curl, python scripts, etc.
            pass
        except Exception as e:
            logger.warning(f"Terminal research failed: {e}")

        if time.time() - start > self.max_time_per_market:
            return "\n\n".join(research_parts)

        # 3. Browser research (if available and time left)
        # This is slower, so only if we have time and browser
        try:
            browser_tool = self.tool_registry.get("browser")
            if browser_tool and browser_tool.browser and browser_tool.browser.is_running:
                # Don't do heavy browser research for every market in batch mode
                # Only for high-value opportunities
                if market.volume_24h > 50000:
                    b_result = self.tool_registry.execute("browser", action="research", query=question)
                    if b_result:
                        research_parts.append(f"BROWSER RESEARCH:\n{b_result[:1500]}")
        except Exception as e:
            logger.debug(f"Browser research skipped: {e}")

        elapsed = time.time() - start
        combined = "\n\n".join(research_parts)
        logger.info(f"Research done for '{question[:50]}' in {elapsed:.1f}s, got {len(combined)} chars")

        return combined[:3000]  # Truncate for LLM context

    def research_batch(self, markets: List[Market], sentiments: Dict, max_markets: int = 20) -> Dict[str, str]:
        """Research batch of markets (only top by volume to save time)"""
        # Sort by volume and only research top N
        sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:max_markets]
        results = {}
        for market in sorted_markets:
            sentiment = sentiments.get(market.question)
            sentiment_summary = sentiment.summary if sentiment else ""
            research_text = self.research(market, sentiment_summary)
            results[market.id] = research_text
            # Small delay to avoid hammering
            time.sleep(0.5)
        return results
