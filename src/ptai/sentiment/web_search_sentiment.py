"""
Web Search Sentiment - Fallback when X is blocked (which it now is - 404)
Uses DuckDuckGo search + local LLM to build sentiment
Local, no API key, works even when X blocked
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    source: str = "web"

class WebSearchSentiment:
    def __init__(self):
        logger.info("WebSearchSentiment initialized - fallback for when X blocked")

    def search(self, query: str, max_results: int = 5) -> List[NewsItem]:
        """Search web for market-related news"""
        results = []
        try:
            # Try duckduckgo_search
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append(NewsItem(
                            title=r.get("title", ""),
                            snippet=r.get("body", "")[:400],
                            url=r.get("href", ""),
                            source="duckduckgo"
                        ))
                if results:
                    logger.info(f"DDGS found {len(results)} results for '{query}'")
                    return results
            except ImportError:
                logger.debug("duckduckgo-search not installed")
            except Exception as e:
                logger.debug(f"DDGS failed: {e}")

            # Fallback to requests + duckduckgo html
            try:
                import requests
                from bs4 import BeautifulSoup
                url = f"https://html.duckduckgo.com/html/?q={query}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.select(".result")[:max_results]:
                    title_elem = result.select_one(".result__a")
                    snippet_elem = result.select_one(".result__snippet")
                    if title_elem:
                        results.append(NewsItem(
                            title=title_elem.get_text()[:100],
                            snippet=snippet_elem.get_text()[:400] if snippet_elem else "",
                            url=title_elem.get("href", ""),
                            source="ddg_html"
                        ))
                if results:
                    logger.info(f"DDG HTML found {len(results)} for '{query}'")
                    return results
            except Exception as e:
                logger.debug(f"DDG HTML failed: {e}")

        except Exception as e:
            logger.error(f"Web search failed {query}: {e}")

        return results

    def search_batch(self, queries: List[str], max_per: int = 3) -> Dict[str, List[NewsItem]]:
        batch = {}
        for q in queries:
            # Clean query
            clean_q = q.replace("Will ", "").replace("?", "")[:60]
            batch[q] = self.search(clean_q, max_results=max_per)
        return batch
