"""
Agent Tools - Gives agent ability to use computer, browser, terminal
- web_search
- browser research
- terminal commands
- file system
- X sentiment
"""
import subprocess
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from loguru import logger

class Tool:
    name: str
    description: str

    def execute(self, *args, **kwargs) -> str:
        raise NotImplementedError

class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information about a market question"

    def __init__(self, use_local: bool = True):
        self.use_local = use_local

    def execute(self, query: str, max_results: int = 5) -> str:
        """Simple web search - uses DuckDuckGo or local search"""
        try:
            # Try duckduckgo_search package if available
            try:
                from duckduckgo_search import DDGS
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append(f"{r['title']}: {r['body'][:300]} ({r['href']})")
                return "\n".join(results)
            except ImportError:
                pass

            # Fallback to requests + duckduckgo html
            import requests
            from bs4 import BeautifulSoup
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for result in soup.select(".result")[:max_results]:
                title = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if title:
                    results.append(f"{title.get_text()}: {snippet.get_text()[:300] if snippet else ''}")
            return "\n".join(results) if results else f"No results for {query}"

        except Exception as e:
            logger.error(f"Web search failed {query}: {e}")
            return f"Search failed: {e}"

class TerminalTool(Tool):
    name = "terminal"
    description = "Execute terminal commands locally"

    def __init__(self, allowed_commands: List[str] = None):
        # For safety, we could restrict, but user wants full computer use
        self.allowed_commands = allowed_commands  # None = allow all (user's PC)

    def execute(self, command: str, timeout: int = 30) -> str:
        """Execute shell command"""
        try:
            logger.info(f"Terminal executing: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nReturn code: {result.returncode}"
            return output[:5000]  # Truncate
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Command failed: {e}"

class BrowserTool(Tool):
    name = "browser"
    description = "Use browser to research or interact with websites"

    def __init__(self, browser_executor=None):
        self.browser = browser_executor

    def execute(self, action: str, url: str = "", query: str = "") -> str:
        """Browser actions: navigate, research, screenshot"""
        if not self.browser:
            return "Browser not initialized"

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if action == "navigate" and url:
                loop.run_until_complete(self.browser.start())
                success = loop.run_until_complete(self.browser.navigate(url))
                return f"Navigated to {url}: {success}"

            elif action == "research" and query:
                loop.run_until_complete(self.browser.start())
                text = loop.run_until_complete(self.browser.research_market(query))
                return text[:5000]

            elif action == "screenshot":
                # Take screenshot
                if not self.browser.is_running:
                    loop.run_until_complete(self.browser.start())
                path = f"./logs/screenshot_{int(time.time())}.png"
                Path("./logs").mkdir(exist_ok=True)
                loop.run_until_complete(self.browser.page.screenshot(path=path))
                return f"Screenshot saved to {path}"

            else:
                return f"Unknown browser action {action}"

        except Exception as e:
            return f"Browser tool failed: {e}"

class FileTool(Tool):
    name = "file"
    description = "Read/write local files"

    def execute(self, action: str, path: str, content: str = "") -> str:
        try:
            p = Path(path)
            if action == "read":
                if not p.exists():
                    return f"File not found: {path}"
                return p.read_text()[:10000]
            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                return f"Wrote {len(content)} chars to {path}"
            elif action == "list":
                if not p.exists():
                    return f"Path not found: {path}"
                if p.is_dir():
                    files = list(p.iterdir())[:50]
                    return "\n".join([str(f) for f in files])
                else:
                    return f"Not a directory: {path}"
            else:
                return f"Unknown file action {action}"
        except Exception as e:
            return f"File tool failed: {e}"

class ToolRegistry:
    def __init__(self, browser_executor=None):
        self.tools: Dict[str, Tool] = {
            "web_search": WebSearchTool(),
            "terminal": TerminalTool(),
            "browser": BrowserTool(browser_executor),
            "file": FileTool()
        }
        logger.info(f"ToolRegistry initialized with {list(self.tools.keys())}")

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def execute(self, tool_name: str, **kwargs) -> str:
        tool = self.get(tool_name)
        if not tool:
            return f"Tool {tool_name} not found"
        return tool.execute(**kwargs)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
