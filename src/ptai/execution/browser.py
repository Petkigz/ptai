"""
Browser automation - Executes trades in its own browser
Local, stealth, human-like
Uses Playwright with persistent context
"""
import asyncio
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from loguru import logger

@dataclass
class BrowserConfig:
    headless: bool = False
    persistent_dir: str = "./browser/profiles/default"
    stealth: bool = True
    timeout: int = 30000
    human_delay: bool = True

class BrowserExecutor:
    """
    Own browser that can:
    - Navigate to Polymarket, Kalshi, etc.
    - Execute trades via UI if API fails
    - Research via web search
    - Read X sentiment via logged-in session
    """
    def __init__(self, config: Optional[BrowserConfig] = None):
        from ..config import get_settings
        settings = get_settings()
        self.config = config or BrowserConfig(
            headless=settings.browser_headless,
            persistent_dir=settings.browser_persistent_dir,
            stealth=settings.browser_stealth,
            timeout=settings.browser_timeout
        )
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_running = False
        logger.info(f"BrowserExecutor config: headless={self.config.headless}, dir={self.config.persistent_dir}")

    async def start(self):
        """Start browser with persistent profile"""
        try:
            from playwright.async_api import async_playwright
            # Optional stealth
            try:
                from playwright_stealth import stealth_async
                has_stealth = True
            except ImportError:
                has_stealth = False
                logger.warning("playwright-stealth not installed, running without stealth")

            self.playwright = await async_playwright().start()

            # Ensure profile dir exists
            profile_path = Path(self.config.persistent_dir)
            profile_path.mkdir(parents=True, exist_ok=True)

            # Launch persistent context
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=self.config.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-infobars",
                    "--window-size=1280,800"
                ],
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                ignore_https_errors=True,
                timeout=self.config.timeout
            )

            # Get or create page
            if len(self.context.pages) > 0:
                self.page = self.context.pages[0]
            else:
                self.page = await self.context.new_page()

            if has_stealth and self.config.stealth:
                await stealth_async(self.page)

            self.is_running = True
            logger.success(f"Browser started, profile: {profile_path}")

        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop browser"""
        try:
            if self.context:
                await self.context.close()
            if self.playwright:
                await self.playwright.stop()
            self.is_running = False
            logger.info("Browser stopped")
        except Exception as e:
            logger.error(f"Browser stop error: {e}")

    async def human_delay(self, min_ms: int = 300, max_ms: int = 1200):
        """Human-like delay"""
        if self.config.human_delay:
            delay = random.randint(min_ms, max_ms) / 1000
            await asyncio.sleep(delay)

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> bool:
        """Navigate to URL"""
        if not self.is_running or not self.page:
            logger.error("Browser not running")
            return False
        try:
            await self.page.goto(url, wait_until=wait_until, timeout=self.config.timeout)
            await self.human_delay(500, 1500)
            logger.info(f"Navigated to {url}")
            return True
        except Exception as e:
            logger.error(f"Navigate failed {url}: {e}")
            return False

    async def execute_polymarket_trade_browser(self, market_slug: str, side: str, amount_usd: float, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute trade via Polymarket UI
        This is fallback if API fails, or for sites without API
        """
        if dry_run:
            logger.info(f"[DRY RUN BROWSER] Would trade {side} ${amount_usd} on {market_slug}")
            return {"status": "dry_run", "slug": market_slug, "side": side, "amount": amount_usd}

        if not self.is_running:
            await self.start()

        try:
            url = f"https://polymarket.com/event/{market_slug}"
            success = await self.navigate(url)
            if not success:
                return {"status": "failed", "error": "navigation failed"}

            # Wait for market to load
            await self.page.wait_for_selector("body", timeout=10000)
            await self.human_delay(1000, 2000)

            # Take screenshot for debugging
            screenshot_path = f"./logs/browser_{int(time.time())}.png"
            Path("./logs").mkdir(exist_ok=True)
            await self.page.screenshot(path=screenshot_path)
            logger.info(f"Screenshot saved {screenshot_path}")

            # Here would be actual UI interaction:
            # - Find YES/NO buttons
            # - Click appropriate side
            # - Enter amount
            # - Confirm
            # This is highly dependent on Polymarket's current DOM, so we provide a template

            # Example (pseudo):
            # yes_button = await self.page.query_selector('button:has-text("Yes")')
            # if side == "YES" and yes_button:
            #     await yes_button.click()
            #     await self.human_delay()
            #     amount_input = await self.page.query_selector('input[placeholder*="Amount"]')
            #     await amount_input.fill(str(amount_usd))
            #     ...

            logger.warning("Browser trade execution is template - requires DOM adaptation for current Polymarket UI")
            return {"status": "template", "message": "Browser execution needs DOM selectors updated", "screenshot": screenshot_path}

        except Exception as e:
            logger.error(f"Browser trade failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def research_market(self, question: str, max_pages: int = 3) -> str:
        """Use browser to research a market question"""
        if not self.is_running:
            await self.start()

        try:
            # Search via Google or direct
            search_q = question.replace(" ", "+")
            url = f"https://www.google.com/search?q={search_q}"

            await self.navigate(url)
            await self.page.wait_for_selector("body", timeout=10000)

            # Get page content
            content = await self.page.content()
            # Extract text
            text = await self.page.inner_text("body")
            # Truncate
            text = text[:5000]

            logger.info(f"Researched '{question[:50]}' - got {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return f"Research failed: {e}"

    async def scrape_x_via_browser(self, query: str, limit: int = 20) -> list:
        """Scrape X using logged-in browser session"""
        if not self.is_running:
            await self.start()

        try:
            # Navigate to X search
            search_q = query.replace(" ", "%20")
            url = f"https://x.com/search?q={search_q}&f=live"

            await self.navigate(url)
            await self.page.wait_for_selector("body", timeout=10000)
            await self.human_delay(2000, 4000)

            # Scroll to load tweets
            for _ in range(3):
                await self.page.mouse.wheel(0, 1000)
                await self.human_delay(1000, 2000)

            # Try to extract tweets (X DOM is complex, this is best effort)
            tweets = []
            try:
                # Look for tweet articles
                articles = await self.page.query_selector_all('article')
                for article in articles[:limit]:
                    try:
                        text = await article.inner_text()
                        tweets.append(text[:500])
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Tweet extraction failed: {e}")

            logger.info(f"Browser X scrape found {len(tweets)} tweets for '{query}'")
            return tweets

        except Exception as e:
            logger.error(f"Browser X scrape failed: {e}")
            return []

    # Sync wrappers for ease of use in non-async code
    def start_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start())
        return loop

    def navigate_sync(self, url: str):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.navigate(url))

    def research_sync(self, question: str) -> str:
        loop = asyncio.get_event_loop()
        if not self.is_running:
            loop.run_until_complete(self.start())
        return loop.run_until_complete(self.research_market(question))
