"""
Polymarket execution orchestrator - chooses API vs Browser
"""
from typing import Dict, Any, Optional
from loguru import logger

from ..markets.polymarket import PolymarketExecutor as APIExecutor
from .browser import BrowserExecutor
from ..config import get_settings

class ExecutionOrchestrator:
    def __init__(self, api_executor: Optional[APIExecutor] = None, browser_executor: Optional[BrowserExecutor] = None):
        self.settings = get_settings()
        self.api_executor = api_executor or APIExecutor(
            private_key=self.settings.polymarket_private_key,
            funder=self.settings.polymarket_funder_address,
            chain_id=self.settings.polymarket_chain_id,
            signature_type=self.settings.polymarket_signature_type,
            host=self.settings.polymarket_host
        )
        self.browser_executor = browser_executor
        self.mode = self.settings.scan_order_by  # Actually execution mode is separate
        # Get execution mode from env
        import os
        self.exec_mode = os.getenv("EXECUTION_MODE", "hybrid")
        self.dry_run = self.settings.dry_run
        logger.info(f"ExecutionOrchestrator mode={self.exec_mode} dry_run={self.dry_run}")

    async def execute(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading opportunity
        opportunity dict contains:
        - market_id, token_id, market_price, fair_value, edge, side, position_size_usd, event_slug
        """
        market_id = opportunity.get("market_id")
        token_id = opportunity.get("token_id") or opportunity.get("yes_token_id")
        side = opportunity.get("side", "BUY")
        amount = opportunity.get("position_size_usd", 1.0)
        price = opportunity.get("market_price", 0.5)
        event_slug = opportunity.get("event_slug", "")

        # Safety: if token_id missing, can't execute via API
        if not token_id:
            logger.warning(f"No token_id for {market_id}, trying browser with slug {event_slug}")
            if self.browser_executor and not self.dry_run:
                return await self.browser_executor.execute_polymarket_trade_browser(
                    market_slug=event_slug,
                    side=side,
                    amount_usd=amount,
                    dry_run=self.dry_run
                )
            else:
                logger.info(f"[DRY RUN] Would execute {side} ${amount} on {event_slug} but no token_id")
                return {"status": "dry_run", "reason": "no token_id", "opportunity": opportunity}

        # Try API first (faster, more reliable)
        if self.exec_mode in ["api", "hybrid"]:
            try:
                # For Polymarket, price is the limit price, size is in shares or USD?
                # py-clob-client expects size as shares, but we can use market order by amount
                # We'll use limit order at market price + slippage
                slippage = 0.02
                limit_price = price * (1 + slippage) if side == "BUY" else price * (1 - slippage)
                limit_price = max(0.01, min(0.99, limit_price))

                # Convert USD amount to shares: shares = amount / price
                shares = amount / price if price > 0 else amount

                result = self.api_executor.place_order(
                    token_id=token_id,
                    price=limit_price,
                    size=shares,
                    side=side,
                    dry_run=self.dry_run
                )
                logger.success(f"API execution result: {result}")
                return {"status": "executed_api", "result": result, "opportunity": opportunity}
            except Exception as e:
                logger.error(f"API execution failed: {e}, falling back to browser if hybrid")
                if self.exec_mode == "api":
                    return {"status": "failed", "error": str(e)}

        # Fallback to browser
        if self.exec_mode in ["browser", "hybrid"]:
            if not self.browser_executor:
                logger.warning("Browser executor not initialized")
                return {"status": "failed", "error": "browser not initialized"}

            try:
                result = await self.browser_executor.execute_polymarket_trade_browser(
                    market_slug=event_slug,
                    side=side,
                    amount_usd=amount,
                    dry_run=self.dry_run
                )
                return {"status": "executed_browser", "result": result, "opportunity": opportunity}
            except Exception as e:
                logger.error(f"Browser execution failed: {e}")
                return {"status": "failed", "error": str(e)}

        return {"status": "failed", "error": "no execution mode succeeded"}

    def execute_sync(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Sync wrapper"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If already in async context, create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.execute(opportunity))
                return future.result()
        else:
            return loop.run_until_complete(self.execute(opportunity))
