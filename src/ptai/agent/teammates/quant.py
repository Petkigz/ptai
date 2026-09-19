"""
Quant Teammate - Superforecaster, builds fair value, flags mispricing >8%
Signs into LM Studio, Ollama, Grok - uses LLM reasoning + base rates
"""
from typing import Dict, Any, List
from loguru import logger
import os
from .base import BaseTeammate, Task
from ...markets.base import Market

class QuantTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, llm_router=None, brain=None, storage=None):
        super().__init__(name="Quant", role="Superforecaster - Fair Value & Edge Hunter", vault=vault, memory=memory, llm_router=llm_router)
        self.storage = storage
        self.sign_in_to_tool("lm_studio")
        self.sign_in_to_tool("ollama")
        self.sign_in_to_tool("grok")  # future
        
        if brain:
            self.brain = brain
        else:
            try:
                from ..brain import Brain
                self.brain = Brain()
            except Exception as e:
                logger.warning(f"Quant brain init failed: {e}")
                self.brain = None
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "build_fair_value":
            markets = task.payload.get("markets", [])
            sentiments = task.payload.get("sentiments", {})
            research_dict = task.payload.get("research", {})
            
            logger.info(f"[Quant] Building fair value for {len(markets)} markets")
            
            if not self.brain:
                return {"opportunities": [], "analyzed": 0, "error": "No brain"}
            
            # Auto-detect R1 for top_n
            model_name = "local-model"
            try:
                if hasattr(self.brain, 'llm_router') and self.brain.llm_router:
                    detected = self.brain.llm_router.get_detected_models() if hasattr(self.brain.llm_router, 'get_detected_models') else []
                    if detected:
                        model_name = " ".join(detected).lower()
                    provider = self.brain.llm_router.get_provider() if hasattr(self.brain.llm_router, 'get_provider') else None
                    if provider and hasattr(provider, 'model'):
                        model_name = (provider.model + " " + model_name).lower()
            except:
                pass
            
            is_r1 = "r1" in model_name or "distill" in model_name
            is_70b = "70b" in model_name or "72b" in model_name
            is_32b = "32b" in model_name
            
            if is_r1 and is_32b:
                top_n = 5
            elif is_r1 and is_70b:
                top_n = 3
            elif is_70b:
                top_n = 30
            elif is_32b:
                top_n = 50
            else:
                top_n = 50
            
            top_n = int(os.getenv("MAX_DEEP_ANALYZE", top_n))
            logger.info(f"[Quant] Model {model_name[:80]} -> top_n={top_n} is_r1={is_r1}")
            
            sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)
            top_markets = sorted_markets[:top_n]
            
            results = []
            opportunities = []
            
            for market in top_markets:
                try:
                    sentiment = sentiments.get(market.question) or sentiments.get(market.id)
                    research_text = research_dict.get(market.id, "")
                    fv = self.brain.estimate_fair_value(market, sentiment, research_text=research_text)
                    
                    if self.storage:
                        try:
                            self.storage.log_market_analysis({
                                "market_id": market.id,
                                "question": market.question,
                                "market_price": fv.market_price,
                                "fair_value": fv.fair_value,
                                "edge": fv.edge,
                                "sentiment_score": fv.sentiment_score,
                                "sentiment_summary": fv.sentiment_summary,
                                "reasoning": fv.reasoning,
                                "confidence": fv.confidence,
                                "should_trade": fv.should_trade,
                                "raw_data": fv.raw
                            })
                        except Exception as e:
                            logger.debug(f"Log market analysis failed: {e}")
                    
                    item = {
                        "market": market,
                        "fair_value_result": fv,
                        "market_price": fv.market_price,
                        "fair_value": fv.fair_value,
                        "edge": fv.edge,
                        "confidence": fv.confidence,
                        "should_trade": fv.should_trade,
                        "side": fv.side,
                        "sentiment": sentiment,
                        "reasoning": fv.reasoning,
                        "event_slug": market.event_slug,
                        "market_id": market.id,
                        "token_id": market.yes_token_id,
                        "yes_token_id": market.yes_token_id,
                        "no_token_id": market.no_token_id,
                        "question": market.question,
                        "llm_provider": fv.llm_provider
                    }
                    results.append(item)
                    
                    # Check if opportunity
                    from ...config import get_settings
                    settings = get_settings()
                    if abs(item["edge"]) >= settings.min_edge_pct and item["should_trade"]:
                        opportunities.append(item)
                
                except Exception as e:
                    logger.error(f"[Quant] Fair value failed {market.question[:50]}: {e}")
                    continue
            
            logger.success(f"[Quant] Done: {len(results)} analyzed, {len(opportunities)} opps >8%")
            
            # Learning: remember best opportunities
            if self.memory and opportunities:
                for opp in opportunities[:3]:
                    self.memory.remember(f"Quant found opportunity: {opp['question'][:80]} edge {opp['edge']:.1%} conf {opp['confidence']:.2f} reasoning: {opp['reasoning'][:200]}")
            
            return {
                "results": results,
                "opportunities": opportunities,
                "analyzed": len(results),
                "opportunities_count": len(opportunities),
                "model": model_name,
                "top_n": top_n,
                "message": f"Analyzed {len(results)}, {len(opportunities)} opportunities"
            }
        
        else:
            raise ValueError(f"Quant doesn't know {task.type}")
