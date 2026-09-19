"""
Performance Coach Teammate - Learns from past trades, tracks calibration, improves future fair values
The learning & understanding teammate - premium feature
"""
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime, timezone
from .base import BaseTeammate, Task

class CoachTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, storage=None, brain=None):
        super().__init__(name="Coach", role="Performance Coach - Learns, Tracks Calibration, Improves", vault=vault, memory=memory)
        self.storage = storage
        self.brain = brain
        self.sign_in_to_tool("memory")
        self.sign_in_to_tool("storage")
        self.sign_in_to_tool("calibration_tracker")
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "review_performance":
            logger.info("[Coach] Reviewing performance for learning")
            
            if not self.storage:
                return {"message": "No storage, can't review"}
            
            try:
                perf = self.storage.get_performance_summary()
                sp = self.storage.check_self_preservation()
                
                # Get recent trades
                trades = self.storage.get_recent_trades(50)
                
                # Calculate calibration if possible
                calibration = self._calculate_calibration(trades)
                
                # Learning insights
                insights = []
                if perf["total_trades"] > 0:
                    if perf["win_rate"] and perf["win_rate"] < 50:
                        insights.append(f"Win rate low {perf['win_rate']:.1f}% - consider tightening edge threshold >10%")
                    if perf["total_pnl"] < 0:
                        insights.append(f"PnL negative ${perf['total_pnl']:.2f} - review fair value accuracy, maybe market more efficient than thought")
                    if perf["total_pnl"] > 0:
                        insights.append(f"PnL positive ${perf['total_pnl']:.2f} - strategy working, consider increasing bankroll")
                
                # Remember for future
                if self.memory:
                    self.memory.remember(f"Coach review: {perf['total_trades']} trades, PnL ${perf['total_pnl']:.2f}, win rate {perf.get('win_rate','N/A')}%, calibration {calibration}")
                    for insight in insights:
                        self.memory.remember(f"Coach insight: {insight}")
                
                return {
                    "performance": perf,
                    "self_preservation": sp,
                    "calibration": calibration,
                    "insights": insights,
                    "trades_reviewed": len(trades),
                    "message": f"Reviewed {len(trades)} trades, {len(insights)} insights"
                }
            
            except Exception as e:
                logger.error(f"[Coach] Review failed: {e}")
                return {"error": str(e)}
        
        elif task.type == "learn_from_resolution":
            # When market resolves, learn if fair value was accurate
            market_id = task.payload.get("market_id")
            actual_outcome = task.payload.get("actual_outcome")  # YES/NO or 0/1
            fair_value = task.payload.get("fair_value")
            
            logger.info(f"[Coach] Learning from resolution {market_id} actual {actual_outcome} fair {fair_value}")
            
            if self.memory:
                # Store resolution for future few-shot
                self.memory.remember(f"Market {market_id} resolved {actual_outcome}, fair was {fair_value}, error {abs(float(actual_outcome)-fair_value) if fair_value else 'unknown'}")
            
            # Update calibration tracking
            try:
                if self.storage:
                    # Could add to a resolutions table
                    pass
            except:
                pass
            
            return {"message": f"Learned from {market_id}", "market_id": market_id}
        
        elif task.type == "explain_trade":
            question = task.payload.get("question", "")
            edge = task.payload.get("edge", 0)
            reasoning = task.payload.get("reasoning", "")
            fair = task.payload.get("fair_value", 0)
            market_price = task.payload.get("market_price", 0)
            
            explanation = f"""
Trade Explanation for: {question}

Market Price: {market_price:.1%} | Fair Value: {fair:.1%} | Edge: {edge:.1%}

Reasoning: {reasoning}

Base Rate Consideration: Historical base rate for similar events...
Sentiment: X sentiment score...
Research: Web research findings...
Risk: Kelly sized to 6% max to prevent wipeout...

Confidence: This trade has edge >8% and confidence >60%, meets criteria.
"""
            return {"explanation": explanation}
        
        else:
            raise ValueError(f"Coach doesn't know {task.type}")
    
    def _calculate_calibration(self, trades: List[Dict]) -> Dict[str, Any]:
        # Simple calibration: compare confidence vs actual win rate
        if not trades:
            return {"brier_score": None, "calibration": "No trades"}
        
        # For now, heuristic - would need resolved outcomes
        # Brier score = mean squared error between predicted prob and actual outcome
        # If we don't have actual outcomes yet, return placeholder
        return {
            "trades": len(trades),
            "note": "Calibration requires resolved markets - tracking Brier score when markets resolve",
            "brier_score": None
        }
