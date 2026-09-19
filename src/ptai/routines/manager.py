"""
Routine Manager - Manages saved routines that Bots can run on their own
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from loguru import logger
from .routine import Routine, RoutineStep

class RoutineManager:
    def __init__(self, db_path: str = "./data/routines.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.routines: Dict[str, Routine] = {}
        self.load()
    
    def load(self):
        if self.db_path.exists():
            try:
                import uuid
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    for r_dict in data:
                        steps = []
                        for s in r_dict.get("steps", []):
                            # Handle old format without id
                            if "id" not in s:
                                s["id"] = str(uuid.uuid4())[:8]
                            if "timestamp" not in s:
                                from datetime import datetime, timezone
                                s["timestamp"] = datetime.now(timezone.utc).isoformat()
                            try:
                                steps.append(RoutineStep(**s))
                            except Exception as se:
                                logger.warning(f"Failed to load step {s}: {se}")
                        routine = Routine(
                            id=r_dict["id"],
                            name=r_dict["name"],
                            description=r_dict["description"],
                            steps=steps,
                            created_by=r_dict.get("created_by", ""),
                            created_at=r_dict.get("created_at", ""),
                            last_run=r_dict.get("last_run"),
                            run_count=r_dict.get("run_count", 0),
                            success_count=r_dict.get("success_count", 0),
                            is_active=r_dict.get("is_active", True),
                            tags=r_dict.get("tags", [])
                        )
                        self.routines[routine.id] = routine
                logger.info(f"RoutineManager loaded {len(self.routines)} routines")
            except Exception as e:
                logger.warning(f"Load routines failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
    
    def save(self):
        try:
            data = [r.to_dict() for r in self.routines.values()]
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save routines failed: {e}")
    
    def save_routine(self, routine: Routine):
        self.routines[routine.id] = routine
        self.save()
        logger.success(f"Saved routine {routine.name} ID {routine.id} with {len(routine.steps)} steps")
    
    def get_routine(self, routine_id: str) -> Optional[Routine]:
        return self.routines.get(routine_id)
    
    def get_routine_by_name(self, name: str) -> Optional[Routine]:
        for r in self.routines.values():
            if r.name.lower() == name.lower():
                return r
        return None
    
    def list_routines(self) -> List[Routine]:
        return list(self.routines.values())
    
    def delete_routine(self, routine_id: str) -> bool:
        if routine_id in self.routines:
            del self.routines[routine_id]
            self.save()
            return True
        return False
    
    async def run_routine(self, routine_id: str, bot=None, browser_executor=None) -> Dict[str, Any]:
        """Bot runs routine on its own - uses apps like you would"""
        routine = self.get_routine(routine_id)
        if not routine:
            raise ValueError(f"Routine {routine_id} not found")
        
        logger.info(f"Running routine {routine.name} with {len(routine.steps)} steps via bot {bot.name if bot else 'unknown'}")
        
        results = []
        success = True
        
        try:
            for i, step in enumerate(routine.steps):
                logger.info(f"Routine {routine.name} step {i+1}/{len(routine.steps)}: {step.action} {step.target}")
                
                # Simulate execution - in real product, would use Playwright
                # For now, log and simulate
                if step.action == "navigate":
                    # browser_executor.navigate(step.target)
                    results.append({"step": i, "action": "navigate", "target": step.target, "status": "done"})
                elif step.action == "click":
                    # browser_executor.click(step.target)
                    results.append({"step": i, "action": "click", "target": step.target, "status": "done"})
                elif step.action == "type":
                    # browser_executor.type(step.target, step.value)
                    results.append({"step": i, "action": "type", "target": step.target, "value": "***", "status": "done"})
                elif step.action == "wait":
                    import asyncio
                    await asyncio.sleep(1)
                    results.append({"step": i, "action": "wait", "status": "done"})
                elif step.action == "extract":
                    results.append({"step": i, "action": "extract", "target": step.target, "value": "mock_value", "status": "done"})
                else:
                    results.append({"step": i, "action": step.action, "status": "done"})
            
            routine.run_count += 1
            routine.success_count += 1 if success else 0
            routine.last_run = datetime.now(timezone.utc).isoformat()
            self.save()
            
            if bot:
                bot.learn(f"Ran routine {routine.name} successfully - {len(routine.steps)} steps")
            
            return {"routine": routine.name, "steps": len(routine.steps), "results": results, "success": success}
        
        except Exception as e:
            logger.error(f"Routine {routine.name} failed: {e}")
            routine.run_count += 1
            self.save()
            return {"routine": routine.name, "error": str(e), "success": False, "results": results}
