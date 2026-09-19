"""
Routine Recorder - Show a Bot how it's done, it follows along and saves as routine
Premium feature: workflow recording via Playwright
"""
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime, timezone
import uuid
from .routine import Routine, RoutineStep

class RoutineRecorder:
    """
    Records user workflow as Bot follows along
    Show a Bot how it's done once, it saves as routine and runs on its own next time
    """
    def __init__(self, bot=None, vault=None):
        self.bot = bot
        self.vault = vault
        self.is_recording = False
        self.current_routine: Optional[Routine] = None
        self.recorded_steps: List[RoutineStep] = []
        logger.info("RoutineRecorder initialized - ready to show Bot how it's done")
    
    def start_recording(self, name: str, description: str = "", created_by: str = "user") -> Routine:
        """Start recording - Bot follows along as you complete workflow"""
        self.is_recording = True
        self.current_routine = Routine(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            created_by=created_by
        )
        self.recorded_steps = []
        logger.success(f"Started recording routine {name} ID {self.current_routine.id} - Bot following along")
        return self.current_routine
    
    def record_step(self, action: str, target: str, value: str = None, description: str = ""):
        """Record a step as user does it"""
        if not self.is_recording or not self.current_routine:
            logger.warning("Not recording, ignoring step")
            return None
        
        step = self.current_routine.add_step(action=action, target=target, value=value, description=description)
        self.recorded_steps.append(step)
        logger.info(f"Recorded step {len(self.recorded_steps)}: {action} {target} {value or ''} - {description}")
        return step
    
    def stop_recording(self) -> Optional[Routine]:
        """Stop recording and save routine"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        routine = self.current_routine
        self.current_routine = None
        
        if routine and len(routine.steps) > 0:
            logger.success(f"Stopped recording routine {routine.name} with {len(routine.steps)} steps - Bot can now run it on its own")
            
            # Save to manager
            try:
                from .manager import RoutineManager
                manager = RoutineManager()
                manager.save_routine(routine)
                
                # Bot learns the routine
                if self.bot:
                    self.bot.routines.append(routine.id)
                    self.bot.learn(f"Learned routine {routine.name} with {len(routine.steps)} steps - can run on own next time")
                    self.bot.keep_context(f"routine_{routine.id}", routine.to_dict())
            except Exception as e:
                logger.warning(f"Save routine failed: {e}")
            
            return routine
        else:
            logger.warning("Stopped recording but no steps recorded")
            return None
    
    def get_recording_status(self) -> Dict[str, Any]:
        return {
            "is_recording": self.is_recording,
            "current_routine": self.current_routine.name if self.current_routine else None,
            "steps_recorded": len(self.recorded_steps),
            "steps": [{"action": s.action, "target": s.target, "value": s.value} for s in self.recorded_steps[-10:]]
        }

# Example usage for user to show Bot how it's done:
"""
recorder = RoutineRecorder(bot=my_bot)

# User says: Show a Bot how it's done
recorder.start_recording(name="Polymarket Login", description="Login to Polymarket and check portfolio")

# User does workflow, Bot records each step:
recorder.record_step(action="navigate", target="https://polymarket.com", description="Go to Polymarket")
recorder.record_step(action="click", target="button:has-text('Log In')", description="Click Log In")
recorder.record_step(action="type", target="input[type='email']", value="user@example.com", description="Type email")
recorder.record_step(action="click", target="button:has-text('Continue')", description="Click Continue")
recorder.record_step(action="wait", target="2s", description="Wait for login")
recorder.record_step(action="extract", target=".portfolio-value", description="Extract portfolio value")

# User finishes
routine = recorder.stop_recording()
# Bot now has routine saved and can run it on its own next time

# Next time:
# Bot runs routine automatically without user
"""
