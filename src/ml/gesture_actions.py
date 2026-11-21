"""
MotionMind AI - Interaction Logic: Gesture Action Dispatcher
Maps recognized gestures to discrete application actions with cooldowns, hysteresis, and event callbacks.
"""

import time
from typing import Dict, Callable, Optional, Any


# Standard semantic action types triggered by gestures
ACTION_MAP = {
    "Open Palm": "INTERACT_FREE",
    "Fist": "GRAB_LOCK",
    "Peace": "SPECIAL_EFFECT",
    "Thumbs Up": "CONFIRM_ACTION",
    "Thumbs Down": "RESET_VIEW",
    "OK Sign": "SNAP_GRID",
    "Pointing": "AIM_TARGET",
    "Rock Horns": "BOOST_SPIN",
    "Call Me": "TOGGLE_AUDIO",
    "Hand Gun": "TRIGGER_FIRE",
    "Pinch": "SCALE_OBJECT",
    "Three": "SELECT_SLOT_3",
    "Four": "SELECT_SLOT_4",
    "Stop Sign": "EMERGENCY_STOP",
    "PSL Salaam": "SIGN_GREETING",
    "PSL Shukriya": "SIGN_GRATITUDE",
}


class GestureActionDispatcher:
    """
    Stateful action dispatcher that receives stable gestures and triggers callbacks with cooldown throttling.
    """

    def __init__(self, default_cooldown_sec: float = 1.2):
        self.default_cooldown_sec = default_cooldown_sec
        self.last_triggered_time: Dict[str, float] = {}
        self.active_gesture: str = "None"
        self.callbacks: Dict[str, Callable[[str, Dict[str, Any]], None]] = {}

    def register_callback(self, action_name: str, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a handler callback for an action event."""
        self.callbacks[action_name] = callback

    def process_gesture(
        self,
        gesture_name: str,
        is_stable: bool,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Evaluate gesture event. If stable and cooldown elapsed, fire action.
        Returns the action name if triggered, else None.
        """
        if not is_stable or gesture_name not in ACTION_MAP:
            self.active_gesture = "None"
            return None

        action = ACTION_MAP[gesture_name]
        now = time.time()
        last_time = self.last_triggered_time.get(action, 0.0)

        # Continuous actions (like scaling and free interaction) bypass cooldown
        continuous_actions = {"INTERACT_FREE", "SCALE_OBJECT"}
        if action not in continuous_actions:
            if (now - last_time) < self.default_cooldown_sec:
                return None  # In cooldown

        self.last_triggered_time[action] = now
        self.active_gesture = gesture_name

        payload = {
            "gesture": gesture_name,
            "confidence": confidence,
            "timestamp": now,
        }
        if metadata:
            payload.update(metadata)

        # Trigger callback if registered
        if action in self.callbacks:
            try:
                self.callbacks[action](action, payload)
            except Exception as e:
                print(f"[GestureAction] Callback error for {action}: {e}")

        return action

    def get_action_for_gesture(self, gesture_name: str) -> str:
        """Lookup action name mapped to a gesture."""
        return ACTION_MAP.get(gesture_name, "NO_ACTION")
