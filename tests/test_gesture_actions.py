"""
MotionMind AI - Unit Tests: Gesture Action Dispatcher
Tests action mapping, cooldown timing, and callback triggering.
"""

import sys
import os
import unittest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.gesture_actions import GestureActionDispatcher, ACTION_MAP


class TestGestureActions(unittest.TestCase):

    def setUp(self):
        self.dispatcher = GestureActionDispatcher(default_cooldown_sec=0.2)

    def test_unstable_gesture_triggers_no_action(self):
        action = self.dispatcher.process_gesture("Fist", is_stable=False, confidence=0.95)
        self.assertIsNone(action)

    def test_stable_gesture_triggers_action(self):
        action = self.dispatcher.process_gesture("Peace", is_stable=True, confidence=0.98)
        self.assertEqual(action, "SPECIAL_EFFECT")

    def test_cooldown_prevents_rapid_fire(self):
        action1 = self.dispatcher.process_gesture("Thumbs Up", is_stable=True, confidence=0.99)
        self.assertEqual(action1, "CONFIRM_ACTION")

        # Immediate follow-up should be blocked by cooldown
        action2 = self.dispatcher.process_gesture("Thumbs Up", is_stable=True, confidence=0.99)
        self.assertIsNone(action2)

        # After waiting for cooldown period, should trigger again
        time.sleep(0.25)
        action3 = self.dispatcher.process_gesture("Thumbs Up", is_stable=True, confidence=0.99)
        self.assertEqual(action3, "CONFIRM_ACTION")

    def test_callback_invoked(self):
        called = []

        def on_reset(action_name, payload):
            called.append((action_name, payload["gesture"]))

        self.dispatcher.register_callback("RESET_VIEW", on_reset)
        self.dispatcher.process_gesture("Thumbs Down", is_stable=True, confidence=0.92)

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][0], "RESET_VIEW")
        self.assertEqual(called[0][1], "Thumbs Down")


if __name__ == "__main__":
    unittest.main()
