"""
MotionMind AI - Integration Tests: Level 2 End-to-End Pipeline
Validates the entire pipeline from canonical 3D landmarks to feature extraction,
real-time classifier inference, debounce consensus, and action dispatch payload formatting.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.gesture_features import extract_gesture_features
from src.features.hand_features import extract_hand_transform
from src.features.smoothing import TransformSmoother
from src.ml.gesture_classifier import RealTimeGestureClassifier
from src.ml.gesture_actions import GestureActionDispatcher
from src.ml.dataset_generator import generate_canonical_gesture, augment_skeleton


class TestLevel2Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = "models/gesture_classifier.joblib"
        if not os.path.exists(model_path):
            cls.skipTest(cls, "Trained gesture model missing.")
        cls.clf = RealTimeGestureClassifier(model_path=model_path, history_len=5, min_confidence=0.60)
        cls.dispatcher = GestureActionDispatcher(default_cooldown_sec=0.1)
        cls.smoother = TransformSmoother(alpha=0.35)

    def test_pipeline_peace_to_action(self):
        canonical = generate_canonical_gesture("Peace")

        # Simulate 5 consecutive frames
        last_action = None
        for _ in range(5):
            hand = augment_skeleton(canonical)
            raw_transform = extract_hand_transform(hand)
            smooth_transform = self.smoother.update(raw_transform)

            name, conf, is_stable = self.clf.predict(hand)
            action = self.dispatcher.process_gesture(name, is_stable, conf)
            if action:
                last_action = action

        # Peace should trigger SPECIAL_EFFECT
        self.assertEqual(last_action, "SPECIAL_EFFECT")

    def test_pipeline_fist_to_lock(self):
        self.clf.reset()
        canonical = generate_canonical_gesture("Fist")

        last_action = None
        for _ in range(5):
            hand = augment_skeleton(canonical)
            name, conf, is_stable = self.clf.predict(hand)
            action = self.dispatcher.process_gesture(name, is_stable, conf)
            if action:
                last_action = action

        self.assertEqual(last_action, "GRAB_LOCK")

    def test_payload_structure(self):
        canonical = generate_canonical_gesture("Thumbs Up")
        hand = augment_skeleton(canonical)
        raw_transform = extract_hand_transform(hand)
        smooth_transform = self.smoother.update(raw_transform)
        name, conf, is_stable = self.clf.predict(hand)

        payload = {
            "is_tracking": True,
            "transform": smooth_transform,
            "fps": 30.0,
            "gesture": {
                "name": name,
                "confidence": conf,
                "stable": is_stable,
                "action": "CONFIRM_ACTION"
            }
        }

        self.assertTrue(payload["is_tracking"])
        self.assertIn("position", payload["transform"])
        self.assertIn("rotation", payload["transform"])
        self.assertEqual(payload["gesture"]["name"], "Thumbs Up")


if __name__ == "__main__":
    unittest.main()
