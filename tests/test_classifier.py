"""
MotionMind AI - Unit Tests: Gesture Classifier Inference Engine
Tests model loading, prediction output format, confidence thresholding, and temporal voting.
"""

import sys
import os
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.gesture_classifier import RealTimeGestureClassifier


def make_lm(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def make_open_palm():
    """Create a rough open-palm landmark configuration."""
    return [
        make_lm(0.50, 0.80, 0.00),  # Wrist
        make_lm(0.44, 0.74, -0.01), make_lm(0.39, 0.66, -0.02),
        make_lm(0.36, 0.58, -0.03), make_lm(0.33, 0.52, -0.04),
        make_lm(0.44, 0.58, -0.01), make_lm(0.43, 0.48, -0.02),
        make_lm(0.42, 0.40, -0.02), make_lm(0.41, 0.33, -0.03),
        make_lm(0.50, 0.57, 0.00), make_lm(0.50, 0.46, -0.01),
        make_lm(0.50, 0.37, -0.02), make_lm(0.50, 0.29, -0.02),
        make_lm(0.56, 0.58, 0.01), make_lm(0.57, 0.48, 0.01),
        make_lm(0.58, 0.41, 0.00), make_lm(0.58, 0.35, -0.01),
        make_lm(0.61, 0.62, 0.02), make_lm(0.63, 0.55, 0.02),
        make_lm(0.65, 0.49, 0.02), make_lm(0.66, 0.44, 0.01),
    ]


class TestGestureClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = "models/gesture_classifier.joblib"
        if not os.path.exists(model_path):
            cls.skipTest(cls, "Trained model not found, skipping classifier tests.")
        cls.clf = RealTimeGestureClassifier(model_path=model_path)

    def test_predict_returns_tuple(self):
        hand = make_open_palm()
        result = self.clf.predict(hand)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_predict_returns_string_label(self):
        hand = make_open_palm()
        name, conf, stable = self.clf.predict(hand)
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    def test_confidence_range(self):
        hand = make_open_palm()
        _, conf, _ = self.clf.predict(hand)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_stability_after_repeated_predictions(self):
        self.clf.reset()
        hand = make_open_palm()
        for _ in range(6):
            _, _, stable = self.clf.predict(hand)
        # After multiple consistent predictions, stability should be True
        self.assertTrue(stable)

    def test_reset_clears_history(self):
        hand = make_open_palm()
        for _ in range(5):
            self.clf.predict(hand)
        self.clf.reset()
        self.assertEqual(len(self.clf.history), 0)
        self.assertEqual(self.clf.last_stable_gesture, "None")


if __name__ == "__main__":
    unittest.main()
