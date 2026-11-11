"""
MotionMind AI - Automated Unit Tests
Tests hand feature extraction and smoothing algorithms.
"""

import sys
import os
import unittest
from types import SimpleNamespace

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.hand_features import (
    get_palm_center,
    get_pinch_distance,
    get_hand_rotation,
    extract_hand_transform,
)
from src.features.smoothing import TransformSmoother


def create_mock_landmark(x=0.5, y=0.5, z=0.0):
    """Helper to generate a mock landmark with x, y, z attributes."""
    return SimpleNamespace(x=x, y=y, z=z)


def create_mock_hand_landmarks():
    """Create a 21-element list mimicking MediaPipe landmarks."""
    landmarks = [create_mock_landmark(0.5, 0.5, 0.0) for _ in range(21)]
    # Set wrist (0)
    landmarks[0] = create_mock_landmark(0.5, 0.7, 0.0)
    # Set thumb tip (4)
    landmarks[4] = create_mock_landmark(0.4, 0.4, 0.0)
    # Set index mcp (5) and pinky mcp (17)
    landmarks[5] = create_mock_landmark(0.45, 0.45, 0.0)
    landmarks[17] = create_mock_landmark(0.55, 0.45, 0.0)
    # Set index tip (8)
    landmarks[8] = create_mock_landmark(0.42, 0.4, 0.0)
    # Set middle mcp (9)
    landmarks[9] = create_mock_landmark(0.5, 0.4, 0.0)
    return landmarks


class TestHandFeatures(unittest.TestCase):
    def setUp(self):
        self.landmarks = create_mock_hand_landmarks()

    def test_palm_center_normalized(self):
        x, y, z = get_palm_center(self.landmarks)
        # Verify coordinates are finite floats
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)
        # With raw midpoint at (0.5, 0.55), norm_x should be near 0
        self.assertAlmostEqual(x, 0.0, places=2)

    def test_pinch_distance(self):
        dist = get_pinch_distance(self.landmarks)
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 1.0)

    def test_hand_rotation(self):
        pitch, yaw, roll = get_hand_rotation(self.landmarks)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)

    def test_extract_hand_transform(self):
        transform = extract_hand_transform(self.landmarks)
        self.assertIn("position", transform)
        self.assertIn("rotation", transform)
        self.assertIn("scale", transform)
        self.assertIn("pinch_distance", transform)
        self.assertGreaterEqual(transform["scale"], 0.5)
        self.assertLessEqual(transform["scale"], 2.5)

    def test_transform_smoother(self):
        smoother = TransformSmoother(alpha=0.5)
        t1 = {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": 1.0,
            "pinch_distance": 0.1,
        }
        res1 = smoother.update(t1)
        self.assertEqual(res1["position"]["x"], 1.0)

        # Feed shifted position
        t2 = {
            "position": {"x": 2.0, "y": 2.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": 1.0,
            "pinch_distance": 0.1,
        }
        res2 = smoother.update(t2)
        # With alpha=0.5, (0.5*2 + 0.5*1) = 1.5
        self.assertAlmostEqual(res2["position"]["x"], 1.5, places=2)


if __name__ == "__main__":
    unittest.main()
