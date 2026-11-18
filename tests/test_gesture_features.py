"""
MotionMind AI - Unit Tests: Gesture Feature Extraction
Verifies invariant feature vector shape, translation invariance, and scale invariance.
"""

import sys
import os
import unittest
import math
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.gesture_features import (
    extract_gesture_features,
    compute_euclidean_distance,
    compute_joint_angle,
    get_landmark_coords,
)


def make_lm(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def make_default_hand():
    """Create a 21-landmark hand with simple spread layout."""
    lms = []
    for i in range(21):
        lms.append(make_lm(0.5 + i * 0.01, 0.8 - i * 0.02, -0.001 * i))
    return lms


class TestGestureFeatures(unittest.TestCase):

    def test_feature_vector_length(self):
        hand = make_default_hand()
        feats = extract_gesture_features(hand)
        self.assertEqual(len(feats), 75)

    def test_all_features_finite(self):
        hand = make_default_hand()
        feats = extract_gesture_features(hand)
        for f in feats:
            self.assertTrue(math.isfinite(f), f"Non-finite feature found: {f}")

    def test_translation_invariance(self):
        hand_a = make_default_hand()
        hand_b = [make_lm(lm.x + 0.3, lm.y - 0.2, lm.z + 0.1) for lm in hand_a]
        feats_a = extract_gesture_features(hand_a)
        feats_b = extract_gesture_features(hand_b)
        for fa, fb in zip(feats_a, feats_b):
            self.assertAlmostEqual(fa, fb, places=4)

    def test_scale_invariance(self):
        hand_a = make_default_hand()
        wrist = hand_a[0]
        hand_b = [
            make_lm(
                wrist.x + (lm.x - wrist.x) * 2.0,
                wrist.y + (lm.y - wrist.y) * 2.0,
                wrist.z + (lm.z - wrist.z) * 2.0,
            )
            for lm in hand_a
        ]
        feats_a = extract_gesture_features(hand_a)
        feats_b = extract_gesture_features(hand_b)
        for fa, fb in zip(feats_a[:60], feats_b[:60]):
            self.assertAlmostEqual(fa, fb, places=4)

    def test_euclidean_distance(self):
        d = compute_euclidean_distance((0, 0, 0), (3, 4, 0))
        self.assertAlmostEqual(d, 5.0, places=5)

    def test_joint_angle_straight(self):
        angle = compute_joint_angle((0, 0, 0), (1, 0, 0), (2, 0, 0))
        self.assertAlmostEqual(angle, math.pi, places=4)

    def test_joint_angle_right(self):
        angle = compute_joint_angle((0, 0, 0), (0, 1, 0), (1, 1, 0))
        self.assertAlmostEqual(angle, math.pi / 2, places=4)

    def test_wrong_landmark_count_raises(self):
        with self.assertRaises(ValueError):
            extract_gesture_features([make_lm(0, 0, 0)] * 10)


if __name__ == "__main__":
    unittest.main()
