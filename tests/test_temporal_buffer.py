"""
MotionMind AI - Unit Tests: Temporal Feature Buffer
Tests sliding window capacity, instantaneous velocity calculation, speed, and trajectory displacement.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.temporal_buffer import TemporalLandmarkBuffer


class TestTemporalBuffer(unittest.TestCase):

    def setUp(self):
        self.buf = TemporalLandmarkBuffer(window_size=5)

    def test_empty_buffer_returns_zero_velocity(self):
        self.assertEqual(self.buf.get_velocity(), (0.0, 0.0, 0.0))
        self.assertEqual(self.buf.get_speed(), 0.0)

    def test_single_frame_returns_zero_velocity(self):
        self.buf.add_frame((0.5, 0.5, 0.0), timestamp=10.0)
        self.assertEqual(self.buf.get_velocity(), (0.0, 0.0, 0.0))
        self.assertEqual(self.buf.get_speed(), 0.0)

    def test_velocity_calculation(self):
        self.buf.add_frame((0.0, 0.0, 0.0), timestamp=0.0)
        self.buf.add_frame((1.0, 2.0, 0.0), timestamp=1.0)
        vx, vy, vz = self.buf.get_velocity()
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(vy, 2.0)
        self.assertAlmostEqual(vz, 0.0)

    def test_window_size_cap(self):
        for i in range(10):
            self.buf.add_frame((float(i), 0.0, 0.0), timestamp=float(i))
        self.assertEqual(len(self.buf.history), 5)
        self.assertEqual(self.buf.history[0], (5.0, 0.0, 0.0))
        self.assertEqual(self.buf.history[-1], (9.0, 0.0, 0.0))

    def test_net_displacement(self):
        self.buf.add_frame((1.0, 1.0, 0.0), timestamp=1.0)
        self.buf.add_frame((2.0, 3.0, 0.0), timestamp=2.0)
        dx, dy, dz = self.buf.get_trajectory_displacement()
        self.assertAlmostEqual(dx, 1.0)
        self.assertAlmostEqual(dy, 2.0)


if __name__ == "__main__":
    unittest.main()
