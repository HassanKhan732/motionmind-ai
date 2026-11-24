"""
MotionMind AI - Unit Tests: Level 2 Demo Runner
Tests ASCII progress bar formatting and simulation loop execution.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo_level2 import render_ascii_bar, run_simulation_demo


class TestDemoLevel2(unittest.TestCase):

    def test_ascii_bar_min(self):
        bar = render_ascii_bar(0.0, length=10)
        self.assertIn("[----------]", bar)
        self.assertIn("0.0%", bar)

    def test_ascii_bar_max(self):
        bar = render_ascii_bar(1.0, length=10)
        self.assertIn("[==========]", bar)
        self.assertIn("100.0%", bar)

    def test_ascii_bar_mid(self):
        bar = render_ascii_bar(0.5, length=10)
        self.assertIn("[=====-----]", bar)
        self.assertIn("50.0%", bar)

    def test_simulation_runs_without_crash(self):
        # Run 1 rapid cycle without opening WebSocket server
        try:
            run_simulation_demo(num_cycles=1, delay_sec=0.001, stream_ws=False)
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
