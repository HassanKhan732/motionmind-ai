"""
MotionMind AI - Temporal Feature Buffer: Sliding Window & Velocity Extractor
Tracks sequential hand landmark motions over time, calculating instantaneous velocity,
acceleration, and trajectory vectors for dynamic motion-based gesture intelligence.
"""

import collections
import math
import time
from typing import List, Tuple, Optional, Any
import numpy as np


class TemporalLandmarkBuffer:
    """
    Sliding window buffer for hand landmarks over N consecutive frames.
    Computes spatial velocity (dx/dt, dy/dt, dz/dt) and trajectory features.
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)
        self.timestamps = collections.deque(maxlen=window_size)

    def reset(self):
        """Clear all historical state."""
        self.history.clear()
        self.timestamps.clear()

    def add_frame(self, primary_wrist_coords: Tuple[float, float, float], timestamp: Optional[float] = None):
        """Append a 3D coordinate point to the sliding trajectory window."""
        t = timestamp if timestamp is not None else time.time()
        self.history.append(primary_wrist_coords)
        self.timestamps.append(t)

    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Calculate instantaneous 3D velocity vector (units per second) from the last two frames.
        """
        if len(self.history) < 2:
            return (0.0, 0.0, 0.0)

        p1 = self.history[-2]
        p2 = self.history[-1]
        dt = self.timestamps[-1] - self.timestamps[-2]

        if dt <= 0:
            return (0.0, 0.0, 0.0)

        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        vz = (p2[2] - p1[2]) / dt
        return (vx, vy, vz)

    def get_speed(self) -> float:
        """Calculate scalar magnitude of 3D velocity."""
        vx, vy, vz = self.get_velocity()
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def get_trajectory_displacement(self) -> Tuple[float, float, float]:
        """Calculate net displacement across the full sliding window."""
        if len(self.history) < 2:
            return (0.0, 0.0, 0.0)

        first_pt = self.history[0]
        last_pt = self.history[-1]
        return (
            last_pt[0] - first_pt[0],
            last_pt[1] - first_pt[1],
            last_pt[2] - first_pt[2]
        )

    def is_moving(self, threshold_speed: float = 0.15) -> bool:
        """Return True if hand speed exceeds the movement threshold."""
        return self.get_speed() > threshold_speed
