"""
MotionMind AI - Feature Module: Transform Smoother
Applies exponential moving average (EMA) to prevent jitter and provide organic, fluid 3D motion.
"""


class TransformSmoother:
    """
    Applies temporal exponential smoothing to 3D transform properties (position, rotation, scale).
    """

    def __init__(self, alpha=0.35):
        """
        Args:
            alpha (float): Smoothing weight [0.0 - 1.0].
                           Lower value = smoother motion, higher value = faster response.
        """
        self.alpha = alpha
        self.smoothed_state = None

    def reset(self):
        """Reset state when hand is lost from camera view."""
        self.smoothed_state = None

    def update(self, transform):
        """
        Smooth the incoming transform dictionary.
        Returns the smoothed transform with the same schema.
        """
        if self.smoothed_state is None:
            # First frame: initialize baseline state
            self.smoothed_state = {
                "position": dict(transform["position"]),
                "rotation": dict(transform["rotation"]),
                "scale": transform["scale"],
                "pinch_distance": transform["pinch_distance"],
            }
            return transform

        # Smooth position (X, Y, Z)
        for axis in ("x", "y", "z"):
            curr_pos = transform["position"][axis]
            prev_pos = self.smoothed_state["position"][axis]
            self.smoothed_state["position"][axis] = (
                self.alpha * curr_pos + (1.0 - self.alpha) * prev_pos
            )

        # Smooth rotation (Pitch, Yaw, Roll)
        for axis in ("x", "y", "z"):
            curr_rot = transform["rotation"][axis]
            prev_rot = self.smoothed_state["rotation"][axis]
            self.smoothed_state["rotation"][axis] = (
                self.alpha * curr_rot + (1.0 - self.alpha) * prev_rot
            )

        # Smooth scale & pinch
        curr_scale = transform["scale"]
        prev_scale = self.smoothed_state["scale"]
        self.smoothed_state["scale"] = (
            self.alpha * curr_scale + (1.0 - self.alpha) * prev_scale
        )

        curr_pinch = transform["pinch_distance"]
        prev_pinch = self.smoothed_state["pinch_distance"]
        self.smoothed_state["pinch_distance"] = (
            self.alpha * curr_pinch + (1.0 - self.alpha) * prev_pinch
        )

        return {
            "position": {
                k: round(v, 3) for k, v in self.smoothed_state["position"].items()
            },
            "rotation": {
                k: round(v, 3) for k, v in self.smoothed_state["rotation"].items()
            },
            "scale": round(self.smoothed_state["scale"], 3),
            "pinch_distance": round(self.smoothed_state["pinch_distance"], 3),
        }
