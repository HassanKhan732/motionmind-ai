"""
MotionMind AI - Feature Module: Hand Features
Calculates spatial position, orientation angles, and pinch distance from 21 hand landmarks.
Supports MediaPipe landmark objects, dicts, and list/tuple coordinates.
"""

import math
from typing import Sequence, Any, Tuple


def _get_xyz(lm: Any) -> Tuple[float, float, float]:
    """Safely extract (x, y, z) from MediaPipe Landmark, dict, or list/tuple."""
    if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
        return float(lm.x), float(lm.y), float(lm.z)
    elif isinstance(lm, dict):
        return float(lm["x"]), float(lm["y"]), float(lm["z"])
    elif isinstance(lm, (list, tuple)):
        return float(lm[0]), float(lm[1]), float(lm[2])
    raise ValueError(f"Unsupported landmark type: {type(lm)}")


def get_palm_center(landmarks: Sequence[Any]) -> Tuple[float, float, float]:
    """
    Calculate the center point of the palm using Wrist (0) and Middle MCP (9).
    Returns normalized coordinates mapped to [-1.0, 1.0] where (0, 0) is the screen center.
    """
    w_x, w_y, w_z = _get_xyz(landmarks[0])
    m_x, m_y, m_z = _get_xyz(landmarks[9])

    # Average point between wrist and middle knuckle
    raw_x = (w_x + m_x) / 2.0
    raw_y = (w_y + m_y) / 2.0
    raw_z = (w_z + m_z) / 2.0

    # Invert X for mirroring, and invert Y so moving hand up increases Y
    norm_x = (raw_x - 0.5) * 2.0
    norm_y = -(raw_y - 0.5) * 2.0
    norm_z = -raw_z * 2.0

    return norm_x, norm_y, norm_z


def get_pinch_distance(landmarks: Sequence[Any]) -> float:
    """
    Calculate Euclidean distance between Thumb Tip (4) and Index Finger Tip (8).
    Used as an intuitive gesture to scale or manipulate 3D objects.
    """
    t_x, t_y, t_z = _get_xyz(landmarks[4])
    i_x, i_y, i_z = _get_xyz(landmarks[8])

    dx = t_x - i_x
    dy = t_y - i_y
    dz = t_z - i_z

    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance


def get_hand_rotation(landmarks: Sequence[Any]) -> Tuple[float, float, float]:
    """
    Calculate rotation angles (pitch, yaw, roll) in radians from landmark geometry.
    - Roll: Angle formed between Wrist (0) and Middle MCP (9) in 2D image plane.
    - Pitch/Yaw: Approximated using relative Z and X disparities.
    """
    w_x, w_y, w_z = _get_xyz(landmarks[0])
    m_x, m_y, m_z = _get_xyz(landmarks[9])
    i_x, i_y, i_z = _get_xyz(landmarks[5])
    p_x, p_y, p_z = _get_xyz(landmarks[17])

    # Roll (tilt left/right): angle of wrist-to-middle vector
    dx = m_x - w_x
    dy = m_y - w_y
    # Invert dy because image coordinates have top-left as (0,0)
    roll = math.atan2(-dy, dx) - (math.pi / 2.0)

    # Yaw (turn left/right): depth difference between index and pinky MCP
    yaw = (p_z - i_z) * 3.0

    # Pitch (tilt up/down): depth difference between wrist and middle MCP
    pitch = (w_z - m_z) * 3.0

    return pitch, yaw, roll


def extract_hand_transform(landmarks: Sequence[Any]) -> dict:
    """
    Extract a comprehensive transform dictionary ready for 3D engine consumption.
    """
    pos_x, pos_y, pos_z = get_palm_center(landmarks)
    pitch, yaw, roll = get_hand_rotation(landmarks)
    pinch_dist = get_pinch_distance(landmarks)

    # Map pinch distance (typically 0.03 to 0.25) to a scale factor (0.5 to 2.2)
    # Clamp to avoid extreme sizes
    raw_scale = 0.5 + (pinch_dist * 6.0)
    clamped_scale = max(0.5, min(2.5, raw_scale))

    return {
        "position": {
            "x": round(pos_x * 2.5, 3),  # Scale to 3D scene space
            "y": round(pos_y * 1.8, 3),
            "z": round(pos_z, 3),
        },
        "rotation": {
            "x": round(pitch, 3),
            "y": round(yaw, 3),
            "z": round(roll, 3),
        },
        "scale": round(clamped_scale, 3),
        "pinch_distance": round(pinch_dist, 3),
    }
