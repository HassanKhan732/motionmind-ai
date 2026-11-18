"""
MotionMind AI - Feature Engineering: Invariant Gesture Feature Extractor
Extracts translation-invariant, scale-invariant geometric feature vectors from 21 MediaPipe hand landmarks.
"""

import math
from typing import List, Sequence, Any


# Landmark indices for fingertips and base joints
FINGERTIP_INDICES = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
MCP_INDICES = [2, 5, 9, 13, 17]            # Base joints
PIP_INDICES = [3, 6, 10, 14, 18]           # Middle joints


def get_landmark_coords(lm: Any) -> tuple:
    """Safely extract (x, y, z) from either a landmark object or dict/tuple."""
    if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
        return float(lm.x), float(lm.y), float(lm.z)
    elif isinstance(lm, (dict, )):
        return float(lm["x"]), float(lm["y"]), float(lm["z"])
    elif isinstance(lm, (list, tuple)):
        return float(lm[0]), float(lm[1]), float(lm[2])
    raise ValueError(f"Unsupported landmark type: {type(lm)}")


def compute_euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def compute_joint_angle(p_a: tuple, p_b: tuple, p_c: tuple) -> float:
    """
    Calculate the 3D joint angle at vertex B formed by vector BA and vector BC.
    Returns angle in radians [0, pi].
    """
    v_ba = (p_a[0] - p_b[0], p_a[1] - p_b[1], p_a[2] - p_b[2])
    v_bc = (p_c[0] - p_b[0], p_c[1] - p_b[1], p_c[2] - p_b[2])

    dot = v_ba[0] * v_bc[0] + v_ba[1] * v_bc[1] + v_ba[2] * v_bc[2]
    mag_ba = math.sqrt(v_ba[0] ** 2 + v_ba[1] ** 2 + v_ba[2] ** 2)
    mag_bc = math.sqrt(v_bc[0] ** 2 + v_bc[1] ** 2 + v_bc[2] ** 2)

    if mag_ba * mag_bc == 0:
        return 0.0

    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.acos(cosine)


def extract_gesture_features(landmarks: Sequence[Any]) -> List[float]:
    """
    Transform raw 21 hand landmarks into a 75-dimensional invariant feature vector:
      - 60 relative normalized coordinates (landmarks 1..20 relative to wrist 0)
      - 5 fingertip-to-wrist normalized distances
      - 5 adjacent fingertip normalized distances
      - 5 finger joint articulation angles (extension status)
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected exactly 21 hand landmarks, received {len(landmarks)}")

    coords = [get_landmark_coords(lm) for lm in landmarks]
    wrist = coords[0]

    # 1. Translation Invariance: Shift all coordinates relative to wrist origin (0, 0, 0)
    relative_coords = [
        (c[0] - wrist[0], c[1] - wrist[1], c[2] - wrist[2])
        for c in coords
    ]

    # 2. Scale Invariance: Compute maximum spread distance from wrist to normalize scale
    max_dist = max(
        math.sqrt(rc[0] ** 2 + rc[1] ** 2 + rc[2] ** 2)
        for rc in relative_coords[1:]
    )
    if max_dist == 0:
        max_dist = 1.0  # Prevent zero division

    normalized_coords = [
        (rc[0] / max_dist, rc[1] / max_dist, rc[2] / max_dist)
        for rc in relative_coords
    ]

    features: List[float] = []

    # A. 60 relative normalized coordinates (landmarks 1 through 20)
    for nc in normalized_coords[1:]:
        features.extend([nc[0], nc[1], nc[2]])

    # B. 5 Fingertip to wrist normalized distances
    for tip_idx in FINGERTIP_INDICES:
        tip_coord = normalized_coords[tip_idx]
        dist = math.sqrt(tip_coord[0] ** 2 + tip_coord[1] ** 2 + tip_coord[2] ** 2)
        features.append(dist)

    # C. 5 Inter-fingertip normalized distances
    # Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky, Thumb-Pinky
    tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]
    for idx_a, idx_b in tip_pairs:
        d = compute_euclidean_distance(normalized_coords[idx_a], normalized_coords[idx_b])
        features.append(d)

    # D. 5 Finger Joint Articulation Angles (MCP -> PIP -> TIP)
    for mcp_idx, pip_idx, tip_idx in zip(MCP_INDICES, PIP_INDICES, FINGERTIP_INDICES):
        angle = compute_joint_angle(coords[mcp_idx], coords[pip_idx], coords[tip_idx])
        features.append(angle)

    return features
