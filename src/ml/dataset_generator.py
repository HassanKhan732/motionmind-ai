"""
MotionMind AI - Machine Learning: Dataset Generator & Augmentation Pipeline
Generates robust, multi-person, multi-angle gesture datasets with 3D rotation, scaling, and landmark noise augmentation.
"""

import math
import random
from typing import List, Dict, Any
import numpy as np

from src.features.gesture_features import extract_gesture_features
from src.ml.gestures import GESTURE_CLASSES, GESTURE_TO_ID


def get_base_hand_skeleton() -> List[List[float]]:
    """
    Standard base resting hand landmark skeleton (21 points) in normalized space.
    Rooted at Wrist (0) at [0.5, 0.8, 0.0].
    """
    return [
        [0.50, 0.80, 0.00],  # 0: Wrist
        # Thumb: CMC(1), MCP(2), IP(3), TIP(4)
        [0.44, 0.74, -0.01], [0.39, 0.66, -0.02], [0.36, 0.58, -0.03], [0.33, 0.52, -0.04],
        # Index: MCP(5), PIP(6), DIP(7), TIP(8)
        [0.44, 0.58, -0.01], [0.43, 0.48, -0.02], [0.42, 0.40, -0.02], [0.41, 0.33, -0.03],
        # Middle: MCP(9), PIP(10), DIP(11), TIP(12)
        [0.50, 0.57, 0.00], [0.50, 0.46, -0.01], [0.50, 0.37, -0.02], [0.50, 0.29, -0.02],
        # Ring: MCP(13), PIP(14), DIP(15), TIP(16)
        [0.56, 0.58, 0.01], [0.57, 0.48, 0.01], [0.58, 0.41, 0.00], [0.58, 0.35, -0.01],
        # Pinky: MCP(17), PIP(18), DIP(19), TIP(20)
        [0.61, 0.62, 0.02], [0.63, 0.55, 0.02], [0.65, 0.49, 0.02], [0.66, 0.44, 0.01],
    ]


def curl_finger(landmarks: List[List[float]], mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int, curl_factor: float = 1.0):
    """Curl a finger toward the palm base."""
    mcp = np.array(landmarks[mcp_idx])
    wrist = np.array(landmarks[0])
    target_base = mcp * 0.7 + wrist * 0.3

    for idx, weight in [(pip_idx, 0.3), (dip_idx, 0.6), (tip_idx, 0.9)]:
        curr = np.array(landmarks[idx])
        curled = mcp + (target_base - mcp) * weight
        curled[2] += 0.04 * weight  # Fold slightly in depth Z
        landmarks[idx] = list(curr * (1.0 - curl_factor) + curled * curl_factor)


def generate_canonical_gesture(gesture_name: str) -> List[List[float]]:
    """Synthesize the accurate 3D landmark configuration for each gesture class."""
    skeleton = [list(pt) for pt in get_base_hand_skeleton()]

    if gesture_name == "Open Palm":
        pass  # All extended naturally

    elif gesture_name == "Fist":
        curl_finger(skeleton, 5, 6, 7, 8, 1.0)
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[4] = [0.45, 0.60, 0.02]  # Thumb folded over fingers

    elif gesture_name == "Peace":
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[4] = [0.46, 0.64, 0.01]  # Thumb tucked
        # Spread index and middle
        skeleton[8][0] -= 0.03
        skeleton[12][0] += 0.03

    elif gesture_name == "Thumbs Up":
        curl_finger(skeleton, 5, 6, 7, 8, 1.0)
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[1] = [0.47, 0.72, -0.01]
        skeleton[2] = [0.45, 0.63, -0.02]
        skeleton[3] = [0.43, 0.54, -0.03]
        skeleton[4] = [0.42, 0.44, -0.04]  # Thumb pointing straight UP

    elif gesture_name == "Thumbs Down":
        curl_finger(skeleton, 5, 6, 7, 8, 1.0)
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[4] = [0.44, 0.88, -0.03]  # Thumb pointing straight DOWN

    elif gesture_name == "OK Sign":
        curl_finger(skeleton, 13, 14, 15, 16, 0.1)  # Ring slightly open
        # Touch thumb tip to index tip
        touch_point = [0.38, 0.46, -0.02]
        skeleton[4] = list(touch_point)
        skeleton[8] = [touch_point[0] + 0.01, touch_point[1] + 0.01, touch_point[2]]

    elif gesture_name == "Pointing":
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[4] = [0.46, 0.65, 0.02]  # Thumb tucked
        # Index extended firmly

    elif gesture_name == "Rock Horns":
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)  # Middle curled
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)  # Ring curled
        skeleton[4] = [0.48, 0.62, 0.02]  # Thumb holds curled fingers

    elif gesture_name == "Call Me":
        curl_finger(skeleton, 5, 6, 7, 8, 1.0)    # Index curled
        curl_finger(skeleton, 9, 10, 11, 12, 1.0)  # Middle curled
        curl_finger(skeleton, 13, 14, 15, 16, 1.0) # Ring curled
        skeleton[4][0] -= 0.06  # Thumb extended outward
        skeleton[20][0] += 0.06  # Pinky extended outward

    elif gesture_name == "Hand Gun":
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)
        skeleton[4] = [0.42, 0.50, -0.03]  # Thumb up
        skeleton[8] = [0.35, 0.40, -0.03]  # Index pointing forward

    elif gesture_name == "Pinch":
        curl_finger(skeleton, 9, 10, 11, 12, 0.7)
        curl_finger(skeleton, 13, 14, 15, 16, 0.8)
        curl_finger(skeleton, 17, 18, 19, 20, 0.9)
        # Thumb and index tips close together
        skeleton[4] = [0.43, 0.44, 0.0]
        skeleton[8] = [0.44, 0.43, 0.0]

    elif gesture_name == "Three":
        curl_finger(skeleton, 13, 14, 15, 16, 1.0)  # Ring curled
        curl_finger(skeleton, 17, 18, 19, 20, 1.0)  # Pinky curled
        # Thumb, Index, Middle extended

    elif gesture_name == "Four":
        skeleton[4] = [0.49, 0.68, 0.02]  # Thumb folded into palm
        # All 4 fingers extended

    elif gesture_name == "Stop Sign":
        for i in range(21):
            skeleton[i][1] -= 0.05  # Raised slightly
            skeleton[i][2] -= 0.03  # Tilted forward perpendicular to camera

    elif gesture_name == "PSL Salaam":
        # Flat hand held with fingers tight and thumb alongside index
        skeleton[4] = [0.42, 0.52, -0.01]
        for idx in [8, 12, 16, 20]:
            skeleton[idx][0] += 0.02  # Closed finger spread

    elif gesture_name == "PSL Shukriya":
        # Hand cupped gently forward, thumb relaxed
        curl_finger(skeleton, 5, 6, 7, 8, 0.25)
        curl_finger(skeleton, 9, 10, 11, 12, 0.25)
        curl_finger(skeleton, 13, 14, 15, 16, 0.25)
        curl_finger(skeleton, 17, 18, 19, 20, 0.25)

    return skeleton


def apply_3d_rotation(points: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotate 3D landmarks around the wrist origin."""
    origin = points[0].copy()
    shifted = points - origin

    # Convert degrees to radians
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Rotation matrices
    R_z = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    R = R_z @ R_y @ R_x
    rotated = (R @ shifted.T).T + origin
    return rotated


def augment_skeleton(skeleton: List[List[float]]) -> List[List[float]]:
    """
    Apply comprehensive realistic variations:
    - 3D rotations (roll +/-30 deg, yaw +/-25 deg, pitch +/-20 deg)
    - Scale variations (0.75x to 1.25x)
    - Translation shifts (+/-0.15)
    - Sensor jitter noise
    """
    pts = np.array(skeleton, dtype=np.float32)

    # 1. 3D Rotation
    yaw = random.uniform(-25.0, 25.0)
    pitch = random.uniform(-20.0, 20.0)
    roll = random.uniform(-30.0, 30.0)
    pts = apply_3d_rotation(pts, yaw, pitch, roll)

    # 2. Scale variation around wrist
    scale = random.uniform(0.75, 1.25)
    origin = pts[0].copy()
    pts = origin + (pts - origin) * scale

    # 3. Translation
    dx = random.uniform(-0.15, 0.15)
    dy = random.uniform(-0.15, 0.15)
    dz = random.uniform(-0.10, 0.10)
    pts[:, 0] += dx
    pts[:, 1] += dy
    pts[:, 2] += dz

    # 4. Realistic landmark jitter
    noise = np.random.normal(0.0, 0.006, pts.shape)
    pts += noise

    return pts.tolist()


def generate_augmented_dataset(samples_per_class: int = 350) -> (np.ndarray, np.ndarray):
    """
    Synthesize a rich, balanced dataset across all 16 gesture classes.
    Returns (X_features, y_labels).
    """
    all_features = []
    all_labels = []

    print(f"Generating dataset: {len(GESTURE_CLASSES)} classes, {samples_per_class} samples each...")

    for gesture_idx, gesture_name in enumerate(GESTURE_CLASSES):
        canonical = generate_canonical_gesture(gesture_name)

        for _ in range(samples_per_class):
            augmented = augment_skeleton(canonical)
            feats = extract_gesture_features(augmented)
            all_features.append(feats)
            all_labels.append(gesture_idx)

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)
