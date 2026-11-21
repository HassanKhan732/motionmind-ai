"""
MotionMind AI - Machine Learning: Model Robustness & Stress-Testing Suite
Tests the trained gesture classifier under geometric rotations, scaling, noise, and simulated occlusions.
"""

import os
import sys
import copy
import json
import random
from typing import Dict, List, Any
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.gesture_features import extract_gesture_features
from src.ml.gestures import GESTURE_CLASSES, GESTURE_TO_ID
from src.ml.dataset_generator import (
    generate_canonical_gesture,
    apply_3d_rotation,
)


def perturb_rotation(landmarks: List[List[float]], max_deg: float = 25.0) -> List[List[float]]:
    """Apply random 3D rotations up to +/- max_deg degrees."""
    pts = np.array(landmarks, dtype=np.float32)
    yaw = random.uniform(-max_deg, max_deg)
    pitch = random.uniform(-max_deg, max_deg)
    roll = random.uniform(-max_deg, max_deg)
    rotated = apply_3d_rotation(pts, yaw, pitch, roll)
    return rotated.tolist()


def perturb_scale(landmarks: List[List[float]], min_s: float = 0.7, max_s: float = 1.35) -> List[List[float]]:
    """Simulate distance changes / hand size variations by scaling around wrist."""
    pts = np.array(landmarks, dtype=np.float32)
    scale = random.uniform(min_s, max_s)
    origin = pts[0].copy()
    scaled = origin + (pts - origin) * scale
    return scaled.tolist()


def perturb_jitter(landmarks: List[List[float]], sigma: float = 0.010) -> List[List[float]]:
    """Simulate camera sensor noise, poor lighting, or camera vibration."""
    pts = np.array(landmarks, dtype=np.float32)
    noise = np.random.normal(0.0, sigma, pts.shape)
    jittered = pts + noise
    return jittered.tolist()


def perturb_combined(landmarks: List[List[float]]) -> List[List[float]]:
    """Simulate complex real-world conditions (rotation + scale + jitter)."""
    h = perturb_rotation(landmarks, max_deg=20.0)
    h = perturb_scale(h, min_s=0.75, max_s=1.25)
    return perturb_jitter(h, sigma=0.008)


def evaluate_condition(
    model,
    condition_name: str,
    samples_per_class: int = 40,
    noise_fn=None
) -> Dict[str, Any]:
    """
    Evaluate model accuracy under a specific perturbation condition using fast vectorized batch prediction.
    """
    all_features = []
    ground_truth = []
    class_indices = []

    for gesture_name in GESTURE_CLASSES:
        expected_id = GESTURE_TO_ID[gesture_name]
        for _ in range(samples_per_class):
            base_hand = generate_canonical_gesture(gesture_name)
            perturbed_hand = copy.deepcopy(base_hand)

            if noise_fn:
                perturbed_hand = noise_fn(perturbed_hand)

            features = extract_gesture_features(perturbed_hand)
            all_features.append(features)
            ground_truth.append(expected_id)
            class_indices.append(gesture_name)

    X_batch = np.array(all_features, dtype=np.float32)
    y_true = np.array(ground_truth, dtype=np.int32)
    y_pred = model.predict(X_batch)

    correct = int(np.sum(y_pred == y_true))
    total = len(y_true)

    per_class_correct = {name: 0 for name in GESTURE_CLASSES}
    for pred, true_cls, name in zip(y_pred, y_true, class_indices):
        if pred == true_cls:
            per_class_correct[name] += 1

    overall_acc = correct / total if total > 0 else 0.0
    return {
        "condition": condition_name,
        "total_tested": total,
        "correct": correct,
        "accuracy": round(overall_acc, 4),
        "per_class_accuracy": {
            k: round(v / samples_per_class, 3) for k, v in per_class_correct.items()
        }
    }


def run_robustness_suite(model_path: str = "models/gesture_classifier.joblib") -> Dict[str, Any]:
    """Execute all stress tests against the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)

    results = {
        "baseline_clean": evaluate_condition(model, "Clean Canonical Baseline", noise_fn=None),
        "rotation_stress": evaluate_condition(model, "3D Rotation (+/-25 deg)", noise_fn=perturb_rotation),
        "scale_stress": evaluate_condition(model, "Scale Variation (0.7x-1.35x)", noise_fn=perturb_scale),
        "jitter_stress": evaluate_condition(model, "Lighting / Sensor Jitter", noise_fn=perturb_jitter),
        "combined_stress": evaluate_condition(model, "Combined Real-World Stress", noise_fn=perturb_combined),
    }

    return results


def print_robustness_summary(results: Dict[str, Any]):
    """Display tabular stress-test metrics."""
    print("=" * 65)
    print(" MOTIONMIND AI - CLASSIFIER ROBUSTNESS & STRESS-TEST REPORT")
    print("=" * 65)
    print(f" {'Condition Tested':<35} | {'Accuracy':<10} | {'Status':<10}")
    print("-" * 65)

    all_passed = True
    for key, data in results.items():
        acc = data["accuracy"] * 100
        passed = acc >= 80.0
        if not passed:
            all_passed = False
        status = "EXCELLENT" if acc >= 92.0 else ("PASSED" if acc >= 80.0 else "MARGINAL")
        print(f" {data['condition']:<35} | {acc:>7.2f}%   | {status:<10}")

    print("=" * 65)
    overall_status = "ALL TESTS PASSED (Target >= 80% achieved under stress)" if all_passed else "SOME TESTS MARGINAL"
    print(f" VERDICT: {overall_status}")
    print("=" * 65)


if __name__ == "__main__":
    rep = run_robustness_suite()
    print_robustness_summary(rep)

    os.makedirs("docs", exist_ok=True)
    with open("docs/robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("Saved robustness report to docs/robustness_report.json")
