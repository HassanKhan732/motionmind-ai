"""
MotionMind AI - Performance Benchmarking: Latency & Throughput Profiler
Measures per-frame feature extraction, ML inference latency, debounce filtering, and theoretical max FPS.
Validates system against targets: > 30 FPS and < 50 ms latency.
"""

import os
import sys
import time
import json
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.gesture_features import extract_gesture_features
from src.ml.gesture_classifier import RealTimeGestureClassifier
from src.ml.dataset_generator import get_base_hand_skeleton


def benchmark_pipeline(num_iterations: int = 500) -> Dict[str, Any]:
    """
    Profile the entire perception-to-inference pipeline over hundreds of frames.
    """
    model_path = "models/gesture_classifier.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model missing at: {model_path}")

    clf = RealTimeGestureClassifier(model_path=model_path, history_len=5, min_confidence=0.60)
    # Set n_jobs=1 for low-latency single sample inference
    if hasattr(clf.model, "n_jobs"):
        clf.model.n_jobs = 1
    raw_landmarks = get_base_hand_skeleton()

    # 1. Warm-up run
    for _ in range(10):
        clf.predict(raw_landmarks)

    # 2. Benchmark Feature Extraction
    feat_latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        feats = extract_gesture_features(raw_landmarks)
        t1 = time.perf_counter()
        feat_latencies.append((t1 - t0) * 1000.0)  # ms

    # 3. Benchmark Classifier Inference alone
    inf_latencies = []
    features_arr = np.array(feats, dtype=np.float32)
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        clf.model.predict_proba([features_arr])
        t1 = time.perf_counter()
        inf_latencies.append((t1 - t0) * 1000.0)  # ms

    # 4. Benchmark Full Pipeline (Landmarks -> Features -> Model -> Temporal Debounce)
    pipeline_latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        clf.predict(raw_landmarks)
        t1 = time.perf_counter()
        pipeline_latencies.append((t1 - t0) * 1000.0)  # ms

    def stats(arr):
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
        }

    mean_pipe_ms = float(np.mean(pipeline_latencies))
    theoretical_fps = round(1000.0 / mean_pipe_ms if mean_pipe_ms > 0 else 0.0, 1)

    # Target verifications
    target_fps_met = theoretical_fps >= 30.0
    target_latency_met = mean_pipe_ms < 50.0

    report = {
        "iterations": num_iterations,
        "feature_extraction": stats(feat_latencies),
        "classifier_inference": stats(inf_latencies),
        "end_to_end_pipeline": stats(pipeline_latencies),
        "theoretical_max_fps": theoretical_fps,
        "targets": {
            "target_fps": 30.0,
            "achieved_fps": theoretical_fps,
            "fps_target_passed": target_fps_met,
            "target_max_latency_ms": 50.0,
            "achieved_latency_ms": round(mean_pipe_ms, 3),
            "latency_target_passed": target_latency_met,
        }
    }
    return report


def print_benchmark_report(report: Dict[str, Any]):
    """Print formatted performance profiling summary."""
    print("=" * 65)
    print(" MOTIONMIND AI - LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 65)
    print(f" Sample Count Tested  : {report['iterations']} frames")
    print("-" * 65)
    print(f" {'Subsystem Stage':<26} | {'Mean (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 65)
    print(f" Feature Extraction        | {report['feature_extraction']['mean_ms']:>8.3f} ms | {report['feature_extraction']['p95_ms']:>8.3f} ms")
    print(f" ML Inference (Probabilities)| {report['classifier_inference']['mean_ms']:>8.3f} ms | {report['classifier_inference']['p95_ms']:>8.3f} ms")
    print(f" End-to-End Prediction     | {report['end_to_end_pipeline']['mean_ms']:>8.3f} ms | {report['end_to_end_pipeline']['p95_ms']:>8.3f} ms")
    print("-" * 65)
    print(f" Max Theoretical Engine FPS : {report['theoretical_max_fps']} FPS")
    print(f" Latency Target (< 50ms)    : {'PASSED' if report['targets']['latency_target_passed'] else 'FAILED'}")
    print(f" FPS Target (>= 30 FPS)     : {'PASSED' if report['targets']['fps_target_passed'] else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    rep = benchmark_pipeline(num_iterations=100)
    print_benchmark_report(rep)

    os.makedirs("docs", exist_ok=True)
    with open("docs/latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("Saved benchmark metrics to docs/latency_benchmark.json")
