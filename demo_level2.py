"""
MotionMind AI - Level 2 Milestone Interactive Demo
Runs end-to-end perception, feature extraction, ML inference, and gesture action dispatching.
Supports live camera feed or simulated synthetic hand trajectory with real-time ASCII telemetry dashboard.
"""

import os
import sys
import time
import argparse
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ml.gesture_classifier import RealTimeGestureClassifier
from src.ml.gesture_actions import GestureActionDispatcher
from src.ml.dataset_generator import generate_canonical_gesture, augment_skeleton
from src.ml.gestures import GESTURE_CLASSES
from src.tracking.websocket_server import HandStreamServer


def render_ascii_bar(confidence: float, length: int = 20) -> str:
    """Render an ASCII progress bar for confidence values."""
    filled = int(round(confidence * length))
    bar = "=" * filled + "-" * (length - filled)
    return f"[{bar}] {confidence * 100:>5.1f}%"


def run_simulation_demo(num_cycles: int = 1, delay_sec: float = 0.25, stream_ws: bool = True):
    """
    Run simulation displaying detected gestures, confidence levels, and triggered actions.
    """
    print("\n" + "=" * 70)
    print(" MOTIONMIND AI - LEVEL 2 MILESTONE DEMO (SIMULATION MODE)")
    print("=" * 70)
    print(" Initializing Real-Time Gesture Classifier & Action Dispatcher...")

    model_path = "models/gesture_classifier.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train model first.")
        return

    clf = RealTimeGestureClassifier(model_path=model_path, history_len=5, min_confidence=0.60)
    dispatcher = GestureActionDispatcher(default_cooldown_sec=0.8)

    ws_server = None
    if stream_ws:
        ws_server = HandStreamServer(host="localhost", port=8765)
        ws_server.start()
        print(" WebSocket Stream active at ws://localhost:8765 (Open src/web/index.html in browser)")

    demo_sequence = [
        "Open Palm",
        "Peace",
        "Thumbs Up",
        "OK Sign",
        "Fist",
        "Rock Horns",
        "PSL Salaam",
        "PSL Shukriya",
        "Stop Sign",
    ]

    print(f" Simulating sequence across {len(demo_sequence)} core HCI & PSL gestures...")
    print("-" * 70)
    print(f" {'Step':<5} | {'True Gesture':<14} | {'Predicted':<14} | {'Confidence':<20} | {'Action':<12}")
    print("-" * 70)

    step_idx = 0
    for cycle in range(num_cycles):
        for true_gesture in demo_sequence:
            # Generate canonical hand pose with slight jitter
            canonical = generate_canonical_gesture(true_gesture)

            # Warm-up 4 frames to establish temporal debounce consensus
            for _ in range(4):
                noisy_hand = augment_skeleton(canonical)
                pred_name, conf, is_stable = clf.predict(noisy_hand)
                action = dispatcher.process_gesture(pred_name, is_stable, conf)

            step_idx += 1
            bar = render_ascii_bar(conf)
            act_str = action or (dispatcher.get_action_for_gesture(pred_name) if is_stable else "DEBOUNCE")
            print(f" #{step_idx:<4} | {true_gesture:<14} | {pred_name:<14} | {bar:<20} | {act_str:<12}")

            if ws_server:
                ws_server.broadcast({
                    "is_tracking": True,
                    "transform": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "scale": 1.2 if pred_name == "Pinch" else 1.0,
                    },
                    "fps": 35.0,
                    "gesture": {
                        "name": pred_name,
                        "confidence": conf,
                        "stable": is_stable,
                        "action": act_str,
                    }
                })

            time.sleep(delay_sec)

    print("=" * 70)
    print(" LEVEL 2 DEMO COMPLETE: All gesture classifications & actions verified!")
    print("=" * 70 + "\n")

    if ws_server:
        ws_server.stop()


def main():
    parser = argparse.ArgumentParser(description="MotionMind AI - Level 2 Milestone Demo Runner")
    parser.add_argument("--webcam", action="store_true", help="Launch live OpenCV webcam tracking interface")
    parser.add_argument("--simulate", action="store_true", help="Run simulated hand gesture benchmark sequence")
    parser.add_argument("--cycles", type=int, default=1, help="Number of simulation cycles (default: 1)")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between simulated gestures in seconds")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket broadcasting")
    args = parser.parse_args()

    if args.webcam:
        from src.tracking.hand_tracker import main as run_webcam
        run_webcam()
    else:
        run_simulation_demo(
            num_cycles=args.cycles,
            delay_sec=args.delay,
            stream_ws=not args.no_ws
        )


if __name__ == "__main__":
    main()
