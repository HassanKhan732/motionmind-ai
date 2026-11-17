"""Capture labeled, local-only hand landmark samples from a webcam."""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.gesture_dataset import append_sample
from src.tracking.hand_tracker import create_hand_landmarker, draw_hand_landmarks


def positive_int(value):
    """Parse a positive integer for command-line options."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record labeled hand landmarks to a local CSV file."
    )
    parser.add_argument("--label", required=True, help="Gesture label, e.g. open_palm")
    parser.add_argument(
        "--output",
        default="data/gesture_landmarks.csv",
        help="CSV file to append samples to (default: data/gesture_landmarks.csv)",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=100,
        help="Number of samples to collect (default: 100)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--interval-ms",
        type=positive_int,
        default=250,
        help="Minimum time between samples (default: 250 ms)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.label.strip():
        raise SystemExit("Error: --label cannot be empty.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Error: Could not access webcam at index {args.camera}.")

    landmarker = None
    samples_collected = 0
    last_sample_time = 0.0
    interval_seconds = args.interval_ms / 1000.0

    try:
        landmarker = create_hand_landmarker()
        print(f"Collecting '{args.label}' samples into {args.output}")
        print("Hold the gesture in view. Press 'q' to stop early.")

        while samples_collected < args.samples:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            now = time.monotonic()
            result = landmarker.detect_for_video(mp_image, int(now * 1000))

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                draw_hand_landmarks(frame, [hand_landmarks])

                if now - last_sample_time >= interval_seconds:
                    append_sample(args.output, args.label, hand_landmarks)
                    samples_collected += 1
                    last_sample_time = now

            cv2.putText(
                frame,
                f"{args.label}: {samples_collected}/{args.samples}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("MotionMind AI - Gesture Dataset", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(f"Saved {samples_collected} sample(s) to {args.output}.")
    finally:
        if landmarker is not None:
            landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
