"""
MotionMind AI - Step 2 & 3: Hand Landmarks with Real-Time FPS Counter
Tracks hands using MediaPipe HandLandmarker Tasks API and visualizes 21 landmarks + live FPS.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path for the MediaPipe task model bundle
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_exists():
    """Ensure the hand landmarker model file is downloaded and ready."""
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Model downloaded successfully.")


def create_hand_landmarker():
    """Initialize and return the MediaPipe HandLandmarker in VIDEO running mode."""
    ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(frame, hand_landmarks_list):
    """Draw 21 hand landmarks and skeleton connections on the frame."""
    for hand_landmarks in hand_landmarks_list:
        vision.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=vision.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=vision.drawing_styles.get_default_hand_connections_style(),
        )


def draw_fps(frame, fps):
    """Draw a clean FPS overlay with a subtle dark background pill for high contrast."""
    text = f"FPS: {int(fps)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    x, y = 20, 45

    # Background rectangle for easy readability
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x - 8, y - text_h - 8),
        (x + text_w + 8, y + baseline + 4),
        (20, 20, 20),
        cv2.FILLED,
    )

    # Color coded: green if >= 30 FPS, yellow/orange if below 30 FPS
    color = (0, 255, 0) if fps >= 30 else (0, 200, 255)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not access webcam at index {camera_index}.")
        return

    print("Initializing MediaPipe HandLandmarker...")
    landmarker = create_hand_landmarker()
    print("Hand tracker started. Press 'q' in the window to quit.")

    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Mirror horizontally for natural, intuitive interaction
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Current timestamp in milliseconds (monotonically increasing)
        timestamp_ms = int(time.time() * 1000)

        # Detect hand landmarks synchronously for the current video frame
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # If any hands are detected, draw landmarks on the frame
        if detection_result.hand_landmarks:
            draw_hand_landmarks(frame, detection_result.hand_landmarks)

        # Calculate FPS using delta time
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            # Exponential smoothing to prevent erratic jitter in the counter
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

        # Draw the FPS counter on screen
        draw_fps(frame, fps)

        # Display output
        cv2.imshow("MotionMind AI - Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting hand tracking...")
            break

    # Clean up resources
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

