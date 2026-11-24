"""
MotionMind AI - Step 2, 3 & 5: Hand Landmarks, FPS, and WebSocket 3D Streaming
Tracks hands using MediaPipe HandLandmarker Tasks API and streams smoothed 3D transform telemetry.
"""

import os
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure project root is on sys.path for direct module execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.hand_features import extract_hand_transform
from src.features.smoothing import TransformSmoother
from src.tracking.websocket_server import HandStreamServer
from src.ml.gesture_classifier import RealTimeGestureClassifier
from src.ml.gesture_actions import GestureActionDispatcher

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


def draw_hud(frame, fps, is_tracking, client_count, transform=None, gesture_info=None):
    """Draw modern telemetry HUD overlay on OpenCV frame."""
    # 1. FPS Badge
    fps_text = f"FPS: {int(fps)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    x, y = 20, 35

    (w, h), base = cv2.getTextSize(fps_text, font, scale, thickness)
    cv2.rectangle(frame, (x - 6, y - h - 6), (x + w + 6, y + base + 4), (20, 20, 20), cv2.FILLED)
    color = (0, 255, 0) if fps >= 30 else (0, 200, 255)
    cv2.putText(frame, fps_text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # 2. WebSocket Status Badge
    ws_text = f"WS: {client_count} client{'s' if client_count != 1 else ''}"
    ws_x = x + w + 18
    (ws_w, ws_h), ws_base = cv2.getTextSize(ws_text, font, 0.55, 1)
    cv2.rectangle(frame, (ws_x - 6, y - ws_h - 6), (ws_x + ws_w + 6, y + ws_base + 4), (20, 20, 20), cv2.FILLED)
    ws_color = (0, 255, 200) if client_count > 0 else (120, 120, 120)
    cv2.putText(frame, ws_text, (ws_x, y - 2), font, 0.55, ws_color, 1, cv2.LINE_AA)

    # 3. Gesture Recognition Badge
    if gesture_info:
        g_name = gesture_info[0]
        g_conf = gesture_info[1]
        g_stable = gesture_info[2]
        g_action = gesture_info[3] if len(gesture_info) > 3 else "IDLE"
        badge_color = (0, 255, 0) if g_stable else (0, 200, 255)
        stability_tag = "STABLE" if g_stable else "..."
        gesture_text = f"Gesture: {g_name} ({g_conf*100:.0f}%) [{stability_tag}] | Act: {g_action}"
        cv2.putText(frame, gesture_text, (20, 70), font, 0.60, badge_color, 2, cv2.LINE_AA)

    # 4. Hand Tracking & Transform Telemetry
    if is_tracking and transform:
        pos = transform["position"]
        scale_val = transform["scale"]
        telemetry = f"Pos: ({pos['x']:+.2f}, {pos['y']:+.2f}) | Scale: {scale_val:.2f}x"
        cv2.putText(frame, telemetry, (20, frame.shape[0] - 20), font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Waiting for hand in view...", (20, frame.shape[0] - 20), font, 0.55, (100, 100, 255), 1, cv2.LINE_AA)


def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not access webcam at index {camera_index}.")
        return

    print("Initializing MediaPipe HandLandmarker...")
    landmarker = create_hand_landmarker()

    # Initialize and start WebSocket server for Three.js bridge
    ws_server = HandStreamServer(host="localhost", port=8765)
    ws_server.start()

    smoother = TransformSmoother(alpha=0.35)
    action_dispatcher = GestureActionDispatcher(default_cooldown_sec=1.0)

    # Initialize gesture classifier (graceful fallback if model not found)
    gesture_clf = None
    try:
        gesture_clf = RealTimeGestureClassifier()
        print("Gesture classifier loaded successfully.")
    except FileNotFoundError:
        print("Warning: Gesture model not found. Running without gesture recognition.")

    print("Hand tracker started. Streaming to ws://localhost:8765")
    print("Press 'q' in the window to quit.")

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

        # Timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Detect hand landmarks
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        is_tracking = bool(detection_result.hand_landmarks)
        current_transform = None
        gesture_info = None

        if is_tracking:
            # Draw skeleton landmarks
            draw_hand_landmarks(frame, detection_result.hand_landmarks)

            # Extract spatial transform from primary detected hand
            primary_hand = detection_result.hand_landmarks[0]
            raw_transform = extract_hand_transform(primary_hand)
            current_transform = smoother.update(raw_transform)

            # Run gesture prediction if classifier is loaded
            if gesture_clf:
                g_name, g_conf, g_stable = gesture_clf.predict(primary_hand)
                triggered_action = action_dispatcher.process_gesture(g_name, g_stable, g_conf)
                active_action = triggered_action or (action_dispatcher.get_action_for_gesture(g_name) if g_stable else "IDLE")
                gesture_info = (g_name, g_conf, g_stable, active_action)

            # Stream smoothed payload to Three.js frontend
            payload = {
                "is_tracking": True,
                "transform": current_transform,
                "fps": round(fps, 1),
            }
            if gesture_info:
                payload["gesture"] = {
                    "name": gesture_info[0],
                    "confidence": gesture_info[1],
                    "stable": gesture_info[2],
                    "action": gesture_info[3],
                }
            ws_server.broadcast(payload)
        else:
            smoother.reset()
            if gesture_clf:
                gesture_clf.reset()
            ws_server.broadcast({
                "is_tracking": False,
                "fps": round(fps, 1),
            })

        # Calculate FPS
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

        # Draw HUD overlays
        client_count = len(ws_server.clients)
        draw_hud(frame, fps, is_tracking, client_count, current_transform, gesture_info)

        # Display window
        cv2.imshow("MotionMind AI - Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting hand tracking...")
            break

    # Clean up resources
    ws_server.stop()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


