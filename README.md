# MotionMind AI

An intelligent, gesture-based human-computer interaction (HCI) system combining Computer Vision, Machine Learning, and 3D Graphics.

## Pipeline Architecture
```
Webcam 
  -> Perception (MediaPipe HandLandmarker Tasks API) 
  -> Feature Extraction (Palm center, pinch distance, rotation) 
  -> Transform Smoothing (Exponential Moving Average Filter) 
  -> Transport (Low-Latency WebSocket Streaming) 
  -> 3D Engine (Three.js WebGL with smooth Lerp interpolation)
```

## Tech Stack
- **Perception:** OpenCV, MediaPipe (Tasks API)
- **Features & Math:** NumPy, Exponential Moving Average (EMA)
- **Transport:** WebSockets (asyncio, daemon thread)
- **Response & 3D:** Three.js (WebGL, ES Modules)

---

## Quickstart Guide

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```powershell
python -m unittest tests/test_features.py tests/test_gesture_dataset.py
```

### 3. Run Level 1 (Hand-Controlled 3D Cube)

**Terminal 1 — Start Web Server:**
```powershell
python -m http.server 8000
```
Open **[http://localhost:8000/src/web/](http://localhost:8000/src/web/)** in your browser.

**Terminal 2 — Start Hand Tracker & WebSocket Broadcaster:**
```powershell
python src/tracking/hand_tracker.py
```

### Controls:
- **Move Hand:** Translates 3D cube along X, Y, Z space.
- **Tilt Hand:** Rotates 3D cube (pitch, yaw, roll).
- **Pinch Fingers:** Scales 3D cube in real-time.
- **Press 'q':** Exits tracking window cleanly.

## Collect a Gesture Dataset (Level 2)

The collector saves numeric landmark data locally; it does not save webcam frames
or video. Each sample is wrist-relative and scaled by the wrist-to-middle-finger
length, reducing sensitivity to hand position and size.

Run the collector once for each gesture label:
```powershell
python src/ml/collect_gestures.py --label open_palm --samples 100
python src/ml/collect_gestures.py --label fist --samples 100
```

Samples are appended to `data/gesture_landmarks.csv` by default. Press `q` to stop
early. Keep the same label for the same gesture each time, and vary lighting and
viewing conditions between sessions. The `data/` directory is ignored by Git so
personal samples are not pushed to the repository.
