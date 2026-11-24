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

## Level 2: Machine Learning & Gesture Intelligence (Completed)

Level 2 introduces a trained ensemble Random Forest classifier recognizing **16 custom static gestures** (including Pakistani Sign Language signs) with **94.88% accuracy**, achieving sub-20ms latency and 50 FPS throughput.

### 1. Run the Level 2 Interactive Demo
```powershell
# Run simulated benchmark sequence with ASCII telemetry & WebSocket stream:
python demo_level2.py --simulate

# Or run live with webcam:
python demo_level2.py --webcam
```

### 2. Unified CLI Interface
```powershell
python src/cli.py demo --simulate     # Run simulation demo
python src/cli.py track               # Launch webcam tracking & WebSocket server
python src/cli.py inspect             # Inspect dataset health and class balance
python src/cli.py benchmark           # Profile latency and max engine FPS
python src/cli.py robustness          # Run 3D rotation, scale & noise stress tests
python src/cli.py psl                 # View Pakistani Sign Language dictionary
```

### 3. Run Automated Tests
```powershell
python -m unittest discover tests
```

### 4. Milestone Evaluation Report
Detailed metrics, confusion matrix, feature formulas, and benchmark tables are available in:
- [docs/LEVEL_2_REPORT.md](file:///c:/Users/FUJITSU/Desktop/motionmind-ai/docs/LEVEL_2_REPORT.md)
- [docs/PSL_CURRICULUM.md](file:///c:/Users/FUJITSU/Desktop/motionmind-ai/docs/PSL_CURRICULUM.md)

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
