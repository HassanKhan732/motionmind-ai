# MotionMind AI

An intelligent, gesture-based human-computer interaction (HCI) system combining Computer Vision, Machine Learning, and 3D Graphics.

## Tech Stack
- **Perception:** OpenCV, MediaPipe
- **Intelligence:** scikit-learn, PyTorch
- **Response:** Three.js

## Project Setup

1. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run webcam test (Step 1):**
   ```powershell
   python src/tracking/webcam_test.py
   ```

4. **Run hand tracking (Step 2 & 3):**
   ```powershell
   python src/tracking/hand_tracker.py
   ```

5. **Run 3D Scene in browser (Step 4):**
   ```powershell
   python -m http.server 8000
   ```
   Open `http://localhost:8000/src/web/` in your browser.
