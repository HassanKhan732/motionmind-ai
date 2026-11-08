# MotionMind AI: Project Brief

> **Instructions for the AI coding assistant:** Read this entire document before doing anything.
> Follow the working agreement in Section 4 at all times.
> Begin only with the tasks in Section 12 ("Start Here"). Do not work ahead of the current step.

---

## 1. Overview

**MotionMind AI** is an intelligent, gesture-based human-computer interaction (HCI) system.
It combines three technology areas into one pipeline:

| Layer | Technology | Role |
|---|---|---|
| Perception | Computer Vision (OpenCV, MediaPipe) | Track hands, face and body from a webcam |
| Intelligence | Machine Learning (scikit-learn, PyTorch) | Recognize gestures using custom-trained models |
| Response | 3D Graphics (Three.js) | Render a 3D world that reacts to the user in real time |

**Pipeline:** Camera -> Tracking -> Feature extraction -> ML model -> Interaction logic -> 3D response

This is a personal, learning-driven project. It is not a final year project and must go well beyond a basic demo.

## 2. Objectives (in priority order)

1. **Learning:** The developer must genuinely understand every component (CV, ML, 3D, system design). Understanding is more important than speed.
2. **Portfolio quality:** The finished project must look professional: clean architecture, tests, documentation, measured results, a live demo and a demo video.
3. **Social impact:** In later phases, extend the system to recognize **Pakistani Sign Language (PSL)** to help children who cannot speak or hear communicate.

## 3. Product Principles

- **Own the data and the models.** Prefer custom datasets and trained models over ready-made black boxes.
- **Measure everything.** Report accuracy, precision/recall, confusion matrices, FPS and latency (targets: 30+ FPS, under 50 ms end-to-end latency).
- **Modular design.** Hand, face and body tracking must be separable so sign-language support can be added later without rewrites.
- **Incremental delivery.** Every level ends with something that runs, a git tag and a short demo.
- **Privacy by default.** Camera frames must be processed locally. Never store or transmit video without explicit user consent.

## 4. Working Agreement (mandatory rules for the AI assistant)

1. **One small step at a time.** Complete only the current step, then STOP and wait for the developer to say "next".
2. **Explain in simple Roman Urdu** (Urdu written in English letters). Keep code, file names, commands and technical terms in English.
3. **Before writing code,** state in 2 to 4 lines what will be done and why.
4. **After writing code,** explain each important part in simple words.
5. **Give the exact command to run** and describe what the correct output looks like.
6. **When an error occurs,** the developer will paste it. Explain the cause simply, then fix it.
7. **No scope creep.** Do not add features, files, dependencies or refactors that were not requested.
8. **Code quality:** clear names, short helpful comments, small functions, no clever one-liners that hurt readability. Follow PEP 8 for Python.
9. **Verify before installing.** Check library versions and Python compatibility (for example, MediaPipe supports only specific Python versions). Use the currently supported MediaPipe API, not deprecated ones.
10. **Environment hygiene:** use a virtual environment (`venv`) and keep `requirements.txt` up to date.
11. **Version control:** after each completed step, suggest a git commit with a clear message.
12. **Check understanding:** occasionally ask the developer one short question (never more than one at a time) to confirm they understood.
13. **Be honest:** if something is uncertain, risky or likely to fail, say so before proceeding.

## 5. Technology Stack

- **Language:** Python 3 (backend, ML) and JavaScript (frontend)
- **Computer Vision:** OpenCV, MediaPipe (Hands, Face, Pose)
- **Machine Learning:** NumPy, scikit-learn, later PyTorch (MLP, then LSTM/Transformer), ONNX export
- **Backend (later):** FastAPI with WebSockets
- **Frontend:** Three.js (browser), later a physics engine such as Rapier
- **Tooling:** Git, VS Code, a virtual environment, and experiment tracking (for example Weights & Biases) in later levels
- **Development environment:** Windows

## 6. System Architecture (target)

```
Webcam
  -> Tracking module (MediaPipe landmarks)
  -> Feature module (normalization, angles, distances)
  -> Gesture model (static classifier, then temporal model)
  -> Interaction logic (gesture -> action, smoothing, activation rules)
  -> Transport (WebSocket)
  -> 3D scene (Three.js)
```

## 7. Roadmap

Each level builds on the previous one. Do not start a level until the previous one is complete.

### Level 1: Foundation
- **Build:** Webcam capture, hand tracking, landmark visualization, then a basic 3D scene controlled by the hand.
- **Learn:** OpenCV basics, landmarks and coordinate systems, 3D fundamentals (scene, camera, lights, meshes).
- **Done when:** 21 hand landmarks are drawn live, a 3D cube moves/rotates/scales with the hand, FPS is displayed.

### Level 2: Machine Learning
- **Build:** A custom gesture dataset recorded by the developer, plus a trained classifier.
- **Learn:** Data collection, feature engineering, train/validation/test splits, overfitting, confusion matrices.
- **Done when:** 15 to 20 custom gestures are recognized with 95%+ accuracy, tested on more than one person and lighting condition.

### Level 3: Temporal Intelligence
- **Build:** Movement-based gestures (swipe, circle, wave) using LSTM or Transformer models, and the first small set of sign-language words.
- **Learn:** Sequence modelling, sliding windows, augmentation, real-time inference, ONNX export.
- **Done when:** The system recognizes both pose and motion in real time.

### Level 4: Advanced 3D Graphics
- **Build:** A rigged 3D avatar that mirrors body and face movement, with physics-based interaction.
- **Learn:** Skeletal animation, inverse kinematics, shaders (GLSL), physics engines, exporting models from Blender.
- **Done when:** The avatar and interactive 3D world run smoothly in the browser.

### Level 5: Intelligence and Production
- **Build:** Personalization, natural-language/voice commands, performance optimization, Dockerized deployment.
- **Learn:** WebSocket/WebRTC, quantization, latency profiling, CI/CD, monitoring.
- **Done when:** The system is deployed and usable by other people through a link.

### Sign-language track (after Level 3)
Start with 10 to 20 simple PSL signs or letters, then short phrases, then a 3D avatar that displays signs from text.
Sign language includes hand shape, motion, facial expression and body posture, so sign accuracy should be reviewed by a PSL-fluent person or a member of the deaf community.

## 8. Repository Structure (create folders only when needed)

```
motionmind-ai/
  README.md
  MOTIONMIND_AI_BRIEF.md
  requirements.txt
  .gitignore
  src/
    tracking/      # webcam + MediaPipe
    features/      # feature extraction
    ml/            # dataset tools, training, inference
    web/           # Three.js frontend (later)
  data/            # recorded gesture data (not committed if large)
  models/          # trained models
  tests/           # automated tests
  docs/            # notes, screenshots, results
```

## 9. Coding Standards

- Python: PEP 8, type hints where helpful, docstrings for public functions.
- Keep functions short and single-purpose. Separate capture, processing and display code.
- No hard-coded magic numbers; use named constants or a config file.
- Handle errors gracefully (for example, webcam not found).
- Never commit virtual environments, large datasets or secrets.

## 10. Git Workflow

- Small, frequent commits with clear messages (for example: `feat: show live webcam feed`).
- One tag per completed level (`v0.1-level1`, `v0.2-level2`, and so on).
- Keep `README.md` current with setup and run instructions.

## 11. Evaluation Plan

- **Recognition:** accuracy, precision, recall, F1, confusion matrix.
- **Performance:** FPS, per-frame latency, end-to-end latency.
- **Robustness:** different lighting, backgrounds, camera angles and people.
- **Usability:** a short user survey (for example SUS) with 10 to 15 participants in later levels.
- **Ablation:** measure how much each feature or model component contributes.

## 12. Start Here (Level 1, Step 0 and Step 1 only)

**Step 0: Check and prepare the environment**
- Show how to check the Python version and pip in the VS Code terminal on Windows.
- Create a virtual environment inside the project folder and show how to activate it.
- Create `requirements.txt` and install only what Step 1 needs (OpenCV, and MediaPipe if it is compatible with this Python version).
- Then STOP and wait for "next".

**Step 1: Display the live webcam feed**
- Create `src/tracking/webcam_test.py` that opens the webcam, shows the live video in a window, and exits when `q` is pressed. Handle the case where the webcam cannot be opened.
- Explain each line simply, give the exact run command, and describe the expected result.
- Then STOP and wait for "next".

Do not begin hand tracking until instructed.

## 13. Progress Tracker (the assistant updates this after each completed step)

- [x] Step 0: Environment ready
- [x] Step 1: Live webcam feed
- [x] Step 2: Hand landmarks drawn live
- [x] Step 3: FPS counter
- [ ] Step 4: Basic 3D scene
- [ ] Step 5: Hand controls the 3D object
- [ ] Level 1 complete (git tag and short demo)
