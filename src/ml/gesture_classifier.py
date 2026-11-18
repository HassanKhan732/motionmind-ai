"""
MotionMind AI - Machine Learning: Real-Time Gesture Inference Engine
Runs low-latency predictions on landmark feature vectors with confidence thresholding and temporal debounce filtering.
"""

import os
import collections
from typing import Sequence, Any, Tuple, Optional
import joblib
import numpy as np

from src.features.gesture_features import extract_gesture_features
from src.ml.gestures import GESTURE_CLASSES, get_gesture_name


class RealTimeGestureClassifier:
    """
    Inference wrapper for static gesture recognition models.
    Applies temporal voting across recent frames to eliminate single-frame prediction noise.
    """

    def __init__(
        self,
        model_path: str = "models/gesture_classifier.joblib",
        min_confidence: float = 0.60,
        history_len: int = 5,
        required_consensus: int = 3,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

        self.model = joblib.load(model_path)
        self.min_confidence = min_confidence
        self.history_len = history_len
        self.required_consensus = required_consensus

        # Rolling history of recent predictions for temporal debounce
        self.history = collections.deque(maxlen=history_len)
        self.last_stable_gesture = "None"

    def reset(self):
        """Reset temporal history buffer when tracking is lost."""
        self.history.clear()
        self.last_stable_gesture = "None"

    def predict(self, landmarks: Sequence[Any]) -> Tuple[str, float, bool]:
        """
        Predict gesture from 21 hand landmarks.

        Returns:
            gesture_name (str): The predicted gesture label.
            confidence (float): Probability score [0.0 - 1.0].
            is_stable (bool): True if consensus threshold is reached across recent frames.
        """
        try:
            features = extract_gesture_features(landmarks)
            feat_arr = np.array([features], dtype=np.float32)

            # Predict probabilities
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(feat_arr)[0]
                pred_id = int(np.argmax(probas))
                confidence = float(probas[pred_id])
            else:
                pred_id = int(self.model.predict(feat_arr)[0])
                confidence = 1.0

            if confidence < self.min_confidence:
                gesture_name = "Uncertain"
            else:
                gesture_name = get_gesture_name(pred_id)

            # Push to temporal rolling history
            self.history.append(gesture_name)

            # Majority consensus check
            counts = collections.Counter(self.history)
            most_common, freq = counts.most_common(1)[0]

            is_stable = (freq >= self.required_consensus) and (most_common != "Uncertain")
            if is_stable:
                self.last_stable_gesture = most_common

            return gesture_name, round(confidence, 3), is_stable

        except Exception as err:
            return f"Error: {err}", 0.0, False
