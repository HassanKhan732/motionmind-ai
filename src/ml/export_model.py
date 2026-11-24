"""
MotionMind AI - Machine Learning: Model Export & Serialization Utility
Exports trained scikit-learn models into portable, lightweight formats for edge/browser deployment.
"""

import os
import sys
import json
from typing import Dict, Any
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.gestures import GESTURE_CLASSES


def export_model_metadata(
    model_path: str = "models/gesture_classifier.joblib",
    output_meta_path: str = "models/model_manifest.json"
) -> Dict[str, Any]:
    """
    Extract model hyperparameters, classes, and feature expectations into a portable manifest.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    file_size_kb = round(os.path.getsize(model_path) / 1024, 2)

    manifest = {
        "model_type": type(model).__name__,
        "classes": GESTURE_CLASSES,
        "num_classes": len(GESTURE_CLASSES),
        "expected_feature_dim": 75,
        "source_file": model_path,
        "file_size_kb": file_size_kb,
        "hyperparameters": {
            "n_estimators": getattr(model, "n_estimators", None),
            "max_depth": getattr(model, "max_depth", None),
            "min_samples_split": getattr(model, "min_samples_split", None),
        }
    }

    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    meta = export_model_metadata()
    print("Exported Model Manifest:")
    print(json.dumps(meta, indent=2))
