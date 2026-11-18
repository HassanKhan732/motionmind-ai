"""
MotionMind AI - Machine Learning: Multi-Classifier Training & Benchmarking Pipeline
Trains, validates, and benchmarks Random Forest, SVM, and MLP classifiers on invariant hand features.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.gestures import GESTURE_CLASSES


def load_dataset(data_path="data/gestures.csv"):
    """Load pre-generated dataset or synthesize on-the-fly."""
    if not os.path.exists(data_path):
        from scripts.generate_data import main as run_gen
        run_gen()

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label_id"].values.astype(np.int32)
    return X, y


def build_candidate_models():
    """Define candidate ML models for benchmarking."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        ),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10.0, kernel="rbf", probability=True, random_state=42))
        ]),
        "MLP_NeuralNet": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=350, random_state=42, early_stopping=True))
        ]),
    }


def train_and_benchmark():
    """Execute complete cross-validation and evaluation pipeline."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=== Loading Gesture Dataset ===")
    X, y = load_dataset()
    print(f"Total Samples: {X.shape[0]} | Features per Sample: {X.shape[1]} | Classes: {len(np.unique(y))}")

    # 1. Stratified 70% Train / 15% Validation / 15% Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )

    print(f"Dataset Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 2. Benchmark Candidates with 5-Fold Stratified Cross-Validation
    models = build_candidate_models()
    benchmark_results = {}
    best_model_name = None
    best_val_acc = 0.0
    best_model = None

    print("\n=== Benchmarking Candidate Classifiers (5-Fold CV) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        train_time = time.time() - t0

        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        benchmark_results[name] = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "val_accuracy": float(val_acc),
            "train_time_sec": round(train_time, 2),
        }
        print(f"[{name}] CV Acc: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f}) | Val Acc: {val_acc:.4f} ({train_time:.2f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = name
            best_model = model

    print(f"\nWinner Model: {best_model_name} (Validation Accuracy: {best_val_acc * 100:.2f}%)")

    # 3. Final Test Set Evaluation
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"=== Final Test Set Accuracy: {test_acc * 100:.2f}% ===")

    report_dict = classification_report(
        y_test, y_pred, target_names=GESTURE_CLASSES, output_dict=True
    )
    conf_mat = confusion_matrix(y_test, y_pred)

    # 4. Save Best Model
    model_export_path = "models/gesture_classifier.joblib"
    joblib.dump(best_model, model_export_path)
    print(f"Model saved to {model_export_path}")

    # 5. Export Metrics Summary
    metrics_summary = {
        "best_model": best_model_name,
        "test_accuracy": float(test_acc),
        "validation_accuracy": float(best_val_acc),
        "benchmark_comparison": benchmark_results,
        "classification_report": report_dict,
        "confusion_matrix": conf_mat.tolist(),
    }
    with open("docs/model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return best_model_name, test_acc


if __name__ == "__main__":
    train_and_benchmark()
