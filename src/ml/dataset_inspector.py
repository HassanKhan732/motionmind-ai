"""
MotionMind AI - Machine Learning: Dataset Inspector & Validation Tool
Inspects gesture datasets for class distribution, null/corrupted features, and dimensionality.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.gestures import GESTURE_CLASSES


def inspect_dataset(data_path: str = "data/gestures.csv") -> Dict[str, Any]:
    """
    Load and evaluate the structural integrity and distribution of a gesture dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    total_rows = len(df)
    total_features = len(feature_cols)

    # Check for NaN / null values
    null_counts = df[feature_cols].isnull().sum().sum()

    # Per-class sample counts
    label_col = "label_name" if "label_name" in df.columns else ("label" if "label" in df.columns else "label_id")
    class_counts = df[label_col].value_counts().to_dict()

    # Calculate balance metric (ratio of min samples to max samples)
    counts_list = list(class_counts.values())
    balance_ratio = (min(counts_list) / max(counts_list)) if counts_list else 0.0

    report = {
        "dataset_path": data_path,
        "total_samples": int(total_rows),
        "feature_count": int(total_features),
        "total_classes": len(class_counts),
        "has_nulls": bool(null_counts > 0),
        "null_count": int(null_counts),
        "balance_ratio": float(balance_ratio),
        "class_distribution": class_counts,
    }

    return report


def print_inspection_report(report: Dict[str, Any]):
    """Print a clean terminal dashboard of the dataset inspection."""
    print("=" * 60)
    print(" MOTIONMIND AI - DATASET HEALTH & INSPECTION REPORT")
    print("=" * 60)
    print(f" Dataset Path     : {report['dataset_path']}")
    print(f" Total Samples    : {report['total_samples']}")
    print(f" Feature Dimension: {report['feature_count']} features per sample")
    print(f" Total Classes    : {report['total_classes']} / {len(GESTURE_CLASSES)}")
    print(f" Integrity Status : {'PASSED (No Nulls)' if not report['has_nulls'] else 'FAILED'}")
    print(f" Class Balance    : {report['balance_ratio'] * 100:.1f}% uniformity")
    print("-" * 60)
    print(f" {'Gesture Class':<22} | {'Samples':<10} | {'Percentage':<10}")
    print("-" * 60)

    total = report['total_samples']
    for cls_name, count in sorted(report['class_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        print(f" {cls_name:<22} | {count:<10} | {pct:>6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/gestures.csv"
    try:
        rep = inspect_dataset(path)
        print_inspection_report(rep)
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        sys.exit(1)
