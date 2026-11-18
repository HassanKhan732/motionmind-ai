"""
MotionMind AI - Evaluation: Confusion Matrix Heatmap Generator
Renders high-resolution confusion matrix heatmap for all 16 gesture classes.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.gestures import GESTURE_CLASSES


def plot_confusion_matrix(metrics_json="docs/model_metrics.json", output_img="docs/confusion_matrix.png"):
    with open(metrics_json, "r") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    test_acc = data["test_accuracy"]
    best_model = data["best_model"]

    # Normalize by row (recall per class)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 11))
    sns.set_theme(style="dark")

    # Custom styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=GESTURE_CLASSES,
        yticklabels=GESTURE_CLASSES,
        cbar_kws={"label": "Normalized Ratio"},
        linewidths=0.5,
        linecolor="#1e293b",
    )

    plt.title(f"MotionMind AI - Confusion Matrix ({best_model} | Test Acc: {test_acc*100:.2f}%)", fontsize=15, pad=18, fontweight="bold")
    plt.xlabel("Predicted Gesture Label", fontsize=12, labelpad=12)
    plt.ylabel("True Gesture Label", fontsize=12, labelpad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_img}")


if __name__ == "__main__":
    plot_confusion_matrix()
