"""
MotionMind AI - Data Generation Entrypoint
Synthesizes 5,600 augmented gesture samples and saves to data/gestures.csv.
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.dataset_generator import generate_augmented_dataset
from src.ml.gestures import GESTURE_CLASSES


def main():
    os.makedirs("data", exist_ok=True)
    samples_per_class = 350
    X, y = generate_augmented_dataset(samples_per_class=samples_per_class)

    # 75 feature column names
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["label_id"] = y
    df["label_name"] = [GESTURE_CLASSES[idx] for idx in y]

    output_csv = "data/gestures.csv"
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated {len(df)} samples across {len(GESTURE_CLASSES)} classes.")
    print(f"Saved dataset to {output_csv} (File size: {os.path.getsize(output_csv) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
