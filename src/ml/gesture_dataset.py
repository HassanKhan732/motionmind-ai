"""Helpers for building a labeled dataset from hand landmarks."""

import csv
import math
from pathlib import Path


LANDMARK_COUNT = 21
LANDMARK_COLUMNS = [
    f"landmark_{index:02d}_{axis}"
    for index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]
CSV_COLUMNS = ["label", *LANDMARK_COLUMNS]


def normalize_landmarks(landmarks):
    """Return wrist-relative landmarks scaled by the wrist-to-middle-finger length."""
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks, got {len(landmarks)}.")

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    scale = math.sqrt(
        (middle_mcp.x - wrist.x) ** 2
        + (middle_mcp.y - wrist.y) ** 2
        + (middle_mcp.z - wrist.z) ** 2
    )
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("Cannot normalize landmarks with a zero or invalid palm length.")

    normalized = []
    for landmark in landmarks:
        coordinates = (
            (landmark.x - wrist.x) / scale,
            (landmark.y - wrist.y) / scale,
            (landmark.z - wrist.z) / scale,
        )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Landmark coordinates must be finite numbers.")
        normalized.extend(coordinates)

    return normalized


def append_sample(output_path, label, landmarks):
    """Append one labeled hand sample, creating a compatible CSV when needed."""
    label = label.strip()
    if not label:
        raise ValueError("Gesture label cannot be empty.")

    row = [label, *normalize_landmarks(landmarks)]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as dataset_file:
            existing_columns = next(csv.reader(dataset_file), [])
        if existing_columns != CSV_COLUMNS:
            raise ValueError(
                f"{path} has an incompatible header. Use a new output file."
            )

    with path.open("a", newline="", encoding="utf-8") as dataset_file:
        writer = csv.writer(dataset_file)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)
