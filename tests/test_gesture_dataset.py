"""Tests for labeled hand-landmark dataset creation."""

import csv
import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.gesture_dataset import (
    CSV_COLUMNS,
    append_sample,
    normalize_landmarks,
)


def create_landmarks(scale=1.0, offset=0.0):
    landmarks = [
        SimpleNamespace(
            x=offset + index * 0.01 * scale,
            y=offset + index * 0.02 * scale,
            z=offset + index * 0.005 * scale,
        )
        for index in range(21)
    ]
    landmarks[0] = SimpleNamespace(x=offset, y=offset, z=offset)
    landmarks[9] = SimpleNamespace(
        x=offset + 0.09 * scale,
        y=offset + 0.18 * scale,
        z=offset + 0.045 * scale,
    )
    return landmarks


class TestGestureDataset(unittest.TestCase):
    def test_normalization_removes_position_and_scale(self):
        baseline = normalize_landmarks(create_landmarks())
        shifted_and_scaled = normalize_landmarks(create_landmarks(scale=2, offset=4))
        for actual, expected in zip(shifted_and_scaled, baseline):
            self.assertAlmostEqual(actual, expected)

    def test_append_sample_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "samples.csv"
            append_sample(output_path, " open_palm ", create_landmarks())
            append_sample(output_path, "fist", create_landmarks())

            with output_path.open(newline="", encoding="utf-8") as dataset_file:
                rows = list(csv.reader(dataset_file))

        self.assertEqual(rows[0], CSV_COLUMNS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "open_palm")
        self.assertEqual(len(rows[1]), len(CSV_COLUMNS))
        self.assertEqual(rows[2][0], "fist")

    def test_rejects_bad_landmarks_and_incompatible_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "samples.csv"
            output_path.write_text("wrong,header\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "21 landmarks"):
                normalize_landmarks(create_landmarks()[:20])
            with self.assertRaisesRegex(ValueError, "incompatible header"):
                append_sample(output_path, "fist", create_landmarks())


if __name__ == "__main__":
    unittest.main()
