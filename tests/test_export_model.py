"""
MotionMind AI - Unit Tests: Model Export & Serialization
Tests manifest structure, class verification, and export file creation.
"""

import sys
import os
import unittest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.export_model import export_model_metadata


class TestExportModel(unittest.TestCase):

    def setUp(self):
        self.out_manifest = "models/test_manifest.json"

    def tearDown(self):
        if os.path.exists(self.out_manifest):
            os.remove(self.out_manifest)

    def test_export_manifest_contents(self):
        manifest = export_model_metadata(
            model_path="models/gesture_classifier.joblib",
            output_meta_path=self.out_manifest
        )
        self.assertEqual(manifest["expected_feature_dim"], 75)
        self.assertEqual(manifest["num_classes"], 16)
        self.assertTrue(os.path.exists(self.out_manifest))

        with open(self.out_manifest, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["model_type"], manifest["model_type"])


if __name__ == "__main__":
    unittest.main()
