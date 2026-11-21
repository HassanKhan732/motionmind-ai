"""
MotionMind AI - Unit Tests: Pakistani Sign Language (PSL) Taxonomy
Tests sign retrieval, dictionary completeness, and UI formatters.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.psl_gestures import (
    PSL_STATIC_DICTIONARY,
    get_psl_metadata,
    list_psl_signs,
    format_psl_display,
)


class TestPSLGestures(unittest.TestCase):

    def test_dictionary_non_empty(self):
        signs = list_psl_signs()
        self.assertGreaterEqual(len(signs), 5)
        self.assertIn("PSL_SALAAM", signs)
        self.assertIn("PSL_SHUKRIYA", signs)

    def test_metadata_structure(self):
        meta = get_psl_metadata("PSL_SALAAM")
        self.assertIsNotNone(meta)
        self.assertIn("urdu_name", meta)
        self.assertIn("english", meta)
        self.assertIn("category", meta)
        self.assertIn("hands", meta)

    def test_display_formatting(self):
        disp = format_psl_display("PSL_SHUKRIYA")
        self.assertIn("شکریہ", disp)
        self.assertIn("Thank You", disp)

    def test_missing_key_fallback(self):
        disp = format_psl_display("NON_EXISTENT")
        self.assertEqual(disp, "NON_EXISTENT")


if __name__ == "__main__":
    unittest.main()
