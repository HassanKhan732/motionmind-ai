"""
MotionMind AI - Unit Tests: PSL Curriculum & Flashcard Generator
Tests curriculum generation levels, sign assignment, and markdown rendering.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.psl_flashcards import generate_psl_curriculum, export_markdown_guide


class TestPSLFlashcards(unittest.TestCase):

    def test_curriculum_contains_levels(self):
        curriculum = generate_psl_curriculum()
        self.assertGreaterEqual(len(curriculum), 3)
        self.assertEqual(curriculum[0]["level"], 1)

    def test_export_markdown_creates_file(self):
        tmp_path = "docs/test_curriculum.md"
        try:
            export_markdown_guide(output_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Pakistani Sign Language", content)
            self.assertIn("السلام علیکم", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
