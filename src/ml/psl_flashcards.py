"""
MotionMind AI - Pakistani Sign Language (PSL): Educational Curriculum & Flashcard Generator
Generates structured learning guides, interactive flashcards, and practice modules for PSL signers.
"""

import os
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.psl_gestures import PSL_STATIC_DICTIONARY


def generate_psl_curriculum() -> List[Dict[str, Any]]:
    """
    Structure signs into sequential difficulty levels (Beginner, Everyday Needs, Social).
    """
    levels = [
        {"level": 1, "category": "Greeting", "title": "First Greetings (پہلی بات چیت)"},
        {"level": 2, "category": "Etiquette", "title": "Polite Words (شائستہ الفاظ)"},
        {"level": 3, "category": "Response", "title": "Basic Responses (بنیادی جوابات)"},
        {"level": 4, "category": "Basic Needs", "title": "Daily Needs (روزمرہ ضروریات)"},
        {"level": 5, "category": "Emergency / Needs", "title": "Assistance & Help (مدد)"},
    ]

    curriculum = []
    for lvl in levels:
        cat = lvl["category"]
        matching_signs = [
            {"sign_id": k, **v} for k, v in PSL_STATIC_DICTIONARY.items() if v["category"] == cat
        ]
        curriculum.append({
            "level": lvl["level"],
            "title": lvl["title"],
            "category": cat,
            "signs": matching_signs,
        })
    return curriculum


def export_markdown_guide(output_path: str = "docs/PSL_CURRICULUM.md"):
    """Render curriculum into a clean bilingual markdown guide."""
    curriculum = generate_psl_curriculum()

    lines = [
        "# MotionMind AI - Pakistani Sign Language (PSL) Curriculum",
        "",
        "> Educational curriculum designed to help children and educators learn and practice PSL using computer vision.",
        "",
        "| Level | Sign (Urdu) | English Meaning | Category | Hand Shape & Motion |",
        "|---|---|---|---|---|",
    ]

    for item in curriculum:
        for s in item["signs"]:
            lines.append(
                f"| {item['level']} | **{s['urdu_name']}** | {s['english']} | {s['category']} | {s['description']} |"
            )

    lines.append("")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    export_markdown_guide()
    print("Exported PSL curriculum guide to docs/PSL_CURRICULUM.md")
