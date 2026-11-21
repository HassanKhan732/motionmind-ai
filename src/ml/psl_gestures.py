"""
MotionMind AI - Pakistani Sign Language (PSL) Module: Sign Taxonomy & Metadata
Defines static PSL letters and signs to empower communication for deaf and mute individuals.
"""

from typing import Dict, List, Optional, Any


PSL_STATIC_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "PSL_SALAAM": {
        "urdu_name": "السلام علیکم (Salaam)",
        "english": "Peace / Hello",
        "category": "Greeting",
        "hands": 1,
        "description": "Flat open hand raised near temple/forehead, fingertips touching forehead gently",
        "landmark_pattern": "flat_palm_temple",
    },
    "PSL_SHUKRIYA": {
        "urdu_name": "شکریہ (Shukriya)",
        "english": "Thank You",
        "category": "Etiquette",
        "hands": 1,
        "description": "Fingertips touching the chin, then moving gently outward toward the listener",
        "landmark_pattern": "chin_touch_forward",
    },
    "PSL_HAAN": {
        "urdu_name": "ہاں (Haan)",
        "english": "Yes",
        "category": "Response",
        "hands": 1,
        "description": "Fist with thumb tucked, nodding motion downwards like a head nod",
        "landmark_pattern": "fist_nod",
    },
    "PSL_NAHI": {
        "urdu_name": "نہیں (Nahi)",
        "english": "No",
        "category": "Response",
        "hands": 1,
        "description": "Index finger extended and waving side to side horizontally",
        "landmark_pattern": "pointing_side_sweep",
    },
    "PSL_PAANI": {
        "urdu_name": "پانی (Paani)",
        "english": "Water",
        "category": "Basic Needs",
        "hands": 1,
        "description": "Three middle fingers extended (W shape) near mouth",
        "landmark_pattern": "w_mouth_tap",
    },
    "PSL_MADAD": {
        "urdu_name": "مدد (Madad)",
        "english": "Help",
        "category": "Emergency / Needs",
        "hands": 2,
        "description": "Closed fist on one hand placed on the open palm of the other hand, lifted upward",
        "landmark_pattern": "fist_on_palm_lift",
    },
    "PSL_DOST": {
        "urdu_name": "دوست (Dost)",
        "english": "Friend",
        "category": "Social",
        "hands": 2,
        "description": "Both index fingers hooked together in friendship link",
        "landmark_pattern": "hooked_index_clasp",
    },
}


def get_psl_metadata(sign_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve educational metadata and translation for a PSL sign."""
    return PSL_STATIC_DICTIONARY.get(sign_key)


def list_psl_signs() -> List[str]:
    """Return all supported PSL sign keys."""
    return list(PSL_STATIC_DICTIONARY.keys())


def format_psl_display(sign_key: str) -> str:
    """Format sign for HUD and UI banners."""
    data = get_psl_metadata(sign_key)
    if not data:
        return sign_key
    return f"{data['urdu_name']} - {data['english']}"
