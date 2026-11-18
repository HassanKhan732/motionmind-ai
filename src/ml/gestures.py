"""
MotionMind AI - Machine Learning: Gesture Taxonomy
Defines the 16 static gesture classes including core HCI gestures and foundational PSL signs.
"""

# Ordered mapping of gesture IDs to human-readable names
GESTURE_CLASSES = [
    "Open Palm",      # 0: Open hand, fingers extended
    "Fist",           # 1: All fingers curled into palm
    "Peace",          # 2: Index and middle fingers extended (V sign)
    "Thumbs Up",      # 3: Thumb pointing upward, other fingers curled
    "Thumbs Down",    # 4: Thumb pointing downward, other fingers curled
    "OK Sign",        # 5: Thumb and index touching in circle, others extended
    "Pointing",       # 6: Index finger extended, others curled
    "Rock Horns",     # 7: Index and pinky extended, middle and ring curled
    "Call Me",        # 8: Thumb and pinky extended
    "Hand Gun",       # 9: Thumb and index extended at right angle
    "Pinch",          # 10: Thumb and index tips closely pinched
    "Three",          # 11: Thumb, index, and middle extended
    "Four",           # 12: Four fingers extended, thumb folded
    "Stop Sign",      # 13: Flat palm facing forward firmly
    "PSL Salaam",     # 14: Pakistani Sign Language: Flat hand near temple/forehead
    "PSL Shukriya",   # 15: Pakistani Sign Language: Open hand touching chin moving forward
]

GESTURE_TO_ID = {name: idx for idx, name in enumerate(GESTURE_CLASSES)}
ID_TO_GESTURE = {idx: name for idx, name in enumerate(GESTURE_CLASSES)}


def get_gesture_name(class_id: int) -> str:
    """Return the readable gesture label for a class integer ID."""
    return ID_TO_GESTURE.get(class_id, "Unknown")


def get_gesture_id(name: str) -> int:
    """Return the class ID for a gesture name string."""
    return GESTURE_TO_ID.get(name, -1)


def get_total_classes() -> int:
    """Return the total number of recognized gesture classes."""
    return len(GESTURE_CLASSES)
