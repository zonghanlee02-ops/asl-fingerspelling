"""
tracker.py - MediaPipe hand tracking (a SUPPORTING tool).

MediaPipe only finds the 21 hand landmarks. It does not know any ASL.
The letter itself is decided by our own TensorFlow model (classifier.py).
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .features import points_from_mediapipe


@dataclass
class HandObservation:
    points: np.ndarray        # raw (21, 3) landmarks in the (mirrored) frame
    handedness: str           # "Left" / "Right" as reported by MediaPipe
    handedness_score: float


class HandTracker:
    """Wraps mediapipe.solutions.hands.Hands.

    video_mode=True  -> uses tracking between consecutive frames (live camera,
                        imported videos). Faster and steadier.
    video_mode=False -> every image treated independently (photo datasets).
    """

    def __init__(self, video_mode=True, min_detection_confidence=0.6,
                 min_tracking_confidence=0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=not video_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bgr(self, frame_bgr) -> Optional[HandObservation]:
        """Find one hand in a BGR image (OpenCV format). None if no hand."""
        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        label, score = "Unknown", 0.0
        if result.multi_handedness:
            c = result.multi_handedness[0].classification[0]
            label, score = c.label, float(c.score)
        return HandObservation(points_from_mediapipe(hand), label, score)

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
