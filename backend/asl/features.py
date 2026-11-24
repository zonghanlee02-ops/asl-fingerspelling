"""
features.py - The wrist-centred finger shape of one hand: the first 63 of the
sequence model's 68 features (asl/seq/features.py). The maths came from the
original baseline project; changing it changes what every trained sequence
model sees, so do not change it.
"""
import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def normalise_points(points):
    """Normalise raw landmark points.

    points: array-like of shape (21, 3) with MediaPipe x, y (0..1 image
            coordinates) and z (relative depth).
    returns: float32 array of shape (63,)
    """
    pts = np.array(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[0]                                         # wrist -> origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))    # hand "size" in 2D
    if scale > 0:
        pts /= scale
    return pts.flatten()


def points_from_mediapipe(hand_landmarks):
    """MediaPipe NormalizedLandmarkList -> numpy (21, 3) raw points."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32)
