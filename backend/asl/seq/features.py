"""
features.py - The 68 numbers the sequence model sees for every video frame.

Why not reuse the baseline's 63 numbers alone?  The baseline subtracts the
wrist, so the hand is always "at the origin". That is right for a single
held letter, but it erases the movement of the whole hand (J, Z, the small
slide of a double letter, the transition between letters). So we keep it
and add where the wrist is and how fast it moves.

Per frame (FEATURE_NAMES lists them in order):
    0..62  finger coordinates: the baseline asl.features.normalise_points()
           (wrist moved to the origin, divided by hand size, x0,y0,z0,...).
           Indices 0..2 are the wrist itself and are always 0.
    63,64  wrist position x, y in image coordinates minus 0.5 (centre = 0)
    65,66  wrist velocity x, y in HAND SIZES PER FRAME (so the number does
           not depend on how close the hand is to the camera)
    67     validity flag: 1 = a hand was tracked in this frame, 0 = not

Rules (all causal = only the current and past frames are used):
* Signing hand: one hand per sequence (the person's signing hand). A LEFT
  hand is mirrored (x -> 1 - x) so every sequence looks like a right hand.
* Missing / invalid detection (None, NaN, inf, or a hand smaller than
  min_hand_scale): all 68 numbers are 0, including the flag.
* Velocity: difference to the LAST VALID frame divided by the number of
  frames since then, if that gap is <= max_gap_frames; otherwise 0 (after a
  long gap the old position says nothing about current movement).
  The first valid frame of a sequence has velocity 0. Clipped to +-vel_clip.
* Frame rate: the model counts in frames, so live camera frames are
  resampled to target_fps (CausalResampler) before featurisation.

The SAME SequenceFeaturizer.step() is used for training data (Colab) and
the live camera (laptop), so the two cannot drift apart.
"""
import numpy as np

from ..features import normalise_points, NUM_LANDMARKS

FEATURE_VERSION = "seq68-v1"

FEATURE_NAMES = ([f"hand{i}_{a}" for i in range(NUM_LANDMARKS) for a in "xyz"]
                 + ["wrist_x", "wrist_y", "wrist_vx", "wrist_vy", "valid"])
NUM_FEATURES = len(FEATURE_NAMES)                   # 68
VALID = NUM_FEATURES - 1                            # index of the flag
WRIST_POS = slice(63, 65)
WRIST_VEL = slice(65, 67)
FINGERS = slice(0, 63)

DEFAULT_CONFIG = {
    "feature_version": FEATURE_VERSION,
    "num_features": NUM_FEATURES,
    "feature_names": FEATURE_NAMES,
    "canonical_hand": "right",       # left hands are mirrored to look like right hands
    "target_fps": 30.0,              # ASSUMPTION about the dataset; see docs/SEQUENCE_MODEL.md
    "max_gap_frames": 3,
    "vel_clip": 1.0,                 # hand sizes per frame
    "min_hand_scale": 1e-3,          # smaller "hands" are treated as tracking failures
}


def canonical_points(points, hand):
    """Mirror a left hand so that all sequences use right-hand geometry.
    points: (..., 21, 3), e.g. one frame or a whole (T, 21, 3) sequence."""
    pts = np.array(points, dtype=np.float32, copy=True)
    if str(hand).lower().startswith("l"):
        pts[..., 0] = 1.0 - pts[..., 0]
    return pts


def _hand_scale(pts):
    return float(np.max(np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1)))


class SequenceFeaturizer:
    """Turns one frame at a time into 68 features, remembering only the past."""

    def __init__(self, config=None):
        self.cfg = dict(DEFAULT_CONFIG, **(config or {}))
        self.reset()

    def reset(self):
        self._last_wrist = None
        self._since_valid = 0

    def step(self, points):
        """points: (21, 3) raw landmarks of the (already canonical) signing
        hand, or None when no hand was found. Returns float32 (68,)."""
        out = np.zeros(NUM_FEATURES, np.float32)
        pts = None
        if points is not None:
            pts = np.asarray(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
            if not np.all(np.isfinite(pts)) or _hand_scale(pts) < self.cfg["min_hand_scale"]:
                pts = None
        self._since_valid += 1
        if pts is None:
            return out

        scale = _hand_scale(pts)
        wrist = pts[0, :2]
        out[FINGERS] = normalise_points(pts)
        out[WRIST_POS] = wrist - 0.5
        gap = self._since_valid
        if self._last_wrist is not None and gap <= self.cfg["max_gap_frames"]:
            vel = (wrist - self._last_wrist) / gap / scale
            c = self.cfg["vel_clip"]
            out[WRIST_VEL] = np.clip(vel, -c, c)
        out[VALID] = 1.0
        self._last_wrist = wrist.copy()
        self._since_valid = 0
        return out

    def featurize(self, points_seq):
        """(T, 21, 3) with NaN rows for missing frames (or a list that may
        contain None) -> (T, 68). Resets first."""
        self.reset()
        rows = [self.step(p) for p in points_seq]
        return np.stack(rows) if rows else np.zeros((0, NUM_FEATURES), np.float32)


def featurize_sequence(points_seq, config=None):
    return SequenceFeaturizer(config).featurize(points_seq)


class Normalizer:
    """Per-feature mean/std, fitted on VALID frames of the TRAINING split only.

    Invalid frames stay all-zero after normalisation; the flag is never scaled.
    """

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

    def fit(self, feature_arrays):
        n = 0
        s = np.zeros(NUM_FEATURES, np.float64)
        ss = np.zeros(NUM_FEATURES, np.float64)
        for f in feature_arrays:
            v = f[f[:, VALID] > 0.5].astype(np.float64)
            n += len(v)
            s += v.sum(0)
            ss += (v ** 2).sum(0)
        if n == 0:
            raise ValueError("No valid frames to fit the normaliser")
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
        std[std < 1e-6] = 1.0                # constant columns (wrist = origin) stay as they are
        mean[VALID], std[VALID] = 0.0, 1.0
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        return self

    def apply(self, feats):
        f = np.asarray(feats, np.float32)
        valid = f[..., VALID:VALID + 1] > 0.5
        out = np.where(valid, (f - self.mean) / self.std, 0.0).astype(np.float32)
        out[..., VALID] = valid[..., 0]
        return out

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def preprocessing_dict(normalizer, feature_cfg=None, **extra):
    """Everything the laptop needs to reproduce training preprocessing."""
    return {"features": dict(DEFAULT_CONFIG, **(feature_cfg or {})),
            "normalizer": normalizer.to_dict(), **extra}


class CausalResampler:
    """Converts camera frames with timestamps to a fixed frame rate.

    For every target tick between the previous and the current camera frame
    it linearly interpolates the landmarks (both frames valid) or takes the
    nearer frame. Only frames that have already arrived are used, so the
    extra delay is at most one camera frame.
    """

    def __init__(self, target_fps=30.0):
        self.dt = 1.0 / float(target_fps)
        self.reset()

    def reset(self):
        self._prev = None            # (t, points or None)
        self._next_tick = None

    def push(self, t, points):
        """Add one camera frame; returns a list of resampled frames (points or None)."""
        pts = None if points is None else np.asarray(points, np.float32).reshape(NUM_LANDMARKS, 3)
        out = []
        if self._prev is None:
            out.append(pts)
            self._next_tick = t + self.dt
        else:
            t0, p0 = self._prev
            while self._next_tick <= t + 1e-9:
                tau = self._next_tick
                w = (tau - t0) / (t - t0) if t > t0 else 1.0
                if p0 is not None and pts is not None:
                    out.append((1 - w) * p0 + w * pts)
                else:
                    out.append(p0 if w < 0.5 else pts)
                self._next_tick += self.dt
        self._prev = (t, pts)
        return out


def chirality_score(points_seq):
    """Fraction of valid frames where the thumb base (landmark 1) is to the
    RIGHT of the little-finger base (landmark 17) in the image.

    Used only as a sanity check that the dataset's canonical hand and the
    live camera's canonical hand have the same left/right geometry (mirror
    convention). Values near 0 and near 1 mean opposite conventions.
    """
    p = np.asarray(points_seq, np.float32).reshape(-1, NUM_LANDMARKS, 3)
    ok = np.all(np.isfinite(p.reshape(len(p), -1)), axis=1)
    if not ok.any():
        return float("nan")
    return float(np.mean(p[ok, 1, 0] > p[ok, 17, 0]))
