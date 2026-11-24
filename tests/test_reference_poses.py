"""Reference poses from your recordings (scripts/app/build_reference_poses.py --from-recordings)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "app"))
import build_reference_poses as brp  # noqa: E402


def hand(x=0.5, y=0.5, spread=0.1):
    p = np.zeros((21, 3), np.float32)
    p[:, 0] = x + spread * np.linspace(-1, 1, 21)
    p[:, 1] = y - spread * np.linspace(0, 1, 21)
    return p


def test_stillest_frame_near_the_confident_moment():
    times = np.arange(10) / 30
    points = [hand(0.5 + 0.02 * k) for k in range(10)]        # moving right
    points[6] = points[5].copy()                               # still at frame 6 only
    points[7] = None                                           # no hand at frame 7
    assert brp.stillest_near(times, points, t_peak=6 / 30) == 6
    assert brp.stillest_near(times, points, t_peak=2 / 30) in (1, 2, 3, 4, 5)   # nothing outside 0.1 s
    assert brp.stillest_near(times, [None] * 10, t_peak=0.1) is None


class FakeRecognizer:
    """Reads 'GO': most sure of G at frame 2, of O at frame 7 (30 frames a second)."""
    fps = 30.0

    def recognize(self, times, points, hand, image_size):
        logits = np.zeros((len(times), 28), np.float32)
        logits[:, 0] = 5                                        # blank
        logits[2, 7], logits[7, 15] = 10, 10                    # G = 7, O = 15
        return "GO", logits


def test_confident_samples_take_the_frame_at_the_peak_unmirrored_and_isotropic():
    pts = [hand() for _ in range(10)]
    pts[2] = hand(spread=0.2)                                   # the G frame looks different
    take = {"t": np.arange(10) / 30, "used": pts, "hand": "right", "image_size": (640, 480),
            "mirror_input": True, "word": "GO"}
    out = brp.confident_samples(take, FakeRecognizer(), ["G"])
    assert [L for L, _ in out] == ["G"]                         # only the letters asked for
    raw = np.array(pts[2])
    raw[:, 0] = 1 - raw[:, 0]                                   # back to the camera's view
    raw[:, 1] *= 480 / 640                                      # same units for x and y
    from asl.features import normalise_points
    np.testing.assert_allclose(out[0][1], normalise_points(raw), atol=1e-6)
