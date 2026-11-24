"""
build_reference_poses.py - Make the 3D hand's reference poses FROM REAL DATA.

For every letter we pick the MEDOID sample of that letter in
data/landmarks.npz: the real recorded hand whose landmarks are closest to
all the other samples of the same letter. It is an actual recorded hand
shape (not an average, which can produce impossible fingers, and not a pose
we invented).

Output: frontend/public/reference_poses.json

Every pose starts with verification.status = "unverified". A person must
compare each pose with a reliable ASL fingerspelling reference and mark it
verified with scripts/app/verify_poses.py. The interface shows the status.

Motion letters: J and Z are signed WITH movement. The dataset photo only
shows one moment, so their motion path is added from the standard written
description and labelled as such.

Run:  python scripts/app/build_reference_poses.py

From your own recordings instead, for some letters (the others are kept):

    python scripts/app/build_reference_poses.py --from-recordings G,J,M,N,P,Q,T,X,Z

final_model_i reads each saved take in backend/storage/my_recordings. Each time it is surest of a
letter of the take's word, the stillest hand frame within 0.1 s of that moment is taken (checked:
the model is sure of a letter while the shape is shown, so looking further back picks the previous
letter). The pose is the medoid of those frames. This shows how YOU sign the letter, so every such
pose starts unverified again.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "sequence_model"))

POSES_PATH = ROOT / "frontend" / "public" / "reference_poses.json"
BACKUP_PATH = ROOT / "data" / "reference_poses_kaggle_backup.json"
RECORDINGS = ROOT / "backend" / "storage" / "my_recordings"
WINDOW_S = 0.1

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# Result of a visual check of every rendered pose (front, side and top
# views) made while developing this project. It is NOT a verification
# against an ASL reference: it only flags poses whose MediaPipe landmark
# estimate visibly fails to show the handshape (typically closed-fist
# letters, where curled fingers hide each other in the photo).
DEV_FLAGGED = {
    L: "Landmark estimate does not show the fingers curled into the palm; "
       "the 3D pose may be inaccurate for this letter."
    for L in "AEMNSTXZ"
}

MOTION = {
    "J": "Starts from the I handshape; the little finger draws a J (down, then a hook "
         "towards the signer) while the wrist turns.",
    "Z": "Index finger draws a Z in the air (right, diagonal down-left, right).",
}


def medoid(X, max_n=400, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return X[np.argmin(d.sum(axis=1))]


def load_source():
    src = ROOT / "data" / "landmarks.npz"
    if not src.exists():
        sys.exit("data/landmarks.npz not found (hand points from the ASL Alphabet photos; "
                 "the script that made it was removed with the baseline model)")
    d = np.load(src)
    return d["X"], d["y"], "data/landmarks.npz (ASL Alphabet dataset, Kaggle)"


def stillest_near(times, points, t_peak, window=WINDOW_S):
    """Index of the frame with a hand, at most `window` s from t_peak, whose hand moved least
    since the previous frame (movement in hand sizes). None if there is no such frame."""
    best, best_move = None, np.inf
    for i in range(1, len(points)):
        if not abs(times[i] - t_peak) <= window + 1e-6 or points[i] is None or points[i - 1] is None:
            continue
        a, b = np.asarray(points[i - 1])[:, :2], np.asarray(points[i])[:, :2]
        size = max(float(np.max(np.linalg.norm(b - b[0], axis=1))), 1e-6)
        move = float(np.mean(np.linalg.norm(b - a, axis=1))) / size
        if move < best_move:
            best, best_move = i, move
    return best


def confident_samples(take, wr, letters):
    """[(letter, 63 numbers)]: one hand frame for every moment the model was surest of a letter of the word."""
    from asl.features import normalise_points
    from asl.seq import vocab

    times = np.asarray(take["t"]) - take["t"][0]
    _, logits = wr.recognize(times.tolist(), take["used"], hand=take["hand"], image_size=take["image_size"])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    best = p.argmax(axis=1)
    w, h = take["image_size"]
    out = []
    for L in sorted(set(take["word"]) & set(letters)):
        k = vocab.CHAR_TO_ID[L]
        frames = np.flatnonzero(best == k)
        for run in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1):    # one run = one reading of L
            if not len(run):
                continue
            peak = run[np.argmax(p[run, k])]
            i = stillest_near(times, take["used"], peak / wr.fps)
            if i is None:
                continue
            pts = np.array(take["used"][i], np.float32)
            if take["mirror_input"]:
                pts[:, 0] = 1.0 - pts[:, 0]        # back to the camera's own view, like the dataset photos
            pts[:, 1] *= h / w                     # x and y in the same units
            out.append((L, normalise_points(pts)))
    return out


def from_recordings(letters, model_dir):
    """Replace the poses of `letters` in reference_poses.json with poses from your recordings."""
    from asl.seq.inference import WordRecognizer
    from seq_pack_my_recordings import load_take

    if not BACKUP_PATH.exists():
        shutil.copy2(POSES_PATH, BACKUP_PATH)
        print(f"Kept the current poses in {BACKUP_PATH}")
    data = json.loads(POSES_PATH.read_text(encoding="utf-8"))
    wr = WordRecognizer(model_dir)
    samples = {L: [] for L in letters}
    takes = {L: set() for L in letters}
    files = sorted(RECORDINGS.glob("*.npz"))
    for f in files:
        for L, x in confident_samples(load_take(f), wr, letters):
            samples[L].append(x)
            takes[L].add(f.name)
    for L in letters:
        if not samples[L]:
            print(f"{L}: the model was never sure of it in your takes; pose kept")
            continue
        m = medoid(np.stack(samples[L]))
        data["poses"][L] = {
            "landmarks": np.round(m.reshape(21, 3), 4).tolist(),
            "source": f"medoid of {len(samples[L])} frames of '{L}' from {len(takes[L])} of your recorded takes "
                      f"(backend/storage/my_recordings): in each, the stillest hand frame within {WINDOW_S} s "
                      f"of the moment {Path(model_dir).name} was surest of '{L}'",
            "n_samples": len(samples[L]),
            "motion": MOTION.get(L),
            "dev_check": {"result": "not checked", "notes": None,
                          "method": "built from your own signing; not a verification against an ASL reference"},
            "verification": {"status": "unverified", "by": None, "reference": None, "date": None, "notes": None},
        }
        print(f"{L}: {len(samples[L])} frames from {len(takes[L])} takes")
    POSES_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {POSES_PATH} ({len(files)} takes read)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-recordings", metavar="LETTERS",
                    help="comma-separated letters to rebuild from your recordings, e.g. G,J,M")
    ap.add_argument("--model", default=str(ROOT / "backend" / "models" / "final_model_i"))
    args = ap.parse_args()
    if args.from_recordings:
        letters = [L.strip().upper() for L in args.from_recordings.split(",") if L.strip()]
        bad = [L for L in letters if L not in LETTERS]
        if bad:
            sys.exit(f"Not letters A-Z: {', '.join(bad)}")
        return from_recordings(letters, args.model)

    X, y, source_name = load_source()
    flags = DEV_FLAGGED
    out_path = POSES_PATH
    old = json.loads(out_path.read_text()) if out_path.exists() else {"poses": {}}
    poses = {}
    for L in LETTERS:
        Xi = X[y == L]
        if not len(Xi):
            continue
        m = medoid(Xi)
        prev = old.get("poses", {}).get(L, {})
        landmarks = np.round(m.reshape(21, 3), 4).tolist()
        if prev.get("landmarks") != landmarks:
            prev = {}                  # a different pose: an earlier verification does not apply to it
        poses[L] = {
            "landmarks": landmarks,
            "source": f"medoid of {len(Xi)} samples of '{L}' in {source_name}",
            "n_samples": int(len(Xi)),
            "motion": MOTION.get(L),
            "dev_check": {
                "result": "flagged" if L in flags else "plausible",
                "notes": flags.get(L),
                "method": "visual inspection of the 3D render during development; "
                          "not a verification against an ASL reference",
            },
            # keep an existing human verification if the same pose is rebuilt
            "verification": prev.get("verification") or {
                "status": "unverified", "by": None, "reference": None, "date": None, "notes": None},
        }
    out = {
        "format": "21 MediaPipe hand landmarks, wrist-normalised (same as model features); "
                  "x right, y down, z towards camera negative; viewer's perspective as in "
                  "the dataset photos",
        "generated_by": "scripts/app/build_reference_poses.py",
        "how_to_verify": "python scripts/app/verify_poses.py --letters A --by NAME --reference SOURCE",
        "poses": poses,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {len(poses)} poses to {out_path}")


if __name__ == "__main__":
    main()
