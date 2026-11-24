"""
fsboard_video.py - All 21 hand points from FSboard's VIDEOS, ready for training.

FSboard's own landmark files hold only 8 of the 21 hand points (see fsboard.py),
so here we run MediaPipe Hands on the videos, the same tracker family the webcam
app uses. Used by notebooks/fsboard_tcn_ctc_colab.ipynb.

    metadata -> choose clips -> for each clip:
        download video to temporary disk -> check its MD5 -> turn it upright
        -> mirror it (exactly like asl/engine.py does with the webcam)
        -> MediaPipe Hands -> save a small .npz -> delete the video

What each saved file keeps: left and right hand as (frames, 21, 3), NaN where that
hand was not found, a mask for each hand, frame times, image size, and the clip's
phrase, signer and official split. x and y are fractions of the upright, mirrored
image. Nothing is normalised at this stage, so the features can change later without
touching the videos again.

Why MediaPipe runs in its own small Python environment: mediapipe 0.10.14 needs older
package versions than Colab's TensorFlow. ensure_mediapipe_env() creates a separate
Python 3.11 environment (the same versions as the laptop), and extraction runs there:

    python -m asl.seq.fsboard_video --clips clips.json --out DIR --tmp DIR
"""
import argparse
import html
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import fsboard as fb
from . import vocab
from ..features import normalise_points
from .ctc import character_error_rate, collapse_doubles, has_double_letter
from .features import CausalResampler, SequenceFeaturizer, canonical_points

SPLITS = ("train", "validation", "test")
FORMAT_VERSION = "fsboard-video-npz-v1"
LETTER_LIMITS = (2, 100)          # letters-only phrases of any sensible length
MEDIAPIPE_VERSION = "0.10.14"     # same as requirements.txt
MEDIAPIPE_PYTHON = "3.11"         # same as the laptop
FEATURE_SETTINGS = {
    "aspect_correction": True,    # see isotropic_points()
    "trim_to_annotation": False,  # True = drop the idle moments before and after signing
}


def metadata_path(split):
    return f"daun_v3/metadata/daun_v3-{split}.json"


def clip_path(row):
    return f"daun_v3/video_clips/daun_v3-{row['split']}/{row['filename']}"


# ============================================================ choosing clips
def select_clips(records, split, max_clips=0, seed=42, min_letters=LETTER_LIMITS[0],
                 max_letters=LETTER_LIMITS[1], skip_sensitive=True, phrases=False):
    """Usable clips of one official split, plus a count of why others were skipped.

    phrases=False: one-word clips (letters only), as before Phase 2B.
    phrases=True:  ONLY clips whose phrase has words separated by spaces, such as
                   "new york"; one-word clips are left to the other call, so the two
                   groups never overlap.
    max_clips = 0 means all of them. The same seed always gives the same clips.
    """
    rows, skipped = fb.eligible_rows(records, min_letters, max_letters, skip_sensitive, allow_spaces=phrases)
    if phrases:
        one_word = [r for r in rows if " " not in r["phrase"]]
        rows = [r for r in rows if " " in r["phrase"]]
        if one_word:
            skipped["one word (chosen with the letters-only clips)"] += len(one_word)
    index = fb.metadata_index(records)
    for row in rows:
        r = index[row["filename"]]
        row["split"] = split
        row["clip_start_s"] = r.get("clipStartTimeS")
        row["annotation_start_s"] = r.get("annotationStartTimeS")
        row["annotation_end_s"] = r.get("annotationEndTimeS")
    return limit_clips(rows, max_clips, seed), skipped


def limit_clips(rows, max_clips=0, seed=42):
    """A repeatable random subset, in filename order. 0 keeps them all."""
    rows = sorted(rows, key=lambda r: r["filename"])
    if max_clips and len(rows) > max_clips:
        rows = sorted(random.Random(seed).sample(rows, max_clips), key=lambda r: r["filename"])
    return rows


def choose_extra(pool, score, how_many, seed=42):
    """The `how_many` clips of `pool` that your `score` function wants most.

    score(text) gets the phrase in capitals ("NEW YORK") and returns a number: higher is
    picked earlier, 0 or less is never picked. Equal scores are broken at random, the same
    way on every run. Returns the chosen rows in filename order."""
    scored = []
    for row in sorted(pool, key=lambda r: r["filename"]):
        try:
            value = float(score(row["answer"]))
        except Exception as e:
            raise ValueError(f"Your clip_score function failed on {row['answer']!r}: {type(e).__name__}: {e}\n"
                             "What to check: it must take one text and return a number.") from e
        if value > 0:
            scored.append((value, row))
    random.Random(seed).shuffle(scored)                    # a fair, repeatable order among equal scores
    scored.sort(key=lambda item: -item[0])                 # stable: keeps that order within a score
    return sorted((row for _, row in scored[:max(0, int(how_many))]), key=lambda r: r["filename"])


def drop_seen_signers(rows, seen_signers):
    """Evaluation clips from people the model never trained on, and how many were removed."""
    kept = [r for r in rows if r["signer_id"] not in seen_signers]
    return kept, len(rows) - len(kept)


def html_table(rows, columns=None):
    """A small readable table for the notebook (list of dicts -> HTML)."""
    if not rows:
        return "<p><i>(nothing to show)</i></p>"
    columns = columns or list(rows[0])
    head = "".join(f"<th style='text-align:left;padding:2px 10px'>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td style='padding:2px 10px'>{html.escape(str(r.get(c, '')))}</td>"
                                    for c in columns) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


# ======================================================= reading video frames
def mp4_rotation(path):
    """(degrees clockwise to display upright, coded width, coded height) from an MP4.

    Many FSboard clips are stored sideways with a rotation flag, which browsers obey
    and OpenCV may not. Returns (0, 0, 0) if the file has no readable flag.
    """
    def boxes(buf, start, end):
        while start + 8 <= end:
            size, kind = struct.unpack(">I4s", buf[start:start + 8])
            head = 8
            if size == 1:
                size = struct.unpack(">Q", buf[start + 8:start + 16])[0]
                head = 16
            elif size == 0:
                size = end - start
            if size < head:
                return
            yield kind, start + head, start + size
            start += size

    with open(path, "rb") as f:
        total = os.path.getsize(path)
        pos = 0
        moov = None
        while pos + 8 <= total and moov is None:
            f.seek(pos)
            head = f.read(16)
            size, kind = struct.unpack(">I4s", head[:8])
            skip = 8
            if size == 1:
                size = struct.unpack(">Q", head[8:16])[0]
                skip = 16
            elif size == 0:
                size = total - pos
            if size < skip:
                break
            if kind == b"moov":
                f.seek(pos + skip)
                moov = f.read(size - skip)
            pos += size
    if moov is None:
        return 0, 0, 0
    for kind, a, b in boxes(moov, 0, len(moov)):
        if kind != b"trak":
            continue
        for sub, c, _ in boxes(moov, a, b):
            if sub != b"tkhd":
                continue
            matrix_at = c + (52 if moov[c] == 1 else 40)
            m = struct.unpack(">9i", moov[matrix_at:matrix_at + 36])
            width, height = struct.unpack(">II", moov[matrix_at + 36:matrix_at + 44])
            if not width or not height:
                continue                       # not the video track
            turn = (round(m[0] / 65536), round(m[1] / 65536), round(m[3] / 65536), round(m[4] / 65536))
            degrees = {(1, 0, 0, 1): 0, (0, 1, -1, 0): 90, (-1, 0, 0, -1): 180, (0, -1, 1, 0): 270}.get(turn, 0)
            return degrees, width >> 16, height >> 16
    return 0, 0, 0


def upright(frame, degrees, coded_w, coded_h):
    """Apply the stored rotation, unless the video reader already did."""
    import cv2

    h, w = frame.shape[:2]
    if degrees in (90, 270) and coded_w != coded_h and (w, h) == (coded_h, coded_w):
        return frame                           # already turned by the reader
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees)
    return frame if code is None else cv2.rotate(frame, code)


def landmarks_from_video(path, detection=0.5, tracking=0.5, long_side=960):
    """Every frame of one video -> both hands as (T, 21, 3) with masks and times.

    Each frame is turned upright and mirrored before tracking, like the webcam in
    asl/engine.py, so "left" and "right" mean the same as in the live app. Frames are
    shrunk first only for speed: the results are fractions of the image.
    """
    import cv2
    import mediapipe as mp

    degrees, coded_w, coded_h = mp4_rotation(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    found = {"left": [], "right": []}
    stamps = []
    size = None
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=detection,
                                  min_tracking_confidence=tracking) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000))
            stamps.append(max(stamp, stamps[-1] + 1) if stamps else stamp)
            frame = cv2.flip(upright(frame, degrees, coded_w, coded_h), 1)
            h, w = frame.shape[:2]
            size = (w, h)
            scale = long_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            best = {}
            for landmarks, handed in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
                label, score = handed.classification[0].label.lower(), handed.classification[0].score
                if label not in best or score > best[label][0]:
                    best[label] = (score, np.array([[p.x, p.y, p.z] for p in landmarks.landmark], np.float32))
            for hand in found:
                found[hand].append(best[hand][1] if hand in best else None)
    cap.release()
    if not stamps:
        raise ValueError("no frames could be read from this video")
    T = len(stamps)
    out = {"timestamp_us": np.array(stamps, np.int64), "frame_index": np.arange(T, dtype=np.int32),
           "frame_rate": fps, "image_width": size[0], "image_height": size[1]}
    for hand, frames in found.items():
        arr = np.full((T, 21, 3), np.nan, np.float32)
        for t, points in enumerate(frames):
            if points is not None:
                arr[t] = points
        out[f"{hand}_hand"] = arr
        out[f"{hand}_mask"] = np.array([p is not None for p in frames], bool)
    return out


def build_recording(row, landmarks):
    """Everything saved for one clip. Its answer comes from its own metadata row."""
    answer = fb.training_answer(row["phrase"])
    start = row.get("clip_start_s") or 0.0
    return dict(landmarks,
                clip_id=row["filename"], video_name=row["filename"], signer_id=row["signer_id"],
                split=row["split"], phrase=row["phrase"], answer=answer,
                label_ids=np.array(vocab.encode(answer), np.int32),
                landmark_ids=np.arange(21, dtype=np.int32), layout=fb.LAYOUT_COMPLETE,
                mirrored_input=True,
                annotation_start_s=float((row.get("annotation_start_s") or start) - start),
                annotation_end_s=float((row.get("annotation_end_s") or start) - start),
                source_shard="video", source_record=-1, format_version=FORMAT_VERSION)


def recording_path(row, out_dir):
    return Path(out_dir) / row["split"] / (Path(row["filename"]).stem + ".npz")


def save_recording(recording, path):
    """Appears under its final name only when complete, so a rerun can trust it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **recording)
    tmp.replace(path)


def extract_clips(clips, out_dir, tmp_dir, detection=0.5, tracking=0.5, log=print, stop_after_failures=5):
    """Download, track, save and delete, one clip at a time. Finished clips are skipped."""
    out_dir, tmp_dir = Path(out_dir), Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report = {"total": len(clips), "already_done": 0, "done": 0, "failed": [], "seconds": 0.0, "stopped": ""}
    started = time.time()
    in_a_row = 0
    for i, row in enumerate(clips, 1):
        out = recording_path(row, out_dir)
        if out.exists():
            report["already_done"] += 1
            continue
        video = tmp_dir / row["filename"]
        try:
            fb.download(clip_path(row), video, row["size_bytes"], row["md5"], log=lambda *_: None)
            recording = build_recording(row, landmarks_from_video(video, detection, tracking))
            save_recording(recording, out)
            in_a_row = 0
            report["done"] += 1
            if report["done"] <= 5 or report["done"] % 25 == 0:
                done = report["done"]
                left = (time.time() - started) / done * (len(clips) - i)
                hands = int(np.sum(recording["left_mask"] | recording["right_mask"]))
                log(f"[{i}/{len(clips)}] {row['answer']}: {len(recording['frame_index'])} frames, "
                    f"hand in {hands}; about {left / 60:.0f} min left")
        except Exception as problem:                       # one bad clip must not end the run
            in_a_row += 1
            report["failed"].append({"clip": row["filename"], "problem": str(problem)[:300]})
            log(f"[{i}/{len(clips)}] {row['filename']} FAILED: {str(problem).splitlines()[0][:150]}")
            if in_a_row >= stop_after_failures:
                report["stopped"] = str(problem)
                log(f"Stopping: {in_a_row} clips in a row failed. The last problem was:\n{problem}")
                break
        finally:
            video.unlink(missing_ok=True)
            video.with_name(video.name + ".part").unlink(missing_ok=True)
    report["seconds"] = round(time.time() - started, 1)
    return report


# =================================== running MediaPipe in its own environment
def check_mediapipe(python_exe):
    """Versions from that Python, or a plain-English error."""
    code = ("import json, cv2, numpy, mediapipe as mp; mp.solutions.hands; "
            "print(json.dumps({'mediapipe': mp.__version__, 'opencv': cv2.__version__, 'numpy': numpy.__version__}))")
    r = subprocess.run([str(python_exe), "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, GLOG_minloglevel="3"))
    if r.returncode:
        raise fb.FsboardError(f"MediaPipe could not start in {python_exe}.\nWhat to check: delete that folder in the "
                              f"Files panel and run this cell again.\nDetails: {r.stderr.strip()[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def ensure_mediapipe_env(env_dir, log=print):
    """A separate Python 3.11 + mediapipe 0.10.14 environment. Returns its python."""
    env_dir = Path(env_dir)
    python_exe = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python_exe.exists():
        try:
            log(f"MediaPipe environment already here: {check_mediapipe(python_exe)}")
            return python_exe
        except fb.FsboardError:
            shutil.rmtree(env_dir, ignore_errors=True)
    log(f"Installing Python {MEDIAPIPE_PYTHON} and mediapipe {MEDIAPIPE_VERSION} in their own folder ...")
    uv = shutil.which("uv")
    commands = [] if uv else [[sys.executable, "-m", "pip", "install", "-q", "uv"]]
    uv = uv or "uv"
    commands += [[uv, "venv", "--python", MEDIAPIPE_PYTHON, str(env_dir)],
                 [uv, "pip", "install", "-q", "--python", str(python_exe), f"mediapipe=={MEDIAPIPE_VERSION}", "numpy<2"]]
    for command in commands:
        r = subprocess.run(command, capture_output=True, text=True)
        if r.returncode:
            raise fb.FsboardError(f"Installing MediaPipe failed at: {' '.join(command[:3])}\nWhat to check: is this Colab "
                                  f"session still connected? Run the cell again.\nDetails: {r.stderr.strip()[-500:]}")
    log(f"Installed: {check_mediapipe(python_exe)}")
    return python_exe


def run_extraction(clips, out_dir, tmp_dir, python_exe, code_dir, log=print, detection=0.5, tracking=0.5):
    """Run extract_clips() in the MediaPipe environment and pass its progress along."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clips_file, report_file, errors_file = (tmp_dir / n for n in ("clips.json", "report.json", "errors.txt"))
    clips_file.write_text(json.dumps(clips), encoding="utf-8")
    report_file.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=str(code_dir), PYTHONUNBUFFERED="1", GLOG_minloglevel="3",
               TF_CPP_MIN_LOG_LEVEL="3")
    command = [str(python_exe), "-m", "asl.seq.fsboard_video", "--clips", str(clips_file), "--out", str(out_dir),
               "--tmp", str(tmp_dir), "--report", str(report_file), "--base-url", fb.BASE_URL,
               "--detection", str(detection), "--tracking", str(tracking)]
    with open(errors_file, "w", encoding="utf-8") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True, env=env)
        for line in process.stdout:
            log(line.rstrip())
        code = process.wait()
    if code or not report_file.exists():
        tail = errors_file.read_text(encoding="utf-8", errors="replace")[-800:]
        raise fb.FsboardError("Hand extraction stopped unexpectedly.\nWhat to check: run this cell again; finished "
                              f"clips are kept and skipped.\nDetails: {tail}")
    return json.loads(report_file.read_text(encoding="utf-8"))


# ======================================================== recordings -> features
def isotropic_points(points, width, height):
    """Undo stretching from the image's shape, so a hand has the same proportions in a
    portrait phone video and a landscape webcam.

    x and y are fractions of the image's width and height, which are different lengths.
    Here both become fractions of the LONGER side, measured about the image's centre.
    """
    p = np.array(points, np.float32, copy=True)
    longest = float(max(width, height))
    p[..., 0] = (p[..., 0] - 0.5) * width / longest + 0.5
    p[..., 1] = (p[..., 1] - 0.5) * height / longest + 0.5
    p[..., 2] = p[..., 2] * width / longest    # MediaPipe's z is on the same scale as x
    return p


def signing_hand(recording):
    """One hand per recording: the one found in more frames (ties: right)."""
    return "right" if recording["right_mask"].sum() >= recording["left_mask"].sum() else "left"


def recording_features(recording, feature_cfg):
    """A saved recording -> UNnormalised (T, 68) features at target_fps."""
    hand = signing_hand(recording)
    points = recording[f"{hand}_hand"]
    if feature_cfg.get("aspect_correction"):
        points = isotropic_points(points, recording["image_width"], recording["image_height"])
    points = canonical_points(points, hand)
    mask = recording[f"{hand}_mask"]
    seconds = (recording["timestamp_us"] - recording["timestamp_us"][0]) / 1e6
    keep = np.ones(len(seconds), bool)
    if feature_cfg.get("trim_to_annotation"):
        keep = (seconds >= recording["annotation_start_s"]) & (seconds <= recording["annotation_end_s"])
    resampler = CausalResampler(feature_cfg["target_fps"])
    featurizer = SequenceFeaturizer(feature_cfg)
    rows = []
    for t in np.flatnonzero(keep):
        for q in resampler.push(float(seconds[t]), points[t] if mask[t] else None):
            rows.append(featurizer.step(q))
    return np.stack(rows) if rows else np.zeros((0, 68), np.float32)


def build_examples(rows, out_dir, feature_cfg, max_frames=1500):
    """Saved recordings -> training examples. Unusable ones are excluded whole, with a reason."""
    examples, excluded = [], Counter()
    for row in rows:
        path = recording_path(row, out_dir)
        if not path.exists():
            excluded["no saved file (extraction failed)"] += 1
            continue
        recording = fb.load_example(path)
        labels = recording["label_ids"].tolist()
        needed = vocab.required_frames(labels)
        hands = int(recording[f"{signing_hand(recording)}_mask"].sum())
        if hands == 0:
            excluded["no hand found in any frame"] += 1
            continue
        if hands < needed:
            excluded["too few hand frames for its letters"] += 1
            continue
        x = recording_features(recording, feature_cfg)
        if len(x) < needed:
            excluded["too short for its letters"] += 1
        elif len(x) > max_frames:
            excluded["too long"] += 1
        else:
            examples.append({"x": x, "y": recording["label_ids"], "text": recording["answer"],
                             "sequence_id": recording["clip_id"], "signer": recording["signer_id"],
                             "split": recording["split"], "path": str(path),
                             "hand_share": hands / len(recording["frame_index"])})
    return examples, excluded


def check_recording(recording):
    """fsboard.check_example, for any official split, plus a check for all 21 points."""
    results = [r for r in fb.check_example(dict(recording, split="train")) if r[0] != "official split is train"]
    results.append(("all 21 hand points", recording["left_hand"].shape[1:] == (21, 3),
                    f"shape {recording['left_hand'].shape}"))
    return results


# ================================================================== evaluation
def letter_report(refs, hyps):
    """For each true letter: how often it was right, replaced by another, or missed.
    Also the most common confusions (true letter -> predicted letter)."""
    stats = {}
    pairs = Counter()
    for ref, hyp in zip(refs, hyps):
        n, m = len(ref), len(hyp)
        d = np.zeros((n + 1, m + 1), np.int32)
        d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
                s = stats.setdefault(ref[i - 1], Counter())
                s["seen"] += 1
                if ref[i - 1] == hyp[j - 1]:
                    s["right"] += 1
                else:
                    s["replaced"] += 1
                    pairs[f"{_shown(ref[i - 1])} -> {_shown(hyp[j - 1])}"] += 1
                i, j = i - 1, j - 1
            elif j > 0 and d[i, j] == d[i, j - 1] + 1:
                j -= 1
            else:
                s = stats.setdefault(ref[i - 1], Counter())
                s["seen"] += 1
                s["missed"] += 1
                i -= 1
    rows = [{"letter": "space" if c == " " else c, "seen": s["seen"], "right": s["right"], "replaced": s["replaced"], "missed": s["missed"],
             "accuracy": round(s["right"] / s["seen"], 3)} for c, s in stats.items()]
    rows.sort(key=lambda r: (r["accuracy"], r["letter"]))
    return {"letters": rows, "confusions": [{"pair": p, "count": c} for p, c in pairs.most_common(15)]}


def _shown(c):
    return "space" if c == " " else c


def _edit_distance(a, b):
    """Edits (insert, delete, replace) between two sequences of letters or words."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def _matched_spaces(ref, hyp):
    """Spaces of ref that the best alignment pairs with a space in hyp."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, found = n, m, 0
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            found += ref[i - 1] == " " == hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i, j - 1] + 1:
            j -= 1
        else:
            i -= 1
    return found


def phrase_scores(refs, hyps):
    """Scores for phrases with spaces (Phase 2B), next to the usual letter scores.

    letters_only_error_rate  wrong letters when spaces are ignored on both sides,
                             comparable with the one-word models
    word_error_rate          wrong, missing or extra WORDS per true word
    spaces_found             share of true spaces the model also wrote, in the right place
    extra_spaces             spaces the model wrote that are not in the truth
    """
    if not refs:
        return {"recordings": 0}
    plain = [r.replace(" ", "") for r in refs], [h.replace(" ", "") for h in hyps]
    true_spaces = sum(r.count(" ") for r in refs)
    matched = sum(_matched_spaces(r, h) for r, h in zip(refs, hyps))
    words = sum(len(r.split()) for r in refs)
    return {"recordings": len(refs),
            "character_error_rate": round(character_error_rate(refs, hyps), 4),
            "letters_only_error_rate": round(character_error_rate(*plain), 4),
            "word_error_rate": round(sum(_edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
                                     / max(1, words), 4),
            "exact_phrase_accuracy": round(float(np.mean([r == h for r, h in zip(refs, hyps)])), 4),
            "true_spaces": true_spaces,
            "spaces_found": round(matched / true_spaces, 4) if true_spaces else None,
            "extra_spaces": int(sum(h.count(" ") for h in hyps) - matched)}


def score_texts(refs, hyps):
    """The same scores for any model, so two models can be compared fairly."""
    doubled = [k for k, r in enumerate(refs) if has_double_letter(r)]
    return {"recordings": len(refs),
            "character_error_rate": round(character_error_rate(refs, hyps), 4) if refs else None,
            "exact_word_accuracy": round(float(np.mean([r == h for r, h in zip(refs, hyps)])), 4) if refs else None,
            "double_letter_words": len(doubled),
            "double_letter_word_accuracy": round(float(np.mean([refs[k] == hyps[k] for k in doubled])), 4) if doubled else None,
            "doubles_collapsed": int(sum(hyps[k] == collapse_doubles(refs[k]) for k in doubled))}


def main():
    ap = argparse.ArgumentParser(description="Extract 21 hand points from FSboard videos")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--base-url", default=fb.BASE_URL)
    ap.add_argument("--detection", type=float, default=0.5)
    ap.add_argument("--tracking", type=float, default=0.5)
    args = ap.parse_args()
    fb.BASE_URL = args.base_url
    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    report = extract_clips(clips, args.out, args.tmp, args.detection, args.tracking)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
