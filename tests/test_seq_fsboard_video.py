"""Tests for 21-point extraction from FSboard videos and the all-in-one Colab notebook.
They use fake metadata, a fake Kaggle server and small videos made from the real hand
photo inside tests/data/old_live_screenshot.png (the app's first screenshot), so they download nothing."""
import ast
import hashlib
import http.server
import json
import os
import struct
import sys
import threading
import types
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest

from asl.seq import fsboard as fb
from asl.seq import fsboard_video as fsv
from asl.seq import vocab
from asl.seq.features import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "fsboard_tcn_ctc_colab.ipynb"
CFG = dict(DEFAULT_CONFIG, **fsv.FEATURE_SETTINGS)


def meta(name, phrase="hello", signer="a", **kw):
    return dict({"signingType": "fs", "clipFilename": name, "phrase": phrase, "signerId": signer,
                 "clipFileMd5": "0" * 32, "clipFileSize": 10, "sensitiveContent": False,
                 "clipStartTimeS": 10.0, "annotationStartTimeS": 10.5, "annotationEndTimeS": 12.0}, **kw)


def recording(T=30, fps=30.0, left=(), right=None, phrase="hello", w=1836, h=3264):
    rng = np.random.default_rng(0)
    right = range(T) if right is None else right
    land = {"timestamp_us": (np.arange(T) * 1e6 / fps).astype(np.int64), "frame_index": np.arange(T, dtype=np.int32),
            "frame_rate": fps, "image_width": w, "image_height": h}
    for hand, frames in (("left", set(left)), ("right", set(right))):
        arr = np.full((T, 21, 3), np.nan, np.float32)
        for t in frames:
            arr[t] = rng.uniform(0.3, 0.7, (21, 3))
        land[f"{hand}_hand"], land[f"{hand}_mask"] = arr, np.array([t in frames for t in range(T)])
    row = {"filename": "a_1.mp4", "phrase": phrase, "signer_id": "a", "split": "train",
           "clip_start_s": 10.0, "annotation_start_s": 10.5, "annotation_end_s": 11.0}
    return fsv.build_recording(row, land)


# ------------------------------------------------------------- choosing clips
def test_only_letter_phrases_are_chosen_and_paths_follow_the_split():
    rows, skipped = fsv.select_clips([meta("a.mp4", "hello"), meta("b.mp4", "2067 pso costa"),
                                      meta("c.mp4", "x" * 20)], "validation")
    assert [r["answer"] for r in rows] == ["HELLO", "X" * 20]
    assert skipped == {"contains a digit": 1}
    assert fsv.clip_path(rows[0]) == "daun_v3/video_clips/daun_v3-validation/a.mp4"
    assert fsv.metadata_path("test") == "daun_v3/metadata/daun_v3-test.json"
    assert rows[0]["annotation_start_s"] == 10.5


def test_clip_limit_is_repeatable_and_zero_means_all():
    records = [meta(f"{i:03d}.mp4") for i in range(50)]
    first, _ = fsv.select_clips(records, "train", max_clips=10, seed=1)
    again, _ = fsv.select_clips(list(reversed(records)), "train", max_clips=10, seed=1)
    assert len(first) == 10 and [r["filename"] for r in first] == [r["filename"] for r in again]
    assert len(fsv.select_clips(records, "train", max_clips=0)[0]) == 50


def test_training_signers_are_removed_from_evaluation():
    rows, _ = fsv.select_clips([meta("a.mp4", signer="s1"), meta("b.mp4", signer="s2")], "test")
    kept, removed = fsv.drop_seen_signers(rows, {"s1"})
    assert [r["signer_id"] for r in kept] == ["s2"] and removed == 1


def test_extra_clips_follow_your_score_and_are_the_same_every_run():
    answers = ["QUIZ", "NEW YORK", "ANNA", "BOB", "QUIZ", "ZOO", "TEN", "QUIZ", "AT"]
    pool = [{"filename": f"{i:02d}.mp4", "answer": a} for i, a in enumerate(answers)]

    def score(text):
        return text.count("Q") * 2 + text.count("Z") + text.count("Y")

    top = fsv.choose_extra(pool, score, 4)
    assert sorted(r["answer"] for r in top) == ["QUIZ", "QUIZ", "QUIZ", "ZOO"] or \
        sorted(r["answer"] for r in top) == ["NEW YORK", "QUIZ", "QUIZ", "QUIZ"]     # ZOO and NEW YORK tie at 1
    assert top == fsv.choose_extra(list(reversed(pool)), score, 4)                    # repeatable, any input order
    assert [r["filename"] for r in top] == sorted(r["filename"] for r in top)
    everything = fsv.choose_extra(pool, score, 100)
    assert {r["answer"] for r in everything} == {"QUIZ", "NEW YORK", "ZOO"}           # score 0 is never picked
    assert fsv.choose_extra(pool, score, 0) == [] and fsv.choose_extra([], score, 5) == []
    assert fsv.choose_extra(pool, lambda text: -1, 5) == []
    with pytest.raises(ValueError, match="clip_score function failed on 'QUIZ'"):
        fsv.choose_extra(pool, lambda text: text.undefined, 2)


def test_phrase_selection_takes_only_spaced_phrases_so_groups_never_overlap():
    records = [meta("a.mp4", "hello"), meta("b.mp4", "new york"), meta("c.mp4", "i love you"),
               meta("d.mp4", "2067 pso costa"), meta("e.mp4", "new  york")]
    words, _ = fsv.select_clips(records, "train")
    phrases, skipped = fsv.select_clips(records, "train", phrases=True)
    assert [r["answer"] for r in words] == ["HELLO"]
    assert [r["answer"] for r in phrases] == ["NEW YORK", "I LOVE YOU"]
    assert skipped["one word (chosen with the letters-only clips)"] == 1 and skipped["contains a digit"] == 1
    saved = fsv.build_recording(dict(phrases[1], split="train"), recording(phrase="i love you"))
    assert saved["label_ids"].tolist()[1] == vocab.SPACE_ID and saved["answer"] == "I LOVE YOU"


def test_phrase_scores_count_words_and_spaces():
    s = fsv.phrase_scores(["I LOVE YOU", "NEW YORK"], ["I LOVEYOU", "NEW YORK"])
    assert s["true_spaces"] == 3 and s["spaces_found"] == round(2 / 3, 4) and s["extra_spaces"] == 0
    assert s["letters_only_error_rate"] == 0.0 and s["exact_phrase_accuracy"] == 0.5
    assert s["word_error_rate"] == round(2 / 5, 4)                 # LOVE YOU -> LOVEYOU: 1 replaced + 1 missing
    extra = fsv.phrase_scores(["CAT"], ["C AT"])
    assert extra["extra_spaces"] == 1 and extra["spaces_found"] is None
    assert fsv.phrase_scores([], [])["recordings"] == 0
    report = fsv.letter_report(["A B"], ["AB"])
    assert any(r["letter"] == "space" and r["missed"] == 1 for r in report["letters"])


def test_html_table_escapes_text():
    page = fsv.html_table([{"a": "<x>", "b": 1}])
    assert "&lt;x&gt;" in page and "<th" in page
    assert "nothing" in fsv.html_table([])


# ------------------------------------------------------------------- rotation
def fake_mp4(path, matrix, w=3264, h=1836, version=0, moov_last=True):
    """A minimal MP4 structure: enough boxes to hold a track's rotation flag."""
    body = bytes([version]) + b"\0\0\0" + b"\0" * (32 if version else 20) + b"\0" * 16
    body += struct.pack(">9i", *[v * 65536 for v in matrix[:2]], 0, *[v * 65536 for v in matrix[2:4]], 0, 0, 0, 1 << 30)
    body += struct.pack(">II", w << 16, h << 16)
    tkhd = struct.pack(">I4s", len(body) + 8, b"tkhd") + body
    trak = struct.pack(">I4s", len(tkhd) + 8, b"trak") + tkhd
    moov = struct.pack(">I4s", len(trak) + 8, b"moov") + trak
    mdat = struct.pack(">I4s", 108, b"mdat") + b"\0" * 100
    Path(path).write_bytes(mdat + moov if moov_last else moov + mdat)


@pytest.mark.parametrize("matrix,degrees", [((1, 0, 0, 1), 0), ((0, 1, -1, 0), 90), ((-1, 0, 0, -1), 180),
                                            ((0, -1, 1, 0), 270)])
@pytest.mark.parametrize("version", [0, 1])
def test_rotation_flag_is_read_wherever_it_is(tmp_path, matrix, degrees, version):
    for last in (True, False):
        fake_mp4(tmp_path / "v.mp4", matrix, version=version, moov_last=last)
        assert fsv.mp4_rotation(tmp_path / "v.mp4") == (degrees, 3264, 1836)


def test_a_file_without_the_flag_counts_as_upright(tmp_path):
    (tmp_path / "v.mp4").write_bytes(struct.pack(">I4s", 12, b"free") + b"\0\0\0\0")
    assert fsv.mp4_rotation(tmp_path / "v.mp4") == (0, 0, 0)


def test_upright_turns_a_sideways_frame_the_way_real_fsboard_clips_need():
    pytest.importorskip("cv2")
    frame = np.zeros((1836, 3264, 3), np.uint8)
    frame[10, 20] = 255                                     # near the top left
    up = fsv.upright(frame, 270, 3264, 1836)
    assert up.shape[:2] == (3264, 1836)
    # Checked on real clips: x_upright = y_stored, y_upright = 1 - x_stored
    assert up[3264 - 1 - 20, 10].max() == 255
    assert fsv.upright(up, 270, 3264, 1836) is up           # reader already turned it
    assert fsv.upright(frame, 0, 3264, 1836) is frame


# ------------------------------------------------------------------- features
def test_aspect_correction_makes_portrait_and_landscape_hands_match():
    width_px, height_px = 100.0, 200.0                       # a hand's real size, in pixels
    for w, h in ((1080, 1920), (1920, 1080), (640, 480)):
        pts = np.array([[0.5 - width_px / w / 2, 0.5, 0.0], [0.5, 0.5 + height_px / h / 2, 0.0]], np.float32)
        out = fsv.isotropic_points(pts, w, h)
        ratio = (out[1, 1] - out[0, 1]) / (out[1, 0] - out[0, 0])
        assert ratio == pytest.approx(height_px / 2 / (width_px / 2), rel=1e-5)
    centre = fsv.isotropic_points(np.array([[0.5, 0.5, 0.0]], np.float32), 1080, 1920)
    np.testing.assert_allclose(centre[0, :2], [0.5, 0.5])    # the middle stays the middle


def test_signing_hand_is_the_one_found_most():
    assert fsv.signing_hand(recording(left=[1, 2, 3], right=[1])) == "left"
    assert fsv.signing_hand(recording(left=[1], right=[1])) == "right"


def test_features_use_real_time_to_reach_30_fps_with_all_21_points():
    x = fsv.recording_features(recording(T=31, fps=15.0), CFG)
    assert x.shape == (61, 68) and np.all(np.isfinite(x))    # 2 s at 30 fps
    assert np.all(x[:, 67] == 1)


def test_missing_frames_become_zero_features_with_flag_off():
    x = fsv.recording_features(recording(T=30, right=range(0, 15)), CFG)
    assert x[5, 67] == 1 and x[25, 67] == 0 and np.all(x[25] == 0)


def test_trimming_keeps_only_the_annotated_part_when_asked():
    full = fsv.recording_features(recording(T=60), CFG)
    cut = fsv.recording_features(recording(T=60), dict(CFG, trim_to_annotation=True))
    assert 59 <= len(full) <= 60 and 14 <= len(cut) <= 17   # 2 s and 0.5 s (times round to microseconds)


def test_examples_leave_out_unusable_clips_with_reasons(tmp_path):
    rows = []
    for name, rec in (("good", recording(T=40)), ("nohand", recording(T=40, right=[])),
                      ("few", recording(T=40, right=[1, 2]))):
        row = {"filename": f"{name}.mp4", "split": "train"}
        fsv.save_recording(rec, fsv.recording_path(row, tmp_path))
        rows.append(row)
    rows.append({"filename": "lost.mp4", "split": "train"})
    examples, excluded = fsv.build_examples(rows, tmp_path, CFG)
    assert len(examples) == 1 and examples[0]["text"] == "HELLO"
    assert excluded == {"no hand found in any frame": 1, "too few hand frames for its letters": 1,
                        "no saved file (extraction failed)": 1}


def test_saved_recording_passes_checks_for_any_official_split(tmp_path):
    row = {"filename": "x.mp4", "split": "test"}
    path = fsv.recording_path(row, tmp_path)
    fsv.save_recording(dict(recording(T=40), split="test"), path)
    assert path.parent.name == "test" and not list(path.parent.glob("*.tmp.npz"))
    results = fsv.check_recording(fb.load_example(path))
    assert all(ok for _, ok, _ in results), results
    assert ("all 21 hand points", True, "shape (40, 21, 3)") in results


# ----------------------------------------------------------------- evaluation
def test_letter_report_counts_right_replaced_and_missed():
    report = fsv.letter_report(["HELLO", "CAT"], ["HELO", "CUT"])
    by = {r["letter"]: r for r in report["letters"]}
    assert by["L"]["seen"] == 2 and by["L"]["right"] == 1 and by["L"]["missed"] == 1
    assert by["A"]["replaced"] == 1 and by["A"]["accuracy"] == 0
    assert report["confusions"] == [{"pair": "A -> U", "count": 1}]
    assert report["letters"][0]["accuracy"] <= report["letters"][-1]["accuracy"]


def test_scores_measure_collapsed_doubles():
    s = fsv.score_texts(["HELLO", "CAT"], ["HELO", "CAT"])
    assert s["exact_word_accuracy"] == 0.5 and s["character_error_rate"] == pytest.approx(1 / 8)
    assert s["double_letter_words"] == 1 and s["doubles_collapsed"] == 1
    assert fsv.score_texts([], [])["recordings"] == 0


# ---------------------------------------------------- real MediaPipe extraction
SCREENSHOT = ROOT / "tests" / "data" / "old_live_screenshot.png"
DETECT = 0.2                 # this small, blurry test photo needs a low threshold
needs_photos = pytest.mark.skipif(not SCREENSHOT.exists(), reason="project screenshot missing")


def hand_photo():
    """The real hand inside the screenshot, with the drawn landmarks painted out and
    space added around it: MediaPipe finds the hand in this version."""
    import cv2

    crop = cv2.imread(str(SCREENSHOT))[137:408, 785:1056]
    dots = (crop[..., 0].astype(int) - crop[..., 2].astype(int)) > 60
    lines = crop.min(axis=2) > 225
    mask = cv2.dilate((dots | lines).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    clean = cv2.resize(cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA), (400, 400), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(clean, 133, 133, 133, 133, cv2.BORDER_REPLICATE)


def photo_video(path, plan, fps=15):
    """plan: list of (anything for "hand visible", or None for "no hand", seconds)."""
    import cv2

    photo = hand_photo()
    size = photo.shape[0]
    blank = np.full((size, size, 3), 235, np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    k = 0
    for what, seconds in plan:
        for _ in range(round(seconds * fps)):
            k += 1
            writer.write(blank if what is None else np.roll(photo, (k % 5) - 2, axis=1))
    writer.release()
    return path


@needs_photos
def test_real_mediapipe_finds_21_points_and_the_masks_match_the_video(tmp_path):
    pytest.importorskip("mediapipe")
    video = photo_video(tmp_path / "v.mp4", [("B", 1.0), (None, 0.6), ("B", 1.0)])
    land = fsv.landmarks_from_video(video, detection=DETECT)
    T = len(land["frame_index"])
    assert T == 39 and land["image_width"] == land["image_height"] == 666
    hand = land["left_mask"] | land["right_mask"]
    assert hand[:15].mean() > 0.5 and hand[-15:].mean() > 0.5    # hand in the photos
    assert hand[15:24].mean() < 0.3                              # blank wall in the middle
    which = "left" if land["left_mask"].sum() > land["right_mask"].sum() else "right"
    pts = land[f"{which}_hand"]
    assert pts.shape == (39, 21, 3)
    assert np.isfinite(pts[land[f"{which}_mask"]]).all() and np.isnan(pts[~land[f"{which}_mask"]]).all()
    assert np.all(np.diff(land["timestamp_us"]) > 0)


@needs_photos
def test_sideways_video_is_turned_upright_before_tracking(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    import cv2

    video = photo_video(tmp_path / "v.mp4", [("B", 1.0)])
    plain = fsv.landmarks_from_video(video, detection=DETECT)
    # Store the same frames sideways, and pretend the file carries the rotation flag.
    cap, frames = cv2.VideoCapture(str(video)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
    cap.release()
    writer = cv2.VideoWriter(str(tmp_path / "s.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, frames[0].shape[1::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(fsv, "mp4_rotation", lambda path: (270, 666, 400))
    turned = fsv.landmarks_from_video(tmp_path / "s.mp4", detection=DETECT)
    a, b = ("left_hand", "left_mask") if plain["left_mask"].sum() else ("right_hand", "right_mask")
    both = plain[b] & turned[b]
    assert both.sum() >= 5
    assert np.nanmedian(np.abs(plain[a][both][:, :, :2] - turned[a][both][:, :, :2])) < 0.03


# ------------------------------------------------------------ fake Kaggle server
class Service(http.server.BaseHTTPRequestHandler):
    files = {}
    requested = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.lstrip("/")
        type(self).requested.append(name)
        if name not in self.files:
            self.send_error(404)
            return
        data = self.files[name]
        spec = self.headers.get("Range")
        if spec:
            a, b = spec.split("=")[1].split("-")
            a = int(a)
            b = int(b) if b else len(data) - 1
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def kaggle(monkeypatch):
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    Service.files, Service.requested = {}, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    monkeypatch.setattr(fb, "BASE_URL", url)
    yield types.SimpleNamespace(url=url)
    server.shutdown()
    server.server_close()


def clip_row(name, data, phrase="bob", signer="a", split="train"):
    return {"filename": name, "phrase": phrase, "answer": phrase.upper(), "signer_id": signer, "split": split,
            "md5": hashlib.md5(data).hexdigest(), "size_bytes": len(data), "clip_start_s": 0.0,
            "annotation_start_s": 0.1, "annotation_end_s": 1.0}


@needs_photos
def test_extraction_saves_points_deletes_videos_and_skips_finished_clips(tmp_path, kaggle):
    pytest.importorskip("mediapipe")
    data = photo_video(tmp_path / "src.mp4", [("B", 1.0), ("O", 1.0)]).read_bytes()
    row = clip_row("one.mp4", data)
    Service.files[quote(fsv.clip_path(row), safe="")] = data
    videos = tmp_path / "videos"
    report = fsv.run_extraction([row], tmp_path / "out", videos, sys.executable, ROOT / "backend",
                                log=lambda *_: None, detection=DETECT)
    assert report["done"] == 1 and report["failed"] == []
    saved = fb.load_example(fsv.recording_path(row, tmp_path / "out"))
    assert saved["answer"] == "BOB" and saved["signer_id"] == "a" and saved["left_hand"].shape[1:] == (21, 3)
    assert saved["mirrored_input"] and saved["annotation_start_s"] == pytest.approx(0.1)
    assert not list(videos.glob("*.mp4")) and not list(videos.glob("*.part"))
    asked = len(Service.requested)
    again = fsv.run_extraction([row], tmp_path / "out", videos, sys.executable, ROOT / "backend", log=lambda *_: None)
    assert again["already_done"] == 1 and len(Service.requested) == asked   # nothing downloaded twice


def test_bad_clips_are_reported_and_repeated_failures_stop_the_run(tmp_path, kaggle):
    rows = [clip_row(f"{i}.mp4", b"x") for i in range(9)]
    messages = []
    report = fsv.extract_clips(rows, tmp_path / "out", tmp_path / "v", log=messages.append)
    assert len(report["failed"]) == 5 and report["done"] == 0
    assert "does not exist" in report["stopped"] and any("Stopping" in m for m in messages)


def test_a_broken_mediapipe_gets_a_specific_message(tmp_path):
    with pytest.raises((fb.FsboardError, OSError)):
        fsv.check_mediapipe(tmp_path / "missing" / "python")
    with pytest.raises(fb.FsboardError, match="MediaPipe could not start"):
        fsv.check_mediapipe(_python_without_mediapipe(tmp_path))


def _python_without_mediapipe(tmp_path):
    script = tmp_path / ("fake.bat" if os.name == "nt" else "fake.sh")
    script.write_text("@echo off\r\nexit /b 1\r\n" if os.name == "nt" else "#!/bin/sh\nexit 1\n")
    script.chmod(0o755)
    return script


# ------------------------------------------------------------------ the notebook
def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_carries_the_exact_repository_code():
    cells = code_cells()
    files = next(ast.literal_eval(n.value) for n in ast.parse(cells[0]).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_FILES")
    assert "asl/seq/fsboard_video.py" in files and "asl/seq/train.py" in files
    for name, text in files.items():
        assert text == (ROOT / "backend" / name).read_text(encoding="utf-8"), \
            f"{name} changed: run scripts/sequence_model/build_tcn_ctc_notebook.py"
    assert not any("asl_mlp" in c or "BASELINE_DIR" in c for c in cells[1:])   # the baseline copy was removed


def test_notebook_has_no_outputs_and_safe_defaults():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    cells = code_cells()
    assert any("CLIPS_TRAIN = 0 " in c for c in cells) and any("INCLUDE_PHRASES = True" in c for c in cells)
    assert any("def clip_score(text):" in c and "EXTRA_TRAIN_CLIPS = 3000 " in c for c in cells)
    assert any("DOWNLOAD_TO_MY_COMPUTER = False" in c for c in cells)
    assert any("FINE_TUNE_ON_MY_RECORDINGS = False" in c for c in cells)      # off unless switched on
    assert sum("strain.fit(" in c for c in cells) == 3                        # tiny test, training, fine-tune


@needs_photos
def test_whole_notebook_runs_from_metadata_to_exported_model(tmp_path, kaggle, monkeypatch, capsys):
    """Every cell in order, like Run all, against the fake server and fake Colab."""
    pytest.importorskip("mediapipe")
    tf = pytest.importorskip("tensorflow")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    # Small videos from the photos, one signer per clip.
    plans = {"bob": [("B", .5), ("O", .5), ("B", .5)], "hello": [("H", .4), ("E", .4), ("L", .4), ("O", .4)],
             "boo": [("B", .5), ("O", .8)], "oak": [("O", .4), ("B", .4), ("K", .4)]}
    records = {"train": [], "validation": [], "test": []}
    number = 0
    for split, phrases in (("train", ["bob", "hello", "boo", "oak", "bob oak"]), ("validation", ["bob", "boo", "boo bob"]),
                           ("test", ["oak", "hello", "hello bob"])):
        for phrase in phrases:
            number += 1
            name = f"s{number}_0_20220101_000000_0_{number}.mp4"
            plan = [step for word in phrase.split() for step in plans[word]]
            data = photo_video(tmp_path / f"src{number}.mp4", plan, fps=15).read_bytes()
            Service.files[quote(f"daun_v3/video_clips/daun_v3-{split}/{name}", safe="")] = data
            records[split].append(meta(name, phrase, f"s{number}", clipFileMd5=hashlib.md5(data).hexdigest(),
                                       clipFileSize=len(data), clipStartTimeS=0.0, annotationStartTimeS=0.0,
                                       annotationEndTimeS=1.0))
    records["train"].append(meta("skip.mp4", "2067 pso costa", "s1"))
    records["test"].append(meta("seen.mp4", "bob", "s1"))           # a training signer: must be dropped
    for split, recs in records.items():
        Service.files[quote(fsv.metadata_path(split), safe="")] = json.dumps(recs).encode()

    downloads = []
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: (_ for _ in ()).throw(KeyError(name)))
    colab.files = types.SimpleNamespace(download=downloads.append)
    colab.drive = types.SimpleNamespace(mount=lambda path: pytest.fail("Drive should not be needed in this test"))
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "colab", colab, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    shown = []
    display = types.ModuleType("IPython.display")
    display.HTML, display.display = (lambda text: text), shown.append
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    ipython.version_info = (8, 0, 0)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setenv("FSBOARD_WORK", str(tmp_path / "content"))
    monkeypatch.setenv("FSBOARD_DRIVE", str(tmp_path / "drive"))
    monkeypatch.setenv("FSBOARD_MP_PYTHON", sys.executable)

    # Your earlier long-memory model (27 outputs, no space): step 6 starts from it, steps 7-8 compare with it.
    from asl.seq.features import Normalizer, preprocessing_dict
    from asl.seq.model import DEFAULT_MODEL_CONFIG, build_tcn_ctc, save_inference_bundle
    earlier = tmp_path / "drive" / "runs" / "train8732_val2103_test2131_letters2-100_phrases40__speed_long_memory" / "best_model"
    earlier_cfg = dict(DEFAULT_MODEL_CONFIG, dilations=[1, 2, 4, 8, 16], num_classes=27)
    save_inference_bundle(earlier, build_tcn_ctc(earlier_cfg), earlier_cfg,
                          preprocessing_dict(Normalizer(np.zeros(68), np.ones(68)), dict(fsv.FEATURE_SETTINGS)))

    faster = [('"epochs": 300', '"epochs": 3'), ("RUN_TINY_TEST = False", "RUN_TINY_TEST = True"),
              ("EPOCHS = 40 ", "EPOCHS = 3 "), ("DETECTION_CONFIDENCE = 0.5 ", f"DETECTION_CONFIDENCE = {DETECT} ")]
    scope = {"__name__": "__main__"}
    for i, source in enumerate(code_cells()):
        for old, new in faster:
            source = source.replace(old, new)
        exec(compile(source, f"cell{i}", "exec"), scope)
        if i == 0:
            monkeypatch.setattr(scope["fb"], "BASE_URL", kaggle.url)
    out = capsys.readouterr().out

    run = tmp_path / "drive" / "runs" / scope["RUN_NAME"]
    assert scope["RUN_NAME"] == "train5_val3_test3_letters2-100_phrases40__speed_long_memory"
    assert "Warm start from train8732_val2103_test2131_letters2-100_phrases40__speed_long_memory: copied" in out
    assert "extended ['logits'] with the space output" in out
    assert "Best error rate on the memorised clips" in out and "Skipped: tick FINE_TUNE_ON_MY_RECORDINGS" in out
    assert "seen.mp4" not in " ".join(r["filename"] for r in scope["clips"]["test"])
    assert {e["split"] for e in scope["examples"]["train"]} == {"train"}
    assert "failed checks: 0" in out and "signers in both training and evaluation: 0" in out
    # 21 points on Drive, videos gone from temporary disk
    assert not list((tmp_path / "content" / "videos").glob("*.mp4"))
    assert len(list((tmp_path / "drive" / "sequences").rglob("*.npz"))) == 11
    # resumable checkpoint + best model + report + export
    assert (run / "checkpoints" / "last" / "optimizer.npz").exists()
    report = json.loads((run / "best_model" / "evaluation_report.json").read_text())
    assert set(report) == {"validation", "test", "comparison"}
    assert {(r["model"], r["group"]) for r in report["comparison"]} == {
        (m, g) for m in ("this run", "earlier model") for g in ("one-word test clips", "phrase test clips")}
    phrase_row = next(r for r in report["comparison"] if r["group"] == "phrase test clips")
    assert phrase_row["recordings"] == 1 and "spaces_found" in phrase_row
    speed = json.loads((run / "best_model" / "speed_report.json").read_text())
    assert len(speed) == 8 and {r["model"] for r in speed} == {"this run", "earlier model"}
    assert report["test"]["signers_unfamiliar"] is True
    assert report["test"]["scores"]["recordings"] == 3
    assert report["test"]["by_letter"]["letters"]
    assert (tmp_path / "drive" / "export" / f"{scope['RUN_NAME']}.zip").exists()
    assert downloads == []                                                # nothing sent to the laptop by default
    # the saved model reproduces itself
    from asl.seq.model import check_reference_io, load_inference_bundle
    model, _, prep = load_inference_bundle(run / "best_model")
    assert check_reference_io(run / "best_model", model)["ok"]
    assert prep["features"]["aspect_correction"] is True
    tf.keras.backend.clear_session()


def test_training_resumes_after_a_disconnect(tmp_path):
    """A finished short run continues to more epochs from its checkpoint."""
    tf = pytest.importorskip("tensorflow")
    from asl.seq import features, model as smodel, train as strain

    rng = np.random.default_rng(0)
    ex = []
    for k, word in enumerate(["BOB", "HELLO", "BOO"]):
        x = rng.normal(size=(40, 68)).astype(np.float32)
        x[:, 67] = 1
        ex.append({"x": x, "y": np.array(vocab.encode(word), np.int32), "text": word, "signer": str(k), "sequence_id": word})
    n = features.Normalizer().fit([e["x"] for e in ex])
    prep = features.preprocessing_dict(n)
    logs = []
    cfg = {"epochs": 2, "batch_size": 2, "augment": False, "show_examples": 0, "seed": 1}
    strain.fit(smodel.build_tcn_ctc(), ex, ex, n, tmp_path / "run", prep, {}, cfg=cfg, log=logs.append)
    _, state = strain.fit(smodel.build_tcn_ctc(), ex, ex, n, tmp_path / "run", prep, {}, cfg=dict(cfg, epochs=3), log=logs.append)
    assert any("Resumed from epoch 2" in m for m in logs) and state["epoch"] == 3
    tf.keras.backend.clear_session()
