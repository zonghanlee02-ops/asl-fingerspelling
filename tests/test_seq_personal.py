"""Phase 2A: word lists, word-list recording sessions, packing your recordings,
and the Colab fine-tuning cell (run here end to end on fake data)."""
import hashlib
import json
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts" / "sequence_model"
NOTEBOOK = ROOT / "notebooks" / "fsboard_tcn_ctc_colab.ipynb"


def words_in(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


# ------------------------------------------------------------------ word lists
def test_word_lists_are_letters_only_unique_and_never_overlap():
    train, test = words_in(DATA / "my_words_train.txt"), words_in(DATA / "my_words_test.txt")
    assert len(train) == 100 and len(test) == 40
    assert len(set(train)) == len(train) and len(set(test)) == len(test)
    assert all(w.isascii() and w.isalpha() and w.isupper() for w in train + test)
    assert not set(train) & set(test)


def test_word_lists_cover_every_letter_double_letters_and_moving_letters():
    train, test = words_in(DATA / "my_words_train.txt"), words_in(DATA / "my_words_test.txt")
    learned = [w for i, w in enumerate(train) if i % 7 != 6]       # what is left after validation
    for words in (learned, test):
        assert set("".join(words)) == set(string.ascii_uppercase)
        assert sum(any(a == b for a, b in zip(w, w[1:])) for w in words) >= 5
    counts = Counter("".join(learned))
    assert min(counts.values()) >= 4 and counts["J"] >= 4 and counts["Z"] >= 4


def test_phrase_and_double_letter_lists_are_clean_and_never_leak_test_words():
    lists = {name: words_in(DATA / name) for name in ("my_words_train.txt", "my_words_test.txt",
                                                       "my_phrases_train.txt", "my_phrases_test.txt",
                                                       "my_doubles_train.txt", "my_i_words_train.txt")}
    everything = [w for words in lists.values() for w in words]
    assert len(everything) == len(set(everything))                  # nothing listed twice anywhere
    for name in ("my_phrases_train.txt", "my_phrases_test.txt"):
        assert all(" " in p and "  " not in p and p.replace(" ", "").isalpha() and p.isupper() for p in lists[name])
    test_words = set(lists["my_words_test.txt"])
    assert not {w for p in lists["my_phrases_train.txt"] for w in p.split()} & test_words
    assert all(any(a == b for a, b in zip(w, w[1:])) for w in lists["my_doubles_train.txt"])
    assert len(lists["my_phrases_train.txt"]) >= 40 and len(lists["my_phrases_test.txt"]) >= 10
    i_words = lists["my_i_words_train.txt"]
    assert i_words[0] == "I" and all(w.startswith("I") for w in i_words)  # "I" alone is never validation
    assert not any(a == b for w in i_words for a, b in zip(w, w[1:]))       # no double letters mixed in


# ----------------------------------------------------------------- recorder
@pytest.fixture
def recorder(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import seq_record_word
    return seq_record_word


def moving_take(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return (100 + np.arange(n) / 30).tolist(), [rng.uniform(0.3, 0.7, (21, 3)).astype(np.float32) for _ in range(n)]


def test_word_list_reading_rejects_bad_lines(recorder, tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("# comment\n\n cat \nDog\n")
    assert recorder.read_word_list(good) == ["CAT", "DOG"]
    for text, problem in (("CAT\n2 DOGS\n", "line 2"), ("CAT\ncat\n", "twice"), ("# only a comment\n", "no words")):
        bad = tmp_path / "bad.txt"
        bad.write_text(text)
        with pytest.raises(ValueError, match=problem):
            recorder.read_word_list(bad)


def test_paces_and_prompts(recorder):
    assert recorder.parse_paces("slow, Normal") == ["slow", "normal"]
    for wrong in ("", "slow,slow", "fast"):
        with pytest.raises(Exception, match="comma-separated"):
            recorder.parse_paces(wrong)
    prompts = recorder.session_prompts(["CAT", "DOG"], ["slow", "normal"])
    assert prompts == [("CAT", "slow"), ("CAT", "normal"), ("DOG", "slow"), ("DOG", "normal")]
    done = Counter({("CAT", "slow"): 1, ("DOG", "normal"): 2})
    assert recorder.remaining_prompts(prompts, done) == [("CAT", "normal"), ("DOG", "slow")]


def test_saved_takes_are_counted_and_discarded_ones_are_not(recorder, tmp_path):
    t, pts = moving_take(10)
    a = recorder.save_recording(tmp_path, t, pts, ["right"] * 10, "right", "CAT", True, (640, 480), 1.0, "CAT", "slow")
    b = recorder.save_recording(tmp_path, t, pts, ["right"] * 10, "right", "CAT", True, (640, 480), 1.0, "CAT", "slow")
    assert a != b and a.name.startswith("CAT_slow_") and b.exists()          # same second, both kept
    with np.load(a, allow_pickle=False) as z:
        assert str(z["pace"]) == "slow" and str(z["hand_mode"]) == "auto"
    assert recorder.recorded_takes(tmp_path) == Counter({("CAT", "slow"): 2})
    moved = recorder.discard_recording(b)
    assert moved.parent.name == "discarded" and moved.exists() and not b.exists()
    assert recorder.recorded_takes(tmp_path) == Counter({("CAT", "slow"): 1})


@pytest.mark.parametrize("n,hand_every,word,flipped,problem", [
    (60, 1, "CAT", False, None),
    (60, 0, "CAT", False, "no hand"),
    (10, 1, "CAT", False, "too short"),
    (60, 3, "CAT", False, "lost in"),
    (20, 1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", False, "too short for the letters"),
    (60, 1, "CAT", True, "flipped"),
])
def test_a_bad_take_is_explained(recorder, n, hand_every, word, flipped, problem):
    t, pts = moving_take(n)
    used = [p if hand_every and i % hand_every == 0 else None for i, p in enumerate(pts)]
    found = recorder.recording_problem(t, used, word, 30.0, flipped)
    assert (found is None) if problem is None else (problem in found)


def test_word_list_session_saves_skips_redoes_and_refuses_bad_takes(recorder, monkeypatch, tmp_path, capsys):
    from test_seq_model import _fake_camera

    words = tmp_path / "words.txt"
    words.write_text("CAT\nDOG\n")
    save = tmp_path / "takes"
    t, pts = moving_take(30)
    recorder.save_recording(save, t, pts, ["right"] * 30, "right", "CAT", True, (640, 480), 1.0, "CAT", "slow")
    keys = ("." + "s" + "." * 30 + "f"        # CAT normal: saved
            + "n"                             # DOG slow: skipped
            + "s" + "." * 30 + "f"            # DOG normal: saved
            + "r"                             # ...that take was wrong: discarded, asked again
            + "s" + ".." + "f"                # too short: refused, not saved
            + "q")
    _fake_camera(monkeypatch, recorder, ["--untrained", "--mirror-input", "--words", str(words), "--paces",
                                         "slow,normal", "--save", str(save)], keys, [])
    out = capsys.readouterr().out
    assert "4 takes, 1 already saved" in out and "skipped DOG (slow)" in out and "NOT saved: the recording is too short" in out
    kept = sorted(p.name.rsplit("_", 2)[0] for p in save.glob("*.npz"))
    assert kept == ["CAT_normal", "CAT_slow"]
    assert [p.name.rsplit("_", 2)[0] for p in (save / "discarded").glob("*.npz")] == ["DOG_normal"]
    with np.load(next(save.glob("CAT_normal_*.npz")), allow_pickle=False) as z:
        assert str(z["word"]) == "CAT" and str(z["word_list"]) == "words" and len(z["t"]) == 31


def test_word_list_needs_a_save_folder_and_not_a_single_word(recorder, monkeypatch, tmp_path):
    words = tmp_path / "words.txt"
    words.write_text("CAT\n")
    for args in (["--untrained", "--words", str(words)],
                 ["--untrained", "--words", str(words), "--save", str(tmp_path), "--word", "DOG"]):
        monkeypatch.setattr(sys, "argv", ["seq_record_word.py", *args])
        with pytest.raises(SystemExit):
            recorder.main()


# ------------------------------------------------------------------ packing
@pytest.fixture
def packer(recorder):
    import seq_pack_my_recordings
    return seq_pack_my_recordings


@pytest.fixture
def bundle(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from asl.seq.features import Normalizer, preprocessing_dict
    from asl.seq.fsboard_video import FEATURE_SETTINGS
    from asl.seq.model import DEFAULT_MODEL_CONFIG, build_tcn_ctc, save_inference_bundle

    tf.keras.utils.set_random_seed(0)
    cfg = dict(DEFAULT_MODEL_CONFIG, channels=8, dilations=[1, 2])
    rng = np.random.default_rng(1)
    n = Normalizer().fit([np.c_[rng.normal(size=(40, 67)), np.ones(40)].astype(np.float32)])
    folder = save_inference_bundle(tmp_path / "base" / "best_model", build_tcn_ctc(cfg), cfg,
                                   preprocessing_dict(n, dict(FEATURE_SETTINGS)))
    (tmp_path / "base" / "train_config.json").write_text(json.dumps({"augment": {
        "stretch": [0.5, 3.5], "rotate_deg": 10.0, "scale": [0.9, 1.1], "aspect": [0.9, 1.1],
        "wrist_shift": 0.05, "noise": 0.01, "drop_frame": 0.03}}))
    return folder


def make_takes(recorder, folder, phrases=False):
    """Takes for train list CAT DOG SUN MAP BUS PEN HAT (HAT = validation) and test list FOX;
    with phrases=True also the training phrase GO HOME and the test phrase BE KIND."""
    plan = [("CAT", "slow", 60, True), ("DOG", "normal", 50, True), ("MAP", "quick", 45, True),
            ("HAT", "slow", 60, True), ("FOX", "slow", 70, True), ("FOX", "normal", 40, True),
            ("ZIP", "slow", 60, True),            # in neither list
            ("BUS", "slow", 60, False),           # recorded without --mirror-input
            ("SUN", "slow", 2, True),             # too short
            ("", "", 60, True)]                   # no word
    if phrases:
        plan += [("GO HOME", "slow", 80, True), ("BE KIND", "slow", 80, True)]
    for i, (word, pace, n, mirror) in enumerate(plan):
        t, pts = moving_take(n, seed=i)
        recorder.save_recording(folder, t, pts, ["right"] * n, "right", word, mirror, (640, 480), 1.0, word, pace)
    t, pts = moving_take(60, seed=99)
    discarded = recorder.save_recording(folder, t, pts, ["right"] * 60, "right", "PEN", True, (640, 480), 1.0, "PEN", "slow")
    recorder.discard_recording(discarded)


def write_lists(tmp_path, phrases=False):
    (tmp_path / "train.txt").write_text("CAT\nDOG\nSUN\nMAP\nBUS\nPEN\nHAT\n")
    (tmp_path / "test.txt").write_text("FOX\n")
    (tmp_path / "phrases.txt").write_text("GO HOME\n")
    (tmp_path / "test_phrases.txt").write_text("BE KIND\n")
    train = [str(tmp_path / "train.txt")] + ([str(tmp_path / "phrases.txt")] if phrases else [])
    test = [str(tmp_path / "test.txt")] + ([str(tmp_path / "test_phrases.txt")] if phrases else [])
    return ["--train-words", *train, "--test-words", *test]


def test_groups_follow_the_word_lists_and_overlap_is_refused(packer):
    groups = packer.word_groups([["A", "B", "C"], ["GO HOME", "BIG DOG", "RED"]], [["D"], ["SEE YOU"]], every=3)
    assert groups == {"A": "train", "B": "train", "C": "validation", "GO HOME": "train", "BIG DOG": "train",
                      "RED": "validation", "D": "test", "SEE YOU": "test"}
    with pytest.raises(ValueError, match="both"):
        packer.word_groups([["CAT"]], [["CAT"]])
    with pytest.raises(ValueError, match="two training lists"):
        packer.word_groups([["CAT"], ["CAT"]], [["DOG"]])
    with pytest.raises(ValueError, match="contain a test word"):
        packer.word_groups([["MY MOTHER"]], [["MOTHER"]])


def test_pack_keeps_good_takes_with_the_webcam_apps_exact_inputs(recorder, packer, bundle, tmp_path, capsys):
    from asl.seq.inference import WordRecognizer, frame_features

    takes = tmp_path / "takes"
    make_takes(recorder, takes)
    out = tmp_path / "pack.npz"
    summary = packer.main([str(takes), *write_lists(tmp_path), "--bundle", str(bundle), "--out", str(out)])
    assert summary["excluded"] == {"word is in neither word list": 1, "recorded without --mirror-input": 1,
                                   "too few frames with a hand for its letters": 1, "no word saved with it": 1}
    assert {s: g["takes"] for s, g in summary["groups"].items()} == {"train": 3, "validation": 1, "test": 2}
    assert "Saved" in capsys.readouterr().out

    examples, cfg = packer.load_pack(out)
    assert cfg == json.loads((bundle / "preprocessing.json").read_text())["features"]
    cat = next(e for e in examples if e["text"] == "CAT")
    assert cat["split"] == "train" and cat["pace"] == "slow" and cat["y"].tolist() == [3, 1, 20]
    t, pts = moving_take(60, seed=0)
    assert np.allclose(cat["x"], frame_features([s - t[0] for s in t], pts, "right", cfg, (640, 480)))
    wr = WordRecognizer(bundle)                       # live app = the same inputs, then normalised
    assert np.allclose(wr.normalizer.apply(cat["x"]),
                       wr.features_from_frames([s - t[0] for s in t], pts, "right", (640, 480)), atol=1e-5)
    assert len(cat["x"]) == pytest.approx(60, abs=1)  # real speed: no speed-up applied


# ---------------------------------------------------- phrases (Phase 2B)
def test_phrases_and_a_very_slow_pace_can_be_recorded(recorder, tmp_path):
    assert recorder.target_word(" i love you ") == "I LOVE YOU"
    with pytest.raises(Exception, match="single spaces"):
        recorder.target_word("i  love you")
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("I LOVE YOU\nthank you\n")
    assert recorder.read_word_list(phrases) == ["I LOVE YOU", "THANK YOU"]
    assert recorder.parse_paces("veryslow,slow") == ["veryslow", "slow"]
    t, pts = moving_take(40)
    path = recorder.save_recording(tmp_path, t, pts, ["right"] * 40, "right", "", True, (640, 480), 1.0,
                                   "I LOVE YOU", "veryslow")
    assert path.name.startswith("I-LOVE-YOU_veryslow_")
    assert recorder.recorded_takes(tmp_path) == Counter({("I LOVE YOU", "veryslow"): 1})
    assert recorder.recording_problem(t, pts, "I LOVE YOU", 30.0) is None


def test_pack_reads_several_lists_and_keeps_the_space_in_phrases(recorder, packer, bundle, tmp_path):
    from asl.seq import vocab

    takes = tmp_path / "takes"
    make_takes(recorder, takes, phrases=True)
    out = tmp_path / "pack.npz"
    summary = packer.main([str(takes), *write_lists(tmp_path, phrases=True), "--bundle", str(bundle), "--out", str(out)])
    assert summary["groups"]["train"]["takes"] == 4 and summary["groups"]["train"]["phrase_takes"] == 1
    assert summary["groups"]["test"]["takes"] == 3 and summary["groups"]["test"]["phrase_takes"] == 1
    examples, _ = packer.load_pack(out)
    home = next(e for e in examples if e["text"] == "GO HOME")
    assert home["y"].tolist()[2] == vocab.SPACE_ID and home["split"] == "train"


# --------------------------------------------------------- the Colab cells
def fake_clip(rng, text, T=40):
    from asl.seq import vocab

    x = np.c_[rng.normal(size=(T, 67)), np.ones(T)].astype(np.float32)
    return {"x": x, "y": np.array(vocab.encode(text), np.int32), "text": text, "signer": "s", "sequence_id": text}


def colab_scope(drive, run_name, examples, shown):
    """What the notebook's Setup cell and earlier steps leave behind, for running a pasted cell."""
    from asl.seq import features, vocab
    from asl.seq import fsboard as fb
    from asl.seq import fsboard_video as fsv
    from asl.seq import model as smodel
    from asl.seq import train as strain

    return {"np": np, "json": json, "shutil": __import__("shutil"), "Path": Path, "HTML": lambda s: s,
            "display": shown.append, "vocab": vocab, "features": features, "smodel": smodel,
            "strain": strain, "fsv": fsv, "fb": fb, "RUN_NAME": run_name, "DRIVE": drive, "examples": examples,
            "need_data": lambda trained=False: None}      # step 1's loader: this session already has the data


def notebook_cell(marker):
    """The notebook's code cell that contains `marker`."""
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    found = [c for c in cells if marker in c]
    assert len(found) == 1, marker
    return found[0]


def run_cell(marker, scope, replace=()):
    source = notebook_cell(marker)
    for old, new in replace:
        assert old in source
        source = source.replace(old, new)
    exec(compile(source, marker, "exec"), scope)
    return scope


FINE_TUNE = [("FINE_TUNE_ON_MY_RECORDINGS = False", "FINE_TUNE_ON_MY_RECORDINGS = True"),
             ("TUNE_EPOCHS = 40", "TUNE_EPOCHS = 2")]


def test_colab_cell_fine_tunes_reports_and_leaves_the_base_model_alone(recorder, packer, bundle, tmp_path, capsys):
    drive = tmp_path / "drive"
    run_name = "train4_val2_test2_letters2-100_phrases40"
    base_dir = drive / "runs" / f"{run_name}__speed_long_memory"
    base_dir.parent.mkdir(parents=True)
    (tmp_path / "base").rename(base_dir)                 # a model WITH the space output (28)
    takes = tmp_path / "takes"
    make_takes(recorder, takes, phrases=True)
    packer.main([str(takes), *write_lists(tmp_path, phrases=True), "--bundle", str(base_dir / "best_model"),
                 "--out", str(drive / "my_recordings" / "my_recordings_pack.npz")])
    weights_before = hashlib.md5((base_dir / "best_model" / "model.weights.h5").read_bytes()).hexdigest()

    rng = np.random.default_rng(5)
    examples = {"train": [fake_clip(rng, w) for w in ("ONE", "TWO", "SIX", "TEN", "NEW YORK")],
                "validation": [fake_clip(rng, "RED")],
                "test": [fake_clip(rng, "BIG"), fake_clip(rng, "TOP"), fake_clip(rng, "HI THERE")]}
    shown = []
    run_cell("FINE_TUNE_ON_MY_RECORDINGS =", colab_scope(drive, run_name, examples, shown), FINE_TUNE)

    out = capsys.readouterr().out
    assert "Training on 5 FSboard clips + 4 of your takes x 1" in out
    personal = drive / "runs" / f"{run_name}__speed_long_memory__personal" / "best_model"
    report = json.loads((personal / "personal_report.json").read_text())
    assert {(r["model"], r["group"]) for r in report} == {
        (m, g) for m in ("before (base model)", "after (personal)")
        for g in ("your test words", "your test words, slow", "your test words, normal", "your test phrases",
                  "your validation takes", "FSboard one-word test clips", "FSboard phrase test clips")}
    phrase_row = next(r for r in report if r["group"] == "your test phrases")
    assert phrase_row["recordings"] == 1 and "spaces_found" in phrase_row and "word_error_rate" in phrase_row
    prep = json.loads((personal / "preprocessing.json").read_text())
    base_prep = json.loads((base_dir / "best_model" / "preprocessing.json").read_text())
    assert prep["normalizer"] == base_prep["normalizer"] and prep["base_model"] == base_dir.name
    assert (drive / "export" / f"{run_name}__speed_long_memory__personal.zip").exists()
    assert hashlib.md5((base_dir / "best_model" / "model.weights.h5").read_bytes()).hexdigest() == weights_before

    from asl.seq.inference import WordRecognizer
    WordRecognizer(personal)                          # the laptop can load the new model


def test_colab_cell_refuses_phrases_for_a_model_without_a_space_output(recorder, packer, bundle, tmp_path):
    from asl.seq.features import Normalizer, preprocessing_dict
    from asl.seq.fsboard_video import FEATURE_SETTINGS
    from asl.seq.model import DEFAULT_MODEL_CONFIG, build_tcn_ctc, save_inference_bundle

    drive = tmp_path / "drive"
    run_name = "train4_val2_test2_letters2-100"
    cfg = dict(DEFAULT_MODEL_CONFIG, channels=8, dilations=[1, 2], num_classes=27)
    prep = json.loads((bundle / "preprocessing.json").read_text())
    save_inference_bundle(drive / "runs" / f"{run_name}__speed_long_memory" / "best_model", build_tcn_ctc(cfg), cfg,
                          preprocessing_dict(Normalizer.from_dict(prep["normalizer"]), dict(FEATURE_SETTINGS)))
    takes = tmp_path / "takes"
    make_takes(recorder, takes, phrases=True)
    packer.main([str(takes), *write_lists(tmp_path, phrases=True), "--bundle", str(bundle),
                 "--out", str(drive / "my_recordings" / "my_recordings_pack.npz")])
    rng = np.random.default_rng(5)
    examples = {"train": [fake_clip(rng, "ONE")], "validation": [], "test": []}
    with pytest.raises(ValueError, match="no space output"):
        run_cell("FINE_TUNE_ON_MY_RECORDINGS =", colab_scope(drive, run_name, examples, []), FINE_TUNE)


def test_step2_cell_keeps_the_one_word_clips_and_adds_phrases_from_unfamiliar_signers(tmp_path):
    import types
    from asl.seq import fsboard as fb

    def meta(name, phrase, signer):
        return {"signingType": "fs", "clipFilename": name, "phrase": phrase, "signerId": signer,
                "clipFileMd5": "0" * 32, "clipFileSize": 10, "sensitiveContent": False,
                "clipStartTimeS": 1.0, "annotationStartTimeS": 1.5, "annotationEndTimeS": 3.0}

    records = {"train": [meta("t1.mp4", "hello", "a"), meta("t2.mp4", "world", "b"), meta("t3.mp4", "new york", "a"),
                         meta("t4.mp4", "i love you", "b"), meta("t5.mp4", "good day", "a"),
                         meta("t6.mp4", "call 911", "a")],
               "validation": [meta("v1.mp4", "cat", "v"), meta("v2.mp4", "big dog", "v"),
                              meta("v3.mp4", "red car", "a")],
               "test": [meta("s1.mp4", "sun", "s"), meta("s2.mp4", "hot tea", "s")]}
    fake_fb = types.SimpleNamespace(
        download=lambda name, dest, expected_size=None: dest, remote_size=lambda name: 0,
        load_metadata=lambda path: records[Path(path).stem.split("-")[-1]], FsboardError=fb.FsboardError)
    step2 = "INCLUDE_PHRASES = True"

    scope = colab_scope(tmp_path / "drive", None, None, [])
    scope.update(fb=fake_fb, WORK=tmp_path / "work")
    small = [("PHRASE_CLIPS_TRAIN = 8000", "PHRASE_CLIPS_TRAIN = 2")]
    run_cell(step2, scope, small + [("EXTRA_TRAIN_CLIPS = 3000", "EXTRA_TRAIN_CLIPS = 0")])
    clips = scope["clips"]
    assert [r["answer"] for r in clips["train"]][:2] == ["HELLO", "WORLD"]           # one-word clips: all
    assert len([r for r in clips["train"] if " " in r["answer"]]) == 2                 # phrases: up to the limit
    assert [r["answer"] for r in clips["validation"]] == ["CAT", "BIG DOG"]            # "red car": a training signer
    assert [r["answer"] for r in clips["test"]] == ["SUN", "HOT TEA"]
    assert scope["RUN_NAME"] == "train4_val2_test2_letters2-100_phrases40"
    assert (scope["RUN_DIR"] / "selection.json").exists()

    # Extra training clips by your own rule: here, only clips with a K (NEW YORK) score above 0.
    extra = colab_scope(tmp_path / "drive3", None, None, [])
    extra.update(fb=fake_fb, WORK=tmp_path / "work")
    rule = [('WEAK_LETTERS = "QXYUZP"', 'WEAK_LETTERS = "K"'),
            ("        score += 1\n    return score", "        score += 0\n    return score")]
    run_cell(step2, extra, [("PHRASE_CLIPS_TRAIN = 8000", "PHRASE_CLIPS_TRAIN = 1")] + rule)
    train = [r["answer"] for r in extra["clips"]["train"]]
    assert len(train) == len(set(train)) and train[:2] == ["HELLO", "WORLD"]
    assert "NEW YORK" in train and len(train) in (3, 4)          # 4 if the random phrase was another one
    assert [r["answer"] for r in extra["clips"]["validation"]] == ["CAT", "BIG DOG"]   # evaluation is untouched
    assert [r["answer"] for r in extra["clips"]["test"]] == ["SUN", "HOT TEA"]
    saved = json.loads((extra["RUN_DIR"] / "selection.json").read_text())["extra"]
    if len(train) == 4:
        assert extra["RUN_NAME"].startswith("train4_val2_test2_letters2-100_phrases40_extra1-")
        assert saved["chosen"] == ["t3.mp4"] and saved["weak_letters"] == "K"
    else:                                                        # NEW YORK was already the random phrase
        assert extra["RUN_NAME"] == "train3_val2_test2_letters2-100_phrases40" and saved["chosen"] == []

    # With every clip already chosen there is nothing to add, and the run name is the old one.
    full = colab_scope(tmp_path / "drive4", None, None, [])
    full.update(fb=fake_fb, WORK=tmp_path / "work")
    run_cell(step2, full, [])
    assert len(full["clips"]["train"]) == 5 and full["RUN_NAME"] == "train5_val2_test2_letters2-100_phrases40"

    words_only = colab_scope(tmp_path / "drive2", None, None, [])
    words_only.update(fb=fake_fb, WORK=tmp_path / "work")
    run_cell(step2, words_only, [("INCLUDE_PHRASES = True", "INCLUDE_PHRASES = False"),
                                 ("EXTRA_TRAIN_CLIPS = 3000", "EXTRA_TRAIN_CLIPS = 0")])
    assert {s: [r["answer"] for r in c] for s, c in words_only["clips"].items()} == {
        "train": ["HELLO", "WORLD"], "validation": ["CAT"], "test": ["SUN"]}
    assert words_only["RUN_NAME"] == "train2_val1_test1_letters2-100"                  # as before Phase 2B


def test_fine_tuning_is_off_by_default_so_run_all_skips_it(capsys):
    run_cell("FINE_TUNE_ON_MY_RECORDINGS =", {})
    assert "Skipped: tick FINE_TUNE_ON_MY_RECORDINGS" in capsys.readouterr().out
