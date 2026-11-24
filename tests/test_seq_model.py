"""Tests for the TCN-CTC model that need TensorFlow: causality, padding,
CTC loss with true lengths, save/reload equivalence (inference bundle and
resumable checkpoint), offline vs streaming inference, tiny overfit."""
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from asl.seq import vocab                                           # noqa: E402
from asl.seq.features import Normalizer, NUM_FEATURES, VALID, preprocessing_dict  # noqa: E402
from asl.seq.inference import StreamingRecognizer, WordRecognizer   # noqa: E402
from asl.seq.model import (build_tcn_ctc, check_reference_io, load_inference_bundle,  # noqa: E402
                           receptive_field, save_inference_bundle)
from asl.seq.train import CTCTrainer, fit                           # noqa: E402

CFG = {"dropout": 0.1}


@pytest.fixture(scope="module")
def model():
    tf.keras.utils.set_random_seed(0)
    return build_tcn_ctc(CFG)


def rand_x(T, seed=0):
    x = np.random.default_rng(seed).normal(size=(T, NUM_FEATURES)).astype(np.float32)
    x[:, VALID] = 1
    return x


def test_output_shape_one_score_per_frame_plus_blank(model):
    y = model(rand_x(50)[None]).numpy()
    assert y.shape == (1, 50, 28) and np.all(np.isfinite(y))       # blank, A-Z, space
    assert receptive_field() == 61


def test_model_is_causal_future_inputs_do_not_change_past_outputs(model):
    a = rand_x(100, 1)
    b = a.copy()
    b[60:] = rand_x(40, 2)
    ya = model(a[None], training=False).numpy()[0]
    yb = model(b[None], training=False).numpy()[0]
    np.testing.assert_allclose(ya[:60], yb[:60], atol=1e-5)
    assert not np.allclose(ya[60:], yb[60:])


def test_right_padding_does_not_change_real_frames(model):
    x = rand_x(30, 3)
    padded = np.zeros((1, 80, NUM_FEATURES), np.float32)
    padded[0, :30] = x
    np.testing.assert_allclose(model(x[None]).numpy()[0], model(padded).numpy()[0, :30], atol=1e-5)


def test_ctc_loss_uses_true_lengths_padding_is_ignored(model):
    tr = CTCTrainer(model)
    x = rand_x(40, 4)
    y = np.array(vocab.encode("HELLO"), np.int32)
    single = tr.ctc_loss(model(x[None]), np.array([40], np.int32), y[None], np.array([5], np.int32))
    Xp = np.zeros((1, 90, NUM_FEATURES), np.float32)
    Xp[0, :40] = x
    Yp = np.zeros((1, 12), np.int32)
    Yp[0, :5] = y
    padded = tr.ctc_loss(model(Xp), np.array([40], np.int32), Yp, np.array([5], np.int32))
    assert float(single) == pytest.approx(float(padded), rel=1e-4)
    assert np.isfinite(float(single))


def _bundle(tmp_path, model):
    n = Normalizer().fit([rand_x(20, 9)])
    prep = preprocessing_dict(n)
    return save_inference_bundle(tmp_path / "bundle", model, CFG, prep,
                                 reference_inputs=[rand_x(25, 5), rand_x(70, 6)])


def test_bundle_save_reload_gives_identical_predictions(tmp_path, model):
    folder = _bundle(tmp_path, model)
    m2, cfg, prep = load_inference_bundle(folder)
    x = rand_x(64, 7)[None]
    np.testing.assert_allclose(model(x).numpy(), m2(x).numpy(), atol=1e-6)
    assert check_reference_io(folder, m2)["ok"]


def _letters_only_bundle(tmp_path, seed=1):
    """A bundle like the ones made before Phase 2B: 27 outputs, no space."""
    tf.keras.utils.set_random_seed(seed)
    cfg = dict(CFG, num_classes=27)
    model = build_tcn_ctc(cfg)
    return model, save_inference_bundle(tmp_path / "old", model, cfg,
                                        preprocessing_dict(Normalizer().fit([rand_x(20, 9)])))


def test_letters_only_bundles_still_load_and_read(tmp_path):
    import json
    from asl.seq.model import has_space_output

    model, folder = _letters_only_bundle(tmp_path)
    saved = json.loads((folder / "vocab.json").read_text())
    assert saved["symbols"] == vocab.VOCAB[:27] and saved["version"] == "blank0-AZ-v1"
    m2, cfg, _ = load_inference_bundle(folder)
    assert cfg["num_classes"] == 27 and not has_space_output(cfg) and has_space_output(dict(cfg, num_classes=28))
    x = rand_x(40, 3)[None]
    np.testing.assert_allclose(model(x).numpy(), m2(x).numpy(), atol=1e-6)
    text, _ = WordRecognizer(folder).recognize(np.arange(30) / 30, [None] * 30)
    assert " " not in text


def test_warm_start_copies_every_layer_and_adds_the_space_output(tmp_path):
    from asl.seq.model import warm_start

    old, folder = _letters_only_bundle(tmp_path)
    new = build_tcn_ctc(CFG)                                   # 28 outputs: A-Z + space
    info = warm_start(new, folder)
    assert info["extended"] == ["logits"] and "proj" in info["copied"] and "b3_conv1" in info["copied"]
    x = rand_x(50, 4)[None]
    a, b = old(x, training=False).numpy(), new(x, training=False).numpy()
    assert b.shape[-1] == 28 and np.allclose(a, b[..., :27], atol=1e-5)   # reads letters exactly as before
    with pytest.raises(ValueError, match="built differently"):
        warm_start(build_tcn_ctc(dict(CFG, dilations=[1, 2, 4, 8, 16])), folder)


def test_streaming_equals_offline_on_same_recording(tmp_path, model):
    wr = WordRecognizer(_bundle(tmp_path, model))
    x = rand_x(200, 8)
    x[50:70] = 0                                   # a stretch with no hand
    offline = wr.logits(x)
    for chunk, arrive in ((1, 1), (4, 3), (7, 13), (64, 5)):
        s = StreamingRecognizer(wr, chunk_frames=chunk)
        got = []
        run = s._run
        s._run = lambda: got.append(run()) or got[-1]      # record each chunk's new outputs
        for i in range(0, len(x), arrive):                  # frames arrive in groups
            s.push_features(x[i:i + arrive])
        s.finish()
        np.testing.assert_allclose(np.concatenate(got), offline, atol=1e-4)


def test_warm_up_compiles_once_for_every_length(tmp_path, model):
    wr = WordRecognizer(_bundle(tmp_path, model))
    assert wr.warm_up() > 0
    traces = wr._predict.experimental_get_tracing_count()
    for n in (1, 4, 37, 300):                      # live chunks and whole takes of any length
        assert wr.logits(rand_x(n, 3)).shape[0] == n
    assert wr._predict.experimental_get_tracing_count() == traces   # no new compile during a take


def test_streaming_text_equals_offline_text(tmp_path, model):
    wr = WordRecognizer(_bundle(tmp_path, model))
    x = rand_x(150, 11)
    from asl.seq.ctc import greedy_decode_text
    s = StreamingRecognizer(wr, chunk_frames=5)
    for i in range(0, len(x), 3):
        s.push_features(x[i:i + 3])
    assert s.finish()["confirmed"] == greedy_decode_text(wr.logits(x))


def test_checkpoint_restores_weights_optimizer_and_progress(tmp_path):
    tf.keras.utils.set_random_seed(1)
    m = build_tcn_ctc(CFG)
    tr = CTCTrainer(m)
    X = rand_x(30, 1)[None]
    args = (X, np.array([30], np.int32), np.array([vocab.encode("AB")], np.int32),
            np.array([2], np.int32))
    for _ in range(3):
        tr.train_step(*args)
    tr.save(tmp_path / "last", {"epoch": 3, "step": 3})
    tf.keras.utils.set_random_seed(2)
    m2 = build_tcn_ctc(CFG)
    tr2 = CTCTrainer(m2)
    assert tr2.restore(tmp_path / "last")["epoch"] == 3
    for a, b in zip(tr.optimizer.variables, tr2.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_allclose(m(X, training=False).numpy(), m2(X, training=False).numpy(), atol=1e-6)


def _synthetic_examples(words, seed=0):
    """Each letter = its own fixed random pattern held ~6 frames, with
    blank-ish transition frames between letters. Easy on purpose."""
    rng = np.random.default_rng(seed)
    patterns = rng.normal(size=(27, NUM_FEATURES)).astype(np.float32)
    exs = []
    for k, w in enumerate(words):
        rows = []
        for c in w:
            rows += [patterns[vocab.CHAR_TO_ID[c]]] * 6 + [np.zeros(NUM_FEATURES, np.float32)] * 2
        x = np.stack(rows) + rng.normal(0, 0.05, (len(rows), NUM_FEATURES)).astype(np.float32)
        x[:, VALID] = 1
        exs.append({"x": x, "y": np.array(vocab.encode(w), np.int32), "text": w,
                    "signer": k % 3, "sequence_id": k})
    return exs


def test_tiny_overfit_learns_words_including_double_letters(tmp_path):
    words = ["HELLO", "BOOK", "CAT", "ALL", "ZOO", "JAZZ", "ASL", "DOG"]
    exs = _synthetic_examples(words)
    n = Normalizer().fit([e["x"] for e in exs])
    tf.keras.utils.set_random_seed(0)
    m = build_tcn_ctc(CFG)
    _, state = fit(m, exs, exs, n, tmp_path / "run", preprocessing_dict(n), CFG,
                   cfg={"epochs": 120, "batch_size": 8, "augment": False, "patience": 200,
                        "learning_rate": 3e-3, "show_examples": 0}, log=lambda *a: None)
    assert state["best_cer"] < 0.05, state["history"][-1]
    # resuming a finished run does not train again and keeps progress
    _, state2 = fit(build_tcn_ctc(CFG), exs, exs, n, tmp_path / "run", preprocessing_dict(n), CFG,
                    cfg={"epochs": 120, "batch_size": 8, "augment": False, "patience": 200,
                         "learning_rate": 3e-3, "show_examples": 0}, log=lambda *a: None)
    assert state2["epoch"] == state["epoch"]


def test_camera_frames_are_resampled_and_recognised(tmp_path, model):
    """15 fps camera with some missing hands -> about twice as many 30 fps frames."""
    wr = WordRecognizer(_bundle(tmp_path, model))
    rng = np.random.default_rng(0)
    ts = [k / 15 for k in range(30)]
    pts = [None if k % 7 == 0 else rng.uniform(0.4, 0.6, (21, 3)).astype(np.float32) for k in range(30)]
    f = wr.features_from_frames(ts, pts, hand="left")
    assert len(f) == 59 and np.all(np.isfinite(f))
    text, logits = wr.recognize(ts, pts, hand="left")
    assert isinstance(text, str) and logits.shape == (59, vocab.NUM_CLASSES)


# ------------------------------------------- laptop input must match training
def _aspect_bundle(tmp_path, model):
    from asl.seq.fsboard_video import FEATURE_SETTINGS

    n = Normalizer().fit([rand_x(20, 9)])
    return save_inference_bundle(tmp_path / "aspect", model, CFG, preprocessing_dict(n, dict(FEATURE_SETTINGS)))


def _camera_recording(hand, width, height, T=45, fps=20.0):
    """Moving hand with a gap, in the form both the laptop and training code receive."""
    rng = np.random.default_rng(3)
    arr = np.full((T, 21, 3), np.nan, np.float32)
    mask = np.ones(T, bool)
    mask[18:22] = False
    for t in np.flatnonzero(mask):
        arr[t] = rng.uniform(0.3, 0.7, (21, 3))
    other = np.full((T, 21, 3), np.nan, np.float32)
    rec = {"timestamp_us": (np.arange(T) * 1e6 / fps).astype(np.int64), "image_width": width, "image_height": height,
           f"{hand}_hand": arr, f"{hand}_mask": mask,
           f"{'left' if hand == 'right' else 'right'}_hand": other,
           f"{'left' if hand == 'right' else 'right'}_mask": np.zeros(T, bool)}
    seconds = (rec["timestamp_us"] - rec["timestamp_us"][0]) / 1e6
    return rec, seconds.tolist(), [arr[t] if mask[t] else None for t in range(T)]


@pytest.mark.parametrize("hand,width,height", [("right", 640, 480), ("left", 640, 480),
                                               ("right", 1836, 3264), ("left", 1280, 720)])
def test_laptop_features_are_identical_to_training_features(tmp_path, model, hand, width, height):
    from asl.seq.fsboard_video import recording_features

    wr = WordRecognizer(_aspect_bundle(tmp_path, model))
    rec, seconds, points = _camera_recording(hand, width, height)
    training = wr.normalizer.apply(recording_features(rec, wr.feature_cfg))
    laptop = wr.features_from_frames(seconds, points, hand, image_size=(width, height))
    np.testing.assert_array_equal(laptop, training)


def test_the_image_shape_really_changes_the_input(tmp_path, model):
    wr = WordRecognizer(_aspect_bundle(tmp_path, model))
    _, seconds, points = _camera_recording("right", 640, 480)
    wide = wr.features_from_frames(seconds, points, "right", image_size=(1280, 720))
    tall = wr.features_from_frames(seconds, points, "right", image_size=(720, 1280))
    assert not np.allclose(wide, tall)


def test_a_model_trained_with_correction_refuses_input_without_image_size(tmp_path, model):
    wr = WordRecognizer(_aspect_bundle(tmp_path, model))
    _, seconds, points = _camera_recording("right", 640, 480)
    with pytest.raises(ValueError, match="image_size"):
        wr.recognize(seconds, points, "right")
    with pytest.raises(ValueError, match="image_size"):
        StreamingRecognizer(wr)


def test_streaming_camera_frames_give_the_same_text_as_a_whole_recording(tmp_path, model):
    wr = WordRecognizer(_aspect_bundle(tmp_path, model))
    _, seconds, points = _camera_recording("left", 640, 480)
    whole, _ = wr.recognize(seconds, points, "left", image_size=(640, 480))
    s = StreamingRecognizer(wr, hand="left", chunk_frames=4, image_size=(640, 480))
    for t, p in zip(seconds, points):
        s.push_frame(t, p)
    assert s.finish()["confirmed"] == whole


# ------------------------------------------------------- the webcam recorder
@pytest.fixture
def recorder(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "scripts" / "sequence_model"))
    import seq_record_word
    return seq_record_word


def test_hand_is_the_label_seen_most_and_other_frames_become_missing(recorder):
    assert recorder.choose_hand(["left", "left", "right", None]) == "left"
    assert recorder.choose_hand(["left", "right"]) == "right" and recorder.choose_hand([None]) == "right"
    assert recorder.keep_signing_hand(["a", "b", None], ["right", "left", None], "right") == ["a", None, None]


def test_speed_up_compresses_time_from_zero(recorder):
    assert recorder.speed_times([10.0, 11.0, 13.0], 2.0) == [0.0, 0.5, 1.5]
    assert recorder.speed_times([], 2.0) == []


def test_speed_up_gives_fewer_model_frames_at_the_same_rate(recorder, tmp_path, model):
    wr = WordRecognizer(_aspect_bundle(tmp_path, model))
    _, seconds, points = _camera_recording("right", 640, 480, T=60, fps=30.0)
    normal = wr.features_from_frames(recorder.speed_times(seconds, 1.0), points, "right", image_size=(640, 480))
    faster = wr.features_from_frames(recorder.speed_times(seconds, 3.0), points, "right", image_size=(640, 480))
    assert len(normal) == pytest.approx(60, abs=1) and len(faster) == pytest.approx(20, abs=1)


@pytest.mark.parametrize("yours,training,warns", [(1.0, 0.054, True), (0.0, 0.054, False), (0.57, 0.054, False),
                                                  (1.0, None, False), (float("nan"), 0.054, False)])
def test_mirror_warning_only_for_a_clearly_flipped_hand(recorder, yours, training, warns):
    assert bool(recorder.mirror_warning(yours, training)) is warns


def test_target_word_must_be_letters(recorder):
    assert recorder.target_word(" hello ") == "HELLO"
    with pytest.raises(Exception, match="A-Z"):
        recorder.target_word("2 cats")
    with pytest.raises(Exception, match="more than 0"):
        recorder.positive_number("0")


def test_saved_recording_keeps_word_image_size_and_settings(recorder, tmp_path):
    pts = [np.zeros((21, 3), np.float32), None]
    path = recorder.save_recording(tmp_path, [1.0, 1.1], pts, ["right", None], "right", "CAT", True, (640, 480), 2.5, "CAT")
    assert path.name.startswith("CAT_")
    z = np.load(path, allow_pickle=False)
    assert str(z["word"]) == "CAT" and int(z["image_width"]) == 640 and int(z["image_height"]) == 480
    assert float(z["speed_up"]) == 2.5 and z["hand_labels"].tolist() == ["right", ""]
    assert np.isnan(z["points"][1]).all() and z["t"].tolist() == [1.0, 1.1]
    plain = recorder.save_recording(tmp_path / "x", [1.0], pts[:1], ["left"], "left", "", False, (640, 480), 1.0)
    assert plain.name.startswith("rec_")


def _fake_camera(monkeypatch, recorder, args, keys, labels):
    """Runs the recorder's real loop with a fake camera, tracker and keyboard."""
    import time as _time
    from asl.tracker import HandObservation

    frame = np.zeros((480, 640, 3), np.uint8)
    points = np.random.default_rng(0).uniform(0.3, 0.7, (21, 3)).astype(np.float32)
    keys, labels = iter(keys), iter(labels)
    clock = {"t": 100.0}

    class Camera:
        def __init__(self, index): pass
        def isOpened(self): return True
        def read(self): return True, frame
        def release(self): pass

    class Tracker:
        def __init__(self, video_mode): pass
        def process_bgr(self, image):
            assert image.shape == frame.shape
            label = next(labels, "Right")
            return None if label is None else HandObservation(points, label, 0.9)
        def close(self): pass

    def monotonic():
        clock["t"] += 1 / 30
        return clock["t"]

    monkeypatch.setattr(recorder.cv2, "VideoCapture", Camera)
    monkeypatch.setattr(recorder, "HandTracker", Tracker)
    monkeypatch.setattr(recorder.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(recorder.cv2, "waitKey", lambda _: ord(next(keys, "q")))
    monkeypatch.setattr(recorder.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(_time, "monotonic", monotonic)
    monkeypatch.setattr(sys, "argv", ["seq_record_word.py", *args])
    recorder.main()


def test_recorder_loop_detects_the_hand_and_saves_a_labelled_recording(recorder, monkeypatch, tmp_path, capsys):
    keys = "." + "s" + "." * 30 + "f" + "q"
    labels = ["Left"] * 25 + ["Right"] * 2 + ["Left"] * 10
    _fake_camera(monkeypatch, recorder, ["--untrained", "--mirror-input", "--speed-up", "2", "--word", "cat",
                                         "--save", str(tmp_path)], keys, labels)
    out = capsys.readouterr().out
    assert "target CAT:" in out and "using left hand, speed-up 2" in out
    files = list(tmp_path.glob("CAT_*.npz"))
    assert len(files) == 1
    z = np.load(files[0], allow_pickle=False)
    assert str(z["hand"]) == "left" and str(z["word"]) == "CAT" and float(z["speed_up"]) == 2
    assert int(z["image_width"]) == 640 and "left" in z["hand_labels"].tolist()
    assert len(z["t"]) == 31


def test_recorder_loop_with_live_letters_and_manual_hand(recorder, monkeypatch, capsys):
    keys = "." + "s" + "." * 20 + "f" + "q"
    _fake_camera(monkeypatch, recorder, ["--untrained", "--mirror-input", "--stream", "--hand", "left"], keys, ["Right"] * 40)
    out = capsys.readouterr().out
    assert "streaming letters:" in out and "using left hand" in out
