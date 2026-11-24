"""Tests for the experimental sequence backend that need only numpy:
vocabulary, CTC decoding (repeats, blanks), features (missing landmarks,
velocity across gaps, causality), padding, resampling, split, augmentation."""
import numpy as np
import pytest

from asl.features import normalise_points
from asl.seq import ctc, vocab
from asl.seq import data as sdata
from asl.seq.features import (CausalResampler, Normalizer, SequenceFeaturizer, VALID, WRIST_VEL,
                              WRIST_POS, NUM_FEATURES, canonical_points, featurize_sequence)

A, L, B = vocab.CHAR_TO_ID["A"], vocab.CHAR_TO_ID["L"], vocab.BLANK


def onehot(ids, C=vocab.NUM_CLASSES):
    s = np.full((len(ids), C), -5.0, np.float32)
    s[np.arange(len(ids)), ids] = 5.0
    return s


def hand(x=0.5, y=0.5, size=0.1, seed=0):
    """A plausible (21, 3) hand: wrist at (x, y), fingers spread around it."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((21, 3), np.float32)
    pts[1:, :2] = rng.uniform(-0.6, 0.6, (20, 2)) * size   # all within "size"
    pts[1:, 2] = rng.uniform(-0.05, 0.05, 20)
    pts[17, :2] = [size, 0]                       # guarantee hand size = size
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


# ------------------------------------------------------------------ vocab
def test_vocabulary_blank_is_zero_letters_1_to_26_and_space_27():
    assert vocab.VOCAB[0] == "<blank>" and vocab.NUM_CLASSES == 28 and vocab.SPACE_ID == 27
    assert vocab.encode("hello ") == [8, 5, 12, 12, 15]
    assert vocab.decode_ids([8, 0, 5, 12, 12, 15]) == "HELLO"
    assert vocab.encode("i love you") == [9, 27, 12, 15, 22, 5, 27, 25, 15, 21]
    assert vocab.decode_ids([9, 27, 0, 27, 12]) == "I  L"      # decoding writes what the model says


def test_older_letters_only_vocabulary_is_still_understood():
    assert vocab.is_known_vocabulary(vocab.VOCAB[:27]) and vocab.is_known_vocabulary(vocab.VOCAB)
    assert not vocab.is_known_vocabulary(vocab.VOCAB[:26]) and not vocab.is_known_vocabulary(["x"] * 27)


@pytest.mark.parametrize("text,word,phrase", [("Hello", True, True), ("new york", False, True),
                                              (" new york ", False, True), ("new  york", False, False),
                                              ("3 creek", False, False), ("o'neil", False, False),
                                              ("", False, False), ("straße", False, False)])
def test_incompatible_transcripts_are_excluded_not_edited(text, word, phrase):
    assert vocab.is_compatible(text) is word                       # one word: the default
    assert vocab.is_compatible(text, allow_spaces=True) is phrase  # phrases only when asked for
    if not phrase:
        with pytest.raises(ValueError):
            vocab.encode(text)


def test_required_frames_counts_blank_between_repeats():
    assert vocab.required_frames(vocab.encode("HELLO")) == 6
    assert vocab.required_frames(vocab.encode("AAA")) == 5
    assert vocab.required_frames(vocab.encode("CAT")) == 3


# -------------------------------------------------------------------- CTC
def test_ctc_repeated_letter_needs_blank_between():
    assert ctc.greedy_decode_text(onehot([L, L, B, L])) == "LL"
    assert ctc.greedy_decode_text(onehot([L, L, L])) == "L"
    assert ctc.greedy_decode_text(onehot([B, A, A, B, B, L, B, L, L])) == "ALL"


def test_ctc_blank_only_and_length_limit():
    assert ctc.greedy_decode_text(onehot([B] * 10)) == ""
    # frames beyond the true length (padding) are ignored
    assert ctc.greedy_decode_text(onehot([A, B, L, L]), length=2) == "A"


@pytest.mark.parametrize("seed", range(5))
def test_streaming_decoder_equals_offline_for_any_chunking(seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice([B, B, B, A, L], size=80)
    scores = onehot(ids)
    dec = ctc.StreamingGreedyDecoder()
    i = 0
    while i < len(scores):
        n = int(rng.integers(1, 9))
        dec.push(scores[i:i + n])
        i += n
    dec.finish()
    assert dec.confirmed == ctc.greedy_decode(scores)


def test_streaming_decoder_repeat_across_chunk_boundary_and_provisional():
    dec = ctc.StreamingGreedyDecoder()
    dec.push(onehot([L, L]))
    assert dec.text() == ("", "L")              # run still going: provisional
    dec.push(onehot([L]))
    assert dec.text() == ("", "L")              # same run continues over the boundary: no new letter
    dec.push(onehot([B]))
    assert dec.text() == ("L", "")              # run ended: confirmed
    dec.push(onehot([L, L]))
    dec.finish()
    assert dec.text() == ("LL", "")


def test_edit_operations_and_cer():
    assert ctc.edit_operations("HELLO", "HELO") == (0, 0, 1)
    assert ctc.edit_operations("CAT", "CART") == (0, 1, 0)
    assert ctc.edit_operations("CAT", "CUT") == (1, 0, 0)
    assert ctc.character_error_rate(["HELLO", "CAT"], ["HELO", "CAT"]) == pytest.approx(1 / 8)
    assert ctc.collapse_doubles("BOOKKEEPER") == "BOKEPER"


# --------------------------------------------------------------- features
def test_feature_layout_and_reuse_of_baseline_finger_normalisation():
    f = featurize_sequence([hand(0.3, 0.6)])
    assert f.shape == (1, NUM_FEATURES) == (1, 68)
    np.testing.assert_allclose(f[0, :63], normalise_points(hand(0.3, 0.6)), atol=1e-6)
    np.testing.assert_allclose(f[0, WRIST_POS], [-0.2, 0.1], atol=1e-6)
    assert f[0, VALID] == 1 and np.all(f[0, WRIST_VEL] == 0)     # first frame: no velocity


@pytest.mark.parametrize("bad", [None, "nan", "inf", "tiny"])
def test_missing_or_invalid_landmarks_give_zero_features_and_flag(bad):
    p = {None: None, "nan": np.full((21, 3), np.nan), "inf": np.full((21, 3), np.inf),
         "tiny": np.zeros((21, 3))}[bad]
    if bad == "nan":
        p = hand()
        p[5, 1] = np.nan                          # even ONE missing coordinate -> invalid
    fz = SequenceFeaturizer()
    out = fz.step(p)
    assert out.shape == (68,) and np.all(out == 0) and np.all(np.isfinite(out))


def test_velocity_is_wrist_motion_in_hand_sizes_per_frame_across_short_gaps():
    fz = SequenceFeaturizer({"max_gap_frames": 3})
    fz.step(hand(0.50, 0.5, size=0.1))
    f1 = fz.step(hand(0.51, 0.5, size=0.1))
    np.testing.assert_allclose(f1[WRIST_VEL], [0.1, 0.0], atol=1e-5)    # 0.01 / 0.1
    fz.step(None)                                                         # 1 missing frame
    f3 = fz.step(hand(0.53, 0.5, size=0.1))
    np.testing.assert_allclose(f3[WRIST_VEL], [0.1, 0.0], atol=1e-5)    # 0.02 over 2 frames
    for _ in range(3):
        fz.step(None)                                                     # gap of 4 > 3
    f7 = fz.step(hand(0.60, 0.5, size=0.1))
    assert np.all(f7[WRIST_VEL] == 0)


def test_velocity_is_clipped_on_tracking_jumps():
    fz = SequenceFeaturizer({"vel_clip": 1.0})
    fz.step(hand(0.1, 0.5, size=0.05))
    f = fz.step(hand(0.9, 0.5, size=0.05))
    assert f[WRIST_VEL][0] == pytest.approx(1.0)


def test_featurizer_is_causal_future_frames_do_not_change_past():
    seq = [hand(0.5 + 0.01 * k, 0.5, seed=k) for k in range(10)]
    a = featurize_sequence(seq)
    seq2 = seq[:6] + [None, hand(0.9, 0.1, seed=99), None, None]
    b = featurize_sequence(seq2)
    np.testing.assert_array_equal(a[:6], b[:6])


def test_left_hand_is_mirrored_to_right_hand_geometry():
    p = hand(0.3, 0.5)
    m = canonical_points(p, "left")
    np.testing.assert_allclose(m[:, 0], 1 - p[:, 0])
    np.testing.assert_array_equal(canonical_points(p, "right"), p)
    seq = np.stack([p, p])
    assert canonical_points(seq, "Left").shape == (2, 21, 3)


def test_normaliser_uses_valid_frames_only_and_keeps_invalid_zero():
    f = featurize_sequence([hand(0.4, 0.5, seed=k) for k in range(20)] + [None] * 5)
    n = Normalizer().fit([f])
    g = n.apply(f)
    assert np.all(g[-5:] == 0)
    np.testing.assert_allclose(g[:20, 63:65].mean(0), 0, atol=1e-5)
    assert np.all(g[:20, VALID] == 1)
    n2 = Normalizer.from_dict(n.to_dict())
    np.testing.assert_allclose(n2.apply(f), g)


def test_resampler_makes_fixed_rate_and_interpolates():
    r = CausalResampler(30.0)
    out = []
    for k in range(16):                                  # 15 fps for ~1 s
        out += r.push(k / 15, hand(0.5 + 0.01 * k, 0.5))
    assert len(out) == 31                                # 0 .. 1 s at 30 fps
    np.testing.assert_allclose(out[1][0, 0], 0.505, atol=1e-5)   # halfway between frames
    r = CausalResampler(30.0)
    r.push(0.0, hand())
    mid = r.push(1 / 15, None)                           # interpolation impossible
    assert len(mid) == 2 and mid[-1] is None


# ---------------------------------------------------------------- batching
def _example(T, text, signer=1):
    x = featurize_sequence([hand(0.5, 0.5, seed=k) for k in range(T)])
    return {"x": x, "y": np.array(vocab.encode(text), np.int32), "text": text,
            "signer": signer, "sequence_id": T}


def test_padding_on_the_right_keeps_true_lengths():
    exs = [_example(5, "AB"), _example(9, "HELLO")]
    X, xl, Y, yl = sdata.pad_batch(exs)
    assert X.shape == (2, 9, 68) and list(xl) == [5, 9] and list(yl) == [2, 5]
    assert np.all(X[0, 5:] == 0) and np.all(Y[0, 2:] == 0)
    np.testing.assert_array_equal(X[0, :5], exs[0]["x"])


def test_batches_cover_every_example_once():
    exs = [_example(5 + k, "AB") for k in range(10)]
    n = Normalizer().fit([e["x"] for e in exs])
    seen = sum(len(b[1]) for b in sdata.iterate_batches(exs, 3, n, np.random.default_rng(0)))
    assert seen == 10


def test_augmentation_never_breaks_ctc_length_and_keeps_invalid_zero():
    rng = np.random.default_rng(0)
    x = featurize_sequence([hand(seed=k) if k % 4 else None for k in range(12)])
    for _ in range(50):
        y = sdata.augment(x, min_len=12, rng=rng, cfg=sdata.DEFAULT_AUGMENT)
        assert len(y) >= 12 and np.all(np.isfinite(y))
        assert np.all(y[y[:, VALID] < 0.5] == 0)


def test_shard_roundtrip_without_pickle(tmp_path):
    recs = [{"sequence_id": 7, "hand": "left", "points": np.random.rand(4, 21, 3).astype(np.float32),
             "frames": np.arange(4, dtype=np.int32), "valid_right": 0, "valid_left": 4,
             "inserted_frames": 1, "duplicate_frames": 0},
            {"sequence_id": 9, "hand": "right", "points": np.random.rand(2, 21, 3).astype(np.float32),
             "frames": np.arange(2, dtype=np.int32), "valid_right": 2, "valid_left": 0,
             "inserted_frames": 0, "duplicate_frames": 0}]
    sdata.save_shard(tmp_path / "s.npz", recs)
    back = sdata.load_shard(tmp_path / "s.npz")
    assert [r["sequence_id"] for r in back] == [7, 9] and back[0]["hand"] == "left"
    np.testing.assert_array_equal(back[1]["points"], recs[1]["points"])


class _Column:
    def __init__(self, values):
        self.values = np.array(values)

    def unique(self):
        return np.unique(self.values)


def test_signer_split_has_no_overlap():
    ids = list(range(100, 120)) * 3                     # several recordings per signer
    meta = {"participant_id": _Column(ids)}
    split = sdata.split_by_signer(meta, seed=3)
    assert set(split) == set(ids) and split == sdata.split_by_signer(meta, seed=3)
    groups = {s: {p for p, v in split.items() if v == s} for s in ("train", "val", "test")}
    assert all(groups.values())
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"])
