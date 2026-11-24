"""Tests for the FSboard preparation code: filtering, pairing, the coordinate
layout found in the real files, export and reload, and downloads against a local
fake server. Names and numbers below follow real FSboard records."""
import csv
import hashlib
import http.server
import threading
from collections import Counter

import numpy as np
import pytest

from asl.seq import fsboard as fb
from asl.seq import vocab


def meta_row(name="aaaa_1_20220101_000000_1000_2000.mp4", phrase="hello", signer="aaaa", **kw):
    return dict({"signingType": "fs", "clipFilename": name, "phrase": phrase, "signerId": signer,
                 "clipFileMd5": "0" * 32, "clipFileSize": 10, "sensitiveContent": False}, **kw)


# ------------------------------------------------------------------- filter
@pytest.mark.parametrize("phrase,reason", [
    ("celebstyle", ""), ("hello", ""), ("ab", ""), ("a" * 12, ""),
    ("a", "too few letters"), ("a" * 13, "too many letters"),
    ("2067 pso costa", "contains a digit"), ("new york", "contains a space"),
    ("o'neil", "contains a symbol"), ("straße", "contains a symbol"),
    ("ﬁne", "contains a symbol"), ("hello ", "contains a space"), ("", "empty phrase")])
def test_complete_phrase_must_be_2_to_12_letters_and_nothing_else(phrase, reason):
    assert fb.phrase_problem(phrase) == reason


@pytest.mark.parametrize("phrase,reason", [
    ("new york", ""), ("i love you", ""), ("hello", ""), ("a b", ""), ("a", "too few letters"),
    ("ab " + "c" * 11, "too many letters"),                      # letters are counted, not spaces
    ("new  york", "space at the start or end, or two spaces"), ("hello ", "space at the start or end, or two spaces"),
    ("2067 pso costa", "contains a digit"), ("o'neil street", "contains a symbol")])
def test_phrases_with_single_spaces_are_accepted_only_when_asked(phrase, reason):
    assert fb.phrase_problem(phrase, allow_spaces=True) == reason
    rows, _ = fb.eligible_rows([meta_row(phrase=phrase)], allow_spaces=True)
    assert [r["answer"] for r in rows] == ([phrase.upper()] if not reason else [])


def test_answer_is_uppercase_and_keeps_repeated_letters():
    assert fb.training_answer("letter") == "LETTER"
    assert fb.training_answer("zoo") == "ZOO" and len(vocab.encode("zoo")) == 3


def test_unsupported_recording_is_skipped_whole_not_edited():
    rows, skipped = fb.eligible_rows([meta_row(phrase="2067 pso costa"), meta_row(phrase="book")])
    assert [r["answer"] for r in rows] == ["BOOK"]         # never "PSOCOSTA"
    assert skipped == Counter({"contains a digit": 1})


def test_non_fingerspelling_and_sensitive_clips_are_left_out():
    rows, skipped = fb.eligible_rows([meta_row(signingType="other"), meta_row(sensitiveContent=True)])
    assert rows == [] and skipped == Counter({"not fingerspelling": 1, "tagged sensitiveContent": 1})
    rows, _ = fb.eligible_rows([meta_row(sensitiveContent=True)], skip_sensitive=False)
    assert len(rows) == 1


def test_preview_uses_different_signers_and_is_repeatable():
    recs = [meta_row(f"s{i % 7}_{i}.mp4", "hello", f"s{i % 7}") for i in range(40)]
    rows, _ = fb.eligible_rows(recs)
    chosen = fb.select_preview(rows, 5)
    assert len(chosen) == 5 and len({r["signer_id"] for r in chosen}) == 5
    assert chosen == fb.select_preview(list(reversed(rows)), 5)


def test_fewer_signers_than_five_gives_fewer_rows():
    rows, _ = fb.eligible_rows([meta_row("a_1.mp4", signer="a"), meta_row("a_2.mp4", signer="a")])
    assert len(fb.select_preview(rows)) == 1


def test_empty_filter_explains_what_to_check():
    with pytest.raises(fb.FsboardError, match="No recording passed the filter"):
        fb.select_preview([])


def test_row_number_outside_table_explains_itself():
    with pytest.raises(fb.FsboardError, match="rows 0 to 1"):
        fb.pick_row([{}, {}], 7)


def test_metadata_missing_a_field_names_it(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('[{"phrase": "x"}]')
    with pytest.raises(fb.FsboardError, match="clipFilename"):
        fb.load_metadata(path)
    path.write_text('{"not": "a list"}')
    with pytest.raises(fb.FsboardError, match="not a list"):
        fb.load_metadata(path)


def test_video_html_puts_written_answer_above_the_video(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"fake")
    page = fb.video_html(clip, "HELLO")
    assert page.index("HELLO") < page.index("<video") and "base64,ZmFrZQ==" in page


# ------------------------------------------------------------------- pairing
@pytest.mark.parametrize("video,clip", [
    ("51ee_379_20220813_203034_619.011_625.644.mp4", "51ee_379_20220813_203034_619011_625644.mp4"),
    ("9e67_859_20220826_101007_752.45_757.5.mp4", "9e67_859_20220826_101007_752450_757500.mp4"),
    ("e96d_420_20221008_202204_519.911_524.0.mp4", "e96d_420_20221008_202204_519911_524000.mp4"),
    ("x_1_20220101_000000_0.29_1.001.mp4", "x_1_20220101_000000_290_1001.mp4"),
    ("03ad_0_20220813_131811_103550_108200.mp4", "03ad_0_20220813_131811_103550_108200.mp4")])
def test_record_names_convert_to_metadata_names(video, clip):
    assert fb.clip_filename_from_video_name(video) == clip


def _ctx(video="a_1_20220101_000000_1.5_2.5.mp4", prompt="hello", signer="a"):
    return {"video_name": video, "prompt": prompt, "signer_id": signer}


def test_pairing_requires_name_phrase_and_signer_to_agree():
    index = fb.metadata_index([meta_row("a_1_20220101_000000_1500_2500.mp4", "hello", "a")])
    assert fb.match_metadata(_ctx(), index)[0] is not None
    assert fb.match_metadata(_ctx(prompt="world"), index) == (None, "phrase differs between record and metadata")
    assert fb.match_metadata(_ctx(signer="b"), index) == (None, "signer differs between record and metadata")
    assert fb.match_metadata(_ctx(video="z_1.mp4"), index) == (None, "no metadata row with this clip name")


# ---------------------------------------------------------- coordinate layout
FLAT = np.arange(1, 64, dtype=np.float32) / 100          # x0 y0 z0 x1 y1 z1 ...


def real_layout_lists(flat=FLAT):
    return flat[0:21], flat[1:22], flat[2:23]


def test_shifted_windows_decode_to_landmarks_0_to_7():
    layout, pts = fb.decode_hand_frame(*real_layout_lists())
    assert layout == fb.LAYOUT_PARTIAL and pts.shape == (8, 3)
    np.testing.assert_array_equal(pts[:7], FLAT[:21].reshape(7, 3))
    np.testing.assert_array_equal(pts[7, :2], FLAT[21:23])    # x and y of landmark 7
    assert np.isnan(pts[7, 2])                                 # its z is not stored


def test_values_from_a_real_frame_decode_as_expected():
    # Numbers printed from a real record (9e67_859_..., right hand, frame 27)
    x = [.372, .766, 0, .471, .737, -.045, .534, .672, -.068, .545, .614, -.093, .554, .567, -.115,
         .468, .597, -.05, .483, .52, -.099]
    y = x[1:] + [.484]
    z = y[1:] + [.469]
    layout, pts = fb.decode_hand_frame(x, y, z)
    assert layout == fb.LAYOUT_PARTIAL
    np.testing.assert_allclose(pts[0], [.372, .766, 0], atol=1e-6)      # wrist
    np.testing.assert_allclose(pts[4], [.554, .567, -.115], atol=1e-6)  # thumb tip
    np.testing.assert_allclose(pts[7, :2], [.484, .469], atol=1e-6)     # index DIP


def test_true_21_landmark_lists_would_be_read_as_complete():
    rng = np.random.default_rng(0)
    x, y, z = rng.random(21), rng.random(21), rng.random(21)
    layout, pts = fb.decode_hand_frame(x, y, z)
    assert layout == fb.LAYOUT_COMPLETE and pts.shape == (21, 3)
    np.testing.assert_allclose(pts[3], [x[3], y[3], z[3]])


def test_wrong_list_length_is_an_unknown_layout():
    assert fb.decode_hand_frame(np.zeros(25), np.zeros(25), np.zeros(25)) == (None, None)


# -------------------------------------------------------------------- records
tf = pytest.importorskip("tensorflow")


def make_record(video, prompt, signer, T=12, left=(), right=None, fps=30.0, layout="shifted", seed=0):
    """A tf.train.SequenceExample built like the real ones: empty features for
    frames where a hand was not detected."""
    rng = np.random.default_rng(seed)
    right = range(T) if right is None else right
    ex = tf.train.SequenceExample()
    c = ex.context.feature
    c["video_name"].bytes_list.value.append(video.encode())
    c["prompt"].bytes_list.value.append(prompt.encode())
    c["signer_id"].bytes_list.value.append(signer.encode())
    c["num_frames"].int64_list.value.append(T)
    c["image/width"].int64_list.value.append(1836)
    c["image/height"].int64_list.value.append(3264)
    c["image/frame_rate"].float_list.value.append(fps)
    fl = ex.feature_lists.feature_list
    for t in range(T):
        fl["timestamp"].feature.add().int64_list.value.append(33333 * t)
    for hand, frames in (("left_hand", set(left)), ("right_hand", set(right))):
        for t in range(T):
            flat = rng.random(63).astype(np.float32) * 0.8
            lists = real_layout_lists(flat) if layout == "shifted" else (flat[0::3], flat[1::3], flat[2::3])
            for a, vals in zip("xyz", lists):
                f = fl[f"{a}_{hand}"].feature.add()
                if t in frames:
                    f.float_list.value.extend(vals.tolist())
    return ex


def test_decode_hands_keeps_identity_order_timing_and_masks():
    ex = make_record("a_1.mp4", "hello", "a", T=12, left=[2, 3], right=[0, 1, 2, 9])
    h = fb.decode_hands(ex)
    assert h["layout"] == fb.LAYOUT_PARTIAL and h["left_hand"].shape == (12, 8, 3)
    assert np.flatnonzero(h["left_mask"]).tolist() == [2, 3]
    assert np.flatnonzero(h["right_mask"]).tolist() == [0, 1, 2, 9]
    assert np.isnan(h["left_hand"][0]).all() and np.isfinite(h["left_hand"][2, :7]).all()
    assert h["timestamp_us"].tolist() == [33333 * t for t in range(12)]
    # frame 9 of the right hand is exactly what was stored for frame 9
    stored = ex.feature_lists.feature_list["x_right_hand"].feature[9].float_list.value
    np.testing.assert_allclose(h["right_hand"][9, 0], stored[:3])


def test_decode_hands_reads_complete_layout_when_data_has_it():
    h = fb.decode_hands(make_record("a_1.mp4", "hello", "a", layout="complete"))
    assert h["layout"] == fb.LAYOUT_COMPLETE and h["right_hand"].shape == (12, 21, 3)
    assert h["landmark_ids"].tolist() == list(range(21))


def test_format_preview_shows_the_real_structure():
    text = fb.format_preview(make_record("a_1.mp4", "hello", "a"))
    assert "prompt" in text and "x list" in text and fb.LAYOUT_PARTIAL in text


CLIPS = [("b_2_20220101_000000_3.0_4.0.mp4", "b_2_20220101_000000_3000_4000.mp4", "letter", "b"),
         ("a_1_20220101_000000_1.5_2.5.mp4", "a_1_20220101_000000_1500_2500.mp4", "hello", "a")]


def _built(video, clip, phrase, signer, **kw):
    ex = make_record(video, phrase, signer, **kw)
    return fb.build_example(fb.record_context(ex), meta_row(clip, phrase, signer), fb.decode_hands(ex),
                            "daun_v3-train.arrow-00000-of-00100", 3)


def test_export_reload_round_trip_and_all_checks_pass(tmp_path):
    built = _built(*CLIPS[0], left=[1, 2])
    path = fb.save_example(built, tmp_path)
    back = fb.load_example(path)
    assert back["clip_id"] == CLIPS[0][1] and back["answer"] == "LETTER" and back["signer_id"] == "b"
    assert back["split"] == "train" and back["label_ids"].tolist() == vocab.encode("LETTER")
    np.testing.assert_array_equal(back["right_mask"], built["right_mask"])
    np.testing.assert_array_equal(back["timestamp_us"], built["timestamp_us"])
    results = fb.check_example(back)
    assert all(ok for _, ok, _ in results), results
    assert "L-E-T-T-E-R" in fb.model_sentence(back)


def test_checks_catch_broken_files(tmp_path):
    good = fb.load_example(fb.save_example(_built(*CLIPS[0]), tmp_path))

    def failed(**change):
        return {name for name, ok, _ in fb.check_example(dict(good, **change)) if not ok}

    assert "sequence and mask lengths agree" in failed(left_mask=good["left_mask"][:-1],
                                                       left_hand=good["left_hand"])
    assert "frames are in order" in failed(timestamp_us=good["timestamp_us"][::-1])
    no_hand = np.zeros_like(good["right_mask"])
    assert "usable hand coordinates exist" in failed(right_mask=no_hand,
                                                     right_hand=np.full_like(good["right_hand"], np.nan))
    assert "mask matches the coordinates" in failed(right_mask=no_hand)
    assert "answer matches the phrase" in failed(answer="LETER")       # collapsed double
    assert "official split is train" in failed(split="test")


def test_plot_of_exported_example_runs(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fig = fb.plot_example(fb.load_example(fb.save_example(_built(*CLIPS[0], left=[1, 2]), tmp_path)))
    assert len(fig.axes) == 7
    matplotlib.pyplot.close(fig)


def test_only_the_training_split_can_be_exported(tmp_path):
    assert fb.shard_split("daun_v3/landmarks/daun_v3-validation.arrow-00000-of-00020") == "validation"
    for name in ("daun_v3-validation.arrow-00000-of-00020", "daun_v3-test.arrow-00000-of-00020"):
        with pytest.raises(fb.FsboardError, match="training data only"):
            fb.export_from_shard(tmp_path / "missing", [], tmp_path, shard_name=name)


def test_shard_numbers_outside_the_range_explain_themselves():
    assert fb.train_shard_path(0) == "daun_v3/landmarks/daun_v3-train.arrow-00000-of-00100"
    with pytest.raises(fb.FsboardError, match="0 to 99"):
        fb.train_shard_path(100)


def write_shard(path, examples):
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([("serialized", pa.binary())])
    options = pa.ipc.IpcWriteOptions(compression="zstd")
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, schema, options=options) as w:
        for ex in examples:
            w.write_batch(pa.record_batch([pa.array([ex.SerializeToString()], pa.binary())], schema=schema))


def _shard_and_meta(tmp_path, seed=0):
    """Metadata is deliberately in a different order from the shard."""
    recs = [make_record(video, phrase, signer, seed=seed + i) for i, (video, _, phrase, signer) in enumerate(CLIPS)]
    recs.insert(1, make_record("c_3_20220101_000000_5.0_6.0.mp4", "2067 pso costa", "c"))
    recs.append(make_record("d_4_20220101_000000_7.0_8.0.mp4", "nohands", "d", right=[]))
    recs.append(make_record("e_5_20220101_000000_9.0_9.1.mp4", "manyletters", "e", right=[0, 1]))
    recs.append(make_record("f_6_20220101_000000_10.0_11.0.mp4", "stranger", "f"))        # not in metadata
    recs.append(make_record("g_7_20220101_000000_12.0_13.0.mp4", "prompt", "g"))          # phrase differs
    meta = [meta_row("g_7_20220101_000000_12000_13000.mp4", "another", "g"),
            meta_row("e_5_20220101_000000_9000_9100.mp4", "manyletters", "e"),
            meta_row("d_4_20220101_000000_7000_8000.mp4", "nohands", "d"),
            meta_row("c_3_20220101_000000_5000_6000.mp4", "2067 pso costa", "c"),
            meta_row(CLIPS[1][1], "hello", "a"), meta_row(CLIPS[0][1], "letter", "b")]
    path = tmp_path / "daun_v3-train.arrow-00000-of-00100"
    write_shard(path, recs)
    return path, recs, meta


def test_export_pairs_by_identity_and_reports_every_skip_reason(tmp_path):
    shard, recs, meta = _shard_and_meta(tmp_path)
    out = tmp_path / "out"
    report = fb.export_from_shard(shard, meta, out, target=20, log=lambda *_: None)
    assert report["records_read"] == report["records_in_shard"] == 7 and not report["stopped_early"]
    assert sorted(report["exported_now"]) == sorted(c[1] for c in CLIPS)
    assert report["skipped"] == {"contains a digit": 1, "no hand detected in any frame": 1,
                                 "too few hand frames for its letters": 1,
                                 "no metadata row with this clip name": 1,
                                 "phrase differs between record and metadata": 1}
    # Each file holds the coordinates of ITS OWN record, though the orders differ
    rows = {r["answer"]: r for r in fb.read_index(out)}
    assert set(rows) == {"LETTER", "HELLO"}
    for k, (video, clip, phrase, signer) in enumerate(CLIPS):
        loaded = fb.load_example(out / rows[phrase.upper()]["file"])
        assert loaded["clip_id"] == clip and loaded["signer_id"] == signer and loaded["phrase"] == phrase
        np.testing.assert_allclose(fb.decode_hands(recs[0 if k == 0 else 2])["right_hand"][5],
                                   loaded["right_hand"][5])
    assert (out / "README.txt").read_text().startswith("FSboard pilot sequences")


def test_index_csv_has_a_readable_row_per_sequence(tmp_path):
    shard, _, meta = _shard_and_meta(tmp_path)
    fb.export_from_shard(shard, meta, tmp_path / "out", log=lambda *_: None)
    with open(tmp_path / "out" / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and set(fb.INDEX_COLUMNS) == set(rows[0])
    row = [r for r in rows if r["answer"] == "LETTER"][0]
    assert row["signer_id"] == "b" and row["split"] == "train" and row["num_frames"] == "12"
    assert row["landmarks_per_hand"] == "8" and row["source_shard"] == "daun_v3-train.arrow-00000-of-00100"


def test_reading_stops_once_the_target_is_reached(tmp_path):
    shard, _, meta = _shard_and_meta(tmp_path)
    report = fb.export_from_shard(shard, meta, tmp_path / "out", target=1, log=lambda *_: None)
    assert report["exported_total"] == 1 and report["stopped_early"] and report["records_read"] == 1


def test_rerun_keeps_exports_without_duplicates(tmp_path):
    shard, _, meta = _shard_and_meta(tmp_path)
    fb.export_from_shard(shard, meta, tmp_path / "out", log=lambda *_: None)
    again = fb.export_from_shard(shard, meta, tmp_path / "out", log=lambda *_: None)
    assert again["exported_now"] == [] and again["exported_total"] == 2
    assert again["skipped"]["already exported earlier"] == 2
    assert len(fb.read_index(tmp_path / "out")) == 2


def test_shortfall_report_gives_count_and_reason_before_suggesting_a_download():
    report = {"shard": "s", "records_in_shard": 739, "records_read": 739, "stopped_early": False,
              "paired_with_metadata": 739, "exported_now": ["a", "b", "c"], "exported_total": 3,
              "target": 20, "skipped": {"contains a digit": 400}}
    text = fb.report_text(report, 439)
    assert text.index("3 found") < text.index("more shards") and "4.4 per shard" in text
    assert "Nothing further was downloaded" in text


def test_a_truncated_shard_gets_a_specific_message(tmp_path):
    pytest.importorskip("pyarrow")
    bad = tmp_path / "daun_v3-train.arrow-00000-of-00100"
    bad.write_bytes(b"not an arrow file")
    with pytest.raises(fb.FsboardError, match="download may be incomplete"):
        fb.open_shard(bad)


def test_missing_tensorflow_gets_a_specific_message(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", None)
    with pytest.raises(fb.FsboardError, match="pip install tensorflow"):
        fb.parse_example(b"")


def test_zip_contains_the_exports(tmp_path):
    import zipfile

    shard, _, meta = _shard_and_meta(tmp_path)
    out = tmp_path / "fsboard_pilot"
    fb.export_from_shard(shard, meta, out, log=lambda *_: None)
    made = fb.zip_outputs(out, tmp_path / "fsboard_pilot.zip")
    names = zipfile.ZipFile(made).namelist()
    assert "fsboard_pilot/index.csv" in names and "fsboard_pilot/README.txt" in names
    assert sum(n.endswith(".npz") for n in names) == 2


# ------------------------------------------------------------------ downloads
class FakeKaggle(http.server.BaseHTTPRequestHandler):
    """Behaves like the real service: Kaggle redirects to a storage host, and
    storage supports Range requests."""
    files = {}
    seen = []
    drop_first_at = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        cls = type(self)
        cls.seen.append((self.path, self.headers.get("Authorization"), self.headers.get("Range")))
        if self.path.startswith("/kaggle/"):
            name = self.path[len("/kaggle/"):]
            code = {"denied": 403, "missing": 404, "slow": 429}.get(name)
            if code:
                self.send_error(code)
                return
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/storage/{name}")
            self.end_headers()
            return
        data = cls.files[self.path[len("/storage/"):]]
        start = 0
        if self.headers.get("Range"):
            spec = self.headers["Range"].split("=")[1]
            start = int(spec.split("-")[0])
            end = int(spec.split("-")[1]) if spec.split("-")[1] else len(data) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            body = data[start:end + 1]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.drop_first_at is not None:
            cut, cls.drop_first_at = cls.drop_first_at, None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def kaggle(monkeypatch):
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    FakeKaggle.files, FakeKaggle.seen, FakeKaggle.drop_first_at = {}, [], None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeKaggle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(fb, "BASE_URL", f"http://127.0.0.1:{server.server_port}/kaggle/")
    yield FakeKaggle
    server.shutdown()
    server.server_close()


def test_download_verifies_md5_and_size(tmp_path, kaggle):
    data = b"video-bytes" * 1000
    kaggle.files["clip.mp4"] = data
    got = fb.download("clip.mp4", tmp_path / "clip.mp4", len(data), hashlib.md5(data).hexdigest(),
                      log=lambda *_: None)
    assert got.read_bytes() == data


def test_second_download_reuses_the_verified_file(tmp_path, kaggle):
    data = b"abc" * 100
    kaggle.files["clip.mp4"] = data
    fb.download("clip.mp4", tmp_path / "clip.mp4", len(data), hashlib.md5(data).hexdigest(), log=lambda *_: None)
    calls = len(kaggle.seen)
    fb.download("clip.mp4", tmp_path / "clip.mp4", len(data), hashlib.md5(data).hexdigest(), log=lambda *_: None)
    assert len(kaggle.seen) == calls


def test_wrong_md5_is_reported_and_the_bad_file_removed(tmp_path, kaggle):
    kaggle.files["clip.mp4"] = b"tampered"
    with pytest.raises(fb.FsboardError, match="clipFileMd5"):
        fb.download("clip.mp4", tmp_path / "clip.mp4", expected_md5="0" * 32, log=lambda *_: None)
    assert not (tmp_path / "clip.mp4").exists()


def test_wrong_size_is_reported(tmp_path, kaggle):
    kaggle.files["clip.mp4"] = b"short"
    with pytest.raises(fb.FsboardError, match="should have 999"):
        fb.download("clip.mp4", tmp_path / "clip.mp4", expected_size=999, log=lambda *_: None)


def test_interrupted_download_continues_where_it_stopped(tmp_path, kaggle):
    data = bytes(range(256)) * 4000
    kaggle.files["shard"] = data
    kaggle.drop_first_at = 300_000
    got = fb.download("shard", tmp_path / "shard", len(data), hashlib.md5(data).hexdigest(), log=lambda *_: None)
    assert got.read_bytes() == data
    assert any(r and r.startswith("bytes=300000-") for _, _, r in kaggle.seen)


@pytest.mark.parametrize("name,code,text", [("denied", 403, "KAGGLE_API_TOKEN"),
                                            ("missing", 404, "does not exist"), ("slow", 429, "wait")])
def test_http_failures_explain_what_to_check(tmp_path, kaggle, name, code, text):
    with pytest.raises(fb.FsboardError, match=text) as e:
        fb.download(name, tmp_path / "x", log=lambda *_: None)
    assert str(code) in str(e.value)


def test_nested_paths_are_requested_with_encoded_slashes(tmp_path, kaggle):
    kaggle.files["daun_v3%2Fmetadata%2Fx.json"] = b"[]"
    fb.download("daun_v3/metadata/x.json", tmp_path / "x.json", log=lambda *_: None)
    assert kaggle.seen[0][0] == "/kaggle/daun_v3%2Fmetadata%2Fx.json"


def test_credentials_go_to_kaggle_only_never_to_storage(tmp_path, kaggle, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret-token")
    kaggle.files["clip.mp4"] = b"abc"
    fb.download("clip.mp4", tmp_path / "clip.mp4", log=lambda *_: None)
    by_host = {path.split("/")[1]: auth for path, auth, _ in kaggle.seen}
    assert by_host == {"kaggle": "Bearer secret-token", "storage": None}


def test_username_and_key_are_sent_as_basic_auth(monkeypatch):
    for name in ("KAGGLE_API_TOKEN",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert fb._auth_headers() == {"Authorization": "Basic bWU6aw=="}


def test_remote_size_reads_without_downloading(kaggle):
    kaggle.files["shard"] = b"x" * 5000
    assert fb.remote_size("shard") == 5000
    assert ("/storage/shard", None, "bytes=0-0") in kaggle.seen
