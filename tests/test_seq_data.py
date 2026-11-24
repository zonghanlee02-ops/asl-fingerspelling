"""Dataset-side tests (pandas / pyarrow). They are skipped on the laptop if
those packages are not installed, and run in the Colab notebook (cell 6)."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from asl.seq import data as sdata   # noqa: E402


def _frame_rows(seq_id, frames, right_valid, left_valid):
    rows = []
    for f, rv, lv in zip(frames, right_valid, left_valid):
        r = {"sequence_id": seq_id, "frame": f}
        for h, ok in (("right", rv), ("left", lv)):
            for c in sdata.hand_columns(h):
                r[c] = np.random.rand() if ok else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def test_compatibility_annotation_keeps_reasons():
    meta = pd.DataFrame({"phrase": ["hello", "new york", "3 creek", "o'neil", "ZOO "],
                         "participant_id": 1, "sequence_id": range(5), "path": "a"})
    m = sdata.annotate_compatibility(meta)
    assert m["compatible"].tolist() == [True, False, False, False, True]
    assert m["exclude_reason"].tolist() == ["", "space", "space", "other_symbol", ""]
    assert m.loc[4, "transcript"] == "ZOO"


def test_sequence_record_orders_frames_fills_gaps_and_picks_signing_hand():
    df = _frame_rows(5, [12, 10, 11, 14], [0, 0, 1, 0], [1, 1, 1, 1])   # frame 13 missing
    r = sdata.sequence_record(5, df)
    assert r["hand"] == "left" and r["valid_left"] == 4 and r["valid_right"] == 1
    assert r["frames"].tolist() == [10, 11, 12, 13, 14] and r["inserted_frames"] == 1
    assert np.all(np.isnan(r["points"][3]))                  # inserted frame is invalid
    assert r["points"].shape == (5, 21, 3)


def test_pilot_selection_respects_budget_and_only_compatible():
    meta = sdata.annotate_compatibility(pd.DataFrame({
        "path": ["f1"] * 4 + ["f2"] * 4 + ["f3"] * 2,
        "sequence_id": range(10), "participant_id": [1, 2, 3, 4] * 2 + [5, 6],
        "phrase": ["cat", "dog", "1 a", "zoo", "ab", "cd", "ef", "g h", "hi", "yo"]}))
    sizes = pd.DataFrame({"name": ["f1", "f2", "f3"], "bytes": [2e9, 1e9, 5e9]})
    sel, summary = sdata.select_pilot_subset(meta, sizes, budget_gb=3.5, max_sequences=100)
    assert set(sel["path"]) == {"f1", "f2"} and summary["download_gb"] == 3.0
    assert sel["compatible"].all() and len(sel) == 6


def test_read_hands_from_parquet_reads_only_selected_sequences(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.concat([_frame_rows(1, [0, 1], [1, 1], [0, 0]), _frame_rows(2, [0, 1, 2], [1, 0, 1], [0, 0, 0])])
    df["x_face_0"] = 0.1                                     # extra columns are not read
    df.set_index("sequence_id").to_parquet(tmp_path / "f.parquet")
    got = sdata.read_hands(tmp_path / "f.parquet", [2])
    assert set(got["sequence_id"]) == {2} and "x_face_0" not in got.columns
    r = sdata.sequence_record(2, got[got["sequence_id"] == 2])
    assert r["hand"] == "right" and r["valid_right"] == 2
