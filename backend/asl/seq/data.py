"""
data.py - Google "ASL Fingerspelling" (Kaggle competition `asl-fingerspelling`)
metadata, pilot-subset selection, remote extraction, signer split and batching.

Not to be confused with Google's ISOLATED SIGNS dataset (Kaggle
`asl-signs`, 250 single word signs). That one has no fingerspelled text.

What the dataset gives us (verified in the notebook, not assumed):
    train.csv / supplemental_metadata.csv
        path, file_id, sequence_id, participant_id, phrase
    <path>.parquet   one row per (sequence_id, frame), MediaPipe Holistic
        landmarks as columns such as x_right_hand_0 ... z_left_hand_20
        (NaN when that hand was not detected). One parquet file holds MANY
        sequences, so we download one file, keep only the selected
        sequences' hand columns, save a compact shard and delete the file.
What it does NOT give: timestamps / frame rate, session or recording-device
IDs, video. These gaps are written into the manifest.

Heavy imports (pandas, pyarrow, kaggle) are done inside functions so the
rest of asl.seq works on the laptop without them.
"""
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import vocab
from .fsboard import kaggle_auth_from_colab_secrets  # noqa: F401  (used by the notebook)
from .features import (SequenceFeaturizer, canonical_points, VALID, WRIST_VEL,
                       FINGERS, WRIST_POS, NUM_FEATURES)

COMPETITION = "asl-fingerspelling"
METADATA_FILES = ["train.csv", "supplemental_metadata.csv"]
HANDS = ("right", "left")
PROVENANCE_GAPS = [
    "No per-frame timestamps or frame rate in the dataset; target_fps is an assumption.",
    "No session or recording-device IDs; each sequence_id is treated as one recording.",
    "Camera image size/aspect ratio not given; landmark x and y are in normalised image units.",
    "Signing hand is chosen per sequence as the hand tracked in more frames.",
]


# ===================================================================== Kaggle
def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_competition_files(api, competition=COMPETITION, page_size=200):
    """All files with their sizes in bytes (follows pagination)."""
    import pandas as pd

    rows, token = [], None
    while True:
        resp = api.competition_list_files(competition, page_token=token, page_size=page_size)
        for f in resp.files:
            size = getattr(f, "total_bytes", None)
            if size is None:
                size = getattr(f, "totalBytes", None)
            rows.append({"name": f.name, "bytes": int(size or 0)})
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    return pd.DataFrame(rows)


def download_competition_file(api, name, dest_dir, competition=COMPETITION):
    """Download ONE file into dest_dir (unzipping if Kaggle zipped it).
    Returns the local path of the file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    if target.exists():
        return target
    api.competition_download_file(competition, name, path=str(dest_dir), quiet=True)
    zipped = target.with_name(target.name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            member = [m for m in z.namelist() if m.endswith(Path(name).name)][0]
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zipped.unlink()
    if not target.exists():
        raise FileNotFoundError(f"Download of {name} did not produce {target}")
    return target


def free_gb(path):
    return shutil.disk_usage(str(path)).free / 1e9


# ================================================================== metadata
def load_metadata(csv_paths):
    import pandas as pd

    frames = []
    for p in csv_paths:
        df = read_csv_text(p)
        df["source"] = Path(p).stem
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    return annotate_compatibility(meta)


def read_csv_text(path):
    """read_csv that keeps words such as "NA" or "NAN" as text (pandas would
    otherwise turn them into missing values)."""
    import pandas as pd

    return pd.read_csv(path, keep_default_na=False)


def read_manifest(path):
    return read_csv_text(path)


def _incompatibility(text):
    t = vocab.normalise_transcript(text)
    if not t:
        return "empty"
    if vocab.is_compatible(t):
        return ""
    if " " in t:
        return "space"
    if any(c.isdigit() for c in t):
        return "digit"
    return "other_symbol"


def annotate_compatibility(meta):
    meta = meta.copy()
    meta["transcript"] = meta["phrase"].astype(str).map(vocab.normalise_transcript)
    meta["exclude_reason"] = meta["phrase"].astype(str).map(_incompatibility)
    meta["compatible"] = meta["exclude_reason"] == ""
    meta["label_len"] = meta["transcript"].str.len()
    return meta


def select_pilot_subset(meta, file_sizes, budget_gb=5.0, max_sequences=3000,
                        min_signers=6, seed=0):
    """Choose which parquet files to download and which sequences to keep.

    Files are ranked by "compatible sequences per GB", then taken while the
    total download stays under budget_gb. Sequences are then sampled (with
    a fixed seed) up to max_sequences. Returns (selection DataFrame, summary).
    """
    ok = meta[meta["compatible"]]
    sizes = dict(zip(file_sizes["name"], file_sizes["bytes"]))
    per_file = ok.groupby("path").agg(n=("sequence_id", "size"),
                                      signers=("participant_id", "nunique")).reset_index()
    per_file["bytes"] = per_file["path"].map(sizes).fillna(0).astype(np.int64)
    per_file = per_file[per_file["bytes"] > 0]
    if per_file.empty:
        raise ValueError("No metadata 'path' matched a listed file name; compare "
                         "meta['path'].head() with file_sizes['name'].head()")
    per_file["per_gb"] = per_file["n"] / (per_file["bytes"] / 1e9)
    per_file = per_file.sort_values("per_gb", ascending=False)

    chosen, total = [], 0
    for r in per_file.itertuples():
        if total + r.bytes > budget_gb * 1e9:
            continue
        chosen.append(r.path)
        total += r.bytes
        if ok[ok["path"].isin(chosen)].shape[0] >= max_sequences * 1.2:
            break
    sel = ok[ok["path"].isin(chosen)]
    if len(sel) > max_sequences:
        sel = sel.sample(n=max_sequences, random_state=seed)
    sel = sel.sort_values(["path", "sequence_id"]).reset_index(drop=True)
    summary = {
        "budget_gb": budget_gb, "download_gb": round(total / 1e9, 3),
        "files": len(chosen), "sequences": int(len(sel)),
        "signers": int(sel["participant_id"].nunique()), "seed": seed,
        "max_sequences": max_sequences,
        "warning": (f"only {sel['participant_id'].nunique()} signers (< {min_signers}); "
                    "increase the budget for a meaningful signer split")
        if sel["participant_id"].nunique() < min_signers else "",
    }
    return sel, summary


# ================================================================ extraction
def hand_columns(hand):
    return [f"{a}_{hand}_hand_{i}" for i in range(21) for a in "xyz"]


def parquet_schema(path):
    import pyarrow.parquet as pq

    return pq.read_schema(str(path)).names


def read_hands(path, sequence_ids):
    """Read only frame + both hands' columns for the given sequences."""
    import pyarrow.parquet as pq

    cols = ["frame"] + hand_columns("right") + hand_columns("left")
    names = parquet_schema(path)
    missing = [c for c in cols if c not in names]
    if missing:
        raise KeyError(f"{path}: expected columns not found, e.g. {missing[:3]}")
    if "sequence_id" in names:
        table = pq.read_table(str(path), columns=["sequence_id"] + cols,
                              filters=[("sequence_id", "in", list(map(int, sequence_ids)))])
        df = table.to_pandas()
        if "sequence_id" not in df.columns:      # pandas metadata turned it back into the index
            df = df.reset_index()
    else:  # stored only as pandas index
        import pandas as pd
        df = pd.read_parquet(path, columns=cols)
        df = df[df.index.isin(sequence_ids)].reset_index()
    return df


def sequence_record(sequence_id, df_seq):
    """One sequence -> compact record with the signing hand only.

    Frames are sorted; missing frame numbers inside the range are inserted
    as NaN (invalid) rows so time stays evenly spaced.
    """
    df_seq = df_seq.sort_values("frame")
    frames = df_seq["frame"].to_numpy(np.int64)
    full = np.arange(frames.min(), frames.max() + 1)
    pos = np.searchsorted(full, frames)
    hands = {}
    counts = {}
    for h in HANDS:
        arr = df_seq[hand_columns(h)].to_numpy(np.float32).reshape(-1, 21, 3)
        pts = np.full((len(full), 21, 3), np.nan, np.float32)
        pts[pos] = arr
        hands[h] = pts
        counts[h] = int(np.all(np.isfinite(arr.reshape(len(arr), -1)), axis=1).sum())
    hand = "right" if counts["right"] >= counts["left"] else "left"
    return {"sequence_id": int(sequence_id), "hand": hand, "points": hands[hand],
            "frames": full.astype(np.int32), "valid_right": counts["right"],
            "valid_left": counts["left"], "inserted_frames": int(len(full) - len(frames)),
            "duplicate_frames": int(len(frames) - len(np.unique(frames)))}


def save_shard(path, records):
    """Compact .npz without pickled objects (safe to load anywhere)."""
    lengths = np.array([len(r["points"]) for r in records], np.int64)
    np.savez_compressed(
        path,
        sequence_id=np.array([r["sequence_id"] for r in records], np.int64),
        hand=np.array([r["hand"] for r in records]),
        lengths=lengths,
        points=np.concatenate([r["points"] for r in records]) if records else np.zeros((0, 21, 3), np.float32),
        frames=np.concatenate([r["frames"] for r in records]) if records else np.zeros(0, np.int32),
        valid_right=np.array([r["valid_right"] for r in records], np.int64),
        valid_left=np.array([r["valid_left"] for r in records], np.int64),
        inserted_frames=np.array([r["inserted_frames"] for r in records], np.int64),
        duplicate_frames=np.array([r["duplicate_frames"] for r in records], np.int64),
    )


def load_shard(path):
    z = np.load(path, allow_pickle=False)
    ends = np.cumsum(z["lengths"])
    starts = ends - z["lengths"]
    out = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        out.append({"sequence_id": int(z["sequence_id"][i]), "hand": str(z["hand"][i]),
                    "points": z["points"][a:b], "frames": z["frames"][a:b],
                    "valid_right": int(z["valid_right"][i]), "valid_left": int(z["valid_left"][i]),
                    "inserted_frames": int(z["inserted_frames"][i]),
                    "duplicate_frames": int(z["duplicate_frames"][i])})
    return out


def shard_name(parquet_path):
    return str(parquet_path).replace("/", "__").replace(".parquet", ".npz")


def prepare_shards(api, selection, tmp_dir, shard_dir, keep_raw=False, log=print):
    """For each selected parquet file: download to TEMPORARY disk, extract the
    selected sequences, save a compact shard to PERSISTENT storage, delete
    the raw file. Already finished shards are skipped (safe to rerun)."""
    tmp_dir, shard_dir = Path(tmp_dir), Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    groups = selection.groupby("path")["sequence_id"].apply(list)
    for i, (path, ids) in enumerate(groups.items(), 1):
        out = shard_dir / shard_name(path)
        if out.exists():
            log(f"[{i}/{len(groups)}] {path}: shard exists, skipped")
            continue
        local = download_competition_file(api, path, tmp_dir)
        df = read_hands(local, ids)
        records = [sequence_record(sid, g) for sid, g in df.groupby("sequence_id")]
        found = {r["sequence_id"] for r in records}
        lost = sorted(set(map(int, ids)) - found)
        tmp_out = out.with_name(out.stem + ".tmp.npz")
        save_shard(tmp_out, records)
        tmp_out.replace(out)                      # appears only when complete
        if not keep_raw:
            local.unlink()
        log(f"[{i}/{len(groups)}] {path}: kept {len(records)} sequences "
            f"({out.stat().st_size / 1e6:.1f} MB)" + (f", {len(lost)} not found" if lost else "")
            + f"; temp disk free {free_gb(tmp_dir):.1f} GB")


def load_all_shards(shard_dir):
    recs = {}
    for p in sorted(Path(shard_dir).glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        for r in load_shard(p):
            recs[r["sequence_id"]] = r
    return recs


# ===================================================================== split
def split_by_signer(meta, val_frac=0.15, test_frac=0.15, seed=0):
    """Assign whole SIGNERS (participant_id) to train / val / test.

    A signer's recordings (and any augmented copies, which are only made on
    the fly for the training split) never appear in two splits.
    Returns {participant_id: split}.
    """
    signers = np.array(sorted(meta["participant_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(signers)
    n = len(signers)
    if n < 3:
        raise ValueError(f"Need at least 3 signers for a signer split, got {n}")
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    out = {}
    for i, s in enumerate(signers):
        out[int(s)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def write_manifests(folder, selection, signer_split, summary, examples_info=None):
    """Save what was selected and how it was split, so runs can be repeated."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sel = selection.copy()
    sel["split"] = sel["participant_id"].map(signer_split)
    sel.to_csv(folder / "selection_manifest.csv", index=False)
    info = dict(summary, provenance_gaps=PROVENANCE_GAPS,
                split_counts=sel["split"].value_counts().to_dict(),
                split_signers=sel.groupby("split")["participant_id"].nunique().to_dict())
    if examples_info:
        info["examples"] = examples_info
    (folder / "dataset_summary.json").write_text(json.dumps(info, indent=2, default=int))
    return sel


# ================================================================== examples
def record_features(record, feature_cfg=None):
    """Raw record -> UNnormalised (T, 68) features, using the shared featurizer."""
    pts = canonical_points(record["points"], record["hand"])
    return SequenceFeaturizer(feature_cfg).featurize(pts)


def build_examples(selection, records, feature_cfg=None, max_frames=1000):
    """Join manifest rows with extracted landmarks. Recordings that cannot be
    used are EXCLUDED (never truncated): too short for their transcript
    under CTC, too long, missing, or with no valid hand frame."""
    examples, excluded = [], {"missing": 0, "too_short_for_ctc": 0, "too_long": 0, "no_hand": 0}
    for row in selection.itertuples():
        rec = records.get(int(row.sequence_id))
        if rec is None:
            excluded["missing"] += 1
            continue
        y = vocab.encode(row.transcript)
        x = record_features(rec, feature_cfg)
        if x[:, VALID].sum() == 0:
            excluded["no_hand"] += 1
        elif len(x) < vocab.required_frames(y):
            excluded["too_short_for_ctc"] += 1
        elif len(x) > max_frames:
            excluded["too_long"] += 1
        else:
            examples.append({"x": x, "y": np.array(y, np.int32), "text": row.transcript,
                             "sequence_id": int(row.sequence_id),
                             "signer": int(row.participant_id),
                             "split": getattr(row, "split", None)})
    return examples, excluded


# ================================================================== batching
def pad_batch(examples, xs=None):
    """Right-pad features and labels; keep the true lengths.
    Padding values: features 0, labels 0 (ignored by CTC beyond label_length)."""
    xs = xs if xs is not None else [e["x"] for e in examples]
    x_len = np.array([len(x) for x in xs], np.int32)
    y_len = np.array([len(e["y"]) for e in examples], np.int32)
    X = np.zeros((len(xs), x_len.max(), NUM_FEATURES), np.float32)
    Y = np.zeros((len(xs), max(1, y_len.max())), np.int32)
    for i, (x, e) in enumerate(zip(xs, examples)):
        X[i, :len(x)] = x
        Y[i, :len(e["y"])] = e["y"]
    return X, x_len, Y, y_len


def iterate_batches(examples, batch_size, normalizer, rng=None, augment_cfg=None):
    """Length-bucketed batches (similar lengths together = less padding).
    Shuffled when rng is given. Augmentation only if augment_cfg is given."""
    order = np.argsort([len(e["x"]) for e in examples], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    for idx in batches:
        exs = [examples[i] for i in idx]
        xs = [e["x"] for e in exs]
        if augment_cfg:
            xs = [augment(x, vocab.required_frames(e["y"].tolist()), rng, augment_cfg)
                  for x, e in zip(xs, exs)]
        xs = [normalizer.apply(x) for x in xs]
        yield pad_batch(exs, xs)


DEFAULT_AUGMENT = {"stretch": (0.8, 1.25), "rotate_deg": 10.0, "scale": (0.9, 1.1),
                   "aspect": (0.9, 1.1), "wrist_shift": 0.05, "noise": 0.01, "drop_frame": 0.03}


def augment(x, min_len, rng, cfg=DEFAULT_AUGMENT):
    """Training-only random changes to UNnormalised features (T, 68).
    Never shortens a sequence below what its transcript needs (min_len)."""
    x = x.copy()
    T = len(x)
    lo, hi = cfg["stretch"]
    r = rng.uniform(lo, hi)
    new_T = int(round(T * r))
    if new_T >= max(min_len, 1) and new_T != T:
        idx = np.minimum((np.arange(new_T) / r).astype(int), T - 1)
        x = x[idx]
        x[:, WRIST_VEL] /= r                      # faster playback = larger steps per frame
    valid = x[:, VALID] > 0.5
    f = x[:, FINGERS].reshape(len(x), 21, 3)
    a = np.deg2rad(rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    f[..., :2] = f[..., :2] @ rot.T
    f *= rng.uniform(*cfg["scale"])
    f[..., 0] *= rng.uniform(*cfg["aspect"])
    f += rng.normal(0, cfg["noise"], f.shape).astype(np.float32)
    f[:, 0] = 0.0                                 # wrist stays the origin
    x[:, FINGERS] = f.reshape(len(x), 63)
    x[:, WRIST_POS] += rng.uniform(-cfg["wrist_shift"], cfg["wrist_shift"], 2).astype(np.float32)
    drop = rng.random(len(x)) < cfg["drop_frame"]
    x[~valid | drop] = 0.0                        # invalid frames stay all-zero
    return x
