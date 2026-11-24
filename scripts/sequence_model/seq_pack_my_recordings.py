"""
seq_pack_my_recordings.py - Phase 2A: turn your own word-list recordings into one
small file of training examples for the Colab fine-tuning cell.

    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_pack_my_recordings.py

What it does, for every take in backend/storage/my_recordings (not in discarded/):
  1. reads the word (or phrase) and decides its group from the lists in data/:
       test        my_words_test.txt, my_phrases_test.txt          -> only for the honest score
       validation  every 7th line of each training list           -> picks the best epoch
       train       the other lines of my_words_train.txt,
                   my_phrases_train.txt, my_doubles_train.txt,
                   my_i_words_train.txt                           -> the model learns from these
     Each word or phrase is in exactly one group, and no training phrase may contain a test
     word, so the scores are on words the model never trained on.
  2. builds the model inputs exactly as the webcam app does (same code: asl.seq.inference.
     frame_features), at the real signing speed, with the bundle's feature settings.
  3. leaves out takes that cannot train well, and says why.

The output (default backend/storage/my_recordings_pack.npz) holds only hand-point numbers
and words, never video. It is personal data: upload it to your own Google Drive only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asl.seq import vocab  # noqa: E402
from asl.seq.inference import frame_features  # noqa: E402
from seq_record_word import keep_signing_hand, read_word_list, speed_times  # noqa: E402

VALIDATION_EVERY = 7          # every 7th training word is held back for validation
MAX_FRAMES = 1500             # the same limit as the FSboard clips
PACK_VERSION = "my-recordings-v1"


TRAIN_LISTS = ("my_words_train.txt", "my_phrases_train.txt", "my_doubles_train.txt", "my_i_words_train.txt")
TEST_LISTS = ("my_words_test.txt", "my_phrases_test.txt")


def word_groups(train_lists, test_lists, every=VALIDATION_EVERY):
    """word or phrase -> 'train' | 'validation' | 'test'.

    train_lists, test_lists: lists of word lists (one per file). Every `every`-th line of
    each training list is validation. Raises if anything is listed twice, is in both a
    training and a test list, or if a training phrase contains a test word."""
    groups = {}
    for words in train_lists:
        for i, w in enumerate(words):
            if w in groups:
                raise ValueError(f"{w} is in two training lists")
            groups[w] = "validation" if i % every == every - 1 else "train"
    test = [w for words in test_lists for w in words]
    both = sorted(set(groups) & set(test))
    if both:
        raise ValueError(f"these are in both a training and a test list: {', '.join(both)}")
    if len(set(test)) != len(test):
        raise ValueError("a test word or phrase is listed twice")
    leaks = sorted({f"{w} ({part})" for w in groups for part in w.split() if " " in w and part in test})
    if leaks:
        raise ValueError(f"these training phrases contain a test word: {', '.join(leaks)}")
    groups.update({w: "test" for w in test})
    return groups


def load_take(path):
    with np.load(path, allow_pickle=False) as z:
        points = [None if np.isnan(p).all() else p for p in z["points"]]
        labels = [str(x) or None for x in z["hand_labels"]]
        hand = str(z["hand"])
        hand_mode = str(z["hand_mode"]) if "hand_mode" in z.files else "auto"
        return {"file": path.name, "word": str(z["word"]), "pace": str(z["pace"]) if "pace" in z.files else "",
                "t": z["t"].astype(np.float64), "hand": hand, "mirror_input": bool(z["mirror_input"]),
                "image_size": (int(z["image_width"]), int(z["image_height"])),
                "used": keep_signing_hand(points, labels, hand) if hand_mode == "auto" else points}


def take_features(take, feature_cfg):
    """Model inputs (unnormalised) at the real signing speed, as the webcam app builds them."""
    return frame_features(speed_times(take["t"], 1.0), take["used"], take["hand"], feature_cfg, take["image_size"])


def pack(folder, train_lists, test_lists, feature_cfg, every=VALIDATION_EVERY, max_frames=MAX_FRAMES):
    """All usable takes -> (arrays for np.savez, summary dict). The lists are lists of word lists."""
    groups = word_groups(train_lists, test_lists, every)
    kept, excluded = [], Counter()
    for path in sorted(Path(folder).glob("*.npz")):
        try:
            take = load_take(path)
        except (OSError, KeyError, ValueError):
            excluded["not a recording from seq_record_word.py"] += 1
            continue
        word = take["word"]
        if not word:
            excluded["no word saved with it"] += 1
            continue
        if word not in groups:
            excluded["word is in neither word list"] += 1
            continue
        if not take["mirror_input"]:
            excluded["recorded without --mirror-input"] += 1
            continue
        needed = vocab.required_frames(vocab.encode(word))
        n_hand = sum(p is not None for p in take["used"])
        if n_hand < needed:
            excluded["too few frames with a hand for its letters"] += 1
            continue
        x = take_features(take, feature_cfg)
        if len(x) < needed:
            excluded["too short for its letters"] += 1
        elif len(x) > max_frames:
            excluded["too long"] += 1
        else:
            kept.append(dict(take, x=x.astype(np.float32), split=groups[word]))

    arrays = {
        "x": np.concatenate([k["x"] for k in kept]) if kept else np.zeros((0, 68), np.float32),
        "lengths": np.array([len(k["x"]) for k in kept], np.int32),
        "words": np.array([k["word"] for k in kept], dtype=str),
        "paces": np.array([k["pace"] for k in kept], dtype=str),
        "splits": np.array([k["split"] for k in kept], dtype=str),
        "files": np.array([k["file"] for k in kept], dtype=str),
        "feature_cfg": np.array(json.dumps(feature_cfg, sort_keys=True)),
        "pack_version": np.array(PACK_VERSION),
    }
    summary = {"excluded": dict(excluded), "groups": {}}
    for split in ("train", "validation", "test"):
        rows = [k for k in kept if k["split"] == split]
        wanted = sorted(w for w, g in groups.items() if g == split)
        summary["groups"][split] = {
            "takes": len(rows), "words": len({k["word"] for k in rows}), "words_in_list": len(wanted),
            "phrase_takes": sum(" " in k["word"] for k in rows),
            "paces": dict(Counter(k["pace"] or "none" for k in rows)),
            "letters_missing": sorted(set(vocab.LETTERS) - set("".join(k["word"] for k in rows))),
        }
    return arrays, summary


def load_pack(path):
    """The packed file -> list of example dicts (the form the training code uses) + feature settings."""
    with np.load(path, allow_pickle=False) as z:
        ends = np.cumsum(z["lengths"])
        starts = ends - z["lengths"]
        examples = [{"x": z["x"][s:e], "y": np.array(vocab.encode(str(w)), np.int32), "text": str(w),
                     "pace": str(p), "split": str(g), "sequence_id": str(f), "signer": "me"}
                    for s, e, w, p, g, f in zip(starts, ends, z["words"], z["paces"], z["splits"], z["files"])]
        return examples, json.loads(str(z["feature_cfg"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folder", nargs="?", default=str(ROOT / "backend" / "storage" / "my_recordings"))
    ap.add_argument("--train-words", nargs="+", help="training lists (default: the ones in data/ that exist: "
                    + ", ".join(TRAIN_LISTS) + ")")
    ap.add_argument("--test-words", nargs="+", help="test lists (default: the ones in data/ that exist: "
                    + ", ".join(TEST_LISTS) + ")")
    ap.add_argument("--bundle", default=str(ROOT / "backend" / "models" / "long_memory_model"),
                    help="the model you will fine-tune: its feature settings are used")
    ap.add_argument("--out", default=str(ROOT / "backend" / "storage" / "my_recordings_pack.npz"))
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"No recordings folder at {folder}. Record with seq_record_word.py --words ... --save {folder}")
    prep = json.loads((Path(args.bundle) / "preprocessing.json").read_text())
    train_files = args.train_words or [str(ROOT / "data" / n) for n in TRAIN_LISTS if (ROOT / "data" / n).exists()]
    test_files = args.test_words or [str(ROOT / "data" / n) for n in TEST_LISTS if (ROOT / "data" / n).exists()]
    print("Training lists:", ", ".join(Path(f).name for f in train_files))
    print("Test lists:    ", ", ".join(Path(f).name for f in test_files))
    arrays, summary = pack(folder, [read_word_list(f) for f in train_files],
                           [read_word_list(f) for f in test_files], prep["features"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)

    print(f"{'group':<11}{'takes':>6}{'words':>9}   paces")
    for split, g in summary["groups"].items():
        print(f"{split:<11}{g['takes']:>6}{g['words']:>4} of {g['words_in_list']:<3}  {g['paces']}"
              + (f"  ({g['phrase_takes']} phrase takes)" if g["phrase_takes"] else "")
              + (f"   letters never seen: {''.join(g['letters_missing'])}" if g["takes"] and g["letters_missing"] else ""))
    for reason, n in summary["excluded"].items():
        print(f"left out: {n} take(s), {reason}")
    if not summary["groups"]["train"]["takes"] or not summary["groups"]["validation"]["takes"]:
        print("NOTE: fine-tuning needs takes in both 'train' and 'validation'. Record more of my_words_train.txt.")
    if not summary["groups"]["test"]["takes"]:
        print("NOTE: no test takes yet, so there is no honest score. Record data/my_words_test.txt too.")
    print(f"\nSaved {out} ({out.stat().st_size / 1e6:.2f} MB): hand-point numbers and words only, no video.")
    print("Upload it to Google Drive: MyDrive/asl_fingerspelling/fsboard_tcn_ctc/my_recordings/")
    return summary


if __name__ == "__main__":
    main()
