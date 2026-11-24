"""
build_tcn_ctc_notebook.py - Writes notebooks/fsboard_tcn_ctc_colab.ipynb.

The notebook carries exact copies of the backend code it needs, so uploading that one
file to Colab is enough. It is the single source of the notebook: edit the cells here,
not in Colab, then run this again.
tests/test_seq_fsboard_video.py fails if a copy and the repository differ.
After changing that code, run this again:

    venv\\Scripts\\python.exe scripts\\sequence_model\\build_tcn_ctc_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fsboard_tcn_ctc_colab.ipynb"
EMBEDDED = ("asl/__init__.py", "asl/features.py", "asl/seq/__init__.py",
            "asl/seq/vocab.py", "asl/seq/features.py", "asl/seq/ctc.py", "asl/seq/model.py",
            "asl/seq/fsboard.py", "asl/seq/fsboard_video.py", "asl/seq/data.py", "asl/seq/train.py",
            "asl/seq/evaluate.py")
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text, form=False):
    meta = {"cellView": "form"} if form else {}
    CELLS.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


def setup_cell():
    blocks = []
    for name in EMBEDDED:
        text = (ROOT / "backend" / name).read_text(encoding="utf-8")
        if "'''" in text or text.endswith("\\"):
            raise ValueError(f"{name} cannot be embedded in a raw triple-quoted string")
        blocks.append(f'    "{name}": r\'\'\'{text}\'\'\',\n')
    return ('#@title Setup: run this first (the code stays hidden; click "Show code" to read it)\n'
            "# Writes the project's code to the temporary Colab disk. These are exact copies\n"
            "# of the files named below, from backend/ in the repository.\n"
            "import importlib, json, os, shutil, sys\n"
            "from pathlib import Path\n\n"
            'WORK = Path(os.environ.get("FSBOARD_WORK", "/content/fsboard_tcn"))   # temporary Colab disk\n'
            'CODE = WORK / "code"\n\n'
            "_FILES = {\n" + "".join(blocks) + "}\n"
            "for _name, _text in _FILES.items():\n"
            "    _path = CODE / _name\n"
            "    _path.parent.mkdir(parents=True, exist_ok=True)\n"
            '    _path.write_text(_text, encoding="utf-8")\n'
            "if str(CODE) not in sys.path:\n"
            "    sys.path.insert(0, str(CODE))\n\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import HTML, display\n"
            "import asl.seq.vocab as vocab, asl.seq.features as features, asl.seq.fsboard as fb\n"
            "import asl.seq.fsboard_video as fsv, asl.seq.model as smodel, asl.seq.train as strain\n"
            "import asl.seq.evaluate as seval\n"
            "for _m in (vocab, features, fb, fsv, smodel, strain, seval):\n"
            "    importlib.reload(_m)\n\n"
            "try:\n"
            "    _found = fb.kaggle_auth_from_colab_secrets()\n"
            "except ImportError:\n"
            "    _found = []\n"
            'print("Setup finished. Temporary working folder:", WORK)\n'
            'print("Kaggle key from Colab Secrets:", ", ".join(_found) if _found else\n'
            '      "none set (fine: FSboard downloaded without one when last tested)")\n')


md(r"""
# ASL fingerspelling: train and evaluate the TCN + CTC model on FSboard

**What this notebook does.** It teaches a small network to read fingerspelled words and short phrases from video,
letter by letter and with spaces between words, in order. Everything happens here in Google Colab. Nothing from the
dataset goes on your laptop.

| Step | What happens |
|---|---|
| 1 | Check the GPU and connect Google Drive |
| 2 | Choose your data: one-word clips and phrases with spaces |
| 3 | Run MediaPipe on the videos to get **all 21 hand points** |
| 4 | Turn the points into model inputs and check them |
| 5 | Optional tiny test: memorise a few clips, to prove the pipeline works |
| 6 | Train the model (speed range, longer memory, start from your earlier model) |
| 7 | Evaluate it on signers it has never seen, next to your earlier model |
| 8 | Speed check: the same test clips signed faster and slower |
| 9 | Export a small model file |
| 10 | Optional: fine-tune on your own webcam recordings (off unless you switch it on) |

**Why videos?** FSboard's ready-made landmark files hold only 8 of the 21 hand points, so we make our own.

### How to use it
1. Upload this one file at [colab.research.google.com](https://colab.research.google.com) (**File → Upload notebook**).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.** A Google pop-up asks to connect your Drive.
4. The settings are those of the full run. Step 3 can take hours the first time; the progress lines estimate it.

### If Colab disconnects, or in a new session
Colab forgets every variable when a session ends. Step 4 saves its result to Drive, so you don't repeat steps 2–4:
run **Setup** and **step 1**, then go straight to the step you want (5 to 10). It prints which run it loaded.
Run steps 2, 3 and 4 again only when you change which clips to use. They need no GPU, so a CPU runtime saves your limit.
Training continues from the last saved epoch.

### Where files go
| Place | What | How long it lasts |
|---|---|---|
| **Temporary Colab disk** | code, metadata, MediaPipe, one video at a time | erased when the session ends |
| **Google Drive** `MyDrive/asl_fingerspelling/fsboard_tcn_ctc/` | hand points, checkpoints, best models, reports, zips | until you delete it |
| **Your laptop** | nothing, unless you choose to download a zip | |

Earlier runs on your Drive are never changed: every new setting gets its own run folder.

**Please keep the videos private.** FSboard's rules ask you to blur signers' faces when publicizing examples. This
notebook never displays a video. Credit: FSboard by Georg et al., CC BY 4.0
([paper](https://arxiv.org/abs/2407.15806), [dataset](https://www.kaggle.com/datasets/googleai/fsboard)).
""")

code(setup_cell(), form=True)

md(r"""
---
## 1. Check the GPU and connect Google Drive

**In plain English.** A GPU makes training much faster. Google Drive keeps your work safe, because Colab's own disk
is erased when the session ends.

**What success looks like:** a line starting `GPU: [PhysicalDevice`, `Mounted at /content/drive`, and your free space.
If it says `GPU: NONE`, use **Runtime → Change runtime type → T4 GPU**. A CPU also works, just more slowly.

`RESUME_RUN` is only for new sessions. Empty means "the data step 4 saved last". To use an older run, put its
folder name from `runs/` here.
""")

code(r"""
import platform
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("Python", platform.python_version(), "| TensorFlow", tf.__version__)
print("GPU:", gpus if gpus else "NONE (Runtime > Change runtime type > T4 GPU)")

if os.environ.get("FSBOARD_DRIVE"):
    DRIVE = Path(os.environ["FSBOARD_DRIVE"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/asl_fingerspelling/fsboard_tcn_ctc")
SEQUENCES = DRIVE / "sequences"
SEQUENCES.mkdir(parents=True, exist_ok=True)
print("Saved work goes to:", DRIVE)
print(f"Free space: Colab disk {fb.free_gb(WORK):.0f} GB, Drive {fb.free_gb(DRIVE):.1f} GB")

# ---- Save and reload the prepared data, so steps 5-10 work in a new session without steps 2-4 ----
RESUME_RUN = ""  #@param {type:"string"}


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def save_prepared():
    # End of step 4: write the model inputs and their settings to Drive as one file.
    base = RUN_NAME.split("__")[0]
    rows = [(split, e) for split in fsv.SPLITS for e in examples[split]]
    info = {"run": base, "seed": SEED, "feature_cfg": FEATURE_CFG, "prep": PREP, "unfamiliar": unfamiliar,
            "examples": [dict({k: _plain(v) for k, v in e.items() if k not in ("x", "y")}, group=split)
                         for split, e in rows]}
    local = WORK / "prepared.npz"
    np.savez_compressed(local, x=np.concatenate([e["x"] for _, e in rows]),
                        x_lengths=np.array([len(e["x"]) for _, e in rows], np.int64),
                        y=np.concatenate([np.asarray(e["y"]) for _, e in rows]),
                        y_lengths=np.array([len(e["y"]) for _, e in rows], np.int64),
                        info=np.array(json.dumps(info)))
    size_gb = local.stat().st_size / 1e9
    if fb.free_gb(DRIVE) < size_gb + 0.2:
        print(f"NOT saved for later sessions: the file is {size_gb:.2f} GB and Drive has "
              f"{fb.free_gb(DRIVE):.1f} GB free. This session still works; free some Drive space and run step 4 again.")
        return
    target = DRIVE / "runs" / base / "prepared.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local, target)
    (DRIVE / "last_run.json").write_text(json.dumps({"run": base}))
    print(f"Saved for later sessions: {target} ({size_gb * 1000:.0f} MB, {len(rows)} clips). "
          "In a new session, run Setup and step 1, then go straight to step 5 or later.")


def need_data(trained=False):
    # First line of steps 5-10. Does nothing if this session already has the data;
    # otherwise loads what step 4 saved. trained=True also finds the trained model's folder.
    g = globals()
    if not all(k in g for k in ("examples", "normalizer", "PREP", "RUN_NAME", "RUN_DIR", "SEED", "unfamiliar")):
        pointer = DRIVE / "last_run.json"
        base = RESUME_RUN.split("__")[0] or (json.loads(pointer.read_text())["run"] if pointer.exists() else "")
        saved = DRIVE / "runs" / base / "prepared.npz"
        if not base or not saved.exists():
            raise FileNotFoundError(
                "This session has no prepared data, and none is saved on Drive"
                + (f" for the run '{base}'" if base else "") + ".\n"
                "What to do: run steps 2, 3 and 4 once (a CPU runtime is enough). Step 4 saves the result, "
                "and later sessions then only need Setup and step 1.")
        local = WORK / "prepared_loaded.npz"
        shutil.copy(saved, local)
        with np.load(local, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            x, y = z["x"], z["y"]
            x_ends, y_ends = np.cumsum(z["x_lengths"]), np.cumsum(z["y_lengths"])
            x_lengths, y_lengths = z["x_lengths"], z["y_lengths"]
        loaded = {split: [] for split in fsv.SPLITS}
        for i, row in enumerate(info["examples"]):
            e = dict(row)
            e["x"] = x[x_ends[i] - x_lengths[i]:x_ends[i]]
            e["y"] = y[y_ends[i] - y_lengths[i]:y_ends[i]]
            loaded[e.pop("group")].append(e)
        g.update(examples=loaded, PREP=info["prep"], FEATURE_CFG=info["feature_cfg"], SEED=info["seed"],
                 unfamiliar=info["unfamiliar"], normalizer=features.Normalizer.from_dict(info["prep"]["normalizer"]),
                 MODEL_CFG=dict(smodel.DEFAULT_MODEL_CONFIG), RUN_NAME=info["run"],
                 RUN_DIR=DRIVE / "runs" / info["run"])
        print(f"Loaded the prepared data of run '{info['run']}' from Drive:",
              ", ".join(f"{len(loaded[s])} {s}" for s in fsv.SPLITS), "clips.")
        print("   If you changed step 2 since then, run steps 2, 3 and 4 again instead.")
    if trained and not (g["RUN_DIR"] / "best_model" / "model.weights.h5").exists():
        base = g["RUN_NAME"].split("__")[0]
        asked = DRIVE / "runs" / RESUME_RUN / "best_model" / "model.weights.h5"
        if "__" in RESUME_RUN and RESUME_RUN.split("__")[0] == base and asked.exists():
            found = [asked]
        else:
            found = sorted((p for p in (DRIVE / "runs").glob(base + "__*/best_model/model.weights.h5")
                            if not p.parents[1].name.endswith("__personal")), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"No trained model for the run '{base}' on Drive yet.\n"
                                    "What to do: run step 6 first.")
        g.update(RUN_DIR=found[-1].parents[1], RUN_NAME=found[-1].parents[1].name)
        print(f"Using the trained model of run '{g['RUN_NAME']}' (the newest for this data). "
              "For another one, put its run name in RESUME_RUN in step 1 and run step 1 again.")
""")

md(r"""
---
## 2. Choose your data

**In plain English.** Each clip has a phrase, a signer and an official group: **train**, **validation** or **test**.
Only metadata is downloaded here (about 70 MB). This shows what you would get **before** any video is downloaded.

**Two kinds of clips:**
* **One-word clips** (letters only), exactly as in your earlier runs. `0` means all of them, so the same 131 test
  clips can still be compared with earlier models.
* **Phrases with spaces**, such as `new york`, up to the `PHRASE_CLIPS_*` limits. Only single spaces between words.

**What you can change:**
* `CLIPS_TRAIN`, `CLIPS_VALIDATION`, `CLIPS_TEST`: how many one-word clips (`0` = all).
* `INCLUDE_PHRASES` and `PHRASE_CLIPS_*`: how many phrases. More phrases take longer in step 3. Clips already
  extracted are skipped, so you can start small and raise the limits later.
* `MAX_LETTERS`, `MAX_PHRASE_LETTERS`: the longest word and phrase to accept (letters only, spaces not counted).

**Extra training clips, chosen by a rule you write.** The clips above are picked at random. On top of them,
`EXTRA_TRAIN_CLIPS` more **training** clips are added: the ones your `clip_score` function scores highest among the
clips not chosen yet. Validation and test clips are never touched, so results stay comparable with earlier runs.

`clip_score(text)` is plain Python at the top of the cell. It gets one phrase in capitals, such as `NEW YORK`, and
returns a number. Higher is picked earlier; `0` is never picked. As written it gives 1 point for every letter in
`WEAK_LETTERS` (copy those from step 7's "weakest letters first" table) and 1 point for a double letter.

| You want | What to write inside `clip_score` |
|---|---|
| Other weak letters | change `WEAK_LETTERS = "QXYUZP"` |
| One letter to count more | `score += 2 * text.count("Q")` |
| More phrases with several words | `score += text.count(" ")` |
| Only short clips | `if len(text) > 15: return 0` as the first line |
| Double letters to count more | change `score += 1` to `score += 3` |
| No extra clips | set `EXTRA_TRAIN_CLIPS = 0` |

After the table, the cell prints how often each weak letter is in the training clips **before -> after**, and the
highest-scoring extra clips. Check those two lines whenever you change the rule. A new rule gets a new run folder.

**Skipped whole:** anything with a digit or symbol, such as `2067 pso costa`, because the hands also spell those and
the model has no outputs for them. Nothing is ever cut out of a phrase.

**Validation and test** use only people who are **not** in the training metadata, so the results describe unfamiliar signers.

**What success looks like:** a table of clips per group, the weak-letter counts before and after, a few example
phrases, and the run folder's name.
""")

code(r"""
import hashlib

CLIPS_TRAIN = 0                 #@param {type:"integer"}
CLIPS_VALIDATION = 0            #@param {type:"integer"}
CLIPS_TEST = 0                  #@param {type:"integer"}
MIN_LETTERS = 2                 #@param {type:"integer"}
MAX_LETTERS = 100               #@param {type:"integer"}
INCLUDE_PHRASES = True          #@param {type:"boolean"}
PHRASE_CLIPS_TRAIN = 8000       #@param {type:"integer"}
PHRASE_CLIPS_VALIDATION = 2000  #@param {type:"integer"}
PHRASE_CLIPS_TEST = 2000        #@param {type:"integer"}
MAX_PHRASE_LETTERS = 40         #@param {type:"integer"}
SEED = 42                       #@param {type:"integer"}
SKIP_SENSITIVE = True           #@param {type:"boolean"}
EXTRA_TRAIN_CLIPS = 3000        #@param {type:"integer"}
WEAK_LETTERS = "QXYUZP"         #@param {type:"string"}


# ============================== YOUR SELECTION RULE: edit this function ==============================
def clip_score(text):
    # How much you want this clip as an EXTRA training clip.
    # text is the phrase in capitals, for example "NEW YORK". Return a number:
    # higher = picked earlier, 0 = never picked.
    score = sum(text.count(letter) for letter in WEAK_LETTERS)       # 1 point for every weak letter
    if any(a == b != " " for a, b in zip(text, text[1:])):           # 1 point if it has a double letter
        score += 1
    return score
# ====================================== end of your rule =============================================


wanted = {"train": CLIPS_TRAIN, "validation": CLIPS_VALIDATION, "test": CLIPS_TEST}
wanted_phrases = {"train": PHRASE_CLIPS_TRAIN, "validation": PHRASE_CLIPS_VALIDATION, "test": PHRASE_CLIPS_TEST}
limits = dict(min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, skip_sensitive=SKIP_SENSITIVE)
phrase_limits = dict(min_letters=MIN_LETTERS, max_letters=MAX_PHRASE_LETTERS, skip_sensitive=SKIP_SENSITIVE,
                     phrases=True)
metadata = {}
for split in fsv.SPLITS:
    name = fsv.metadata_path(split)
    metadata[split] = fb.load_metadata(fb.download(name, WORK / "metadata" / Path(name).name,
                                                   expected_size=fb.remote_size(name)))

train_signers = {r["signerId"] for r in metadata["train"]}
clips, summary, unfamiliar, extra_rows, random_train = {}, [], {}, [], []
for split in fsv.SPLITS:
    rows, skipped = fsv.select_clips(metadata[split], split, **limits)
    available = len(rows)
    unfamiliar[split] = split != "train"
    if split != "train":
        kept, removed = fsv.drop_seen_signers(rows, train_signers)
        if kept:
            rows = kept
        else:                               # nothing left: use them, and say so in the report
            unfamiliar[split] = False
            print(f"WARNING: every usable {split} clip is from a training signer.")
    clips[split] = fsv.limit_clips(rows, wanted[split], SEED)
    one_word = len(clips[split])
    print(f"{split}: one-word clips skipped whole because:", dict(skipped.most_common()))

    phrase_rows, phrases_usable, found = [], 0, []
    if INCLUDE_PHRASES:
        found, phrase_skipped = fsv.select_clips(metadata[split], split, **phrase_limits)
        if split != "train":
            found, _ = fsv.drop_seen_signers(found, train_signers)   # evaluation: unfamiliar signers only
        phrases_usable = len(found)
        phrase_rows = fsv.limit_clips(found, wanted_phrases[split], SEED)
        clips[split] = clips[split] + phrase_rows
        print(f"{split}: phrases skipped whole because:", dict(phrase_skipped.most_common(6)))
    extra = []
    if split == "train" and EXTRA_TRAIN_CLIPS > 0:          # training only: evaluation clips stay as they were
        random_train = list(clips[split])
        taken = {r["filename"] for r in random_train}
        extra = extra_rows = fsv.choose_extra([r for r in rows + found if r["filename"] not in taken],
                                              clip_score, EXTRA_TRAIN_CLIPS, SEED)
        clips[split] = sorted(random_train + extra, key=lambda r: r["filename"])
    summary.append({"group": split, "clips in metadata": f"{len(metadata[split]):,}",
                    "one-word usable": available, "one-word chosen": one_word,
                    "phrases usable": phrases_usable, "phrases chosen": len(phrase_rows),
                    "extra by your rule": len(extra),
                    "signers": len({r["signer_id"] for r in clips[split]}),
                    "video download (GB)": round(sum(r["size_bytes"] for r in clips[split]) / 1e9, 2)})
display(HTML(fsv.html_table(summary)))
if not clips["train"] or not clips["validation"] or not clips["test"]:
    raise fb.FsboardError("A group has no clips.\nWhat to check: the limits at the top of this cell.")
print("Examples of chosen phrases:", [r["answer"] for r in clips["train"] if " " in r["answer"]][:8])
if EXTRA_TRAIN_CLIPS > 0:
    def letter_counts(rows):
        return {c: sum(r["answer"].count(c) for r in rows) for c in WEAK_LETTERS}
    was, now = letter_counts(random_train), letter_counts(clips["train"])
    print(f"Your rule added {len(extra_rows)} training clips (asked for {EXTRA_TRAIN_CLIPS}).")
    if len(extra_rows) < EXTRA_TRAIN_CLIPS:
        print("   Fewer than asked: no more unused training clips score above 0.")
    print("Weak letters in the training clips, before -> after:",
          ", ".join(f"{c} {was[c]} -> {now[c]}" for c in WEAK_LETTERS))
    top = sorted(extra_rows, key=lambda r: -clip_score(r["answer"]))[:8]
    print("Highest-scoring extra clips:", [(r["answer"], clip_score(r["answer"])) for r in top])

# Different data gets its own run folder, so these results never mix with earlier runs.
RUN_NAME = f"train{len(clips['train'])}_val{len(clips['validation'])}_test{len(clips['test'])}_letters{MIN_LETTERS}-{MAX_LETTERS}"
if INCLUDE_PHRASES:
    RUN_NAME += f"_phrases{MAX_PHRASE_LETTERS}"
if extra_rows:                             # a different rule picks different clips: give it its own folder
    picked = hashlib.md5("\n".join(r["filename"] for r in extra_rows).encode()).hexdigest()[:6]
    RUN_NAME += f"_extra{len(extra_rows)}-{picked}"
RUN_DIR = DRIVE / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "selection.json").write_text(json.dumps({"seed": SEED, "limits": limits, "wanted": wanted,
                                                    "phrase_limits": phrase_limits if INCLUDE_PHRASES else None,
                                                    "wanted_phrases": wanted_phrases if INCLUDE_PHRASES else None,
                                                    "unfamiliar_signers": unfamiliar,
                                                    "extra": {"asked": EXTRA_TRAIN_CLIPS, "weak_letters": WEAK_LETTERS,
                                                              "chosen": [r["filename"] for r in extra_rows]},
                                                    "clips": clips}, indent=1))
print("Run folder:", RUN_DIR)
""")

md(r"""
---
## 3. Extract all 21 hand points from the videos

**In plain English.** For each clip: download the video to temporary disk, check its fingerprint (MD5), turn it upright,
mirror it the way your webcam app does, run MediaPipe, save the small result to Drive, and **delete the video**.

MediaPipe gets its own small Python environment, because it needs older package versions than Colab's TensorFlow.
It is the same Python (3.11) and MediaPipe (0.10.14) as your laptop.

Safe to interrupt: finished clips are kept and skipped next time. One bad clip doesn't stop the run, but five failures
in a row do, because that usually means a connection or permission problem.

`DETECTION_CONFIDENCE` is how sure MediaPipe must be before accepting a hand. If step 4 shows few frames with a hand,
try a lower value such as `0.3`, but first delete `sequences/` from the Drive folder, or finished clips are skipped.

**What success looks like:** a line with the MediaPipe version, progress lines with minutes left, then `0 failed`. The
time depends on the number of clips and hasn't been measured on Colab yet; the progress lines estimate it.
""")

code(r"""
DETECTION_CONFIDENCE = 0.5  #@param {type:"number"}

os.environ["MPLBACKEND"] = "Agg"   # the MediaPipe helper must not inherit Colab's plotting backend
if os.environ.get("FSBOARD_MP_PYTHON"):
    mediapipe_python = os.environ["FSBOARD_MP_PYTHON"]
    print("MediaPipe:", fsv.check_mediapipe(mediapipe_python))
else:
    mediapipe_python = fsv.ensure_mediapipe_env(WORK / "mediapipe_env")

all_clips = [row for split in fsv.SPLITS for row in clips[split]]
report = fsv.run_extraction(all_clips, SEQUENCES, WORK / "videos", mediapipe_python, CODE,
                            detection=DETECTION_CONFIDENCE)
(RUN_DIR / "extraction_report.json").write_text(json.dumps(report, indent=1))
print(f"Done: {report['done']} new, {report['already_done']} already saved earlier, {len(report['failed'])} failed")
for failure in report["failed"][:10]:
    print("   ", failure["clip"], "-", failure["problem"].splitlines()[0][:120])
if report["stopped"]:
    raise fb.FsboardError(f"Extraction stopped after repeated failures.\n{report['stopped']}")
size_mb = sum(p.stat().st_size for p in SEQUENCES.rglob("*.npz")) / 1e6
print(f"Hand points on Drive: {size_mb:.1f} MB")
""")

md(r"""
---
## 4. Turn the points into model inputs, and check them

**In plain English.** Each frame becomes 68 numbers:
* **63 finger numbers:** finger positions relative to the wrist, scaled to the hand's size.
* **Wrist position and speed:** so movement isn't lost.
* **A flag:** "a hand was found here".

Three more things happen:
* **One hand per clip:** a left hand is mirrored to look like a right hand.
* **Same speed:** every clip is converted to 30 frames per second using its real timestamps.
* **Same proportions:** hand shapes are corrected so a portrait phone video and a landscape webcam match.

The scaling numbers (mean and spread) are computed **only from training clips**, so test data can't leak in.

**What success looks like:** a table of examples per group, `failed checks: 0`, and a picture of a hand with **all five
fingers**. Clips with too few detected hand frames are left out, with the reason shown.
""")

code(r"""
FEATURE_CFG = dict(features.DEFAULT_CONFIG, **fsv.FEATURE_SETTINGS)
examples, table = {}, []
for split in fsv.SPLITS:
    examples[split], excluded = fsv.build_examples(clips[split], SEQUENCES, FEATURE_CFG)
    share = [e["hand_share"] for e in examples[split]]
    table.append({"group": split, "chosen": len(clips[split]), "usable": len(examples[split]),
                  "median share of frames with a hand": round(float(np.median(share)), 2) if share else "-",
                  "left out": dict(excluded) or "none"})
display(HTML(fsv.html_table(table)))
if not examples["train"] or not examples["validation"] or not examples["test"]:
    raise fb.FsboardError("A group has no usable clips.\nWhat to check: the extraction report in step 3, and use more clips.")

failed = 0
for split in fsv.SPLITS:
    for e in examples[split]:
        results = fsv.check_recording(fb.load_example(e["path"]))
        bad = [name for name, passed, _ in results if not passed]
        if bad or not np.all(np.isfinite(e["x"])) or e["x"].shape[1] != features.NUM_FEATURES:
            failed += 1
            print("Problem:", e["sequence_id"], bad)
print("failed checks:", failed)
overlap = {e["signer"] for e in examples["train"]} & {e["signer"] for s in ("validation", "test") for e in examples[s]}
print("signers in both training and evaluation:", len(overlap))

normalizer = features.Normalizer().fit([e["x"] for e in examples["train"]])
canonical = []
for e in examples["train"][:200]:
    rec = fb.load_example(e["path"])
    hand = fsv.signing_hand(rec)
    canonical.append(features.canonical_points(rec[f"{hand}_hand"], hand))
PREP = features.preprocessing_dict(normalizer, FEATURE_CFG, chirality_train=round(features.chirality_score(np.concatenate(canonical)), 3),
                                   source="FSboard videos, MediaPipe Hands 0.10.14, mirrored like the webcam app",
                                   vocab_version=vocab.VOCAB_VERSION)
(RUN_DIR / "preprocessing.json").write_text(json.dumps(PREP, indent=1))
MODEL_CFG = dict(smodel.DEFAULT_MODEL_CONFIG)

sample = max(examples["train"], key=lambda e: e["hand_share"])
fb.plot_example(fb.load_example(sample["path"]))
plt.show()
save_prepared()                           # later sessions load this instead of repeating steps 2-4
""")

md(r"""
---
## 5. Optional tiny test: memorise a few clips

**In plain English.** The model trains and is tested on the **same 16 clips**, with no random changes. It should simply
memorise them. This only proves the pipeline works. It says nothing about real accuracy. It is off by default, because
the pipeline has already worked on the full data; tick `RUN_TINY_TEST` after changing the code.

**What success looks like:** the error rate (CER) falls from about 1.0 towards 0, and predicted words start to match.
If it stays near 1.0 or the loss becomes `nan`, don't continue: something in the pipeline is wrong.
""")

code(r"""
RUN_TINY_TEST = False  #@param {type:"boolean"}

if RUN_TINY_TEST:
    need_data()                           # in a new session: loads what step 4 saved
    tiny_dir =RUN_DIR.parent / (RUN_NAME.split("__")[0] + "_tiny")
    shutil.rmtree(tiny_dir, ignore_errors=True)
    tiny = examples["train"][:16]
    tf.keras.utils.set_random_seed(SEED)
    _, tiny_state = strain.fit(smodel.build_tcn_ctc(MODEL_CFG), tiny, tiny, normalizer, tiny_dir, PREP, MODEL_CFG,
                               cfg={"epochs": 300, "batch_size": 4, "augment": False, "learning_rate": 3e-3,
                                    "patience": 300, "show_examples": 3, "log_every": 25, "seed": SEED})
    print("Best error rate on the memorised clips:", round(tiny_state["best_cer"], 3))
else:
    print("Skipped.")
""")

md(r"""
---
## 6. Train the model

**In plain English.** The model studies the training clips again and again. After each round (*epoch*) it is scored on
the validation signers, and you see a few true and predicted texts.

**The settings, and why:**
* `EXPERIMENT = "speed_long_memory"`: each output sees about 4 seconds (125 frames) instead of 2. It did best on
  slowed-down clips in your earlier runs. `"speed"` keeps the 2-second view.
* **Speed range:** every training clip is randomly played between 2× faster and 3.5× slower, so the model meets slow
  signing like yours.
* `WARM_START_FROM`: the run whose model this one starts from, instead of from zero. All its layers are copied, so
  it starts out reading as well as that model and only has to improve on the new clips. An older model without a
  space output works too: it gets one new output for the **space**. Leave it empty to start from zero.
  It must be a `speed_long_memory` run when `EXPERIMENT` is `speed_long_memory` (the layers must match).

Saved every epoch: `checkpoints/last/`, so training continues after a disconnect, and `best_model/`, the version with
the lowest validation error. Each `EXPERIMENT` gets its own run folder.

**What success looks like:** `Warm start from ...: copied ... layers`, then epochs where the
validation error falls, then two plots. A big gap between low training loss and high validation error means it's
memorising rather than learning.
""")

code(r"""
EXPERIMENT = "speed_long_memory"  #@param ["speed_long_memory", "speed"]
WARM_START_FROM = "train8732_val2103_test2131_letters2-100_phrases40__speed_long_memory"  #@param {type:"string"}
EPOCHS = 40        #@param {type:"integer"}
BATCH_SIZE = 16    #@param {type:"integer"}
PATIENCE = 10      #@param {type:"integer"}

need_data()                               # in a new session: loads what step 4 saved
RUN_NAME = RUN_NAME.split("__")[0] + "__" + EXPERIMENT   # its own folder, so nothing resumes or overwrites
RUN_DIR = DRIVE / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "preprocessing.json").write_text(json.dumps(PREP, indent=1))
MODEL_CFG = dict(smodel.DEFAULT_MODEL_CONFIG)
if EXPERIMENT == "speed_long_memory":
    MODEL_CFG["dilations"] = [1, 2, 4, 8, 16]              # sees about 4 seconds instead of 2
SPEED_AUGMENT = dict(strain.sdata.DEFAULT_AUGMENT, stretch=(0.5, 3.5))   # 2x faster to 3.5x slower
print("Run:", RUN_NAME, "| the model sees", smodel.receptive_field(MODEL_CFG), "frames |",
      MODEL_CFG["num_classes"], "outputs (blank, A-Z, space)")

VERSIONS = strain.environment_versions()
TRAIN_CFG = dict(strain.DEFAULT_TRAIN_CONFIG, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, seed=SEED,
                 augment=SPEED_AUGMENT)
tf.keras.utils.set_random_seed(SEED)
model = smodel.build_tcn_ctc(MODEL_CFG)
if WARM_START_FROM:
    source = DRIVE / "runs" / WARM_START_FROM / "best_model"
    if not source.exists():
        raise FileNotFoundError(f"No model to start from at {source}.\n"
                                "What to check: WARM_START_FROM, or leave it empty to start from zero.")
    warm = smodel.warm_start(model, source)
    print(f"Warm start from {WARM_START_FROM}: copied {len(warm['copied'])} layers"
          + (f", extended {warm['extended']} with the space output." if warm["extended"] else "."))
_, state = strain.fit(model, examples["train"], examples["validation"], normalizer, RUN_DIR, PREP, MODEL_CFG,
                      cfg=TRAIN_CFG, versions=VERSIONS)
history = state["history"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
epochs = [h["epoch"] for h in history]
ax[0].plot(epochs, [h["train_loss"] for h in history], label="training")
ax[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
ax[1].plot(epochs, [h["val_cer"] for h in history])
ax[1].set(title="Validation character error rate", xlabel="epoch")
plt.show()
print("Best validation error rate", round(state["best_cer"], 3), "at epoch", state["best_epoch"])
""")

md(r"""
---
## 7. Evaluate, next to your earlier model

**In plain English.** This scores the **best saved model** on signers it never trained on:

| Measure | Meaning |
|---|---|
| **Character error rate (CER)** | wrong letters and spaces per true character; lower is better |
| **Letters-only error rate** | the same with spaces ignored, so it compares fairly with models that have no space |
| **Word error rate** | wrong, missing or extra words per true word |
| **Spaces found / extra spaces** | how many true spaces it wrote in the right place, and how many it invented |
| **Exact accuracy** | texts that are completely right |
| **Double letters** | words like HELLO, and how often LL collapsed to L |
| **Slow, normal, fast** | signing speed, measured as frames per letter |
| **Per letter** | which letters are weak, and what they get confused with |
| **No hand visible** | whether it invents letters when no hand is shown |

`COMPARE_WITH` is your earlier model. Both are scored on the **same test clips**, each with its own scaling numbers.
An earlier model without a space output can only be judged on letters when the clips are phrases.

Look at the **test** numbers, and don't tune settings on them. They describe these clips only.

**What success looks like:** the tables below and a saved `evaluation_report.json`.
""")

code(r"""
COMPARE_WITH = "train8732_val2103_test2131_letters2-100_phrases40__speed_long_memory"  #@param {type:"string"}


def load_for_scoring(folder):
    # A saved model and its own scaling numbers.
    scored, _, scored_prep = smodel.load_inference_bundle(folder)
    return strain.CTCTrainer(scored), features.Normalizer.from_dict(scored_prep["normalizer"])


def all_scores(refs, hyps):
    extra = {k: v for k, v in fsv.phrase_scores(refs, hyps).items()
             if k in ("letters_only_error_rate", "word_error_rate", "spaces_found", "extra_spaces")}
    return dict(fsv.score_texts(refs, hyps), **extra)


need_data(trained=True)                   # in a new session: loads the data and finds the trained model
evaluator, _ = load_for_scoring(RUN_DIR / "best_model")
reports = {}
for split in ("validation", "test"):
    group = examples[split]
    report = seval.evaluate(evaluator, group, normalizer, examples["train"], split,
                            fps=PREP["features"]["target_fps"])
    seval.print_report(report)
    hyps, _ = strain.predict_examples(evaluator, group, normalizer)
    refs = [e["text"] for e in group]
    report["scores"] = all_scores(refs, hyps)
    report["by_letter"] = fsv.letter_report(refs, hyps)
    report["signers_unfamiliar"] = unfamiliar[split]
    reports[split] = report

score_groups = {"one-word test clips": [e for e in examples["test"] if " " not in e["text"]],
                "phrase test clips": [e for e in examples["test"] if " " in e["text"]]}
score_models = {"this run": RUN_DIR / "best_model"}
compare_dir = DRIVE / "runs" / COMPARE_WITH / "best_model"
if COMPARE_WITH and COMPARE_WITH != RUN_NAME and compare_dir.exists():
    score_models["earlier model"] = compare_dir
elif COMPARE_WITH and COMPARE_WITH != RUN_NAME:
    print("No earlier model at", compare_dir, "- comparison skipped.")
comparison, compare_hyps = [], {}
for label, folder in score_models.items():
    scored_eval, scored_norm = load_for_scoring(folder)
    for group_name, items in score_groups.items():
        if items:
            hyps, _ = strain.predict_examples(scored_eval, items, scored_norm)
            compare_hyps[(label, group_name)] = hyps
            comparison.append(dict(model=label, group=group_name, **all_scores([e["text"] for e in items], hyps)))
reports["comparison"] = comparison

print("\nThis run next to the earlier model, on the same test clips (lower error rates are better):")
display(HTML(fsv.html_table(comparison)))
print("\nTest clips, weakest letters first:")
display(HTML(fsv.html_table(reports["test"]["by_letter"]["letters"])))
print("Most common confusions:")
display(HTML(fsv.html_table(reports["test"]["by_letter"]["confusions"])))
for group_name, items in score_groups.items():
    if items:
        print(f"Examples, {group_name}:")
        shown = [{"true": e["text"], "this run": h} for e, h in zip(items, compare_hyps[("this run", group_name)])]
        if ("earlier model", group_name) in compare_hyps:
            for row, h in zip(shown, compare_hyps[("earlier model", group_name)]):
                row["earlier model"] = h
        display(HTML(fsv.html_table(shown[:15])))
seval.save_report(reports, RUN_DIR / "best_model" / "evaluation_report.json")
print("Saved:", RUN_DIR / "best_model" / "evaluation_report.json")
""")

md(r"""
---
## 8. Speed check: the same test clips, faster and slower

**In plain English.** The one-word test clips are played 2× faster, at normal speed, 2× slower and 3× slower (by
repeating or skipping frames), and scored again, for this run and the earlier model. Your own signing is about 3×
slower than FSboard's signers. This is a simulation: real slow signing moves differently, so your webcam is the real test.

**What success looks like:** a table with four rows per model. Flat numbers across the rows mean the model copes with
different speeds.
""")

code(r"""
RUN_SPEED_CHECK = True  #@param {type:"boolean"}


def at_speed(x, slower):
    # Features as if signed `slower` times slower (0.5 = twice as fast).
    n = max(1, round(len(x) * slower))
    y = x[np.minimum((np.arange(n) / slower).astype(int), len(x) - 1)].copy()
    y[:, features.WRIST_VEL] /= slower          # wrist speed changes with signing speed
    return y


need_data(trained=True)                   # in a new session: loads the data and finds the trained model
if "score_models" not in globals():       # step 7 was not run in this session: score this run only
    def load_for_scoring(folder):
        scored, _, scored_prep = smodel.load_inference_bundle(folder)
        return strain.CTCTrainer(scored), features.Normalizer.from_dict(scored_prep["normalizer"])
    score_groups = {"one-word test clips": [e for e in examples["test"] if " " not in e["text"]]}
    score_models = {"this run": RUN_DIR / "best_model"}

if RUN_SPEED_CHECK and score_groups["one-word test clips"]:
    speed_rows = []
    words = score_groups["one-word test clips"]
    for label, folder in score_models.items():
        scored_eval, scored_norm = load_for_scoring(folder)
        for signing, slower in (("2x faster", 0.5), ("normal", 1.0), ("2x slower", 2.0), ("3x slower", 3.0)):
            group = [dict(e, x=at_speed(e["x"], slower)) for e in words]
            hyps, _ = strain.predict_examples(scored_eval, group, scored_norm)
            speed_rows.append(dict(model=label, signing=signing, **fsv.score_texts([e["text"] for e in group], hyps)))
    display(HTML(fsv.html_table(speed_rows)))
    (RUN_DIR / "best_model" / "speed_report.json").write_text(json.dumps(speed_rows, indent=1))
else:
    print("Skipped.")
""")

md(r"""
---
## 9. Export the small model file

**In plain English.** This makes one small zip on your Drive, holding the trained model, its settings and its reports.
Nothing comes to your laptop unless you tick `DOWNLOAD_TO_MY_COMPUTER`.

**On the laptop:** unzip it into a **new** folder, for example `backend\models\phrase_model`, so your other models stay
as they are. The folder must directly contain `model.weights.h5`. Then try it:
`venv\Scripts\python.exe scripts\sequence_model\seq_record_word.py backend\models\phrase_model --mirror-input`

**What success looks like:** a line naming a zip in `MyDrive/asl_fingerspelling/fsboard_tcn_ctc/export/`, under 1 MB.
""")

code(r"""
DOWNLOAD_TO_MY_COMPUTER = False  #@param {type:"boolean"}

need_data(trained=True)                   # in a new session: finds the trained model on Drive
bundle = RUN_DIR / "best_model"
for name in ("train_config.json", "history.json", "selection.json", "extraction_report.json"):
    for folder in (RUN_DIR, RUN_DIR.parent / RUN_NAME.split("__")[0]):
        if (folder / name).exists():
            shutil.copy(folder / name, bundle / name)
            break
(bundle / "README.txt").write_text(
    "TCN + CTC fingerspelling model (blank=0, A-Z, space), trained on FSboard videos with MediaPipe Hands.\n"
    f"Run: {RUN_NAME}. Dataset: FSboard by Georg et al., CC BY 4.0.\n"
    "Inputs use aspect_correction=True (fsboard_video.isotropic_points): live code must apply it too.\n")
(DRIVE / "export").mkdir(exist_ok=True)
export_zip = fb.zip_outputs(bundle, DRIVE / "export" / f"{RUN_NAME}.zip")
print(f"Saved: {export_zip} ({export_zip.stat().st_size / 1e6:.2f} MB)")
if DOWNLOAD_TO_MY_COMPUTER:
    from google.colab import files
    files.download(str(export_zip))
""")

md(r"""
---
## 10. Optional: fine-tune on your own webcam recordings

**Off by default.** Use it only if the model from steps 6–9 reads your own signing poorly. Then:
1. On the laptop, record words and phrases with `seq_record_word.py --words ...` and pack them with
   `seq_pack_my_recordings.py` (see `docs/SEQUENCE_MODEL.md`).
2. Upload `backend/storage/my_recordings_pack.npz` to `MyDrive/asl_fingerspelling/fsboard_tcn_ctc/my_recordings/`.
3. Tick `FINE_TUNE_ON_MY_RECORDINGS` and run this cell. In a new Colab session it only needs Setup and step 1
   first (the data step 4 saved and the model from step 6 are read from Drive).

**What it does.** It continues training a **copy** of the model from step 6 (the model's own folder is never changed),
on all FSboard training clips plus your recordings, which are repeated so they make up about `YOUR_SHARE` of each
epoch. It uses a lower learning rate, so it adjusts gently. The best epoch is chosen on your validation takes; your
test takes, which are never trained on, give the honest score.

**What success looks like:** a before/after table. Keep the new model only if **your test words** improve and the
**FSboard** rows don't clearly get worse. On the laptop, unzip it into a new folder such as `backend\models\personal_phrase_model`.
""")

code(r"""
FINE_TUNE_ON_MY_RECORDINGS = False  #@param {type:"boolean"}
BASE_EXPERIMENT = "speed_long_memory"  #@param ["speed_long_memory", "speed"]
TUNE_EPOCHS = 40                    #@param {type:"integer"}
TUNE_PATIENCE = 10                  #@param {type:"integer"}
TUNE_LEARNING_RATE = 0.0003         #@param {type:"number"}
YOUR_SHARE = 0.5                    #@param {type:"number"}
DOWNLOAD_PERSONAL_MODEL = False     #@param {type:"boolean"}

if not FINE_TUNE_ON_MY_RECORDINGS:
    print("Skipped: tick FINE_TUNE_ON_MY_RECORDINGS to fine-tune on your own recordings.")
else:
    need_data()                           # in a new session: loads what step 4 saved
    BASE_RUN = RUN_NAME.split("__")[0] + "__" + BASE_EXPERIMENT   # the model trained in step 6
    BASE_DIR = DRIVE / "runs" / BASE_RUN
    PERSONAL_RUN = BASE_RUN + "__personal"
    PERSONAL_DIR = DRIVE / "runs" / PERSONAL_RUN
    PACK = DRIVE / "my_recordings" / "my_recordings_pack.npz"
    if not PACK.exists():
        raise FileNotFoundError(f"Upload my_recordings_pack.npz from your laptop to {PACK.parent}")
    if not (BASE_DIR / "best_model").exists():
        raise FileNotFoundError(f"No trained model at {BASE_DIR / 'best_model'}: run step 6 first, "
                                "or check BASE_EXPERIMENT")

    # 1. The model you start from, and the normaliser it was trained with (kept unchanged,
    #    so the laptop tools work with the new model exactly as before).
    base_model, base_cfg, base_prep = smodel.load_inference_bundle(BASE_DIR / "best_model")
    base_norm = features.Normalizer.from_dict(base_prep["normalizer"])

    # 2. Your recordings, packed on the laptop by seq_pack_my_recordings.py.
    with np.load(PACK, allow_pickle=False) as z:
        pack_cfg = json.loads(str(z["feature_cfg"]))
        ends = np.cumsum(z["lengths"])
        mine = [{"x": z["x"][e - n:e], "y": np.array(vocab.encode(str(w)), np.int32), "text": str(w),
                 "pace": str(p), "split": str(g), "sequence_id": str(f), "signer": "me"}
                for e, n, w, p, g, f in zip(ends, z["lengths"], z["words"], z["paces"], z["splits"], z["files"])]
    if any(" " in e["text"] for e in mine) and not smodel.has_space_output(base_cfg):
        raise ValueError("Your recordings include phrases with spaces, but this model has no space output. "
                         "Train with phrases in steps 2 and 6, or pack without the phrase lists.")
    if json.dumps(pack_cfg, sort_keys=True) != json.dumps(base_prep["features"], sort_keys=True):
        raise ValueError("Your recordings were packed with different feature settings than this model uses. "
                         "On the laptop, pack again with --bundle pointing at this model's folder.")
    mine_by = {s: [e for e in mine if e["split"] == s] for s in ("train", "validation", "test")}
    display(HTML(fsv.html_table([{"group": s, "takes": len(g), "different words": len({e["text"] for e in g})}
                                 for s, g in mine_by.items()])))
    if not mine_by["train"] or not mine_by["validation"]:
        raise ValueError("Fine-tuning needs your takes in both 'train' and 'validation'. Record more training words.")

    # 3. Training data: all FSboard training clips plus yours, repeated so they are about
    #    YOUR_SHARE of what the model sees. Keeping FSboard in stops it forgetting other signers.
    repeat = max(1, min(10, round(YOUR_SHARE / (1 - YOUR_SHARE) * len(examples["train"]) / len(mine_by["train"]))))
    train_set = examples["train"] + mine_by["train"] * repeat
    if max(int(np.max(e["y"])) for e in train_set) >= base_cfg["num_classes"]:
        raise ValueError("The training clips include spaces, but this model has no space output: check step 6.")
    print(f"Training on {len(examples['train'])} FSboard clips + {len(mine_by['train'])} of your takes x {repeat} "
          f"(yours are {len(mine_by['train']) * repeat / len(train_set):.0%} of each epoch).")
    print(f"The best epoch is chosen on your {len(mine_by['validation'])} validation takes; "
          f"your {len(mine_by['test'])} test takes are only used for the final score.")

    # 4. Continue training a copy of the base model, gently (lower learning rate), with the same
    #    random changes (augmentation) the base model was trained with. Safe to re-run after a
    #    disconnect: it resumes from PERSONAL_DIR/checkpoints/last.
    base_train_cfg = BASE_DIR / "train_config.json"
    augment = json.loads(base_train_cfg.read_text()).get("augment", True) if base_train_cfg.exists() else True
    personal_model = smodel.build_tcn_ctc(base_cfg)
    personal_model.set_weights(base_model.get_weights())
    PERSONAL_PREP = dict(base_prep, base_model=BASE_RUN,
                         fine_tuned_on="FSboard training clips + your own recordings")
    TUNE_CFG = dict(strain.DEFAULT_TRAIN_CONFIG, epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
                    learning_rate=TUNE_LEARNING_RATE, augment=augment, seed=42)
    strain.fit(personal_model, train_set, mine_by["validation"], base_norm, PERSONAL_DIR, PERSONAL_PREP, base_cfg,
               cfg=TUNE_CFG, versions=strain.environment_versions())

    # 5. Before vs after, on the same takes. Your test words are the honest score: never trained on.
    after_model, _, _ = smodel.load_inference_bundle(PERSONAL_DIR / "best_model")
    test_words = [e for e in mine_by["test"] if " " not in e["text"]]
    groups = {"your test words": test_words}
    for pace in sorted({e["pace"] for e in test_words}):
        groups[f"your test words, {pace}"] = [e for e in test_words if e["pace"] == pace]
    groups["your test phrases"] = [e for e in mine_by["test"] if " " in e["text"]]
    groups["your validation takes"] = mine_by["validation"]
    groups["FSboard one-word test clips"] = [e for e in examples["test"] if " " not in e["text"]]
    groups["FSboard phrase test clips"] = [e for e in examples["test"] if " " in e["text"]]
    personal_rows, predictions = [], {}
    for label, m in (("before (base model)", base_model), ("after (personal)", after_model)):
        before_after = strain.CTCTrainer(m)
        for group, items in groups.items():
            if not items:
                continue
            hyps, _ = strain.predict_examples(before_after, items, base_norm)
            predictions[(label, group)] = hyps
            refs = [e["text"] for e in items]
            extra = {k: v for k, v in fsv.phrase_scores(refs, hyps).items()
                     if k in ("letters_only_error_rate", "word_error_rate", "spaces_found", "extra_spaces")}
            personal_rows.append(dict(model=label, group=group, **fsv.score_texts(refs, hyps), **extra))
    display(HTML(fsv.html_table(personal_rows)))
    for group in ("your test words", "your test phrases"):
        if groups[group]:
            print(f"{group.capitalize()} (first 20):")
            display(HTML(fsv.html_table([
                {"true": e["text"], "pace": e["pace"], "before": b, "after": a}
                for e, b, a in zip(groups[group], predictions[("before (base model)", group)],
                                   predictions[("after (personal)", group)])][:20])))

    # 6. Save the report and a small zip of the new model for the laptop.
    personal_bundle = PERSONAL_DIR / "best_model"
    for name in ("train_config.json", "history.json"):
        if (PERSONAL_DIR / name).exists():
            shutil.copy(PERSONAL_DIR / name, personal_bundle / name)
    (personal_bundle / "personal_report.json").write_text(json.dumps(personal_rows, indent=1))
    (personal_bundle / "README.txt").write_text(
        f"TCN + CTC fingerspelling model: {BASE_RUN} fine-tuned on FSboard + your own recordings.\n"
        "Inputs use aspect_correction=True (fsboard_video.isotropic_points): live code must apply it too.\n"
        "Contains no recordings, only model weights and settings.\n")
    (DRIVE / "export").mkdir(exist_ok=True)
    personal_zip = fb.zip_outputs(personal_bundle, DRIVE / "export" / f"{PERSONAL_RUN}.zip")
    print(f"Saved: {personal_zip} ({personal_zip.stat().st_size / 1e6:.2f} MB). "
          "On the laptop, unzip it into a new folder in backend\\models")
    if DOWNLOAD_PERSONAL_MODEL:
        from google.colab import files
        files.download(str(personal_zip))
""")

md(r"""
---
## Where this leaves you
You have a model that reads letters and spaces, reports on unfamiliar signers next to your earlier model, a speed check,
and a small zip, all on Google Drive. Your webcam is the final test: try it on words and phrases you actually sign.
""")


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "name": OUT.name},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
