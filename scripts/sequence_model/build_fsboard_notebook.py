"""
build_fsboard_notebook.py - Writes notebooks/fsboard_prepare_data_colab.ipynb.

The notebook carries an exact copy of the helper code (asl/seq/fsboard.py and
what it needs), so uploading that one file to Colab is enough.
tests/test_seq_fsboard_notebook.py fails if the copy and the repository differ.
After changing that code, run this again:

    venv\\Scripts\\python.exe scripts\\sequence_model\\build_fsboard_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fsboard_prepare_data_colab.ipynb"
EMBEDDED = ("asl/__init__.py", "asl/seq/__init__.py", "asl/seq/vocab.py", "asl/seq/fsboard.py")


def setup_cell():
    blocks = []
    for name in EMBEDDED:
        text = (ROOT / "backend" / name).read_text(encoding="utf-8")
        if "'''" in text or text.endswith("\\"):
            raise ValueError(f"{name} cannot be embedded in a raw triple-quoted string")
        blocks.append(f'    "{name}": r\'\'\'{text}\'\'\',\n')
    return (
        '#@title Setup: run this first (the code stays hidden; click "Show code" to read it)\n'
        "# Writes this project's helper code to the temporary Colab disk. These are exact\n"
        "# copies of the files named below, from backend/ in the repository.\n"
        "import importlib, os, shutil, sys\n"
        "from pathlib import Path\n\n"
        'WORK = Path(os.environ.get("FSBOARD_WORK", "/content/fsboard"))   # temporary Colab disk\n'
        'CODE = WORK / "code"\n'
        'OUT = WORK / "fsboard_pilot"                                      # exported sequences\n\n'
        "_FILES = {\n" + "".join(blocks) + "}\n"
        "for _name, _text in _FILES.items():\n"
        "    _path = CODE / _name\n"
        "    _path.parent.mkdir(parents=True, exist_ok=True)\n"
        '    _path.write_text(_text, encoding="utf-8")\n'
        "if str(CODE) not in sys.path:\n"
        "    sys.path.insert(0, str(CODE))\n\n"
        "import asl.seq.vocab\n"
        "import asl.seq.fsboard as fb\n"
        "importlib.reload(asl.seq.vocab)\n"
        "importlib.reload(fb)\n\n"
        "import pandas as pd\n"
        "from IPython.display import HTML, display\n"
        'pd.set_option("display.max_colwidth", None)\n\n'
        "SKIP_SENSITIVE = True       # leave out clips the publisher tagged sensitiveContent\n"
        "RUN_LANDMARK_PREP = False   # section 3 is off until you switch it on there\n\n"
        "try:\n"
        "    _found = fb.kaggle_auth_from_colab_secrets()\n"
        "except ImportError:\n"
        "    _found = []\n"
        'print("Setup finished. Temporary working folder:", WORK)\n'
        'print("Kaggle key from Colab Secrets:", ", ".join(_found) if _found else\n'
        '      "none set (fine: the files downloaded without one when this was tested)")\n'
    )


CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text, form=False):
    meta = {"cellView": "form"} if form else {}
    CELLS.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Prepare FSboard data for continuous ASL fingerspelling

**Project goal:** turn one complete recording of someone fingerspelling into the letters they spelled, in order.

**This notebook's goal:** get real data ready and prove it is usable. It does not train a model; that is
the next milestone. Your webcam app and single-letter model are not touched.

### How to use it
1. Upload this one file at [colab.research.google.com](https://colab.research.google.com) (**File → Upload notebook**).
2. Choose **Runtime → Run all**. This is safe. It downloads about **57 MB** in total to Colab: the 52 MB
   metadata file and one 5 MB video. Section 3, the 643 MB download, stays **off** until you switch it on.

A CPU runtime is enough; no GPU is needed. No Kaggle account was needed when the download links were tested.
If Kaggle refuses, the error message says exactly what to do.

### Where files go
| Place | What | How long it lasts |
|---|---|---|
| **Temporary Colab disk** `/content/fsboard/` | metadata, preview video, landmark shard, exported sequences | **erased when the Colab session ends** |
| **Your computer** (optional, section 3e) | `fsboard_pilot.zip`, tens of KB | until you delete it |
| **Google Drive** (optional, section 3e) | a copy of the same zip | until you delete it |

### Two things found in the real data
These were measured on the real files when this notebook was written.

1. **Few phrases qualify.** Only 436 of the 70,699 training clips (0.6%) have a phrase of 2–12 letters and
   nothing else. Most phrases contain digits, symbols or spaces. One landmark shard holds only a handful.
2. **The landmark files hold only part of each hand.** Section 3 explains this. Read it before you download.
""")

code(setup_cell(), form=True)

md("""
**What success looks like:** `Setup finished.` and a line about the Kaggle key.
""")

md("""
---
## 1. Which videos, and what is each person spelling?

**In plain English.** Every recording has an entry in a metadata file: its filename, the phrase that was
spelled, and who signed it. We download only the training group's metadata file, then look at the fields
it really has before using them.

A recording is used only if its **whole phrase is 2–12 English letters and nothing else**. The training
answer must be exactly what the hands spell. Take the real clip labelled `2067 pso costa`: it is fine for
checking that playback works, but if we deleted the digits and spaces the answer would become `PSOCOSTA`
while the hands still spell `2067` and the gaps. So the **whole recording is skipped**, never edited.
Repeated letters stay: `letter` becomes `LETTER`, with both Ts.

The cell picks up to five recordings from **different signers**, always the same five.

**What success looks like:** a list of 17 fields that includes `clipFilename`, `phrase`, `signerId` and
`clipFileMd5`, then `70,699 clips` and `436 usable, from 85 different signers`, then this table:

| row | Filename | Original phrase | Training answer | Signer ID |
|---|---|---|---|---|
| 0 | `03ad_0_20220813_131811_629544_633267.mp4` | coquetteaabc | COQUETTEAABC | 03ad |
| 1 | `0ba8_954_20220916_164150_58611_61433.mp4` | rnom | RNOM | 0ba8 |
| 2 | `0bea_541_20220831_120130_19878_24156.mp4` | haikyuufess | HAIKYUUFESS | 0bea |
| 3 | `0d7b_324_20221004_231934_118750_121267.mp4` | moneros | MONEROS | 0d7b |
| 4 | `110f_263_20220807_154205_770144_773367.mp4` | scarlettvkm | SCARLETTVKM | 110f |

The dataset's phrases are synthetic, which is why many are not everyday words.
""")

code("""
# Download only the training metadata file, then read it.
metadata_file = fb.download(fb.TRAIN_METADATA, WORK / "daun_v3-train.json",
                            expected_size=fb.remote_size(fb.TRAIN_METADATA))
records = fb.load_metadata(metadata_file)

print("The fields this file really has:")
display(pd.DataFrame(fb.describe_fields(records), columns=["field", "clips with it", "type", "example"]))

eligible, skipped = fb.eligible_rows(records, skip_sensitive=SKIP_SENSITIVE)
print(f"{len(records):,} clips in the training metadata")
print(f"{len(eligible):,} usable, from {len({r['signer_id'] for r in eligible})} different signers")
print("Skipped whole, and why:")
for reason, count in skipped.most_common():
    print(f"   {count:7,}  {reason}")

for r in records:
    if r["phrase"] == "2067 pso costa":
        print(f"\\nExample: '{r['phrase']}' ({r['clipFilename']}) is skipped: {fb.phrase_problem(r['phrase'])}.")
        break

preview_rows = fb.select_preview(eligible, n=5)
table = pd.DataFrame({"Filename": [r["filename"] for r in preview_rows],
                      "Original phrase": [r["phrase"] for r in preview_rows],
                      "Training answer": [r["answer"] for r in preview_rows],
                      "Signer ID": [r["signer_id"] for r in preview_rows]})
table.index.name = "row"
print("\\nThe recordings selected for preview:")
display(table)
""")

md("""
---
## 2. Play one selected video

**In plain English.** This downloads one video, the row you choose, and nothing else. It then compares the
file's fingerprint (its **MD5**) with `clipFileMd5` in the metadata, which proves every byte arrived intact.
Then it shows the written answer directly above the video.

**To preview another example:** change `PREVIEW_ROW` to a row number from the table and run this cell again.

**What success looks like:** `MD5 check passed`, then **COQUETTEAABC** in large letters above an upright video
about 4 seconds long. The video starts a little before the signing and ends a little after it.

**Please keep the video private.** FSboard's responsible-use rules ask you to blur signers' faces when
publicizing examples. The video is stored in this notebook's output, so use **Edit → Clear all outputs**
before sharing the notebook or putting it on GitHub.
""")

code("""
PREVIEW_ROW = 0  #@param {type:"integer"}

try:
    chosen = fb.pick_row(preview_rows, PREVIEW_ROW)
    print(f"Row {PREVIEW_ROW}: {chosen['filename']} ({chosen['size_bytes'] / 1e6:.1f} MB)")
    clip = fb.download(fb.clip_path(chosen), WORK / "clips" / chosen["filename"],
                       chosen["size_bytes"], chosen["md5"])
    print(f"MD5 check passed: {fb.file_md5(clip)} matches clipFileMd5")
    display(HTML(fb.video_html(clip, chosen["answer"])))
except fb.FsboardError as problem:
    print(f"The video could not be shown.\\n{problem}")
""")

md("""
---
## 3. OPTIONAL: prepare real training inputs from the existing landmarks

**This section is off, even with "Run all".** Nothing here downloads until you tick `RUN_LANDMARK_PREP`
in the next cell. Then run cells 3a to 3e, and section 4.

**In plain English.** *Landmarks* are numbers giving the position of points on the hand in each video frame.
Google already computed them with a tool called MediaPipe, so we don't have to process video. They come in
big files called *shards*. Shard 00000 holds 739 recordings and is **about 643 MB**. The five preview videos
may live in other shards, so we select eligible examples from the shard actually downloaded.

### Read this before you download: only part of each hand is stored
The publisher's script `convert_arrow_to_parquet.py` was the starting point for reading these files. Its column
names suggest 21 points per hand, each with an x, y and z. **The real numbers are different.** In every one
of the 5,447 hand frames checked across train, validation and test shards:

```
"x" list = x0 y0 z0 x1 y1 z1 ... x6 y6 z6
"y" list =    y0 z0 x1 y1 z1 ... x6 y6 z6 x7
"z" list =       z0 x1 y1 z1 ... x6 y6 z6 x7 y7
```

So there are only 23 different numbers per hand per frame:

| Stored | Not stored |
|---|---|
| wrist, 4 thumb joints, 3 index-finger joints (the z of the last one is missing) | index fingertip, middle, ring and little fingers |

This reading was confirmed against MediaPipe run directly on a real video (CELEBSTYLE): all 65 stored frames
matched within 2% of the image, typically 0.1%. Reading the lists the way their names suggest was off by about 28%.

**Tracking quality also varies by clip.** On the other two eligible clips in shard 00000, where the hand is very
close to the camera, the stored hand was found in only 16% and 38% of frames. The stored points differed from
direct MediaPipe by about 6-8% of the image, and in the 12 frames inspected by eye they often did not sit on the hand.

**What this means for you.** This code decodes the numbers correctly and labels the files
`partial-8-landmarks`. Those files are correctly paired and readable, so they are good for building and testing
the data pipeline. But many letters differ mainly by the fingers that are missing, so expect them to limit
accuracy. Your webcam code would have to use the same 8 points. If Google corrects the dataset, the same code
reads complete 21-point hands automatically.
""")

code("""
RUN_LANDMARK_PREP = False  #@param {type:"boolean"}
SHARD_NUMBER = 0           #@param {type:"integer"}
TARGET_SEQUENCES = 20      #@param {type:"integer"}

shard_on_kaggle = fb.train_shard_path(SHARD_NUMBER)
shard_file = WORK / "shards" / Path(shard_on_kaggle).name
if RUN_LANDMARK_PREP:
    print(f"Section 3 is ON. It will download {shard_on_kaggle}")
else:
    print("Section 3 is OFF. Nothing will be downloaded.")
    print(f"If you switch it on, it downloads {shard_on_kaggle} (about 0.6 GB; shard 00000 is 643 MB).")
""")

md("""
### 3a. Download one shard
**What success looks like:** the path and `Approximate size: 643 MB` appear **before** the download starts,
then progress lines, then `Download complete`. It usually takes a few minutes. If it is interrupted, run
the cell again and it continues.
""")

code("""
if not RUN_LANDMARK_PREP:
    print("Skipped: section 3 is off.")
else:
    size = fb.remote_size(shard_on_kaggle)
    print(f"About to download: {shard_on_kaggle}")
    print(f"Approximate size: {size / 1e6:,.0f} MB, to temporary Colab disk ({fb.free_gb(WORK):,.0f} GB free)")
    if fb.free_gb(WORK) < 1.5 * size / 1e9:
        raise fb.FsboardError("There is not enough temporary disk for this shard.\\nWhat to check: delete "
                              "earlier shards in the Files panel, or restart the runtime.")
    fb.download(shard_on_kaggle, shard_file, expected_size=size)
    print(f"Download complete: {shard_file} ({shard_file.stat().st_size / 1e6:,.0f} MB, size verified)")
""")

md("""
### 3b. Look inside the real file
Instead of assuming the format, this opens the shard and prints one record. Compare the three printed lists:
each one is the previous list moved along by one place.

**What success looks like:** `Column: serialized`, `739 records`, the clip's own `prompt`, `signer_id` and
`video_name`, and `layout found from these numbers: partial-8-landmarks`.
""")

code("""
if not RUN_LANDMARK_PREP:
    print("Skipped: section 3 is off.")
else:
    reader = fb.open_shard(shard_file)
    print("Column:", ", ".join(reader.schema.names), f"| {reader.num_record_batches} records")
    print(fb.format_preview(fb.parse_example(next(fb.iter_serialized(reader)))))
""")

md("""
### 3c. Pair, select and export
**In plain English.** For each record, in shard order:

1. **Pairing.** The record carries its own `video_name`, `prompt` (the phrase) and `signer_id`, so hands and
   answer are already together. We double-check against the metadata by an explicit name match. The record says
   `..._165.367_167.833.mp4` in seconds where the metadata says `..._165367_167833.mp4` in milliseconds. The
   phrase and signer must then agree too. **Row position is never used.**
2. **Selection.** The phrase must pass the same 2–12 letter rule, and the clip must have enough frames with a
   detected hand for its letters.
3. **Export.** One small `.npz` file per recording, plus `index.csv`. Left and right hands are kept separate.
   Frame order, timestamps, and a mask of missing detections are all preserved. Reading stops as soon as the target is reached.

**What success looks like for shard 00000** (these numbers come from scanning that real shard):
* `Pairing check: 739 of 739 records matched`
* **3 exported:** SURYADIESEL, STUDENTWORK and CELEBSTYLE
* Skipped: 487 with a digit, 175 with a symbol, 72 with a space, 2 with too many letters
* A message saying **fewer than 20 were found**, and why. **No other shard is downloaded for you.** To add one,
  change `SHARD_NUMBER`, then run 3a to 3e again. Earlier exports are kept, and no recording is exported twice.

Validation and test data stay untouched: the export refuses any shard that is not in the official training split.
""")

code("""
if not RUN_LANDMARK_PREP:
    print("Skipped: section 3 is off.")
else:
    report = fb.export_from_shard(shard_file, records, OUT, target=TARGET_SEQUENCES,
                                  shard_name=shard_on_kaggle, skip_sensitive=SKIP_SENSITIVE)
    print(fb.report_text(report, len(eligible)))
    exported = pd.DataFrame(fb.read_index(OUT))
    if len(exported):
        exported["% frames with a hand"] = (100 * exported["frames_any_hand"].astype(float)
                                            / exported["num_frames"].astype(float)).round(0)
        print()
        display(exported[["file", "answer", "signer_id", "split", "num_frames", "frames_left_hand",
                          "frames_right_hand", "% frames with a hand", "landmarks_per_hand"]])
    if fb.LAYOUT_PARTIAL in report["layouts"]:
        print("\\n" + fb.PARTIAL_WARNING)
    print("\\nThe file format and coordinate order (also saved as README.txt beside the files):\\n")
    print(fb.FORMAT_NOTES)
""")

md("""
### 3d. Where the exported files are
Open the **Files** panel (the folder icon on the left) and go to `/content/fsboard/fsboard_pilot/`:

```
fsboard_pilot/
  index.csv            one row per recording
  README.txt           the file format and coordinate order
  export_report.json   counts and skip reasons
  sequences/*.npz      one small file per recording, about 8-15 KB each
```

### 3e. Save the output before the session ends
Colab erases this disk when the session ends. This cell makes one small zip, about 30 KB for 3 recordings.
Your browser may ask permission to download. **What success looks like:** a `fsboard_pilot.zip` download.
On your laptop, unzip it into `data/fsboard_pilot/`; git ignores that folder.
""")

code("""
SAVE_TO_MY_COMPUTER = True    #@param {type:"boolean"}
COPY_TO_GOOGLE_DRIVE = False  #@param {type:"boolean"}

if not RUN_LANDMARK_PREP:
    print("Skipped: section 3 is off.")
elif not fb.read_index(OUT):
    print("Nothing has been exported yet, so there is nothing to save. Run 3a to 3c first.")
else:
    zip_file = fb.zip_outputs(OUT, WORK / "fsboard_pilot.zip")
    print(f"Created {zip_file} ({zip_file.stat().st_size / 1e3:.0f} KB)")
    if COPY_TO_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        target = Path("/content/drive/MyDrive/asl_fingerspelling/fsboard_pilot.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(zip_file, target)
        print("Copied to Google Drive:", target)
    if SAVE_TO_MY_COMPUTER:
        from google.colab import files
        files.download(str(zip_file))
""")

md("""
---
## 4. Prove the prepared data is usable

**In plain English.** This reloads an exported file from disk, as your future training code will, and tests
it. It runs only if section 3 has produced files. Change `CHECK_ROW` to look at another recording.

**What success looks like for row 0 of shard 00000:** clip `eac7_766_20221004_154135_580433_585578.mp4`,
transcript `SURYADIESEL`, `153` frames, coordinate shape `(153, 8, 3)`, `24` frames with a detected hand,
every check `PASS`, and a picture of the hand moving. Row 2 (CELEBSTYLE) has the best hand coverage,
65 of 74 frames, so its movement is the easiest to see.

Gaps in the picture are frames where no hand was detected. A small number of hand frames, as in row 0, is a
real weakness of that recording.
""")

code("""
import matplotlib.pyplot as plt

CHECK_ROW = 0  #@param {type:"integer"}

index_rows = fb.read_index(OUT)
if not index_rows:
    print("No exported files yet, so there is nothing to check. That is expected while section 3 is off.")
elif not 0 <= CHECK_ROW < len(index_rows):
    print(f"CHECK_ROW is {CHECK_ROW}, but there are rows 0 to {len(index_rows) - 1}. Change it and run again.")
else:
    example = fb.load_example(OUT / index_rows[CHECK_ROW]["file"])
    print("Clip ID:                  ", example["clip_id"])
    print("Signer ID:                ", example["signer_id"])
    print("Official split:           ", example["split"])
    print("Transcript:               ", example["answer"], f"(original phrase: {example['phrase']})")
    print("Frame count:              ", len(example["frame_index"]))
    print("Coordinate shape:         ", example["left_hand"].shape, "(frames, hand points, x/y/z), for each hand")
    print("Frames with a detected hand:", fb.hand_frame_count(example),
          f"(left {int(example['left_mask'].sum())}, right {int(example['right_mask'].sum())})")
    print("\\nChecks on this file:")
    results = fb.check_example(example)
    for name, passed, detail in results:
        print(f"   {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    every = [fb.check_example(fb.load_example(OUT / row["file"])) for row in index_rows]
    good = sum(all(passed for _, passed, _ in checks) for checks in every)
    print(f"\\nAll exported files: {good} of {len(every)} pass every check.")

    fb.plot_example(example)
    plt.show()

    print("\\nWhat goes into the future model, and what comes out:")
    print(fb.model_sentence(example))
    if example["layout"] == fb.LAYOUT_PARTIAL:
        print("\\n" + fb.PARTIAL_WARNING)
""")

md("""
---
## Where this leaves you
This milestone ends here, with real recordings correctly paired with their full written answers, in small files
you can save and reload. FSboard's validation and test groups were not used; they stay separate for later
evaluation. Training the TCN-CTC model is the next milestone.

**Decide before training:** the existing landmarks hold only 8 of 21 hand points in each frame, and their tracking
quality varies. To train on complete hands, the 21 points would have to be extracted from the FSboard videos with
MediaPipe, the same tool the webcam app uses. See `docs/SEQUENCE_MODEL.md` in the project.

**Credit:** FSboard by Georg et al., CC BY 4.0 ([paper](https://arxiv.org/abs/2407.15806),
[dataset](https://www.kaggle.com/datasets/googleai/fsboard)).
""")


def main():
    nb = {"cells": CELLS,
          "metadata": {"colab": {"provenance": [], "name": OUT.name},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
