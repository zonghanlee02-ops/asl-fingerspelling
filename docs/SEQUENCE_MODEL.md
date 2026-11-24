# Sequence model: causal TCN + CTC

**Status (23 Nov 2026):** trained in Colab on FSboard and the author's own takes; `final_model_i` is the
model the browser app reads with. The old one-letter baseline model was removed on 23 Nov 2026.
The sections below are in the order the work was done, so the early ones describe the prototype stage.

## The idea in plain words

The baseline looks at **one frame at a time** and asks "which letter is this
handshape?". Hand-written rules then wait until a letter is held still.
That is why fast signing loses letters.

The sequence model looks at **a stream of frames** and outputs the letters
**in order**, including the movement between letters (and the motion
letters J and Z). It is trained only with the whole word as the answer
("HELLO"). Nobody tells it which frame shows which letter. **CTC**
(Connectionist Temporal Classification) is the training method that makes
this possible:

* The model gives, for every frame, a score for 27 symbols:
  **blank** (index 0, "no new letter here") and **A–Z** (1–26).
  Blank is *not* a space.
* Decoding: take the best symbol per frame, merge identical neighbours,
  drop blanks. `L L _ L` → **LL**; `L L L` → **L**. A double letter
  therefore needs a blank in between, so a word needs at least
  *letters + doubled pairs* frames (`vocab.required_frames`).

**TCN** (temporal convolution network) means small filters that slide along
time. Every filter is **causal**: it sees the current frame and past frames,
never future ones. That is what allows live use later.

```
68 features per frame
 → Dense(64) + LayerNorm + ReLU                      (per-frame projection)
 → 4 residual blocks, dilation 1, 2, 4, 8,
   each: [causal Conv1D(64, kernel 3) → LayerNorm → ReLU → Dropout 0.1] × 2
 → Dense(27)                                         (raw scores per frame)
```
106,139 parameters. Stride 1, so there is one output per input frame.
Each output sees **61 frames** (about 2 s if the data is 30 fps).
LayerNormalization works per frame, so it never mixes time steps.

These are **first experimental settings**, not tuned values.

## Features (68 per frame), `backend/asl/seq/features.py`

| Index | Meaning |
|---|---|
| 0–62 | Finger coordinates: exactly the baseline's `asl.features.normalise_points` (wrist = origin, divided by hand size). 0–2 is the wrist itself and is always 0. |
| 63, 64 | Wrist position x, y in image coordinates minus 0.5 |
| 65, 66 | Wrist velocity in **hand sizes per frame** (independent of distance to the camera) |
| 67 | Validity flag: 1 = hand tracked, 0 = not |

Why not the 63 baseline numbers alone? They put the wrist at the origin
in every frame, so the hand's movement is erased.

Rules, all **causal**:
* **One signing hand per sequence.** A left hand is mirrored (`x → 1 − x`) so
  everything looks like a right hand. In the dataset, the signing hand is the
  hand tracked in more frames of that recording. That is equivalent to knowing
  the signer's dominant hand, which the laptop gets from `--hand`.
* **Missing or invalid detection** (no hand, any NaN/inf, or a degenerate tiny
  hand): all 68 numbers are 0, including the flag.
* **Velocity across gaps:** measured against the last valid frame and divided
  by the number of frames in between, if that gap is ≤ 3 frames; otherwise 0.
  The first valid frame has velocity 0. Clipped to ±1 hand size per frame.
* **Normaliser:** mean/std per feature, from valid frames of the **training
  split only**. Invalid frames stay 0; the flag is not scaled.
* **Frame rate:** the model counts frames. Live camera frames (about 15 fps in
  the app) are resampled to `target_fps` by `CausalResampler`. It interpolates
  between the previous and current frame, adding at most one camera frame of delay.

The same `SequenceFeaturizer.step()` runs in Colab and on the laptop.

## Data: Google ASL Fingerspelling (Kaggle competition `asl-fingerspelling`)

* Continuous fingerspelled phrases with their full text, as MediaPipe
  Holistic landmarks in Parquet files. One file holds many recordings.
  This is **not** the isolated-sign dataset (`asl-signs`).
* Public descriptions say most phrases are addresses, phone numbers, URLs
  and names. The notebook counts how many are plain A–Z words; that count
  has not been checked yet.
* **Compatible transcript:** after `strip().upper()`, only `A–Z` (one word).
  Anything with a space, digit or symbol is **excluded whole**, never edited.
* **Pilot subset:** rank Parquet files by compatible recordings per GB, take
  them until `DOWNLOAD_BUDGET_GB`, then sample up to `MAX_SEQUENCES` with a
  fixed seed. Each file is downloaded to temporary Colab disk, reduced to
  the selected recordings' hand columns, saved as a compact `.npz` shard on
  Drive, and deleted.
* **Kept per recording:** sequence_id, file path, participant_id (signer),
  phrase, normalised transcript, frame numbers (gaps filled with invalid
  frames), which hand was used, valid-frame counts.
* **Split:** whole signers go to train / val / test (70/15/15 of signers).
  Augmented versions are made on the fly for training batches only, so
  they never leave the training split.
* **Not in the dataset:** timestamps / frame rate, session or device IDs,
  image size. These gaps are recorded in `dataset_summary.json`.
* **Excluded, not truncated:** recordings shorter than their word needs
  under CTC, longer than `MAX_FRAMES`, or with no hand at all.
* Manifests (`selection_manifest.csv`, `dataset_summary.json`) are saved and
  reused, so a run can be repeated exactly.

Licence and terms: you must accept the competition rules on Kaggle. Check
the data licence on the competition's Data tab before publishing anything
derived from it (including a trained model).

## FSboard data: `notebooks/fsboard_prepare_data_colab.ipynb`

[FSboard](https://www.kaggle.com/datasets/googleai/fsboard) (CC BY 4.0) is the fuller release of
the competition data above: videos plus official train / validation / test splits. The notebook
selects recordings, plays one, and optionally exports training sequences. Code: `backend/asl/seq/fsboard.py`.
The notebook embeds an exact copy of that code; after changing it, run
`scripts/sequence_model/build_fsboard_notebook.py` (a test fails if they differ).

### What the real files contain (measured November 2026)

* **Training metadata:** 70,699 clips, 94 signers. Only **439 phrases are 2–12 letters and nothing else**
  (436 after leaving out 3 tagged `sensitiveContent`). The rest contain digits (44,860), symbols (16,933),
  spaces (8,174) or too many letters (293).
* **Shard 00000:** 643 MB, 739 records, one per Arrow record batch, ZSTD-compressed.
  **3 are eligible:** SURYADIESEL, STUDENTWORK, CELEBSTYLE.
* **Names:** a record's `video_name` is in seconds (`..._619.011_625.644.mp4`), metadata's `clipFilename` in
  milliseconds (`..._619011_625644.mp4`). A direct match finds 0 of 739; the conversion rule finds 739 of 739, and
  phrase and signer then agree on all 739. Records carry their own `prompt` and `signer_id`.
* **Timing:** `timestamp` is in microseconds per frame and does not always start at 0. Frame rate is
  per clip: in shard 00000, 710 clips round to 30 fps, 25 to 24 fps, 4 others. So resample by timestamps; the
  `target_fps = 30` above is a good default but not universal.
* **Image size:** 1836×3264 (632 clips), 1080×1920 (76), 1512×2688 (31), all portrait. Many MP4s are stored
  sideways with a rotation flag. Browsers honour it; OpenCV 4.11 reads them sideways, so rotate 90° counter-clockwise.
* **Frames:** a record covers the annotated signing only. The MP4 is longer (for CELEBSTYLE, 74 record frames
  against 121 video frames, with record frame 0 at video frame 16).

### Only 8 of 21 hand points are stored

The lists named `x_right_hand`, `y_right_hand`, `z_right_hand` each hold 21 numbers, but they are overlapping
windows of one flattened list, not 21 x, y and z values:

```
x list = x0 y0 z0 x1 y1 z1 ... x6 y6 z6
y list =    y0 z0 x1 y1 z1 ... x6 y6 z6 x7
z list =       z0 x1 y1 z1 ... x6 y6 z6 x7 y7
```

That is 23 distinct numbers: the wrist, four thumb joints, two index joints, and x and y of a third. The index
fingertip and the middle, ring and little fingers are absent. The publisher's `convert_arrow_to_parquet.py` copies
these lists into columns such as `x_right_hand_3` unchanged, so those columns do not mean what their names say.
Pose and face lists show the same overlap in the frame viewed. `presence_*_hand` and `visibility_*_hand`
were all zero in the frame viewed, and are not exported.

Evidence: all 5,447 hand frames tested follow this pattern (25 records of train shard 0; 2 records each
of train shards 50 and 99, validation shard 0 and test shard 0). MediaPipe Hands run on the CELEBSTYLE video
agrees: all 65 stored frames lie within 2% of the image, median 0.1%, against 28% for the literal reading.

**Tracking quality varies.** SURYADIESEL has a hand in 16% of its frames, STUDENTWORK 38%, CELEBSTYLE 88%.
On the first two the hand is very close to the camera, stored points differ from direct MediaPipe by a median
6–8% of the image, and in the 12 frames inspected by eye they often did not sit on the hand. That is a small sample.

The code detects the layout from every frame. A corrected dataset would be read as complete hands, with
`layout = complete-21-landmarks`, without code changes.

### Exported files and the coordinate order

`README.txt` in the export (`fsboard.FORMAT_NOTES`) is the full reference. In short, `left_hand[t, i, c]`:
`t` is the frame, `i` the position in `landmark_ids` (MediaPipe numbering, the same order as
`asl.features.points_from_mediapipe`), `c` is 0 = x, 1 = y, 2 = z. Missing detections are NaN, marked by `left_mask` /
`right_mask`. x and y are fractions of the upright image's width and height.

For webcam code to supply the same inputs:

* Take landmarks 0–7 of MediaPipe's 21, and ignore the z of landmark 7.
* **Aspect ratio:** training coordinates are fractions of a portrait image; a webcam image is landscape. A
  fraction means something different on each, so convert using `image_width` and `image_height`. Not yet decided.
* **Mirroring:** `backend/asl/engine.py` flips the camera image before tracking, which turns x into 1 − x. FSboard's
  orientation looked unmirrored on the one clip examined; confirm this before training.
* **Left and right** are dataset labels. STUDENTWORK has 35 frames with both labels at once, so do not assume exactly
  one signing hand per frame.

### Decision before training: where do complete hands come from?

The exported files prove the pipeline, but 8 points per hand will likely limit accuracy, because many letters differ
by the missing fingers. Options:

1. **Run MediaPipe Hands on the FSboard MP4s (recommended).** Same tracker as the webcam app, all 21 points, and
   eligible clips can be picked directly from metadata at about 8 MB each, so 20 clips is about 150 MB with no shard
   needed. In a first try on the 3 clips above it found a hand in 89/121, 99/187 and 160/241 video frames. Accuracy of
   those points has not been measured.
2. **The competition's Parquet files** (the earlier notebook). Reportedly has all 21 points, but not checked here,
   and it needs the competition rules accepted.

## All-in-one notebook: `notebooks/fsboard_tcn_ctc_colab.ipynb`

The one to use. Upload it alone to Colab with a T4 GPU and choose Run all. Its defaults are the full run.
It is generated by `scripts/sequence_model/build_tcn_ctc_notebook.py`: change cells there, not in Colab, and
build again (since 7 Nov 2026 there are no paste-in cells).

**Steps:** 1 GPU and Drive; 2 choose one-word clips and phrases from metadata; 3 extract 21 points from videos;
4 build and check features; 5 optional tiny test; 6 train (`EXPERIMENT`, speed range 0.5–3.5, warm start from
`WARM_START_FROM`); 7 evaluate on unfamiliar signers next to `COMPARE_WITH`; 8 speed check; 9 export a small zip;
10 optional fine-tune on your own recordings (`FINE_TUNE_ON_MY_RECORDINGS`, off by default).

**How the 21 points are made** (`backend/asl/seq/fsboard_video.py`): each frame is turned upright using the MP4's rotation
flag, mirrored like the webcam in `backend/asl/engine.py`, then tracked with MediaPipe Hands (two hands, one fresh
tracker per clip). Raw points are saved before any normalisation, and each video is deleted once processed.

**MediaPipe's own environment:** Python 3.11 and mediapipe 0.10.14, the laptop's versions, in a separate folder created with
`uv`, so it can't clash with Colab's TensorFlow. Extraction runs there as a separate process.

**Aspect correction:** `isotropic_points` turns x and y into fractions of the image's longer side, so portrait and landscape
hands keep the same proportions. Training and the baseline comparison both use it. The laptop tools
`inference.py` and `seq_record_word.py` apply it too (tests compare them number for number); run the recorder with `--mirror-input`.

**Signing speed:** a replay of 17 of the user's webcam recordings gave 32% wrong letters at normal speed and about 11% at 3x, because they had about 30-50 frames per letter against about 12-15 in training. `--speed-up X` on the recorder is a stopgap for this. Retraining with a wider speed range is the likely real fix (not yet done).

**Recorder hand setting:** `--hand auto` (default) uses MediaPipe's label, as training did, and warns if the mirror check looks flipped. `--word` and `--save` store the word, image size and settings in each recording.

**Baseline comparison:** removed from the notebook on 7 Nov 2026 (it could not load in Colab's Keras, and it was no
longer needed). `fsv.baseline_text` was removed with the baseline on 23 Nov 2026 (the current notebook file still
carries an old copy of it and of `text_builder.py`; they are not called).

**What has been tested** (3 November 2026, on the laptop): automated tests, including every notebook cell run in order against a
fake Kaggle server. These used real MediaPipe on small videos made from the hand photo in `tests/data/old_live_screenshot.png`.
**Not yet run:** anything in Colab, any FSboard download through this notebook, the `uv` install, and the
`daun_v3-validation` video folder path (train and test paths were seen in Kaggle's listing). No model has been trained on real data.

## Training, `backend/asl/seq/train.py`

* Explicit loop: `tf.nn.ctc_loss(labels, logits, label_length, logit_length,
  logits_time_major=False, blank_index=0)` on raw logits, divided by word
  length and averaged. Then global-norm gradient clipping (1.0) and Adam (1e-3).
* Batches: grouped by similar length, padded on the **right** with zeros.
  True frame and letter lengths go to the loss.
* Augmentation (training only): time stretch 0.8–1.25 (never below the CTC
  minimum), small rotation, scaling, aspect change, noise, wrist shift and
  dropped frames.
* `checkpoints/last/` stores weights, optimizer state (`optimizer.npz`) and
  `state.json` (epoch, step, history, best CER). It is replaced atomically
  every epoch, and the data order per epoch is seeded, so a resumed run
  continues where it stopped.
* `best_model/` is the **inference bundle** of the best validation CER, kept separate.

## Bundle and version compatibility

The bundle stores **weights only** (`model.weights.h5`) plus
`model_config.json`. The laptop rebuilds the architecture from the same
code and loads the weights. That avoids depending on the full-model file
format matching between Colab's and the laptop's Keras versions.
`reference_io.npz` holds a few inputs and the logits Colab produced.
`scripts/sequence_model/seq_check_bundle.py` must report **Reference check ... OK** before you trust a bundle.

Laptop environment (measured 31 Oct 2026): Python 3.11.9, TensorFlow 2.17.1,
Keras 3.15.1, NumPy 1.26.4. Colab's versions are written to `versions.json`
when the notebook runs. No new laptop dependencies are needed: pandas and
pyarrow are only used in Colab.

## How to run

**1. Laptop: pack the code** (about 35 KB: code only, no data):
```powershell
venv\Scripts\python.exe scripts\sequence_model\seq_pack_code.py
```
Upload `dist\asl_seq_code.zip` to Google Drive → `MyDrive/asl_seq/`.

**2. Colab:** open `notebooks/asl_seq_tcn_ctc_colab.ipynb` (colab.research.google.com
→ File → Upload notebook), choose a T4 GPU runtime, and run the cells in
order. Cell 3 explains the Kaggle login, which uses Colab Secrets.

| Location | Contents |
|---|---|
| Temporary Colab disk `/content/` | code, one raw Parquet file at a time |
| Google Drive `MyDrive/asl_seq/` | `metadata/`, `manifests/<data>/`, `shards/<data>/`, `runs/<run>/checkpoints/last`, `runs/<run>/best_model`, `export/<run>_bundle.zip` |
| Laptop `backend/models/seq/<run>/best_model/` | exported bundle |

**3. Laptop: check the bundle, then record a word:**
```powershell
venv\Scripts\python.exe scripts\sequence_model\seq_check_bundle.py backend\models\seq\tcn_ctc_v1\best_model
venv\Scripts\python.exe scripts\sequence_model\seq_record_word.py backend\models\seq\tcn_ctc_v1\best_model --mirror-input --speed-up 2.5
```
In the camera window press **S**, fingerspell one word, press **F**. It prints
the predicted word, frame rate, per-frame time and a **mirror check**.
`--stream` also shows letters live: confirmed text, then the provisional letter in brackets.

## Evaluation report (`best_model/evaluation_report.json`)

For validation and test signers: CER, exact word accuracy,
substitutions/insertions/deletions, double-letter words (and how often the
double collapsed), slow/normal/fast (frames per letter, training tertiles),
per-signer CER, and letters emitted for "no hand" input.
These numbers describe this pilot subset only. Don't compare them directly
with other datasets or with the baseline's single-photo accuracy.

## Streaming

`StreamingRecognizer` keeps the last 60 frames as context. For each chunk of
new frames it runs the model on context + new frames, keeps only the new
frames' outputs, and passes them to a greedy decoder whose state survives
chunk boundaries. Each frame is decoded once, so double letters across a
boundary are kept, and nothing is stitched from separately decoded windows.
*Confirmed* = a letter whose run of frames has ended; *provisional* = the
letter currently being emitted. Tests check that chunked logits and text equal offline ones.

CTC does **not** detect the end of a word or sentence, or idle periods.
Start/Finish (and later explicit controls or a separate detector) still decide that.

## Phase 2A: fine-tune on your own recordings

Stretching FSboard clips (Phase 1) helped on simulated slow clips, but not on real
webcam signing. On 31 webcam recordings at natural pace, all three models scored
80–81% letters right. Phase 2A teaches the model your hand, camera and pace.

| Step | Where | Command or file |
|---|---|---|
| 1. Record training words (100 words × slow, normal, quick) | laptop | `seq_record_word.py backend\models\long_memory_model --mirror-input --words data\my_words_train.txt --save backend\storage\my_recordings` |
| 2. Record test words (40 other words × slow, normal) | laptop | the same, with `--words data\my_words_test.txt --paces slow,normal` |
| 3. Pack into one small file | laptop | `seq_pack_my_recordings.py`, which writes `backend\storage\my_recordings_pack.npz` |
| 4. Upload the pack | Google Drive | `MyDrive/asl_fingerspelling/fsboard_tcn_ctc/my_recordings/` |
| 5. Fine-tune and compare | Colab | step 10 of the notebook: tick `FINE_TUNE_ON_MY_RECORDINGS` |
| 6. Use the new model | laptop | unzip to `backend\models\personal_model` |

Rules that keep the score honest:
* A word belongs to exactly one group. **test** words come from `my_words_test.txt`
  and are never trained on. **validation** is every 7th training word, and picks
  the best epoch. **train** is the rest.
* The recorder saves only takes with a hand in view, long enough for their letters,
  and not mirror-flipped. N skips a word; R moves the last take to `discarded/`.
  Takes already saved are skipped, so a session can stop and continue later.
* Inputs are built by the same function as the live app (`inference.frame_features`),
  at real speed. The pack is refused in Colab if its feature settings differ from the model's.
* Fine-tuning starts from a copy of `…__speed_long_memory/best_model`, keeps its
  normaliser, mixes your takes with all FSboard training clips (yours are about half
  of each epoch) and uses a lower learning rate (3e-4). The base model's folder is not changed.
* The cell prints before vs after for your test words (all, slow, normal), your
  validation words, and the FSboard test clips. Keep the personal model only if
  your test words improve and the FSboard test score does not clearly get worse.

Tested on this laptop (6 Nov 2026) with fake data only (`tests/test_seq_personal.py`).
No real fine-tuning has run yet.

## Phase 2B: spaces and phrases

Result of Phase 2A (7 Nov 2026): on 40 test words the model never trained on, the personal
model got 91.5% of letters right (before: 71.7%), or 49 of 80 words exactly right. The laptop
reproduces this exactly. Three weaknesses remain: double letters (0 of 14 double-letter test
takes exactly right), a small loss on FSboard (letter error 10.8% → 12.2%), and very slow
signing (your Phase 2A takes averaged about 19 camera frames per letter; earlier recordings
averaged 36).

What Phase 2B adds:
* **Vocabulary v2:** output 27 is a space. Models made earlier keep 27 outputs and still load
  and read as before (`vocab.is_known_vocabulary`). Phrases are accepted only where asked
  (`allow_spaces=True`), and only with single spaces between words.
* **FSboard phrases:** `fsv.select_clips(..., phrases=True)` chooses only phrases with spaces,
  so they never overlap the one-word clips. Digits and symbols are still excluded whole.
* **Warm start:** `smodel.warm_start` copies the long-memory model's layers into the new model.
  The output layer keeps its 27 letter outputs and gets one new output for the space.
* **Scores:** `fsv.phrase_scores` reports letter errors with spaces ignored (comparable with
  older models), word errors, and the share of spaces found.
* **Your phrases:** `data/my_phrases_train.txt` (49), `data/my_phrases_test.txt` (14),
  `data/my_doubles_train.txt` (21), and the `veryslow` pace in the recorder.

In Colab: upload the current `notebooks/fsboard_tcn_ctc_colab.ipynb` and Run all. Step 2 adds the phrases, step 6
warm-starts from the long-memory run, steps 7–8 compare with it, and step 9 exports the zip.

Tested on this laptop (7 Nov 2026) with fake data only: every notebook cell runs in order against a fake Kaggle server
with phrase clips (`tests/test_seq_fsboard_video.py`), `tests/test_seq_personal.py` runs steps 2 and 10, and
`tests/test_seq_model.py` checks warm start and older models.

Real results (Colab, 8–10 Nov 2026; the laptop reproduces the "your test words" numbers exactly):

| Model (folder in `backend/models`) | Your 40 test words, letters right | Exact (of 80 takes) | Writes spaces |
|---|---:|---:|---|
| `long_memory_model` | 71.7% | 12 | no |
| `personal_model` (Phase 2A) | 91.5% | 49 | no |
| `phrase_model` (FSboard words + 1,500 phrases, 40 epochs) | 83.9% | 29 | yes |
| `phrase_personal_model` (step 10 on top of it) | 93.7% | 52 | yes |

On FSboard strangers the phrase model has 9.4% letter error on one-word clips (long-memory: 10.8%), 13.2% on
phrases (16.1%), and finds 39% of the spaces. Still weak: double letters in unseen words (0 of 14), spaces, and
fast signers. You have recorded no phrases, so spaces in your own signing are unmeasured; played one after the
other, two of your one-word takes come out with no space.

### Extra training clips by your own rule (step 2, 16 Nov 2026)

Step 2 picks its one-word clips and phrases at random, as before (defaults now 8,000 / 2,000 / 2,000 phrases,
the sizes of the 8,732-clip run). On top of them it adds `EXTRA_TRAIN_CLIPS` (default 3,000) **training** clips:
the unused ones that the `clip_score(text)` function at the top of the cell scores highest
(`fsv.choose_extra`). As written, a clip gets 1 point per letter in `WEAK_LETTERS` (default `QXYUZP`, the letters
under 90% in the 8,732-clip evaluation) and 1 point for a double letter. Edit that function to change the
selection; validation and test clips are never touched. A run with extra clips gets `_extraN-xxxxxx` in its name,
where `xxxxxx` identifies which clips were picked, so a changed rule never resumes another rule's training.
Steps 6 and 7 now start from and compare with the 8,732-clip run. Tested with fake data only; not yet run in Colab.

## Phase 3: word correction (`backend/asl/seq/correct.py`)

The model writes its best letter per moment and never asks whether the result is a word. Phase 3 adds that
check after the model. Nothing is retrained and no service is called; it uses a word list and the model's own
per-frame scores.

For each stretch of letters between the spaces the model wrote:

1. **Listed word:** kept.
2. **Near word:** listed words one or two letters away compete with keeping the letters. Score = how well the
   text fits what the model saw (CTC probability over all timings) + 0.2 × how common the word is − 2.0 for each
   letter the model wrote that the option drops + 1.0 if the option only doubles a letter. An unlisted word
   counts as very rare (ln probability −23).
3. **Split** (opt-in, `split_words=True` / `--split-words`): letters that no near word explains are cut into
   words from the 3,000 most common, changing no letter (`ILOVEYOU` → `I LOVE YOU`).
4. **Join:** two neighbouring stretches become one listed word if that scores better (`OXY GEN` → `OXYGEN`).

**Word list:** `data/words_en.txt`, 30,000 words with counts, made by `scripts/sequence_model/build_word_list.py`
from Peter Norvig's `count_1w.txt` (norvig.com/ngrams; counts derived from the Google Web Trillion Word Corpus).
Check that source's terms before publishing the file. Add names and other words of your own to
`data/my_extra_words.txt`.

**Measured on your recordings** (11 Nov 2026, `phrase_personal_model`, splitting off;
`scripts/sequence_model/seq_score_correction.py` prints this):

| | Exact words | Letters right | Fixed | Broken | Changed to another wrong word | Hidden-word check |
|---|---:|---:|---:|---:|---:|---:|
| Training words, 300 takes (settings chosen here) | 254 → 282 | 96.1% → 98.3% | 28 | 0 | 2 | 3 of 254 |
| Test words, 80 takes (the honest score) | 52 → 65 | 93.5% → 96.1% | 13 | 0 | 1 | 0 of 52 |

"Letters right" here counts a wrong space as an error, so it starts at 93.5%, not 93.7%. The hidden-word check
takes each rightly read word out of the list, as with a name, and counts how often the corrector then changes it.

**Double letters stay the main gap:** of the 14 double-letter test takes, 0 were right before and 4 after.
Where the model saw no trace of the second letter (`CHESE`, `SOCER`, `TENIS`), the fit is too poor for the
corrector to accept the word. A larger doubled-letter bonus did not help on the training words and raised the
hidden-word check (`FOX` → `FOXX`), so it stays at 1.0; recording `data/my_doubles_train.txt` is the real fix.

**Splitting is off by default.** With it on, the hidden-word check rises to 13 of 254 and 4 of 52 (for example
`TAXI` → `TAX I`), and a wrong reading can be cut (`TENIS` → `TEN IS`). It recovers 54 of your 63 phrases
when every letter is right, and 16 of 51 simulated two-word takes. It has not been tried on real phrase signing.

**Not fixed by Phase 3:** a wrong reading that is itself a word (`READ` for `BREAD`), words outside the list,
readings three or more letters off, and double letters the model saw no trace of (`JAZ` → `JAP`, not `JAZZ`).
Not yet used by the app or by live streaming; `seq_record_word.py` shows it after each recording
(`--no-correct` switches it off) and saves it as `corrected` next to the model's own `predicted`.

## Stage 2: the word decoder (`backend/asl/seq/word_decoder.py`, 17 Nov 2026)

Phase 3 repairs the letters after the model has chosen them. The word decoder chooses letters and words
together in one pass over the frames (word beam search), so it can also separate words the model ran together
(`ILOVEYOU` → `I LOVE YOU`), use a letter the model found second-best, and restore a double letter written
once. It is now what `seq_record_word.py` and `seq_score_correction.py` use; `--old-correct` brings Phase 3 back.

**How it reads a recording.** It follows the 40 best sentences at once. At each frame a sentence may stay
(blank, or the same letter held), add a letter, or end its word. A word may end at a space the model wrote
(free) or without one (cost 1.0): the model's space is a hint, not an order. Any letter may be added, so names
still come out; a word that is not in the list pays 3.0 once. When a word ends it is scored by how common it is
(weight 0.1) and by how much more often than chance it follows the word before it (weight 1.0, from
`data/word_pairs_en.txt`: `LOVE YOU`, `THANK YOU`). Restoring a double letter costs 1.0. The exact letters the
model itself wrote get 1.0 of extra trust, and no word may hold the same letter three times in a row.
It takes about 7 ms per recording on the laptop.

**Word pairs:** `data/word_pairs_en.txt`, the 100,000 most common two-word sequences of the listed words (2 MB),
made by `scripts/sequence_model/build_word_pairs.py` from Peter Norvig's `count_2w.txt`. Check that source's
terms before publishing the file.

**Risk levels** (`--risk`): `careful` trusts the model's letters more and separates less; `bold` repairs and
separates more and changes more names. `normal` is the default.

**Measured** with `new_model`. Settings were chosen on training material only; the test columns are the honest
score. "Two words" are two of your one-word takes played back to back (no real pause, so the hardest case);
"likely pairs" are the same with words that often go together (`GOOD MORNING`); "phrases, no pause" are your
phrase lists spelled perfectly with no space, to test separation alone. The hidden-word check takes one rightly
read word out of the list, as with a name, and counts how often the reading then changes.

| Training material (exactly right) | Model alone | Phase 3 | Decoder careful | Decoder normal | Decoder bold |
|---|---:|---:|---:|---:|---:|
| One word, 363 takes | 284 | 338 | 330 | 342 | 351 |
| Two words, 150 takes | 40 | 54 | 66 | 87 | 117 |
| Likely pairs, 91 takes | 25 | 29 | 74 | 88 | 88 |
| Phrases, no pause, 49 | 0 | 0 | 18 | 31 | 31 |
| Hidden-word check, one word | | 0 of 284 | 0 of 284 | 0 of 284 | 27 of 284 |

| Test material (exactly right) | Model alone | Phase 3 | Decoder careful | Decoder normal | Decoder bold |
|---|---:|---:|---:|---:|---:|
| One word, 80 takes | 58 | 71 | 66 | 71 | 74 |
| Two words, 40 takes | 12 | 16 | 16 | 22 | 30 |
| Likely pairs, 6 takes | 1 | 2 | 5 | 6 | 6 |
| Phrases, no pause, 14 | 0 | 0 | 5 | 10 | 8 |
| Hidden-word check, one word | | 0 of 58 | 0 of 58 | 0 of 58 | 5 of 58 |
| Right readings broken | | 0 | 0 | 1 | 1 |

On single test words the decoder at `normal` equals Phase 3 (71 of 80, letters right 94.8% → 97.8%); its gain is
in separating words. The one broken test reading is `WEATHER MOTHER` → `WE AT HER MOTHER` (the pair `HER MOTHER`
is very common). When the best sentence is less than 2.0 ahead of the next, or holds an unlisted word, the
recorder prints `NOT SURE`: on the training material, sentences of listed words that were at least 2.0 ahead
were right 384 times out of 385, against 133 of 152 below that.

**Not measured:** real phrase signing (no phrase has been recorded yet; `data/my_phrases_test.txt` is the list),
live use, and any model other than `new_model`. **Not fixed:** a letter the model was sure of and got wrong
(`BUEAD`, `OUANGE`, `BROKTHER`), words outside the list, `NOWHERE` against `NOW HERE` without a pause. The
decoder runs after Finish, not while signing.

### Live words on the webcam (19 Nov 2026)

`seq_record_word.py ... --stream` now shows words while you sign: confirmed words, which never change again,
then `[the current guess]`, which may still change. `LiveWords` in `word_decoder.py` reads all unconfirmed
frames every 4 model frames (0.13 s). A word is confirmed once 6 readings in a row (0.8 s) have agreed on it and
on every word before it, while a further word is already being signed, so a pause between two letters never
confirms half a word. After 300 unconfirmed frames (10 s) the frames of confirmed words are dropped and the last
confirmed word is passed on to the word-pair table. After F the whole recording is read once more; that is the
final answer, and the recorder prints whether the live text matched it. `StreamingRecognizer` now keeps every
frame's scores in `frame_logits` for this.

A first rule that confirmed words at any pause was rejected: pauses between letters look like pauses between
words, and it showed `HUNGRY` as `H U N G R Y`.

**Measured** by replaying saved takes camera frame by camera frame with `final_model` (not yet tried live):

| Replayed takes | Live text at the end = Finish result | A confirmed word changed later | Time per reading, median / 95% / max |
|---|---:|---:|---|
| 148 of your one-word takes | 148 | 0 | 1.8 / 7.5 / 12.8 ms |
| 40 two-word takes (two takes with the hand held still for 0.6 s between) | 39 | 0 | 7.1 / 18.1 / 152 ms |
| Your 6 webcam tries from 16 Nov (I LOVE YOU) | 6 | 0 | 5.7 / 13.0 / 14.7 ms |

The one mismatch: live `RAIN APLE` (RAIN had been confirmed), Finish `RAINAPLE`. On the webcam tries the
decoder reads `LOVE YOU` five times and `I LOVE YOU` once where the model wrote `LOVEYOU` or `ILOVEYOU`
(one try stays `LOUVEYOU`). Because words are confirmed only after 0.8 s of agreement, most words stay
"current" until Finish in quick signing; that is by design. Camera tracking adds about 18 ms per frame.

The first live take used to freeze for about 0.7 s: TensorFlow compiles the model on its first call. The
recorder now calls `WordRecognizer.warm_up()` before the camera opens (`final_model`: 671 ms at startup, then
4.4 ms for the first chunk instead of 728 ms), and the timing line after F also prints the slowest frame.

### final_model with the word decoder (19 Nov 2026)

`final_model` is the step-10 fine-tune of `train8829_val2103_test2131_letters2-100_phrases40_weak97-ad84fd`,
trained on the re-recorded double letters. Its training numbers are high because it trained on those takes.

| Your 80 test takes | Model alone | Phase 3 corrector | Word decoder (normal) |
|---|---:|---:|---:|
| `new_model` exact | 58 | 71 | 71 |
| `final_model` exact | 67 | 76 | 75 |
| `final_model` letters right | 96.7% | | 98.7% |
| `final_model` hidden-word check | | 0 of 67 | 0 of 67 |

The 5 test takes still wrong are all a letter the model was sure of: `BROKTHER`, `OUANGE`, `OUANHE`, `QUEATION`,
`SUGAU`. The decoder's settings were chosen with `new_model` and have not been re-tuned for `final_model`.

### The missing I in I LOVE YOU, and final_model_i (22 Nov 2026)

In I LOVE YOU the model often wrote `LOVEYOU`, although the hand points show the I shape held for 0.3-0.8 s.
No training word started with I (only the test word ISLAND does), so the model had never learned an I with no
letter before it. Distance mattered too: the 16 Nov tries were farther from the camera (hand size 0.20 of the
picture) than the training takes (median 0.25). `data/my_i_words_train.txt` (I, IF, IN, ... 14 words, 42 takes)
was recorded and added to the pack (348 train / 57 validation / 80 test; the old pack is kept as
`backend/storage/my_recordings_pack_before_i_words.npz`), then step 10 was run again from the same FSboard base:
`final_model_i`. `final_model` is kept unchanged.

| | `final_model` | `final_model_i` |
|---|---:|---:|
| I LOVE YOU tries, never trained on (6 from 16 Nov, 7 from 21 Nov): I written | 7 of 13 | **12 of 13** |
| Test takes (80), model alone, exact | 67 | **70** |
| Test takes, after the word decoder (normal), exact | **75** | 73 |
| Test letters right, after the decoder | 98.7% | 98.3% |
| Hidden-word check on test | 0 of 67 | 0 of 70 |
| Test double-letter words, model alone (Colab) | 11 of 14 | 12 of 14 |
| FSboard one-word CER / phrase CER (Colab) | 10.58% / 11.15% | 10.64% / 11.13% |
| FSboard phrase spaces found / extra spaces (Colab) | 70.8% / 190 | 72.1% / 203 |

The two test takes the decoder no longer repairs (`OXY GEN`, `SITER`) get the same raw reading from both models;
`final_model` won them by 0.1-0.4 points, and `final_model_i` is more confident in its own reading. One I LOVE YOU
try read `ILOVEYJOU` (the decoder fixes it): J and I share the pinky-up shape, worth watching.

**Decoder re-tuned for `final_model_i`, and the settings were kept.** It trained on the training takes, so at normal
speed it reads 400 of 405 of them right, which leaves too few mistakes to tune on. The material was therefore those
takes at 1x, 1.5x and 2x playback speed (2x: 364 of 405 right), plus simulated two-word takes, likely pairs and
phrases with no pause. 288 settings were tried on that training material. The settings that scored best there
trust the model's own letters more (`raw_bonus` 1.25, `pair_weight` 1.5). On the test material they were worse,
not better (one word at 1.5x 76 → 74, at 2x 77 → 75, two-word takes 28 → 23 of 40), and none of them repairs
`OXY GEN` or `SITER`. So the defaults stay. With the decoder, the two models end about level on test takes at all
speeds (1x / 1.5x / 2x: `final_model` 75 / 75 / 77, `final_model_i` 73 / 76 / 77; simulated two-word takes 31 vs
28 of 40).

### Bigger word lists were tested and rejected (19 Nov 2026)

With `new_model`, 30,000 / 50,000 / 100,000 words gave 71 / 68 / 66 exact test words (training 342 / 334 / 330).
The bigger lists contain web misspellings that are exactly this model's mistakes (`JAZ`, `HELO`, `FET`,
`BALLON`, and in the 100,000 list `TENIS`, `PEPER`), so the decoder accepts them instead of repairing them.
Add your own names to `data/my_extra_words.txt` instead.

### Double letters signed again (17 Nov 2026)

All double-letter takes recorded before 17 Nov were signed the wrong way, so every earlier double-letter number
in this file measured the wrong movement. The 86 old takes are kept in
`backend/storage/my_recordings/wrong_doubles/` and are not packed. The 24 double-letter training words and the
7 test words were recorded again, and `data/my_doubles_train.txt` (21 words) for the first time: 443 takes in all.

`new_model` (which trained on the old signing) on the new takes, before any retraining:

| Takes | Exact, model alone | Exact, after Phase 3 | Double written by the model |
|---|---:|---:|---:|
| 7 test words with a double, old signing (14) | 1 | 8 | 2 |
| The same, new signing (14) | 4 | 12 | 4 |
| 24 training words with a double, new signing (72) | 35 | 57 | 37 |
| 21 new double words, never trained on (63) | 31 | 60 | 34 |
| All 80 test takes, new signing | 58 | 71 | |

The model alone still writes the double in only about half of the new takes; the word list repairs most of the
rest. Rerunning notebook step 10 with the repacked recordings is the next step and has not been run.

## What was actually tested (31 Oct 2026, this laptop)

* `tests/test_seq_core.py`, `tests/test_seq_model.py`: 42 tests, all pass.
  Covered: CTC repeats and blanks, padding with true lengths, CTC loss
  ignoring padding, invalid/missing landmarks, velocity across gaps,
  causality of features and model, save/reload equivalence, checkpoint
  restore of optimizer state, offline = streaming, and a tiny synthetic
  overfit reaching CER < 0.05, and camera-rate resampling into the recogniser.
* `tests/test_seq_data.py`: 4 tests with pandas/pyarrow. They pass when
  those are installed (they are skipped on the laptop's venv and run in Colab cell 2b).
* The **whole notebook** ran end to end against a fake Kaggle API serving
  synthetic Parquet files in the dataset's column layout. Simulated zipped
  downloads and pagination, the shards, signer split, validation, a tiny overfit
  (CER 0.0 on its memorised examples), a 4-epoch run, evaluation, resume and
  export all worked. Synthetic results say nothing about real accuracy.
* Laptop CPU timing (i5-8265U), compiled model, *untrained weights* (same
  compute as trained): feature step 0.08 ms per frame; streaming model call
  2.2 ms (60 context frames + 1–4 new); a 3 s word offline 2.4 ms. Camera
  decoding and MediaPipe tracking are extra: about 18 ms per frame in the
  baseline benchmark (24.6 ms total minus 6.2 ms for its model).

## Still needs data or verification

1. Run the notebook on the real dataset. That checks columns, the number
   of compatible words and the signer count.
2. **Frame rate:** `target_fps = 30` is an *assumption*; the dataset doesn't
   state it. Compare frames per letter in the dataset with your own
   Start/Finish recordings.
3. **Mirror convention:** compare the dataset's chirality score with your
   own (printed by `seq_record_word.py`). Toggle `--mirror-input` if they disagree.
4. **Aspect ratio:** dataset image sizes are unknown; augmentation
   partly covers this.
5. Real latency end to end with the camera, and idle/transition insertions
   on your own recordings.
6. Accuracy on *your* hand. That needs labelled word recordings (see Roadmap).

### The browser app: final_model_i and the baseline side by side (23 Nov 2026, baseline later removed)

`start.bat` / `python -m backend.server` opens the previous Live screen with a **final_model_i | Baseline**
switch (final_model_i is the default), so the two models can be compared on the same signing.
* The camera turns on by itself; **Start** begins reading with the chosen model, **Stop** pauses it.
* **final_model_i** (`/ws/live`, `backend/asl/seq/live_session.py`, the same steps as `seq_record_word.py
  --stream`): 640 x 480 at 30 fps with each frame's capture time; the panel shows confirmed words + `[current
  guess]`; **Finish sentence** reads the whole sentence once more (like F), stores it, speaks it and keeps reading.
  **Delete** removes the last word of the last finished sentence and **Edit** corrects it (words still being read
  can change, so they are not edited).
* **Baseline** (`/ws/recognize`, unchanged rules): hold each letter steady, Confidence / Hold bars, top-3,
  **Space**, **Delete**, **Clear**, **Edit**, **Finish sentence**, as before.
* Switching models keeps the finished sentences and drops the sentence being read.
* Keys on the camera screen: **S** start / stop, **Delete**, **C** clear, **F** finish sentence (ignored while
  typing in a text box). final_model_i's live words are plain text (confirmed and guessed words look alike).
* The A symbol at the top left opens a side strip like Chrome's vertical tabs: **Live** and **ASL Alphabet**
  (the former pose review page: every reference handshape in 3D, with its verification status).
* Speech to ASL fingerspelling has a **Clear** button; under the 3D hand only the letter is shown.
* The 3D hand is always on screen: a resting open hand, which follows your hand while reading (both models).
  It reuses the points the server already finds, is drawn in the browser, sized by the palm and lightly smoothed
  for display only, so it does not slow the model.
* Not shown for now: the Dataset and Models screens and the Word test (they only worked for the baseline);
  Alphabet / Digits (final_model_i has no digits).

**Later the same day the baseline was removed** at the user's request: the `baseline/` folder (model, scripts,
results), its live path (`/ws/recognize`, `classifier.py`, `engine.py`, `text_builder.py`), the old training
system (`registry.py`, `dataset.py`, `training.py`, the Dataset / Models / Word test screens and their API) and
the ASL Alphabet photos, all sent to the Recycle Bin. The model switch, the baseline's Confidence / Hold bars and
the Space button are gone from the page; everything else above is unchanged. `data/landmarks.npz` is kept
because the 3D reference poses are built from it.

Server time per frame, measured on the laptop: `final_model_i` + live words median 0.3 ms, 95% 14.9 ms, max 21 ms;
JPEG decode + mirror + MediaPipe about 18-26 ms, so expect about 25-30 fps; the camera panel shows the real fps
and server ms. The model loads when reading first starts (a few seconds). Check without a webcam:
`set ASL_LIVE_REPLAY=<a saved take .npz>` before starting the server, then open `http://127.0.0.1:8000/?replay=1`.
