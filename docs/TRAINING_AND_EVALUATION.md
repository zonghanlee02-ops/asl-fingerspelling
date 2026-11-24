# Training, data and evaluation

## The model the app uses

`final_model_i`: a causal TCN trained with CTC loss (`backend/asl/seq/model.py`), first on
Google's FSboard videos, then fine-tuned on the author's own recorded takes (notebook step 10).
Training, data, evaluation and every result are described in [SEQUENCE_MODEL.md](SEQUENCE_MODEL.md).

* Training runs in Google Colab: `notebooks/fsboard_tcn_ctc_colab.ipynb`.
* Record your own takes on the laptop: `scripts/sequence_model/seq_record_word.py`.
* Pack them for Colab: `scripts/sequence_model/seq_pack_my_recordings.py` (train / validation / test words
  come from separate word lists, so test words are never trained on).
* Check an exported model on the laptop: `scripts/sequence_model/seq_check_bundle.py`.

More data does not automatically improve accuracy, so always compare a new model with the current
one on the same test takes before switching.

## The removed baseline (history)

Until 23 Nov 2026 the project also had a one-letter-per-frame baseline: a small multilayer
perceptron on 63 landmark numbers, trained on the Kaggle ASL Alphabet photos, with hold-still
rules joining letters into words, plus screens to import videos and retrain it. It was removed
because `final_model_i` replaced it. Its design and results are kept in
[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md). Its 0.990 test accuracy came from a random
frame split of consecutive photos of one signer, so it was optimistic.

## Reference poses

`scripts/app/build_reference_poses.py` picks, for each letter, the real sample
closest to all others (medoid) from `data/landmarks.npz` (hand points from the ASL Alphabet photos;
the script that made that file was removed with the baseline, so keep the file if you may rebuild the poses).
Status values:

* `unverified` (default) – shown as "Data-derived pose · not yet verified".
* `dev_check: flagged` – during development the rendered poses for
  **A, E, M, N, S, T, X, Z** were seen not to show curled fingers clearly
  (MediaPipe's estimate for closed-hand photos). They are labelled
  "Pose may be inaccurate". This was a visual check, *not* an ASL verification.
* `verified` / `rejected` – set by a person who compared the pose with a
  reliable ASL reference: `python scripts/app/verify_poses.py --letters A --by NAME --reference SOURCE`.
  Review them on the **ASL Alphabet** screen (click the A symbol at the top left, then ASL Alphabet).

J and Z motion paths come from their standard written descriptions, not from data.

**Poses from your own signing (23 Nov 2026).** The photo poses for G, J, M, N, P, Q, T, X and Z looked wrong
in 3D, so they were rebuilt from your recorded takes:

```powershell
python scripts/app/build_reference_poses.py --from-recordings G,J,M,N,P,Q,T,X,Z
```

`final_model_i` reads every take in `backend/storage/my_recordings` (485 takes, 19 s on the laptop). Each time
it is surest of a letter of the take's word, the stillest hand frame within 0.1 s of that moment is kept; the
pose is the medoid of those frames (17 frames for Q up to 147 for N). The frames are turned back to the camera's
own view and x / y are put in the same units, so they face the same way as the photo poses (checked on B, L
and V built the same way). A first try looked 0.4 s back from that moment and picked the previous letter: the
model is sure of a letter while the shape is shown, not after it.

Result, judged from the 3D render: G and Z now show the right handshape; P and Q point down as they should but
show extra straight fingers; X shows a straight, not hooked, index; J shows the little finger out, at one moment
of the movement; M, N and T are still loose claws, because MediaPipe cannot see fingers folded over the thumb
from any camera. These poses show how YOU sign, so they start unverified again. The previous poses are kept in
`data/reference_poses_kaggle_backup.json`.

**Verified (24 Nov 2026).** The author checked all 26 poses on the ASL Alphabet screen and judged each
handshape correct; all 26 are marked `verified` (by "project author"). The development flags on A, E and S
are no longer shown for verified letters. Rebuilding a letter with a different result resets it to
`unverified`, so a new pose is never shown as verified.

## Commands

```powershell
python scripts/app/build_reference_poses.py        # all 26 from the photos: undoes the 9 letters from your takes
python scripts/app/build_reference_poses.py --from-recordings G,J,M,N,P,Q,T,X,Z
python -m pytest                            # Python tests
npm --prefix frontend test                       # front-end planner tests
```
