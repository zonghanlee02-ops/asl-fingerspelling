# Roadmap (next training phases)

Order agreed for expansion:

1. **Done – alphabet + live words.** Existing static classifier, improved
   word construction (repeat cues, states, corrections), browser interface.
2. **Done – video annotation and retraining tools.** Local import, labelled
   segments, provenance, grouped split, versioned models, comparison.
3. **Ready, needs data – digits 0–9.** Separate `digits` mode and model.
   Record labelled videos of digits from several signers, extract, train
   with `mode=digits`, compare, activate. Digits and letters stay separate
   until a combined model is evaluated against both.
4. **In progress – movement-aware continuous fingerspelling (sequence model).**
   See [SEQUENCE_MODEL.md](SEQUENCE_MODEL.md).

## Phase 4: temporal model (TCN + CTC)

The project's sequence bonus requires the **trained model itself** to output
the ordered letters from video. Processing video frame by frame and
assembling letters with rules (what the app does now) does **not** meet it,
and the app does not claim it.

Stages:

1. **Done (prototype)** – data and Colab setup: Google ASL Fingerspelling
   pilot subset with a download budget, remote extraction to compact shards,
   A–Z words only (incompatible transcripts excluded whole), signer split,
   manifests. Code in `backend/asl/seq/`, notebook in `notebooks/`.
2. **Done (prototype)** – 68-feature causal preprocessing shared by Colab and laptop.
3. **Done (prototype)** – compact causal TCN with CTC (blank = 0, A–Z = 1–26).
4. **In progress: FSboard data** – `notebooks/fsboard_prepare_data_colab.ipynb` selects
   recordings, plays one, exports verified training sequences. FSboard's landmark files hold only
   8 of 21 hand points, so `notebooks/fsboard_tcn_ctc_colab.ipynb` extracts all 21 from the videos, then trains and
   evaluates. **Next: run it in Colab**
   ([details](SEQUENCE_MODEL.md#fsboard-data-notebooksfsboard_prepare_data_colabipynb)).
   Then validation, tiny overfit, pilot training, resumable checkpoints.
   *No real training has been run yet.*
5. **Then** – evaluation report (CER, word accuracy, repeats, signers, speed,
   idle); laptop check of the exported bundle; Start/Finish word test
   (`scripts/sequence_model/seq_record_word.py`); measured laptop latency.
6. **Only after 5 works** – streaming in the app. `StreamingRecognizer`
   (context buffer, decoder state across chunks, confirmed vs provisional)
   exists and is tested offline = chunked. It is not yet wired into the
   server or UI, and word/sentence end and idle detection still need controls
   or a separate detector.
7. Later – your own labelled word recordings (done with `seq_record_word.py`)
   for fine-tuning and a personal evaluation; space token for multi-word
   phrases; beam search or a word list as an optional decoder.

J and Z become learnable because motion is part of the input.

## Interface and other features (long-term list, keep)

Several of these exist in the app (see README). Since 23 Nov 2026 the app reads with
`final_model_i`; the old baseline model was removed:

1. **Local browser interface** – Python backend; full-window responsive
   layout; Apple-inspired spacing, typography and controls with Windows
   system-font fallbacks; 3D reference hand on the left, live camera on the
   right, interpreted text in a bottom-centre caption. *(Exists, with `final_model_i`.)*
2. **3D hand shows the accepted letter's verified reference pose**, not a
   mirror of every movement. *(Changed 23 Nov 2026 at the user's request: on the camera screen the 3D hand
   follows your hand; reference poses show in speech mode and on the ASL Alphabet screen. Poses still need
   human verification.)*
3. **Optional text-to-speech** – read the finished text once on *Finish
   sentence*; enable/disable, replay and stop. *(Exists.)*
4. **Optional speech-to-fingerspelling** – speech → editable transcript →
   ordered ASL letter animations, local speech recognition only, never
   silently sending audio to the cloud. Labelled as fingerspelling, not full
   ASL translation. *(Exists with Vosk.)*
5. **Local video import and annotation** – label recordings, extract
   features, prepare training data, compare model versions. Importing a
   video never implies automatic learning. *(The single-letter version was removed with the baseline on
   23 Nov 2026; recording and packing whole words is done with `seq_record_word.py` and
   `seq_pack_my_recordings.py`. A browser version is planned.)*
6. **Digits and a defined vocabulary of ASL word signs** – word signs are a
   separate task from continuous fingerspelling. ASL only.
7. **Documentation** – keep README, dependency lists, dataset attribution
   and limitations complete and reproducible.

## Common signs (later)

A defined, small vocabulary (e.g. HELLO, THANK-YOU, YES, NO) as whole-sign
classes in a separate mode, trained from labelled video segments with the
same provenance and grouped evaluation. Labels for ASL only; other sign
languages would need their own datasets, label sets and evaluation and are
never mixed under the same labels.

Out of scope for the first release: full grammatical ASL translation and
unrestricted word-sign recognition.
