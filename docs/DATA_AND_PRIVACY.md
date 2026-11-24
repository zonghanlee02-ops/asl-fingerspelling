# Data sources, licences and privacy

## Data used

| Data | Where | Provenance | Published in the repo? |
|---|---|---|---|
| ASL Alphabet (Kaggle, `grassknoted/asl-alphabet`) | removed on 23 Nov 2026 with the baseline model | Downloaded by the user; one signer, consecutive frames | No |
| Landmarks extracted from it | `data/landmarks.npz` | Made by the removed baseline's extractor; source of the 3D reference poses | No (git-ignored) |
| Your recorded takes and packs | `backend/storage/` | `seq_record_word.py` / `seq_pack_my_recordings.py` | **No** (`backend/storage/` is git-ignored) |
| Trained sequence models | `backend/models/<name>/` | Colab notebook; fine-tuned on your takes (weights and settings only, no recordings) | Only `final_model_i` is published (the app needs it); every other model folder is git-ignored |
| Reference poses | `frontend/public/reference_poses.json` | 17 medoid landmark sets from the ASL Alphabet dataset; G, J, M, N, P, Q, T, X, Z from your own recorded takes (hand points only, no image) | Yes — check the dataset's licence on its Kaggle page first, and decide whether to publish poses of your own hand |
| FSboard (Kaggle, `googleai/fsboard`, CC BY 4.0) | Colab only; optional small export in `data/fsboard_pilot/` | Google; 147 paid, consenting Deaf signers; synthetic phrases | **No** (`data/fsboard*/` is git-ignored) |
| Vosk English model | `backend/models/vosk/` | alphacephei.com, Apache 2.0 | No (download script) |

## FSboard responsible use

FSboard's videos show signers' faces. The publisher asks that faces be blurred when publicizing examples,
and that nobody tries to re-identify signers or use their likeness to generate content. The Colab notebook
embeds the preview video in its output, so clear outputs (Edit → Clear all outputs) before sharing it. The
copy in this repository has no outputs. Cite the FSboard paper if you use the data (arXiv:2407.15806).

## Personal recordings

* Recordings stay on this computer (`backend/storage/`). The app has no upload to
  the internet and binds to `127.0.0.1` only.
* Only record other people with their consent.
* `.gitignore` excludes `backend/storage/` and the large files in `data/`, so `git add .` will not
  publish personal recordings. The model folders in `backend/models/` are **not** excluded and
  were fine-tuned on your recordings. Check `git status` before every push.

## Speech

Speech recognition uses the local Vosk engine. Recording starts only after
**Start microphone** and shows a red recording indicator. Audio is streamed
to the local backend only and is not stored. If Vosk is not installed the
feature says so; there is no fallback to cloud transcription (browser
`SpeechRecognition` is deliberately not used because it may send audio to a
remote service and is not available in every browser).

Text-to-speech uses the browser's `speechSynthesis` with English voices.
On Windows these are the installed system voices.
