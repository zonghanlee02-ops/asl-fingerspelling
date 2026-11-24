# Tools

Run these from the project folder with `venv\Scripts\python.exe`.

**`app/`: for the browser app**

| Script | Purpose |
|---|---|
| `build_reference_poses.py` | rebuilds the 3D reference poses (needs `data/landmarks.npz`) |
| `verify_poses.py` | records that a person checked a 3D pose |
| `download_vosk_model.py` | downloads the optional speech model |

**`sequence_model/`: for the sequence model**

| Script | Purpose |
|---|---|
| `seq_check_bundle.py` | checks a trained bundle on this laptop and measures its speed |
| `seq_record_word.py` | fingerspell one word to the webcam and let the model read it |
| `seq_pack_code.py` | makes `dist/asl_seq_code.zip` for the training notebook |
| `build_tcn_ctc_notebook.py` | rebuilds the all-in-one notebook after its code changes (it overwrites edits made in the notebook) |
| `build_fsboard_notebook.py` | rebuilds the older FSboard notebook after its code changes |
