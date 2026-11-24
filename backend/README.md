# Backend: the Python side of the browser app

Runs only on your computer, at http://127.0.0.1:8000.

```powershell
venv\Scripts\python.exe -m backend.server
```

| Folder | What it is |
|---|---|
| `server/` | FastAPI web server: serves the interface, API and live connections |
| `asl/` | hand tracking, finger features, speech |
| `asl/seq/` | the sequence model (TCN + CTC): training code, live reading, word decoder, FSboard data code |
| `models/` | trained sequence models (the app uses `final_model_i`) and `vosk/` (speech, git-ignored) |
| `storage/` | your recorded takes and packs. Personal data, git-ignored |

The old baseline model was removed on 23 Nov 2026. Details: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).
