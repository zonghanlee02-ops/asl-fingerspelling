"""
asl - core package for the ASL fingerspelling project.

Modules
-------
features      Wrist-centred finger shape of one hand (used inside the sequence features).
tracker       MediaPipe hand tracking wrapper (supporting tool, not the recogniser).
speech        Optional local speech-to-text (Vosk). Never required for recognition.
seq/          The sequence model: TCN + CTC, live reading and the word decoder.

The old one-letter-per-frame baseline model was removed on 23 Nov 2026.
"""
from pathlib import Path

# Where everything lives. This file is <project>/backend/asl/__init__.py
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent                         # the project folder
MODELS_DIR = BACKEND_DIR / "models"               # sequence models and the speech model
STORAGE_DIR = BACKEND_DIR / "storage"             # your recordings and packs
FRONTEND_DIST = ROOT / "frontend" / "dist"        # built browser interface
