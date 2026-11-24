"""
download_vosk_model.py - Download the small English Vosk model (about 40 MB)
for the OPTIONAL "Speech to ASL fingerspelling" feature.

After this, speech recognition runs fully offline on this computer.

    python scripts/app/download_vosk_model.py

Model: vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models
(Apache 2.0 licence). It is saved to backend/models/vosk/ (git-ignored).
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
DEST = Path(__file__).resolve().parents[2] / "backend" / "models" / "vosk"


def main():
    target = DEST / NAME
    if target.exists():
        print(f"Already installed: {target}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    zpath = DEST / f"{NAME}.zip"
    print(f"Downloading {URL} ...")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(100, blocks * block_size * 100 // total)
            print(f"\r  {done}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(URL, zpath, progress)
    except Exception as e:
        sys.exit(f"\nDownload failed: {e}\nDownload the zip manually from {URL} "
                 f"and unzip it into {DEST}")
    print("\nExtracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DEST)
    zpath.unlink()
    print(f"Installed: {target}")


if __name__ == "__main__":
    main()
