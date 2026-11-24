"""
Speech-to-text tests (optional feature). Skipped unless the Vosk model is
installed (python scripts/app/download_vosk_model.py) AND a 16 kHz mono WAV is
given in the ASL_TEST_WAV environment variable together with its expected
words in ASL_TEST_WAV_TEXT, e.g.

    set ASL_TEST_WAV=C:\\path\\hello.wav
    set ASL_TEST_WAV_TEXT=hello book
    python -m pytest tests/test_speech.py
"""
import json
import os
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from asl.speech import SpeechEngine  # noqa: E402

WAV = os.environ.get("ASL_TEST_WAV")
TEXT = os.environ.get("ASL_TEST_WAV_TEXT", "")
ready = SpeechEngine().status()["available"] and WAV and Path(WAV).exists()
pytestmark = pytest.mark.skipif(not ready, reason="Vosk model or ASL_TEST_WAV not available")


def test_speech_websocket_streams_pcm_and_returns_transcript(tmp_path):
    from fastapi.testclient import TestClient
    import server.main as m

    with TestClient(m.app) as client, client.websocket_connect("/ws/speech") as ws:
        assert ws.receive_json()["type"] == "ready"
        with wave.open(WAV, "rb") as w:
            while chunk := w.readframes(1600):          # 0.1 s chunks, like the browser
                ws.send_bytes(chunk)
                ws.receive_json()
        ws.send_text(json.dumps({"type": "stop"}))
        final = ws.receive_json()
        while final["type"] != "final":
            final = ws.receive_json()
    for word in TEXT.split():
        assert word in final["transcript"]
