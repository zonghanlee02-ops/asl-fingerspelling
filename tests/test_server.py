"""
Server tests: health, and live reading with final_model_i over /ws/live.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

BLANK = np.full((480, 640, 3), 235, np.uint8)   # empty wall: no hand


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server.main as m
    with TestClient(m.app) as c:
        yield c


def jpeg(frame):
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def test_health_and_the_old_baseline_is_gone(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["speech"]["engine"] == "vosk" and "models" not in h
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/recognize") as ws:   # the baseline's connection was removed
            ws.receive_json()


def test_the_page_is_never_an_old_saved_copy(client):
    import server.main as m
    if not (m.DIST / "index.html").exists():
        pytest.skip("frontend not built")
    for path in ("/", "/index.html", "/some/screen"):
        assert client.get(path).headers.get("cache-control") == "no-cache"


# ------------------------------------------- live reading with final_model_i (/ws/live)
LIVE_MODEL = ROOT / "backend" / "models" / "final_model_i"
ILY_TAKES = sorted((ROOT / "backend" / "storage" / "i_love_you_test").glob("*.npz"))
needs_live = pytest.mark.skipif(not LIVE_MODEL.exists() or not ILY_TAKES,
                                reason="final_model_i or your saved I LOVE YOU takes missing")


@needs_live
def test_live_frames_carry_time_and_no_hand(client):
    import numpy as np
    live = client.get("/api/health").json()["live_model"]
    assert live["name"] == "final_model_i" and live["found"]
    with client.websocket_connect("/ws/live") as ws:
        for k in range(3):
            ws.send_bytes(np.float64(1000.0 + 33.3 * k).tobytes() + jpeg(BLANK))
            r = ws.receive_json()
            assert r["type"] == "frame" and r["hand"] is None and r["model"] == "final_model_i"
        ws.send_text('{"type":"command","action":"finish"}')
        assert ws.receive_json() == {"type": "text", "finished_text": "", "sentences": []}


@needs_live
def test_live_replay_of_a_saved_take_reads_i_love_you(client, monkeypatch):
    monkeypatch.setenv("ASL_LIVE_REPLAY", str(ILY_TAKES[1]))      # a take final_model_i reads right
    frames = []
    with client.websocket_connect("/ws/live?replay=1") as ws:
        import numpy as np
        n = len(np.load(ILY_TAKES[1])["t"])
        while len(frames) < n:
            frames.append(ws.receive_json())
        ws.send_text('{"type":"command","action":"finish"}')
        done = ws.receive_json()
    hands = [f["hand"] for f in frames if f["hand"]]
    assert len(hands) > n * 0.8 and all(len(h) == 21 and len(h[0]) == 3 for h in hands)   # 3D points for the hand
    assert frames[-1]["confirmed"] or frames[-1]["current"]                               # live words shown
    assert done["finished_text"] == "I LOVE YOU" and done["sentences"] == ["I LOVE YOU"]
