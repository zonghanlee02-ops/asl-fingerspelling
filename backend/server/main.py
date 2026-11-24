"""
server/main.py - Local FastAPI backend for the browser interface.

Run:   python -m backend.server            (then open http://127.0.0.1:8000)

Everything runs on this computer. The browser sends camera frames to /ws/live;
our own sequence model (final_model_i, TCN + CTC + word decoder) reads them.

Endpoints (summary)
    GET  /api/health                     status of the live model, speech engine, frontend
    WS   /ws/live                        live reading (time + JPEG in, JSON out)
    WS   /ws/speech                      optional local speech-to-text (Vosk)
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from asl import FRONTEND_DIST, MODELS_DIR
from asl.speech import SpeechEngine

app = FastAPI(title="ASL Fingerspelling (local)")
speech_engine = SpeechEngine()


@app.on_event("startup")
def _warm_up_speech():
    # Load the (optional) Vosk model in the background so the first
    # recording starts quickly. Failures are reported by /api/speech/status.
    if speech_engine.status()["available"]:
        threading.Thread(target=lambda: _safe(speech_engine.model), daemon=True).start()


def _safe(fn):
    try:
        fn()
    except Exception as e:  # keep the server running; status explains the problem
        print(f"Speech engine could not load: {e}")


class NoModelError(Exception):
    pass


# ----------------------------------------------------------------- health
@app.get("/api/health")
def health():
    return {"ok": True, "speech": speech_engine.status(),
            "live_model": {"name": LIVE_MODEL_DIR.name, "found": (LIVE_MODEL_DIR / "model_config.json").exists(),
                           "loaded": bool(_live)},
            "frontend_built": (FRONTEND_DIST / "index.html").exists()}


# ------------------------------------------- live recognition, sequence model
# The browser app reads with this one model only (TCN + CTC + word decoder).
LIVE_MODEL_DIR = Path(os.environ.get("ASL_LIVE_MODEL") or MODELS_DIR / "final_model_i")
_live = {}
_live_lock = threading.Lock()


def live_model():
    """(WordRecognizer, WordDecoder), loaded at the first camera start and compiled before any frame
    is read, so the first sign does not freeze. Later connections reuse them."""
    with _live_lock:
        if not _live:
            from asl.seq.inference import WordRecognizer
            from asl.seq.word_decoder import WordDecoder
            if not (LIVE_MODEL_DIR / "model_config.json").exists():
                raise NoModelError(f"The live model is missing: {LIVE_MODEL_DIR}")
            wr = WordRecognizer(LIVE_MODEL_DIR)
            wr.warm_up()
            _live.update(wr=wr, decoder=WordDecoder())
        return _live["wr"], _live["decoder"]


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Live reading with the sequence model.

    Client -> server: binary = 8-byte little-endian float64 capture time in ms, then one JPEG;
                      text = {"type":"command","action":"finish"|"clear"}.
    Server -> client: {"type":"frame", "hand": 21 x [x,y,z] or null, "confirmed", "current", ...}
                      and {"type":"text", "finished_text", "sentences"} after a command.
    One frame at a time: the browser waits for each reply before sending the next frame.
    With ASL_LIVE_REPLAY=<a saved take .npz> and ?replay=1, the server plays that take instead
    of reading the camera (for checks and screenshots without a webcam).
    """
    from asl.seq.live_session import LiveSession

    await ws.accept()
    try:
        wr, decoder = await asyncio.to_thread(live_model)
    except Exception as e:
        await ws.send_json({"type": "error", "fatal": True, "message": str(e)})
        await ws.close()
        return
    session = await asyncio.to_thread(LiveSession, wr, decoder, LIVE_MODEL_DIR.name)
    lock = asyncio.Lock()
    replay = os.environ.get("ASL_LIVE_REPLAY") if ws.query_params.get("replay") else None

    async def play(path):
        with np.load(path, allow_pickle=False) as z:
            pts = [None if np.isnan(p).all() else p for p in z["points"]]
            labels = [str(x) or None for x in z["hand_labels"]]
            ts, size = z["t"] - z["t"][0], (int(z["image_width"]), int(z["image_height"]))
        begin = time.monotonic()
        for t, p, label in zip(ts, pts, labels):
            await asyncio.sleep(max(0.0, begin + float(t) - time.monotonic()))
            async with lock:
                result = await asyncio.to_thread(session.process_points, float(t), p, label, size)
            await ws.send_json(result)

    try:
        player = asyncio.create_task(play(replay)) if replay else None
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None and not replay:
                data = msg["bytes"]
                t = float(np.frombuffer(data[:8], "<f8")[0]) / 1000.0
                async with lock:
                    result = await asyncio.to_thread(session.process_jpeg, data[8:], t)
                await ws.send_json(result)
            elif msg.get("text"):
                action = json.loads(msg["text"]).get("action")
                async with lock:
                    if action == "finish":
                        reply = await asyncio.to_thread(session.finish)
                    elif action == "clear":
                        reply = session.clear()
                    else:
                        reply = {"type": "error", "message": f"Unknown command: {action}"}
                await ws.send_json(reply)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if replay and player:
            player.cancel()
        await asyncio.to_thread(session.close)

# ----------------------------------------------------------------- speech
@app.get("/api/speech/status")
def speech_status():
    return speech_engine.status()


@app.websocket("/ws/speech")
async def ws_speech(ws: WebSocket):
    """Binary: 16 kHz mono int16 PCM chunks. Text {"type":"stop"} -> final."""
    await ws.accept()
    try:
        session = await asyncio.to_thread(speech_engine.recognizer)
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return
    await ws.send_json({"type": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                out = await asyncio.to_thread(session.feed, msg["bytes"])
                await ws.send_json(out)
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                await ws.send_json(await asyncio.to_thread(session.finish))
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# --------------------------------------------------------------- frontend
DIST = FRONTEND_DIST
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if not DIST.exists():
        return HTMLResponse(
            "<h2>Frontend not built</h2><p>Run <code>npm --prefix frontend install</code> and "
            "<code>npm --prefix frontend run build</code>, then reload. Or use the dev server: "
            "<code>npm --prefix frontend run dev</code> and open http://localhost:5173</p>")
    f = (DIST / path).resolve()
    if path and f.is_file() and DIST.resolve() in f.parents and f.name != "index.html":
        return FileResponse(f)
    # no-cache: the browser asks every time, so after a rebuild it loads the new page code
    # (the files in assets/ have new names on every build, so they can stay cached)
    return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
