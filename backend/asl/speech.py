"""
speech.py - OPTIONAL local speech-to-text with Vosk (supporting feature).

* Runs fully offline once the model folder is downloaded
  (python scripts/app/download_vosk_model.py).
* If Vosk or its model is missing, status() explains how to fix it and the
  rest of the application keeps working. There is NO silent fallback to any
  cloud transcription service.

Audio format expected: 16 kHz, mono, signed 16-bit little-endian PCM
(the browser's audio worklet converts the microphone to this).
"""
import json
import os
from pathlib import Path

from . import MODELS_DIR

SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = MODELS_DIR / "vosk"


def find_model_dir():
    env = os.environ.get("ASL_VOSK_MODEL")
    if env:
        return Path(env)
    if DEFAULT_MODEL_DIR.exists():
        # Use the first folder that looks like a Vosk model (has am/ or conf/)
        for d in sorted(DEFAULT_MODEL_DIR.iterdir()):
            if d.is_dir() and ((d / "am").exists() or (d / "conf").exists()):
                return d
    return None


class SpeechEngine:
    def __init__(self):
        self._model = None
        self.error = None
        self.model_dir = None

    def status(self):
        try:
            import vosk  # noqa: F401
        except ImportError:
            return {"available": False, "engine": "vosk",
                    "message": "Vosk is not installed. Run: pip install vosk"}
        d = find_model_dir()
        if d is None or not d.exists():
            return {"available": False, "engine": "vosk",
                    "message": "No Vosk English model found. Run: "
                               "python scripts/app/download_vosk_model.py"}
        return {"available": True, "engine": "vosk", "model": d.name,
                "message": f"Local speech recognition ready ({d.name})"}

    def model(self):
        if self._model is None:
            st = self.status()
            if not st["available"]:
                raise RuntimeError(st["message"])
            import vosk
            vosk.SetLogLevel(-1)
            self.model_dir = find_model_dir()
            self._model = vosk.Model(str(self.model_dir))
        return self._model

    def recognizer(self):
        import vosk
        rec = vosk.KaldiRecognizer(self.model(), SAMPLE_RATE)
        rec.SetWords(False)
        return SpeechSession(rec)


class SpeechSession:
    """Feed PCM chunks, get partial and final transcripts."""

    def __init__(self, rec):
        self.rec = rec
        self.final_parts = []

    def feed(self, pcm_bytes):
        if self.rec.AcceptWaveform(pcm_bytes):
            text = json.loads(self.rec.Result()).get("text", "")
            if text:
                self.final_parts.append(text)
            return {"type": "final_part", "text": text, "transcript": self.transcript()}
        partial = json.loads(self.rec.PartialResult()).get("partial", "")
        return {"type": "partial", "text": partial, "transcript": self.transcript()}

    def finish(self):
        text = json.loads(self.rec.FinalResult()).get("text", "")
        if text:
            self.final_parts.append(text)
        return {"type": "final", "transcript": self.transcript()}

    def transcript(self):
        return " ".join(p for p in self.final_parts if p).strip()


def transcribe_wav(path, engine=None):
    """Helper for tests / command line: transcribe a 16 kHz mono WAV file."""
    import wave

    engine = engine or SpeechEngine()
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("WAV must be 16 kHz, mono, 16-bit")
        s = engine.recognizer()
        while True:
            data = w.readframes(4000)
            if not data:
                break
            s.feed(data)
        return s.finish()["transcript"]
