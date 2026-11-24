"""
live_session.py - One live webcam session for the browser app, with the sequence model.

    JPEG + capture time from the browser
      -> mirror (the model trained on mirrored video, like seq_record_word.py --mirror-input)
      -> MediaPipe hand points
      -> StreamingRecognizer (the model, a few frames at a time)
      -> LiveWords (confirmed words + the current guess)
    "finish": the whole sentence is read once more (like F in the recorder); that is the final text.

The same steps as `seq_record_word.py --stream`, so the app reads exactly as the recorder does.
"""
import time
from collections import Counter

import cv2
import numpy as np

from ..tracker import HandTracker
from .inference import StreamingRecognizer
from .word_decoder import LiveWords


class LiveSession:
    def __init__(self, recognizer, decoder, name="", mirror_input=True):
        """recognizer: a loaded WordRecognizer (shared, warmed up); decoder: a shared WordDecoder."""
        self.wr, self.decoder, self.name = recognizer, decoder, name
        self.mirror_input = mirror_input
        self.tracker = HandTracker(video_mode=True)
        self.sentences = []
        self.image_size = None
        self._new_sentence()

    def _new_sentence(self):
        self.stream = None                 # made at the first frame, when the image size is known
        self.live = LiveWords(self.decoder)
        self.fed = 0
        self.t0 = None
        self.times, self.points, self.labels = [], [], []

    def process_jpeg(self, jpeg_bytes, t):
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"type": "error", "message": "Could not decode camera frame."}
        start = time.perf_counter()
        if self.mirror_input:
            frame = cv2.flip(frame, 1)
        obs = self.tracker.process_bgr(frame)
        image_size = (frame.shape[1], frame.shape[0])
        return self.process_points(t, None if obs is None else obs.points,
                                   None if obs is None else obs.handedness.lower(), image_size, start)

    def process_points(self, t, points, label, image_size, start=None):
        """One frame of hand points (None = no hand) at capture time t in seconds."""
        start = start or time.perf_counter()
        if self.stream is None or image_size != self.image_size:
            if self.stream is not None:
                self._new_sentence()       # camera size changed: start the sentence again
            self.image_size = image_size
            self.stream = StreamingRecognizer(self.wr, hand="right", image_size=image_size)
        if label and not any(self.labels):
            self.stream.hand = label       # the first hand seen decides, as in the recorder
        self.t0 = t if self.t0 is None else self.t0
        self.stream.push_frame(t - self.t0, points)
        new = self.stream.frame_logits[self.fed:]
        self.fed = len(self.stream.frame_logits)
        if len(new):
            self.live.push(np.stack(new))
        self.times.append(t - self.t0)
        self.points.append(points)
        self.labels.append(label)
        state = self.live.state()
        hand3d = None
        if points is not None:             # x, y, z in the same units (pixels / width), for the 3D hand
            w, h = image_size
            hand3d = np.round(points[:, :3] * [1.0, h / w, 1.0], 4).tolist()
        return {"type": "frame", "hand": hand3d, "confirmed": state["confirmed"], "current": state["current"],
                "sentences": self.sentences[-5:], "model": self.name,
                "process_ms": round((time.perf_counter() - start) * 1000, 1)}

    def finish(self):
        """Read the whole sentence once more; that is the final text."""
        text = ""
        if any(p is not None for p in self.points):
            counts = Counter(l for l in self.labels if l)
            hand = "right" if counts["right"] >= counts["left"] else "left"
            used = [p if l == hand else None for p, l in zip(self.points, self.labels)]
            _, logits = self.wr.recognize(self.times, used, hand=hand, image_size=self.image_size)
            text = self.decoder.decode(logits).text
        if text:
            self.sentences.append(text)
        self._new_sentence()
        return {"type": "text", "finished_text": text, "sentences": self.sentences[-5:]}

    def clear(self):
        self._new_sentence()
        return {"type": "text", "finished_text": "", "sentences": self.sentences[-5:]}

    def close(self):
        self.tracker.close()
