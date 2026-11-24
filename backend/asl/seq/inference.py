"""
inference.py - Using a trained bundle on the laptop.

WordRecognizer      record one whole word (Start ... Finish), then predict.
StreamingRecognizer feed camera frames as they arrive; letters appear while
                    signing. Built on the same model and preprocessing.

Both apply exactly the training preprocessing:
    camera frames (any rate, with timestamps)
      -> aspect correction, if the bundle was trained with it
      -> canonical signing hand (left hands mirrored)
      -> CausalResampler to the bundle's target_fps
      -> SequenceFeaturizer (68 features) -> Normalizer (training mean/std)
      -> causal TCN -> greedy CTC decoding

Streaming details: the model sees `receptive_field` frames, so we keep that
many past frames as context, run the model on context + new frames, and
decode only the outputs of the NEW frames. Each frame is decoded exactly
once and the decoder keeps its state between chunks, so the result is the
same as decoding the whole recording at once (tests/test_seq_model.py).
"""
import time

import numpy as np

from .ctc import StreamingGreedyDecoder, greedy_decode
from .features import (CausalResampler, Normalizer, SequenceFeaturizer, canonical_points,
                       NUM_FEATURES)
from .fsboard_video import isotropic_points
from .model import load_inference_bundle, receptive_field
from .vocab import decode_ids


def prepare_points(points, hand, feature_cfg, image_size=None):
    """One frame's raw landmarks -> the form the model saw in training."""
    if points is None:
        return None
    if feature_cfg.get("aspect_correction"):
        points = isotropic_points(points, *image_size)
    return canonical_points(points, hand)


def frame_features(timestamps, points_list, hand, feature_cfg, image_size=None):
    """Camera frames -> UNnormalised (T, 68) features at feature_cfg's target_fps.

    The single place that turns camera frames into model inputs, used for live
    recognition and for packing your own recordings into training examples."""
    if feature_cfg.get("aspect_correction") and image_size is None:
        raise ValueError("This model was trained with aspect correction, so it needs "
                         "image_size=(width, height) of the camera image.")
    res = CausalResampler(float(feature_cfg["target_fps"]))
    fz = SequenceFeaturizer(feature_cfg)
    rows = []
    for t, p in zip(timestamps, points_list):
        for q in res.push(t, prepare_points(p, hand, feature_cfg, image_size)):
            rows.append(fz.step(q))
    return np.stack(rows) if rows else np.zeros((0, NUM_FEATURES), np.float32)


class WordRecognizer:
    def __init__(self, bundle_dir=None, model=None, model_cfg=None, preprocessing=None):
        if bundle_dir is not None:
            model, model_cfg, preprocessing = load_inference_bundle(bundle_dir)
        self.model = model
        self.model_cfg = model_cfg
        self.prep = preprocessing
        self.feature_cfg = preprocessing["features"]
        self.normalizer = Normalizer.from_dict(preprocessing["normalizer"])
        self.fps = float(self.feature_cfg["target_fps"])
        self.rf = receptive_field(model_cfg)
        import tensorflow as tf
        # A compiled graph that accepts any length: a plain eager model call
        # costs ~50 ms on a laptop CPU, the compiled one a few ms.
        self._predict = tf.function(lambda x: model(x, training=False), input_signature=[
            tf.TensorSpec([None, None, model_cfg["num_features"]], tf.float32)])

    def warm_up(self):
        """Compile the model now (TensorFlow does it on the first call), so the first live take does not freeze.
        Returns the time it took in ms. Inputs of any length use the same compiled graph afterwards."""
        t = time.perf_counter()
        self.logits(np.zeros((max(self.rf, 8), self.model_cfg["num_features"]), np.float32))
        return (time.perf_counter() - t) * 1000

    def logits(self, feats_normalised):
        x = np.asarray(feats_normalised, np.float32)
        if len(x) == 0:
            return np.zeros((0, self.model_cfg["num_classes"]), np.float32)
        return self._predict(x[None]).numpy()[0]

    def check_image_size(self, image_size):
        """A model trained with aspect correction must never get uncorrected input."""
        if self.feature_cfg.get("aspect_correction") and image_size is None:
            raise ValueError("This model was trained with aspect correction, so it needs "
                             "image_size=(width, height) of the camera image.")

    def prepare_points(self, points, hand, image_size=None):
        """One frame's raw landmarks -> the form the model saw in training."""
        return prepare_points(points, hand, self.feature_cfg, image_size)

    def features_from_frames(self, timestamps, points_list, hand="right", image_size=None):
        """Camera frames -> normalised (T, 68) at the training frame rate."""
        self.check_image_size(image_size)
        return self.normalizer.apply(frame_features(timestamps, points_list, hand, self.feature_cfg, image_size))

    def recognize(self, timestamps, points_list, hand="right", image_size=None):
        """Offline: one complete recording -> (text, logits)."""
        logits = self.logits(self.features_from_frames(timestamps, points_list, hand, image_size))
        return decode_ids(greedy_decode(logits)), logits


class StreamingRecognizer:
    def __init__(self, word_recognizer, hand="right", chunk_frames=4, image_size=None):
        word_recognizer.check_image_size(image_size)
        self.wr = word_recognizer
        self.hand = hand
        self.chunk = int(chunk_frames)
        self.image_size = image_size
        self.reset()

    def reset(self):
        self.resampler = CausalResampler(self.wr.fps)
        self.featurizer = SequenceFeaturizer(self.wr.feature_cfg)
        self.decoder = StreamingGreedyDecoder()
        self.context = np.zeros((0, NUM_FEATURES), np.float32)   # already decoded frames
        self.pending = []                                         # not yet decoded
        self.frame_logits = []      # the model's scores for every frame read so far (for the live word decoder)

    def push_frame(self, t, points):
        """One camera frame (points None if no hand). Returns state dict."""
        p = self.wr.prepare_points(points, self.hand, self.image_size)
        for q in self.resampler.push(t, p):
            self.pending.append(self.wr.normalizer.apply(self.featurizer.step(q)))
        if len(self.pending) >= self.chunk:
            self._run()
        return self.state()

    def push_features(self, feats_normalised):
        """Already normalised resampled features (used by the tests)."""
        self.pending.extend(list(np.asarray(feats_normalised, np.float32)))
        if len(self.pending) >= self.chunk:
            self._run()
        return self.state()

    def _run(self):
        new = np.stack(self.pending)
        self.pending = []
        x = np.concatenate([self.context, new])
        logits = self.wr.logits(x)[len(self.context):]
        self.decoder.push(logits)
        self.frame_logits.extend(np.asarray(logits))
        self.context = x[-(self.wr.rf - 1):] if self.wr.rf > 1 else x[:0]
        return logits

    def finish(self):
        if self.pending:
            self._run()
        self.decoder.finish()
        return self.state()

    def state(self):
        confirmed, provisional = self.decoder.text()
        return {"confirmed": confirmed, "provisional": provisional}
