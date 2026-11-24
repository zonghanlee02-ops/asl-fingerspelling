"""
train.py - Explicit CTC training loop with checkpoints that survive Colab
disconnects.

Why a hand-written loop instead of model.fit()?  CTC needs FOUR inputs per
batch (features, their true lengths, labels, label lengths). Writing the
step out makes it obvious which lengths go where:

    logits = model(features)                       (batch, T, 27) raw scores
    loss   = tf.nn.ctc_loss(labels, logits,
                            label_length = true letters per example,
                            logit_length = true frames per example,
                            blank_index  = 0)
    gradients clipped by global norm, then Adam.

Two kinds of saved output (both in the run folder, on Google Drive):
    checkpoints/last/   weights + optimizer state + progress  -> resume training
    best_model/         inference bundle of the best validation CER -> the laptop
"""
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np

from . import data as sdata
from .ctc import greedy_decode, character_error_rate
from .model import save_inference_bundle
from .vocab import decode_ids

DEFAULT_TRAIN_CONFIG = {
    "epochs": 60,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "clip_norm": 1.0,
    "seed": 42,
    "augment": True,
    "patience": 15,            # stop if validation CER has not improved for this many epochs
    "show_examples": 5,
    "log_every": 1,            # print every N-th epoch (best epochs are always printed)
}


def set_seeds(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def environment_versions():
    import tensorflow as tf

    v = {"python": platform.python_version(), "platform": platform.platform(),
         "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "",
         "numpy": np.__version__,
         "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}
    try:
        import keras
        v["keras"] = keras.__version__
    except Exception:
        pass
    for name in ("pandas", "pyarrow", "kaggle"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            pass
    return v


class CTCTrainer:
    def __init__(self, model, learning_rate=1e-3, clip_norm=1.0):
        import tensorflow as tf

        self.tf = tf
        self.model = model
        self.clip_norm = float(clip_norm)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.optimizer.build(model.trainable_variables)
        sig = [tf.TensorSpec([None, None, model.input_shape[-1]], tf.float32),
               tf.TensorSpec([None], tf.int32),
               tf.TensorSpec([None, None], tf.int32),
               tf.TensorSpec([None], tf.int32)]
        self._train_step = tf.function(self._train_step_impl, input_signature=sig)
        self._eval_step = tf.function(self._eval_step_impl, input_signature=sig)
        self._logits = tf.function(lambda x: model(x, training=False), input_signature=sig[:1])

    def ctc_loss(self, logits, x_len, y, y_len):
        """Mean CTC loss per letter (dividing by label length keeps long and
        short words on a similar scale)."""
        tf = self.tf
        loss = tf.nn.ctc_loss(labels=y, logits=logits, label_length=y_len, logit_length=x_len,
                              logits_time_major=False, blank_index=0)
        return tf.reduce_mean(loss / tf.cast(y_len, tf.float32))

    def _train_step_impl(self, x, x_len, y, y_len):
        tf = self.tf
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.ctc_loss(logits, x_len, y, y_len)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, norm = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, norm

    def _eval_step_impl(self, x, x_len, y, y_len):
        logits = self.model(x, training=False)
        return self.ctc_loss(logits, x_len, y, y_len), logits

    def train_step(self, X, x_len, Y, y_len):
        loss, norm = self._train_step(X, x_len, Y, y_len)
        return float(loss), float(norm)

    def logits(self, X):
        return self._logits(X).numpy()

    # ------------------------------------------------------- checkpoints
    def save(self, folder, state):
        """Atomically replace folder with weights + optimizer + state.json."""
        folder = Path(folder)
        tmp = folder.with_name(folder.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.model.save_weights(str(tmp / "model.weights.h5"))
        np.savez(tmp / "optimizer.npz", *[v.numpy() for v in self.optimizer.variables])
        (tmp / "state.json").write_text(json.dumps(state, indent=2))
        old = folder.with_name(folder.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if folder.exists():
            folder.rename(old)
        tmp.rename(folder)
        shutil.rmtree(old, ignore_errors=True)

    def restore(self, folder):
        folder = Path(folder)
        if not (folder / "state.json").exists():
            old = folder.with_name(folder.name + ".old")   # crash between the two renames
            if (old / "state.json").exists():
                folder = old
            else:
                return None
        self.model.load_weights(str(folder / "model.weights.h5"))
        z = np.load(folder / "optimizer.npz")
        values = [z[f"arr_{i}"] for i in range(len(z.files))]
        if len(values) != len(self.optimizer.variables):
            raise ValueError("Optimizer state does not match this model/optimizer")
        for var, val in zip(self.optimizer.variables, values):
            var.assign(val)
        return json.loads((folder / "state.json").read_text())


def predict_examples(trainer, examples, normalizer, batch_size=64):
    """Greedy-decoded text for each example (in the given order) + logits."""
    texts, all_logits = [], []
    for i in range(0, len(examples), batch_size):
        exs = examples[i:i + batch_size]
        X, x_len, _, _ = sdata.pad_batch(exs, [normalizer.apply(e["x"]) for e in exs])
        L = trainer.logits(X)
        for k, n in enumerate(x_len):
            all_logits.append(L[k, :n])
            texts.append(decode_ids(greedy_decode(L[k], n)))
    return texts, all_logits


def validate(trainer, examples, normalizer, batch_size=64):
    losses = []
    for X, x_len, Y, y_len in sdata.iterate_batches(examples, batch_size, normalizer):
        loss, _ = trainer._eval_step(X, x_len, Y, y_len)
        losses.append((float(loss), len(X)))
    hyps, _ = predict_examples(trainer, examples, normalizer, batch_size)
    refs = [e["text"] for e in examples]
    return {"loss": sum(l * n for l, n in losses) / max(1, sum(n for _, n in losses)),
            "cer": character_error_rate(refs, hyps),
            "word_acc": float(np.mean([r == h for r, h in zip(refs, hyps)])) if refs else 0.0,
            "pairs": list(zip(refs, hyps))}


def fit(model, train_examples, val_examples, normalizer, run_dir, preprocessing,
        model_cfg, cfg=None, log=print, versions=None):
    """Train with resume support. Safe to call again after a disconnect:
    it continues from run_dir/checkpoints/last if that exists."""
    cfg = dict(DEFAULT_TRAIN_CONFIG, **(cfg or {}))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))
    set_seeds(cfg["seed"])
    trainer = CTCTrainer(model, cfg["learning_rate"], cfg["clip_norm"])
    last = run_dir / "checkpoints" / "last"
    state = trainer.restore(last) or {"epoch": 0, "step": 0, "best_cer": None,
                                      "best_epoch": None, "history": []}
    if state["epoch"]:
        log(f"Resumed from epoch {state['epoch']} (best val CER {state['best_cer']})")
    aug = (sdata.DEFAULT_AUGMENT if cfg["augment"] is True else cfg["augment"]) or None

    for epoch in range(state["epoch"] + 1, cfg["epochs"] + 1):
        t0 = time.time()
        rng = np.random.default_rng(cfg["seed"] * 1000 + epoch)   # same order after a resume
        losses = []
        for X, x_len, Y, y_len in sdata.iterate_batches(train_examples, cfg["batch_size"],
                                                        normalizer, rng, aug):
            loss, _ = trainer.train_step(X, x_len, Y, y_len)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Loss became {loss} at epoch {epoch}; check data/lengths")
            losses.append(loss)
            state["step"] += 1
        v = validate(trainer, val_examples, normalizer)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": v["loss"],
               "val_cer": v["cer"], "val_word_acc": v["word_acc"], "seconds": round(time.time() - t0, 1)}
        state["history"].append(row)
        improved = state["best_cer"] is None or v["cer"] < state["best_cer"]
        if improved:
            state["best_cer"], state["best_epoch"] = v["cer"], epoch
            save_inference_bundle(run_dir / "best_model", model, model_cfg, preprocessing,
                                  versions=versions,
                                  reference_inputs=[normalizer.apply(e["x"]) for e in val_examples[:3]])
        state["epoch"] = epoch
        trainer.save(last, state)
        if improved or epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"]:
            log(f"epoch {epoch:3d}  train loss {row['train_loss']:.3f}  val loss {v['loss']:.3f}  "
                f"val CER {v['cer']:.3f}  word acc {v['word_acc']:.3f}  ({row['seconds']}s)"
                + ("  * best, saved" if improved else ""))
            for ref, hyp in v["pairs"][:cfg["show_examples"]]:
                log(f"      true {ref:<16s} predicted {hyp}")
        if epoch - state["best_epoch"] >= cfg["patience"]:
            log(f"Stopping: no improvement for {cfg['patience']} epochs")
            break
    (run_dir / "history.json").write_text(json.dumps(state["history"], indent=2))
    return trainer, state
