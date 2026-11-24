"""
model.py - The compact causal TCN with a CTC output, and its saved bundle.

TCN = temporal convolution network: small filters slide along TIME. Each
filter looks at the current frame and a few PAST frames ("causal": never
future ones), so the model can run live without waiting for the end.
Dilation spaces those few frames out (1, 2, 4, 8 frames apart), so four
blocks together see 61 frames (about 2 s at 30 fps) at a low cost.

    features (batch, T, 68)
      -> Dense 64 per frame (projection) + LayerNorm + ReLU
      -> 4 residual blocks, dilation 1, 2, 4, 8:
           [causal Conv1D(64, k=3) -> LayerNorm -> ReLU -> Dropout] x 2  + skip
      -> Dense 27 per frame = raw scores (logits): blank + A..Z
    output (batch, T, 27)   - one output per input frame (no downsampling)

LayerNormalization normalises the 64 channels of each frame separately, so
it never mixes time steps (keeps the model causal and padding-safe).
The settings below are a first experiment, not tuned optima.
"""
import json
from pathlib import Path

import numpy as np

from .features import NUM_FEATURES
from .vocab import NUM_CLASSES, SPACE_ID, is_known_vocabulary

DEFAULT_MODEL_CONFIG = {
    "architecture": "causal-tcn-ctc-v1",
    "num_features": NUM_FEATURES,
    "num_classes": NUM_CLASSES,
    "channels": 64,
    "kernel_size": 3,
    "dilations": [1, 2, 4, 8],
    "convs_per_block": 2,
    "dropout": 0.1,
}


def receptive_field(cfg=None):
    """Number of input frames (current + past) that one output can see."""
    c = dict(DEFAULT_MODEL_CONFIG, **(cfg or {}))
    return 1 + c["convs_per_block"] * (c["kernel_size"] - 1) * sum(c["dilations"])


def build_tcn_ctc(cfg=None):
    from tensorflow import keras
    from tensorflow.keras import layers

    c = dict(DEFAULT_MODEL_CONFIG, **(cfg or {}))
    inp = keras.Input(shape=(None, c["num_features"]), name="features")
    x = layers.Dense(c["channels"], name="proj")(inp)
    x = layers.LayerNormalization(name="proj_norm")(x)
    x = layers.ReLU(name="proj_relu")(x)
    for b, d in enumerate(c["dilations"]):
        h = x
        for j in range(c["convs_per_block"]):
            h = layers.Conv1D(c["channels"], c["kernel_size"], dilation_rate=d,
                              padding="causal", name=f"b{b}_conv{j}")(h)
            h = layers.LayerNormalization(name=f"b{b}_norm{j}")(h)
            h = layers.ReLU(name=f"b{b}_relu{j}")(h)
            h = layers.Dropout(c["dropout"], name=f"b{b}_drop{j}")(h)
        x = layers.Add(name=f"b{b}_residual")([x, h])
    logits = layers.Dense(c["num_classes"], name="logits")(x)
    return keras.Model(inp, logits, name="tcn_ctc")


# ------------------------------------------------------------- bundle I/O
# An inference bundle is a small folder that the laptop needs:
#   model_config.json      architecture settings (rebuilt with build_tcn_ctc)
#   model.weights.h5       trained weights only (robust across Keras 3 versions)
#   preprocessing.json     feature settings + normaliser mean/std
#   vocab.json             index -> symbol
#   reference_io.npz       a few inputs and the logits Colab produced for them,
#                          to check the laptop reproduces the same numbers
#   versions.json          package versions used for training
#   evaluation_report.json (added by evaluate.py)

def save_inference_bundle(folder, model, model_cfg, preprocessing, versions=None,
                          reference_inputs=None):
    from .vocab import VOCAB, VOCAB_VERSION

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(folder / "model.weights.h5"))
    cfg = dict(DEFAULT_MODEL_CONFIG, **model_cfg)
    (folder / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (folder / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
    n = int(cfg["num_classes"])                   # a 27-output model keeps the A-Z vocabulary
    (folder / "vocab.json").write_text(json.dumps({
        "version": VOCAB_VERSION if n == len(VOCAB) else "blank0-AZ-v1",
        "symbols": VOCAB[:n], "blank_index": 0}, indent=2))
    if versions:
        (folder / "versions.json").write_text(json.dumps(versions, indent=2))
    if reference_inputs is not None:
        xs = [np.asarray(x, np.float32) for x in reference_inputs]
        logits = [model(x[None], training=False).numpy()[0] for x in xs]
        np.savez_compressed(folder / "reference_io.npz",
                            **{f"x{i}": x for i, x in enumerate(xs)},
                            **{f"y{i}": y for i, y in enumerate(logits)})
    return folder


def load_inference_bundle(folder):
    """Returns (model, model_cfg, preprocessing dict)."""
    folder = Path(folder)
    model_cfg = json.loads((folder / "model_config.json").read_text())
    prep = json.loads((folder / "preprocessing.json").read_text())
    vocab = json.loads((folder / "vocab.json").read_text())
    if (vocab.get("blank_index") != 0 or len(vocab["symbols"]) != model_cfg["num_classes"]
            or not is_known_vocabulary(vocab["symbols"])):
        raise ValueError("vocab.json does not match this code's vocabulary (blank=0, A-Z, optional space)")
    model = build_tcn_ctc(model_cfg)
    model.load_weights(str(folder / "model.weights.h5"))
    return model, model_cfg, prep


def has_space_output(model_cfg):
    """Models made before Phase 2B have 27 outputs and can never write a space."""
    return int(model_cfg["num_classes"]) > SPACE_ID


def warm_start(model, source_bundle):
    """Copy a trained bundle's weights into a new model with the same layers.

    Every layer must match, except the output layer may have MORE outputs (27 -> 28,
    adding the space): its existing outputs are copied and the new ones keep their
    fresh starting values. So the new model starts out reading letters as well as
    the source model did, and only has to learn the space.
    Returns {"copied": [...], "extended": [...]} layer names. Raises ValueError if the
    layers differ (for example different dilations)."""
    source, source_cfg, _ = load_inference_bundle(source_bundle)
    by_name = {layer.name: layer for layer in source.layers}
    copied, extended = [], []
    for layer in model.layers:
        new = layer.get_weights()
        if not new:
            continue
        if layer.name not in by_name:
            raise ValueError(f"The source model has no layer {layer.name!r}: the two models are built "
                             f"differently (source dilations {source_cfg['dilations']}).")
        old = by_name[layer.name].get_weights()
        if [w.shape for w in old] == [w.shape for w in new]:
            layer.set_weights(old)
            copied.append(layer.name)
        elif (layer.name == "logits" and len(old) == len(new)
              and all(o.shape[:-1] == n.shape[:-1] and o.shape[-1] <= n.shape[-1] for o, n in zip(old, new))):
            for o, n in zip(old, new):
                n[..., :o.shape[-1]] = o
            layer.set_weights(new)
            extended.append(layer.name)
        else:
            raise ValueError(f"Layer {layer.name!r} has different sizes in the two models "
                             f"({[w.shape for w in old]} vs {[w.shape for w in new]}).")
    return {"copied": copied, "extended": extended}


def check_reference_io(folder, model, atol=1e-3):
    """Max |difference| between saved reference logits and this machine's logits."""
    path = Path(folder) / "reference_io.npz"
    if not path.exists():
        return None
    z = np.load(path)
    worst = 0.0
    i = 0
    while f"x{i}" in z:
        y = model(z[f"x{i}"][None], training=False).numpy()[0]
        worst = max(worst, float(np.max(np.abs(y - z[f"y{i}"]))))
        i += 1
    return {"max_abs_diff": worst, "ok": worst <= atol, "n": i}
