"""
seq_check_bundle.py - Check a sequence-model bundle on THIS laptop.

1. Prints the package versions used in Colab next to the local ones.
2. Loads the model and checks it reproduces the logits Colab saved
   (reference_io.npz) - proves training and laptop environments agree.
3. Measures model + preprocessing time on this CPU (no camera involved;
   camera + MediaPipe time is measured by scripts/sequence_model/seq_record_word.py).

    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_check_bundle.py backend\\models\\seq\\<run>\\best_model
    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_check_bundle.py --untrained   # timing only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from asl.seq.features import (Normalizer, SequenceFeaturizer, NUM_FEATURES,  # noqa: E402
                              preprocessing_dict)
from asl.seq.inference import StreamingRecognizer, WordRecognizer  # noqa: E402
from asl.seq.model import DEFAULT_MODEL_CONFIG, build_tcn_ctc, check_reference_io  # noqa: E402
from asl.seq.train import environment_versions  # noqa: E402


def untrained_recognizer():
    """Random weights + identity normaliser: predictions are meaningless,
    but the amount of computation (and so the timing) is the same."""
    n = Normalizer(np.zeros(NUM_FEATURES), np.ones(NUM_FEATURES))
    return WordRecognizer(model=build_tcn_ctc(), model_cfg=dict(DEFAULT_MODEL_CONFIG),
                          preprocessing=preprocessing_dict(n))


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--untrained", action="store_true")
    args = ap.parse_args()
    if not args.bundle and not args.untrained:
        ap.error("give a bundle folder or --untrained")

    local = environment_versions()
    if args.untrained:
        wr = untrained_recognizer()
        print("Untrained model (timing only).")
    else:
        folder = Path(args.bundle)
        wr = WordRecognizer(folder)
        trained = json.loads((folder / "versions.json").read_text()) if (folder / "versions.json").exists() else {}
        print(f"{'package':<12s} {'Colab (training)':<22s} laptop")
        for k in ("python", "tensorflow", "keras", "numpy"):
            print(f"{k:<12s} {str(trained.get(k, '?')):<22s} {local.get(k)}")
        ref = check_reference_io(folder, wr.model)
        if ref is None:
            print("No reference_io.npz in bundle: cannot check equivalence.")
        else:
            print(f"Reference check on {ref['n']} saved inputs: max difference {ref['max_abs_diff']:.2e} "
                  f"-> {'OK' if ref['ok'] else 'MISMATCH - do not trust this bundle here'}")

    rng = np.random.default_rng(0)
    pts = rng.uniform(0.4, 0.6, (21, 3)).astype(np.float32)
    fz = SequenceFeaturizer(wr.feature_cfg)
    print(f"\nTimings on this computer ({local['platform']}):")
    print(f"  feature step per frame         {timeit(lambda: wr.normalizer.apply(fz.step(pts)), 2000):7.3f} ms")
    for chunk in (1, 4):
        s = StreamingRecognizer(wr, chunk_frames=chunk)
        s.push_features(rng.normal(size=(wr.rf, NUM_FEATURES)).astype(np.float32))
        x = rng.normal(size=(chunk, NUM_FEATURES)).astype(np.float32)
        ms = timeit(lambda: s.push_features(x), 200)
        print(f"  streaming model, {chunk} new frame(s) {ms:7.2f} ms per call "
              f"(context {wr.rf - 1} frames)")
    x3 = rng.normal(size=(int(3 * wr.fps), NUM_FEATURES)).astype(np.float32)
    print(f"  offline, 3 s recording         {timeit(lambda: wr.logits(x3), 20):7.2f} ms")


if __name__ == "__main__":
    main()
