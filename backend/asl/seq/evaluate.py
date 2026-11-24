"""
evaluate.py - Evaluation report for the sequence model.

All numbers come from the split you pass in (normally the TEST signers, who
were never used for training or model selection). They describe THIS
dataset subset only and must not be compared directly with numbers from
other datasets, other subsets, or the baseline's single-photo accuracy.

Report contents
    cer, word_acc              character error rate, exact word accuracy
    errors                     substitutions / insertions / deletions
    double_letters             words with a repeated letter (LL in HELLO):
                               accuracy, and how often the double was
                               collapsed to one letter
    by_speed                   slow / normal / fast, measured as frames per
                               letter (edges = training-split tertiles)
    per_signer                 CER of each unfamiliar signer
    idle                       letters emitted on inputs with NO hand at all
                               (a proxy only; real idle / transition
                               behaviour needs local recordings)
"""
import json
from pathlib import Path

import numpy as np

from .ctc import (character_error_rate, edit_operations, greedy_decode,
                  has_double_letter, collapse_doubles)
from .features import NUM_FEATURES
from .vocab import decode_ids


def speed_edges(train_examples):
    fpl = [len(e["x"]) / len(e["y"]) for e in train_examples]
    return [float(np.percentile(fpl, 33.3)), float(np.percentile(fpl, 66.7))]


def _summary(refs, hyps):
    if not refs:
        return {"n": 0}
    s = i = d = 0
    for r, h in zip(refs, hyps):
        a, b, c = edit_operations(r, h)
        s, i, d = s + a, i + b, d + c
    return {"n": len(refs), "cer": round(character_error_rate(refs, hyps), 4),
            "word_acc": round(float(np.mean([r == h for r, h in zip(refs, hyps)])), 4),
            "substitutions": s, "insertions": i, "deletions": d}


def idle_insertions(trainer, normalizer, seconds=(1, 3, 10), fps=30.0):
    """Letters the model emits for 'no hand visible' input of a few lengths."""
    out = {}
    for s in seconds:
        x = normalizer.apply(np.zeros((int(s * fps), NUM_FEATURES), np.float32))
        logits = trainer.logits(x[None])[0]
        out[f"{s}s_no_hand"] = decode_ids(greedy_decode(logits))
    return out


def evaluate(trainer, examples, normalizer, train_examples, split_name="test",
             fps=30.0, n_samples=30):
    from .train import predict_examples

    hyps, _ = predict_examples(trainer, examples, normalizer)
    refs = [e["text"] for e in examples]
    report = {"split": split_name, "overall": _summary(refs, hyps)}

    dbl = [k for k, r in enumerate(refs) if has_double_letter(r)]
    report["double_letters"] = dict(
        _summary([refs[k] for k in dbl], [hyps[k] for k in dbl]),
        collapsed_to_single=int(sum(hyps[k] == collapse_doubles(refs[k]) for k in dbl)))

    edges = speed_edges(train_examples)
    buckets = {"slow": [], "normal": [], "fast": []}
    for k, e in enumerate(examples):
        fpl = len(e["x"]) / len(e["y"])            # more frames per letter = slower signing
        name = "fast" if fpl < edges[0] else "normal" if fpl < edges[1] else "slow"
        buckets[name].append(k)
    report["by_speed"] = {"frames_per_letter_edges": edges,
                          **{n: _summary([refs[k] for k in ix], [hyps[k] for k in ix])
                             for n, ix in buckets.items()}}

    signers = sorted({e["signer"] for e in examples})
    report["per_signer"] = {str(s): _summary(*zip(*[(refs[k], hyps[k]) for k, e in enumerate(examples)
                                                    if e["signer"] == s]))
                            for s in signers}
    report["idle"] = idle_insertions(trainer, normalizer, fps=fps)
    report["samples"] = [{"true": r, "predicted": h} for r, h in zip(refs[:n_samples], hyps[:n_samples])]
    report["note"] = ("Measured on a pilot subset of Google's ASL Fingerspelling dataset, "
                      "signer-independent split. Not comparable to other datasets. "
                      "Laptop latency and real idle/transition behaviour are measured separately.")
    return report


def save_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2))


def print_report(report):
    o = report["overall"]
    print(f"[{report['split']}] {o['n']} recordings  CER {o.get('cer')}  word accuracy {o.get('word_acc')}")
    print(f"   errors: {o.get('substitutions')} substituted, {o.get('insertions')} inserted, "
          f"{o.get('deletions')} deleted letters")
    d = report["double_letters"]
    print(f"   double-letter words: {d['n']}  word accuracy {d.get('word_acc')}  "
          f"collapsed to single letter {d['collapsed_to_single']}")
    for n in ("slow", "normal", "fast"):
        b = report["by_speed"][n]
        print(f"   {n:<6s}: {b['n']} recordings  CER {b.get('cer')}  word acc {b.get('word_acc')}")
    print(f"   letters emitted with no hand visible: {report['idle']}")
