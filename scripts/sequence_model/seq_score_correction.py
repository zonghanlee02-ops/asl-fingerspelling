"""
seq_score_correction.py - Measure the word programs on YOUR recordings, on this laptop.

    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_score_correction.py backend\\models\\new_model

It measures the word decoder (stage 2: asl.seq.word_decoder) by default, and the older word
correction (Phase 3: asl.seq.correct) with --old-correct. --risk careful|normal|bold sets how
much the decoder may change.

For every take in backend\\storage\\my_recordings it compares what the model wrote with what
the word program makes of it, separately for your training words and your test words:

  exact      takes read completely right, before -> after
  fixed      wrong before, right after
  broken     right before, wrong after (should be 0)
  changed    wrong before, a different wrong word after
  hidden     "what if this word were not in the word list?" Every take the model read right
             is corrected again with its own word taken out of the list, as happens with a
             name. The number is how many of those right readings get changed (lower is better).

The settings were chosen on the training words. The test words are the honest score.
Read-only: nothing is saved or changed.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from asl.seq.correct import Corrector, WordList  # noqa: E402
from asl.seq.word_decoder import RISK_LEVELS, WordDecoder  # noqa: E402
from asl.seq.ctc import edit_distance  # noqa: E402

DATA = ROOT / "data"


def read_words(path):
    return [line.strip().upper() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def score(rows, corrector, examples=12):
    """rows: [{'word', 'logits'}] -> summary dict. Pure, so the tests can call it."""
    out = {"takes": len(rows), "exact_before": 0, "exact_after": 0, "fixed": 0, "broken": 0, "changed": 0,
           "hidden_changed": 0, "hidden_checked": 0, "errors_before": 0, "errors_after": 0, "letters": 0,
           "examples": []}
    for r in rows:
        word = r["word"]
        c = corrector.correct(r["logits"])
        before, after = c.raw == word, c.text == word
        out["exact_before"] += before
        out["exact_after"] += after
        out["fixed"] += after and not before
        out["broken"] += before and not after
        out["changed"] += (not before) and (not after) and c.text != c.raw
        out["errors_before"] += edit_distance(c.raw, word)
        out["errors_after"] += edit_distance(c.text, word)
        out["letters"] += len(word)
        if c.changed and len(out["examples"]) < examples:
            why = c.why if hasattr(c, "why") else "; ".join(w.reason for w in c.words if w.chosen != w.raw)
            out["examples"].append((word, c.raw, c.text, why))
        if before and all(w in corrector.words for w in word.split()):
            out["hidden_checked"] += 1
            saved = {w: corrector.words.counts.pop(w) for w in set(word.split())}
            try:
                out["hidden_changed"] += corrector.correct(r["logits"]).text != word
            finally:
                corrector.words.counts.update(saved)
    return out


def show(title, s):
    n = max(s["takes"], 1)
    print(f"\n{title}: {s['takes']} takes")
    print(f"  exact words     {s['exact_before']} -> {s['exact_after']}  "
          f"({s['exact_before'] / n:.1%} -> {s['exact_after'] / n:.1%})")
    print(f"  letters right   {1 - s['errors_before'] / max(s['letters'], 1):.1%} -> "
          f"{1 - s['errors_after'] / max(s['letters'], 1):.1%}")
    print(f"  fixed {s['fixed']}, broken {s['broken']}, changed to another wrong word {s['changed']}")
    print(f"  hidden-word check: {s['hidden_changed']} of {s['hidden_checked']} right readings would be changed")
    for word, raw, text, why in s["examples"]:
        print(f"      {word:14s} {raw!r:18s} -> {text!r:18s} {'RIGHT' if text == word else 'wrong'}  ({why})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="model folder, e.g. backend\\models\\phrase_personal_model")
    ap.add_argument("--recordings", default=str(ROOT / "backend" / "storage" / "my_recordings"))
    ap.add_argument("--old-correct", action="store_true", help="measure the older word correction instead")
    ap.add_argument("--risk", choices=sorted(RISK_LEVELS), default="normal",
                    help="how much the word decoder may change (default normal)")
    ap.add_argument("--split-words", action="store_true",
                    help="with --old-correct: also cut unlisted letters into common words")
    args = ap.parse_args(argv)

    from asl.seq.inference import WordRecognizer
    from seq_pack_my_recordings import TEST_LISTS, load_take
    test_words = {w for name in TEST_LISTS if (DATA / name).exists() for w in read_words(DATA / name)}
    wr = WordRecognizer(args.bundle)
    groups = {"Your training words (settings were chosen here)": [], "Your test words (the honest score)": []}
    for path in sorted(Path(args.recordings).glob("*.npz")):
        tk = load_take(path)
        if not tk["word"]:
            continue
        t = tk["t"] - tk["t"][0]
        _, logits = wr.recognize(t.tolist(), tk["used"], hand=tk["hand"], image_size=tk["image_size"])
        key = list(groups)[1 if tk["word"] in test_words else 0]
        groups[key].append({"word": tk["word"], "logits": logits})
    if args.old_correct:
        corrector = Corrector(WordList.load(), split_words=args.split_words)
        print(f"Older word correction. Word list: {len(corrector.words)} words. Splitting unlisted letters "
              f"into words: {'on' if args.split_words else 'off'}.")
    else:
        corrector = WordDecoder(risk=args.risk)
        print(f"Word decoder, risk {args.risk}. Word list: {len(corrector.words)} words, "
              f"{len(corrector.lexicon.pairs)} word pairs.")
    for title, rows in groups.items():
        show(title, score(rows, corrector))


if __name__ == "__main__":
    main()
