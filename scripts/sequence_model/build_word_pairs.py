"""
build_word_pairs.py - Make data/word_pairs_en.txt, the "which word follows which" table used by
the word decoder (backend/asl/seq/word_decoder.py).

Source: Peter Norvig's count_2w.txt (https://norvig.com/ngrams/count_2w.txt, 5.5 MB), the most
frequent two-word sequences of the Google Web Trillion Word Corpus with their counts. Download it
yourself, then:

    venv\\Scripts\\python.exe scripts\\sequence_model\\build_word_pairs.py PATH\\TO\\count_2w.txt

What is kept (one "FIRST SECOND count" per line, most frequent first):
  - only pairs where BOTH words are in data/words_en.txt
  - the 100,000 most frequent of them (LOVE YOU, THANK YOU, MY NAME ...)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEEP = 100000


def build(lines, words, keep=KEEP):
    """Lines of 'first second<TAB>count' -> [(FIRST, SECOND, count)], most frequent first.

    words: the allowed words (upper case)."""
    counts = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        a, b = parts[0].upper(), parts[1].upper()
        if a in words and b in words:                 # "See you" and "see you" are added together
            counts[a, b] = counts.get((a, b), 0) + int(parts[2])
    out = sorted(((a, b, c) for (a, b), c in counts.items()), key=lambda row: (-row[2], row[0], row[1]))
    return out[:keep]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="count_2w.txt from norvig.com/ngrams")
    ap.add_argument("--words", default=str(ROOT / "data" / "words_en.txt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "word_pairs_en.txt"))
    args = ap.parse_args(argv)
    words = {line.split()[0] for line in Path(args.words).read_text(encoding="utf-8").splitlines() if line.split()}
    rows = build(Path(args.source).read_text(encoding="utf-8", errors="replace").splitlines(), words)
    if len(rows) < 1000:
        sys.exit("That does not look like count_2w.txt (expected lines of 'first second<TAB>count').")
    Path(args.out).write_text("".join(f"{a} {b} {c}\n" for a, b, c in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} word pairs to {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
