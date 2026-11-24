"""
build_word_list.py - Make data/words_en.txt, the word list used by word correction (Phase 3).

Source: Peter Norvig's count_1w.txt (https://norvig.com/ngrams/count_1w.txt, 5 MB), the 333,333
most frequent words of the Google Web Trillion Word Corpus with their counts. Download it
yourself, then:

    venv\\Scripts\\python.exe scripts\\sequence_model\\build_word_list.py PATH\\TO\\count_1w.txt

What is kept (one "WORD count" per line, most frequent first):
  - only words made of the letters A-Z
  - the 30,000 most frequent of them
  - one-letter words: only A and I
  - two-letter words: only real English ones (the web list is full of abbreviations such as
    PM, CD, TX, which would let any letter jumble be "split into words")
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEEP = 30000
ONE_LETTER = {"A", "I"}
TWO_LETTER = set("AM AN AS AT BE BY DO GO HE HI IF IN IS IT ME MY NO OF OH OK ON OR SO TO UP US WE".split())


def build(lines, keep=KEEP):
    """Lines of 'word<TAB>count' (most frequent first) -> [(WORD, count)]."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[a-z]+", parts[0]) or not parts[1].isdigit():
            continue
        word = parts[0].upper()
        if len(out) < keep:
            if (len(word) == 1 and word not in ONE_LETTER) or (len(word) == 2 and word not in TWO_LETTER):
                continue
            out.append((word, int(parts[1])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="count_1w.txt from norvig.com/ngrams")
    ap.add_argument("--out", default=str(ROOT / "data" / "words_en.txt"))
    args = ap.parse_args(argv)
    rows = build(Path(args.source).read_text(encoding="utf-8").splitlines())
    if len(rows) < 1000:
        sys.exit("That does not look like count_1w.txt (expected lines of 'word<TAB>count').")
    Path(args.out).write_text("".join(f"{w} {c}\n" for w, c in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} words to {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
