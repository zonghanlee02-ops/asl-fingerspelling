"""
correct.py - Phase 3: word correction after the sequence model.

The model writes the best letter at each moment (greedy decoding) and stops there:
it never asks whether CHESE is a word. This module adds that check, using only a
word list and the model's own per-frame scores. Nothing is retrained.

For every stretch of letters between two spaces the model wrote:

  1. LISTED WORD: kept as it is (the model is trusted on real words).
  2. NEAR WORD: otherwise, listed words one or two letters away are compared with
     keeping the letters                                    CHESE -> CHEESE
         fit    how probable that text is given what the model saw in the video
                (the CTC probability, added up over every way of timing the letters)
       + weight * how common the word is (an unlisted word counts as very rare)
       - a cost for every letter the model wrote that the option drops
       + a bonus if the option is the same letters with one letter doubled
         (writing a double letter once is this model's most common mistake)
     The best score wins, so a replacement must be BOTH close to what the model
     saw and a plausible word. A name the model read clearly stays a name.
  3. SPLIT (off unless split_words=True): if no near word wins, but the letters
     can be cut into very common words without changing a letter, they are cut
                                                           ILOVEYOU -> I LOVE YOU
     This step looks at the letters only, so it also cuts some unlisted single
     words (TAXI -> TAX I when TAXI is not listed). Measured on your recordings it
     changed about 6% of rightly read words that were hidden from the list, against
     about 1% for step 2, which is why it is opt-in.
  4. JOIN: two neighbouring stretches are joined when together they are a listed
     word and that scores better than the two apart        OXY GEN -> OXYGEN

Limits: a wrong reading that happens to be a real word (READ for BREAD) is kept;
words outside the list are never produced (add names to data/my_extra_words.txt);
a phrase is only split if every letter in it was read right.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ctc import collapse_doubles, greedy_decode
from .vocab import BLANK, CHAR_TO_ID, LETTERS, SPACE, SPACE_ID, decode_ids

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORD_LIST = ROOT / "data" / "words_en.txt"
DEFAULT_EXTRA_WORDS = ROOT / "data" / "my_extra_words.txt"

# Chosen on the TRAINING-word takes only; the test-word takes gave the final score
# (scripts/sequence_model/seq_score_correction.py prints both).
PRIOR_WEIGHT = 0.2           # how much "a common word" counts against "what the model saw"
UNKNOWN_LOG_PROB = -23.0     # ln probability given to a word outside the list (about 1 in 10 billion)
REMOVED_LETTER_COST = 2.0    # extra doubt for an option that drops a letter the model wrote
DOUBLED_LETTER_BONUS = 1.0   # extra trust in an option that only doubles a letter (TENIS -> TENNIS)
MAX_EDITS = 2
MIN_LETTERS_FOR_TWO_EDITS = 5
MAX_CANDIDATES = 40
SPLIT_TOP_RANK = 3000        # a split may only use the 3,000 most common words
EXTRA_WORD_RANK = 3000       # your own extra words count as common as the word of this rank


# ------------------------------------------------------------------ word list
class WordList:
    """Words with how often they are used. All upper case."""

    def __init__(self, counts, extra=()):
        self.counts = {str(w).upper(): float(c) for w, c in dict(counts).items() if float(c) > 0}
        if not self.counts:
            raise ValueError("The word list is empty.")
        ranked = sorted(self.counts.values(), reverse=True)
        extra_count = ranked[min(EXTRA_WORD_RANK, len(ranked)) - 1]
        for w in extra:
            w = str(w).strip().upper()
            if w and all(c in LETTERS for c in w):
                self.counts[w] = max(self.counts.get(w, 0.0), extra_count)
        self.total = sum(self.counts.values())
        self.longest = max(map(len, self.counts))
        self.rank = {w: i for i, w in enumerate(sorted(self.counts, key=lambda w: -self.counts[w]))}

    @classmethod
    def load(cls, path=DEFAULT_WORD_LIST, extra_path=DEFAULT_EXTRA_WORDS):
        """'WORD count' per line; extra_path holds your own words, one per line (# starts a comment)."""
        counts = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                counts[parts[0]] = float(parts[1])
        extra = []
        if extra_path and Path(extra_path).exists():
            extra = [ln.split("#")[0].strip() for ln in Path(extra_path).read_text(encoding="utf-8").splitlines()]
        return cls(counts, [w for w in extra if w])

    def __contains__(self, word):
        return word in self.counts

    def __len__(self):
        return len(self.counts)

    def log_prob(self, word):
        """Natural log of the word's share of all word uses."""
        return math.log(self.counts[word] / self.total)

    def is_common(self, word, top=SPLIT_TOP_RANK):
        return word in self.counts and self.rank[word] < top


def edits1(word):
    """Every string one letter away: one letter removed, added, changed, or two swapped."""
    parts = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = {a + b[1:] for a, b in parts if b}
    out.update(a + b[1] + b[0] + b[2:] for a, b in parts if len(b) > 1)
    out.update(a + c + b[1:] for a, b in parts if b for c in LETTERS)
    out.update(a + c + b for a, b in parts for c in LETTERS)
    out.discard(word)
    return out


def near_words(word, words, max_edits=MAX_EDITS):
    """{listed word: edits away} for words 1 (or, for longer words, 2) letters from `word`."""
    first = edits1(word)
    found = {w: 1 for w in first if w in words}
    if max_edits >= 2 and len(word) >= MIN_LETTERS_FOR_TWO_EDITS:
        for e in first:
            for w in edits1(e):
                if w in words and w not in found and w != word:
                    found[w] = 2
    return found


def best_split(letters, words, top_rank=SPLIT_TOP_RANK):
    """The most plausible way to cut `letters` into two or more very common words, changing
    no letter; None if there is none."""
    n = len(letters)
    best = [None] * (n + 1)                            # best[i]: (log prob, [words]) for letters[:i]
    best[0] = (0.0, [])
    for end in range(1, n + 1):
        for start in range(max(0, end - words.longest), end):
            piece = letters[start:end]
            if best[start] is not None and words.is_common(piece, top_rank):
                lp = best[start][0] + words.log_prob(piece)
                if best[end] is None or lp > best[end][0]:
                    best[end] = (lp, best[start][1] + [piece])
    return best[n][1] if best[n] is not None and len(best[n][1]) > 1 else None


# ------------------------------------------------------------------ model fit
def log_softmax(logits):
    x = np.asarray(logits, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def ctc_log_prob(log_probs, ids):
    """ln P(label sequence | frames): the CTC forward algorithm, every timing of the letters added up.

    log_probs: (T, C) log-softmax scores. ids: label ids without blanks."""
    lp = np.asarray(log_probs, np.float64)
    ids = [int(i) for i in ids]
    if len(lp) == 0:
        return 0.0 if not ids else -math.inf
    if not ids:
        return float(lp[:, BLANK].sum())
    ext = np.full(2 * len(ids) + 1, BLANK)
    ext[1::2] = ids                                    # blank, l1, blank, l2, ... blank
    may_skip = np.zeros(len(ext), bool)                # jump over a blank: only between DIFFERENT letters
    may_skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    emit = lp[:, ext]
    alpha = np.full(len(ext), -math.inf)
    alpha[0], alpha[1] = emit[0, 0], emit[0, 1]
    for t in range(1, len(lp)):
        stay_or_step = np.logaddexp(alpha, np.concatenate(([-math.inf], alpha[:-1])))
        skip = np.concatenate(([-math.inf, -math.inf], alpha[:-2]))
        skip[~may_skip] = -math.inf
        alpha = np.logaddexp(stay_or_step, skip) + emit[t]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def encode_text(text):
    return [CHAR_TO_ID[c] for c in text]


def letter_stretches(best):
    """Per-frame best symbols -> [(first frame, end frame, letters)] between the spaces the model wrote.

    The stretches and the runs of space frames between them cover every frame once."""
    out, start, ids, prev, in_space = [], 0, [], BLANK, False
    for t, k in enumerate(list(best) + [None]):
        space = k == SPACE_ID
        if (k is None or (space and not in_space)) and ids:
            out.append((start, t, decode_ids(ids)))
            ids = []
        if in_space and not space:
            start = t
        if k is not None and not space and k != BLANK and k != prev:
            ids.append(int(k))
        prev, in_space = k, space
    return out


# ------------------------------------------------------------------ the corrector
@dataclass
class WordChoice:
    raw: str                   # the letters the model wrote for this stretch
    chosen: str                # what we show (contains spaces after a split)
    reason: str                # "listed word" | "unknown word kept" | "corrected" | "split" | "joined"
    margin: float = 0.0        # for "corrected": score of the choice minus score of keeping the letters


@dataclass
class Correction:
    raw: str                   # exactly what the model wrote
    text: str                  # after correction
    words: list = field(default_factory=list)

    @property
    def changed(self):
        return self.text != self.raw


class Corrector:
    def __init__(self, words=None, prior_weight=PRIOR_WEIGHT, unknown_log_prob=UNKNOWN_LOG_PROB,
                 removed_letter_cost=REMOVED_LETTER_COST, doubled_letter_bonus=DOUBLED_LETTER_BONUS,
                 max_edits=MAX_EDITS, split_words=False):
        self.words = words if words is not None else WordList.load()
        self.prior_weight = float(prior_weight)
        self.unknown_log_prob = float(unknown_log_prob)
        self.removed_letter_cost = float(removed_letter_cost)
        self.doubled_letter_bonus = float(doubled_letter_bonus)
        self.max_edits = int(max_edits)
        self.split_words = bool(split_words)

    def score(self, log_probs, text):
        """fit + weight * how common the words are."""
        prior = sum(self.words.log_prob(w) if w in self.words else self.unknown_log_prob for w in text.split(SPACE))
        return ctc_log_prob(log_probs, encode_text(text)) + self.prior_weight * prior

    def choose(self, log_probs, letters):
        """One stretch of letters and its frames -> WordChoice."""
        if letters in self.words:
            return WordChoice(letters, letters, "listed word")
        keep = self.score(log_probs, letters)
        best = WordChoice(letters, letters, "unknown word kept")
        near = near_words(letters, self.words, self.max_edits)
        for word in sorted(near, key=lambda w: (near[w], -self.words.log_prob(w)))[:MAX_CANDIDATES]:
            margin = self.score(log_probs, word) - keep - self.removed_letter_cost * max(0, len(letters) - len(word))
            if len(word) > len(letters) and collapse_doubles(word) == collapse_doubles(letters):
                margin += self.doubled_letter_bonus
            if margin > best.margin:
                best = WordChoice(letters, word, "corrected", margin)
        if best.reason == "unknown word kept" and self.split_words:
            pieces = best_split(letters, self.words)
            if pieces:
                best = WordChoice(letters, SPACE.join(pieces), "split")
        return best

    def correct(self, logits):
        """Per-frame model scores (T, 27 or 28) of one recording -> Correction."""
        logits = np.asarray(logits)
        if logits.ndim != 2 or len(logits) == 0:
            return Correction("", "")
        lp = log_softmax(logits)
        raw = decode_ids(greedy_decode(lp))
        stretches = letter_stretches(lp.argmax(-1))

        # JOIN neighbours that together are a listed word, if that scores better than the two apart.
        i = 0
        while i + 1 < len(stretches):
            (s0, _, a), (_, e1, b) = stretches[i], stretches[i + 1]
            if a + b in self.words and self.score(lp[s0:e1], a + b) > self.score(lp[s0:e1], a + SPACE + b):
                stretches[i:i + 2] = [(s0, e1, a + b)]
            else:
                i += 1

        before = raw.split()
        choices = [self.choose(lp[s:e], letters) for s, e, letters in stretches]
        for c in choices:
            if c.reason == "listed word" and c.raw not in before:
                c.reason = "joined"
        return Correction(raw, SPACE.join(c.chosen for c in choices), choices)
