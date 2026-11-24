"""
ctc.py - Turning per-frame scores into letters, and measuring errors.

CTC in one paragraph: at every frame the model gives a score for "blank"
and for each letter. Greedy decoding takes the best symbol per frame, then
  1) merges neighbouring identical symbols  (L L L -> L)
  2) removes blanks.
So "L L <blank> L" -> "LL" (a real double letter) but "L L L" -> "L".
A blank between two identical letters is the ONLY way to produce a double
letter, which is why vocab.required_frames() counts one extra frame for it.

Greedy decoding only; beam search / word lists can come later.
"""
import numpy as np

from .vocab import BLANK, decode_ids


def greedy_decode(scores, length=None):
    """scores: (T, C) logits or probabilities. Returns list of label ids."""
    s = np.asarray(scores)
    if length is not None:
        s = s[:int(length)]
    best = s.argmax(-1)
    out, prev = [], BLANK
    for k in best:
        if k != BLANK and k != prev:
            out.append(int(k))
        prev = k
    return out


def greedy_decode_text(scores, length=None):
    return decode_ids(greedy_decode(scores, length))


class StreamingGreedyDecoder:
    """Greedy CTC decoding fed a few frames at a time.

    It remembers the last frame's best symbol across chunks, so a letter
    whose frames are split over two chunks is emitted once, and
    "L | <blank> L" across a boundary still gives a double letter.
    Each frame is decoded exactly once; the result equals greedy_decode()
    on the whole recording.

    confirmed   letters whose run of frames has ended (a different symbol
                followed). The causal model never revises them.
    provisional the letter currently being emitted (its run is still going
                on); it becomes confirmed as soon as the run ends.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.confirmed = []
        self._prev = BLANK
        self._frames = 0

    @property
    def provisional(self):
        return [] if self._prev == BLANK else [int(self._prev)]

    def push(self, scores):
        """scores: (n, C) for NEW frames only. Returns letters confirmed by this call."""
        new = []
        for k in np.asarray(scores).argmax(-1):
            if k != self._prev and self._prev != BLANK:
                new.append(int(self._prev))           # previous run just ended
            self._prev = k
            self._frames += 1
        self.confirmed.extend(new)
        return new

    def finish(self):
        """End of recording: the provisional letter (if any) becomes confirmed."""
        tail = self.provisional
        self.confirmed.extend(tail)
        self._prev = BLANK
        return tail

    def text(self):
        return decode_ids(self.confirmed), decode_ids(self.provisional)


def edit_distance(a, b):
    """Levenshtein distance between two sequences (insert/delete/substitute = 1)."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_operations(ref, hyp):
    """Counts of (substitutions, insertions, deletions) on one best alignment."""
    ref, hyp = list(ref), list(hyp)
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    sub = ins = dele = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            ins, j = ins + 1, j - 1
        else:
            dele, i = dele + 1, i - 1
    return int(sub), int(ins), int(dele)


def character_error_rate(refs, hyps):
    """Total edits / total reference letters (corpus CER, not an average of ratios)."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    chars = sum(len(r) for r in refs)
    return edits / max(chars, 1)


def has_double_letter(word):
    return any(a == b for a, b in zip(word, word[1:]))


def collapse_doubles(word):
    out = []
    for c in word:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)
