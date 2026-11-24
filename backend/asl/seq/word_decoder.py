"""
word_decoder.py - Stage 2, method 1: read WORDS from the model's frame scores (word beam search).

correct.py repairs the letters after the model has chosen them. This decoder chooses the
letters and the words together, in one pass over the frames, so it can also

  - find where one word ends and the next begins when the model wrote no space
                                                            ILOVEYOU -> I LOVE YOU
  - ignore a space the model wrote in the middle of a word  SCH OOL  -> SCHOOL
  - prefer a letter the model found second-best when only that letter makes a word
  - restore a double letter the model wrote once            TENIS    -> TENNIS

How it works. It follows many possible sentences at once (the "beam"). At every frame each
sentence may
    stay          a blank frame, or the same letter still being held
    add a letter  any letter may be added; a word that leaves the word list pays a cost once
    end the word  at a space the model wrote (free), or without one (small cost)
and at the end of a word that word is scored by how common it is, and it gains when it is a
known companion of the word before it (word pairs: YOU follows LOVE far more often than
chance). Only the best `beam` sentences are kept after each frame. The score of a sentence is

      what the model saw (the CTC probability, every timing added up)
    + word_weight * ln P(word)                        for every listed word
    + pair_weight * (how many times more often than chance the word follows the one before, as ln)
    - unknown_word_cost                               for every word outside the list
    - no_space_cost                                   for every word end the model did not mark
    - double_cost                                     for every double letter restored
    + raw_bonus                                       if it is exactly what the model itself wrote

No word may hold the same letter three times in a row (EGGG): no English word or name does.

"Space as a hint": the model's space output makes ending a word free, but a sentence may also
treat those frames as a pause inside a word (space_as_pause, 0 = never, 1 = freely).

Live use: LiveWords (below) runs this decoder again and again while you sign and confirms each
word once a pause has followed it.

Limits: it cannot produce a letter the model gave almost no probability; NOWHERE against NOW HERE depends on your pause; a name is kept only if
the model read it clearly enough to beat the unknown-word cost.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .correct import DEFAULT_EXTRA_WORDS, DEFAULT_WORD_LIST, ROOT, WordList, log_softmax
from .ctc import greedy_decode
from .vocab import BLANK, CHAR_TO_ID, LETTERS, SPACE, SPACE_ID, decode_ids

DEFAULT_WORD_PAIRS = ROOT / "data" / "word_pairs_en.txt"

# Chosen on TRAINING material only: your training-word takes, two of them played back to back
# (random pairs, and pairs that often go together), and your training phrases spelled without a
# pause. Test words and test phrases gave the final score (docs/SEQUENCE_MODEL.md has the table).
# All costs are in natural-log units.
WORD_WEIGHT = 0.1            # how much "a common word / a likely next word" counts against what the model saw
PAIR_WEIGHT = 1.0            # how much "these two words often go together" counts (LOVE YOU, THANK YOU)
PAIR_GAIN_CAP = 5.0          # a pair never counts as more than e^5 = 150 times more likely than chance
UNKNOWN_WORD_COST = 3.0      # paid once by every word outside the list
UNKNOWN_LETTER_COST = 0.0    # extra cost per letter of such a word
NO_SPACE_COST = 1.0          # ending a word where the model wrote no space
DOUBLE_COST = 1.0            # restoring a double letter the model wrote once
SPACE_AS_PAUSE = 0.1         # 0..1: how freely a space the model wrote may be read as a pause inside a word
RAW_BONUS = 1.0              # extra trust in exactly the letters and spaces the model wrote
BEAM = 40                    # sentences kept after each frame
MIN_SYMBOL_PROB = 1e-3       # letters below this probability at a frame are not tried
SURE_MARGIN = 2.0            # best minus second-best score from which the answer counts as "sure"

RISK_LEVELS = {}             # filled below: "careful" | "normal" | "bold" -> settings


class Lexicon:
    """The word list, every beginning of a word, and the word-pair counts."""

    def __init__(self, words, pairs=None):
        self.words = words
        self.pairs = dict(pairs or {})
        self.prefixes = {""}
        for w in words.counts:
            for i in range(1, len(w) + 1):
                self.prefixes.add(w[:i])

    @classmethod
    def load(cls, word_path=DEFAULT_WORD_LIST, extra_path=DEFAULT_EXTRA_WORDS, pair_path=DEFAULT_WORD_PAIRS):
        """pair_path: 'FIRST SECOND count' per line; a missing file means no word pairs are used."""
        from pathlib import Path
        words = WordList.load(word_path, extra_path)
        pairs = {}
        if pair_path and Path(pair_path).exists():
            for line in Path(pair_path).read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[0] in words and parts[1] in words:
                    pairs[parts[0], parts[1]] = float(parts[2])
        return cls(words, pairs)

    def __contains__(self, word):
        return word in self.words.counts

    def log_prob(self, word):
        """ln of the word's share of all word uses. `word` must be listed."""
        return math.log(self.words.counts[word] / self.words.total)

    def pair_gain(self, before, word):
        """ln of how many times more often than chance `word` follows `before`; 0 if not more often,
        if the pair is not in the table, or if either word is unlisted. Never above PAIR_GAIN_CAP."""
        counts = self.words.counts
        together = self.pairs.get((before, word))
        if not together or before not in counts or word not in counts:
            return 0.0
        gain = math.log(together / counts[before]) - self.log_prob(word)
        return min(max(gain, 0.0), PAIR_GAIN_CAP)


@dataclass
class Decoded:
    raw: str                                   # exactly what the model wrote (greedy letters)
    text: str                                  # the decoder's best sentence
    score: float = 0.0                         # its score (ln)
    margin: float = 0.0                        # best score minus the second-best sentence's score
    unknown: list = field(default_factory=list)   # words of `text` that are not in the word list
    nbest: list = field(default_factory=list)     # [(sentence, score)], best first

    @property
    def changed(self):
        return self.text != self.raw

    @property
    def sure(self):
        return not self.unknown and self.margin >= SURE_MARGIN

    @property
    def why(self):
        notes = [f"ahead of {self.nbest[1][0]!r} by {self.margin:.1f}" if len(self.nbest) > 1 else "only option"]
        if self.unknown:
            notes.append("not in the word list: " + ", ".join(self.unknown))
        return "; ".join(notes)


class WordDecoder:
    def __init__(self, lexicon=None, word_weight=WORD_WEIGHT, pair_weight=PAIR_WEIGHT,
                 unknown_word_cost=UNKNOWN_WORD_COST,
                 unknown_letter_cost=UNKNOWN_LETTER_COST, no_space_cost=NO_SPACE_COST, double_cost=DOUBLE_COST,
                 space_as_pause=SPACE_AS_PAUSE, raw_bonus=RAW_BONUS, beam=BEAM, min_symbol_prob=MIN_SYMBOL_PROB,
                 risk=None):
        self.lexicon = lexicon if lexicon is not None else Lexicon.load()
        self.words = self.lexicon.words                    # the WordList, as on Corrector
        if risk is not None:
            if risk not in RISK_LEVELS:
                raise ValueError(f"risk must be one of {', '.join(RISK_LEVELS)}, not {risk!r}")
            chosen = RISK_LEVELS[risk]
            word_weight, pair_weight = chosen["word_weight"], chosen["pair_weight"]
            unknown_word_cost, no_space_cost = chosen["unknown_word_cost"], chosen["no_space_cost"]
            double_cost, space_as_pause, raw_bonus = chosen["double_cost"], chosen["space_as_pause"], chosen["raw_bonus"]
        self.word_weight = float(word_weight)
        self.pair_weight = float(pair_weight)
        self.unknown_word_cost = float(unknown_word_cost)
        self.unknown_letter_cost = float(unknown_letter_cost)
        self.no_space_cost = None if no_space_cost is None else float(no_space_cost)   # None: never
        self.double_cost = None if double_cost is None else float(double_cost)         # None: never
        self.space_as_pause = min(max(float(space_as_pause), 0.0), 1.0)
        self.raw_bonus = float(raw_bonus)
        self.beam = max(1, int(beam))
        self.min_symbol_prob = float(min_symbol_prob)

    # ------------------------------------------------------------ the costs
    def _grow(self, part, new):
        """Factor for a word growing from `part` to `new` (one more letter)."""
        if new in self.lexicon.prefixes:
            return 1.0
        if part in self.lexicon.prefixes:                  # it leaves the word list here: pay once
            return math.exp(-self.unknown_word_cost - self.unknown_letter_cost * len(new))
        return math.exp(-self.unknown_letter_cost)

    def _listed(self, before, word):
        """Factor for ending a listed word: how common it is, and how well it follows `before`."""
        lex = self.lexicon
        gain = lex.pair_gain(before, word) if before is not None and self.pair_weight else 0.0
        return math.exp(self.word_weight * lex.log_prob(word) + self.pair_weight * gain)

    def _endings(self, before, part, cache):
        """Ways to end the word being written: [(word, factor)]."""
        key = (before, part)
        if key in cache:
            return cache[key]
        lex = self.lexicon
        if part in lex:
            out = [(part, self._listed(before, part))]
        elif part in lex.prefixes:                         # the beginning of a listed word, used as a word
            out = [(part, math.exp(-self.unknown_word_cost - self.unknown_letter_cost * len(part)))]
        else:
            out = [(part, 1.0)]                            # already paid when it left the word list
        doubled = part + part[-1]
        if self.double_cost is not None and part[-2:-1] != part[-1] and doubled in lex:
            out.append((doubled, self._listed(before, doubled) * math.exp(-self.double_cost)))
        cache[key] = out
        return out

    # ------------------------------------------------------------ the search
    def decode(self, logits, nbest=5, before=None):
        """Per-frame model scores (T, 27 or 28) of one recording -> Decoded.

        before: the word signed just before these frames (live use), so the word-pair table can
        judge the first word too. It is not part of the result."""
        logits = np.asarray(logits)
        if logits.ndim != 2 or len(logits) == 0:
            return Decoded("", "")
        start = (before,) if before else ()
        lp = log_softmax(logits)
        raw = decode_ids(greedy_decode(lp))
        probs = np.exp(lp)
        has_space = probs.shape[1] > SPACE_ID
        letter_probs = probs[:, 1:len(LETTERS) + 1]
        floor = self.min_symbol_prob
        no_space = 0.0 if self.no_space_cost is None else math.exp(-self.no_space_cost)
        double = 0.0 if self.double_cost is None else math.exp(-self.double_cost)
        prefixes = self.lexicon.prefixes
        endings_cache = {}

        # A sentence in progress: (finished words, letters of the word being written).
        # Its two numbers: probability that it ends in a blank or pause, and that it ends in its last letter.
        beams = {(start, ""): [1.0, 0.0]}
        log_scale = 0.0
        for t in range(len(probs)):
            p_blank = float(probs[t, BLANK])
            p_space = float(probs[t, SPACE_ID]) if has_space else 0.0
            row = letter_probs[t]
            letters = [(LETTERS[i], float(row[i])) for i in np.flatnonzero(row >= floor)] if floor > 0 else \
                [(LETTERS[i], float(row[i])) for i in range(len(LETTERS)) if row[i] > 0]
            new = {}

            def add(key, blank_part, letter_part):
                slot = new.get(key)
                if slot is None:
                    new[key] = [blank_part, letter_part]
                else:
                    slot[0] += blank_part
                    slot[1] += letter_part

            for key, (pb, pnb) in beams.items():
                words, part = key
                total = pb + pnb
                last = part[-1] if part else ""
                # 1. nothing new: a blank frame. A space frame too, between words or as a pause in a word.
                add(key, total * (p_blank + (p_space if not part else self.space_as_pause * p_space)), 0.0)
                # 2. the last letter is still being held
                if last and pnb:
                    add(key, 0.0, pnb * float(row[CHAR_TO_ID[last] - 1]))
                endings = self._endings(words[-1] if words else None, part, endings_cache) if part else ()
                # 3. the model's space ends the word
                if part and p_space > 0 and p_space >= floor:
                    for word, factor in endings:
                        add((words + (word,), ""), total * p_space * factor, 0.0)
                for c, pc in letters:
                    base = pb if c == last else total           # the same letter again needs a blank between
                    if c == last and part[-2:-1] == last:
                        base = 0.0                              # never the same letter three times in a row
                    if base:
                        # 4. one more letter of the same word
                        add((words, part + c), 0.0, base * pc * self._grow(part, part + c))
                        # 5. the word ends here without a space, and c starts the next one
                        if part and no_space:
                            for word, factor in endings:
                                add((words + (word,), c), 0.0, base * pc * factor * no_space * self._grow("", c))
                    # 6. the model wrote a double letter once: restore it, then add c
                    if double and last and c != last and part[-2:-1] != last and part + last + c in prefixes:
                        add((words, part + last + c), 0.0, total * pc * double)

            kept = sorted(new.items(), key=lambda item: -(item[1][0] + item[1][1]))[:self.beam]
            top = kept[0][1][0] + kept[0][1][1]
            if top <= 0:
                return Decoded(raw, raw)
            log_scale += math.log(top)
            beams = {key: [pb / top, pnb / top] for key, (pb, pnb) in kept}

        final = {}
        for (words, part), (pb, pnb) in beams.items():
            total = pb + pnb
            if not part:
                final[words] = final.get(words, 0.0) + total
                continue
            for word, factor in self._endings(words[-1] if words else None, part, endings_cache):
                final[words + (word,)] = final.get(words + (word,), 0.0) + total * factor
        as_written = start + tuple(raw.split())
        if as_written in final:
            final[as_written] *= math.exp(self.raw_bonus)
        ranked = sorted(((SPACE.join(words[len(start):]), math.log(v) + log_scale)
                         for words, v in final.items() if v > 0), key=lambda item: -item[1])
        if not ranked:
            return Decoded(raw, raw)
        text, score = ranked[0]
        margin = score - ranked[1][1] if len(ranked) > 1 else math.inf
        return Decoded(raw, text, score, margin, [w for w in text.split() if w not in self.lexicon], ranked[:nbest])

    def correct(self, logits):
        """The same call as Corrector.correct, so the scoring script and the recorder can use either."""
        return self.decode(logits)


# ------------------------------------------------------------------ live use
LIVE_UPDATE_EVERY = 4        # model frames between two readings (4 frames = 0.13 s at 30 frames a second)
LIVE_SETTLE_READINGS = 6     # a word is confirmed after this many readings in a row agree on it (0.8 s)
LIVE_MAX_FRAMES = 300        # never read more than this many unconfirmed frames (10 s)
LIVE_CUT_TRIES = 6           # pauses tried when dropping the frames of confirmed words


def _quiet_runs(x):
    """[(first frame, end frame)] of runs where the model's best symbol is the blank or the space."""
    quiet = list(np.isin(np.asarray(x).argmax(-1), (BLANK, SPACE_ID))) + [False]
    runs, start = [], None
    for t, q in enumerate(quiet):
        if q and start is None:
            start = t
        elif not q and start is not None:
            runs.append((start, t))
            start = None
    return runs


def _after_letters(words, skip):
    """The words left after skipping `skip` letters: (['RAINAPLE'], 4) -> ['APLE']."""
    out = []
    for w in words:
        if skip >= len(w):
            skip -= len(w)
            continue
        out.append(w[skip:])
        skip = 0
    return out


class LiveWords:
    """The word decoder while you sign.

    Feed it the model's frame scores as they arrive (push). Every LIVE_UPDATE_EVERY frames it reads
    all unconfirmed frames again and shows
        confirmed  words that are final on screen: they never change again
        current    the best guess for the rest; this may still change (LOVEYO -> LOVE YOU)
    A word is confirmed when the readings have agreed on it, and on every word before it, for
    LIVE_SETTLE_READINGS readings in a row while a further word was already being signed. A pause
    between two letters can never confirm half a word on its own: the next word must have started.

    So long signing stays fast, once more than LIVE_MAX_FRAMES frames are unconfirmed the frames of
    confirmed words are dropped (at a point where reading the two parts separately gives the same
    words), and the last confirmed word is passed on to the word-pair table.

    After Finish, decode the whole recording once more: that is the final answer. Measured on your
    saved recordings, see docs/SEQUENCE_MODEL.md."""

    def __init__(self, decoder, update_every=LIVE_UPDATE_EVERY, settle_readings=LIVE_SETTLE_READINGS,
                 max_frames=LIVE_MAX_FRAMES):
        self.decoder = decoder
        self.update_every = max(1, int(update_every))
        self.settle_readings = max(1, int(settle_readings))
        self.max_frames = max(4 * self.update_every, int(max_frames))
        self.reset()

    def reset(self):
        self.confirmed = []          # final words
        self.current = ""            # the guess for everything after them
        self.window = []             # frame scores still being read
        self.dropped = 0             # confirmed words whose frames left the window
        self.frames = 0              # frames pushed in all
        self.readings = 0
        self._waiting = 0            # frames pushed since the last reading
        self._agree = []             # for each word position: readings in a row with the same word there
        self._last = []              # the words of the previous reading

    @property
    def text(self):
        return SPACE.join(self.confirmed + ([self.current] if self.current else []))

    def state(self):
        return {"confirmed": SPACE.join(self.confirmed), "current": self.current}

    def push(self, rows):
        """New frame scores (n, 27 or 28). Returns state()."""
        rows = np.asarray(rows)
        if rows.ndim == 2 and len(rows):
            self.window.extend(rows)
            self.frames += len(rows)
            self._waiting += len(rows)
            if self._waiting >= self.update_every:
                self._read()
        return self.state()

    def finish(self):
        """Read whatever arrived since the last reading. Returns state()."""
        if self._waiting:
            self._read()
        return self.state()

    def _before(self):
        return self.confirmed[self.dropped - 1] if self.dropped else None

    def _read(self):
        self._waiting = 0
        self.readings += 1
        words = self.decoder.decode(np.stack(self.window), before=self._before()).text.split()
        kept = self.confirmed[self.dropped:]                 # confirmed words still in the window
        if words[:len(kept)] != kept:
            # The reading no longer agrees with words already on screen. They stay (they are final);
            # after them, show the reading's letters that come after the confirmed letters.
            words = kept + _after_letters(words, sum(map(len, kept)))
        same = 0
        while same < min(len(words), len(self._last)) and words[same] == self._last[same]:
            same += 1
        self._agree = [n + 1 for n in self._agree[:same]] + [1] * (len(words) - same)
        self._last = words
        n = len(kept)
        while n < len(words) - 1 and self._agree[n] >= self.settle_readings:   # never the word being signed
            n += 1
        self.confirmed.extend(words[len(kept):n])
        self.current = SPACE.join(words[n:])
        if len(self.window) > self.max_frames:
            self._drop_confirmed_frames()

    def _drop_confirmed_frames(self):
        """Drop the frames of confirmed words, cutting where the two parts read the same as the whole."""
        x = np.stack(self.window)
        kept = self.confirmed[self.dropped:]
        if not kept:
            return
        runs = [(a + b) // 2 for a, b in _quiet_runs(x) if 0 < a and b < len(x) - self.update_every]
        for cut in runs[::-1][:LIVE_CUT_TRIES]:              # the latest few pauses only, so it stays fast
            head = self.decoder.decode(x[:cut], before=self._before()).text.split()
            if head == kept:
                self.window = self.window[cut:]
                self.dropped = len(self.confirmed)
                self._agree = self._agree[len(kept):]
                self._last = self._last[len(kept):]
                return


# One knob instead of six. "normal" equals the defaults above; the table in docs/SEQUENCE_MODEL.md
# shows what each level gained and broke on your recordings.
_NORMAL = {"word_weight": WORD_WEIGHT, "pair_weight": PAIR_WEIGHT, "unknown_word_cost": UNKNOWN_WORD_COST,
           "no_space_cost": NO_SPACE_COST, "double_cost": DOUBLE_COST, "space_as_pause": SPACE_AS_PAUSE,
           "raw_bonus": RAW_BONUS}
RISK_LEVELS.update({
    "careful": dict(_NORMAL, no_space_cost=2.0, raw_bonus=1.5),     # changes less, keeps more names
    "normal": dict(_NORMAL),
    "bold": dict(_NORMAL, unknown_word_cost=4.0, no_space_cost=2.0, raw_bonus=0.5),   # repairs and separates more
})
