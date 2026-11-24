"""
vocab.py - Output vocabulary for the CTC model.

    index 0      CTC blank  ("no new letter at this frame"; NOT a space)
    index 1..26  A..Z
    index 27     space between words (added in Phase 2B, vocabulary v2)

The blank never appears in a target transcript. Digits and punctuation are
not in the vocabulary, so transcripts containing them are EXCLUDED (the whole
recording is skipped). We never delete characters from a transcript while
keeping the hand movements that produced them, because the model would then
learn that those movements mean "nothing".

Spaces: a phrase such as "I LOVE YOU" may be used only where the caller asks
for it (allow_spaces=True), and only with single spaces between words. Older
models have 27 outputs (no space); index 0-26 mean the same in both versions,
so those models still load and read exactly as before.
"""
import re
import string

BLANK = 0
LETTERS = string.ascii_uppercase                    # "ABC...Z"
SPACE = " "
VOCAB = ["<blank>"] + list(LETTERS) + [SPACE]       # index -> symbol
NUM_CLASSES = len(VOCAB)                            # 28 outputs per frame
LETTERS_ONLY_CLASSES = 27                           # models made before Phase 2B
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB) if i != BLANK}
SPACE_ID = CHAR_TO_ID[SPACE]

VOCAB_VERSION = "blank0-AZ-space-v2"
_WORD = re.compile(r"^[A-Z]+$")
_PHRASE = re.compile(r"^[A-Z]+( [A-Z]+)*$")         # words separated by single spaces


def normalise_transcript(text):
    """Consistent case: strip surrounding whitespace, upper-case.

    Internal characters are NOT removed; incompatible text stays incompatible.
    """
    return str(text).strip().upper()


def is_compatible(text, allow_spaces=False):
    """True if the normalised transcript uses only A-Z (one complete word), or with
    allow_spaces=True, A-Z words separated by single spaces.
    Non-ASCII text never qualifies, even if upper() would turn it into A-Z."""
    pattern = _PHRASE if allow_spaces else _WORD
    return str(text).isascii() and bool(pattern.match(normalise_transcript(text)))


def encode(text):
    """'HELLO' -> [8, 5, 12, 12, 15]; 'HI YOU' -> [8, 9, 27, 25, 15, 21].
    Raises ValueError on anything else."""
    if not is_compatible(text, allow_spaces=True):
        raise ValueError(f"Transcript not in the A-Z + space vocabulary: {text!r}")
    return [CHAR_TO_ID[c] for c in normalise_transcript(text)]


def decode_ids(ids):
    """[8, 5, 12, 12, 15] -> 'HELLO' (blanks are skipped; 27 is a space)."""
    return "".join(VOCAB[i] for i in ids if i != BLANK)


def required_frames(label_ids):
    """Minimum number of output frames CTC needs to produce this label.

    One frame per symbol, plus one blank between every pair of identical
    neighbouring letters (HELLO needs 6: H E L <blank> L O).
    """
    repeats = sum(1 for a, b in zip(label_ids, label_ids[1:]) if a == b)
    return len(label_ids) + repeats


def is_known_vocabulary(symbols):
    """True for this vocabulary or an older one it extends (the 27-output A-Z version)."""
    symbols = list(symbols)
    return len(symbols) >= LETTERS_ONLY_CLASSES and symbols == VOCAB[:len(symbols)]
