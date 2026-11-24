"""Phase 3: word correction (asl.seq.correct), its word list, and the scripts around it."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from asl.seq import correct as C
from asl.seq.vocab import BLANK, CHAR_TO_ID

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts" / "sequence_model"
sys.path.insert(0, str(SCRIPTS))


def frames(spec, classes=28, strong=9.0):
    """Fake per-frame model scores. Each item is one frame: a symbol the model is sure of
    ('_' is the blank, ' ' the space), or {symbol: score} for a frame it is unsure about."""
    rows = []
    for item in spec:
        row = np.zeros(classes, np.float32)
        for symbol, value in (item.items() if isinstance(item, dict) else [(item, strong)]):
            row[BLANK if symbol == "_" else CHAR_TO_ID[symbol]] = value
        rows.append(row)
    return np.stack(rows)


def spelled(text):
    """Clear frames for a text: every letter, then a blank."""
    return [s for c in text for s in (c, "_")]


@pytest.fixture
def words():
    return C.WordList({"THE": 1000, "YOU": 800, "I": 900, "THESE": 300, "LOVE": 200, "CHEESE": 50,
                       "TENNIS": 40, "TEN": 60, "IS": 700, "OXYGEN": 20, "OXY": 2, "GEN": 2, "CHASE": 30})


# ------------------------------------------------------------------ CTC probability
def test_ctc_log_prob_equals_adding_up_every_timing_by_hand():
    rng = np.random.default_rng(0)
    lp = C.log_softmax(rng.normal(size=(4, 3)))                      # 4 frames; blank, A, B
    totals = {}
    for path in itertools.product(range(3), repeat=4):
        label, prev = [], BLANK
        for k in path:
            if k != BLANK and k != prev:
                label.append(k)
            prev = k
        totals[tuple(label)] = totals.get(tuple(label), 0.0) + math.exp(sum(lp[t, k] for t, k in enumerate(path)))
    for label, p in totals.items():
        assert math.exp(C.ctc_log_prob(lp, list(label))) == pytest.approx(p, rel=1e-9)
    assert sum(totals.values()) == pytest.approx(1.0)
    assert C.ctc_log_prob(lp, [1, 2, 1, 2, 1]) == -math.inf           # 5 letters cannot fit in 4 frames
    assert C.ctc_log_prob(lp[:0], []) == 0.0 and C.ctc_log_prob(lp[:0], [1]) == -math.inf


def test_a_double_letter_needs_a_blank_between_its_two_letters():
    without = C.log_softmax(frames(["L", "L", "L"]))
    with_gap = C.log_softmax(frames(["L", "_", "L"]))
    double = [CHAR_TO_ID["L"]] * 2
    assert C.ctc_log_prob(with_gap, double) > -0.1
    assert C.ctc_log_prob(without, double) < -8


# ------------------------------------------------------------------ word list helpers
def test_near_words_finds_listed_words_one_or_two_letters_away(words):
    assert C.near_words("CHESE", words) == {"CHEESE": 1, "THESE": 1, "CHASE": 1}
    assert "TENNIS" in C.near_words("TENIS", words) and C.near_words("TENNIS", words).get("TENNIS") is None
    assert C.near_words("TEIS", words) == {}                           # 2 edits are only tried for 5+ letters
    assert C.near_words("TENXIX", words) == {"TENNIS": 2}


def test_best_split_uses_only_common_words_and_changes_no_letter(words):
    assert C.best_split("ILOVEYOU", words) == ["I", "LOVE", "YOU"]
    assert C.best_split("TENIS", words) == ["TEN", "IS"]
    assert C.best_split("LOVE", words) is None                         # one word is not a split
    assert C.best_split("ILOVEYO", words) is None
    assert C.best_split("OXYGEN", words, top_rank=10) is None          # OXY and GEN are too rare to split on
    assert C.best_split("ILOVEYOU", words, top_rank=3) is None


def test_letter_stretches_cover_the_frames_between_the_spaces():
    best = frames(["_", "H", "H", "_", "I", " ", " ", "_", "Y", "O", "_", "O"]).argmax(-1)
    assert C.letter_stretches(best) == [(0, 5, "HI"), (7, 12, "YOO")]
    assert C.letter_stretches(frames(["_", " ", "_"]).argmax(-1)) == []


def test_word_list_file_and_your_extra_words(tmp_path):
    (tmp_path / "w.txt").write_text("THE 1000\nCAT 10\nbroken line here\n", encoding="utf-8")
    (tmp_path / "x.txt").write_text("# my names\nzonghan  # me\nR2D2\n\n", encoding="utf-8")
    wl = C.WordList.load(tmp_path / "w.txt", tmp_path / "x.txt")
    assert "THE" in wl and "ZONGHAN" in wl and "R2D2" not in wl and len(wl) == 3
    assert wl.log_prob("ZONGHAN") >= wl.log_prob("CAT") and wl.is_common("ZONGHAN")
    assert "ZONGHAN" not in C.WordList.load(tmp_path / "w.txt", tmp_path / "missing.txt")
    with pytest.raises(ValueError):
        C.WordList({})


def test_build_word_list_keeps_real_words_only():
    import build_word_list
    lines = ["the\t100", "a\t90", "b\t80", "pm\t70", "of\t60", "it's\t50", "x9\t40", "cat\t30", "junk", "dog\t20"]
    assert build_word_list.build(lines) == [("THE", 100), ("A", 90), ("OF", 60), ("CAT", 30), ("DOG", 20)]
    assert build_word_list.build(lines, keep=3) == [("THE", 100), ("A", 90), ("OF", 60)]


def test_the_shipped_word_list_knows_the_words_you_sign():
    if not C.DEFAULT_WORD_LIST.exists():
        pytest.skip("data/words_en.txt not built (scripts/sequence_model/build_word_list.py)")
    wl = C.WordList.load()
    assert len(wl) >= 29000
    assert {w for w in wl.counts if len(w) == 1} == {"A", "I"}
    for name in ("my_words_train.txt", "my_words_test.txt", "my_doubles_train.txt", "my_phrases_train.txt",
                 "my_phrases_test.txt"):
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                assert all(w in wl for w in line.split()), line


# ------------------------------------------------------------------ the corrector
def test_a_collapsed_double_letter_is_restored_when_the_model_half_saw_it(words):
    unsure = frames(spelled("CH") + ["E", {"E": 9.0, "_": 8.0}, "E", "_"] + spelled("SE"))
    fix = C.Corrector(words).correct(unsure)
    assert (fix.raw, fix.text, fix.changed) == ("CHESE", "CHEESE", True)
    assert fix.words[0].reason == "corrected" and fix.words[0].margin > 0

    sure = frames(spelled("CH") + ["E", "E", "E", "_"] + spelled("SE"))     # no sign of a second E at all
    kept = C.Corrector(words).correct(sure)
    assert (kept.text, kept.changed, kept.words[0].reason) == ("CHESE", False, "unknown word kept")


def test_listed_words_and_clearly_read_names_are_left_alone(words):
    assert C.Corrector(words).correct(frames(spelled("THESE"))).words[0].reason == "listed word"
    name = C.Corrector(words).correct(frames(spelled("ZONGHAN")))
    assert name.text == "ZONGHAN" and not name.changed


def test_dropping_a_letter_the_model_wrote_costs_extra(words):
    logits = frames(spelled("YOU") + [{"R": 9.0, "_": 8.5}])
    assert C.Corrector(words, removed_letter_cost=0.0).correct(logits).text == "YOU"
    assert C.Corrector(words, removed_letter_cost=50.0).correct(logits).text == "YOUR"


def test_an_option_that_only_doubles_a_letter_gets_a_bonus(words):
    faint = frames(spelled("TE") + ["N", {"N": 9.0, "_": 4.5}, "N", "_"] + spelled("IS"))   # a faint second N
    assert C.Corrector(words, doubled_letter_bonus=0.0).correct(faint).text == "TENIS"
    assert C.Corrector(words, doubled_letter_bonus=3.0).correct(faint).text == "TENNIS"
    other = frames(spelled("CHESE"))                                   # CHASE is near too, but doubles nothing
    assert C.Corrector(words, doubled_letter_bonus=3.0, prior_weight=0.0).correct(other).text == "CHESE"


def test_splitting_is_off_by_default_and_never_beats_a_near_word(words):
    run_together = frames(spelled("ILOVEYOU"))
    assert C.Corrector(words).correct(run_together).text == "ILOVEYOU"
    split = C.Corrector(words, split_words=True).correct(run_together)
    assert split.text == "I LOVE YOU" and split.words[0].reason == "split" and split.raw == "ILOVEYOU"

    tenis = frames(spelled("TE") + ["N", {"N": 9.0, "_": 8.5}, "N", "_"] + spelled("IS"))
    assert C.Corrector(words, split_words=True).correct(tenis).text == "TENNIS"      # not TEN IS


def test_each_word_between_the_models_spaces_is_corrected_on_its_own_frames(words):
    phrase = frames(spelled("CH") + ["E", {"E": 9.0, "_": 8.0}, "E", "_"] + spelled("SE") + [" ", " "]
                    + spelled("YOU"))
    fix = C.Corrector(words).correct(phrase)
    assert (fix.raw, fix.text) == ("CHESE YOU", "CHEESE YOU")
    assert [w.reason for w in fix.words] == ["corrected", "listed word"]


def test_a_doubtful_space_inside_a_listed_word_is_removed(words):
    doubtful = frames(spelled("OXY") + [{" ": 9.0, "_": 8.5}] + spelled("GEN"))
    fix = C.Corrector(words).correct(doubtful)
    assert (fix.raw, fix.text, fix.words[0].reason) == ("OXY GEN", "OXYGEN", "joined")
    certain = frames(spelled("OXY") + [" ", " ", " "] + spelled("GEN"))
    assert C.Corrector(words).correct(certain).text == "OXY GEN"


def test_older_models_without_a_space_output_and_empty_recordings(words):
    old = frames(spelled("CH") + ["E", {"E": 9.0, "_": 8.0}, "E", "_"] + spelled("SE"), classes=27)
    assert C.Corrector(words).correct(old).text == "CHEESE"
    empty = C.Corrector(words).correct(np.zeros((0, 28), np.float32))
    assert (empty.raw, empty.text, empty.changed) == ("", "", False)
    nothing = C.Corrector(words).correct(frames(["_", "_", "_"]))
    assert (nothing.text, nothing.words) == ("", [])


# ------------------------------------------------------------------ scripts
def test_score_script_counts_fixes_and_restores_the_word_list(words):
    import seq_score_correction as ssc
    half_seen = spelled("CH") + ["E", {"E": 9.0, "_": 8.0}, "E", "_"] + spelled("SE")
    rows = [{"word": "CHEESE", "logits": frames(half_seen)},           # fixed
            {"word": "THESE", "logits": frames(spelled("THESE"))},     # right before and after
            {"word": "BREAD", "logits": frames(spelled("ZONGHAN"))}]   # wrong, left alone
    before = dict(words.counts)
    s = ssc.score(rows, C.Corrector(words))
    assert (s["takes"], s["exact_before"], s["exact_after"], s["fixed"], s["broken"], s["changed"]) == (3, 1, 2, 1, 0, 0)
    assert (s["hidden_checked"], s["hidden_changed"]) == (1, 0)        # THESE, hidden from the list, stays THESE
    assert s["errors_after"] < s["errors_before"] and s["examples"][0][:3] == ("CHEESE", "CHESE", "CHEESE")
    assert words.counts == before


def test_recorder_reports_and_saves_the_corrected_word(words, tmp_path):
    import seq_record_word as rec
    half_seen = spelled("CH") + ["E", {"E": 9.0, "_": 8.0}, "E", "_"] + spelled("SE")
    fixed = C.Corrector(words).correct(frames(half_seen))
    assert rec.correction_line(fixed, "CHEESE") == "  word correction: 'CHEESE' (corrected)   target CHEESE: RIGHT"
    assert rec.correction_line(fixed) == "  word correction: 'CHEESE' (corrected)"
    assert "in the word list" in rec.correction_line(C.Corrector(words).correct(frames(spelled("THESE"))))
    assert "kept as the model wrote it" in rec.correction_line(C.Corrector(words).correct(frames(spelled("ZONGHAN"))))
    joined = C.Corrector(words).correct(frames(spelled("OXY") + [{" ": 9.0, "_": 8.5}] + spelled("GEN")))
    assert rec.correction_line(joined) == "  word correction: 'OXYGEN' (joined)"

    path = rec.save_recording(tmp_path, [0.0, 0.1], [None, np.zeros((21, 3), np.float32)], [None, "Right"], "right",
                              "CHESE", True, (640, 480), 1.0, "CHEESE", corrected="CHEESE")
    with np.load(path, allow_pickle=False) as z:
        assert str(z["predicted"]) == "CHESE" and str(z["corrected"]) == "CHEESE"   # the model's letters are kept
