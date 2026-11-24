"""Stage 2, method 1: the word decoder (asl.seq.word_decoder), its word-pair table and scripts."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from asl.seq import correct as C
from asl.seq import word_decoder as W
from test_seq_correct import frames, spelled

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts" / "sequence_model"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def lexicon():
    words = C.WordList({"THE": 1000, "YOU": 800, "I": 900, "LOVE": 200, "TENNIS": 40, "TEN": 60, "IS": 700,
                        "NOW": 300, "HERE": 300, "NOWHERE": 30, "SCHOOL": 100, "BREAD": 50, "YEW": 5,
                        "BIG": 100, "GOAT": 20, "A": 900})
    return W.Lexicon(words, {("I", "LOVE"): 150, ("LOVE", "YOU"): 150, ("NOW", "HERE"): 10})


def test_with_every_cost_off_the_score_is_the_exact_ctc_probability():
    rng = np.random.default_rng(0)
    tiny = W.Lexicon(C.WordList({"AB": 5, "BA": 3, "A": 2}))
    for _ in range(10):
        logits = rng.normal(size=(6, 27)) * 2
        logits[:, 4:] = -50                                            # only blank, A, B, C can appear
        d = W.WordDecoder(tiny, word_weight=0, pair_weight=0, unknown_word_cost=0, no_space_cost=None,
                          double_cost=None, raw_bonus=0, beam=10000, min_symbol_prob=0).decode(logits, nbest=1000)
        lp = C.log_softmax(logits)
        assert len(d.nbest) > 20 and len({text for text, _ in d.nbest}) == len(d.nbest)
        for text, score in d.nbest:
            assert score == pytest.approx(C.ctc_log_prob(lp, C.encode_text(text)), abs=1e-9)
        assert [s for _, s in d.nbest] == sorted((s for _, s in d.nbest), reverse=True)
        assert not any(c * 3 in text for text, _ in d.nbest for c in "ABC")       # never AAA: see below


def test_words_that_ran_together_are_separated(lexicon):
    d = W.WordDecoder(lexicon).decode(frames(spelled("ILOVEYOU")))
    assert (d.raw, d.text, d.unknown) == ("ILOVEYOU", "I LOVE YOU", [])
    assert d.changed and d.nbest[0] == (d.text, d.score) and d.margin > 0
    # With "never end a word without a space" it stays one unknown word.
    kept = W.WordDecoder(lexicon, no_space_cost=None).decode(frames(spelled("ILOVEYOU")))
    assert (kept.text, kept.unknown, kept.sure) == ("ILOVEYOU", ["ILOVEYOU"], False)


def test_a_space_from_the_model_is_a_hint_not_an_order(lexicon):
    dec = W.WordDecoder(lexicon)
    # The pause decides between NOWHERE and NOW HERE ...
    assert dec.decode(frames(spelled("NOW") + [" ", " "] + spelled("HERE"))).text == "NOW HERE"
    assert dec.decode(frames(spelled("NOWHERE"))).text == "NOWHERE"
    # ... but a doubtful space in the middle of a word, where no word ends, is read as a pause.
    doubtful = {" ": 9.0, "_": 8.0}
    assert dec.decode(frames(spelled("SCH") + [doubtful] + spelled("OOL"))).text == "SCHOOL"
    # A strict decoder (space_as_pause=0) may not ignore a space the model was sure of.
    strict = W.WordDecoder(lexicon, space_as_pause=0.0).decode(frames(spelled("SCH") + [" "] + spelled("OOL")))
    assert strict.text == "SCH OOL" and strict.unknown == ["SCH", "OOL"]


def test_the_second_best_letter_wins_when_only_it_makes_a_word(lexicon):
    unsure = {"U": 9.0, "R": 8.0}                                      # the model's R-or-U doubt
    d = W.WordDecoder(lexicon).decode(frames(spelled("B") + [unsure, "_"] + spelled("EAD")))
    assert (d.raw, d.text) == ("BUEAD", "BREAD")
    sure = W.WordDecoder(lexicon).decode(frames(spelled("BUEAD")))     # no trace of an R: left alone
    assert (sure.text, sure.unknown) == ("BUEAD", ["BUEAD"])


def test_a_double_letter_written_once_is_restored(lexicon):
    dec = W.WordDecoder(lexicon)
    assert dec.decode(frames(spelled("TENIS"))).text == "TENNIS"
    assert dec.decode(frames(spelled("TENNIS"))).text == "TENNIS"      # a real double is left as it is
    off = W.WordDecoder(lexicon, double_cost=None).decode(frames(spelled("TENIS")))
    assert off.text in ("TENIS", "TEN IS")


def test_a_clearly_read_name_is_kept(lexicon):
    d = W.WordDecoder(lexicon).decode(frames(spelled("ZONGHAN")))
    assert (d.text, d.unknown, d.changed) == ("ZONGHAN", ["ZONGHAN"], False)
    assert "not in the word list: ZONGHAN" in d.why


def test_word_pairs_choose_the_likelier_next_word(lexicon):
    last = [{"U": 9.0, "W": 9.0}, "_"]                                 # YOU or YEW, the model cannot tell
    middle = [{"O": 9.0, "E": 9.0}, "_"]
    spec = spelled("LOVE") + [" "] + ["Y", "_"] + middle + last
    assert W.WordDecoder(lexicon).decode(frames(spec)).text == "LOVE YOU"
    assert lexicon.pair_gain("LOVE", "YOU") > 0                        # YOU follows LOVE more often than chance
    assert lexicon.pair_gain("NOW", "YOU") == 0 and lexicon.pair_gain("ZONGHAN", "YOU") == 0
    assert lexicon.pair_gain("NOW", "HERE") == 0                       # in the table, but not more than chance
    assert math.exp(lexicon.log_prob("YOU")) == pytest.approx(800 / lexicon.words.total)
    assert W.Lexicon(lexicon.words, {("LOVE", "YEW"): 200}).pair_gain("LOVE", "YEW") == W.PAIR_GAIN_CAP


def test_the_same_letter_can_end_one_word_and_start_the_next(lexicon):
    assert W.WordDecoder(lexicon).decode(frames(spelled("BIG") + [" "] + spelled("GOAT"))).text == "BIG GOAT"
    assert W.WordDecoder(lexicon).decode(frames(spelled("BIGGOAT"))).text == "BIG GOAT"


def test_older_models_empty_recordings_and_risk_levels(lexicon):
    d = W.WordDecoder(lexicon).decode(frames(spelled("ILOVEYOU"), classes=27))
    assert d.text == "I LOVE YOU"
    empty = W.WordDecoder(lexicon).decode(np.zeros((0, 28)))
    assert (empty.raw, empty.text, empty.changed) == ("", "", False)
    silent = W.WordDecoder(lexicon).decode(frames(["_", "_", "_"]))
    assert (silent.text, silent.unknown) == ("", [])
    assert set(W.RISK_LEVELS) == {"careful", "normal", "bold"}
    normal = W.WordDecoder(lexicon, risk="normal")
    plain = W.WordDecoder(lexicon)
    for name in ("word_weight", "pair_weight", "unknown_word_cost", "no_space_cost", "double_cost",
                 "space_as_pause", "raw_bonus"):
        assert getattr(normal, name) == getattr(plain, name)
    careful, bold = W.WordDecoder(lexicon, risk="careful"), W.WordDecoder(lexicon, risk="bold")
    assert careful.raw_bonus > normal.raw_bonus > bold.raw_bonus       # trust in the model's own letters
    assert bold.unknown_word_cost > normal.unknown_word_cost == careful.unknown_word_cost
    with pytest.raises(ValueError, match="risk must be one of"):
        W.WordDecoder(lexicon, risk="wild")
    assert W.WordDecoder(lexicon).correct(frames(spelled("THE"))).text == "THE"


def test_a_small_beam_still_gives_an_answer(lexicon):
    assert W.WordDecoder(lexicon, beam=1).decode(frames(spelled("THE"))).text == "THE"


def test_build_word_pairs_adds_up_spellings_and_keeps_listed_words_only():
    import build_word_pairs as bwp
    lines = ["love you\t100", "Love You\t50", "love xqzt\t999", "see you\t120", "broken line", "a b c\t5"]
    assert bwp.build(lines, {"LOVE", "YOU", "SEE"}) == [("LOVE", "YOU", 150), ("SEE", "YOU", 120)]
    assert bwp.build(lines, {"LOVE", "YOU", "SEE"}, keep=1) == [("LOVE", "YOU", 150)]


def test_the_shipped_word_pairs_load_and_know_everyday_phrases():
    lex = W.Lexicon.load()
    assert len(lex.pairs) > 50000 and all(a in lex and b in lex for a, b in list(lex.pairs)[:2000])
    for a, b in (("THANK", "YOU"), ("I", "LOVE"), ("GOOD", "MORNING"), ("SEE", "YOU")):
        assert lex.pair_gain(a, b) > 0
    assert lex.pair_gain("TAX", "I") == 0
    assert "" in lex.prefixes and "CHEES" in lex.prefixes and "CHEESX" not in lex.prefixes
    no_pairs = W.Lexicon.load(pair_path=DATA / "no_such_file.txt")
    assert no_pairs.pairs == {} and no_pairs.pair_gain("THANK", "YOU") == 0


def test_no_word_holds_the_same_letter_three_times_and_the_models_own_letters_get_extra_trust(lexicon):
    triple = spelled("BIGGG")                                          # G, blank, G, blank, G
    assert "GGG" not in W.WordDecoder(lexicon).decode(frames(triple)).text
    # Two readings the model finds almost equally likely: the one it wrote itself (greedy) wins.
    tie = spelled("ZON") + [{"G": 9.0, "C": 8.99}, "_"] + spelled("HAN")
    d = W.WordDecoder(lexicon).decode(frames(tie))
    assert (d.raw, d.text) == ("ZONGHAN", "ZONGHAN") and d.nbest[1][0] == "ZONCHAN"
    assert d.margin == pytest.approx(W.RAW_BONUS + 0.01, abs=0.02)


def test_score_script_and_recorder_work_with_the_decoder(lexicon):
    import seq_record_word as rec
    import seq_score_correction as ssc
    rows = [{"word": "I LOVE YOU", "logits": frames(spelled("ILOVEYOU"))},     # fixed
            {"word": "SCHOOL", "logits": frames(spelled("SCHOOL"))},           # right before and after
            {"word": "BREAD", "logits": frames(spelled("ZONGHAN"))}]           # wrong, left alone
    before = dict(lexicon.words.counts)
    s = ssc.score(rows, W.WordDecoder(lexicon))
    assert (s["takes"], s["exact_before"], s["exact_after"], s["fixed"], s["broken"]) == (3, 1, 2, 1, 0)
    assert (s["hidden_checked"], s["hidden_changed"]) == (1, 0)         # SCHOOL, hidden from the list, stays
    assert s["examples"][0][:3] == ("I LOVE YOU", "ILOVEYOU", "I LOVE YOU") and "ahead of" in s["examples"][0][3]
    assert lexicon.words.counts == before

    dec = W.WordDecoder(lexicon)
    fixed = dec.decode(frames(spelled("ILOVEYOU")))
    assert rec.correction_line(fixed, "I LOVE YOU").startswith("  word decoder: 'I LOVE YOU'   target I LOVE YOU: RIGHT")
    assert ("NOT SURE" in rec.correction_line(fixed)) == (not fixed.sure)
    assert rec.correction_line(dec.decode(frames(spelled("SCHOOL")))) == "  word decoder: no change (in the word list)"
    assert rec.correction_line(dec.decode(frames(spelled("ZONGHAN")))) == \
        "  word decoder: no change (not in the word list, kept as the model wrote it: ZONGHAN)"


# ------------------------------------------------------------------ live use
def live_run(lexicon, spec, chunk=1, **kw):
    """Feed fake frame scores to LiveWords `chunk` frames at a time; return it and every state seen.
    The fake recordings are short, so it reads every 2 frames and confirms after 2 agreeing readings."""
    kw = dict(dict(update_every=2, settle_readings=2), **kw)
    live = W.LiveWords(W.WordDecoder(lexicon), **kw)
    x = frames(spec)
    seen = [live.push(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    seen.append(live.finish())
    return live, seen, x


def test_live_words_end_with_the_whole_recording_reading_and_confirmed_words_never_change(lexicon):
    spec = spelled("I") + ["_"] * 6 + spelled("LOVE") + ["_"] * 6 + spelled("YOU")
    for chunk in (1, 3, 7):
        live, seen, x = live_run(lexicon, spec, chunk)
        assert live.text == W.WordDecoder(lexicon).decode(x).text == "I LOVE YOU"
        confirmed = [s["confirmed"] for s in seen]
        assert all(b.startswith(a) for a, b in zip(confirmed, confirmed[1:]))      # only ever grows
        assert seen[-1]["confirmed"] in ("I LOVE", "I")                             # YOU is still being signed
        assert live.frames == len(x) and len(x) // max(chunk, 2) - 1 <= live.readings <= len(x)


def test_a_pause_between_two_letters_never_confirms_half_a_word(lexicon):
    spec = spelled("SCH") + ["_"] * 20 + spelled("OOL")                 # a long pause inside SCHOOL
    live, seen, _ = live_run(lexicon, spec)
    assert live.text == "SCHOOL" and all(s["confirmed"] == "" for s in seen)


def test_the_last_confirmed_word_feeds_the_word_pairs_after_old_frames_are_dropped(lexicon):
    last = [{"U": 9.0, "W": 9.0}, "_"]                                 # YOU or YEW, the model cannot tell
    spec = spelled("I") + ["_"] * 4 + spelled("LOVE") + ["_"] * 4 + ["Y", "_", {"O": 9.0, "E": 9.0}, "_"] + last
    live, _, _ = live_run(lexicon, spec, max_frames=8)                # small, so old frames are dropped
    assert live.dropped >= 1 and len(live.window) < len(frames(spec))
    assert live.text == "I LOVE YOU"


def test_live_words_reset_and_an_empty_push(lexicon):
    live, _, _ = live_run(lexicon, spelled("THE"))
    live.reset()
    assert live.state() == {"confirmed": "", "current": ""} and live.text == "" and live.frames == 0
    assert live.push(np.zeros((0, 28))) == {"confirmed": "", "current": ""}
    assert W._after_letters(["RAINAPLE"], 4) == ["APLE"] and W._after_letters(["RAIN", "APLE"], 4) == ["APLE"]


def test_recorder_shows_live_words_and_feeds_only_new_frames(lexicon):
    import seq_record_word as rec

    class FakeStream:
        def __init__(self):
            self.frame_logits = []

        def state(self):
            return {"confirmed": "LOV", "provisional": "E"}

    stream, live = FakeStream(), W.LiveWords(W.WordDecoder(lexicon))
    x = frames(spelled("LOVE"))
    fed = 0
    for row in x:
        stream.frame_logits.append(row)
        fed = rec.feed_live_words(stream, live, fed)
    live.finish()
    assert fed == len(x) and live.frames == len(x)
    assert rec.live_status(stream, live) == " [LOVE]"
    assert rec.live_status(stream, None) == "LOV[E]"                   # letters when the decoder is off
