"""
seq_record_word.py - EXPERIMENTAL: fingerspell words to the webcam and let
the sequence model (TCN-CTC) read them. Separate from the main app.

Try one word at a time:
    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_record_word.py backend\\models\\long_memory_model --mirror-input

Record your own training words (Phase 2A), from a word list:
    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_record_word.py backend\\models\\long_memory_model --mirror-input ^
        --words data\\my_words_train.txt --save backend\\storage\\my_recordings

Keys in the camera window:
    S  start recording        F  finish -> predicted letters are shown, then the words the
                                   word decoder reads in them, if that changed anything
    Q  quit
    with --words also:
    N  skip this word for now (it is asked again next time)
    R  that take was wrong: move the last saved take to "discarded" and sign it again
Options:
    --hand         auto (default) uses MediaPipe's left/right label, as training did.
                   right or left forces the choice, if the mirror check says auto is wrong
    --speed-up X   treat the recording as X times faster (only changes the live prediction;
                   saved recordings keep their real timing)
    --word HELLO   the word you are about to sign: shown as right or wrong and saved.
                   A phrase works too: --word "I LOVE YOU" (needs a model with a space output)
    --words FILE   a word or phrase list (one per line). The window shows which one to sign
                   and at which pace; each good take is saved with its word. Needs --save.
                   Takes already in the --save folder are skipped, so you can stop and
                   continue another day
    --paces LIST   the paces each word is signed at, default slow,normal,quick;
                   veryslow (about one letter per second) is also available
    --stream       read while you sign. With the word decoder on (the default) the window shows
                   words: confirmed words, which no longer change, then [the current guess],
                   which may still change. With --no-correct or --old-correct it shows letters.
                   After F (finish) the whole recording is read once more: that is the final result
    --no-correct   do not run the word decoder (ILOVEYOU -> I LOVE YOU, TENIS -> TENNIS) after
                   each recording. It uses data/words_en.txt, data/word_pairs_en.txt and your
                   own data/my_extra_words.txt; saved recordings always keep the model's letters
    --risk LEVEL   careful, normal (default) or bold: how much the word decoder may change.
                   bold separates and repairs more, and changes more names
    --old-correct  use the older word correction (Phase 3) instead of the word decoder
    --split-words  with --old-correct: also cut unlisted letters into common words
    --mirror-input track on the mirrored image. Use this for models from
                   notebooks/fsboard_tcn_ctc_colab.ipynb, which trained on mirrored
                   video (see also "mirror check" below)
    --save DIR     keep the landmarks of each recording (personal data; DIR
                   should be inside backend/storage/, which git ignores). Each file
                   holds the word, pace, image size, hand and speed-up, so it can be replayed
    --untrained    random model, for testing the camera pipeline and timing only

Mirror check: after each recording it prints the "chirality score" of your
hand next to the training data's, and warns if they are far apart. That usually
means left and right are flipped: use --mirror-input, or set --hand yourself.
"""
import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from asl.seq import vocab  # noqa: E402
from asl.seq.correct import DEFAULT_WORD_LIST, Corrector  # noqa: E402
from asl.seq.word_decoder import RISK_LEVELS, LiveWords, WordDecoder  # noqa: E402
from asl.seq.features import canonical_points, chirality_score  # noqa: E402
from asl.seq.inference import StreamingRecognizer, WordRecognizer  # noqa: E402
from asl.seq.model import has_space_output  # noqa: E402
from asl.tracker import HandTracker  # noqa: E402

MIRROR_WARNING_GAP = 0.7     # single hard words can reach about 0.5 without being flipped
PACES = ("slow", "normal", "quick", "veryslow")
PACE_HINTS = {"slow": "slow and careful", "normal": "your usual pace", "quick": "a bit faster than usual",
              "veryslow": "very slow, about one letter per second"}
DEFAULT_PACES = ("slow", "normal", "quick")
MIN_SECONDS = 0.5            # a take shorter than this is a mistaken key press
MIN_HAND_SHARE = 0.6         # the hand must be found in at least this share of the frames
DISCARDED = "discarded"


def choose_hand(labels):
    """The signing hand of one recording: the label found most (ties: right)."""
    counts = Counter(label for label in labels if label)
    return "right" if counts["right"] >= counts["left"] else "left"


def keep_signing_hand(points, labels, hand):
    """Frames labelled as the other hand count as "no hand", as they did in training."""
    return [p if label == hand else None for p, label in zip(points, labels)]


def speed_times(timestamps, speed):
    """Recording times as if it were signed `speed` times faster, starting at 0."""
    t = np.asarray(timestamps, np.float64)
    return ((t - t[0]) / speed).tolist() if len(t) else []


def positive_number(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return value


def target_word(text):
    """A word, or words separated by single spaces ("I LOVE YOU")."""
    if not vocab.is_compatible(text, allow_spaces=True):
        raise argparse.ArgumentTypeError(f"{text!r} must contain only the letters A-Z, with single spaces "
                                         "between words")
    return vocab.normalise_transcript(text)


def read_word_list(path):
    """Words or phrases from a text file, one per line: A-Z, with single spaces between the
    words of a phrase. Blank lines and # comments are skipped."""
    words = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not vocab.is_compatible(line, allow_spaces=True):
            raise ValueError(f"{path}, line {number}: {line!r} must contain only the letters A-Z, "
                             "with single spaces between words")
        word = vocab.normalise_transcript(line)
        if word in words:
            raise ValueError(f"{path}, line {number}: {word} is listed twice")
        words.append(word)
    if not words:
        raise ValueError(f"{path} contains no words")
    return words


def parse_paces(text):
    paces = [p.strip().lower() for p in str(text).split(",") if p.strip()]
    wrong = [p for p in paces if p not in PACES]
    if not paces or wrong or len(set(paces)) != len(paces):
        raise argparse.ArgumentTypeError(f"use a comma-separated list of {', '.join(PACES)} (each once)")
    return paces


def session_prompts(words, paces):
    """Every (word, pace) take of a session, in signing order."""
    return [(w, p) for w in words for p in paces]


def recorded_takes(folder):
    """How many takes of each (word, pace) are already saved in folder (not in discarded/)."""
    done = Counter()
    for path in Path(folder).glob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as z:
                word = str(z["word"]) if "word" in z.files else ""
                pace = str(z["pace"]) if "pace" in z.files else ""
        except (OSError, ValueError):
            continue
        if word:
            done[(word, pace)] += 1
    return done


def remaining_prompts(prompts, done):
    """The prompts not yet covered by saved takes."""
    left, done = [], Counter(done)
    for prompt in prompts:
        if done[prompt] > 0:
            done[prompt] -= 1
        else:
            left.append(prompt)
    return left


def recording_problem(timestamps, used_points, word, target_fps, flipped=False):
    """Why a take is not good enough to train on, or None if it is fine."""
    n = len(used_points)
    n_hand = sum(p is not None for p in used_points)
    if n_hand == 0:
        return "no hand was found"
    seconds = float(timestamps[-1] - timestamps[0]) if n > 1 else 0.0
    if seconds < MIN_SECONDS:
        return "the recording is too short"
    if n_hand / n < MIN_HAND_SHARE:
        return f"the hand was lost in {1 - n_hand / n:.0%} of the frames: keep it inside the picture"
    needed = vocab.required_frames(vocab.encode(word))
    if seconds * target_fps < needed or n_hand < needed:
        return f"the recording is too short for the letters of {word}"
    if flipped:
        return "left and right look flipped (see the mirror check)"
    return None


def mirror_warning(yours, training):
    """A plain warning if the hand looks flipped compared with training, else None."""
    if training is None or np.isnan(yours) or abs(yours - training) < MIRROR_WARNING_GAP:
        return None
    return (f"  WARNING: left and right look flipped (yours {yours:.2f}, training {training:.2f}), so this "
            "prediction is unreliable. Use --mirror-input and try the other --hand setting.")


def live_status(stream, live_words):
    """The text shown in the window while recording with --stream."""
    if live_words is not None:
        s = live_words.state()
        return s["confirmed"] + (f" [{s['current']}]" if s["current"] else "")
    s = stream.state()
    return s["confirmed"] + (f"[{s['provisional']}]" if s["provisional"] else "")


def feed_live_words(stream, live_words, fed):
    """Give the live word decoder the frame scores the streaming model added since `fed`. Returns the new count."""
    new = stream.frame_logits[fed:]
    if live_words is not None and len(new):
        live_words.push(np.stack(new))
    return len(stream.frame_logits)


def correction_line(fix, target=None):
    """One line saying what the word program did with the model's letters."""
    if hasattr(fix, "nbest"):                              # the word decoder
        if not fix.changed:
            return ("  word decoder: no change (" + ("not in the word list, kept as the model wrote it: " +
                    ", ".join(fix.unknown) if fix.unknown else "in the word list") + ")")
        verdict = "" if not target else f"   target {target}: {'RIGHT' if fix.text == target else 'wrong'}"
        doubt = "" if fix.sure else "   NOT SURE: " + fix.why
        return f"  word decoder: {fix.text!r}{verdict}{doubt}"
    if not fix.changed:
        listed = bool(fix.words) and all(w.reason == "listed word" for w in fix.words)
        return ("  word correction: no change (" +
                ("in the word list" if listed else "not in the word list, kept as the model wrote it") + ")")
    how = ", ".join(sorted({w.reason for w in fix.words if w.chosen != w.raw or w.reason == "joined"}))
    verdict = "" if not target else f"   target {target}: {'RIGHT' if fix.text == target else 'wrong'}"
    return f"  word correction: {fix.text!r} ({how}){verdict}"


def save_recording(folder, timestamps, points, labels, hand, predicted, mirror_input, image_size,
                   speed_up, word=None, pace="", word_list="", hand_mode="auto", corrected=""):
    """One recording, with everything needed to replay, label or train on it later."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.full((21, 3), np.nan, np.float32) if p is None else p for p in points])
    name = (word or "rec").replace(" ", "-")              # I LOVE YOU -> I-LOVE-YOU_slow_...
    stem = "_".join(part for part in (name, pace) if part) + "_" + time.strftime("%Y%m%d_%H%M%S")
    path, n = folder / f"{stem}.npz", 1
    while path.exists():                                   # two takes within one second
        n += 1
        path = folder / f"{stem}_{n}.npz"
    np.savez_compressed(path, t=np.array(timestamps), points=arr, hand=hand, predicted=predicted,
                        mirror_input=mirror_input, word=word or "", hand_labels=np.array([x or "" for x in labels]),
                        image_width=image_size[0], image_height=image_size[1], speed_up=speed_up,
                        pace=pace, word_list=word_list, hand_mode=hand_mode, corrected=corrected)
    return path


def discard_recording(path):
    """Move a take into the discarded/ subfolder (kept, but never used for training)."""
    path = Path(path)
    target = path.parent / DISCARDED / path.name
    target.parent.mkdir(exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto",
                    help="the hand you fingerspell with (auto: MediaPipe's label)")
    ap.add_argument("--speed-up", type=positive_number, default=1.0,
                    help="treat the recording as this many times faster (try 2.5)")
    ap.add_argument("--word", type=target_word, help="the word you are about to sign")
    ap.add_argument("--words", help="a word list to record, one word per line (needs --save)")
    ap.add_argument("--paces", type=parse_paces, default=list(DEFAULT_PACES),
                    help="paces for each word in --words: slow, normal, quick, veryslow "
                         "(default slow,normal,quick)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-correct", action="store_true", help="do not run the word decoder")
    ap.add_argument("--risk", choices=sorted(RISK_LEVELS), default="normal",
                    help="how much the word decoder may change (default normal)")
    ap.add_argument("--old-correct", action="store_true", help="use the older word correction instead")
    ap.add_argument("--split-words", action="store_true",
                    help="word correction may also cut unlisted letters into common words")
    ap.add_argument("--mirror-input", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--untrained", action="store_true")
    args = ap.parse_args()

    queue = None
    if args.words:
        if args.word:
            ap.error("use --word or --words, not both")
        if not args.save:
            ap.error("--words needs --save, the folder your takes are saved in")
        try:
            prompts = session_prompts(read_word_list(args.words), args.paces)
        except (OSError, ValueError) as e:
            ap.error(str(e))
        queue = remaining_prompts(prompts, recorded_takes(args.save))
        print(f"Word list {args.words}: {len(prompts)} takes, {len(prompts) - len(queue)} already saved "
              f"in {args.save}, {len(queue)} to go.")

    if args.untrained:
        from seq_check_bundle import untrained_recognizer
        wr = untrained_recognizer()
    elif args.bundle:
        wr = WordRecognizer(args.bundle)
    else:
        ap.error("give a bundle folder or --untrained")
    print(f"Model compiled in {wr.warm_up():.0f} ms (done now, so your first take does not freeze).")
    corrector = None
    if not args.no_correct and not args.untrained:
        if DEFAULT_WORD_LIST.exists():
            if args.old_correct:
                corrector = Corrector(split_words=args.split_words)
                print(f"Older word correction is on ({len(corrector.words)} words; --no-correct switches it off).")
            else:
                corrector = WordDecoder(risk=args.risk)
                print(f"Word decoder is on, risk {args.risk} ({len(corrector.words)} words, "
                      f"{len(corrector.lexicon.pairs)} word pairs; --no-correct switches it off).")
        else:
            print(f"Word correction is off: {DEFAULT_WORD_LIST} is missing "
                  "(scripts/sequence_model/build_word_list.py makes it).")
    train_chirality = wr.prep.get("chirality_train")
    if "mirrored" in str(wr.prep.get("source", "")) and not args.mirror_input:
        print("WARNING: this model trained on mirrored video. Add --mirror-input, or results will be poor.")
    targets = [args.word] if args.word else [w for w, _ in queue] if queue else []
    if any(" " in w for w in targets) and not has_space_output(wr.model_cfg):
        print("NOTE: this model has no space output, so it reads phrases without spaces and marks them 'wrong'. "
              "Your takes are still saved with the right words for training.")
    if queue is not None and not args.mirror_input:
        print("WARNING: recording training words without --mirror-input. These takes will not match "
              "the training videos, and the packing script will leave them out.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera.")
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read from the camera.")
    image_size = (frame.shape[1], frame.shape[0])          # the model needs its proportions
    tracker = HandTracker(video_mode=True)
    stream, live_words, fed = None, None, 0
    if args.stream:
        stream = StreamingRecognizer(wr, hand="right" if args.hand == "auto" else args.hand, image_size=image_size)
        if isinstance(corrector, WordDecoder):
            live_words = LiveWords(corrector)
            print("Live words are on: confirmed words, then [the current guess].")
    recording, ts, pts, labels, result, frame_ms, shown = False, [], [], [], "", [], ""
    pos, saved_this_session = 0, []                        # word-list progress; (path, pos) of each saved take

    def current():
        if queue is None:
            return args.word, ""
        return queue[pos] if pos < len(queue) else (None, "")

    def announce():
        word, pace = current()
        if queue is not None:
            print(f"\nNext: {word} ({PACE_HINTS[pace]})  [{pos + 1}/{len(queue)}]" if word
                  else "\nAll words in the list are recorded. Press Q to quit.")

    print(f"Camera image {image_size[0]}x{image_size[1]}, speed-up {args.speed_up:g}. S = start, F = finish, Q = quit"
          + ("; N = skip word, R = redo last take" if queue is not None else ""))
    announce()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic()
        t0 = time.perf_counter()
        track_img = cv2.flip(frame, 1) if args.mirror_input else frame
        obs = tracker.process_bgr(track_img)
        p = None if obs is None else obs.points
        live = ""
        if recording:
            label = None if obs is None else obs.handedness.lower()
            if stream:
                if args.hand == "auto" and label and not any(labels):
                    stream.hand = label                    # live mode decides from the first hand seen
                stream.push_frame((t - ts[0]) / args.speed_up if ts else 0.0, p)
                fed = feed_live_words(stream, live_words, fed)
                live = live_status(stream, live_words)
            ts.append(t)
            pts.append(p)
            labels.append(label)
            frame_ms.append((time.perf_counter() - t0) * 1000)

        show = cv2.flip(frame, 1)                          # selfie view for the user
        if p is not None:
            for x, y in p[:, :2]:
                xs = x if args.mirror_input else 1 - x
                cv2.circle(show, (int(xs * show.shape[1]), int(y * show.shape[0])), 3, (0, 255, 0), -1)
        status = f"REC {len(ts)} frames {live}" if recording else f"Result: {shown}   (S start / F finish / Q quit)"
        cv2.putText(show, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        if queue is not None:
            word, pace = current()
            prompt = f"Sign: {word}  ({pace})  {pos + 1}/{len(queue)}" if word else "All done: press Q"
            cv2.putText(show, prompt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Sequence model (experimental)", show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        target, pace = current()
        if key == ord("s") and not recording:
            if queue is not None and target is None:
                print("All words are recorded. Press Q to quit.")
                continue
            recording, ts, pts, labels, frame_ms = True, [], [], [], []
            if stream:
                stream.reset()
                fed = 0
                if live_words is not None:
                    live_words.reset()
        elif key == ord("n") and not recording and queue is not None and target is not None:
            print(f"  skipped {target} ({pace}); it is asked again next time")
            pos += 1
            announce()
        elif key == ord("r") and not recording and queue is not None:
            if not saved_this_session:
                print("  nothing saved in this session to redo")
                continue
            path, pos = saved_this_session.pop()
            moved = discard_recording(path)
            print(f"  moved {moved.name} to {moved.parent}; sign it again")
            announce()
        elif key == ord("f") and recording:
            recording = False
            hand = choose_hand(labels) if args.hand == "auto" else args.hand
            used = keep_signing_hand(pts, labels, hand) if args.hand == "auto" else pts
            t1 = time.perf_counter()
            result, logits = wr.recognize(speed_times(ts, args.speed_up), used, hand=hand, image_size=image_size)
            fix = corrector.correct(logits) if corrector else None
            finish_ms = (time.perf_counter() - t1) * 1000
            corrected = fix.text if fix else ""
            shown = f"{corrected}   (model wrote {result})" if fix and fix.changed else result
            dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
            n_hand = sum(q is not None for q in used)
            verdict = "" if not target else f"   target {target}: {'RIGHT' if result == target else 'wrong'}"
            print(f"\nPredicted: {result!r}{verdict}   ({len(ts)} camera frames in {dur:.1f} s = "
                  f"{len(ts) / max(dur, 1e-6):.1f} fps, hand found in {n_hand}, "
                  f"using {hand} hand, speed-up {args.speed_up:g})")
            if fix:
                print(correction_line(fix, target))
            if frame_ms:
                print(f"  per-frame tracking{' + streaming' if stream else ''}"
                      f"{' + live words' if live_words is not None else ''}: median "
                      f"{np.median(frame_ms):.1f} ms, p95 {np.percentile(frame_ms, 95):.1f} ms, "
                      f"slowest {np.max(frame_ms):.1f} ms; "
                      f"word prediction{' + correction' if fix else ''} after Finish: {finish_ms:.1f} ms")
            if stream:
                print(f"  streaming letters: {stream.finish()['confirmed']!r}")
                if live_words is not None:
                    feed_live_words(stream, live_words, fed)
                    live_words.finish()
                    print(f"  live words at the end: {live_words.text!r} "
                          f"({'same as' if live_words.text == corrected else 'different from'} the final result)")
            valid = [canonical_points(q, hand) for q in used if q is not None]
            warning = None
            if valid:
                yours = chirality_score(np.stack(valid))
                print(f"  mirror check: your chirality {yours:.2f}, "
                      f"training data {train_chirality if train_chirality is not None else 'unknown'}")
                warning = mirror_warning(yours, train_chirality)
                if warning:
                    print(warning)
            if queue is not None:
                problem = recording_problem(ts, used, target, wr.fps, flipped=bool(warning))
                if problem:
                    print(f"  NOT saved: {problem}. Press S to sign {target} again, or N to skip it.")
                    continue
                saved = save_recording(args.save, ts, pts, labels, hand, result, args.mirror_input, image_size,
                                       args.speed_up, target, pace, Path(args.words).stem, args.hand, corrected)
                saved_this_session.append((saved, pos))
                print(f"  saved {saved}")
                pos += 1
                announce()
            elif args.save and pts:
                saved = save_recording(args.save, ts, pts, labels, hand, result, args.mirror_input, image_size,
                                       args.speed_up, args.word, hand_mode=args.hand, corrected=corrected)
                print(f"  saved {saved}")
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
