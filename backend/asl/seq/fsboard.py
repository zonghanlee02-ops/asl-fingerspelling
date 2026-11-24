"""
fsboard.py - Google's FSboard dataset (Kaggle: googleai/fsboard).

Chooses recordings from the metadata, downloads single files, and turns
landmark records into compact training examples. Used by
notebooks/fsboard_prepare_data_colab.ipynb.

What the real files look like (checked on real data, see docs/SEQUENCE_MODEL.md)

Metadata   daun_v3/metadata/daun_v3-train.json: a JSON list, one object per
           clip, with clipFilename, phrase, signerId, clipFileMd5, ...
Videos     daun_v3/video_clips/daun_v3-train/<clipFilename>
Landmarks  daun_v3/landmarks/daun_v3-train.arrow-NNNNN-of-00100: an Arrow file
           with one "serialized" column; each row is a tf.train.SequenceExample
           whose context holds video_name, prompt, signer_id, num_frames, ...

Two surprises the code handles:

1. Names differ. A record says  ..._619.011_625.644.mp4  (seconds) where the
   metadata says  ..._619011_625644.mp4  (milliseconds). They are matched with
   an explicit rule, then the phrase and signer are compared as well. Records
   are never paired by row position.

2. Hand coordinates are not what the names suggest. x_right_hand, y_right_hand
   and z_right_hand each hold 21 numbers, but they are three overlapping
   windows of one flattened list:
       x list = x0 y0 z0 x1 y1 z1 ... x6 y6 z6
       y list =    y0 z0 x1 y1 z1 ... x6 y6 z6 x7
       z list =       z0 x1 y1 z1 ... x6 y6 z6 x7 y7
   So only 23 different numbers exist per hand per frame: landmarks 0-6
   complete, and x and y of landmark 7. The index fingertip and the middle,
   ring and little fingers are not stored. Layout is detected from the numbers
   of every frame, so a corrected dataset (21 true x, y and z values) would be
   read as complete hands without changing this code.
"""
import base64
import csv
import hashlib
import html
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from . import vocab

BASE_URL = "https://www.kaggle.com/api/v1/datasets/download/googleai/fsboard/"
DATASET_PAGE = "https://www.kaggle.com/datasets/googleai/fsboard"
TRAIN_METADATA = "daun_v3/metadata/daun_v3-train.json"
TRAIN_VIDEO_DIR = "daun_v3/video_clips/daun_v3-train/"
NUM_TRAIN_SHARDS = 100
FORMAT_VERSION = "fsboard-npz-v1"

REQUIRED_FIELDS = ("signingType", "clipFilename", "phrase", "signerId", "clipFileMd5",
                   "clipFileSize", "sensitiveContent")
MIN_LETTERS, MAX_LETTERS = 2, 12

# MediaPipe's hand landmark numbering. asl.features.points_from_mediapipe()
# returns the same order, so webcam code can index it with landmark_ids.
HAND_LANDMARK_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")
HAND_BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20))

LAYOUT_PARTIAL = "partial-8-landmarks"     # the shifted windows described above
LAYOUT_COMPLETE = "complete-21-landmarks"  # 21 true x, y and z values

FORMAT_NOTES = """\
FSboard pilot sequences: one .npz file per recording, plus index.csv.
Load with numpy.load(path, allow_pickle=False). No pickled objects are used.

left_hand, right_hand   float32, shape (frames, landmarks, 3). NaN where not known.
                        Read as hand[frame, position, coordinate].
                        coordinate: 0 = x, 1 = y, 2 = z.
                        position: the n-th entry of landmark_ids.
landmark_ids            MediaPipe hand landmark number of each position
                        (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
                        17-20 little finger).
left_mask, right_mask   bool, shape (frames,). True = that hand was detected.
timestamp_us            int64, time of each frame in microseconds, as stored.
frame_index             int32, 0, 1, 2, ... in the original frame order.
answer, label_ids       the full written answer in capitals, and the same letters
                        as numbers (A=1 ... Z=26; 0 is the CTC blank, never in a label).
phrase                  the original text from the metadata.
clip_id, video_name     the clip's metadata filename, and its name in the landmark record.
signer_id, split        the signer, and FSboard's official split (always train here).
image_width, image_height, frame_rate   of the source video.
source_shard, source_record   where it was read from. Tracing only, never used for pairing.
layout                  partial-8-landmarks or complete-21-landmarks.

Coordinates: x is a fraction of the image width from the left, y a fraction of
the height from the top, z a relative depth from MediaPipe. Values can fall
slightly outside 0..1. They refer to the upright video (image_width by
image_height, usually portrait). Left and right are the labels stored in the
dataset; the tracker sometimes gives one physical hand both labels.

Important: with layout partial-8-landmarks each hand has only the wrist, the
four thumb joints and three index-finger joints, and z of landmark 7 is always
NaN. The index fingertip and the middle, ring and little fingers are not in
FSboard's landmark files. To give the same input, webcam code must take
landmarks 0-7 from MediaPipe's 21 and ignore the z of landmark 7.
"""


PARTIAL_WARNING = """\
LIMITATION: these sequences hold 8 of the 21 hand points in each frame: the wrist,
the four thumb joints and three index-finger joints. FSboard's landmark files do not
contain the index fingertip or the middle, ring and little fingers. The files are
correctly paired and readable, but many letters differ mainly by those missing fingers."""


class FsboardError(RuntimeError):
    """A problem with a plain-English explanation of what to check."""

    def _render_traceback_(self):              # notebooks show the message, not a stack trace
        return [f"STOPPED: {self}"]


# ================================================================= downloads
def kaggle_auth_from_colab_secrets():
    """Copy Kaggle credentials from Colab's Secrets panel into this process's
    environment. Nothing is printed or written to disk. Returns the names found."""
    from google.colab import userdata  # only exists inside Colab

    found = []
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
            found.append(name)
    return found


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _auth_headers():
    """Optional Kaggle credentials from the environment. The public files
    downloaded without any when this was written."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        raw = base64.b64encode(f"{user}:{key}".encode()).decode()
        return {"Authorization": f"Basic {raw}"}
    return {}


def _http_problem(code, path):
    if code in (401, 403):
        return FsboardError(
            f"Kaggle refused to give us {path} (HTTP {code}).\nWhat to check:\n"
            f"  1. Sign in at {DATASET_PAGE} and accept anything the page asks you to accept.\n"
            "  2. Kaggle > Settings > API > Create New Token. In Colab, open the key icon (Secrets),\n"
            "     add it as KAGGLE_API_TOKEN (or add KAGGLE_USERNAME and KAGGLE_KEY), and switch on\n"
            "     notebook access.\n"
            "  3. Run the Setup cell again, then this cell. Never paste a key into a code cell.")
    if code == 404:
        return FsboardError(
            f"Kaggle says this file does not exist: {path} (HTTP 404).\nWhat to check:\n"
            f"  1. Look for it on the Data tab of {DATASET_PAGE}\n"
            "  2. If the folders were renamed, update the path in the Setup code.")
    if code == 429:
        return FsboardError(f"Kaggle asked us to slow down while fetching {path} (HTTP 429).\n"
                            "What to check: wait a few minutes, then run this cell again.")
    return FsboardError(f"The download of {path} failed with HTTP {code}.\n"
                        "What to check: run the cell again in a minute. If it keeps failing, "
                        f"see whether {DATASET_PAGE} opens in your browser.")


def _resolve(path):
    """Ask Kaggle where a file lives. Returns (url, headers) for fetching it.

    Kaggle answers with a redirect to a time-limited storage link. Credentials,
    if any, go to Kaggle only and are not forwarded to the storage host.
    """
    url = BASE_URL + urllib.parse.quote(path, safe="")
    auth = _auth_headers()
    req = urllib.request.Request(url, headers={"User-Agent": "fsboard-notebook", **auth})
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=60):
            return url, auth                   # served directly, no redirect
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
            return e.headers["Location"], {}
        raise _http_problem(e.code, path) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise FsboardError(f"Could not reach Kaggle to fetch {path} ({e}).\n"
                           "What to check: is this Colab session still connected? "
                           "Run the cell again.") from None


def remote_size(path):
    """Size in bytes, read from the server without downloading the file."""
    url, headers = _resolve(path)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "fsboard-notebook",
                                               **headers})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            content_range = r.headers.get("Content-Range", "")
            return int(content_range.split("/")[-1]) if "/" in content_range \
                else int(r.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        raise _http_problem(e.code, path) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise FsboardError(f"Could not reach the storage server for {path} ({e}).\n"
                           "What to check: is this Colab session still connected? "
                           "Run the cell again.") from None


def file_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def free_gb(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(folder).free / 1e9


def _fetch_to_part(url, headers, part, expected_size, log, label):
    """One download attempt, continuing from however much part already holds."""
    have = part.stat().st_size if part.exists() else 0
    req_headers = {"User-Agent": "fsboard-notebook", **headers}
    if have:
        req_headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:           # server ignored the Range: start again
            have = 0
        length = r.headers.get("Content-Length")
        total = have + int(length) if length else None
        step = max((expected_size or 0) // 10, 1)
        next_report = (have // step + 1) * step
        with open(part, "ab" if have else "wb") as out:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
                have += len(block)
                if expected_size and have >= next_report and have < expected_size:
                    log(f"   {label}: {have / 1e6:,.0f} of {expected_size / 1e6:,.0f} MB")
                    next_report += step
    if total is not None and have < total:     # a dropped connection can look like a normal end
        raise ConnectionError(f"received {have:,} of {total:,} bytes")


def download(path, dest, expected_size=None, expected_md5=None, log=print, attempts=5):
    """Download ONE dataset file to dest and verify it. Returns dest.

    A verified copy is reused, an interrupted download continues where it
    stopped, and a file that fails verification is deleted.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = dest.name

    def problem():
        size = dest.stat().st_size
        if expected_size is not None and size != expected_size:
            return f"it has {size:,} bytes but should have {expected_size:,}"
        if expected_md5:
            actual = file_md5(dest)
            if actual != expected_md5:
                return f"its MD5 is {actual} but clipFileMd5 says {expected_md5}"
        return ""

    if dest.exists() and not problem():
        log(f"   {label}: already here and verified, not downloading again")
        return dest
    if dest.exists():
        dest.unlink()

    url, headers = _resolve(path)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            _fetch_to_part(url, headers, part, expected_size, log, label)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:                  # the partial file is already complete
                break
            raise _http_problem(e.code, path) from None
        except OSError as e:                   # URLError and timeouts are OSErrors
            if getattr(e, "errno", None) == 28:
                raise FsboardError(f"The Colab disk is full while saving {label}.\nWhat to check: "
                                   f"{free_gb(dest.parent):.1f} GB is free. Delete files in the "
                                   "Files panel, or use Runtime > Disconnect and delete runtime.") from None
            if attempt == attempts:
                raise FsboardError(f"The download of {path} kept stopping ({e}).\nWhat to check: "
                                   "is this Colab session still connected? Run the cell again; "
                                   "it continues from what it already has.") from None
            log(f"   {label}: connection dropped, trying again ({attempt}/{attempts})")
            time.sleep(2 * attempt)
    part.replace(dest)
    bad = problem()
    if bad:
        dest.unlink()
        raise FsboardError(f"The downloaded {label} did not pass verification: {bad}.\n"
                           "What to check: the bad copy was deleted, so run this cell again. "
                           "If it fails twice, the file on Kaggle may have changed.")
    return dest


# ================================================================== metadata
def load_metadata(path):
    """Read a metadata JSON file and confirm the fields this code relies on."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise FsboardError(f"{path} is not a list of clip records.\nWhat to check: open the "
                           "file's preview on Kaggle; the dataset's format may have changed.")
    missing = [k for k in REQUIRED_FIELDS if k not in records[0]]
    if missing:
        raise FsboardError(f"The metadata has no field called {', '.join(missing)}.\nFields it does "
                           f"have: {', '.join(records[0])}.\nWhat to check: the dataset's format "
                           "may have changed; update REQUIRED_FIELDS and the code that uses them.")
    return records


def describe_fields(records):
    """(field, how many clips have it, type, example) for each metadata field."""
    counts, kinds, examples = Counter(), {}, {}
    for r in records:
        for k, v in r.items():
            counts[k] += 1
            kinds.setdefault(k, type(v).__name__)
            examples.setdefault(k, v)
    return [(k, counts[k], kinds[k], examples[k]) for k in counts]


def phrase_problem(phrase, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, allow_spaces=False):
    """Why this complete phrase cannot be a training answer ("" if it can).

    The phrase is judged as it is written. Nothing is removed from it, because
    the hands in the video spell every character. With allow_spaces=True, words
    separated by single spaces are accepted too (the space is an output since
    Phase 2B); the letter limits then count letters only.
    """
    if not isinstance(phrase, str) or not phrase:
        return "empty phrase"
    if re.fullmatch(r"[A-Za-z]+( [A-Za-z]+)*" if allow_spaces else r"[A-Za-z]+", phrase):
        letters = sum(c != " " for c in phrase)
        if letters < min_letters:
            return "too few letters"
        if letters > max_letters:
            return "too many letters"
        return ""
    if any(c.isdigit() for c in phrase):
        return "contains a digit"
    if allow_spaces and re.fullmatch(r"[A-Za-z ]+", phrase):
        return "space at the start or end, or two spaces"
    if not allow_spaces and any(c.isspace() for c in phrase):
        return "contains a space"
    return "contains a symbol"


def training_answer(phrase):
    """'letter' -> 'LETTER'. Repeated letters stay; they are real signs."""
    return vocab.normalise_transcript(phrase)


def eligible_rows(records, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, skip_sensitive=True,
                  allow_spaces=False):
    """All usable clips as simple rows, plus a count of why others were skipped.

    A clip with unsupported characters is skipped whole. Clips the publisher
    tagged sensitiveContent are left out of this pilot unless asked for.
    """
    rows, skipped = [], Counter()
    for r in records:
        reason = phrase_problem(r["phrase"], min_letters, max_letters, allow_spaces)
        if not reason and r["signingType"] != "fs":
            reason = "not fingerspelling"
        if not reason and skip_sensitive and r["sensitiveContent"]:
            reason = "tagged sensitiveContent"
        if reason:
            skipped[reason] += 1
            continue
        rows.append({"filename": r["clipFilename"], "phrase": r["phrase"],
                     "answer": training_answer(r["phrase"]), "signer_id": r["signerId"],
                     "md5": r["clipFileMd5"], "size_bytes": r["clipFileSize"]})
    return rows, skipped


def select_preview(rows, n=5):
    """Up to n clips from different signers (same result on every run)."""
    chosen, signers = [], set()
    for row in sorted(rows, key=lambda r: r["filename"]):
        if row["signer_id"] not in signers:
            chosen.append(row)
            signers.add(row["signer_id"])
        if len(chosen) == n:
            break
    if not chosen:
        raise FsboardError(f"No recording passed the filter ({MIN_LETTERS}-{MAX_LETTERS} letters and "
                           "nothing else).\nWhat to check:\n  1. Did the field list above show "
                           "'phrase' and 'signingType'?\n  2. Was the metadata download complete? "
                           "Delete it in the Files panel and run this cell again.\n  3. Are the "
                           "letter limits in this cell set the way you want?")
    return chosen


def clip_path(row):
    return TRAIN_VIDEO_DIR + row["filename"]


def pick_row(chosen, number):
    if not isinstance(number, int) or not 0 <= number < len(chosen):
        raise FsboardError(f"PREVIEW_ROW is {number!r}, but the table has rows 0 to {len(chosen) - 1}.\n"
                           "What to check: set PREVIEW_ROW to a row number from the table in step 1.")
    return chosen[number]


def video_html(path, answer, width=320):
    """The written answer directly above a playable video. Embedding is needed
    because a Colab browser tab cannot read files on the Colab disk."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<div style="font-family:sans-serif">'
            f'<div style="font-size:14px;color:#555">Written answer</div>'
            f'<div style="font-size:30px;font-weight:700;letter-spacing:2px;margin-bottom:8px">'
            f'{html.escape(answer)}</div>'
            f'<video controls loop playsinline width="{int(width)}" '
            f'src="data:video/mp4;base64,{data}"></video></div>')


# ================================================== records in landmark shards
def train_shard_path(number):
    if not isinstance(number, int) or not 0 <= number < NUM_TRAIN_SHARDS:
        raise FsboardError(f"SHARD_NUMBER is {number!r}, but training shards are numbered 0 to "
                           f"{NUM_TRAIN_SHARDS - 1}.\nWhat to check: use a whole number in that range.")
    return f"daun_v3/landmarks/daun_v3-train.arrow-{number:05d}-of-{NUM_TRAIN_SHARDS:05d}"


def shard_split(shard_name):
    """The official split, read from the file name: daun_v3-train.arrow-... -> train."""
    m = re.search(r"-(train|validation|test)\.arrow-", str(shard_name).replace("\\", "/"))
    if not m:
        raise FsboardError(f"{shard_name} is not named like an FSboard landmark shard.")
    return m.group(1)


def open_shard(path):
    """Open an Arrow shard without reading it all. Each row is decompressed
    only when it is asked for, which is how we read just what the pilot needs."""
    import pyarrow as pa

    try:
        reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
    except (pa.ArrowInvalid, OSError) as e:
        raise FsboardError(f"{path} could not be read as an Arrow file ({e}).\nWhat to check: the "
                           "download may be incomplete. Delete it in the Files panel and run this "
                           "cell again.") from None
    if "serialized" not in reader.schema.names:
        raise FsboardError(f"{path} has columns {reader.schema.names}, not 'serialized'.\nWhat to "
                           "check: compare with convert_arrow_to_parquet.py on the dataset page.")
    return reader


def iter_serialized(reader):
    """Yield the bytes of each record, in file order."""
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for value in batch.column(batch.schema.get_field_index("serialized")):
            yield value.as_py()


def parse_example(serialized):
    try:
        import tensorflow as tf
    except ImportError:
        raise FsboardError("Reading landmark records needs TensorFlow, which this runtime does not "
                           "have.\nWhat to check: run  !pip install tensorflow  in a new cell, then "
                           "run this cell again.") from None
    example = tf.train.SequenceExample()
    example.ParseFromString(serialized)
    return example


def record_context(example):
    """The clip-level values stored inside the record itself."""
    out = {}
    for key, feature in example.context.feature.items():
        kind = feature.WhichOneof("kind")
        if not kind or not getattr(feature, kind).value:
            continue
        value = getattr(feature, kind).value[0]
        out[key] = value.decode("utf-8") if isinstance(value, bytes) else value
    return out


def clip_filename_from_video_name(video_name):
    """Explicit name rule: seconds with a decimal point -> whole milliseconds.
    e4fa_965_20220920_145746_165.367_167.833.mp4 -> ..._165367_167833.mp4"""
    m = re.fullmatch(r"(.+)_(\d+\.\d+)_(\d+\.\d+)\.mp4", video_name)
    if not m:
        return video_name
    start, end = (int(Decimal(t) * 1000) for t in m.groups()[1:])
    return f"{m.group(1)}_{start}_{end}.mp4"


def metadata_index(records):
    return {r["clipFilename"]: r for r in records}


def match_metadata(context, index):
    """Find this record's metadata row by name, then compare its phrase and
    signer with the record's own. Returns (row, "") or (None, reason)."""
    row = index.get(clip_filename_from_video_name(context.get("video_name", "")))
    if row is None:
        return None, "no metadata row with this clip name"
    if row["phrase"] != context.get("prompt"):
        return None, "phrase differs between record and metadata"
    if row["signerId"] != context.get("signer_id"):
        return None, "signer differs between record and metadata"
    return row, ""


def decode_hand_frame(x, y, z):
    """Three stored lists for one hand in one frame -> (layout, points).

    See the top of this file. points has shape (8, 3) for the partial layout
    and (21, 3) for a complete one; (None, None) means an unknown layout.
    """
    x, y, z = (np.asarray(v, np.float32) for v in (x, y, z))
    if not len(x) == len(y) == len(z) == len(HAND_LANDMARK_NAMES):
        return None, None
    if np.allclose(x[1:], y[:-1], atol=1e-7) and np.allclose(y[1:], z[:-1], atol=1e-7):
        points = np.full((8, 3), np.nan, np.float32)
        points.flat[:23] = np.concatenate([x, y[-1:], z[-1:]])
        return LAYOUT_PARTIAL, points
    return LAYOUT_COMPLETE, np.stack([x, y, z], axis=1)


def _float_values(feature):
    return feature.float_list.value if feature.WhichOneof("kind") == "float_list" else ()


def decode_hands(example):
    """Ordered hand coordinates, masks and timing from one record. A frame's
    position in the lists is its frame number, as in the publisher's script."""
    lists = example.feature_lists.feature_list
    stamps = lists["timestamp"].feature
    T = len(stamps)
    decoded, masks, layouts = {}, {}, Counter()
    for hand in ("left", "right"):
        name = f"{hand}_hand"
        xs, ys, zs = (lists[f"{a}_{name}"].feature for a in "xyz")
        if not len(xs) == len(ys) == len(zs) == T:
            raise ValueError(f"{name} lists do not have {T} frames")
        frames = {}
        for t in range(T):
            x, y, z = _float_values(xs[t]), _float_values(ys[t]), _float_values(zs[t])
            if not (x and y and z):
                continue                       # hand not detected in this frame
            layout, points = decode_hand_frame(x, y, z)
            layouts[layout] += 1
            if layout is not None:
                frames[t] = points
        decoded[hand], masks[hand] = frames, np.array([t in frames for t in range(T)], bool)
    if None in layouts or len(layouts) > 1:
        raise ValueError(f"unexpected coordinate layout {dict(layouts)}")
    layout = next(iter(layouts), LAYOUT_PARTIAL)
    count = 8 if layout == LAYOUT_PARTIAL else len(HAND_LANDMARK_NAMES)
    out = {"layout": layout, "landmark_ids": np.arange(count, dtype=np.int32),
           "timestamp_us": np.array([s.int64_list.value[0] if s.WhichOneof("kind") == "int64_list" else -1
                                     for s in stamps], np.int64),
           "frame_index": np.arange(T, dtype=np.int32)}
    for hand in ("left", "right"):
        arr = np.full((T, count, 3), np.nan, np.float32)
        for t, points in decoded[hand].items():
            arr[t] = points
        out[f"{hand}_hand"], out[f"{hand}_mask"] = arr, masks[hand]
    return out


def format_preview(example):
    """A plain-text look inside one record, so the format is shown, not assumed."""
    context = record_context(example)
    lists = example.feature_lists.feature_list
    lines = ["Clip-level values stored inside the record:"]
    lines += [f"   {k:18s} {context[k]}" for k in sorted(context)]
    lines += ["Frame-by-frame lists: " + ", ".join(sorted(lists))]
    for hand in ("right_hand", "left_hand"):
        xs, ys, zs = (lists[f"{a}_{hand}"].feature for a in "xyz")
        found = [t for t in range(len(xs)) if _float_values(xs[t])]
        lines.append(f"{hand}: detected in {len(found)} of {len(xs)} frames")
        if found:
            t = found[0]
            x, y, z = (np.array(_float_values(v[t])) for v in (xs, ys, zs))
            fmt = lambda v: " ".join(f"{a:6.3f}" for a in v[:9])
            layout, _ = decode_hand_frame(x, y, z)
            lines += [f"   frame {t}, first 9 of {len(x)} numbers in each list:",
                      f"   x list: {fmt(x)}", f"   y list: {fmt(y)}", f"   z list: {fmt(z)}",
                      f"   layout found from these numbers: {layout}"]
    return "\n".join(lines)


def build_example(context, row, hands, shard_name, record_number):
    """Everything saved for one recording. Pairing was settled by match_metadata."""
    answer = training_answer(row["phrase"])
    return dict(hands,
                clip_id=row["clipFilename"], video_name=context["video_name"],
                signer_id=row["signerId"], split=shard_split(shard_name),
                phrase=row["phrase"], answer=answer,
                label_ids=np.array(vocab.encode(answer), np.int32),
                frame_rate=float(context.get("image/frame_rate", float("nan"))),
                image_width=int(context.get("image/width", -1)),
                image_height=int(context.get("image/height", -1)),
                source_shard=Path(str(shard_name)).name, source_record=int(record_number),
                format_version=FORMAT_VERSION)


def hand_frame_count(example):
    return int(np.sum(example["left_mask"] | example["right_mask"]))


def save_example(example, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (Path(example["clip_id"]).stem + ".npz")
    np.savez_compressed(path, **example)
    return path


def load_example(path):
    """Read an exported file back into plain values (no pickled objects)."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k, v in out.items():
        if v.ndim == 0:
            out[k] = v.item()
    return out


def check_example(example):
    """Checks on one exported recording: a list of (what, passed, detail)."""
    left, right = example["left_hand"], example["right_hand"]
    lmask, rmask = example["left_mask"], example["right_mask"]
    stamps, T = example["timestamp_us"], len(example["frame_index"])
    labels = example["label_ids"].tolist()
    lengths = {len(left), len(right), len(lmask), len(rmask), len(stamps), T}
    if len(lengths) != 1:
        return [("sequence and mask lengths agree", False, f"array lengths found: {sorted(lengths)}")]
    known = np.ones(left.shape[1:], bool)
    if example["layout"] == LAYOUT_PARTIAL:
        known[7, 2] = False                    # z of landmark 7 is not stored

    def coordinates_agree_with_mask(points, mask):
        return (np.isfinite(points[mask][:, known]).all() and np.isnan(points[~mask]).all())

    used = np.concatenate([left[lmask][:, known], right[rmask][:, known]])
    return [
        ("sequence and mask lengths agree", True, f"{T} frames in every array"),
        ("frames are in order", bool(np.all(np.diff(example["frame_index"]) == 1)
                                     and np.all(np.diff(stamps) > 0)), "frame numbers and times increase"),
        ("mask matches the coordinates",
         bool(coordinates_agree_with_mask(left, lmask) and coordinates_agree_with_mask(right, rmask)),
         "numbers where detected, NaN elsewhere"),
        ("usable hand coordinates exist", bool(used.size and np.abs(used).max() < 5),
         f"{hand_frame_count(example)} frames with a hand"),
        ("enough hand frames for the letters", hand_frame_count(example) >= vocab.required_frames(labels),
         f"needs at least {vocab.required_frames(labels)}"),
        ("answer matches the phrase", bool(example["answer"] == training_answer(example["phrase"])
                                           and vocab.decode_ids(labels) == example["answer"]
                                           and len(labels) == len(example["phrase"])),
         f"{example['phrase']} -> {example['answer']}, repeated letters kept"),
        ("official split is train", example["split"] == "train", example["split"]),
    ]


def model_sentence(example):
    """The one-sentence meaning of an exported file for the next milestone."""
    return (f"For each of its {len(example['frame_index'])} frames the model reads the numbers in "
            "left_hand, right_hand, left_mask and right_mask (with timing from timestamp_us), and "
            f"from those it should predict the letters {'-'.join(example['answer'])} in that order.")


def plot_example(example, panels=6):
    """Hand movement from an exported file: poses over time, then wrist paths."""
    import matplotlib.pyplot as plt

    colors = {"left": "#d1495b", "right": "#00798c"}
    ids = example["landmark_ids"].tolist()
    bones = [(ids.index(a), ids.index(b)) for a, b in HAND_BONES if a in ids and b in ids]
    any_hand = np.flatnonzero(example["left_mask"] | example["right_mask"])
    chosen = any_hand[np.linspace(0, len(any_hand) - 1, min(panels, len(any_hand))).astype(int)]
    seconds = (example["timestamp_us"] - example["timestamp_us"][0]) / 1e6
    hands = [h for h in colors if example[f"{h}_mask"].any()]
    seen = np.concatenate([example[f"{h}_hand"][example[f"{h}_mask"]][:, :, :2].reshape(-1, 2) for h in hands])
    (x0, y0), (x1, y1) = np.nanmin(seen, axis=0) - 0.03, np.nanmax(seen, axis=0) + 0.03
    fig = plt.figure(figsize=(2.2 * max(len(chosen), 3), 6))
    grid = fig.add_gridspec(2, len(chosen))
    for col, t in enumerate(chosen):
        ax = fig.add_subplot(grid[0, col])
        for hand in hands:
            if example[f"{hand}_mask"][t]:
                p = example[f"{hand}_hand"][t]
                for a, b in bones:
                    ax.plot(p[[a, b], 0], p[[a, b], 1], color=colors[hand], lw=2)
                ax.scatter(p[:, 0], p[:, 1], color=colors[hand], s=12, label=hand)
        ax.set(title=f"frame {t}, {seconds[t]:.1f} s", xlim=(x0, x1), ylim=(y1, y0), xticks=[], yticks=[])
        ax.set_aspect(example["image_height"] / example["image_width"]
                      if example["image_width"] > 0 else 1)
        if col == 0:
            ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[1, :])
    for hand in hands:
        ax.plot(seconds, example[f"{hand}_hand"][:, 0, 0], ".-", color=colors[hand], ms=3, label=f"{hand} wrist x")
        ax.plot(seconds, example[f"{hand}_hand"][:, 0, 1], ".:", color=colors[hand], ms=3, label=f"{hand} wrist y")
    ax.set(xlabel="seconds", ylabel="position (fraction of image)",
           title="Wrist position over time (gaps = no hand detected)")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"{example['answer']}  ({example['clip_id']})")
    fig.tight_layout()
    return fig


# ===================================================================== export
INDEX_COLUMNS = ("file", "clip_id", "video_name", "signer_id", "split", "phrase", "answer", "num_letters",
                 "num_frames", "frames_left_hand", "frames_right_hand", "frames_any_hand",
                 "duration_s", "frame_rate", "image_width", "image_height", "landmarks_per_hand",
                 "layout", "source_shard", "source_record")


def index_row(example, file_name):
    stamps = example["timestamp_us"]
    return {"file": file_name, "clip_id": example["clip_id"], "video_name": example["video_name"],
            "signer_id": example["signer_id"], "split": example["split"], "phrase": example["phrase"],
            "answer": example["answer"], "num_letters": len(example["answer"]),
            "num_frames": len(stamps), "frames_left_hand": int(example["left_mask"].sum()),
            "frames_right_hand": int(example["right_mask"].sum()),
            "frames_any_hand": hand_frame_count(example),
            "duration_s": round(float(stamps[-1] - stamps[0]) / 1e6, 3),
            "frame_rate": round(example["frame_rate"], 3), "image_width": example["image_width"],
            "image_height": example["image_height"],
            "landmarks_per_hand": int(len(example["landmark_ids"])), "layout": example["layout"],
            "source_shard": example["source_shard"], "source_record": example["source_record"]}


def read_index(out_dir):
    path = Path(out_dir) / "index.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_index(out_dir, rows):
    path = Path(out_dir) / "index.csv"
    tmp = path.with_name("index.csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


def export_from_shard(shard_path, records, out_dir, target=20, shard_name=None,
                      min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, skip_sensitive=True, log=print):
    """Read a downloaded TRAINING shard and export up to target usable recordings.

    Reading stops as soon as the target is reached. Running it again, or on
    another shard, keeps what was already exported. Returns a report dictionary.
    """
    shard_name = shard_name or str(shard_path)
    split = shard_split(shard_name)
    if split != "train":
        raise FsboardError(f"{shard_name} belongs to the official {split} split.\nWhat to check: this "
                           "milestone exports training data only. Validation and test stay untouched "
                           "for later evaluation.")
    out_dir = Path(out_dir)
    sequences = out_dir / "sequences"
    sequences.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(FORMAT_NOTES, encoding="utf-8")
    index = metadata_index(records)
    rows = read_index(out_dir)
    done = {r["clip_id"] for r in rows}
    reader = open_shard(shard_path)
    skipped, layouts = Counter(), Counter()
    exported = []
    read = paired = 0
    total = reader.num_record_batches
    for number, serialized in enumerate(iter_serialized(reader)):
        if len(rows) >= target:
            break
        read += 1
        if read % 100 == 0:
            log(f"   read {read} of {total} records, {len(rows)} sequences exported so far")
        example = parse_example(serialized)
        context = record_context(example)
        row, mismatch = match_metadata(context, index)
        if row is None:
            skipped[mismatch] += 1
            continue
        paired += 1
        reason = phrase_problem(row["phrase"], min_letters, max_letters)
        if not reason and row["signingType"] != "fs":
            reason = "not fingerspelling"
        if not reason and skip_sensitive and row["sensitiveContent"]:
            reason = "tagged sensitiveContent"
        if not reason and row["clipFilename"] in done:
            reason = "already exported earlier"
        if not reason:
            try:
                hands = decode_hands(example)
            except (ValueError, IndexError) as e:
                reason = f"unreadable landmarks ({e})"
        if not reason:
            built = build_example(context, row, hands, shard_name, number)
            needed = vocab.required_frames(built["label_ids"].tolist())
            layouts[built["layout"]] += 1
            if hand_frame_count(built) == 0:
                reason = "no hand detected in any frame"
            elif hand_frame_count(built) < needed:
                reason = "too few hand frames for its letters"
        if reason:
            skipped[reason] += 1
            continue
        path = save_example(built, sequences)
        rows.append(index_row(built, f"sequences/{path.name}"))
        done.add(built["clip_id"])
        exported.append(built["clip_id"])
        write_index(out_dir, rows)
    report = {"shard": Path(str(shard_name)).name, "records_in_shard": total, "records_read": read,
              "stopped_early": read < total, "paired_with_metadata": paired,
              "exported_now": exported, "exported_total": len(rows), "target": target,
              "skipped": dict(skipped.most_common()), "layouts": dict(layouts)}
    write_index(out_dir, rows)
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def report_text(report, eligible_total):
    """A plain-English summary. If short of the target, it says how many and why
    BEFORE suggesting another download; nothing more is downloaded by itself."""
    lines = [f"Shard {report['shard']}: read {report['records_read']} of {report['records_in_shard']} records"
             + (" (stopped once the target was reached)" if report["stopped_early"] else "") + ".",
             f"Pairing check: {report['paired_with_metadata']} of {report['records_read']} records matched "
             "the metadata by clip name, phrase and signer.",
             f"Exported from this shard: {len(report['exported_now'])}. "
             f"Total exported: {report['exported_total']} of a target of {report['target']}."]
    if report["skipped"]:
        lines.append("Why the other records were skipped:")
        lines += [f"   {count:5d}  {reason}" for reason, count in report["skipped"].items()]
    if report["exported_total"] < report["target"]:
        per_shard = eligible_total / NUM_TRAIN_SHARDS
        missing = report["target"] - report["exported_total"]
        extra = int(np.ceil(missing / max(per_shard, 1e-9)))
        lines += [f"\nFewer than {report['target']} sequences: {report['exported_total']} found. Reason: only "
                  f"{eligible_total:,} training clips in the whole dataset have a phrase of "
                  f"{MIN_LETTERS}-{MAX_LETTERS} letters and nothing else, about {per_shard:.1f} per shard. "
                  "Most phrases contain digits, symbols or spaces.",
                  f"Reaching {report['target']} would take about {extra} more shards "
                  f"(roughly {extra * 0.62:.1f} GB in total) if they are typical. Nothing further was "
                  "downloaded. To add one, change SHARD_NUMBER and run section 3 again; what you "
                  "already exported is kept."]
    return "\n".join(lines)


def zip_outputs(out_dir, zip_path):
    """One small zip of the exported files, replacing any earlier zip."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    made = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_dir.parent,
                               base_dir=out_dir.name)
    return Path(made)
