"""The FSboard Colab notebook: it carries the repository's exact code, is clean to
publish, and really keeps the large download off, even under "Run all". Whole-notebook
runs use a fake Kaggle server and fake Colab modules, so they need no internet."""
import ast
import hashlib
import http.server
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from asl.seq import fsboard as fb

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "fsboard_prepare_data_colab.ipynb"
SHARD = fb.train_shard_path(0)


def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_carries_the_exact_repository_code():
    tree = ast.parse(code_cells()[0])
    files = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_FILES")
    assert set(files) >= {"asl/seq/fsboard.py", "asl/seq/vocab.py"}
    for name, text in files.items():
        assert text == (ROOT / "backend" / name).read_text(encoding="utf-8"), \
            f"{name} changed: run scripts/sequence_model/build_fsboard_notebook.py"


def test_notebook_is_published_without_outputs_or_credentials():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "data:video/mp4;base64,AA" not in text and "KGAT_" not in text


def test_large_download_is_off_by_default():
    assert sum("RUN_LANDMARK_PREP = False" in c for c in code_cells()) == 2     # setup + section 3
    assert not any("RUN_LANDMARK_PREP = True" in c for c in code_cells())


# ---------------------------------------------------------- whole-notebook runs
pytest.importorskip("tensorflow")

from tests.test_seq_fsboard import make_record, meta_row  # noqa: E402

CLIP_BYTES = b"fake mp4 bytes" * 50


class Service(http.server.BaseHTTPRequestHandler):
    files = {}
    requested = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.lstrip("/")
        type(self).requested.append(name)
        if name not in self.files:
            self.send_error(404)
            return
        data = self.files[name]
        spec = self.headers.get("Range")
        if spec:
            a, b = spec.split("=")[1].split("-")
            a = int(a)
            b = int(b) if b else len(data) - 1
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def colab(tmp_path, monkeypatch):
    """A stand-in for Colab: temp folder, fake Kaggle, fake google.colab and IPython."""
    from urllib.parse import quote

    pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")

    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)

    videos = [("a_1_20220101_000000_1.5_2.5.mp4", "a_1_20220101_000000_1500_2500.mp4", "letter", "a"),
              ("b_2_20220101_000000_3.0_4.0.mp4", "b_2_20220101_000000_3000_4000.mp4", "hello", "b")]
    md5 = hashlib.md5(CLIP_BYTES).hexdigest()
    meta = [meta_row(clip, phrase, signer, clipFileMd5=md5, clipFileSize=len(CLIP_BYTES))
            for _, clip, phrase, signer in videos]
    meta.append(meta_row("c_3.mp4", "2067 pso costa", "c"))
    shard = tmp_path / "shard.arrow"
    schema = pa.schema([("serialized", pa.binary())])
    with pa.OSFile(str(shard), "wb") as sink, pa.ipc.new_file(
            sink, schema, options=pa.ipc.IpcWriteOptions(compression="zstd")) as w:
        for i, (video, _, phrase, signer) in enumerate(videos):
            w.write_batch(pa.record_batch(
                [pa.array([make_record(video, phrase, signer, seed=i).SerializeToString()], pa.binary())],
                schema=schema))
    Service.files = {quote(fb.TRAIN_METADATA, safe=""): json.dumps(meta).encode(),
                     quote(SHARD, safe=""): shard.read_bytes()}
    for _, clip, _, _ in videos:
        Service.files[quote(fb.TRAIN_VIDEO_DIR + clip, safe="")] = CLIP_BYTES
    Service.requested = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    shown, downloads = [], []
    fake_colab = types.ModuleType("google.colab")
    fake_colab.userdata = types.SimpleNamespace(get=lambda name: (_ for _ in ()).throw(KeyError(name)))
    fake_colab.files = types.SimpleNamespace(download=downloads.append)
    fake_colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "colab", fake_colab, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    fake_display = types.ModuleType("IPython.display")
    fake_display.HTML = lambda text: ("html", text)
    fake_display.display = shown.append
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None
    fake_ipython.version_info = (8, 0, 0)
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    monkeypatch.setenv("FSBOARD_WORK", str(tmp_path / "content"))
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    state = types.SimpleNamespace(work=tmp_path / "content", shown=shown, downloads=downloads,
                                  url=f"http://127.0.0.1:{server.server_port}/")
    yield state
    server.shutdown()
    server.server_close()


def run_notebook(colab, monkeypatch, change=()):
    """Run every code cell in order, like Runtime > Run all."""
    scope = {"__name__": "__main__"}
    for i, source in enumerate(code_cells()):
        for old, new in change:
            source = source.replace(old, new)
        exec(compile(source, f"cell{i}", "exec"), scope)
        if i == 0:                                  # setup has just loaded the helper code
            monkeypatch.setattr(scope["fb"], "BASE_URL", colab.url)
    return scope


def test_run_all_by_default_never_touches_the_landmark_shard(colab, monkeypatch, capsys):
    scope = run_notebook(colab, monkeypatch)
    out = capsys.readouterr().out
    assert not any("landmarks" in name for name in Service.requested)
    assert not (colab.work / "shards").exists() or not list((colab.work / "shards").iterdir())
    assert not fb.read_index(colab.work / "fsboard_pilot")
    assert out.count("Skipped: section 3 is off.") == 4 and "nothing to check" in out
    assert colab.downloads == []
    # Sections 1 and 2 did run
    assert "2067 pso costa" in out and "MD5 check passed" in out
    assert len(scope["preview_rows"]) == 2
    table = [x for x in colab.shown if hasattr(x, "columns") and "Training answer" in x.columns][0]
    assert table["Training answer"].tolist() == ["LETTER", "HELLO"]
    assert list(table.columns) == ["Filename", "Original phrase", "Training answer", "Signer ID"]
    page = [x for x in colab.shown if isinstance(x, tuple)][0][1]
    assert page.index("LETTER") < page.index("<video")
    # Only the metadata file and ONE video were requested
    clips = {n for n in Service.requested if "video_clips" in n}
    assert len(clips) == 1


def test_preview_row_chooses_another_video(colab, monkeypatch, capsys):
    run_notebook(colab, monkeypatch, [("PREVIEW_ROW = 0", "PREVIEW_ROW = 1")])
    assert "b_2_20220101_000000_3000_4000.mp4" in capsys.readouterr().out
    page = [x for x in colab.shown if isinstance(x, tuple)][0][1]
    assert "HELLO" in page


def test_a_bad_row_or_failed_video_gives_a_message_and_run_all_continues(colab, monkeypatch, capsys):
    run_notebook(colab, monkeypatch, [("PREVIEW_ROW = 0", "PREVIEW_ROW = 9")])
    out = capsys.readouterr().out
    assert "rows 0 to 1" in out and "Skipped: section 3 is off." in out


def test_a_video_failing_its_md5_is_not_displayed(colab, monkeypatch, capsys):
    from urllib.parse import quote

    Service.files[quote(fb.TRAIN_VIDEO_DIR + "a_1_20220101_000000_1500_2500.mp4", safe="")] = b"x" * len(CLIP_BYTES)
    run_notebook(colab, monkeypatch)
    out = capsys.readouterr().out
    assert "clipFileMd5" in out and "did not pass verification" in out
    assert not [x for x in colab.shown if isinstance(x, tuple)]


def test_section_three_when_enabled_exports_and_section_four_proves_it(colab, monkeypatch, capsys):
    scope = run_notebook(colab, monkeypatch, [("RUN_LANDMARK_PREP = False  #@param", "RUN_LANDMARK_PREP = True  #@param")])
    out = capsys.readouterr().out
    assert "Section 3 is ON" in out
    assert out.index("Approximate size:") < out.index("Download complete")
    assert "partial-8-landmarks" in out and "LIMITATION" in out
    assert "Pairing check: 2 of 2" in out and "Total exported: 2 of a target of 20" in out
    assert "Fewer than 20 sequences: 2 found" in out and "Nothing further was downloaded" in out
    index = fb.read_index(colab.work / "fsboard_pilot")
    assert [r["answer"] for r in index] == ["LETTER", "HELLO"]
    assert colab.downloads and colab.downloads[0].endswith("fsboard_pilot.zip")
    assert "FAIL" not in out and "2 of 2 pass every check" in out
    assert "L-E-T-T-E-R" in out
    assert (12, 8, 3) == scope["example"]["left_hand"].shape


def test_notebook_can_be_run_twice_without_trouble(colab, monkeypatch, capsys):
    enable = [("RUN_LANDMARK_PREP = False  #@param", "RUN_LANDMARK_PREP = True  #@param")]
    run_notebook(colab, monkeypatch, enable)
    first = len(Service.requested)
    capsys.readouterr()
    run_notebook(colab, monkeypatch, enable)
    out = capsys.readouterr().out
    assert "already here and verified" in out and "Total exported: 2 of a target of 20" in out
    assert not [n for n in Service.requested[first:] if n.endswith("00100")
                and Service.requested[first:].count(n) > 2]   # at most the size checks, not a re-download
    assert len(fb.read_index(colab.work / "fsboard_pilot")) == 2
