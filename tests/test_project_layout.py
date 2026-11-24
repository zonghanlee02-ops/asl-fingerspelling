"""The project layout: each part is where the code expects it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_location_constants_point_at_real_folders():
    import asl

    assert asl.ROOT == ROOT and asl.BACKEND_DIR == ROOT / "backend"
    assert asl.MODELS_DIR == ROOT / "backend" / "models" and (asl.MODELS_DIR / "final_model_i").exists()
    assert asl.FRONTEND_DIST.parent == ROOT / "frontend"
    assert (ROOT / "frontend" / "package.json").exists()
    assert not (ROOT / "baseline").exists()           # the baseline model was removed on 23 Nov 2026


def test_private_and_generated_folders_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("backend/storage/", "backend/models/vosk/", "backend/models/versions/",
                    "frontend/node_modules/", "frontend/dist/", "data/asl_alphabet_train/"):
        assert pattern in ignored
