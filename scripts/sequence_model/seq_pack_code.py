"""
seq_pack_code.py - Zip ONLY the code the Colab notebook needs (about 60 KB):
asl/__init__.py, asl/features.py, asl/seq/ (from backend/), and the sequence tests.
No data, models, recordings or credentials are included.

    venv\\Scripts\\python.exe scripts\\sequence_model\\seq_pack_code.py
    -> dist\\asl_seq_code.zip   (upload it to Google Drive: MyDrive/asl_seq/)

Alternative: push the repository to GitHub and let the notebook clone it.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
# {name inside the zip: file on disk}. In the zip, asl/ sits at the top level.
FILES = {"tests/conftest.py": ROOT / "tests" / "conftest.py",
         "asl/__init__.py": BACKEND / "asl" / "__init__.py",
         "asl/features.py": BACKEND / "asl" / "features.py"}
FILES.update({f"asl/seq/{p.name}": p for p in sorted((BACKEND / "asl" / "seq").glob("*.py"))})
FILES.update({f"tests/{p.name}": p for p in sorted((ROOT / "tests").glob("test_seq_*.py"))})


def main():
    out = ROOT / "dist" / "asl_seq_code.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in FILES.items():
            z.write(path, name)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(FILES)} files)")
    for f in FILES:
        print("  ", f)


if __name__ == "__main__":
    main()
