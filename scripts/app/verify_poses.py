"""
verify_poses.py - Record that a person checked reference poses against a
reliable ASL fingerspelling source.

Only mark a letter verified after comparing the 3D hand (Live screen, or
the "Pose review" list in the Speech screen) with the reference.

Examples
    python scripts/app/verify_poses.py --list
    python scripts/app/verify_poses.py --letters A B C --by "Your Name" \
        --reference "Lifeprint ASL fingerspelling chart"
    python scripts/app/verify_poses.py --letters R --reject --by "Your Name" \
        --reference "..." --notes "fingers not crossed in medoid sample"
"""
import argparse
import json
from datetime import date
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "frontend" / "public" / "reference_poses.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=[])
    ap.add_argument("--by")
    ap.add_argument("--reference")
    ap.add_argument("--notes")
    ap.add_argument("--reject", action="store_true", help="mark as checked and INCORRECT")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if a.list or not a.letters:
        for L, p in data["poses"].items():
            v = p["verification"]
            print(f"{L}: {v['status']:<10} {v.get('reference') or ''}")
        return
    if not a.by or not a.reference:
        ap.error("--by and --reference are required")
    for L in a.letters:
        L = L.upper()
        if L not in data["poses"]:
            print(f"skip {L}: no pose")
            continue
        data["poses"][L]["verification"] = {
            "status": "rejected" if a.reject else "verified",
            "by": a.by, "reference": a.reference, "date": date.today().isoformat(),
            "notes": a.notes}
        print(f"{L}: {'rejected' if a.reject else 'verified'}")
    PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
