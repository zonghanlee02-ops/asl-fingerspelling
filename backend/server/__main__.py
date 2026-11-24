"""Start the local server, from the project folder:

    python -m backend.server  [--port 8000] [--host 127.0.0.1]
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND.parent)

import uvicorn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this computer can connect (recommended)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"\n  ASL Fingerspelling running at  http://{args.host}:{args.port}\n")
    uvicorn.run("server.main:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
