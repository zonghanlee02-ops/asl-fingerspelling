import os
import sys
from pathlib import Path

# Make "import asl" and "import server" work when running pytest
# from anywhere. (In the Colab code zip, asl/ sits directly under ROOT.)
ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT, ROOT / "backend"):
    sys.path.insert(0, str(folder))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
