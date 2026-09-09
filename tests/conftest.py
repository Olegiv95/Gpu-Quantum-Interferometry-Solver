"""Make repository example and benchmark scripts importable during tests."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for directory in (PROJECT_ROOT / "Examples", PROJECT_ROOT / "Benchmarks"):
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
