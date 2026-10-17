"""PVBench core evaluation package."""
from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
for _path in (_PROJECT_ROOT, _PACKAGE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
