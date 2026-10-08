"""
Path bootstrap: ensures ml/ root is on sys.path so that
`config` and `src.*` modules can always be found regardless of
how Python was invoked (python -m src.train, direct script, etc.)
"""
import sys
from pathlib import Path

# _pathfix.py lives at ml/src/_pathfix.py
# .parent       = ml/src/
# .parent.parent = ml/   <-- _ML_ROOT
_ML_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _ML_ROOT / "src"

for _p in (str(_ML_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
