"""
Path bootstrap: ensures ml/ root is on sys.path so that
`config` and `src.*` modules can always be found regardless of
how Python was invoked (python -m src.train, direct script, etc.)
"""
import sys
from pathlib import Path

# ml/ directory is always this file's parent
_ML_ROOT = Path(__file__).resolve().parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))
