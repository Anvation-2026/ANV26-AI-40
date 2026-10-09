"""pytest configuration and sys.path setup for MedGuard ML tests."""
import sys
from pathlib import Path

# Ensure both ml/ and project root are on sys.path
_ML_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _ML_ROOT.parent
for p in [str(_PROJECT_ROOT), str(_ML_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
