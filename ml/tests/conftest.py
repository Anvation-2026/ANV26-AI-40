"""pytest configuration and sys.path setup for MedGuard ML tests."""
import sys
from pathlib import Path

# Ensure ml/ root is on sys.path regardless of where pytest is run from
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))
