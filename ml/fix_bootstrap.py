"""Clean up the double-try artifact left by the PowerShell attempt."""
import re
from pathlib import Path

CORRECT_BLOCK = (
    "try:\n"
    "    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/\n"
    "except ModuleNotFoundError:\n"
    "    import _pathfix  # noqa: F401  # python x.py from ml/src/"
)

# Match the mangled block: 'try:`n    try:\n    from src ...\nexcept...\n    import _pathfix...'
MANGLED_PATTERN = re.compile(
    r"try:`n\s+try:\s*\n"
    r"    from src import _ML_ROOT[^\n]*\n"
    r"except ModuleNotFoundError:\n"
    r"    import _pathfix[^\n]*",
    re.MULTILINE,
)

src_dir = Path("src")
for f in sorted(src_dir.glob("*.py")):
    if f.name in ("__init__.py", "_pathfix.py", "env_check.py"):
        continue
    text = f.read_text(encoding="utf-8")
    if "try:`n" in text:
        cleaned = MANGLED_PATTERN.sub(CORRECT_BLOCK, text, count=1)
        if cleaned != text:
            f.write_text(cleaned, encoding="utf-8")
            print(f"Fixed: {f.name}")
        else:
            print(f"Pattern did not match in: {f.name} — manual check needed")
    else:
        print(f"Clean: {f.name}")
