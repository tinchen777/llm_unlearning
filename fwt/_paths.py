"""Path setup: the project's `src/` modules are imported as top-level packages."""

from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "configs"
FWT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def ensure_paths() -> None:
    """Make `import data`, `import trainer`, ... and `import fwt` work."""
    for path in (str(SRC_DIR), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_paths()
