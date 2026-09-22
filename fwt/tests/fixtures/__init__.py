"""Test fixtures."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

FIXTURE_DIR = Path(__file__).parent


def load_fixture_rows(name: str = "tofu_like_rows.json") -> List[Dict[str, Any]]:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)
