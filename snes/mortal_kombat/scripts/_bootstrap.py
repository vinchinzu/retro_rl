"""Make ``mortal_kombat`` importable when scripts are run as files."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    root = Path(__file__).resolve().parents[3]
    for extra in (root, root / "snes"):
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)
