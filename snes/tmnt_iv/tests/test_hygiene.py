"""Guard the layout without freezing a file whitelist."""

from __future__ import annotations

import re
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1]
_CLONE_RUNNER = re.compile(
    r"^(?:run_stage\d+_segment|probe_stage\d+_clean|run_stage\d+_bridge)\.py$"
)


def test_scripts_has_no_cloned_runners() -> None:
    clones = sorted(
        path.name for path in (GAME_DIR / "scripts").glob("*.py") if _CLONE_RUNNER.match(path.name)
    )
    assert clones == []


def test_lab_is_not_imported_by_production() -> None:
    roots = [GAME_DIR / "policy.py", GAME_DIR / "assist.py"]
    roots.extend(sorted((GAME_DIR / "tactics").glob("*.py")))
    roots.extend(sorted((GAME_DIR / "run").glob("*.py")))
    hits: list[str] = []
    for path in roots:
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "tmnt_iv.lab" in text or "from tmnt_iv.lab" in text:
            hits.append(path.relative_to(GAME_DIR).as_posix())
    assert hits == []
