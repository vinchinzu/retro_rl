"""Guard the layout without freezing a file whitelist."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1]


def test_one_run_cli_and_no_second_loop() -> None:
    """``scripts/run.py`` is the only runner; ``run_trial`` the only live loop.

    ``lookahead.py`` steps the raw core (``em.step``) inside rollouts only.
    """
    scripts = sorted(p.name for p in (GAME_DIR / "scripts").glob("*.py") if p.name != "__init__.py")
    assert scripts == ["run.py", "setup_rom.py"]
    loops = []
    for path in GAME_DIR.rglob("*.py"):
        if "tests" in path.parts or path.name == "trial.py":
            continue
        if "env.step(" in path.read_text(encoding="utf-8"):
            loops.append(path.relative_to(GAME_DIR).as_posix())
    assert loops == []


def test_deleted_trees_stay_deleted() -> None:
    for name in ("lab", "local_grind", "grind_knobs.py"):
        assert not (GAME_DIR / name).exists(), name
