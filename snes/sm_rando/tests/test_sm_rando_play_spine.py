"""Play spine manifest helpers (no emulator)."""

from __future__ import annotations

from pathlib import Path

from retro_harness.play_spine import RunManifest, fun_hud_lines


def test_run_manifest_write(tmp_path: Path) -> None:
    m = RunManifest(
        game="SMRando-Snes",
        package="sm_rando",
        started_at="2026-08-06T00:00:00Z",
        seed="1337",
    )
    m.add_milestone("ship")
    m.frames = 120
    m.outcome = "session_end"
    path = m.write(tmp_path / "run.json")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "sm_rando" in text
    assert "ship" in text


def test_fun_hud_has_seed() -> None:
    lines = fun_hud_lines(package="sm_rando", seed="99", frame=0)
    assert any("99" in line for line in lines)
