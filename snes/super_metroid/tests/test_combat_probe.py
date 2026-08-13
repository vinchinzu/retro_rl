"""Unit tests for shared combat-probe CLI helpers (no emulator)."""

from __future__ import annotations

from pathlib import Path

from super_metroid.combat.probe import resolve_named_state, write_json_report


def test_resolve_named_state_uses_alias(tmp_path: Path) -> None:
    named = tmp_path / "entry.state"
    named.write_bytes(b"pin")
    assert resolve_named_state("entry", {"entry": named}) == named


def test_resolve_named_state_finds_scratch_basename(tmp_path: Path, monkeypatch) -> None:
    pin = tmp_path / "custom.state"
    pin.write_bytes(b"pin")
    monkeypatch.setattr(
        "super_metroid.combat.probe.DEFAULT_STATE_DIRS",
        (tmp_path,),
    )
    assert resolve_named_state("custom").resolve() == pin.resolve()
    assert resolve_named_state(str(pin)).resolve() == pin.resolve()


def test_write_json_report_prints_and_persists(tmp_path: Path, capsys) -> None:
    out = tmp_path / "report.json"
    write_json_report({"ok": True, "n": 1}, out)
    captured = capsys.readouterr().out
    assert '"ok": true' in captured
    assert out.read_text(encoding="utf-8").startswith("{\n")
