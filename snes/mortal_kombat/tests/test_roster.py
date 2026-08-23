"""Roster pick / v3 filename tests."""

from __future__ import annotations

from pathlib import Path

from mortal_kombat.roster import (
    KIND_RAM_V3,
    KIND_SCRIPT,
    SCRIPT_NAME,
    STAGES,
    PIXEL_FALLBACK,
    backup_on_round_loss,
    record_stage,
    resolve_model,
    slot_for_match,
    v3_filename,
    build_slots,
)


def test_twelve_fights() -> None:
    assert len(STAGES) == 12
    assert STAGES[0][0] == "Fight"
    assert STAGES[-1][0] == "ShangTsung"
    assert STAGES[-2][0] == "Goro"


def test_v3_name() -> None:
    assert v3_filename("Goro") == "mk1_v3_Goro_ppo_final.zip"


def test_prefers_v3_when_present(tmp_path: Path) -> None:
    v3 = tmp_path / v3_filename("Fight")
    v3.write_bytes(b"zip")
    pixel = tmp_path / PIXEL_FALLBACK["Fight"]
    pixel.write_bytes(b"old")
    path, kind = resolve_model("Fight", tmp_path)
    assert path == v3
    assert kind == KIND_RAM_V3


def test_goro_and_shang_by_opponent_id(tmp_path: Path) -> None:
    for prefix in ("Goro", "ShangTsung", "Fight"):
        (tmp_path / v3_filename(prefix)).write_bytes(b"zip")
    slots = build_slots(tmp_path)
    goro = slot_for_match(10, 7, slots)
    shang = slot_for_match(11, 8, slots)
    assert goro is not None and goro.prefix == "Goro"
    assert shang is not None and shang.prefix == "ShangTsung"


def test_round_loss_backup(tmp_path: Path) -> None:
    from mortal_kombat.roster import StageSlot

    fallback = PIXEL_FALLBACK["Goro"]
    (tmp_path / fallback).write_bytes(b"old")
    slot = StageSlot(
        prefix="Goro",
        display="Goro",
        match_id=10,
        model=v3_filename("Goro"),
        kind=KIND_RAM_V3,
        backups=[fallback],
    )
    assert backup_on_round_loss(slot, tmp_path) == fallback


def test_fighters_common_alias_installs() -> None:
    import sys

    from mortal_kombat.compat import install_fighters_common_alias

    install_fighters_common_alias()
    assert "fighters_common" in sys.modules
    assert "fighters_common.fighting_env" in sys.modules


def test_resolve_script_without_zip(tmp_path: Path, monkeypatch) -> None:
    roster = tmp_path / "roster.json"
    record_stage(
        "Fight",
        model=SCRIPT_NAME,
        kind=KIND_SCRIPT,
        win_rate=None,
        attempts=0,
        path=roster,
    )
    monkeypatch.setattr("mortal_kombat.roster.ROSTER_PATH", roster)
    path, kind = resolve_model("Fight", tmp_path)
    assert kind == KIND_SCRIPT
    assert path == tmp_path / SCRIPT_NAME
    assert not path.exists()


def test_build_slots_includes_scripted_without_zip(tmp_path: Path, monkeypatch) -> None:
    roster = tmp_path / "roster.json"
    record_stage(
        "Fight",
        model=SCRIPT_NAME,
        kind=KIND_SCRIPT,
        win_rate=None,
        attempts=0,
        path=roster,
    )
    monkeypatch.setattr("mortal_kombat.roster.ROSTER_PATH", roster)
    slots = build_slots(tmp_path)
    fight = next(slot for slot in slots if slot.prefix == "Fight")
    assert fight.model == SCRIPT_NAME
    assert fight.kind == KIND_SCRIPT


def test_record_stage_roundtrip(tmp_path: Path) -> None:
    roster = tmp_path / "roster.json"
    record_stage(
        "Fight",
        model=v3_filename("Fight"),
        kind=KIND_RAM_V3,
        win_rate=0.5,
        attempts=4,
        path=roster,
    )
    text = roster.read_text()
    assert "mk1_v3_Fight_ppo_final.zip" in text
    assert "0.5" in text
