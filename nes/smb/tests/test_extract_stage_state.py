"""Pin promotion gates for extract_stage_state (no ROM / no emulator)."""

from __future__ import annotations

import inspect

import pytest

from smb.paths import RECORDINGS_DIR
from smb.ram import SmbSnapshot
from smb.reactive_route import snapshot_fingerprint
from smb.scripts.extract_stage_state import (
    CONTROL_X_MAX,
    extract,
    main,
    pin_promotion_issues,
)
from smb.start_presets import (
    normalize_stage_id,
    pin_meta_path,
    pin_state_path,
    pins_dir,
)


def _snap(
    *,
    world: int = 0,
    level: int = 0,
    oper_mode: int = 1,
    player_state: int = 8,
    lives: int = 2,
    player_x: int = 40,
    player_y: int = 176,
    timer: int = 400,
    level_number: int | None = None,
    area_pointer: int = 0,
    x_speed: int = 0,
    y_speed: int = 0,
    screen_x: int = 0,
    in_air: bool = False,
) -> SmbSnapshot:
    return SmbSnapshot(
        frame=0,
        player_state=player_state,
        player_x=player_x,
        player_y=player_y,
        x_page=player_x // 256,
        x_offset=player_x % 256,
        lives=lives,
        world=world,
        level=level,
        level_id=world * 4 + level,
        oper_mode=oper_mode,
        player_power=0,
        timer_hundreds=timer // 100,
        timer=timer,
        area_pointer=area_pointer,
        x_speed=x_speed,
        y_speed=y_speed,
        facing=1,
        screen_x=screen_x,
        player_screen_x=min(player_x, 255),
        in_air=in_air,
        level_number=level_number,
    )


def _issues(stage_id: str, snap: SmbSnapshot, fingerprint: dict | None = None) -> list[str]:
    fp = snapshot_fingerprint(snap) if fingerprint is None else fingerprint
    return pin_promotion_issues(stage_id, loaded=snap, settled=snap, fingerprint=fp)


def test_fingerprint_player_x_mismatch() -> None:
    snap = _snap(world=0, level=2, level_number=2, player_x=40, timer=400)
    fp = snapshot_fingerprint(snap)
    fp["player_x"] = snap.player_x + 1
    issues = _issues("1-3", snap, fp)
    assert any("player_x" in item for item in issues)


def test_missing_fingerprint_keys_are_issues() -> None:
    snap = _snap(world=0, level=2, level_number=2, player_x=40, timer=400)
    fp = snapshot_fingerprint(snap)
    for key in ("grounded", "timer", "timer_mod21", "screen_x"):
        fp.pop(key)
    issues = _issues("1-3", snap, fp)
    assert any(item.startswith("grounded:") for item in issues)
    assert any(item.startswith("timer:") for item in issues)
    assert any(item.startswith("timer_mod21:") for item in issues)
    assert any(item.startswith("screen_x:") for item in issues)


def test_bogus_1_3_is_1_2_underground() -> None:
    snap = _snap(
        world=0,
        level=2,
        level_number=1,
        player_state=8,
        oper_mode=1,
        player_x=40,
        timer=397,
    )
    issues = _issues("1-3", snap)
    assert issues
    joined = " ".join(issues)
    assert "dash_level=1" in joined
    assert "dash_level=2" in joined


def test_real_1_3_control_has_no_issues() -> None:
    snap = _snap(
        world=0,
        level=2,
        level_number=2,
        player_state=8,
        oper_mode=1,
        player_x=40,
        timer=400,
    )
    assert _issues("1-3", snap) == []


def test_mid_stage_2_1_rejected() -> None:
    snap = _snap(world=1, level=0, level_number=0, player_x=2431, timer=253)
    assert snap.player_x > CONTROL_X_MAX
    issues = _issues("2-1", snap)
    assert any("player_x" in item and str(CONTROL_X_MAX) in item for item in issues)


def test_extract_uses_start_presets_pin_paths() -> None:
    expected = RECORDINGS_DIR / "human" / "all_exits_v1_pins"
    assert pins_dir("all_exits_v1") == expected
    assert pin_state_path("all_exits_v1", "1-3") == expected / "1-3.state"
    assert pin_meta_path("all_exits_v1", "1-3") == expected / "1-3.json"
    extract_src = inspect.getsource(extract)
    main_src = inspect.getsource(main)
    assert "pin_state_path(" in extract_src
    assert "pin_meta_path(" in extract_src
    assert "pins_dir(" in main_src
    assert "pins_root" not in extract_src
    assert "pin_paths" not in extract_src
    assert "env.close()" in extract_src
    assert "boot_env.reset()" in extract_src


def test_normalize_stage_id_and_list_without_pins(capsys: pytest.CaptureFixture[str]) -> None:
    assert normalize_stage_id("smb_1_3") == "1-3"
    assert normalize_stage_id("1-3") == "1-3"
    assert main(["--list", "--task", "no_such_pins_task"]) == 0
    assert capsys.readouterr().out == ""


def test_main_requires_stages_unless_list() -> None:
    with pytest.raises(SystemExit):
        main([])
