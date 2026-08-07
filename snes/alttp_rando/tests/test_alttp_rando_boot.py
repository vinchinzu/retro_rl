"""Offline tests for alttp_rando boot helpers and JP path constants."""

from __future__ import annotations

from pathlib import Path

from alttp.ram import AlttpSnapshot
from alttp_rando.boot import BootResult
from alttp_rando.paths import (
    FIRST_PLAY_STATE,
    INTEGRATION,
    SHARED_Z3_JP_ROM,
    SHARED_Z3_US_ROM,
    Z3_JP_XXH32,
)


def _snap(**overrides: object) -> AlttpSnapshot:
    base = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=0x0104,
        indoors=True,
        screen_id=0,
        link_x=2368,
        link_y=8538,
        link_direction=2,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
    )
    base.update(overrides)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_boot_result_links_house_control() -> None:
    snap = _snap()
    result = BootResult(
        ok=True,
        frames=1110,
        snapshot=snap,
        method="alttp_startup",
        detail="test",
        state_path="/tmp/FirstPlay.state",
    )
    d = result.to_dict()
    assert d["ok"] is True
    assert d["has_control"] is True
    assert d["in_links_house"] is True
    assert d["game_mode"] == 0x07
    assert d["room_base_id"] == 0x04
    assert d["method"] == "alttp_startup"


def test_first_play_constant() -> None:
    assert FIRST_PLAY_STATE == "FirstPlay"
    assert INTEGRATION == "ALTTPRando-Snes"


def test_jp_paths_not_usa() -> None:
    assert SHARED_Z3_JP_ROM.name == "zelda3_jp.sfc"
    assert SHARED_Z3_US_ROM.name == "zelda3.sfc"
    assert SHARED_Z3_JP_ROM != SHARED_Z3_US_ROM
    assert Z3_JP_XXH32 == 0x8AC8FD15


def test_validate_z3_jp_rejects_usa_title(tmp_path: Path) -> None:
    from alttp_rando.scripts.setup_rom import validate_z3_jp

    # Minimal fake: 1 MiB with USA title string at LoROM header.
    body = bytearray(0x100000)
    body[0x7FC0 : 0x7FC0 + 21] = b"THE LEGEND OF ZELDA   "
    try:
        validate_z3_jp(bytes(body), path=tmp_path / "fake.sfc")
        raised = False
    except ValueError as exc:
        raised = True
        assert "USA" in str(exc) or "Japanese" in str(exc)
    assert raised
