"""Outdoor Z3 route unit tests (no emulator)."""

from __future__ import annotations

from alttp.primitives import SpriteSnapshot
from smz3.outdoor_route import (
    CORRIDOR_X,
    FORTUNE_TELLER_SCREEN,
    LINKS_HOUSE_OW_SCREEN,
    OUTDOOR_SCREEN_PATH,
    OutdoorSegmentResult,
    choose_outdoor_buttons,
    on_links_house_screen,
    outdoor_path_screens,
    preferred_direction,
)
from smz3.ram import ComboSnapshot


def _snap(**overrides: object) -> ComboSnapshot:
    base = dict(
        frame=0,
        sm_game_state=0,
        sm_room_id=0,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=0,
        sm_max_health=0,
        sm_samus_x=0,
        sm_samus_y=0,
        sm_pose=0,
        z3_module=0x09,
        z3_submodule=0,
        z3_indoors=0,
        z3_room_id=0,
        z3_screen_id=FORTUNE_TELLER_SCREEN,
        z3_link_x=2648,
        z3_link_y=3275,
    )
    base.update(overrides)
    return ComboSnapshot(**base)  # type: ignore[arg-type]


def _sprite(
    *,
    x: int,
    y: int,
    sprite_type: int = 0x41,
    slot: int = 0,
) -> SpriteSnapshot:
    return SpriteSnapshot(
        slot=slot,
        sprite_type=sprite_type,
        state=9,
        hp=4,
        x=x,
        y=y,
    )


def test_outdoor_screen_path() -> None:
    path = outdoor_path_screens()
    assert path == list(OUTDOOR_SCREEN_PATH)
    assert path[0] == FORTUNE_TELLER_SCREEN
    assert path[-1] == LINKS_HOUSE_OW_SCREEN


def test_preferred_direction_from_fortune_teller() -> None:
    snap = _snap(z3_screen_id=0x35)
    assert preferred_direction(snap) == "UP"
    snap2 = _snap(z3_screen_id=0x2D)
    assert preferred_direction(snap2) == "LEFT"
    snap3 = _snap(z3_screen_id=0x2C)
    assert preferred_direction(snap3) is None


def test_corridor_bias_south_then_east() -> None:
    # Near door: prefer DOWN/RIGHT toward corridor, not pure UP into house.
    snap = _snap(z3_link_x=2648, z3_link_y=3275)
    buttons, fleeing = choose_outdoor_buttons(snap, (), house_cleared=False)
    assert not fleeing
    assert "DOWN" in buttons
    assert "UP" not in buttons


def test_corridor_on_band_goes_north() -> None:
    # After house clear + on corridor X → pure north.
    snap = _snap(z3_link_x=CORRIDOR_X, z3_link_y=3450)
    buttons, fleeing = choose_outdoor_buttons(snap, (), house_cleared=True)
    assert not fleeing
    assert buttons == ("UP",)


def test_mid_screen_goes_northwest() -> None:
    snap = _snap(z3_screen_id=0x2D, z3_link_x=2708, z3_link_y=3040)
    buttons, fleeing = choose_outdoor_buttons(snap, (), house_cleared=True)
    assert not fleeing
    assert "UP" in buttons and "LEFT" in buttons


def test_flee_side_steps_without_reversing() -> None:
    # Close enemy on corridor: keep UP, side-step LEFT.
    snap = _snap(z3_link_x=CORRIDOR_X, z3_link_y=3450)
    enemy = _sprite(x=CORRIDOR_X + 12, y=3450)
    buttons, fleeing = choose_outdoor_buttons(snap, (enemy,), house_cleared=True)
    assert fleeing
    assert "UP" in buttons
    assert "LEFT" in buttons
    assert "DOWN" not in buttons


def test_flee_on_mid_prefers_up() -> None:
    # Horizontal primary on $2D: side-step is always UP (NW path).
    snap = _snap(z3_screen_id=0x2D, z3_link_x=2708, z3_link_y=3040)
    # Force phase primary LEFT-ish by putting enemy north so side is UP.
    enemy = _sprite(x=2708, y=3000)
    buttons, fleeing = choose_outdoor_buttons(snap, (enemy,), house_cleared=True)
    assert "UP" in buttons
    assert "DOWN" not in buttons


def test_flee_near_door_forbids_up() -> None:
    # Spawn band: soldier to the east; must not flee UP into the house.
    snap = _snap(z3_link_x=2648, z3_link_y=3275)
    enemy = _sprite(x=2696, y=3268)
    buttons, fleeing = choose_outdoor_buttons(snap, (enemy,), house_cleared=False)
    assert "UP" not in buttons
    assert "DOWN" in buttons


def test_on_links_house_screen() -> None:
    assert on_links_house_screen(
        _snap(z3_screen_id=0x2C, z3_module=0x09, z3_submodule=0, z3_indoors=0)
    )
    assert not on_links_house_screen(_snap(z3_screen_id=0x35))
    assert not on_links_house_screen(_snap(z3_screen_id=0x2C, z3_indoors=1))


def test_result_dict() -> None:
    result = OutdoorSegmentResult(
        ok=True,
        frames=400,
        detail="reached",
        start_screen=0x35,
        final_screen=0x2C,
        screens_visited=[0x35, 0x2D, 0x2C],
        fled_frames=40,
    )
    d = result.to_dict()
    assert d["ok"] is True
    assert d["final_screen"] == "0x2C"
    assert d["screens_visited"] == ["0x35", "0x2D", "0x2C"]
    assert d["fled_frames"] == 40
