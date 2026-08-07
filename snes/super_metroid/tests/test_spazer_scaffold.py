"""Unit locks for Early Spazer Beam mainline K2.2 (geometry + wiring)."""

from __future__ import annotations

from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.kpdr import red_stack, spazer
from super_metroid.routes.kpdr.below_spazer_west import play_below_spazer_floor_to_west
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.kpdr.spazer import SPAZER_BEAM_MASK, geometry, scripts
from super_metroid.routes.kpdr.spazer.geometry import (
    SOLID_TOP_X_MIN,
    SOLID_TOP_Y,
    mid_band,
    on_mid_or_floor,
    on_solid_top,
    on_super_door_approach,
    standing_mid_seat,
)


def _state(
    *,
    room_id: int = ROOM_BELOW_SPAZER,
    x: int = 50,
    y: int = 400,
    pose: int = 1,
    beams: int = 0,
    door: int = 0,
) -> SuperMetroidState:
    return SuperMetroidState(
        frame=0,
        game_state=8,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=room_id,
        area_index=1,
        door_transition=door,
        transition_direction=0,
        samus_x=x,
        samus_y=y,
        velocity_x=0,
        velocity_y=0,
        pose=pose,
        health=99,
        max_health=99,
        reserve_health=0,
        max_reserve_health=0,
        missiles=0,
        max_missiles=0,
        super_missiles=5,
        max_super_missiles=5,
        power_bombs=0,
        max_power_bombs=0,
        selected_item=0,
        equipped_items=0,
        collected_items=0,
        equipped_beams=beams,
        collected_beams=beams,
        timer_type=0,
        escape_timer_frames=0,
        escape_timer_seconds=0,
        escape_timer_minutes=0,
        num_enemies=0,
        enemies_killed=0,
        enemy0_x=0,
        enemy0_y=0,
        enemy0_hp=0,
        enemy0_spritemap=0,
        event_flags=(0,) * 8,
        boss_bits=(0,) * 8,
    )


def test_room_constant() -> None:
    assert spazer.ROOM_SPAZER == 0xA447


def test_spazer_beam_mask() -> None:
    assert SPAZER_BEAM_MASK == 0x0004
    assert geometry.SPAZER_BEAM_MASK == 0x0004


def test_public_exports() -> None:
    for name in (
        "play_below_spazer_to_spazer",
        "play_below_spazer_climb",
        "play_below_spazer_floor_to_mid",
        "play_below_spazer_mid_to_top",
        "play_spazer_collect",
        "play_spazer_return_to_below",
        "play_spazer_top_to_mid",
        "play_spazer_top_to_west",
        "play_spazer_detour",
    ):
        assert name in spazer.__all__
        assert callable(getattr(spazer, name))
    assert callable(play_below_spazer_floor_to_west)
    assert callable(red_stack.play_below_spazer_floor_to_west)
    assert red_stack.play_below_spazer_floor_to_west is play_below_spazer_floor_to_west


def test_guide_rle_shapes() -> None:
    """Compressed guide phases — not full human gold paste."""
    assert len(scripts.FLOOR_MID_RLE) >= 10
    floor_frames = sum(n for n, _ in scripts.FLOOR_MID_RLE)
    assert 100 <= floor_frames <= 400
    floor_btns = {tuple(b) for _, b in scripts.FLOOR_MID_RLE}
    assert ("B", "LEFT", "A") in floor_btns or ("RIGHT", "A") in floor_btns

    assert len(scripts.TOP_MID_RLE) >= 10
    top_frames = sum(n for n, _ in scripts.TOP_MID_RLE)
    assert 200 <= top_frames <= 800
    top_btns = {tuple(b) for _, b in scripts.TOP_MID_RLE}
    assert ("B", "LEFT", "A") in top_btns
    assert ("DOWN",) in top_btns

    assert len(scripts.TOP_DOOR_APPROACH_RLE) >= 8
    door_frames = sum(n for n, _ in scripts.TOP_DOOR_APPROACH_RLE)
    assert 200 <= door_frames <= 600
    door_btns = {tuple(b) for _, b in scripts.TOP_DOOR_APPROACH_RLE}
    assert ("DOWN",) in door_btns
    assert ("X",) in door_btns or ("RIGHT", "X") in door_btns

    assert geometry.WJ_LEFT.into == "LEFT"
    assert geometry.WJ_RIGHT.into == "RIGHT"
    assert geometry.WJ_LEFT.into_frames >= 10


def test_solid_top_predicate() -> None:
    y_lo, y_hi = SOLID_TOP_Y
    assert y_lo < y_hi
    assert SOLID_TOP_X_MIN >= 40

    # Natural node-4 land ~(91,91)p1
    assert on_solid_top(_state(x=91, y=91, pose=1))
    # Place-embed sill ~(130) still solid top
    assert on_solid_top(_state(x=100, y=130, pose=2))
    # Shaft air peak is NOT land (x too low)
    assert not on_solid_top(_state(x=59, y=124, pose=1))
    # Air pose on solid coords is NOT land
    assert not on_solid_top(_state(x=91, y=91, pose=25))
    # Wrong room
    assert not on_solid_top(_state(room_id=0xA447, x=91, y=91, pose=1))


def test_mid_seat_and_mid_or_floor() -> None:
    seat = _state(x=55, y=235, pose=1)
    assert mid_band(seat)
    assert standing_mid_seat(seat)
    assert not standing_mid_seat(_state(x=55, y=235, pose=25))  # spin apex
    assert not mid_band(_state(x=55, y=400, pose=1))  # floor

    assert on_mid_or_floor(_state(x=100, y=220, pose=1))
    assert on_mid_or_floor(_state(x=43, y=395, pose=1))
    assert not on_mid_or_floor(_state(x=380, y=155, pose=1))  # top handoff


def test_super_door_approach_band() -> None:
    lip = _state(x=460, y=139, pose=1)
    assert on_super_door_approach(lip)
    assert not on_super_door_approach(_state(x=400, y=139, pose=1))  # too left
    assert not on_super_door_approach(_state(x=460, y=200, pose=1))  # too low


def test_spazer_segments_registered_mainline() -> None:
    from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS

    for seg_id in (
        "below_spazer_to_spazer",
        "spazer_collect",
        "spazer_return_to_below",
        "spazer_top_to_west",
        "spazer_detour",
        "below_spazer_to_west",
    ):
        assert seg_id in KPDR_SEGMENTS


def test_mainline_west_wires_detour() -> None:
    """Spine hop is the Spazer detour (always collect)."""
    assert red_stack.play_below_spazer_to_west is not play_below_spazer_floor_to_west
    # Registry maps both; detour is the product path for pure probe.
    from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS

    assert KPDR_SEGMENTS["spazer_detour"] is spazer.play_spazer_detour
    assert KPDR_SEGMENTS["below_spazer_to_west"] is red_stack.play_below_spazer_to_west


def test_spazer_modules_do_not_import_red_stack() -> None:
    """Spazer hop modules exit west via below_spazer_west (no red_stack cycle)."""
    from pathlib import Path

    pkg = Path(spazer.__file__).resolve().parent
    for path in pkg.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Docstrings may mention red_stack as the spine caller; ban import edges.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            if "import" in stripped and "red_stack" in stripped:
                raise AssertionError(f"{path.name} imports red_stack: {stripped}")
    west_src = Path(play_below_spazer_floor_to_west.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert "red_stack" not in west_src
