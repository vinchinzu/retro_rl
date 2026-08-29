"""Unit locks for Early Spazer Beam mainline K2.2 (geometry + wiring)."""

from __future__ import annotations

from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.kpdr import red_stack
from super_metroid.routes.kpdr.below_spazer_west import play_below_spazer_floor_to_west
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
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


def test_mainline_west_wires_detour() -> None:
    """Spine west hop is the Spazer detour, not the floor skip."""
    assert red_stack.play_below_spazer_to_west is not play_below_spazer_floor_to_west
