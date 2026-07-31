"""Red Tower descent → Bat → Below Spazer → Warehouse tunnels."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph
_wait_ordinary_room = wait_ordinary_room

def play_red_tower_to_bat(session: ControllerSession) -> SuperMetroidState:
    """Natural Noob-door spawn → Red Tower bottom → Bat Room ``0xA3DD``."""
    _require_room(session, ROOM_RED_TOWER, "red_to_bat")
    _unmorph(session)

    # Zigzag down the tall upper shaft.  x<=45 is intentional: switching at
    # x<=35 leaves Samus perched at x≈37/y≈1499.
    direction = "RIGHT"
    for _ in range(1600):
        state = session.state
        if state.samus_y >= 1600:
            break
        if state.samus_x >= 225:
            direction = "LEFT"
        elif state.samus_x <= 45:
            direction = "RIGHT"
        _hold(session, 1, direction, "B", reason="red_upper_zigzag")
    else:
        raise TimeoutError(f"red_to_bat: upper descent stalled: {state}")
    _hold(session, 40, reason="red_floor_settle")

    # Open the temporary floor and cross it during the bomb timer.
    ensure_morph(session)
    for _ in range(100):
        state = session.state
        if 148 <= state.samus_x <= 152:
            break
        direction = "LEFT" if state.samus_x > 152 else "RIGHT"
        _hold(session, 1, direction, reason="red_floor_bomb_position")
    _hold(session, 2, "X", reason="red_floor_bomb")
    _hold(session, 8, "LEFT", reason="red_floor_bomb_retreat")
    _hold(session, 40, reason="red_floor_bomb_wait")
    for _ in range(180):
        state = _hold(session, 1, "RIGHT", reason="red_floor_cross")
        if state.samus_y >= 1650:
            break
    else:
        raise TimeoutError(f"red_to_bat: timed floor crossing failed: {state}")

    for frame in range(500):
        buttons = ("LEFT", "X") if frame % 45 < 2 else ("LEFT",)
        state = _hold(session, 1, *buttons, reason="red_tunnel_bomb_roll")
        if state.samus_y >= 1880:
            break
    else:
        raise TimeoutError(f"red_to_bat: upper tunnel descent stalled: {state}")

    for _ in range(300):
        state = _hold(session, 1, "RIGHT", reason="red_tunnel_right")
        if state.samus_y >= 2090:
            break
    else:
        raise TimeoutError(f"red_to_bat: lower tunnel entry stalled: {state}")

    direction = "LEFT"
    for _ in range(600):
        state = session.state
        if state.samus_y >= 2440:
            break
        if state.samus_x >= 220:
            direction = "LEFT"
        elif state.samus_x <= 40:
            direction = "RIGHT"
        _hold(session, 1, direction, reason="red_lower_zigzag")
    else:
        raise TimeoutError(f"red_to_bat: lower descent stalled: {state}")

    _hold(session, 40, reason="red_bottom_settle")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(420):
        phase = frame % 30
        if phase < 5:
            buttons = ("RIGHT", "B", "X")
        elif phase >= 21:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="red_bottom_exit")
        if state.room_id == ROOM_BAT:
            break
    else:
        raise TimeoutError(f"red_to_bat: Bat Room door not reached: {state}")
    # Prefer a high left-sill settle (y<=125) so Bat→Below can use the short
    # natural run-up. Continuous door transitions sometimes linger and drop
    # Samus lower; LEFT brake helps, and bat_to_below_spazer also has a low
    # entry path if we still land around y~155.
    for frame in range(240):
        state = _hold(session, 1, "LEFT", reason="red_to_bat_brake")
        if (
            state.room_id == ROOM_BAT
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 8
            and state.samus_y <= 125
        ):
            return state
        if (
            state.room_id == ROOM_BAT
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 40
            and state.velocity_y == 0
        ):
            # Grounded (possibly low left ledge) — hand off to bat_to_below.
            return state
    return _wait_ordinary_room(session, ROOM_BAT, settle_frames=60, label="red_to_bat")



def play_bat_to_below_spazer(session: ControllerSession) -> SuperMetroidState:
    """Cross Bat Room's three dry platforms and enter Below Spazer.

    Two entry heights are supported:
    - High left sill (y<=125): natural door-exit glide + short run-up (pure
      anchors and continuous red→bat when the sill is secured).
    - Low left ledge (~y155) or mid-fall continuous tip: wait to ground, then
      a longer spin-jump reaches the first dry platform.
    """
    _require_room(session, ROOM_BAT, "bat_to_below_spazer")
    _unmorph(session)
    _select_weapon(session, 0)

    state = session.state
    if state.samus_y > 125 or abs(state.velocity_y) > 0:
        for _ in range(60):
            state = _hold(session, 1, reason="bat_land_wait")
            if state.velocity_y == 0 and state.pose in (
                1,
                2,
                25,
                26,
                27,
                28,
                137,
                138,
            ):
                break
        _unmorph(session)

    if session.state.samus_y <= 125:
        # High sill — preserve a short door-exit glide then original timings.
        _hold(session, 5, reason="bat_entry_glide")
        _hold(session, 35, "RIGHT", "B", reason="bat_first_runup")
        _hold(session, 60, "RIGHT", "B", "A", reason="bat_first_jump")
        _hold(session, 30, reason="bat_first_land")
    else:
        # Low left ledge / post-fall continuous entry.
        _hold(session, 8, reason="bat_low_ready")
        _hold(session, 15, "RIGHT", "B", reason="bat_low_runup")
        _hold(session, 80, "RIGHT", "B", "A", reason="bat_low_jump")
        _hold(session, 40, reason="bat_low_land")

    state = session.state
    if not (state.samus_x >= 210 and state.samus_y <= 165):
        raise TimeoutError(
            f"bat_to_below_spazer: missed first platform: {state}"
        )

    _hold(session, 8, "RIGHT", "B", reason="bat_second_runup")
    _hold(session, 20, "RIGHT", "B", "A", reason="bat_second_jump")
    _hold(session, 80, reason="bat_second_land")
    state = session.state
    if not (330 <= state.samus_x <= 400 and state.samus_y <= 185):
        raise TimeoutError(f"bat_to_below_spazer: missed middle platform: {state}")

    _hold(session, 48, "RIGHT", "B", "A", reason="bat_third_jump")
    _hold(session, 60, reason="bat_third_land")
    if session.state.samus_x < 400:
        raise TimeoutError(
            f"bat_to_below_spazer: missed right platform: {session.state}"
        )
    return play_run_shoot_exit(
        session,
        from_room=ROOM_BAT,
        to_room=ROOM_BELOW_SPAZER,
        direction="RIGHT",
        label="bat_to_below_spazer",
        run_frames=20,
        shoot_frames=4,
        spin_frames=30,
        hold_frames=240,
        settle_frames=260,
    )



def play_below_spazer_to_west(session: ControllerSession) -> SuperMetroidState:
    """Below Spazer water room → West Tunnel."""
    _require_room(session, ROOM_BELOW_SPAZER, "below_spazer_to_west")
    # Let the door-exit running pose settle before `_unmorph`; pose 9/10 is
    # intentionally handled by that shared helper and would otherwise turn
    # this ordinary entry glide into an unwanted jump.
    _hold(session, 6, reason="below_spazer_entry_glide")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(2000):
        buttons = ("RIGHT", "B", "X") if frame % 35 < 10 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="below_spazer_right")
        if state.room_id == ROOM_WEST_TUNNEL:
            break
    else:
        raise TimeoutError(f"below_spazer_to_west: West Tunnel not reached: {state}")
    return _wait_ordinary_room(
        session, ROOM_WEST_TUNNEL, settle_frames=260, label="below_spazer_to_west"
    )



def play_west_to_glass(session: ControllerSession) -> SuperMetroidState:
    """West Tunnel → Glass Tunnel."""
    return play_run_shoot_exit(
        session,
        from_room=ROOM_WEST_TUNNEL,
        to_room=ROOM_GLASS,
        direction="RIGHT",
        label="west_to_glass",
        run_frames=80,
        shoot_frames=5,
        spin_frames=50,
        hold_frames=300,
        settle_frames=260,
    )



def play_glass_to_east(session: ControllerSession) -> SuperMetroidState:
    """Glass Tunnel → East Tunnel."""
    return play_run_shoot_exit(
        session,
        from_room=ROOM_GLASS,
        to_room=ROOM_EAST_TUNNEL,
        direction="RIGHT",
        label="glass_to_east",
        run_frames=80,
        shoot_frames=5,
        spin_frames=50,
        hold_frames=300,
        settle_frames=260,
    )



def play_east_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """East Tunnel → natural Warehouse Entrance spawn."""
    _require_room(session, ROOM_EAST_TUNNEL, "east_to_warehouse")
    for frame in range(1600):
        phase = frame % 25
        if phase < 5:
            buttons = ("RIGHT", "B", "X")
        elif phase >= 18:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="east_tunnel_right")
        if state.room_id == ROOM_WAREHOUSE:
            break
    else:
        raise TimeoutError(f"east_to_warehouse: Warehouse not reached: {state}")
    # This multi-screen room commonly remains in game state 11 for >100f.
    return _wait_ordinary_room(
        session, ROOM_WAREHOUSE, settle_frames=900, label="east_to_warehouse"
    )



def play_red_tower_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Compose the verified controller-only Red Tower → Warehouse prefix."""
    play_red_tower_to_bat(session)
    play_bat_to_below_spazer(session)
    play_below_spazer_to_west(session)
    play_west_to_glass(session)
    play_glass_to_east(session)
    return play_east_to_warehouse(session)



