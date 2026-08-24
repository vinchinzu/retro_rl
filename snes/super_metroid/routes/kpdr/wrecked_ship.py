"""Wrecked Ship approach controllers (K6).

Product pure shine chain (compose, natural-entry when sources allow)
--------------------------------------------------------------------
1. **Kihunter / Moat → West Ocean** — :func:`play_moat_to_west_ocean`
   (leave + open door stay + clear + left pin + hop spark).
2. **West Ocean → WS entrance** — :func:`play_west_ocean_to_ws`
   (ocean-floor stutter spark + green Super).
3. **Compose** — :func:`play_moat_to_ws` runs both; lands ``0xCA08`` for
   ship free-record / Phantoon approach (``guided_human --from ws-entrance``).
4. **WS entrance → Main Shaft** — :func:`play_ws_entrance_to_main`
   (unpowered 4-screen dash + beam blue door).
5. **WS Main Shaft → Basement** — :func:`play_ws_main_to_basement`
   (unpowered morph-stair descent + floor pipes + green Super hatch).

Phantoon remains a scaffold placeholder until pure geometry.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    hold_until,
    is_morph,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr import west_ocean as _west_ocean
from super_metroid.routes.rle import load_rle_json, play_script
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.shinespark import ChargeMode

_DATA = Path(__file__).resolve().parent / "data"
_WS_MAIN_RLE = load_rle_json(_DATA / "ws_main_to_basement_rle.json")

# Product pure: over-ocean spark + Super open (re-exported for WS callers).
play_west_ocean_to_ws = _west_ocean.play_west_ocean_to_ws
play_west_ocean_over_ocean_spark = _west_ocean.play_west_ocean_over_ocean_spark


ROOM_KIHUNTER = 0x948C
ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE
ROOM_WS_ENTRANCE = 0xCA08
ROOM_WS_MAIN = 0xCAF6
ROOM_WS_BASEMENT = 0xCC6F
ROOM_PHANTOON = 0xCD13
ROOM_WS_ATTIC = 0xCA52
ROOM_WS_SAVE = 0xCE8A

# Beam, not Super — pin after the green Super still has selected_item=2.
WEAPON_BEAM = 0
WEAPON_SUPER = 2
WEAPON_PB = 3
# Dump from post_ws_poweron: first x>=960 at ~(968,139) p9 speed=4; closed
# blue-door crash wall is x=987 p137. Start beam pressure before that wall.
WS_ENTRANCE_DOOR_X_MIN = 900
WS_ENTRANCE_DOOR_X_MAX = 1024
_WS_ENTRANCE_RUN_TIMEOUT = 400
_WS_ENTRANCE_SETTLE = 200

_MAX_SCAFFOLD_FRAMES = 240

# Main Shaft descent (rr-4btp dumps from post_ws_entrance_to_main).
# Pin (1063,907) p9 is the entry ledge. Save is x≳1240 at this y — do not enter.
# Morph-only hole in the grated floor around x∈[1140,1176] drops onto stairs
# at y≳950. Ping-pong morph-roll reaches the node-3 platform y≳1650.
# Green floor hatch is below x∈[1135,1165]; shoot pipes, Super, drop.
WS_MAIN_SAVE_X = 1240
WS_MAIN_HOLE_Y = 950
WS_MAIN_BOTTOM_Y = 1650
WS_MAIN_HATCH_X_MIN = 1135
WS_MAIN_HATCH_X_MAX = 1165
# Pipe hole after aim-down shots is the hatch well, slightly left of the right lip.
WS_MAIN_SUPER_X_MIN = 1128
WS_MAIN_SUPER_X_MAX = 1145
WS_MAIN_ATTIC_Y = 850
_WS_MAIN_SETTLE = 200
_WS_MAIN_PONG_FRAMES = 2500
_WS_MAIN_DROP_FRAMES = 480


def at_ws_entrance_door_seat(state: SuperMetroidState) -> bool:
    """True on the right-door approach band of unpowered Entrance ``0xCA08``."""
    x = int(state.samus_x)
    return (
        int(state.room_id) == ROOM_WS_ENTRANCE
        and WS_ENTRANCE_DOOR_X_MIN <= x <= WS_ENTRANCE_DOOR_X_MAX
    )


def ws_entrance_to_main_action(state: SuperMetroidState) -> tuple[str, ...]:
    """One-frame buttons. Cycle to beam before any X; never Super the blue door."""
    room = int(state.room_id)
    if room == ROOM_WS_MAIN:
        return ()
    if int(state.selected_item) != WEAPON_BEAM:
        return ("SELECT",)
    if room != ROOM_WS_ENTRANCE:
        return ()
    if int(state.samus_x) < WS_ENTRANCE_DOOR_X_MIN:
        return ("RIGHT", "B")
    return ("RIGHT", "B", "X")


def ws_entrance_main_settled(state: SuperMetroidState) -> bool:
    """Ordinary Main Shaft handoff: room ``0xCAF6`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def ws_main_basement_settled(state: SuperMetroidState) -> bool:
    """Ordinary Basement handoff: room ``0xCC6F`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_main_green_floor(state: SuperMetroidState) -> bool:
    """True over the green floor-hatch band of unpowered Main Shaft."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and WS_MAIN_HATCH_X_MIN <= int(state.samus_x) <= WS_MAIN_HATCH_X_MAX
        and int(state.samus_y) >= WS_MAIN_BOTTOM_Y
    )


def ws_main_to_basement_action(state: SuperMetroidState) -> tuple[str, ...]:
    """One-frame buttons. Never UP (attic). Super the green floor door, not save."""
    room = int(state.room_id)
    if room == ROOM_WS_BASEMENT:
        return ()
    if room != ROOM_WS_MAIN:
        return ()
    y = int(state.samus_y)
    x = int(state.samus_x)
    if y < WS_MAIN_ATTIC_Y:
        return ("DOWN",)
    if y < WS_MAIN_HOLE_Y:
        if not is_morph(int(state.pose)):
            return ("DOWN",)
        return ("RIGHT",)
    if y < WS_MAIN_BOTTOM_Y:
        return ("DOWN", "RIGHT") if int(state.samus_x) < 1150 else ("DOWN", "LEFT")
    if not at_ws_main_green_floor(state):
        return ("RIGHT",) if x < WS_MAIN_HATCH_X_MIN else ("LEFT",)
    if int(state.selected_item) != WEAPON_SUPER:
        return ("SELECT",)
    return ("DOWN", "X")


def _ws_main_guard(session: ControllerSession, label: str) -> None:
    """Fail loud on attic / save / entrance; basement is success."""
    room = int(session.state.room_id)
    if room == ROOM_WS_ATTIC:
        raise TimeoutError(f"{label}: climbed to attic 0xCA52: {session.state}")
    if room == ROOM_WS_SAVE:
        raise TimeoutError(f"{label}: entered save 0xCE8A: {session.state}")
    if room == ROOM_WS_ENTRANCE:
        raise TimeoutError(f"{label}: back through entrance 0xCA08: {session.state}")
    if room not in (ROOM_WS_MAIN, ROOM_WS_BASEMENT):
        raise TimeoutError(
            f"{label}: left Main Shaft into 0x{room:04X}: {session.state}"
        )


def _ws_main_align_x(
    session: ControllerSession, x_lo: int, x_hi: int, *, label: str, budget: int = 160
) -> None:
    if is_morph(int(session.state.pose)):
        unmorph(session)
        hold(session, 6, "UP", reason=f"{label}_align_unmorph")
    for _ in range(budget):
        _ws_main_guard(session, label)
        if int(session.state.room_id) == ROOM_WS_BASEMENT:
            return
        x = int(session.state.samus_x)
        if x_lo <= x <= x_hi:
            return
        hold(session, 1, "RIGHT" if x < x_lo else "LEFT", reason=f"{label}_align")
    # Pit walls can clip x; keep going if we are already over the hatch band.


def _ws_main_morph_hole(session: ControllerSession, label: str) -> None:
    """Morph-roll RIGHT through the grated-floor hole onto the first stairs."""
    ensure_morph(session)
    for _ in range(160):
        st = session.state
        _ws_main_guard(session, label)
        if int(st.room_id) == ROOM_WS_BASEMENT:
            return
        if int(st.samus_y) >= WS_MAIN_HOLE_Y and int(st.velocity_y) == 0:
            return
        if int(st.samus_x) >= WS_MAIN_SAVE_X and int(st.samus_y) < WS_MAIN_HOLE_Y:
            raise TimeoutError(f"{label}: missed morph hole, at save band: {st}")
        hold(session, 1, "RIGHT", reason=f"{label}_hole")
    raise TimeoutError(f"{label}: morph hole did not drop: {session.state}")


def _ws_main_morph_pong(session: ControllerSession, label: str) -> None:
    """LEFT/RIGHT morph-roll switchbacks down the shaft to the node-3 platform."""
    hold(session, 6, reason=f"{label}_hole_settle")
    direction = "LEFT"
    stuck = 0
    prev = (int(session.state.samus_x), int(session.state.samus_y))
    for _ in range(_WS_MAIN_PONG_FRAMES):
        st = session.state
        _ws_main_guard(session, label)
        if int(st.room_id) == ROOM_WS_BASEMENT:
            return
        if int(st.samus_y) >= WS_MAIN_BOTTOM_Y and int(st.velocity_y) == 0:
            return
        if int(st.samus_y) < WS_MAIN_ATTIC_Y:
            raise TimeoutError(f"{label}: went up toward attic: {st}")
        hold(session, 1, direction, reason=f"{label}_pong")
        st = session.state
        x, y = int(st.samus_x), int(st.samus_y)
        dx, dy = abs(x - prev[0]), abs(y - prev[1])
        stuck = stuck + 1 if dx < 2 and dy < 2 and int(st.velocity_y) == 0 else 0
        prev = (x, y)
        if stuck > 22:
            direction = "LEFT" if direction == "RIGHT" else "RIGHT"
            stuck = 0
    raise TimeoutError(f"{label}: stairs ping-pong missed bottom: {session.state}")


def _ws_main_shoot_pipes(session: ControllerSession, label: str) -> None:
    """Morph PB the floor pipes (human s21). Clears shot-block hatch cover."""
    if not is_morph(int(session.state.pose)):
        ensure_morph(session)
    for _ in range(80):
        _ws_main_guard(session, label)
        if int(session.state.room_id) == ROOM_WS_BASEMENT:
            return
        y = int(session.state.samus_y)
        x = int(session.state.samus_x)
        if y >= 1715 and WS_MAIN_HATCH_X_MIN <= x <= WS_MAIN_HATCH_X_MAX:
            break
        hold(session, 1, "RIGHT" if x < 1143 else "LEFT", reason=f"{label}_pb_seat")
    select_weapon(session, WEAPON_PB)
    hold(session, 4, "X", reason=f"{label}_pb")
    hold(session, 90, reason=f"{label}_pb_boom")


def _ws_main_super_drop(session: ControllerSession, label: str) -> None:
    """Jump-down Super the green floor hatch, then hold DOWN through.

    Human s21 hop_01 cadence (full_start_v1): A, DOWN+A, DOWN+A+X, DOWN+X,
    hold DOWN. Morph on the hatch cannot fire. Standing L+X hits the ledge.
    """
    if is_morph(int(session.state.pose)):
        unmorph(session)
        hold(session, 8, "UP", reason=f"{label}_hatch_unmorph")
        hold(session, 10, reason=f"{label}_hatch_unmorph_settle")
    _ws_main_align_x(
        session, WS_MAIN_SUPER_X_MIN, WS_MAIN_SUPER_X_MAX, label=f"{label}_hatch"
    )
    hold(session, 12, reason=f"{label}_hatch_brake")
    hold(session, 8, "LEFT", reason=f"{label}_hatch_face")
    hold(session, 10, reason=f"{label}_hatch_face_settle")
    select_weapon(session, WEAPON_SUPER)
    for _ in range(4):
        if int(session.state.room_id) == ROOM_WS_BASEMENT:
            return
        _ws_main_guard(session, label)
        hold(session, 8, "A", reason=f"{label}_super_hop")
        hold(session, 8, "DOWN", "A", reason=f"{label}_super_downjump")
        hold(session, 8, "DOWN", "A", "X", reason=f"{label}_super_airshot")
        hold(session, 8, "DOWN", "X", reason=f"{label}_super_shot")
        hold(session, 48, "DOWN", reason=f"{label}_super_fall")
        hold(session, 12, reason=f"{label}_super_land")
    for _ in range(_WS_MAIN_DROP_FRAMES):
        st = session.state
        _ws_main_guard(session, label)
        if int(st.room_id) == ROOM_WS_BASEMENT:
            return
        hold(session, 1, "DOWN", reason=f"{label}_drop")
    raise TimeoutError(f"{label}: green floor drop missed: {session.state}")


def _scaffold_exit(
    session: ControllerSession,
    *,
    entry_room: int,
    target_room: int,
    label: str,
) -> SuperMetroidState:
    """Run a bounded placeholder toward the next ship-room door."""
    require_room(session, entry_room, label)

    # TODO(SM-WS-PURE): replace with source-state-driven room geometry.
    for _ in range(_MAX_SCAFFOLD_FRAMES):
        state = hold(session, 1, "RIGHT", "B", reason=f"{label}_scaffold")
        if state.room_id == target_room:
            return state

    state = session.state
    raise TimeoutError(
        f"{label}: scaffold timeout before room 0x{target_room:04X}; "
        f"room=0x{state.room_id:04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y})"
    )


def play_moat_to_west_ocean(
    session: ControllerSession,
    *,
    charge_mode: ChargeMode = "full",
) -> SuperMetroidState:
    """Moat ``0x95FF`` (or Kihunter) → West Ocean via pure shinespark.

    Delegates to :func:`super_metroid.routes.kpdr.moat.play_moat_shinespark`
    (leave Moat → open door stay → clear → left pin → spark). Product charge
    is continuous ``full``; ``short``/``stutter`` are for cramped runways only.
    """
    from super_metroid.routes.kpdr.moat import play_moat_shinespark

    st = session.state
    if st.room_id not in (ROOM_KIHUNTER, ROOM_MOAT):
        raise TimeoutError(
            f"moat_to_west_ocean: expected Kihunter 0x948C or Moat 0x95FF, "
            f"got 0x{st.room_id:04X}"
        )
    return play_moat_shinespark(session, charge_mode=charge_mode)


def play_moat_to_ws(
    session: ControllerSession,
    *,
    moat_charge_mode: ChargeMode = "full",
    wo_charge_mode: ChargeMode = "stutter",
    label: str = "moat_to_ws",
) -> SuperMetroidState:
    """Compose product shine path: Kihunter/Moat → West Ocean → WS ``0xCA08``.

    Room-dispatch for continuous-style natural entry:

    * ``0x948C`` / ``0x95FF`` — Moat spark (full charge product) then over-ocean
    * ``0x93FE`` — over-ocean only (already at Moat handoff)
    * ``0xCA08`` — already in WS entrance (no-op settle)

    Dual pure green from product pin and human Moat end; pin-only until full
    continuous power-on compose. Handoff for ship free-record / Phantoon:

    ``guided_human --from ws-entrance`` / ``practice_takes --segment ws-entrance``.
    """
    st = session.state
    if st.room_id == ROOM_WS_ENTRANCE:
        if st.door_transition == 0 and st.game_state == 8:
            return st
        hold(session, 12, reason=f"{label}_ws_settle")
        return session.state

    if st.room_id in (ROOM_KIHUNTER, ROOM_MOAT):
        play_moat_to_west_ocean(session, charge_mode=moat_charge_mode)
        st = session.state

    if st.room_id == ROOM_WEST_OCEAN:
        return play_west_ocean_to_ws(
            session,
            charge_mode=wo_charge_mode,
            label=f"{label}_wo",
        )

    raise TimeoutError(
        f"{label}: expected Kihunter/Moat/West Ocean/WS entrance, "
        f"got 0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y})"
    )


def play_ws_entrance_to_main(session: ControllerSession) -> SuperMetroidState:
    """Unpowered 4-screen hallway. Walk/run right. Coverns only. Blue door into Main Shaft.

    https://wiki.supermetroid.run/Wrecked_Ship_Entrance

    Do not invent a fight. Energy assist is on — tank Coverns if they touch.
    Shoot the blue door (beam, not Super). Lands ordinary ``gs=8`` in Main
    Shaft ``0xCAF6`` (game state 11 can last 50–100+f).

    Source: ``scratch/post_ws_poweron.state`` ``0xCA08`` ~(57,139) p1 gs=8.
    """
    label = "ws_entrance_to_main"
    require_room(session, ROOM_WS_ENTRANCE, label)
    select_weapon(session, WEAPON_BEAM)

    def _at_seat_or_main(state: SuperMetroidState) -> bool:
        return int(state.room_id) == ROOM_WS_MAIN or at_ws_entrance_door_seat(state)

    hold_until(
        session,
        _at_seat_or_main,
        "RIGHT",
        "B",
        timeout=_WS_ENTRANCE_RUN_TIMEOUT,
        reason=f"{label}_run",
    )
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return wait_ordinary_room(
            session, ROOM_WS_MAIN, settle_frames=_WS_ENTRANCE_SETTLE, label=label
        )
    return play_run_shoot_exit(
        session,
        from_room=ROOM_WS_ENTRANCE,
        to_room=ROOM_WS_MAIN,
        direction="RIGHT",
        label=label,
        run_frames=0,
        shoot_frames=10,
        spin_frames=0,
        hold_frames=200,
        settle_frames=_WS_ENTRANCE_SETTLE,
        super_door=False,
    )


def play_ws_main_to_basement(session: ControllerSession) -> SuperMetroidState:
    """Unpowered first visit toward Phantoon. Descend stairs. Coverns/Sbugs only.

    Pin is mid-height on the right of a wide room (x=1063; save is across —
    do NOT enter the save). Descend the stairs (do NOT go UP = Attic 0xCA52).
    Ignore grey locked doors. Skip the optional left-wall missile. At the
    bottom: shoot the floor pipes (aim down), morph, Super the green floor
    hatch, drop into Basement ``0xCC6F``. Unpowered Atomics stay in glass —
    ignore. Energy assist on — tank.

    https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft

    Human s21 hop body (1091f) from this pin dual-greens: dash switchbacks,
    morph PB the floor pipes, jump-down Super, hold DOWN through the hatch.
    Lands ordinary ``gs=8`` (game state 11 can last 50–100+f).
    """
    label = "ws_main_to_basement"
    require_room(session, ROOM_WS_MAIN, label)
    if ws_main_basement_settled(session.state):
        return session.state
    play_script(
        session,
        _WS_MAIN_RLE,
        reason=f"{label}_body",
        room_id=ROOM_WS_MAIN,
        stop_when=lambda state: int(state.room_id) != ROOM_WS_MAIN,
    )
    return wait_ordinary_room(
        session, ROOM_WS_BASEMENT, settle_frames=_WS_MAIN_SETTLE, label=label
    )


def play_ws_basement_to_phantoon(session: ControllerSession) -> SuperMetroidState:
    """Scaffold WS basement ``0xCC6F`` -> Phantoon ``0xCD13``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_BASEMENT,
        target_room=ROOM_PHANTOON,
        label="ws_basement_to_phantoon",
    )


__all__ = [
    "ROOM_KIHUNTER",
    "ROOM_MOAT",
    "ROOM_WEST_OCEAN",
    "ROOM_WS_ENTRANCE",
    "ROOM_WS_MAIN",
    "ROOM_WS_BASEMENT",
    "ROOM_PHANTOON",
    "ROOM_WS_ATTIC",
    "ROOM_WS_SAVE",
    "WEAPON_BEAM",
    "WEAPON_SUPER",
    "WS_ENTRANCE_DOOR_X_MIN",
    "WS_ENTRANCE_DOOR_X_MAX",
    "WS_MAIN_SAVE_X",
    "WS_MAIN_HOLE_Y",
    "WS_MAIN_BOTTOM_Y",
    "WS_MAIN_HATCH_X_MIN",
    "WS_MAIN_HATCH_X_MAX",
    "at_ws_entrance_door_seat",
    "at_ws_main_green_floor",
    "ws_entrance_to_main_action",
    "ws_entrance_main_settled",
    "ws_main_to_basement_action",
    "ws_main_basement_settled",
    "play_moat_to_west_ocean",
    "play_moat_to_ws",
    "play_west_ocean_over_ocean_spark",
    "play_west_ocean_to_ws",
    "play_ws_entrance_to_main",
    "play_ws_main_to_basement",
    "play_ws_basement_to_phantoon",
]
