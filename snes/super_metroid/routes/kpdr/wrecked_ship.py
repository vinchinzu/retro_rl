"""Wrecked Ship approach controllers (K6).

Product pure shine chain (compose, natural-entry when sources allow)
--------------------------------------------------------------------
1. **Kihunter / Moat → West Ocean** — :func:`play_moat_to_west_ocean`
   (leave + open door stay + clear + left pin + hop spark).
2. **West Ocean → WS entrance** — :func:`play_west_ocean_to_ws`
   (ocean-floor stutter spark + green Super).
3. **Compose** — :func:`play_moat_to_ws` runs both; lands ``0xCA08`` for
   ship free-record / Phantoon approach (``guided_human --from ws-entrance``).

Post-entry ship rooms remain scaffold placeholders until pure geometry.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
from super_metroid.routes.kpdr import west_ocean as _west_ocean
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.shinespark import ChargeMode

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

_MAX_SCAFFOLD_FRAMES = 240


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
    """Scaffold WS entrance ``0xCA08`` -> WS main/attic ``0xCAF6``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_ENTRANCE,
        target_room=ROOM_WS_MAIN,
        label="ws_entrance_to_main",
    )


def play_ws_main_to_basement(session: ControllerSession) -> SuperMetroidState:
    """Scaffold WS main/attic ``0xCAF6`` -> basement ``0xCC6F``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_MAIN,
        target_room=ROOM_WS_BASEMENT,
        label="ws_main_to_basement",
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
    "play_moat_to_west_ocean",
    "play_moat_to_ws",
    "play_west_ocean_over_ocean_spark",
    "play_west_ocean_to_ws",
    "play_ws_entrance_to_main",
    "play_ws_main_to_basement",
    "play_ws_basement_to_phantoon",
]
