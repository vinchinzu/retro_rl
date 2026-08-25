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
6. **WS Basement → Phantoon room** — :func:`play_ws_basement_to_phantoon`
   (unpowered hallway + morph-tunnel bomb + Gadora Super). No fight.
   Fight is the ``--to phantoon`` SpineHop ``phantoon_fight``
   (wiki doppler; :func:`super_metroid.routes.kpdr.k6.phantoon_fight.play_phantoon_room_fight`).
   Loot + left-door exit is ``phantoon_loot_exit``
   (:func:`super_metroid.routes.kpdr.k6.phantoon_leave.play_phantoon_loot_exit`).
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold
from super_metroid.routes.kpdr import west_ocean as _west_ocean
from super_metroid.routes.kpdr.k6 import ws_basement as _ws_basement
from super_metroid.routes.kpdr.k6 import ws_entrance as _ws_entrance
from super_metroid.routes.kpdr.k6 import ws_main as _ws_main
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CRATERIA_KIHUNTER as ROOM_KIHUNTER,
    ROOM_MOAT,
    ROOM_PHANTOON,
    ROOM_WEST_OCEAN,
    ROOM_WS_BASEMENT,
    ROOM_WS_ENTRANCE,
    ROOM_WS_MAIN,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.shinespark import ChargeMode

# Product pure: over-ocean spark + Super open (re-exported for WS callers).
play_west_ocean_to_ws = _west_ocean.play_west_ocean_to_ws
play_west_ocean_over_ocean_spark = _west_ocean.play_west_ocean_over_ocean_spark

ROOM_WS_ATTIC = _ws_main.ROOM_WS_ATTIC
ROOM_WS_SAVE = _ws_main.ROOM_WS_SAVE
WEAPON_BEAM = _ws_entrance.WEAPON_BEAM
WEAPON_SUPER = _ws_main.WEAPON_SUPER
WS_ENTRANCE_DOOR_X_MIN = _ws_entrance.WS_ENTRANCE_DOOR_X_MIN
WS_ENTRANCE_DOOR_X_MAX = _ws_entrance.WS_ENTRANCE_DOOR_X_MAX
WS_MAIN_SAVE_X = _ws_main.WS_MAIN_SAVE_X
WS_MAIN_HOLE_Y = _ws_main.WS_MAIN_HOLE_Y
WS_MAIN_BOTTOM_Y = _ws_main.WS_MAIN_BOTTOM_Y
WS_MAIN_HATCH_X_MIN = _ws_main.WS_MAIN_HATCH_X_MIN
WS_MAIN_HATCH_X_MAX = _ws_main.WS_MAIN_HATCH_X_MAX
_WS_MAIN_RLE = _ws_main._WS_MAIN_RLE

at_ws_entrance_door_seat = _ws_entrance.at_ws_entrance_door_seat
ws_entrance_to_main_action = _ws_entrance.ws_entrance_to_main_action
ws_entrance_main_settled = _ws_entrance.ws_entrance_main_settled
play_ws_entrance_to_main = _ws_entrance.play_ws_entrance_to_main

at_ws_main_green_floor = _ws_main.at_ws_main_green_floor
ws_main_to_basement_action = _ws_main.ws_main_to_basement_action
ws_main_basement_settled = _ws_main.ws_main_basement_settled
play_ws_main_to_basement = _ws_main.play_ws_main_to_basement

play_ws_basement_to_phantoon = _ws_basement.play_ws_basement_to_phantoon
ws_basement_phantoon_settled = _ws_basement.ws_basement_phantoon_settled
at_ws_basement_bomb_blocks = _ws_basement.at_ws_basement_bomb_blocks
at_ws_basement_eye_seat = _ws_basement.at_ws_basement_eye_seat
WS_BASEMENT_ALCOVE_X = _ws_basement.WS_BASEMENT_ALCOVE_X
WS_BASEMENT_BOMB_X_MIN = _ws_basement.WS_BASEMENT_BOMB_X_MIN
WS_BASEMENT_FLOOR_Y = _ws_basement.WS_BASEMENT_FLOOR_Y
WS_BASEMENT_MORPH_X_MIN = _ws_basement.WS_BASEMENT_MORPH_X_MIN


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
    "ws_basement_phantoon_settled",
    "at_ws_basement_bomb_blocks",
    "at_ws_basement_eye_seat",
    "WS_BASEMENT_ALCOVE_X",
    "WS_BASEMENT_BOMB_X_MIN",
    "WS_BASEMENT_FLOOR_Y",
    "WS_BASEMENT_MORPH_X_MIN",
]
