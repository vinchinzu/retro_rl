"""West Ocean shinespark controllers (K6).

Two pure surfaces
-----------------
1. **Over-ocean → green Super WS** (:func:`play_west_ocean_over_ocean_spark`)
   Product path. From Moat handoff lower-left ``~(49,1163)``: stutter/short
   charge → crouch-store → hop → horizontal spark across the ocean floor
   runway → Super pressure into Wrecked Ship Entrance ``0xCA08``.

2. **Spit edge → Bowling** (:func:`play_west_ocean_edge_spark`)
   VOD edge-turn-hop into mid-right blue ``0xC98E``. Useful for store/hop
   practice; **not** the Phantoon entry door.

Harness: **B**=dash, **A**=jump/activate, **DOWN**=store.

Over-ocean measured green (2026-08-10)
-------------------------------------
* Source: ``scratch/post_moat_west_ocean_spark.state`` natural ``~(49,1163)``
* Charge: ``stutter`` (also ``short``) on ocean-floor runway; full fails
  (falls off before echoes=4)
* Store arms ``$0A68``≈179; hop 4f A; pre-stand UP 4f then RIGHT+A
* Spark max_x≈2011 @ y≈1163 (green Super door lip)
* Super open → settle ``0xCA08`` ~(57,139) gs=8 (~627–650f dual)

Spit-edge bowling (prior)
-------------------------
* Spit settle ≈ ``(350, 587)`` after free-place ``(350, 550)``
* Edge ≈ ``(909, 472)``; back 8 / hop 4 → ``0xC98E`` Bowling
"""

from __future__ import annotations

from typing import Literal, Sequence

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, select_weapon
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills import shinespark as spark

ChargeMode = Literal["full", "short", "stutter"]


ROOM_WEST_OCEAN = 0x93FE
ROOM_BOWLING = 0xC98E  # mid-right blue (edge-spark door; not product WS)
ROOM_WS_ENTRANCE = 0xCA08  # lower green Super → product Phantoon entry

# Default hop-carry knobs (green from place spit 2026-08-06).
DEFAULT_BACK_FRAMES = 8
DEFAULT_HOP_FRAMES = 4
DEFAULT_EDGE_BUDGET = 80
DEFAULT_SPARK_TRAVEL = 400
DEFAULT_DOOR_BUDGET = 200

# Over-ocean → green Super (product) knobs — dual pure 2026-08-10.
DEFAULT_OCEAN_HOP_FRAMES = 4
DEFAULT_OCEAN_PRE_STAND = 4
DEFAULT_OCEAN_SPARK_TRAVEL = 500
DEFAULT_SUPER_DOOR_BUDGET = 280
# Ocean-floor free-place (optional); natural Moat handoff needs no place.
OCEAN_FLOOR_PLACE_XY = (48, 1140)

# Free-place spit used when starting from Moat handoff for edge-spark only.
SPIT_PLACE_XY = (350, 550)


def _snap(session: ControllerSession) -> dict:
    try:
        return spark.spark_snapshot(session.env, int(session.frame))
    except Exception:  # noqa: BLE001
        st = session.state
        return {
            "x": st.samus_x,
            "y": st.samus_y,
            "pose": st.pose,
            "room": st.room_id,
            "spark_timer": st.shinespark_timer,
            "speed_echoes": st.speed_counter,
        }


def run_to_water_edge(
    session: ControllerSession,
    *,
    budget: int = DEFAULT_EDGE_BUDGET,
    y_slop: int = 24,
    label: str = "wo_edge",
) -> dict:
    """Hold RIGHT+B until water lip (y drop) or x stall."""
    y0 = int(session.state.samus_y)
    edge_x = int(session.state.samus_x)
    for i in range(budget):
        prev = int(session.state.samus_x)
        hold(session, 1, "RIGHT", "B", reason=f"{label}_{i}")
        st = session.state
        if int(st.samus_y) > y0 + y_slop:
            break
        if i > 4 and int(st.samus_x) <= prev:
            edge_x = int(st.samus_x)
            break
        edge_x = int(st.samus_x)
    return {"edge_x": edge_x, "after": _snap(session)}


def open_green_super_ws(
    session: ControllerSession,
    *,
    budget: int = DEFAULT_SUPER_DOOR_BUDGET,
    settle_frames: int = 120,
    label: str = "wo_super_ws",
) -> SuperMetroidState:
    """Select Supers and pressure the lower green door into ``0xCA08``.

    Expects Samus near the door band (x≳1900, y≈1100–1180) after an
    over-ocean spark (or free-place). Settles ordinary ``game_state==8``.
    """
    require_room(session, ROOM_WEST_OCEAN, label)
    try:
        select_weapon(session, 2)
    except RuntimeError:
        pass

    for i in range(budget):
        st = session.state
        if (
            st.room_id == ROOM_WS_ENTRANCE
            and st.door_transition == 0
            and st.game_state == 8
        ):
            hold(session, 8, reason=f"{label}_settle")
            return session.state
        if st.room_id == ROOM_WS_ENTRANCE:
            hold(session, 1, reason=f"{label}_trans")
            continue
        phase = i % 24
        if phase < 8:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_sup")
        elif phase < 14:
            hold(session, 1, "RIGHT", reason=f"{label}_face")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_run")

    for _ in range(settle_frames):
        st = session.state
        if (
            st.room_id == ROOM_WS_ENTRANCE
            and st.door_transition == 0
            and st.game_state == 8
        ):
            return st
        hold(session, 1, reason=f"{label}_final")

    st = session.state
    if st.room_id == ROOM_WS_ENTRANCE:
        return st
    raise TimeoutError(
        f"{label}: green Super did not open into 0xCA08 "
        f"(room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y})): {st}"
    )


def play_west_ocean_over_ocean_spark(
    session: ControllerSession,
    *,
    hop_frames: int = DEFAULT_OCEAN_HOP_FRAMES,
    pre_stand_frames: int = DEFAULT_OCEAN_PRE_STAND,
    spark_travel: int = DEFAULT_OCEAN_SPARK_TRAVEL,
    super_budget: int = DEFAULT_SUPER_DOOR_BUDGET,
    label: str = "wo_over_ocean",
    charge_mode: ChargeMode = "stutter",
) -> SuperMetroidState:
    """Ocean-floor short-charge → spark → green Super into WS ``0xCA08``.

    Expects Samus in West Ocean at the Moat handoff / ocean-floor runway
    (natural ``~(49,1163)`` after pure Moat spark). Product charge is
    ``stutter`` (``short`` also greens); continuous ``full`` usually falls
    off the runway before echoes=4.

    sm-json: *Over Ocean Spark, In-Room* (node 13 → door 5).
    """
    require_room(session, ROOM_WEST_OCEAN, label)

    charge = spark.charge_until_boost(
        session,
        "RIGHT",
        budget=300,
        label=f"{label}_charge",
        mode=charge_mode,
    )
    if not charge.get("ok"):
        raise TimeoutError(f"{label}: charge failed: {charge} state={session.state}")

    store = spark.crouch_store(session, label=f"{label}_store")
    if not store.get("ok"):
        raise TimeoutError(f"{label}: store failed: {store} state={session.state}")

    hold(session, 2, reason=f"{label}_pre_hop")
    if hop_frames > 0:
        hold(session, hop_frames, "A", reason=f"{label}_hop")

    act = spark.activate_shinespark(
        session,
        "RIGHT",
        pre_stand_frames=pre_stand_frames,
        pre_stand_buttons=("UP",),
        hold_frames=30,
        travel_budget=spark_travel,
        label=f"{label}_spark",
    )
    if not act.get("spark_pose_seen"):
        raise TimeoutError(
            f"{label}: spark did not arm/travel: {act} state={session.state}"
        )

    st = session.state
    if st.room_id == ROOM_WS_ENTRANCE:
        hold(session, 8, reason=f"{label}_spark_enter")
        return session.state

    return open_green_super_ws(
        session,
        budget=super_budget,
        label=f"{label}_super",
    )


def play_west_ocean_to_ws(
    session: ControllerSession,
    *,
    charge_mode: ChargeMode = "stutter",
    hop_frames: int = DEFAULT_OCEAN_HOP_FRAMES,
    label: str = "west_ocean_to_ws",
) -> SuperMetroidState:
    """Product alias: over-ocean spark + Super open → ``0xCA08``."""
    return play_west_ocean_over_ocean_spark(
        session,
        charge_mode=charge_mode,
        hop_frames=hop_frames,
        label=label,
    )


def play_west_ocean_edge_spark(
    session: ControllerSession,
    *,
    back_frames: int = DEFAULT_BACK_FRAMES,
    hop_frames: int = DEFAULT_HOP_FRAMES,
    aim_buttons: Sequence[str] = ("RIGHT",),
    edge_budget: int = DEFAULT_EDGE_BUDGET,
    spark_travel: int = DEFAULT_SPARK_TRAVEL,
    door_budget: int = DEFAULT_DOOR_BUDGET,
    accept_rooms: frozenset[int] | None = None,
    label: str = "wo_edge_spark",
    charge_mode: ChargeMode = "full",
) -> SuperMetroidState:
    """Charge → edge → store → turn back → hop → spark into mid-right door.

    Expects Samus already on a **dry spit** in West Ocean (not lower-left water).
    Default accept room is Bowling Alley ``0xC98E`` (practice / wrong door for
    Phantoon — use :func:`play_west_ocean_over_ocean_spark` for ``0xCA08``).

    ``charge_mode``: ``full`` (continuous dash), ``short`` (magic frames), or
    ``stutter`` (stutter-walk + short). Short modes charge near the spit start
    so the edge run can be a short walk with echoes already up — or charge
    while approaching the edge when space is tight.
    """
    require_room(session, ROOM_WEST_OCEAN, label)
    if accept_rooms is None:
        accept_rooms = frozenset({ROOM_BOWLING})

    # 1) speed charge (full continuous, or short/stutter magic-frame)
    charge = spark.charge_until_boost(
        session,
        "RIGHT",
        budget=250,
        label=f"{label}_charge",
        mode=charge_mode,
    )
    if not charge.get("ok"):
        raise TimeoutError(f"{label}: charge failed: {charge} state={session.state}")

    # 2) water edge (keep dash if full echoes so boost survives the approach)
    edge = run_to_water_edge(session, budget=edge_budget, label=f"{label}_edge")

    # 3) store AT edge (required — turning back without store wipes echoes)
    store = spark.crouch_store(session, label=f"{label}_store")
    if not store.get("ok"):
        raise TimeoutError(
            f"{label}: store failed at edge {edge}: {store} state={session.state}"
        )

    # 4) turn back a few steps
    if back_frames > 0:
        hold(session, back_frames, "LEFT", reason=f"{label}_back")

    # 5) hop up a few tiles
    hold(session, 2, reason=f"{label}_pre_hop")
    if hop_frames > 0:
        hold(session, hop_frames, "A", reason=f"{label}_hop")

    # 6) activate horizontal (or aimed) spark
    act = spark.activate_shinespark(
        session,
        *aim_buttons,
        pre_stand_frames=0 if spark.is_spark_pose(int(session.state.pose)) else 2,
        pre_stand_buttons=("UP",),
        hold_frames=20,
        travel_budget=spark_travel,
        label=f"{label}_spark",
    )

    # 7) finish door (blue: X + RIGHT)
    for i in range(door_budget):
        st = session.state
        if st.room_id in accept_rooms and st.door_transition == 0 and st.game_state == 8:
            hold(session, 8, reason=f"{label}_settle")
            return session.state
        if st.room_id in accept_rooms:
            hold(session, 1, reason=f"{label}_door_trans")
            continue
        if st.room_id != ROOM_WEST_OCEAN:
            # unexpected room — still return if left ocean
            hold(session, 8, reason=f"{label}_other_room")
            return session.state
        if i % 5 < 2:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_door_x")
        elif i < 40:
            hold(session, 1, "RIGHT", "A", reason=f"{label}_door_a")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_door_walk")

    # settle leftover transition
    for _ in range(120):
        st = session.state
        if st.door_transition == 0 and st.game_state == 8 and st.room_id != ROOM_WEST_OCEAN:
            return st
        hold(session, 1, reason=f"{label}_final_settle")

    st = session.state
    if st.room_id in accept_rooms or st.room_id != ROOM_WEST_OCEAN:
        return st
    raise TimeoutError(
        f"{label}: no door after edge spark "
        f"(room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
        f"act={act.get('spark_pose_seen')} max_x={act.get('max_x')} "
        f"edge={edge}): {st}"
    )


__all__ = [
    "DEFAULT_BACK_FRAMES",
    "DEFAULT_HOP_FRAMES",
    "DEFAULT_OCEAN_HOP_FRAMES",
    "DEFAULT_OCEAN_PRE_STAND",
    "OCEAN_FLOOR_PLACE_XY",
    "ROOM_BOWLING",
    "ROOM_WEST_OCEAN",
    "ROOM_WS_ENTRANCE",
    "SPIT_PLACE_XY",
    "open_green_super_ws",
    "play_west_ocean_edge_spark",
    "play_west_ocean_over_ocean_spark",
    "play_west_ocean_to_ws",
    "run_to_water_edge",
]
