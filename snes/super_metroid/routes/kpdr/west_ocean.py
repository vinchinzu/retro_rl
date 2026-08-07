"""West Ocean edge-turn-hop shinespark (K6).

VOD / human recipe (screenshots 2026-08-06 14:50–14:52)
-------------------------------------------------------
On a **dry spit** above the water:

1. ``RIGHT+B`` to full charge (echoes ≥ 4)
2. Run to the **water edge**
3. **Crouch-store** (DOWN) — arms ``$0A68`` ≈ 179
4. **Turn back LEFT** a few steps (store survives; do not re-charge)
5. **Jump up** a few tiles (A hop)
6. **Shinespark RIGHT** (optional RIGHT+UP) into the mid-right door band

Harness: **B**=dash, **A**=jump/activate, **DOWN**=store.

Measured green (place-bootstrap or climb-to-spit)
------------------------------------------------
* Spit settle ≈ ``(350, 587)`` after free-place ``(350, 550)`` on
  ``scratch/post_moat_west_ocean_spark.state`` loadout.
* Edge ≈ ``(909, 472)`` echoes=4
* Store at edge → back 8f LEFT → hop 4f A → ``RIGHT+A`` spark
* Spark travel max_x ≈ 2011, lands door band y≈395
* Door reached: ``0xC98E`` Bowling Alley (mid-right blue) — **not** the lower
  green Super into Wrecked Ship ``0xCA08`` (that door is underwater; no Speed
  charge without Gravity).

Natural climb from Moat handoff lower-left ``~(49,1163)`` onto the spit is
still open work; pure path currently accepts a free-place spit bootstrap or
an already-elevated pin.
"""

from __future__ import annotations

from typing import Sequence

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills import shinespark as spark


ROOM_WEST_OCEAN = 0x93FE
ROOM_BOWLING = 0xC98E  # mid-right blue (current pure door)
ROOM_WS_ENTRANCE = 0xCA08  # lower green Super — not this controller yet

# Default hop-carry knobs (green from place spit 2026-08-06).
DEFAULT_BACK_FRAMES = 8
DEFAULT_HOP_FRAMES = 4
DEFAULT_EDGE_BUDGET = 80
DEFAULT_SPARK_TRAVEL = 400
DEFAULT_DOOR_BUDGET = 200

# Free-place spit used when starting from Moat handoff lower-left water.
# Controllers that own env may place; pure session path expects elevated pin.
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
) -> SuperMetroidState:
    """Charge → edge → store → turn back → hop → spark into mid-right door.

    Expects Samus already on a **dry spit** in West Ocean (not lower-left water).
    Default accept room is Bowling Alley ``0xC98E`` (measured pure).
    """
    require_room(session, ROOM_WEST_OCEAN, label)
    if accept_rooms is None:
        accept_rooms = frozenset({ROOM_BOWLING})

    # 1) full speed charge
    charge = spark.charge_until_boost(session, "RIGHT", budget=250, label=f"{label}_charge")
    if not charge.get("ok"):
        raise TimeoutError(f"{label}: charge failed: {charge} state={session.state}")

    # 2) water edge
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
    "ROOM_BOWLING",
    "ROOM_WEST_OCEAN",
    "ROOM_WS_ENTRANCE",
    "SPIT_PLACE_XY",
    "play_west_ocean_edge_spark",
    "run_to_water_edge",
]
