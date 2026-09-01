"""Morph spine: Ceres prefix + Landing→Morph seeds.

Ceres boot / outbound / escape live in :mod:`super_metroid.routes.kpdr.ceres.spine`.
This module composes that prefix with hash-pinned Crateria seeds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.paths import POLICY_DIR
from super_metroid.progression.types import (
    DoorEdge,
    ProgressCondition,
    ProgressionMilestone,
)
from super_metroid.ram import MORPH_BALL_MASK
from super_metroid.routes.kpdr.ceres import (
    _CERES_ARM_PUMP_PERIOD,
    _arm_pump_dash_spans,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.ceres.spine import (
    CERES_DOOR_EDGES,
    CERES_MILESTONES,
    CERES_SPINE,
    _BOOT_MAX_FRAMES,
    _BOOT_MENU_MASH_FRAMES,
    _boot_spans,
    play_boot_to_ceres,
    play_boot_to_ceres_tas,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CLIMB,
    ROOM_CONSTRUCTION,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.routes.kpdr.spine import SpineHop
from super_metroid.routes.runtime import Action, ActionSpan, RouteSession, Split

__all__ = [
    "MORPH_SPINE",
    "MORPH_DOOR_EDGES",
    "MORPH_MILESTONES",
    "play_morph_hops",
    "play_ship_to_morph",
    "morph_route_hops",
    "continuous_edges_from_morph_spine",
    "validate_morph_spine",
    # Stable re-exports for tests / continuous consumers.
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
    "play_boot_to_ceres",
    "play_boot_to_ceres_tas",
    "_boot_spans",
    "_BOOT_MENU_MASH_FRAMES",
    "_BOOT_MAX_FRAMES",
    "_CERES_ARM_PUMP_PERIOD",
    "_arm_pump_dash_spans",
]


# ---------------------------------------------------------------------------
# Landing → Morph seed hops (hash-pinned room seeds)
# ---------------------------------------------------------------------------


def _load_room_seed(index: int, name: str) -> list[Action]:
    path = POLICY_DIR / f"seg{index:02d}_{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [np.asarray(action, dtype=np.int8) for action in payload["raw_buttons"]]


def _play_seed_to_room(
    session: RouteSession,
    *,
    index: int,
    name: str,
    target_room: int | None,
    align_elevator: bool = False,
    landing_door_adapter: bool = False,
) -> None:
    if align_elevator:
        session.span(ActionSpan((), 17, "elevator_seed_alignment"))
    source_room = session.state.room_id
    session.raw_actions(_load_room_seed(index, name), f"seed_{name}")

    if landing_door_adapter and session.state.room_id == source_room:
        for _ in range(90):
            session.step(buttons("LEFT", "X"), "landing_door_adapter_shoot")
            if session.state.game_state == 11:
                break
        for _ in range(180):
            session.step(buttons("LEFT"), "landing_door_adapter_enter")
            if session.state.room_id != source_room:
                break

    if target_room is not None and session.state.room_id != target_room:
        if session.state.game_state != 11:
            raise RuntimeError(
                f"seed {name} missed 0x{target_room:04X}: {session.state}"
            )
        session.wait_until(
            lambda state, target=target_room: state.room_id == target,
            timeout=180,
            reason=f"seed_{name}_transition_settle",
        )


def play_landing_to_parlor(session: RouteSession) -> None:
    _play_seed_to_room(
        session,
        index=0,
        name="landing_site",
        target_room=ROOM_PARLOR,
        landing_door_adapter=True,
    )


def play_parlor_to_climb(session: RouteSession) -> None:
    from super_metroid.routes.kpdr.crateria.parlor_descent import (
        parlor_moonfall_enabled,
        play_parlor_to_climb_moonfall,
    )

    if parlor_moonfall_enabled(session):
        play_parlor_to_climb_moonfall(session)
        return
    _play_seed_to_room(session, index=1, name="parlor", target_room=ROOM_CLIMB)


def play_climb_to_pit(session: RouteSession) -> None:
    from super_metroid.routes.kpdr.crateria.climb_descent import (
        climb_moonfall_enabled,
        play_climb_to_pit_moonfall,
    )

    if climb_moonfall_enabled(session):
        play_climb_to_pit_moonfall(session)
        return
    _play_seed_to_room(session, index=2, name="climb", target_room=ROOM_PIT)


def play_pit_to_elevator(session: RouteSession) -> None:
    _play_seed_to_room(
        session, index=3, name="pit_room", target_room=ROOM_BLUE_BRINSTAR_ELEVATOR
    )


def play_elevator_to_morph_room(session: RouteSession) -> None:
    """BB elev → Morph. Prefer product seed; re-pin if elev status phase misses.

    Elev standing flag ``$0E16`` toggles each frame. Product open-loop DOWN
    lands on a ``1`` frame; a faster Ceres path (TAS boot) can invert that
    parity so the same seed crouches instead of boarding. First attempt is
    exact product (parity 1, no pad walk). On miss, re-seat pad and try
    parity 0 / 2, then WRAM-reactive board.
    """
    seed = _load_room_seed(4, "bb_elev_hallway")

    def _try_seed(parity: int, *, reseat: bool) -> bool:
        if session.state.room_id == ROOM_MORPH:
            return True
        if session.state.room_id != ROOM_BLUE_BRINSTAR_ELEVATOR:
            return False
        if reseat:
            for _ in range(80):
                st = session.state
                if st.room_id != ROOM_BLUE_BRINSTAR_ELEVATOR:
                    break
                x = int(st.samus_x)
                if 128 <= x <= 148 and abs(int(st.velocity_x)) <= 1:
                    break
                if x < 128:
                    session.step(buttons("RIGHT"), "bb_elev_to_pad")
                elif x > 148:
                    session.step(buttons("LEFT"), "bb_elev_to_pad")
                else:
                    session.step(idle_action(), "bb_elev_to_pad")
        for _ in range(parity):
            session.step(idle_action(), "elevator_seed_parity")
        session.span(ActionSpan((), 17, "elevator_seed_alignment"))
        session.raw_actions(seed, "seed_bb_elev_hallway")
        if session.state.room_id == ROOM_MORPH or (
            session.state.room_id == ROOM_BLUE_BRINSTAR_ELEVATOR
            and session.state.game_state in (9, 11)
        ):
            session.wait_until(
                lambda s: s.room_id == ROOM_MORPH,
                timeout=180,
                reason="seed_bb_elev_hallway_transition_settle",
            )
            return True
        return False

    # Faster Ceres shifts $0E16 phase. Try both parities on the pad before
    # the WRAM board; do not burn a reseat+replay when parity 0 is enough.
    if _try_seed(0, reseat=False):
        return
    if _try_seed(1, reseat=False):
        return
    _play_bb_elev_reactive(session)


def _elev_status(session: RouteSession) -> int:
    """Samus-on-elevator flag at WRAM ``$0E16`` (toggles while standing on pad)."""
    ram = session.env.get_ram()  # type: ignore[attr-defined]
    return int(ram[0x0E16])


def _play_bb_elev_reactive(session: RouteSession) -> None:
    """Board Blue Brinstar elev by elev-status WRAM, ride to Morph."""
    # Settle door if still transitioning.
    for _ in range(200):
        st = session.state
        if st.room_id != ROOM_BLUE_BRINSTAR_ELEVATOR:
            if st.room_id == ROOM_MORPH:
                return
            break
        if st.game_state == 8 and st.door_transition == 0:
            break
        session.step(buttons("RIGHT"), "bb_elev_door")

    # Walk onto pad center (~x128–145). Product boards from x≈142.
    for _ in range(120):
        st = session.state
        if st.room_id == ROOM_MORPH:
            return
        if st.room_id != ROOM_BLUE_BRINSTAR_ELEVATOR:
            break
        x = int(st.samus_x)
        if 128 <= x <= 148 and abs(int(st.velocity_x)) <= 1:
            break
        if x < 128:
            session.step(buttons("RIGHT", "B"), "bb_elev_to_pad")
        elif x > 148:
            session.step(buttons("LEFT"), "bb_elev_to_pad")
        else:
            session.step(buttons("RIGHT"), "bb_elev_to_pad")

    for _ in range(8):
        session.step(idle_action(), "bb_elev_plant")

    # Wait for elev standing flag ($0E16==1) then DOWN (snaps to elev pose 0).
    boarded = False
    for _ in range(40):
        st = session.state
        if st.room_id == ROOM_MORPH:
            return
        if int(st.pose) == 0 or int(st.samus_y) > 150:
            boarded = True
            break
        if _elev_status(session) == 1:
            session.step(buttons("DOWN"), "bb_elev_board")
        else:
            session.step(idle_action(), "bb_elev_wait_flag")
    if not boarded and int(session.state.pose) != 0:
        # One more forced DOWN pair across both parity frames.
        session.step(buttons("DOWN"), "bb_elev_board")
        session.step(buttons("DOWN"), "bb_elev_board")

    # Ride / wait Morph room.
    for _ in range(400):
        st = session.state
        if st.room_id == ROOM_MORPH:
            if st.game_state == 8:
                return
            session.step(idle_action(), "bb_elev_morph_settle")
            continue
        if st.room_id != ROOM_BLUE_BRINSTAR_ELEVATOR:
            break
        if int(st.pose) == 0 or st.game_state in (9, 11) or int(st.samus_y) > 150:
            session.step(idle_action(), "bb_elev_ride")
        elif _elev_status(session) == 1:
            session.step(buttons("DOWN"), "bb_elev_board")
        else:
            session.step(idle_action(), "bb_elev_ride")
    if session.state.room_id != ROOM_MORPH:
        raise RuntimeError(
            f"bb elev reactive missed Morph: {session.state}"
        )
    # Morph open-loop seed expects elev pose 0 entry (product). Do not force
    # ordinary stand here — that desyncs the seed.


def play_morph_ball_collect(session: RouteSession) -> None:
    """Morph Ball room seed. On miss, return to elev pad and re-seed once.

    Product seed expects elev pose 0 @ x≈128. TAS boot + reactive BB elev can
    land with matching kinematics yet still desync the open-loop tape by phase;
    one pad return + idle-14 re-seed is the cheap residual (not thrash).
    """
    seed = _load_room_seed(5, "morph_ball_room")
    _play_seed_to_room(
        session, index=5, name="morph_ball_room", target_room=None
    )
    if session.state.collected_items & MORPH_BALL_MASK:
        return

    # Walk back toward elev pad (product entry ~x128 y292 pose 0).
    for _ in range(400):
        st = session.state
        if st.room_id != ROOM_MORPH:
            break
        if session.state.collected_items & MORPH_BALL_MASK:
            return
        x = int(st.samus_x)
        if int(st.pose) in (137, 138):
            session.step(idle_action(), "morph_seed_kb")
            continue
        if 110 <= x <= 145 and abs(int(st.velocity_x)) <= 1:
            if int(st.pose) == 0 or abs(int(st.velocity_y)) <= 1:
                break
        if x > 145:
            session.step(buttons("LEFT"), "morph_return_elev")
        elif x < 110:
            session.step(buttons("RIGHT"), "morph_return_elev")
        else:
            session.step(idle_action(), "morph_return_elev")

    for _ in range(14):
        session.step(idle_action(), "morph_seed_phase_align")
    session.raw_actions(seed, "seed_morph_ball_room")
    if not session.state.collected_items & MORPH_BALL_MASK:
        raise RuntimeError(f"Morph Ball was not acquired: {session.state}")


# ---------------------------------------------------------------------------
# Morph play spine = Ceres prefix + Landing→Morph seeds
# ---------------------------------------------------------------------------

MORPH_SPINE: tuple[SpineHop, ...] = CERES_SPINE + (
    SpineHop(
        "landing_to_parlor",
        play_landing_to_parlor,
        ROOM_LANDING_SITE,
        ROOM_PARLOR,
        "Parlor",
        "morph",
        use_transition_split=False,
        exit_direction="left",
        entry_direction="right",
        policy_id="legacy_seed_adapter",
    ),
    SpineHop(
        "parlor_to_climb",
        play_parlor_to_climb,
        ROOM_PARLOR,
        ROOM_CLIMB,
        "Climb",
        "morph",
        use_transition_split=False,
        exit_direction="bottom_left",
        entry_direction="top",
        policy_id="legacy_room_seed",
    ),
    SpineHop(
        "climb_to_pit",
        play_climb_to_pit,
        ROOM_CLIMB,
        ROOM_PIT,
        "Pit",
        "morph",
        use_transition_split=False,
        exit_direction="bottom",
        entry_direction="left",
        policy_id="legacy_room_seed",
    ),
    SpineHop(
        "pit_to_elevator",
        play_pit_to_elevator,
        ROOM_PIT,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        "Blue Brinstar Elevator",
        "morph",
        use_transition_split=False,
        exit_direction="right",
        entry_direction="left",
        policy_id="legacy_room_seed",
    ),
    SpineHop(
        "elevator_to_morph",
        play_elevator_to_morph_room,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        ROOM_MORPH,
        "Morph Ball Room",
        "morph",
        use_transition_split=False,
        exit_direction="elevator",
        entry_direction="right",
        policy_id="legacy_room_seed",
    ),
    SpineHop(
        "morph_ball",
        play_morph_ball_collect,
        ROOM_MORPH,
        ROOM_MORPH,
        "Morph Ball collect",
        "morph",
        use_transition_split=False,
        policy_id="legacy_room_seed",
    ),
)


# ---------------------------------------------------------------------------
# Graph tables (morph stage source of truth for progression/stages/morph.py)
# ---------------------------------------------------------------------------

MORPH_DOOR_EDGES: tuple[DoorEdge, ...] = CERES_DOOR_EDGES + (
    # Seed hops — also emitted from MORPH_SPINE door meta (see continuous_edges).
    DoorEdge(
        "landing_to_parlor",
        ROOM_LANDING_SITE,
        ROOM_PARLOR,
        "left",
        "right",
        policy_id="legacy_seed_adapter",
        verification="continuous",
    ),
    DoorEdge(
        "parlor_to_climb",
        ROOM_PARLOR,
        ROOM_CLIMB,
        "bottom_left",
        "top",
        policy_id="legacy_room_seed",
        verification="continuous",
    ),
    DoorEdge(
        "climb_to_pit",
        ROOM_CLIMB,
        ROOM_PIT,
        "bottom",
        "left",
        policy_id="legacy_room_seed",
        verification="continuous",
    ),
    DoorEdge(
        "pit_to_elevator",
        ROOM_PIT,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        "right",
        "left",
        policy_id="legacy_room_seed",
        verification="continuous",
    ),
    DoorEdge(
        "elevator_to_morph",
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        ROOM_MORPH,
        "elevator",
        "right",
        policy_id="legacy_room_seed",
        verification="continuous",
    ),
    # Post-collect continuity into bombs (not a morph play hop).
    DoorEdge(
        "morph_to_construction",
        ROOM_MORPH,
        ROOM_CONSTRUCTION,
        "right",
        "left",
        frozenset({"morph_ball"}),
        "legacy_room_seed",
        "continuous",
    ),
)

MORPH_MILESTONES: tuple[ProgressionMilestone, ...] = CERES_MILESTONES + (
    ProgressionMilestone(
        "morph_ball",
        "Morph Ball collected naturally",
        ProgressCondition(room_id=ROOM_MORPH, collected_items_mask=MORPH_BALL_MASK),
        acquires=frozenset({"morph_ball"}),
        timeout_frames=8_000,
        policy_id="legacy_room_seed",
    ),
)


def continuous_edges_from_morph_spine(
    spine: Sequence[SpineHop] = MORPH_SPINE,
) -> tuple[DoorEdge, ...]:
    """DoorEdges emitted by morph play hops that declare door meta.

    Full topology (including multi-room Ceres intermediates and morph→construction)
    remains :data:`MORPH_DOOR_EDGES`. This helper is for tests / join checks that
    seed hops match the graph.
    """
    return tuple(hop.as_door_edge() for hop in spine if hop.emits_door_edge)


def morph_route_hops(spine: Sequence[SpineHop] = MORPH_SPINE) -> tuple[SpineHop, ...]:
    """Morph spine as hop sequence (same type as Super+ deltas)."""
    return tuple(spine)


def play_morph_hops(
    session: RouteSession,
    splits: list[Split],
    spine: Sequence[SpineHop] = MORPH_SPINE,
) -> None:
    """Run morph spine via shared :func:`~super_metroid.routes.tips.play_hops`."""
    from super_metroid.routes.tips import play_hops

    play_hops(session, splits, spine)


def play_ship_to_morph(
    session: RouteSession,
    splits: list[Split],
) -> None:
    """Run the real Landing Site → Morph Ball suffix of the vanilla spine.

    This is the reusable edge-policy entry point for randomizer consumers that
    begin at a natural Landing Site observation.  It deliberately reuses the
    vanilla room policies and excludes the power-on/Ceres prefix.
    """
    play_morph_hops(session, splits, MORPH_SPINE[3:])


def validate_morph_spine(spine: Sequence[SpineHop] = MORPH_SPINE) -> None:
    """Dev check: morph tip hops unique; door edge ids unique vs hand list."""
    hop_ids: set[str] = set()
    for hop in spine:
        if hop.tip_id != "morph":
            raise RuntimeError(f"Morph spine hop {hop.hop_id!r} tip_id={hop.tip_id!r}")
        if hop.hop_id in hop_ids:
            raise RuntimeError(f"Duplicate morph hop_id {hop.hop_id!r}")
        hop_ids.add(hop.hop_id)
    required = {
        "first_ceres_control",
        "ridley_countdown",
        "zebes_landing",
        "morph_ball",
    }
    missing = required - hop_ids
    if missing:
        raise RuntimeError(f"Morph spine missing milestone hops: {sorted(missing)}")

    hand_ids = {e.edge_id for e in MORPH_DOOR_EDGES}
    for edge in continuous_edges_from_morph_spine(spine):
        if edge.edge_id not in hand_ids:
            raise RuntimeError(
                f"Morph spine DoorEdge {edge.edge_id!r} missing from MORPH_DOOR_EDGES"
            )


validate_morph_spine()
