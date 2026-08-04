"""Early continuous spine — Ceres/Morph power-on as :class:`SpineHop` orchestration.

**Model (same as Super+):** ordered :class:`~super_metroid.routes.kpdr.spine.SpineHop`
rows on :data:`MORPH_SPINE` are registered as the morph TipSpec ``hops`` and run
via :func:`~super_metroid.routes.tips.play_hops`. Hash-pinned room seeds and
open-loop Ceres frame budgets are **unchanged** — only the composition surface
matches post-Supers.

**Graph edges / milestones** for ``MORPH_GRAPH`` live here as the morph-stage
source of truth (imported by ``progression/stages/morph``). Multi-room Ceres bulk
policies still span several DoorEdges per play hop; intermediate edges stay
hand-authored beside the play spine (same verification story as Super+ product
doors vs pure-only reverse edges).

Bombs / Spore / Supers orchestration lives in
:mod:`super_metroid.routes.kpdr.early_post_morph` (same SpineHop model; policy
JSON / boss controllers unchanged; pre-Supers DoorEdges remain in ``progression/stages/``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.paths import POLICY_DIR
from super_metroid.progression.types import (
    DoorEdge,
    ProgressCondition,
    ProgressionMilestone,
)
from super_metroid.ram import MORPH_BALL_MASK
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
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
    "morph_route_hops",
    "continuous_edges_from_morph_spine",
    "validate_morph_spine",
]


# ---------------------------------------------------------------------------
# Unchanged open-loop spans / room seeds (hash-pinned product path)
# ---------------------------------------------------------------------------


def _boot_spans() -> list[ActionSpan]:
    spans = [
        ActionSpan((), 2100, "boot_title_wait"),
        ActionSpan(("A",), 10, "boot_title_confirm"),
        ActionSpan((), 120, "boot_file_menu_wait"),
        ActionSpan(("A",), 10, "boot_file_confirm"),
        ActionSpan((), 300, "boot_prologue_wait"),
        ActionSpan(("A",), 10, "boot_prologue_confirm"),
        ActionSpan((), 30, "boot_prologue_settle"),
    ]
    for _ in range(69):
        spans.append(ActionSpan(("A",), 10, "boot_intro_mash"))
        spans.append(ActionSpan((), 110, "boot_intro_wait"))
    return spans


def _ceres_outbound_spans() -> list[ActionSpan]:
    raw = (
        (("RIGHT", "A"), 24),
        (("RIGHT",), 120),
        (("LEFT",), 120),
        (("RIGHT", "B"), 240),
        ((), 60),
        (("RIGHT",), 24),
        (("RIGHT", "B"), 24),
        (("RIGHT", "B", "A"), 24),
        (("RIGHT", "A"), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT", "B"), 24),
        ((), 12),
        (("RIGHT",), 24),
        ((), 140),
        (("RIGHT",), 160),
        (("LEFT",), 120),
        (("RIGHT", "B"), 96),
        ((), 120),
        (("RIGHT", "B"), 216),
        ((), 150),
        (("RIGHT", "B"), 240),
        ((), 200),
    )
    return [ActionSpan(names, frames, "ceres_outbound") for names, frames in raw]


def _ceres_escape_spans() -> list[ActionSpan]:
    raw = (
        (("LEFT", "A"), 40, "ceres_ridley_exit"),
        (("LEFT",), 1000, "ceres_reverse_rooms"),
        ((), 20, "ceres_magnet_phase_align"),
        (("A",), 16, "ceres_magnet_climb"),
        (("RIGHT", "A"), 124, "ceres_magnet_climb"),
        (("LEFT", "A"), 60, "ceres_magnet_climb"),
        (("LEFT",), 320, "ceres_magnet_exit"),
        (("LEFT", "A"), 40, "ceres_falling_room"),
        (("LEFT",), 380, "ceres_falling_room"),
        (("LEFT", "A"), 70, "ceres_elevator_lower_ledge"),
    )
    return [ActionSpan(names, frames, reason) for names, frames, reason in raw]


def _ceres_shaft_spans() -> list[ActionSpan]:
    raw = (
        (("LEFT", "A"), 70),
        ((), 7),
        (("RIGHT", "A"), 67),
        ((), 1),
        (("LEFT", "A"), 67),
        ((), 19),
        (("RIGHT", "A"), 66),
        ((), 8),
        (("RIGHT", "A"), 86),
        (("LEFT", "A"), 83),
        (("RIGHT", "A"), 72),
        ((), 3),
        (("LEFT", "A"), 38),
        (("LEFT",), 25),
    )
    return [ActionSpan(names, frames, "ceres_elevator_climb") for names, frames in raw]


def _load_room_seed(index: int, name: str) -> list[Action]:
    path = POLICY_DIR / f"seg{index:02d}_{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [np.asarray(action, dtype=np.int8) for action in payload["raw_buttons"]]


# ---------------------------------------------------------------------------
# Hop play callables (RouteSession) — seed bytes / span budgets identical
# ---------------------------------------------------------------------------


def play_boot_to_ceres(session: RouteSession) -> None:
    """Power-on boot mash through first controllable Ceres elevator frame."""
    session.spans(_boot_spans())
    if not (session.state.room_id == ROOM_CERES_ELEVATOR and session.state.game_state == 8):
        raise RuntimeError(f"boot missed first Ceres control: {session.state}")


def play_ceres_outbound_to_ridley(session: RouteSession) -> None:
    """Ceres elevator → Ridley room + natural countdown gate."""
    session.spans(_ceres_outbound_spans())
    session.wait_until(
        lambda state: state.room_id == ROOM_CERES_RIDLEY
        and state.timer_type == 3
        and state.health <= 27,
        timeout=6_000,
        reason="ceres_ridley_natural_countdown",
    )


def play_ceres_escape_to_landing(session: RouteSession) -> None:
    """Ceres reverse rooms + elevator shaft → Zebes Landing Site settle."""
    session.spans(_ceres_escape_spans())
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        raise RuntimeError(f"Ceres reverse route missed elevator: {session.state}")
    session.wait_until(
        lambda state: state.room_id == ROOM_CERES_ELEVATOR
        and state.samus_y == 571
        and state.pose == 2,
        timeout=120,
        reason="ceres_lower_ledge_settle",
    )
    session.spans(_ceres_shaft_spans())
    session.wait_until(
        lambda state: state.room_id == ROOM_LANDING_SITE and state.game_state == 8,
        timeout=3_000,
        reason="zebes_landing_transition",
    )
    stable = 0
    for _ in range(1_200):
        if session.state.samus_y == 1088:
            stable += 1
            if stable >= 30:
                break
        else:
            stable = 0
        session.step(idle_action(), "zebes_ship_final_settle")
    else:
        raise TimeoutError(f"Zebes ship never reached final settle: {session.state}")


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
    _play_seed_to_room(session, index=1, name="parlor", target_room=ROOM_CLIMB)


def play_climb_to_pit(session: RouteSession) -> None:
    _play_seed_to_room(session, index=2, name="climb", target_room=ROOM_PIT)


def play_pit_to_elevator(session: RouteSession) -> None:
    _play_seed_to_room(
        session, index=3, name="pit_room", target_room=ROOM_BLUE_BRINSTAR_ELEVATOR
    )


def play_elevator_to_morph_room(session: RouteSession) -> None:
    _play_seed_to_room(
        session,
        index=4,
        name="bb_elev_hallway",
        target_room=ROOM_MORPH,
        align_elevator=True,
    )


def play_morph_ball_collect(session: RouteSession) -> None:
    _play_seed_to_room(
        session, index=5, name="morph_ball_room", target_room=None
    )
    if not session.state.collected_items & MORPH_BALL_MASK:
        raise RuntimeError(f"Morph Ball was not acquired: {session.state}")


# ---------------------------------------------------------------------------
# Morph play spine (orchestration; seeds/spans unchanged)
# ---------------------------------------------------------------------------

# Power-on has no real source room; use Ceres elevator as graph anchor for both
# ends of the boot hop (milestone split only).
MORPH_SPINE: tuple[SpineHop, ...] = (
    SpineHop(
        "first_ceres_control",
        play_boot_to_ceres,
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_ELEVATOR,
        "Ceres Elevator",
        "morph",
        use_transition_split=False,
        # Bulk multi-room policies below; intermediate DoorEdges stay hand list.
    ),
    SpineHop(
        "ridley_countdown",
        play_ceres_outbound_to_ridley,
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_RIDLEY,
        "Ceres Ridley",
        "morph",
        use_transition_split=False,
        policy_id="ceres_outbound",
    ),
    SpineHop(
        "zebes_landing",
        play_ceres_escape_to_landing,
        ROOM_CERES_RIDLEY,
        ROOM_LANDING_SITE,
        "Landing Site",
        "morph",
        use_transition_split=False,
        policy_id="ceres_escape",
    ),
    # Seed hops: historical morph reports only record milestone splits, not
    # per-door transition splits (elevator transitions are also noisy). Door
    # meta still emits product edges for join checks via continuous_edges_*.
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

MORPH_DOOR_EDGES: tuple[DoorEdge, ...] = (
    # Ceres outbound bulk (play hop ridley_countdown spans all of these).
    DoorEdge(
        "ceres_elevator_to_falling",
        ROOM_CERES_ELEVATOR,
        ROOM_CERES_FALLING,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_falling_to_magnet",
        ROOM_CERES_FALLING,
        ROOM_CERES_MAGNET,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_magnet_to_scientist",
        ROOM_CERES_MAGNET,
        ROOM_CERES_SCIENTIST,
        "bottom_right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_scientist_to_flat",
        ROOM_CERES_SCIENTIST,
        ROOM_CERES_FLAT,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_flat_to_ridley",
        ROOM_CERES_FLAT,
        ROOM_CERES_RIDLEY,
        "right",
        "left",
        policy_id="ceres_outbound",
        verification="continuous",
    ),
    # Ceres escape bulk (play hop zebes_landing).
    DoorEdge(
        "ceres_ridley_to_flat",
        ROOM_CERES_RIDLEY,
        ROOM_CERES_FLAT,
        "left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_flat_to_scientist",
        ROOM_CERES_FLAT,
        ROOM_CERES_SCIENTIST,
        "left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_scientist_to_magnet",
        ROOM_CERES_SCIENTIST,
        ROOM_CERES_MAGNET,
        "left",
        "bottom_right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_magnet_to_falling",
        ROOM_CERES_MAGNET,
        ROOM_CERES_FALLING,
        "upper_left",
        "right",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_falling_to_elevator",
        ROOM_CERES_FALLING,
        ROOM_CERES_ELEVATOR,
        "left",
        "bottom",
        policy_id="ceres_escape",
        verification="continuous",
    ),
    DoorEdge(
        "ceres_to_landing",
        ROOM_CERES_ELEVATOR,
        ROOM_LANDING_SITE,
        "elevator",
        "ship",
        policy_id="ceres_escape",
        verification="continuous",
    ),
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

MORPH_MILESTONES: tuple[ProgressionMilestone, ...] = (
    ProgressionMilestone(
        "first_ceres_control",
        "First controllable Ceres frame",
        ProgressCondition(room_id=ROOM_CERES_ELEVATOR, game_states=frozenset({8})),
        timeout_frames=12_000,
        policy_id="power_on_boot",
    ),
    ProgressionMilestone(
        "ridley_countdown",
        "Natural Ceres countdown",
        ProgressCondition(room_id=ROOM_CERES_RIDLEY, game_states=frozenset({8})),
        timeout_frames=7_000,
        policy_id="ceres_ridley_wait",
    ),
    ProgressionMilestone(
        "zebes_landing",
        "Zebes Landing Site control",
        ProgressCondition(room_id=ROOM_LANDING_SITE, game_states=frozenset({8})),
        timeout_frames=8_000,
        policy_id="ceres_escape",
    ),
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
