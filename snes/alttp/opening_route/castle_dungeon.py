"""Composable first-Hyrule-Castle dungeon prefix.

This module owns the measured room-edge sequence after the continuous opening
tip.  It deliberately stops at room ``0x50``:

``0x61 main hall → 0x60 west hall → 0x50 NW chamber``.

Both edges are verified from their real predecessor in one clean power-on
chain. The prefix is still *not* a Zelda or Sanctuary clear.
Keep any newly measured B1 door here as a typed edge, rather than a
monolithic Zelda macro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alttp.opening_route.room_engine import in_room, run_room_edge
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    AlttpSnapshot,
    zelda_rescued_accepted,
)
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import snapshot_env


@dataclass(frozen=True)
class DungeonRoomEdge:
    """One measured room-engine edge in the castle dungeon prefix."""

    edge_id: str
    map_id: str
    door_label: str
    source_room: int
    target_room: int
    graph_edge_id: str
    verification: str = "continuous"
    clear: bool = True


MAIN_HALL_TO_NW_PREFIX: tuple[DungeonRoomEdge, ...] = (
    DungeonRoomEdge(
        edge_id="main_hall_west_to_0x60",
        map_id="room_61",
        door_label="west_to_0x60",
        source_room=HYRULE_CASTLE_MAIN_HALL_ROOM,
        target_room=HYRULE_CASTLE_MAIN_WEST_ROOM,
        graph_edge_id="main_hall_west_to_0x60",
    ),
    DungeonRoomEdge(
        edge_id="room_60_north_to_0x50",
        map_id="room_60",
        door_label="north_to_0x50",
        source_room=HYRULE_CASTLE_MAIN_WEST_ROOM,
        target_room=HYRULE_CASTLE_NW_ROOM,
        graph_edge_id="room_60_north_to_0x50",
    ),
)


def evaluate_prefix_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Acceptance keys for the measured prefix, plus honest future diagnostics."""
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "main_hall": in_room(snapshot, HYRULE_CASTLE_MAIN_HALL_ROOM),
        "main_west_0x60": in_room(snapshot, HYRULE_CASTLE_MAIN_WEST_ROOM),
        "northwest_0x50": in_room(snapshot, HYRULE_CASTLE_NW_ROOM),
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "in_sanctuary": snapshot.in_sanctuary,
    }


def _prefix_notes(edges: Sequence[DungeonRoomEdge]) -> list[str]:
    return [
        "First Hyrule Castle dungeon prefix via room_engine map authority.",
        "All listed edges passed from their real predecessor in the clean power-on chain.",
        "No Zelda or Sanctuary claim is made by this prefix.",
        *[
            (
                f"{edge.edge_id}: {edge.map_id}/{edge.door_label} "
                f"(verification={edge.verification})"
            )
            for edge in edges
        ],
    ]


def run_room_edge_sequence(
    env: object,
    edges: Sequence[DungeonRoomEdge],
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Run measured room edges in order, failing before a wrong-room action.

    ``run_room_edge`` owns combat, geometry and door pushes.  This thin
    composition layer owns only ordering, room-boundary assertions and a
    cumulative report.  That keeps every future castle-door addition local and
    auditable.
    """
    if not edges:
        snap = snapshot_env(env)
        return SegmentResult(
            ok=False,
            phase="empty_dungeon_prefix",
            frames=0,
            snapshot=snap,
            source=source,
            acceptance=evaluate_prefix_acceptance(snap),
            blocker="no dungeon room edges configured",
        )

    frames = 0
    phases: list[RoutePhaseResult] = []
    notes = _prefix_notes(edges)
    for edge in edges:
        before = snapshot_env(env)
        if not in_room(before, edge.source_room):
            return SegmentResult(
                ok=False,
                phase=f"entry_not_{edge.map_id}",
                frames=frames,
                snapshot=before,
                phases=phases,
                source=source,
                acceptance=evaluate_prefix_acceptance(before),
                blocker=(
                    f"{edge.edge_id} expected room 0x{edge.source_room:02X}, got "
                    f"0x{before.room_base_id:02X} indoors={before.indoors}"
                ),
                notes=notes,
            )

        result = run_room_edge(
            env,
            edge.map_id,
            edge.door_label,
            clear=edge.clear,
            source=source,
            notes=[
                f"dungeon_edge={edge.edge_id}",
                f"graph_edge={edge.graph_edge_id}",
                f"verification={edge.verification}",
            ],
        )
        frames += result.frames
        phases.extend(result.phases)
        snap = result.snapshot
        if not result.ok:
            return SegmentResult(
                ok=False,
                phase=f"{edge.edge_id}:{result.phase}",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=evaluate_prefix_acceptance(snap),
                blocker=result.blocker or f"{edge.edge_id} failed",
                notes=notes,
            )
        if not in_room(snap, edge.target_room):
            return SegmentResult(
                ok=False,
                phase=f"{edge.edge_id}:wrong_destination",
                frames=frames,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=evaluate_prefix_acceptance(snap),
                blocker=(
                    f"{edge.edge_id} expected destination 0x{edge.target_room:02X}, "
                    f"got 0x{snap.room_base_id:02X} indoors={snap.indoors}"
                ),
                notes=notes,
            )

    final = snapshot_env(env)
    return SegmentResult(
        ok=True,
        phase="castle_dungeon_prefix_complete",
        frames=frames,
        snapshot=final,
        phases=phases,
        source=source,
        acceptance=evaluate_prefix_acceptance(final),
        notes=notes,
    )


def run_from_main_hall(
    env: object,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Run the measured ``0x61 → 0x60 → 0x50`` first-dungeon prefix."""
    return run_room_edge_sequence(env, MAIN_HALL_TO_NW_PREFIX, source=source)


def run_from_state(
    state_name: str = "CastleMain",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic for the two-edge prefix from main hall."""
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    try:
        env.reset()  # type: ignore[attr-defined]
        return run_from_main_hall(env, source="state_load_dev")
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]
