"""Compose early SMZ3 segments into one chainable quest runner.

Probes and the future race harness should call :func:`run_early_quest` rather
than hand-chaining portal → outdoor → house in every script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from smz3.boot import make_boot_env
from smz3.house_route import HouseSegmentResult, run_links_house_chest
from smz3.outdoor_route import OutdoorSegmentResult, run_fortune_teller_to_links_house
from smz3.portal_route import (
    STOP_AFTER_PORTAL,
    STOP_AT_RED_DOOR,
    PortalSegmentResult,
    run_landing_to_portal,
)
from smz3.ram import ComboSnapshot
from smz3.route_graph import (
    COARSE_GRAPH,
    N_LANDING,
    N_LINKS_HOUSE_CHEST,
    N_LINKS_HOUSE_OW,
    N_PARLOR,
    N_PORTAL_SETTLED,
    N_RED_DOOR,
    path_with_capabilities,
    plan_early_legs,
)
from smz3.segment import SegmentResult

# Public stop names (also accepted as route-graph node ids).
STOP_PARLOR = "parlor"
STOP_RED_DOOR = "red_door"
STOP_PORTAL = "portal"
STOP_LINKS_HOUSE_OW = "links_house_ow"
STOP_LINKS_HOUSE_CHEST = "links_house_chest"

STOP_TO_NODE = {
    STOP_PARLOR: N_PARLOR,
    STOP_RED_DOOR: N_RED_DOOR,
    STOP_PORTAL: N_PORTAL_SETTLED,
    STOP_LINKS_HOUSE_OW: N_LINKS_HOUSE_OW,
    STOP_LINKS_HOUSE_CHEST: N_LINKS_HOUSE_CHEST,
    N_PARLOR: N_PARLOR,
    N_RED_DOOR: N_RED_DOOR,
    N_PORTAL_SETTLED: N_PORTAL_SETTLED,
    N_LINKS_HOUSE_OW: N_LINKS_HOUSE_OW,
    N_LINKS_HOUSE_CHEST: N_LINKS_HOUSE_CHEST,
}

STOP_CHOICES = (
    STOP_PARLOR,
    STOP_RED_DOOR,
    STOP_PORTAL,
    STOP_LINKS_HOUSE_OW,
    STOP_LINKS_HOUSE_CHEST,
)


@dataclass
class QuestResult(SegmentResult):
    """Aggregated early-quest outcome."""

    goal: str = "early_quest"
    stop: str = STOP_LINKS_HOUSE_CHEST
    portal: PortalSegmentResult | None = None
    outdoor: OutdoorSegmentResult | None = None
    house: HouseSegmentResult | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    planned_legs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "stop": self.stop,
                "portal": self.portal.to_dict() if self.portal else None,
                "outdoor": self.outdoor.to_dict() if self.outdoor else None,
                "house": self.house.to_dict() if self.house else None,
                "segments": list(self.segments),
                "planned_legs": list(self.planned_legs),
            }
        )
        return d


def resolve_stop(stop: str) -> str:
    """Normalize stop name to a short public id."""
    key = stop.strip().lower()
    if key not in STOP_TO_NODE:
        raise ValueError(f"Unknown stop {stop!r}. Choose from {STOP_CHOICES}")
    for short, node in (
        (STOP_PARLOR, N_PARLOR),
        (STOP_RED_DOOR, N_RED_DOOR),
        (STOP_PORTAL, N_PORTAL_SETTLED),
        (STOP_LINKS_HOUSE_OW, N_LINKS_HOUSE_OW),
        (STOP_LINKS_HOUSE_CHEST, N_LINKS_HOUSE_CHEST),
    ):
        if key == short or key == node:
            return short
    return key


def _plan_dicts(stop_node: str, capabilities: frozenset[str]) -> list[dict[str, Any]]:
    try:
        legs = plan_early_legs(stop_node=stop_node, initial_capabilities=capabilities)
        return [pl.to_dict(COARSE_GRAPH.nodes) for pl in legs]
    except ValueError as exc:
        return [{"error": str(exc)}]


def run_early_quest(
    env: Any | None = None,
    *,
    close: bool = False,
    stop: str = STOP_LINKS_HOUSE_CHEST,
    grant_missile_assist: bool = True,
    max_frames: int = 30_000,
    room_timeout_multiplier: float = 3.0,
    on_frame: Callable[[int, ComboSnapshot], None] | None = None,
    skip_outdoor: bool = False,
) -> QuestResult:
    """Power-on (or use *env*) and run early segments until *stop*.

    Stops: ``parlor`` | ``red_door`` | ``portal`` | ``links_house_ow`` |
    ``links_house_chest``.
    """
    stop_id = resolve_stop(stop)
    node = STOP_TO_NODE[stop_id]
    owns = env is None
    if env is None:
        env = make_boot_env(render_mode="rgb_array")
        env.reset()

    caps = frozenset({"missiles"}) if grant_missile_assist else frozenset()
    planned_dicts = _plan_dicts(node, caps)

    segments: list[dict[str, Any]] = []
    portal: PortalSegmentResult | None = None
    outdoor: OutdoorSegmentResult | None = None
    house: HouseSegmentResult | None = None
    total_frames = 0
    final_snap: ComboSnapshot | None = None

    try:
        if stop_id == STOP_PARLOR:
            from smz3.early_route import run_landing_to_parlor

            early = run_landing_to_parlor(
                env,
                close=False,
                max_frames=max_frames,
                room_timeout_multiplier=room_timeout_multiplier,
            )
            segments.append(early.to_dict())
            return QuestResult(
                ok=early.ok,
                goal="early_quest",
                stop=stop_id,
                frames=early.frames,
                detail=early.detail,
                final_snapshot=early.final_snapshot,
                segments=segments,
                planned_legs=planned_dicts,
            )

        portal = run_landing_to_portal(
            env,
            close=False,
            max_frames=max_frames,
            room_timeout_multiplier=room_timeout_multiplier,
            grant_missile_assist=grant_missile_assist,
            stop=STOP_AT_RED_DOOR if stop_id == STOP_RED_DOOR else STOP_AFTER_PORTAL,
        )
        segments.append(portal.to_dict())
        total_frames = portal.frames
        final_snap = portal.final_snapshot

        if stop_id == STOP_RED_DOOR:
            return QuestResult(
                ok=portal.ok,
                goal="early_quest",
                stop=stop_id,
                frames=total_frames,
                detail=portal.detail,
                final_snapshot=final_snap,
                portal=portal,
                segments=segments,
                planned_legs=planned_dicts,
            )

        if not portal.z3_settled and not (portal.ok and portal.portal_started):
            return QuestResult(
                ok=False,
                goal="early_quest",
                stop=stop_id,
                frames=total_frames,
                detail=f"portal failed: {portal.detail}",
                final_snapshot=final_snap,
                portal=portal,
                segments=segments,
                planned_legs=planned_dicts,
            )

        if stop_id == STOP_PORTAL:
            ok = bool(portal.z3_settled or portal.portal_started)
            return QuestResult(
                ok=ok,
                goal="early_quest",
                stop=stop_id,
                frames=total_frames,
                detail=portal.detail,
                final_snapshot=final_snap,
                portal=portal,
                segments=segments,
                planned_legs=planned_dicts,
            )

        # Prefer settled Link for outdoor; residue may still fail outdoor wait.
        if not portal.z3_settled:
            return QuestResult(
                ok=False,
                goal="early_quest",
                stop=stop_id,
                frames=total_frames,
                detail=f"portal did not settle: {portal.detail}",
                final_snapshot=final_snap,
                portal=portal,
                segments=segments,
                planned_legs=planned_dicts,
            )

        if not skip_outdoor:
            outdoor = run_fortune_teller_to_links_house(
                env, start_frame=0, on_frame=on_frame
            )
            segments.append(outdoor.to_dict())
            total_frames += outdoor.frames
            final_snap = outdoor.final_snapshot or final_snap
            if not outdoor.ok:
                return QuestResult(
                    ok=False,
                    goal="early_quest",
                    stop=stop_id,
                    frames=total_frames,
                    detail=f"outdoor failed: {outdoor.detail}",
                    final_snapshot=final_snap,
                    portal=portal,
                    outdoor=outdoor,
                    segments=segments,
                    planned_legs=planned_dicts,
                )

        if stop_id == STOP_LINKS_HOUSE_OW:
            return QuestResult(
                ok=True if outdoor is None else outdoor.ok,
                goal="early_quest",
                stop=stop_id,
                frames=total_frames,
                detail=outdoor.detail if outdoor else "skipped outdoor",
                final_snapshot=final_snap,
                portal=portal,
                outdoor=outdoor,
                segments=segments,
                planned_legs=planned_dicts,
            )

        house = run_links_house_chest(env, start_frame=0, on_frame=on_frame)
        segments.append(house.to_dict())
        total_frames += house.frames
        final_snap = house.final_snapshot or final_snap
        return QuestResult(
            ok=house.ok,
            goal="early_quest",
            stop=stop_id,
            frames=total_frames,
            detail=house.detail,
            final_snapshot=final_snap,
            portal=portal,
            outdoor=outdoor,
            house=house,
            segments=segments,
            planned_legs=planned_dicts,
        )
    finally:
        if owns and close:
            env.close()


def early_path_summary(
    stop: str = STOP_LINKS_HOUSE_CHEST,
    *,
    with_missiles: bool = True,
) -> dict[str, Any]:
    """Offline path / plan summary (no emulator)."""
    stop_id = resolve_stop(stop)
    node = STOP_TO_NODE[stop_id]
    caps = frozenset({"missiles"}) if with_missiles else frozenset()
    path = path_with_capabilities(N_LANDING, node, caps)
    return {
        "stop": stop_id,
        "node": node,
        "capabilities": sorted(caps),
        "path_edge_ids": [e.edge_id for e in path] if path else None,
        "planned_legs": _plan_dicts(node, caps),
    }
