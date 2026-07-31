"""Thin ALTTP session façade over ``retro_harness.snes`` + sparse RAM.

Preferred agent surface for opening-route work:

- selective / cached :class:`~alttp.ram.AlttpSnapshot`
- progress / capability vector from the escape graph
- natural-entry segment play + evidence
- multi-truth anchor matching

Does not own long-running route logic — that stays in
``alttp.opening_route`` and ``alttp.primitives``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from alttp.opening_route.anchors import (
    MultiTruthAnchor,
    anchors_to_report,
    match_anchors,
)
from alttp.opening_route.escape_graph import (
    capabilities_from_snapshot,
    escape_route_graph,
    plan_escape_to_sanctuary,
)
from alttp.opening_route.segment import (
    SegmentEvidence,
    get_segment,
    list_segments,
    segment_registry,
)
from alttp.ram import AlttpSnapshot, read_snapshot, snapshot_to_diag
from alttp.startup import snapshot_env


@dataclass
class AlttpSession:
    """Wrap a live stable-retro env with selective observation helpers."""

    env: object
    _cache: AlttpSnapshot | None = field(default=None, repr=False)
    _frame: int = 0
    source: str = "live"

    def invalidate(self) -> None:
        self._cache = None

    def snapshot(self, *, refresh: bool = True) -> AlttpSnapshot:
        """Return selective RAM snapshot (cached unless ``refresh``)."""
        if refresh or self._cache is None:
            self._cache = snapshot_env(self.env)
        return self._cache

    def read_ram_snapshot(self, ram: object) -> AlttpSnapshot:
        """Parse an already-fetched RAM buffer without touching the env."""
        import numpy as np

        return read_snapshot(np.asarray(ram, dtype=np.uint8))

    def progress_vector(self) -> tuple[int, ...]:
        """Compact progress tuple for logs / tests (not a full bank dump)."""
        s = self.snapshot()
        return (
            int(s.game_mode),
            int(s.screen_id) if not s.indoors else int(s.room_base_id),
            int(s.indoors),
            int(s.sword_level),
            int(s.lamp_level != 0),
            int(s.follower),
            int(s.link_x),
            int(s.link_y),
        )

    def capabilities(self) -> frozenset[str]:
        return capabilities_from_snapshot(self.snapshot())

    def matched_anchors(self) -> list[MultiTruthAnchor]:
        return match_anchors(self.snapshot())

    def anchor_report(self) -> list[dict[str, Any]]:
        return anchors_to_report(self.snapshot())

    def diag(self) -> dict[str, object]:
        return snapshot_to_diag(self.snapshot())

    def plan_to_sanctuary(self) -> list[dict[str, object]]:
        """Capability plan from current inventory toward Sanctuary."""
        planned = plan_escape_to_sanctuary(self.capabilities())
        return [
            {
                "legId": p.leg.leg_id,
                "sourceId": p.leg.source_id,
                "targetId": p.leg.target_id,
                "capabilitiesBefore": sorted(p.capabilities_before),
                "capabilitiesAfter": sorted(p.capabilities_after),
            }
            for p in planned
        ]

    def continuous_tip_node(self) -> str:
        """Best-effort graph node for the current continuous tip location."""
        s = self.snapshot()
        g = escape_route_graph()
        # Prefer specific outdoor pocket over generic grounds.
        from alttp.opening_route.anchors import anchor_by_id

        if s.indoors and s.room_base_id == 0x61 and s.has_fighter_sword:
            return "room_61"
        pocket = anchor_by_id("HyruleCastle_Courtyard_SecretStairsPocket")
        if pocket is not None and pocket.matches(s):
            return "courtyard_secret_pocket"
        if s.in_secret_passage and s.has_fighter_sword:
            if s.link_y >= 2850:
                return "room_55_south"
            return "room_55_sword"
        if s.in_secret_passage:
            return "room_55_uncle"
        if s.on_castle_grounds:
            return "castle_grounds"
        if s.in_zelda_cell:
            return "room_80"
        if s.in_sanctuary:
            return "sanctuary"
        # Fall back: any matched anchor with graph_node_id
        for a in match_anchors(s):
            if a.graph_node_id and a.graph_node_id in g.nodes:
                return a.graph_node_id
        return "unknown"

    def play_segment(self, segment_id: str, **kwargs: Any) -> SegmentEvidence:
        """Execute a registered Segment and return uniform evidence."""
        seg = get_segment(segment_id)
        # Default source tags for clean-chain bookkeeping
        if "source" not in kwargs:
            kwargs["source"] = self.source
        evidence = seg.play_checked(self.env, **kwargs)
        self.invalidate()
        self._frame += evidence.frames
        return evidence

    def list_segments(self) -> list[str]:
        return list_segments()

    def segment_catalog(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in segment_registry().values()]


def bind_env(env: object, *, source: str = "live") -> AlttpSession:
    """Create an :class:`AlttpSession` bound to ``env``."""
    return AlttpSession(env=env, source=source)
