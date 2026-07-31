"""Segment / natural-entry contract for ALTTP opening-route hops.

Parallel in spirit to ``super_metroid.routes.segment``:

- entry requirement (room/screen + position + inventory)
- exit / success predicate
- ``play(env)`` → evidence (frames, final snapshot, phase report)

Existing scripts (``castle_to_sword``, ``sword_to_zelda``) adapt to this
surface. Continuous composition should walk verified Segments; a hop is
route-ready only when it succeeds from the real predecessor continuous
state (no privileged warps in published evidence).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from alttp.opening_route.anchors import MultiTruthAnchor, opening_anchors
from alttp.ram import AlttpSnapshot, snapshot_to_diag
from alttp.route_report import SegmentResult


@runtime_checkable
class Segment(Protocol):
    """Uniform play unit for continuous opening hops."""

    @property
    def id(self) -> str: ...

    def play(self, env: object, **kwargs: Any) -> SegmentResult: ...


@dataclass(frozen=True)
class EntryRequirement:
    """What must be true before a segment may start (natural-entry gate)."""

    description: str
    room_base_id: int | None = None
    screen_id: int | None = None
    require_indoors: bool | None = None
    require_fighter_sword: bool = False
    require_lamp: bool = False
    require_zelda_follower: bool = False
    require_control: bool = True
    # Optional multi-truth anchor ids that all must match
    anchor_ids: tuple[str, ...] = ()
    # Graph node this segment expects as predecessor
    graph_node_id: str = ""

    def matches(self, snapshot: AlttpSnapshot) -> bool:
        if self.require_control and not snapshot.has_control:
            return False
        if self.room_base_id is not None:
            if not snapshot.indoors or snapshot.room_base_id != self.room_base_id:
                return False
        if self.screen_id is not None and not snapshot.indoors:
            if snapshot.screen_id != self.screen_id:
                return False
        if self.require_indoors is True and not snapshot.indoors:
            return False
        if self.require_indoors is False and snapshot.indoors:
            return False
        if self.require_fighter_sword and not snapshot.has_fighter_sword:
            return False
        if self.require_lamp and not snapshot.has_lamp:
            return False
        if self.require_zelda_follower and not snapshot.has_zelda_follower:
            return False
        if self.anchor_ids:
            by_id = {a.anchor_id: a for a in opening_anchors()}
            for aid in self.anchor_ids:
                anchor = by_id.get(aid)
                if anchor is None or not anchor.matches(snapshot):
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "roomBaseId": self.room_base_id,
            "screenId": self.screen_id,
            "requireIndoors": self.require_indoors,
            "requireFighterSword": self.require_fighter_sword,
            "requireLamp": self.require_lamp,
            "requireZeldaFollower": self.require_zelda_follower,
            "requireControl": self.require_control,
            "anchorIds": list(self.anchor_ids),
            "graphNodeId": self.graph_node_id,
        }


@dataclass(frozen=True)
class ExitPredicate:
    """Success condition for a finished segment."""

    description: str
    acceptance_keys: tuple[str, ...] = ()
    # If set, all listed acceptance keys from evaluate_acceptance must be True
    require_all: bool = True
    graph_node_id: str = ""
    verification: str = "planned"  # planned | isolated | natural_entry | continuous

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "acceptanceKeys": list(self.acceptance_keys),
            "requireAll": self.require_all,
            "graphNodeId": self.graph_node_id,
            "verification": self.verification,
        }


@dataclass
class SegmentEvidence:
    """Uniform evidence wrapper around :class:`SegmentResult`."""

    segment_id: str
    ok: bool
    frames: int
    snapshot: AlttpSnapshot
    source: str
    phase: str
    acceptance: dict[str, bool] = field(default_factory=dict)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)
    development_only: bool = True
    matched_anchors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentId": self.segment_id,
            "ok": self.ok,
            "frames": self.frames,
            "source": self.source,
            "phase": self.phase,
            "cleanChain": self.source == "natural_boot" and self.ok,
            "developmentOnly": self.development_only,
            "acceptance": dict(self.acceptance),
            "blocker": self.blocker,
            "notes": list(self.notes),
            "matchedAnchors": list(self.matched_anchors),
            "final": snapshot_to_diag(self.snapshot),
        }

    @classmethod
    def from_segment_result(
        cls,
        segment_id: str,
        result: SegmentResult,
        *,
        matched_anchors: Sequence[str] | None = None,
    ) -> SegmentEvidence:
        return cls(
            segment_id=segment_id,
            ok=result.ok,
            frames=result.frames,
            snapshot=result.snapshot,
            source=result.source,
            phase=result.phase,
            acceptance=dict(result.acceptance),
            blocker=result.blocker,
            notes=list(result.notes),
            development_only=result.source != "natural_boot",
            matched_anchors=list(matched_anchors or ()),
        )


PlayFn = Callable[..., SegmentResult]


@dataclass(frozen=True)
class ScriptSegment:
    """Adapter: existing route script callable → Segment contract."""

    segment_id: str
    play_fn: PlayFn
    entry: EntryRequirement
    exit: ExitPredicate
    label: str = ""
    graph_edge_id: str = ""

    @property
    def id(self) -> str:
        return self.segment_id

    def play(self, env: object, **kwargs: Any) -> SegmentResult:
        return self.play_fn(env, **kwargs)

    def play_checked(self, env: object, **kwargs: Any) -> SegmentEvidence:
        """Play and wrap evidence; does not hard-fail entry (caller enforces)."""
        from alttp.opening_route.anchors import match_anchors
        from alttp.startup import snapshot_env

        result = self.play(env, **kwargs)
        snap = result.snapshot
        matched = [a.anchor_id for a in match_anchors(snap)]
        return SegmentEvidence.from_segment_result(
            self.segment_id, result, matched_anchors=matched
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentId": self.segment_id,
            "label": self.label or self.segment_id,
            "graphEdgeId": self.graph_edge_id,
            "entry": self.entry.to_dict(),
            "exit": self.exit.to_dict(),
        }


def _build_registry() -> dict[str, ScriptSegment]:
    """Lazy registry so heavy route modules import only when needed."""
    from alttp.opening_route import castle_to_sword, pocket_to_main_hall, sword_to_zelda
    from alttp.ram import HYRULE_CASTLE_SCREEN, SECRET_PASSAGE_ROOM

    return {
        "castle_to_sword": ScriptSegment(
            segment_id="castle_to_sword",
            play_fn=castle_to_sword.run_from_castle_grounds,
            entry=EntryRequirement(
                description="Controllable on Hyrule Castle grounds (screen 0x1B)",
                screen_id=HYRULE_CASTLE_SCREEN,
                require_indoors=False,
                graph_node_id="castle_grounds",
                anchor_ids=("HyruleCastle_GroundsSpawn_Controllable",),
            ),
            exit=ExitPredicate(
                description="Fighter sword equip RAM ≥ 1 in secret entrance",
                acceptance_keys=("fighter_sword_ram", "in_secret_passage"),
                graph_node_id="room_55_sword",
                verification="continuous",
            ),
            label="Castle grounds → uncle fighter sword",
            graph_edge_id="grounds_to_hole+hole_to_sword",
        ),
        "sword_to_secret_entrance_clear": ScriptSegment(
            segment_id="sword_to_secret_entrance_clear",
            play_fn=sword_to_zelda.run_from_sword,
            entry=EntryRequirement(
                description="Fighter sword in secret entrance room 0x55",
                room_base_id=SECRET_PASSAGE_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                graph_node_id="room_55_sword",
            ),
            exit=ExitPredicate(
                description="Left secret entrance outdoors (courtyard pocket)",
                acceptance_keys=("left_secret_entrance", "fighter_sword_ram"),
                graph_node_id="courtyard_secret_pocket",
                verification="continuous",
            ),
            label="Post-sword → south chamber → stairs outdoor clear",
            graph_edge_id="sword_to_south_chamber+south_stairs_to_courtyard_pocket",
        ),
        "pocket_to_main_hall": ScriptSegment(
            segment_id="pocket_to_main_hall",
            play_fn=pocket_to_main_hall.run_from_pocket,
            entry=EntryRequirement(
                description=(
                    "Outdoors on castle screen 0x1B with fighter sword "
                    "(courtyard secret pocket or open court)"
                ),
                screen_id=HYRULE_CASTLE_SCREEN,
                require_indoors=False,
                require_fighter_sword=True,
                graph_node_id="courtyard_secret_pocket",
            ),
            exit=ExitPredicate(
                description="Indoors main hall room 0x61",
                acceptance_keys=("main_hall", "fighter_sword_ram"),
                graph_node_id="room_61",
                verification="continuous",
            ),
            label="Courtyard pocket → main castle door → room 0x61",
            graph_edge_id="pocket_to_main_hall",
        ),
    }


_REGISTRY: dict[str, ScriptSegment] | None = None


def segment_registry() -> Mapping[str, ScriptSegment]:
    """Return the opening-route Segment registry (cached)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_segment(segment_id: str) -> ScriptSegment:
    reg = segment_registry()
    if segment_id not in reg:
        known = ", ".join(sorted(reg))
        raise KeyError(f"unknown segment {segment_id!r}; known: {known}")
    return reg[segment_id]


def list_segments() -> list[str]:
    return sorted(segment_registry())
