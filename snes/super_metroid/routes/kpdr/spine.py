"""Post-Supers continuous spine — facade for hop order + tip segments.

**Layout (split before 1k LOC):**

- :mod:`super_metroid.routes.kpdr.spine_types` — ``SpineHop`` / ``TipSegment``
- :mod:`super_metroid.routes.kpdr.spine_hops` — ``POST_SUPERS_SPINE`` hop table
- :mod:`super_metroid.routes.kpdr.tip_segments` — tip metadata rows

This module re-exports the public API and owns query / validation helpers.

**Extend a continuous tip (pragmatic checklist):**

1. Pure controller in ``routes/kpdr/`` (+ ``KPDR_SEGMENTS`` if segment-callable).
2. Append a :class:`SpineHop` on :data:`POST_SUPERS_SPINE` with DoorEdge meta
   (``exit_direction`` / ``entry_direction`` / ``requires`` / ``policy_id``)
   when the hop is a continuous product door transition. In-room milestones
   and hops that reuse an earlier edge omit door meta.
3. If new tip: add a :class:`TipSegment` row in :data:`POST_SUPERS_TIP_SEGMENTS`
   (parent, ``graph_id``, report strings, item flags, **CLI identity** —
   ``display_name`` / ``description`` / ``aliases`` / capability flags).
   First Super+ tip parents to ``"supers"``. Catalog ContinuousTip + NamedRoute
   are **derived** from TipSpec after registration; split suffixes derive via
   :func:`hop_ids_to_tip`.
4. :class:`~super_metroid.routes.tips.TipSpec` rows are **generated** in
   ``hops.py`` from this spine (one tip table for early + Super+), including
   CLI fields copied from TipSegment.
5. Super+ continuous ``DoorEdge`` rows are **generated** via
   :func:`continuous_edges_from_spine` / :func:`continuous_edges_for_tips` and
   composed into staged graphs in ``progression/data.py``. Pre-Supers edges and
   pure-only / reverse / branch K4 edges stay hand-authored.

``frog`` and ``bat_cave`` are **siblings** under ``business`` (not a linear
prefix of each other). Branch by tagging each hop with its ending tip id.

Hop room ids use ``rooms.ROOM_*`` only (no raw hex in the spine).
"""

from __future__ import annotations

from collections.abc import Sequence

from super_metroid.progression.types import DoorEdge
from super_metroid.routes.kpdr.spine_hops import POST_SUPERS_SPINE
from super_metroid.routes.kpdr.spine_types import SpineHop, TipSegment
from super_metroid.routes.kpdr.tip_segments import (
    EARLY_TIP_PARENTS,
    POST_SUPERS_TIP_ORDER,
    POST_SUPERS_TIP_SEGMENTS,
    tip_segment_by_id,
)

__all__ = [
    "SpineHop",
    "TipSegment",
    "POST_SUPERS_SPINE",
    "POST_SUPERS_TIP_SEGMENTS",
    "POST_SUPERS_TIP_ORDER",
    "hops_for_tip",
    "hop_ids_for_tip",
    "hop_ids_to_tip",
    "final_room_for_tip",
    "tip_segment_by_id",
    "continuous_edges_from_spine",
    "continuous_edges_for_tips",
    "validate_spine",
]


def hops_for_tip(tip_id: str) -> tuple[SpineHop, ...]:
    """Hop delta owned by ``tip_id`` (not including parent tips)."""
    hops = tuple(h for h in POST_SUPERS_SPINE if h.tip_id == tip_id)
    if not hops:
        raise KeyError(f"No spine hops for tip_id={tip_id!r}")
    return hops


def hop_ids_for_tip(tip_id: str) -> tuple[str, ...]:
    return tuple(h.hop_id for h in hops_for_tip(tip_id))


def hop_ids_to_tip(tip_id: str) -> tuple[str, ...]:
    """Split suffix from first Super+ tip through ``tip_id`` (parent chain + self).

    Walks :data:`POST_SUPERS_TIP_SEGMENTS` only; early parents (``supers``) stop
    the walk. Sibling branches (``frog`` / ``bat_cave``) each include the linear
    chain to their shared parent and only their own hop delta — not the sibling's.
    """
    by_id = tip_segment_by_id()
    if tip_id not in by_id:
        raise KeyError(f"Unknown Super+ tip {tip_id!r}")
    chain: list[str] = []
    cursor: str | None = tip_id
    seen: set[str] = set()
    while cursor is not None and cursor in by_id:
        if cursor in seen:
            raise RuntimeError(f"Cycle in tip parent chain at {cursor!r}")
        seen.add(cursor)
        chain.append(cursor)
        cursor = by_id[cursor].parent_tip_id
    chain.reverse()
    out: list[str] = []
    for tid in chain:
        out.extend(hop_ids_for_tip(tid))
    return tuple(out)


def final_room_for_tip(tip_id: str) -> int:
    return hops_for_tip(tip_id)[-1].to_room


def continuous_edges_from_spine(
    spine: Sequence[SpineHop] = POST_SUPERS_SPINE,
) -> tuple[DoorEdge, ...]:
    """DoorEdges for Super+ continuous product hops (spine source of truth)."""
    return tuple(hop.as_door_edge() for hop in spine if hop.emits_door_edge)


def continuous_edges_for_tips(
    *tip_ids: str,
    spine: Sequence[SpineHop] = POST_SUPERS_SPINE,
) -> tuple[DoorEdge, ...]:
    """DoorEdges emitted by hops whose ``tip_id`` is in ``tip_ids`` (order preserved)."""
    wanted = set(tip_ids)
    return tuple(
        hop.as_door_edge()
        for hop in spine
        if hop.tip_id in wanted and hop.emits_door_edge
    )


def validate_spine(spine: Sequence[SpineHop] = POST_SUPERS_SPINE) -> None:
    """Dev check: every TipSegment has hops; hop tip_ids are known segments."""
    known = set(POST_SUPERS_TIP_ORDER)
    seen_tip_hops: set[str] = set()
    edge_ids: set[str] = set()
    for hop in spine:
        if hop.tip_id not in known:
            raise RuntimeError(
                f"Spine hop {hop.hop_id!r} has unknown tip_id={hop.tip_id!r}"
            )
        seen_tip_hops.add(hop.tip_id)
        if hop.emits_door_edge:
            eid = hop.edge_id or hop.hop_id
            if eid in edge_ids:
                raise RuntimeError(f"Duplicate spine DoorEdge id {eid!r}")
            edge_ids.add(eid)
            if hop.entry_direction == "":
                raise RuntimeError(
                    f"Spine hop {hop.hop_id!r} emits DoorEdge but entry_direction is empty"
                )
    missing = known - seen_tip_hops
    if missing:
        raise RuntimeError(f"Tip segments with no spine hops: {sorted(missing)}")
    for seg in POST_SUPERS_TIP_SEGMENTS:
        parent = seg.parent_tip_id
        if parent is None:
            raise RuntimeError(
                f"Tip {seg.tip_id!r} parent_tip_id is None; "
                f"first Super+ tip must parent to an early tip (usually 'supers')"
            )
        if parent not in known and parent not in EARLY_TIP_PARENTS:
            raise RuntimeError(
                f"Tip {seg.tip_id!r} parent {parent!r} not in Super+ segments "
                f"or early parents {EARLY_TIP_PARENTS}"
            )
