"""Super+ continuous tip-spec data (built from the spine).

Hop order, tip parent chains, and play callables are declared once in
:mod:`super_metroid.routes.kpdr.spine`. This module builds :class:`TipSpec`
rows and named hop groups as :class:`SpineHop` tuples.

There is no separate RouteHop / PostSupersTipSpec type — use
:class:`~super_metroid.routes.tips.TipSpec` and :class:`SpineHop`.

**Extend a tip:** see the checklist at the top of ``spine.py``.
"""

from __future__ import annotations

from super_metroid.progression import (
    BAT_GRAPH,
    BELOW_SPAZER_GRAPH,
    HIJUMP_GRAPH,
    KRAID_GRAPH,
    RED_TOWER_GRAPH,
    SPEED_GRAPH,
    VARIA_GRAPH,
    WAREHOUSE_GRAPH,
    RoomProgressionGraph,
)
from super_metroid.routes.catalog import SUPERS_SPLITS
from super_metroid.routes.kpdr.spine import (
    POST_SUPERS_SPINE,
    POST_SUPERS_TIP_SEGMENTS,
    SpineHop,
    final_room_for_tip,
    hop_ids_to_tip,
    hops_for_tip,
    validate_spine,
)
from super_metroid.routes.tips import TipSpec, register_tips

# TipSegment.graph_id → staged progression graph (resolved after data builds).
_GRAPH_BY_ID: dict[str, RoomProgressionGraph] = {
    RED_TOWER_GRAPH.graph_id: RED_TOWER_GRAPH,
    BAT_GRAPH.graph_id: BAT_GRAPH,
    BELOW_SPAZER_GRAPH.graph_id: BELOW_SPAZER_GRAPH,
    WAREHOUSE_GRAPH.graph_id: WAREHOUSE_GRAPH,
    HIJUMP_GRAPH.graph_id: HIJUMP_GRAPH,
    KRAID_GRAPH.graph_id: KRAID_GRAPH,
    VARIA_GRAPH.graph_id: VARIA_GRAPH,
    SPEED_GRAPH.graph_id: SPEED_GRAPH,
}

__all__ = [
    "TipSpec",
    "SpineHop",
    "RouteHop",  # alias of SpineHop
    "PostSupersTipSpec",  # alias of TipSpec
    "POST_SUPERS_TIP_SPECS",
    "POST_SUPERS_TIP_BY_ID",
    "SUPER_TIP_SPECS",
    "RED_TOWER_HOPS",
    "BAT_HOPS",
    "BELOW_SPAZER_HOPS",
    "WAREHOUSE_HOPS",
    "HIJUMP_HOPS",
    "KRAID_HOPS",
    "VARIA_HOPS",
    "BUSINESS_RETURN_HOPS",
    "FROG_ONLY_HOPS",
    "FROG_SAVE_HOPS",
    "BAT_CAVE_ONLY_HOPS",
    "_RED_TOWER_HOPS",
    "_BAT_HOPS",
    "_BELOW_SPAZER_HOPS",
    "_WAREHOUSE_HOPS",
    "_HIJUMP_HOPS",
    "_KRAID_HOPS",
    "_VARIA_HOPS",
    "_BUSINESS_RETURN_HOPS",
    "_FROG_ONLY_HOPS",
    "_FROG_SAVE_HOPS",
    "_BAT_CAVE_ONLY_HOPS",
]

# Historical name: RouteHop was a field-subset of SpineHop.
RouteHop = SpineHop
PostSupersTipSpec = TipSpec


validate_spine(POST_SUPERS_SPINE)


def _hops_for_tip(tip_id: str) -> tuple[SpineHop, ...]:
    return hops_for_tip(tip_id)


# Named hop groups (SpineHop tuples; public names for tests / re-exports).
_RED_TOWER_HOPS: tuple[SpineHop, ...] = _hops_for_tip("red_tower")
_BAT_HOPS: tuple[SpineHop, ...] = _hops_for_tip("bat")
_BELOW_SPAZER_HOPS: tuple[SpineHop, ...] = _hops_for_tip("below_spazer")
_WAREHOUSE_HOPS: tuple[SpineHop, ...] = _hops_for_tip("warehouse")
_HIJUMP_HOPS: tuple[SpineHop, ...] = _hops_for_tip("hijump")
_KRAID_HOPS: tuple[SpineHop, ...] = _hops_for_tip("kraid")
_VARIA_HOPS: tuple[SpineHop, ...] = _hops_for_tip("varia")
_BUSINESS_RETURN_HOPS: tuple[SpineHop, ...] = _hops_for_tip("business")
_FROG_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("frog")
_FROG_SAVE_HOPS: tuple[SpineHop, ...] = _BUSINESS_RETURN_HOPS + _FROG_ONLY_HOPS
_BAT_CAVE_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("bat_cave")

RED_TOWER_HOPS = _RED_TOWER_HOPS
BAT_HOPS = _BAT_HOPS
BELOW_SPAZER_HOPS = _BELOW_SPAZER_HOPS
WAREHOUSE_HOPS = _WAREHOUSE_HOPS
HIJUMP_HOPS = _HIJUMP_HOPS
KRAID_HOPS = _KRAID_HOPS
VARIA_HOPS = _VARIA_HOPS
BUSINESS_RETURN_HOPS = _BUSINESS_RETURN_HOPS
FROG_ONLY_HOPS = _FROG_ONLY_HOPS
FROG_SAVE_HOPS = _FROG_SAVE_HOPS
BAT_CAVE_ONLY_HOPS = _BAT_CAVE_ONLY_HOPS


def _build_super_tip_specs() -> tuple[TipSpec, ...]:
    specs: list[TipSpec] = []
    for seg in POST_SUPERS_TIP_SEGMENTS:
        try:
            graph = _GRAPH_BY_ID[seg.graph_id]
        except KeyError as exc:
            raise KeyError(
                f"TipSegment {seg.tip_id!r} graph_id={seg.graph_id!r} not in "
                f"{sorted(_GRAPH_BY_ID)}"
            ) from exc
        specs.append(
            TipSpec(
                tip_id=seg.tip_id,
                parent_tip_id=seg.parent_tip_id,
                hops=_hops_for_tip(seg.tip_id),
                graph=graph,
                kind=seg.kind,
                required_splits=SUPERS_SPLITS + hop_ids_to_tip(seg.tip_id),
                final_room=final_room_for_tip(seg.tip_id),
                success_outcome=seg.success_outcome,
                route_label=seg.route_label,
                source_policy=seg.source_policy,
                timing_source=seg.timing_source,
                entry_condition_key=seg.entry_condition_key,
                ordinary_condition_key=seg.ordinary_condition_key,
                require_hi_jump=seg.require_hi_jump,
                require_varia=seg.require_varia,
            )
        )
    return tuple(specs)


SUPER_TIP_SPECS: tuple[TipSpec, ...] = _build_super_tip_specs()
# Historical names.
POST_SUPERS_TIP_SPECS = SUPER_TIP_SPECS
POST_SUPERS_TIP_BY_ID: dict[str, TipSpec] = {
    spec.tip_id: spec for spec in SUPER_TIP_SPECS
}

register_tips(SUPER_TIP_SPECS)
