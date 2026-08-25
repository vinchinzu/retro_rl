"""Super+ continuous tip-spec data (built from the spine).

Hop order, tip parent chains, and play callables are declared once in
:mod:`super_metroid.routes.kpdr.spine`. This module builds :class:`TipSpec`
rows and named hop groups as :class:`SpineHop` tuples.

Use :class:`~super_metroid.routes.tips.TipSpec` and :class:`SpineHop` — there
are no historical type aliases here.

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
    "SUPER_TIP_SPECS",
    "SUPER_TIP_BY_ID",
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
    "SPEED_ONLY_HOPS",
    "WAVE_ONLY_HOPS",
    "ICE_ONLY_HOPS",
    "ALPHA_PB_ONLY_HOPS",
    "MOAT_ONLY_HOPS",
    "WS_ONLY_HOPS",
    "PHANTOON_ONLY_HOPS",
]


validate_spine(POST_SUPERS_SPINE)


def _hops_for_tip(tip_id: str) -> tuple[SpineHop, ...]:
    return hops_for_tip(tip_id)


# Named hop groups (one public SpineHop tuple per tip / composite).
# TipSpec rows reuse these same tuple objects (identity, not a second call).
RED_TOWER_HOPS: tuple[SpineHop, ...] = _hops_for_tip("red_tower")
BAT_HOPS: tuple[SpineHop, ...] = _hops_for_tip("bat")
BELOW_SPAZER_HOPS: tuple[SpineHop, ...] = _hops_for_tip("below_spazer")
WAREHOUSE_HOPS: tuple[SpineHop, ...] = _hops_for_tip("warehouse")
HIJUMP_HOPS: tuple[SpineHop, ...] = _hops_for_tip("hijump")
KRAID_HOPS: tuple[SpineHop, ...] = _hops_for_tip("kraid")
VARIA_HOPS: tuple[SpineHop, ...] = _hops_for_tip("varia")
BUSINESS_RETURN_HOPS: tuple[SpineHop, ...] = _hops_for_tip("business")
FROG_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("frog")
FROG_SAVE_HOPS: tuple[SpineHop, ...] = BUSINESS_RETURN_HOPS + FROG_ONLY_HOPS
BAT_CAVE_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("bat_cave")
SPEED_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("speed")
WAVE_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("wave")
ICE_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("ice")
ALPHA_PB_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("alpha_pb")
MOAT_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("moat")
WS_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("ws")
PHANTOON_ONLY_HOPS: tuple[SpineHop, ...] = _hops_for_tip("phantoon")

_HOPS_BY_TIP: dict[str, tuple[SpineHop, ...]] = {
    "red_tower": RED_TOWER_HOPS,
    "bat": BAT_HOPS,
    "below_spazer": BELOW_SPAZER_HOPS,
    "warehouse": WAREHOUSE_HOPS,
    "hijump": HIJUMP_HOPS,
    "kraid": KRAID_HOPS,
    "varia": VARIA_HOPS,
    "business": BUSINESS_RETURN_HOPS,
    "frog": FROG_ONLY_HOPS,
    "bat_cave": BAT_CAVE_ONLY_HOPS,
    "speed": SPEED_ONLY_HOPS,
    "wave": WAVE_ONLY_HOPS,
    "ice": ICE_ONLY_HOPS,
    "alpha_pb": ALPHA_PB_ONLY_HOPS,
    "moat": MOAT_ONLY_HOPS,
    "ws": WS_ONLY_HOPS,
    "phantoon": PHANTOON_ONLY_HOPS,
}


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
        try:
            hops = _HOPS_BY_TIP[seg.tip_id]
        except KeyError as exc:
            raise KeyError(
                f"TipSegment {seg.tip_id!r} missing named hop group in "
                f"{sorted(_HOPS_BY_TIP)}"
            ) from exc
        specs.append(
            TipSpec(
                tip_id=seg.tip_id,
                parent_tip_id=seg.parent_tip_id,
                hops=hops,
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
                display_name=seg.display_name,
                description=seg.description,
                aliases=seg.aliases,
                supports_room_timing=seg.supports_room_timing,
                supports_unlimited_energy=seg.supports_unlimited_energy,
                supports_checkpoint=seg.supports_checkpoint,
            )
        )
    return tuple(specs)


SUPER_TIP_SPECS: tuple[TipSpec, ...] = _build_super_tip_specs()
SUPER_TIP_BY_ID: dict[str, TipSpec] = {
    spec.tip_id: spec for spec in SUPER_TIP_SPECS
}

register_tips(SUPER_TIP_SPECS)
