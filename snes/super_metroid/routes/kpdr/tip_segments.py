"""Post-Supers tip-segment metadata (TipSegment rows + parent chain).

Hop order lives in :mod:`super_metroid.routes.kpdr.spine_hops`.
Public facade: :mod:`super_metroid.routes.kpdr.spine`.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.spine_types import TipSegment

__all__ = [
    "POST_SUPERS_TIP_SEGMENTS",
    "POST_SUPERS_TIP_ORDER",
    "EARLY_TIP_PARENTS",
    "tip_segment_by_id",
]


# Early tip ids that may appear as parent_tip_id for Super+ segments.
EARLY_TIP_PARENTS: frozenset[str] = frozenset({"morph", "bombs", "spore", "supers"})

POST_SUPERS_TIP_SEGMENTS: tuple[TipSegment, ...] = (
    TipSegment(
        tip_id="red_tower",
        parent_tip_id="supers",
        graph_id="red_tower",
        kind="red_tower",
        success_outcome="red_tower_entry",
        route_label="red_tower",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1 controllers "
            "(Super→farm→Big Pink main→GHZ→Noob→Red) + phase-guarded resources"
        ),
        timing_source="red_tower",
        entry_condition_key="natural_red_tower_entry",
        ordinary_condition_key="post_red_ordinary",
    ),
    TipSegment(
        tip_id="bat",
        parent_tip_id="red_tower",
        graph_id="bat",
        kind="bat",
        success_outcome="bat_room_entry",
        route_label="bat",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2.0 "
            "controllers (Super→farm→Big Pink main→GHZ→Noob→Red→Bat) + "
            "phase-guarded resources"
        ),
        timing_source="bat",
        entry_condition_key="natural_bat_room_entry",
        ordinary_condition_key="post_bat_ordinary",
    ),
    TipSegment(
        tip_id="below_spazer",
        parent_tip_id="bat",
        graph_id="below_spazer",
        kind="below_spazer",
        success_outcome="below_spazer_entry",
        route_label="below_spazer",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Red→Bat→Below Spazer) + phase-guarded resources"
        ),
        timing_source="below_spazer",
        entry_condition_key="natural_below_spazer_entry",
        ordinary_condition_key="post_below_spazer_ordinary",
    ),
    TipSegment(
        tip_id="warehouse",
        parent_tip_id="below_spazer",
        graph_id="warehouse",
        kind="warehouse",
        success_outcome="warehouse_entry",
        route_label="warehouse",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Bat→Below Spazer→West→Glass→East→Warehouse) + "
            "phase-guarded resources"
        ),
        timing_source="warehouse",
        entry_condition_key="natural_warehouse_entry",
        ordinary_condition_key="post_warehouse_ordinary",
    ),
    TipSegment(
        tip_id="hijump",
        parent_tip_id="warehouse",
        graph_id="hijump",
        kind="hijump",
        success_outcome="hijump_collected",
        route_label="hijump",
        source_policy=(
            "accepted Warehouse continuous prefix + KPDR Hi-Jump controllers "
            "(Warehouse→Business→shaft→HJ room collect) + phase-guarded resources"
        ),
        timing_source="hijump",
        entry_condition_key="natural_hijump_room",
        ordinary_condition_key="post_hijump_ordinary",
        require_hi_jump=True,
    ),
    TipSegment(
        tip_id="kraid",
        parent_tip_id="hijump",
        graph_id="kraid",
        kind="kraid",
        success_outcome="kraid_entry",
        route_label="kraid",
        source_policy=(
            "accepted Warehouse prefix + Hi-Jump collect/return + KPDR Kraid "
            "approach controllers + phase-guarded resources"
        ),
        timing_source="kraid",
        entry_condition_key="natural_kraid_entry",
        ordinary_condition_key="post_kraid_ordinary",
        require_hi_jump=True,
    ),
    TipSegment(
        tip_id="varia",
        parent_tip_id="kraid",
        graph_id="varia",
        kind="varia",
        success_outcome="varia_collected",
        route_label="varia",
        source_policy=(
            "accepted Kraid-entry continuous chain + combat.kraid fight/Varia "
            "policy + phase-guarded resources"
        ),
        timing_source="varia",
        entry_condition_key="natural_varia_room",
        ordinary_condition_key="post_varia_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    TipSegment(
        tip_id="business",
        parent_tip_id="varia",
        graph_id="speed",
        kind="business",
        success_outcome="business_return",
        route_label="business",
        source_policy=(
            "accepted Varia continuous chain + natural K3 return controllers "
            "(Varia→Kraid→Eye→Baby→Kihunter→Zeela→Warehouse→Business) + "
            "phase-guarded resources"
        ),
        timing_source="business",
        entry_condition_key="natural_business_return",
        ordinary_condition_key="post_business_return_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    TipSegment(
        tip_id="frog",
        parent_tip_id="business",
        graph_id="speed",
        kind="frog_save",
        success_outcome="frog_save_reached",
        route_label="frog",
        source_policy=(
            "accepted Business return chain + natural Business elevator descent "
            "and Frog blue-door controller + phase-guarded resources"
        ),
        timing_source="frog",
        entry_condition_key="natural_frog_save",
        ordinary_condition_key="post_frog_save_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
    TipSegment(
        tip_id="bat_cave",
        parent_tip_id="business",
        graph_id="speed",
        kind="bat_cave",
        success_outcome="bat_cave_reached",
        route_label="bat_cave",
        source_policy=(
            "accepted Business return chain + Cathedral first-Bubble pure "
            "controllers (CATH-01…04 + Bubble R19 double-WJ fire / Super door) "
            "+ phase-guarded resources"
        ),
        timing_source="bat_cave",
        entry_condition_key="natural_bat_cave_entry",
        ordinary_condition_key="post_bat_cave_ordinary",
        require_hi_jump=True,
        require_varia=True,
    ),
)

POST_SUPERS_TIP_ORDER: tuple[str, ...] = tuple(
    seg.tip_id for seg in POST_SUPERS_TIP_SEGMENTS
)


def tip_segment_by_id() -> dict[str, TipSegment]:
    return {seg.tip_id: seg for seg in POST_SUPERS_TIP_SEGMENTS}
