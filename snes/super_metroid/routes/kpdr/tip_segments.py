"""Post-Supers tip-segment metadata (TipSegment rows + parent chain).

Hop order lives in :mod:`super_metroid.routes.kpdr.spine_hops`.
Public facade: :mod:`super_metroid.routes.kpdr.spine`.

CLI identity (display_name, aliases, capability flags) lives on each
:class:`TipSegment` and is copied onto generated TipSpec rows in hops.py.
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
        display_name="Power-on → Red Tower (KPDR K1)",
        description=(
            "Supers prefix through farming, Big Pink main, GHZ, Noob, "
            "and natural Red Tower entry."
        ),
        aliases=("red", "k1"),
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
        display_name="Power-on → Bat Room (KPDR K2.0)",
        description=(
            "Red Tower prefix through natural Red Tower descent and "
            "Bat Room entry (first K2 hop)."
        ),
        aliases=("bat_room", "k2_0"),
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
        display_name="Power-on → Below Spazer (KPDR K2.1)",
        description=(
            "Bat prefix through natural three-platform Bat crossing and "
            "Below Spazer entry (Charge Beam collected on Big Pink detour)."
        ),
        aliases=("below", "k2_1"),
        supports_checkpoint=True,
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
            "controllers (…→Charge Big Pink→Bat→Below Spazer→Spazer K2.2 "
            "detour→West/Maridia hallway→Glass→East→Warehouse) + "
            "phase-guarded resources"
        ),
        timing_source="warehouse",
        entry_condition_key="natural_warehouse_entry",
        ordinary_condition_key="post_warehouse_ordinary",
        display_name="Power-on → Warehouse Entrance (KPDR K2.6)",
        description=(
            "Below Spazer prefix (Charge + Spazer mainline detour) through "
            "Maridia hallway West Tunnel, Glass Tunnel, East Tunnel, and "
            "natural Warehouse Entrance. Climb residual until pure green."
        ),
        aliases=("warehouse_entrance", "k2_6"),
        # Charge+Spazer mainline endpoint — dump for pure West/Kraid probes.
        supports_checkpoint=True,
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
        display_name="Power-on → Hi-Jump Boots (KPDR K2.10)",
        description=(
            "Warehouse prefix through Business Center, Hi-Jump shaft, "
            "and natural Hi-Jump Boots collect."
        ),
        aliases=("hi_jump", "hi-jump", "k2_10"),
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
        display_name="Power-on → Kraid entry (KPDR K2.18)",
        description=(
            "Hi-Jump prefix through return to Warehouse, Zeela/Kihunter/"
            "Baby/Eye approach, and natural Kraid room entry."
        ),
        aliases=("kraid_entry", "k2_18"),
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
        display_name="Power-on → Varia Suit (KPDR K3)",
        description=(
            "Kraid-entry prefix through natural Kraid fight, rear exit, "
            "and real Varia PLM collect."
        ),
        aliases=("varia_suit", "k3"),
        supports_checkpoint=True,
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
        display_name="Power-on → Business Center return (KPDR K3→K4)",
        description=(
            "Varia prefix through the natural Kraid return spine and the "
            "right-ledge Warehouse reverse stack into Business Center."
        ),
        aliases=("business_center", "k3_return"),
        supports_checkpoint=True,
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
        display_name="Power-on → Frog Savestation (KPDR K4.0)",
        description=(
            "Business return plus the elevator descent and blue-door exit to "
            "Frog Savestation."
        ),
        aliases=("frog_save", "k4_0"),
        supports_checkpoint=True,
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
        display_name="Power-on → Bat Cave (KPDR K4.4 first Bubble)",
        description=(
            "Business return through Cathedral Entrance → Cathedral → Rising "
            "Tide → Bubble Mountain (R19 double-WJ fire + Super door) into "
            "ordinary Bat Cave. Sibling of Frog Save, not a frog prefix."
        ),
        aliases=("norfair_bat", "bubble_bat", "k4_4", "k4.4"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="speed",
        parent_tip_id="bat_cave",
        graph_id="speed",
        kind="speed",
        success_outcome="speed_collected",
        route_label="speed",
        source_policy=(
            "accepted Bat Cave continuous + pure-green Bat→Speed Hall + "
            "Speed Hall→Speed Booster collect controllers + phase-guarded "
            "resources"
        ),
        timing_source="speed",
        entry_condition_key="natural_speed_room_entry",
        ordinary_condition_key="post_speed_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Speed Booster (KPDR K4.5)",
        description=(
            "Bat Cave tip through Speed Booster Hall and natural Speed "
            "Booster PLM collect. STATUS-promoted default continuous tip "
            "(130,388f ×2 Spazer dual, 2026-08-06)."
        ),
        aliases=("speed_booster", "k4_5", "k4.5"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="wave",
        parent_tip_id="speed",
        graph_id="speed",
        kind="wave",
        success_outcome="wave_collected",
        route_label="wave",
        source_policy=(
            "accepted Speed continuous + pure-green Speed return → Bubble + "
            "Bubble→Single→Double→Wave Super door / PLM controllers + "
            "phase-guarded resources"
        ),
        timing_source="wave",
        entry_condition_key="natural_wave_room_entry",
        ordinary_condition_key="post_wave_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Wave Beam (KPDR K4.10)",
        description=(
            "Speed tip through Speed return → Bubble Mountain, Single "
            "Chamber, Double Chamber (gate + Super door), and natural Wave "
            "Beam PLM collect. Compose tip after pure dual Wave stack "
            "(rr-re9); not STATUS-promoted default."
        ),
        aliases=("wave_beam", "k4_10", "k4.10"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="ice",
        parent_tip_id="wave",
        graph_id="speed",
        kind="ice",
        success_outcome="ice_collected",
        route_label="ice",
        source_policy=(
            "accepted Wave continuous + pure-green Wave→Business return "
            "(rr-vqv3) + Business→Ice Gate→Acid→Snake→Ice PLM (rr-dbu.11; "
            "routes/kpdr/ice/) + phase-guarded resources. Continuous dual "
            "green not claimed until rr-kxge probe evidence."
        ),
        timing_source="ice",
        entry_condition_key="natural_ice_room_entry",
        ordinary_condition_key="post_ice_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Ice Beam (KPDR K4.11)",
        description=(
            "Wave tip through Wave→Business pure return, Business Super door "
            "(floor settle climb/re-pin), Ice Gate, Acid Room (Speed Boost "
            "Blocks), Ice Snake (2WJ), and natural Ice Beam PLM collect. "
            "Compose after rr-vqv3 + rr-dbu.11; not STATUS-promoted without "
            "dual continuous green."
        ),
        aliases=("ice_beam", "k4_11", "k4.11"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="alpha_pb",
        parent_tip_id="ice",
        graph_id="speed",
        kind="alpha_pb",
        success_outcome="alpha_pb_collected",
        route_label="alpha_pb",
        source_policy=(
            "accepted Ice continuous + pure-green Ice return + K5 reverse "
            "tunnels + Red climb + Caterpillar descent + first Alpha PB "
            "(rr-dbu.8; routes/kpdr/ice/ + routes/kpdr/k5/)"
        ),
        timing_source="alpha_pb",
        entry_condition_key="natural_alpha_pb_entry",
        ordinary_condition_key="post_alpha_pb_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Alpha Power Bombs (KPDR K5)",
        description=(
            "Ice tip through Ice return, Business→Warehouse, reverse "
            "tunnels, Red Tower climb, Hellway, Caterpillar descent, and "
            "natural first Alpha PB collect (max 5). Compose after "
            "rr-dbu.8; not STATUS-promoted without dual continuous green."
        ),
        aliases=("alpha_power_bombs", "k5", "k5_0"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="moat",
        parent_tip_id="alpha_pb",
        graph_id="speed",
        kind="moat",
        success_outcome="moat_cleared",
        route_label="moat",
        source_policy=(
            "accepted Ice continuous + K5 Alpha PB pure stack (rr-dbu.8) + "
            "K6 Alpha PB escape / Caterpillar climb / elevator / Kihunter "
            "RLE + Moat spark (rr-dbu.9; routes/kpdr/k6/ + moat.py)"
        ),
        timing_source="moat",
        entry_condition_key="natural_west_ocean_entry",
        ordinary_condition_key="post_moat_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Moat spark / West Ocean (KPDR K6)",
        description=(
            "Alpha PB tip through the Power Bomb escape, Caterpillar "
            "elevator, Crateria Kihunter, Moat, and Speed spark into West "
            "Ocean. Compose after rr-dbu.9; not STATUS-promoted without "
            "dual continuous green."
        ),
        aliases=("west_ocean", "k6", "k6_moat"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="ws",
        parent_tip_id="moat",
        graph_id="speed",
        kind="ws",
        success_outcome="ws_entrance_reached",
        route_label="ws",
        source_policy=(
            "accepted Ice continuous + K5 Alpha PB + K6 Moat spark "
            "(rr-2r06 scratch) + West Ocean over-ocean spark into WS "
            "Entrance (rr-p2bw; routes/kpdr/west_ocean.py)"
        ),
        timing_source="ws",
        entry_condition_key="natural_ws_entrance_entry",
        ordinary_condition_key="post_ws_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Wrecked Ship Entrance (KPDR K6)",
        description=(
            "Moat/West Ocean over-ocean spark + green Super into "
            "Wrecked Ship Entrance 0xCA08. Compose after rr-p2bw; "
            "not STATUS-promoted without dual continuous green."
        ),
        aliases=("wrecked_ship", "ws_entrance", "k6_ws"),
        supports_checkpoint=True,
    ),
    TipSegment(
        tip_id="phantoon",
        parent_tip_id="ws",
        graph_id="speed",
        kind="phantoon",
        success_outcome="phantoon_defeated",
        route_label="phantoon",
        source_policy=(
            "accepted Ice continuous + K5 Alpha PB + K6 Moat/WS spark "
            "(rr-p2bw scratch) + unpowered Entrance→Main (rr-ahjo) + "
            "Main→basement (rr-4btp) + Basement→room (rr-cjpp) + "
            "wiki doppler fight (rr-asyg; routes/kpdr/k6/phantoon_fight.py) + "
            "loot + left-door exit to basement"
        ),
        timing_source="phantoon",
        entry_condition_key="natural_phantoon_entry",
        ordinary_condition_key="post_phantoon_ordinary",
        require_hi_jump=True,
        require_varia=True,
        display_name="Power-on → Phantoon defeat + basement leave (KPDR K6)",
        description=(
            "WS Entrance through unpowered Main Shaft, Basement Gadora, "
            "wiki 2-2-N doppler in 0xCD13 until HP 0 and $D82B bit 0, "
            "then loot and left-door exit to 0xCC6F. Compose after rr-asyg; "
            "not STATUS-promoted without dual continuous green. Default CLI "
            "stays ice. Tip ws still ends at 0xCA08. Charge-only / "
            "charge+missiles / Ice-on X-Factor stay research."
        ),
        aliases=("phan", "k6_phantoon"),
        supports_checkpoint=True,
    ),
)

POST_SUPERS_TIP_ORDER: tuple[str, ...] = tuple(
    seg.tip_id for seg in POST_SUPERS_TIP_SEGMENTS
)


def tip_segment_by_id() -> dict[str, TipSegment]:
    return {seg.tip_id: seg for seg in POST_SUPERS_TIP_SEGMENTS}
