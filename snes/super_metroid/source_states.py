"""Continuous-like pure entry source catalog (code twin of SOURCE_STATES.md).

Pure geometry cards must load a named state whose room (and optional pose/x/y
band) matches the hop. This module validates fingerprints and suggests sources
so executors spend less time hand-picking scratch states.

Binary states live under ``custom_integrations/SuperMetroid-Snes/`` (often
gitignored). The **index** here is source of truth for room expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import SuperMetroidState, probe_pin


@dataclass(frozen=True)
class SourceFingerprint:
    """Expected live RAM fingerprint for a pure entry state."""

    source_id: str
    """Stable id (matches SOURCE_STATES.md row)."""

    relative_path: str
    """Path under SuperMetroid-Snes/ (e.g. ``scratch/post_varia_collected.state``)."""

    room_id: int
    """Expected ``room_id`` after boot (+ short settle)."""

    use_for: str = ""
    """Segment ids / probe names this source feeds (free text)."""

    x_min: int | None = None
    x_max: int | None = None
    y_min: int | None = None
    y_max: int | None = None
    poses: frozenset[int] | None = None
    """When set, pose must be in this set."""

    continuous_like: bool = True
    """False for dev/topology anchors (not continuous evidence)."""

    @property
    def path(self) -> Path:
        return INTEGRATION_DIR / self.relative_path

    def room_hex(self) -> str:
        return f"0x{self.room_id:04X}"


def _fp(
    source_id: str,
    relative_path: str,
    room_id: int,
    use_for: str = "",
    *,
    x: tuple[int, int] | None = None,
    y: tuple[int, int] | None = None,
    poses: frozenset[int] | None = None,
    continuous_like: bool = True,
) -> SourceFingerprint:
    """Compact catalog row. ``x`` / ``y`` are inclusive (min, max) bands."""
    return SourceFingerprint(
        source_id,
        relative_path,
        room_id,
        use_for,
        x_min=None if x is None else x[0],
        x_max=None if x is None else x[1],
        y_min=None if y is None else y[0],
        y_max=None if y is None else y[1],
        poses=poses,
        continuous_like=continuous_like,
    )


# Spine / KPDR continuous-like (keep in sync with docs/SOURCE_STATES.md).
SOURCE_CATALOG: tuple[SourceFingerprint, ...] = (
    _fp("natural_post_spore_spawn", "natural_post_spore_spawn.state", 0x9B5B, "K0–K3 practice; not a continuous tip"),
    _fp("post_varia_collected", "scratch/post_varia_collected.state", 0xA6E2, "varia-to-kraid / varia_to_kraid"),
    _fp("post_varia_continuous", "scratch/post_varia_continuous.state", 0xA6E2, "full-continuous reverse-chain revalidation", x=(100, 140), y=(110, 145), poses=frozenset({81})),
    _fp("post_varia_continuous_to_kraid", "scratch/post_varia_continuous_to_kraid.state", 0xA59F, "natural Varia reverse-chain predecessor"),
    _fp("post_varia_continuous_to_eye", "scratch/post_varia_continuous_to_eye.state", 0xA56B, "natural Varia reverse-chain predecessor"),
    _fp("post_varia_continuous_to_baby", "scratch/post_varia_continuous_to_baby.state", 0xA521, "natural Varia reverse-chain predecessor"),
    _fp("post_varia_to_kraid", "scratch/post_varia_to_kraid_pure.state", 0xA59F, "kraid-to-eye-return / kraid_to_eye_return"),
    _fp("post_kraid_to_eye", "scratch/post_kraid_to_eye_return.state", 0xA56B, "eye-to-baby-return / eye_to_baby_return"),
    _fp("post_eye_to_baby", "scratch/post_eye_to_baby_return.state", 0xA521, "baby-to-kihunter-return / baby_to_kihunter_return"),
    _fp("post_baby_to_kihunter", "scratch/post_baby_to_kihunter_return.state", 0xA4DA, "kihunter-to-zeela-return / kihunter_to_zeela_return"),
    _fp("post_varia_continuous_to_kihunter", "scratch/post_varia_continuous_to_kihunter.state", 0xA4DA, "kihunter-to-zeela-return natural predecessor revalidation", x=(450, 480), y=(390, 400), poses=frozenset({165})),
    _fp("post_varia_continuous_to_zeela", "scratch/post_varia_continuous_to_zeela.state", 0xA471, "zeela-to-warehouse-return natural predecessor revalidation"),
    _fp("post_varia_continuous_to_warehouse", "scratch/post_varia_continuous_to_warehouse.state", 0xA6A1, "warehouse-to-business natural predecessor revalidation"),
    _fp("post_business_continuous", "scratch/post_business_continuous.state", 0xA7DE, "business-to-frog-save / business-to-ice-gate natural continuous predecessor"),
    _fp("post_business_to_ice_gate_pure", "scratch/post_business_to_ice_gate_pure.state", 0xA815, "Ice Gate pin from pre-Speed Business (geometry only; Gate→Acid needs Speed)", x=(1700, 1800), y=(620, 680), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    _fp("post_business_to_ice_gate_wave_speed_pure", "scratch/post_business_to_ice_gate_wave_speed_pure.state", 0xA815, "ice-gate-to-acid pure predecessor (Wave+Speed loadout; rr-9t4)", x=(1700, 1800), y=(620, 680), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    _fp("post_ice_gate_to_acid_pure", "scratch/post_ice_gate_to_acid_pure.state", 0xA75D, "acid-to-snake pure predecessor (rr-9t4 dual handoff)", x=(400, 520), y=(120, 160), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    # Room-clear / pure settle is mid-shaft ~y650 (multi-screen Snake).
    _fp("post_ice_acid_to_snake_pure", "scratch/post_ice_acid_to_snake_pure.state", 0xA8B9, "snake-to-ice pure predecessor (rr-5cf dual handoff)", x=(80, 250), y=(600, 720), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    _fp("post_ice_snake_to_ice_pure", "scratch/post_ice_snake_to_ice_pure.state", 0xA890, "Ice PLM dual pure handoff (rr-5if); pure ice-to-snake return (rr-dbu.8)", x=(160, 220), y=(100, 160), poses=frozenset({1, 2, 9, 10, 75, 77, 81}), continuous_like=False),
    _fp("post_ice_ceres_successor", "scratch/post_ice_ceres_successor.state", 0xA890, "Ceres-successor --to ice dual 146937f leave; ice-to-snake continuous predecessor", x=(160, 220), y=(100, 160), poses=frozenset({1, 2, 9, 10, 75, 77, 81})),
    _fp("post_bubble_entry_continuous", "scratch/post_bubble_entry_continuous.state", 0xACB3, "natural Bubble entry on Ceres-successor spine; bubble-to-bat-cave", x=(20, 80), y=(620, 660), poses=frozenset({1, 2, 9, 25, 81})),
    _fp("post_bat_cave_continuous_ceres_successor", "scratch/post_bat_cave_continuous_ceres_successor.state", 0xB07A, "Ceres-successor --to bat_cave leave (rr-v5c1); bat-cave-to-speed-hall / Ice compose", x=(20, 60), y=(360, 410), poses=frozenset({1, 81})),
    _fp("post_ice_to_snake_pure", "scratch/post_ice_to_snake_pure.state", 0xA8B9, "Ice return pure handoff (rr-dbu.8 ice-to-snake dual 538f); pure snake-to-tutorial", x=(430, 520), y=(360, 430), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    # 81 air residual at export; 164/166 land settle on reload.
    _fp("post_ice_snake_to_tutorial_pure", "scratch/post_ice_snake_to_tutorial_pure.state", 0xA865, "Ice return pure handoff (rr-bf29 snake-to-tutorial dual 2386f); pure tutorial-to-gate", x=(20, 80), y=(100, 160), poses=frozenset({1, 2, 9, 10, 75, 77, 81, 164, 166}), continuous_like=False),
    _fp("post_ice_tutorial_to_gate_pure", "scratch/post_ice_tutorial_to_gate_pure.state", 0xA815, "Ice return pure handoff (rr-81ek tutorial-to-gate dual 969f); pure gate-to-business", x=(450, 900), y=(100, 200), poses=frozenset({1, 2, 9, 10, 75, 77, 81, 164, 166}), continuous_like=False),
    # 25 = turn residual on Super lip after settle; 1/2/9/10 standing.
    _fp("post_ice_gate_to_business_pure", "scratch/post_ice_gate_to_business_pure.state", 0xA7DE, "Ice return pure handoff (rr-e5i6 gate-to-business dual 879f); pure business-to-warehouse / K5 reverse", x=(20, 100), y=(880, 960), poses=frozenset({1, 2, 9, 10, 25, 75, 77, 81, 164, 166}), continuous_like=False),
    # 138 = turn residual after elev exit settle; 1/2/9/10 standing.
    _fp("post_ice_business_to_warehouse_pure", "scratch/post_ice_business_to_warehouse_pure.state", 0xA6A1, "K5 hop 4 pure handoff (rr-3gh9 business-to-warehouse dual 10255f); pure warehouse→east reverse", x=(20, 60), y=(100, 160), poses=frozenset({1, 2, 9, 10, 25, 137, 138}), continuous_like=False),
    # 26 = crouch residual after East multi-screen settle; 1/2/9/10/12 standing.
    _fp("post_ice_warehouse_to_east_pure", "scratch/post_ice_warehouse_to_east_pure.state", 0xCF80, "K5 hop 5 pure handoff (rr-bw2w warehouse-to-east dual 285f); pure east→glass reverse", x=(150, 280), y=(300, 420), poses=frozenset({1, 2, 9, 10, 12, 25, 26}), continuous_like=False),
    # 12 = facing-left standing residual mid Glass floor after settle.
    _fp("post_ice_east_to_glass_pure", "scratch/post_ice_east_to_glass_pure.state", 0xCEFB, "K5 hop 6 pure handoff (rr-68ib east-to-glass dual 253f); pure glass→west reverse", x=(150, 280), y=(350, 420), poses=frozenset({1, 2, 9, 10, 12, 25, 26}), continuous_like=False),
    # 10 = facing-left run residual mid West after settle; 1/2/9/12 standing.
    _fp("post_ice_glass_to_west_pure", "scratch/post_ice_glass_to_west_pure.state", 0xCF54, "K5 hop 7 pure handoff (rr-85c4 glass-to-west dual 211f); pure west→below reverse", x=(150, 280), y=(100, 180), poses=frozenset({1, 2, 9, 10, 12, 25, 26}), continuous_like=False),
    # 82 = facing-right residual after West door settle; standing/run accepted.
    _fp("post_ice_west_to_below_pure", "scratch/post_ice_west_to_below_pure.state", 0xA408, "K5 hop 8 pure handoff (rr-abx5 west-to-below dual 272f); pure below→bat reverse", x=(400, 520), y=(350, 420), poses=frozenset({1, 2, 9, 10, 12, 25, 26, 81, 82}), continuous_like=False),
    # 12 = facing-left stand; 42 = turn residual on reload; standing/run accepted.
    _fp("post_ice_below_to_bat_pure", "scratch/post_ice_below_to_bat_pure.state", 0xA3DD, "K5 hop 9 pure handoff (rr-rp00 below-to-bat dual 485f); pure bat→red reverse", x=(400, 520), y=(100, 180), poses=frozenset({1, 2, 9, 10, 12, 25, 26, 42, 81, 82}), continuous_like=False),
    # 10 = facing-left stand residual Red bottom after Bat left door settle.
    _fp("post_ice_bat_to_red_pure", "scratch/post_ice_bat_to_red_pure.state", 0xA253, "K5 hop 11 pure handoff (rr-0ue1 bat-to-red dual 718f); pure Red→Hellway climb", x=(150, 280), y=(2380, 2500), poses=frozenset({1, 2, 9, 10, 12, 25, 26, 81, 82}), continuous_like=False),
    # 11 = Ice-climb ordinary left-door airborne; 29 = tape morph settle.
    _fp("post_ice_red_to_hellway_pure", "scratch/post_ice_red_to_hellway_pure.state", 0xA2F7, "K5 hop 12 tape leave (rr-av5s 6199f p29); Ice-pin spine is ordinary (39,139) p11", x=(20, 80), y=(120, 180), poses=frozenset({1, 2, 9, 10, 11, 12, 25, 26, 29, 30, 81, 82}), continuous_like=False),
    _fp("post_ice_hellway_to_caterpillar_pure", "scratch/post_ice_hellway_to_caterpillar_pure.state", 0xA322, "K5 hop 13 handoff (rr-bvd1 Hellway-to-Caterpillar dual 2218f); Caterpillar→Alpha PB", x=(20, 80), y=(1380, 1450), poses=frozenset({1, 2, 9, 10, 11, 12, 25, 26, 81, 82}), continuous_like=False),
    _fp("post_ice_caterpillar_to_alpha_pb_pure", "scratch/post_ice_caterpillar_to_alpha_pb_pure.state", 0xA3AE, "K5 hop 14 handoff (rr-dbu.8 dual 1385f); Ice-pin compose Alpha PB 20016f; Moat approach", x=(320, 365), y=(150, 190), poses=frozenset({138}), continuous_like=False),
    _fp("post_alpha_pb_to_caterpillar_pure", "scratch/post_alpha_pb_to_caterpillar_pure.state", 0xA322, "K6 hop 0 handoff (Alpha PB escape dual 2102f); Caterpillar climb", x=(20, 80), y=(1910, 1940), poses=frozenset({1, 2, 9, 10, 164}), continuous_like=False),
    _fp("post_caterpillar_to_elevator_pure", "scratch/post_caterpillar_to_elevator_pure.state", 0x962A, "K6 hop 1 handoff (Caterpillar climb dual 1869f); elevator to Kihunter", x=(110, 145), y=(270, 315), poses=frozenset({155}), continuous_like=False),
    _fp("post_elevator_to_kihunter_pure", "scratch/post_elevator_to_kihunter_pure.state", 0x948C, "K6 hop 2 handoff (elevator to Kihunter dual 627f); Kihunter to Moat", x=(370, 415), y=(670, 725), poses=frozenset({144}), continuous_like=False),
    _fp("post_kihunter_to_moat_pure", "scratch/post_kihunter_to_moat_pure.state", 0x95FF, "K6 hop 3 handoff (Kihunter to Moat dual 1844f); Moat spark", x=(20, 80), y=(120, 170), poses=frozenset({1, 2, 9, 10}), continuous_like=False),
    _fp("post_moat_poweron", "scratch/post_moat_poweron.state", 0x93FE, "power-on --to moat dual 175526f leave (rr-2r06); west-ocean spark to WS", x=(30, 80), y=(1140, 1185), poses=frozenset({1})),
    _fp("post_moat_poweron_wo_to_ws", "scratch/post_moat_poweron_wo_to_ws.state", 0xCA08, "over-ocean spark from power-on moat leave dual 627f; WS entrance", x=(40, 90), y=(120, 160), poses=frozenset({1})),
    _fp("post_ws_poweron", "scratch/post_ws_poweron.state", 0xCA08, "power-on --to ws dual 176141f leave (rr-p2bw); ship interior / Phantoon", x=(40, 90), y=(120, 160), poses=frozenset({1})),
    _fp("post_ws_entrance_to_main", "scratch/post_ws_entrance_to_main.state", 0xCAF6, "pure dual GREEN Entrance→Main (rr-ahjo, 403f ×2); Main Shaft → basement (rr-4btp)", x=(1020, 1100), y=(870, 940), poses=frozenset({9, 81})),
    _fp("post_ws_main_to_basement", "scratch/post_ws_main_to_basement.state", 0xCC6F, "pure dual GREEN Main Shaft→basement (rr-4btp, 1208f ×2); Basement → Phantoon", x=(600, 720), y=(60, 160), poses=frozenset({1, 2, 24})),
    _fp("post_ws_basement_to_phantoon", "scratch/post_ws_basement_to_phantoon.state", 0xCD13, "pure dual GREEN Basement→Phantoon room (rr-cjpp, 718f ×2); Phantoon fight", x=(20, 80), y=(90, 160), poses=frozenset({1, 9, 81})),
    _fp("post_phantoon_leave", "scratch/post_phantoon_leave.state", 0xCC6F, "doppler fight + loot/exit dual GREEN 12455f ×2 (rr-asyg); WS Basement right door; do not clobber post_phantoon_poweron", x=(1180, 1280), y=(120, 160), poses=frozenset({1, 9, 10})),
    _fp("post_frog_continuous", "scratch/post_frog_continuous.state", 0xB167, "frog-save-to-speedway natural continuous predecessor", x=(50, 70), y=(130, 145), poses=frozenset({1})),
    _fp("post_business_to_frog_save_pure", "scratch/post_business_to_frog_save_pure.state", 0xB167, "record of the Business-to-Frog pure handoff; superseded by post_frog_continuous", x=(50, 70), y=(130, 145), poses=frozenset({1})),
    _fp("post_bat_cave_to_speed_hall_pure", "scratch/post_bat_cave_to_speed_hall_pure.state", 0xACF0, "speed-hall-to-speed pure predecessor (Bat→Hall GREEN successor)", continuous_like=False),
    _fp("post_speed_hall_to_speed_pure", "scratch/post_speed_hall_to_speed_pure.state", 0xAD1B, "post Speed Booster pure collect; human Wave/Ice/Moat record", x=(150, 200), y=(100, 150), continuous_like=False),
    _fp("post_speed_collected", "scratch/post_speed_collected.state", 0xAD1B, "speed-return-to-bubble pure / human Wave/Ice/Moat record (standing handoff)", x=(150, 200), y=(100, 150), continuous_like=False),
    _fp("post_speed_return_to_bubble_pure", "scratch/post_speed_return_to_bubble_pure.state", 0xACB3, "bubble-to-single-chamber pure / Wave branch (post Speed return) predecessor", x=(450, 500), y=(90, 160), continuous_like=False),
    _fp("post_bubble_to_single_chamber_pure", "scratch/post_bubble_to_single_chamber_pure.state", 0xAD5E, "single-to-double-chamber pure / Wave branch predecessor", x=(20, 80), y=(100, 180), continuous_like=False),
    _fp("post_single_to_double_chamber_pure", "scratch/post_single_to_double_chamber_pure.state", 0xADAD, "double-chamber-to-wave / Wave PLM pure predecessor", x=(20, 80), y=(100, 180), continuous_like=False),
    _fp("post_double_chamber_to_wave_pure", "scratch/post_double_chamber_to_wave_pure.state", 0xADDE, "Wave pure successor / Ice branch predecessor (STALE 2026-08-09: loads 0xADAD ~(923,311) — use dev_wave_collected)", x=(140, 220), y=(90, 180), continuous_like=False),
    _fp("dev_wave_collected", "scratch/dev_wave_collected.state", 0xADDE, "wave-to-double-chamber pure return / Wave tip handoff (chozo pin)", x=(150, 200), y=(90, 150), continuous_like=False),
    _fp("post_wave_to_double_chamber_pure", "scratch/post_wave_to_double_chamber_pure.state", 0xADAD, "double-to-single return pure predecessor (Wave return stack)", x=(900, 1050), y=(100, 180), continuous_like=False),
    # Live dual pin (rr-qpkd): ~(216,630) after bottom-left door settle.
    _fp("post_double_to_single_chamber_pure", "scratch/post_double_to_single_chamber_pure.state", 0xAD5E, "single-to-bubble return pure predecessor (Wave return stack)", x=(180, 260), y=(580, 680), continuous_like=False),
    _fp("post_single_to_bubble_pure", "scratch/post_single_to_bubble_pure.state", 0xACB3, "bubble-to-farm return pure predecessor (Wave return stack)", x=(400, 560), y=(350, 450), continuous_like=False),
    # Farm right-top settle after Bubble bottom-left leave ~(472–523,139).
    _fp("post_bubble_to_farm_pure", "scratch/post_bubble_to_farm_pure.state", 0xAF72, "farm-to-speedway return pure predecessor (Wave return stack; needs Speed)", x=(400, 560), y=(100, 180), continuous_like=False),
    # Speedway right entry after Farm left blue door (8-screen tunnel) ~(2000–2040,139).
    _fp("post_farm_to_speedway_pure", "scratch/post_farm_to_speedway_pure.state", 0xB106, "speedway-to-frog-save return pure predecessor (Wave return stack; needs Speed)", x=(1950, 2100), y=(100, 180), continuous_like=False),
    # Frog Save right entry after Speedway left leave ~(200–240,139).
    _fp("post_speedway_to_frog_save_pure", "scratch/post_speedway_to_frog_save_pure.state", 0xB167, "frog-save-to-business return pure predecessor (Wave return stack; rr-vsjy)", x=(160, 280), y=(100, 180), continuous_like=False),
    # Business floor after Frog Save left leave (Frog door is floor-right;
    # dual pin ~(216,1419) p12). Ice Super is mid-shaft — compose climbs.
    _fp("post_frog_save_to_business_pure", "scratch/post_frog_save_to_business_pure.state", 0xA7DE, "Wave→Business return stack tip / Ice pure predecessor (rr-vsjy dual)", x=(160, 280), y=(1380, 1460), continuous_like=False),
    _fp("post_kihunter_to_zeela", "scratch/post_kihunter_to_zeela_return.state", 0xA471, "zeela-to-warehouse-return / zeela_to_warehouse_return"),
    _fp("post_zeela_to_warehouse", "scratch/post_zeela_to_warehouse_return.state", 0xA6A1, "warehouse-to-business reverse / warehouse_to_business"),
    _fp("business_climb_entry", "scratch/continuous_like_business_climb_entry.state", 0xA7DE, "business-to-warehouse"),
    _fp("continuous_like_bat", "scratch/continuous_like_bat.state", 0xA3DD, "bat pure / dwell isolation"),
    _fp("post_below_spazer_with_charge_continuous", "scratch/post_below_spazer_with_charge_continuous.state", 0xA408, "Spazer climb pure / below_spazer continuous with Charge", x=(40, 70), y=(380, 410)),
    _fp("post_warehouse_with_spazer_continuous", "scratch/post_warehouse_with_spazer_continuous.state", 0xA6A1, "warehouse/hijump pure with Charge+Spazer; continuous-like Charge pin + pure Spazer detour + West→Warehouse", x=(30, 80), y=(100, 150)),
    # beams OR'd onto pure Hall geometry
    _fp("post_speed_hall_pre_speed_with_spazer", "scratch/post_speed_hall_pre_speed_with_spazer.state", 0xACF0, "speed-hall-to-speed / human pre-Speed with Charge+Spazer", x=(40, 90), y=(110, 150), continuous_like=False),
    # geometry place; inventory continuous-legal
    _fp("pre_spazer_door_with_charge", "scratch/pre_spazer_door_with_charge.state", 0xA408, "below-spazer-to-spazer / below_spazer_to_spazer", x=(440, 490), y=(120, 160), continuous_like=False),
    _fp("post_spazer_entry_pure", "scratch/post_spazer_entry_pure.state", 0xA447, "spazer-collect / spazer_collect"),
    _fp("post_spazer_collect_pure", "scratch/post_spazer_collect_pure.state", 0xA447, "spazer-return-to-below / spazer_return_to_below", x=(150, 200), y=(150, 200)),
    _fp("post_spazer_return_pure", "scratch/post_spazer_return_pure.state", 0xA408, "spazer-top-to-west / spazer_top_to_west / top→mid→West", x=(350, 420), y=(140, 170)),
    _fp("post_spazer_west_pure", "scratch/post_spazer_west_pure.state", 0xCF54, "post Spazer pure West handoff after top→west", x=(20, 80), y=(80, 160)),
    _fp("red_to_warehouse", "scratch/red_to_warehouse_controller.state", 0xA6A1, "warehouse-hijump-kraid"),
    # K6 Moat / West Ocean / WS (shine path; pure dual GREEN product + chain-ws)
    _fp("post_kihunter_pre_moat_spark", "scratch/post_kihunter_pre_moat_spark.state", 0x948C, "moat pure / chain-ws / pre-moat human; not continuous tip", x=(20, 120), y=(100, 220), continuous_like=False),
    _fp("alpha_pb_to_moat_human_end", "scratch/alpha_pb_to_moat_human_end.state", 0x95FF, "moat standing handoff for chain-ws (leave+open+spark); dual green with product pin", continuous_like=False),
    _fp("post_moat_west_ocean_spark", "scratch/post_moat_west_ocean_spark.state", 0x93FE, "west-ocean pure-ws / west_ocean_over_ocean_spark / human west-ocean-to-ws", x=(30, 80), y=(1140, 1185)),
    _fp("post_west_ocean_ws_spark", "scratch/post_west_ocean_ws_spark.state", 0xCA08, "ws-entrance human / ship free-record after pure-ws or chain-ws; Phantoon path", x=(40, 90), y=(120, 160)),
    _fp("ws_ship_human_end", "scratch/ws_ship_human_end.state", 0xCD13, "phantoon_combat strategy entry (human ship tape end)", x=(150, 280), y=(160, 220), continuous_like=False),
    _fp("post_phantoon_defeated", "scratch/post_phantoon_defeated.state", 0xCD13, "post-phantoon human / Gravity path; WS boss bit 0 set", x=(140, 220), y=(160, 220), continuous_like=False),
    _fp("post_gravity_caterpillar", "scratch/post_gravity_caterpillar.state", 0xA322, "post-gravity human / Grapple side-trek + Maridia; tail of gravity_path_human (not Gravity chozo)", x=(40, 120), y=(1380, 1460), continuous_like=False),
    # dumps at leave of Grapple Beam; boot settles into Tutorial 1
    _fp("post_grapple_beam_human", "scratch/post_grapple_beam_human.state", 0xAC00, "post-Grapple human / Maridia return free-record (--from post-grapple); items 0x7125; not continuous tip", x=(100, 400), y=(80, 200), continuous_like=False),
    _fp("post_crocomire_farming_human", "scratch/post_crocomire_farming_human.state", 0xAA82, "post-Croc farm pin from maridia_grapple_human assist-sync replay", x=(0, 80), y=(100, 180), continuous_like=False),
    _fp("post_grapple_main_street", "scratch/post_grapple_main_street.state", 0xCFC9, "standalone maridia_main_street_human F5 (items 0x7125); not the living full_start_v1 seam", x=(350, 450), y=(1900, 2050), continuous_like=False),
    _fp("full_start_v1_main_street", "scratch/full_start_v1_main_street.state", 0xCFC9, "full_start_v1 Grapple→Main Street F5 (items 0x7125); Maridia next / Botwoon free-record; --from main-street", x=(350, 450), y=(1900, 2050), continuous_like=False),
    _fp("post_grapple_croc_escape_human", "scratch/post_grapple_croc_escape_human.state", 0xAA0E, "F6 pin mid maridia_main_street_human (Croc Escape before Business)", x=(180, 250), y=(100, 180), continuous_like=False),
    _fp("full_start_v1_golden_torizo", "scratch/full_start_v1_golden_torizo.state", 0xB283, "full_start_v1 Golden Torizo room enter (items 0x7325 beams 0x100F); left door, GT full HP; --from golden-torizo / ./play gt", x=(20, 80), y=(100, 180), continuous_like=False),
    _fp("full_start_v1_metal_pirates", "scratch/full_start_v1_metal_pirates.state", 0xB62B, "full_start_v1 Metal Pirates right door (items 0x732F beams 0x100F); Screw on, both pirates alive; --from metal-pirates / ./play metal-pirates", x=(680, 760), y=(140, 200), continuous_like=False),
    _fp("full_start_v1_plasma", "scratch/full_start_v1_plasma.state", 0xD2AA, "full_start_v1 Plasma Room F5 (items 0x7325 beams 0x100F); post-Plasma leave / LN next; --from plasma-beam", x=(350, 450), y=(580, 700), continuous_like=False),
    _fp("post_space_jump", "scratch/post_space_jump.state", 0xD9AA, "post-Space Jump collect (items 0x7325); primary next-segment start after maridia_botwoon_path_human; --from post-space-jump", x=(50, 150), y=(120, 200), continuous_like=False),
    _fp("post_space_jump_precious", "scratch/post_space_jump_precious.state", 0xD78F, "Precious Room first return after SJ (items 0x7325)", x=(30, 80), y=(620, 700), continuous_like=False),
    _fp("post_draygon_precious", "scratch/post_draygon_precious.state", 0xD78F, "post-Draygon+SJ Precious F5 end of maridia_botwoon_path_human; --from post-draygon", x=(40, 80), y=(620, 700), continuous_like=False),
    _fp("post_spring_ball", "scratch/post_spring_ball.state", 0xD6D0, "Spring Ball collect (items 0x7327); --from post-spring-ball", x=(340, 420), y=(320, 400), continuous_like=False),
    _fp("post_ln_main_hall", "scratch/post_ln_main_hall.state", 0xB236, "LN Main Hall human end (items 0x7327 beams 0x100F); Ridley / Golden Torizo free-record; --from main-hall", x=(1100, 1200), y=(600, 700), continuous_like=False),
    _fp("post_ln_elevator_save", "scratch/post_ln_elevator_save.state", 0xB1BB, "LN Elevator Save before Main Hall; items 0x7327", x=(160, 240), y=(100, 180), continuous_like=False),
    _fp("post_screw_attack", "scratch/post_screw_attack.state", 0xB6C1, "Screw Attack collect (items 0x732F); item_delta f10857 of post-main-hall; --from post-screw", x=(140, 210), y=(640, 700), continuous_like=False),
    _fp("full_start_v1_ridley", "scratch/full_start_v1_ridley.state", 0xB698, "full_start_v1 Ridley Tank after fight + tank (items 0x732F, Norfair boss bit); --from post-ridley / ./play post-ridley", x=(190, 250), y=(150, 220), continuous_like=False),
    _fp("post_ridley_tank", "scratch/post_ridley_tank.state", 0xB698, "older post-main-hall Ridley Tank pin (items 0x732F Ridley bit); product seam is full_start_v1_ridley", x=(180, 260), y=(100, 180), continuous_like=False),
    _fp("post_ridley_farming", "scratch/post_ridley_farming.state", 0xB37A, "LN Farming after leaving Ridley; exit path; --from post-ridley-farming", x=(30, 80), y=(100, 180), continuous_like=False),
    _fp("post_bosses_landing_site", "scratch/post_bosses_landing_site.state", 0x91F8, "Landing Site post-Ridley return (all 4 bosses, Screw items 0x732F); G4 statues / Tourian free-record; --from post-bosses", x=(1100, 1200), y=(1050, 1150), continuous_like=False),
    # Dev / topology (not continuous evidence)
    _fp("dev_kpdr_kraid_entry", "dev_kpdr_kraid_entry.state", 0xA59F, "kraid fight pure", continuous_like=False),
    _fp("dev_kpdr_varia", "dev_kpdr_varia.state", 0xA6E2, "varia probes", continuous_like=False),
)


_BY_ID: dict[str, SourceFingerprint] = {s.source_id: s for s in SOURCE_CATALOG}


def get_source(source_id: str) -> SourceFingerprint:
    """Return catalog row or raise ``KeyError``."""
    return _BY_ID[source_id]


def suggest_sources_for_room(
    room_id: int,
    *,
    continuous_like_only: bool = True,
    segment_hint: str = "",
) -> tuple[SourceFingerprint, ...]:
    """Rank catalog rows for a pure hop entry room.

    Prefers continuous-like rows whose ``use_for`` mentions ``segment_hint``,
    then any continuous-like match, then dev anchors.
    """
    hint = segment_hint.replace("_", "-").lower()
    matches = [s for s in SOURCE_CATALOG if s.room_id == room_id]
    if continuous_like_only:
        matches = [s for s in matches if s.continuous_like]

    def rank(s: SourceFingerprint) -> tuple[int, int, str]:
        use = s.use_for.replace("_", "-").lower()
        hint_hit = 0 if hint and hint in use else 1
        cont = 0 if s.continuous_like else 1
        return (hint_hit, cont, s.source_id)

    return tuple(sorted(matches, key=rank))


def suggest_source_path(
    room_id: int,
    *,
    segment_hint: str = "",
    continuous_like_only: bool = True,
) -> Path | None:
    """Best catalog path for ``room_id``, or None if unknown."""
    ranked = suggest_sources_for_room(
        room_id,
        continuous_like_only=continuous_like_only,
        segment_hint=segment_hint,
    )
    return ranked[0].path if ranked else None


@dataclass(frozen=True)
class FingerprintCheck:
    ok: bool
    failures: tuple[str, ...]
    pin: dict[str, object]
    source_id: str | None = None


def validate_fingerprint(
    state: SuperMetroidState,
    *,
    expected_room: int | None = None,
    source: SourceFingerprint | None = None,
    source_id: str | None = None,
) -> FingerprintCheck:
    """Validate live state against room / optional catalog fingerprint bands."""
    fp = source
    if fp is None and source_id is not None:
        fp = get_source(source_id)
    room = expected_room if expected_room is not None else (fp.room_id if fp else None)
    failures: list[str] = []
    if room is not None and state.room_id != room:
        failures.append(f"room: expected 0x{room:04X}, got 0x{state.room_id:04X}")
    if fp is not None:
        if fp.x_min is not None and state.samus_x < fp.x_min:
            failures.append(f"x<{fp.x_min}: got {state.samus_x}")
        if fp.x_max is not None and state.samus_x > fp.x_max:
            failures.append(f"x>{fp.x_max}: got {state.samus_x}")
        if fp.y_min is not None and state.samus_y < fp.y_min:
            failures.append(f"y<{fp.y_min}: got {state.samus_y}")
        if fp.y_max is not None and state.samus_y > fp.y_max:
            failures.append(f"y>{fp.y_max}: got {state.samus_y}")
        if fp.poses is not None and state.pose not in fp.poses:
            failures.append(f"pose {state.pose} not in {sorted(fp.poses)}")
    return FingerprintCheck(
        ok=not failures,
        failures=tuple(failures),
        pin=probe_pin(state),
        source_id=fp.source_id if fp else None,
    )


def match_source_by_path(path: Path) -> SourceFingerprint | None:
    """Resolve a filesystem path to a catalog row when possible."""
    resolved = path.resolve()
    for row in SOURCE_CATALOG:
        try:
            if row.path.resolve() == resolved:
                return row
        except OSError:
            continue
        # Also match by suffix relative path string.
        if path.as_posix().endswith(row.relative_path):
            return row
        if path.name == Path(row.relative_path).name:
            # Ambiguous name-only match only if unique.
            same = [
                s for s in SOURCE_CATALOG if Path(s.relative_path).name == path.name
            ]
            if len(same) == 1:
                return same[0]
    return None
