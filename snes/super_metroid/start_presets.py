"""Guided-human start presets and power-on aliases.

Short names → relative state path under SuperMetroid-Snes integration + blurb.
Kept out of ``guided_human.py`` so the recorder CLI stays under 1k lines.

Structure: ``_CANONICAL`` holds one entry per unique state; ``_ALIASES`` maps
alternate short names onto those keys. ``START_PRESETS`` is built at import.
"""

from __future__ import annotations

POWER_ON_STARTS = frozenset({"beginning", "full", "power-on", "poweron", "start"})

# Canonical short name → (relative state path, blurb)
_CANONICAL: dict[str, tuple[str, str]] = {
    # --- full_start_v1 item-to-item seams ---
    "morph": (
        "scratch/full_start_v1_morph.state",
        "full_start_v1 Morph Ball 0x9E9F ~(1125,699) items 0x0004 — continue post-Morph",
    ),
    "bomb": (
        "scratch/full_start_v1_bomb.state",
        "full_start_v1 Bomb Torizo 0x9804 items 0x1004 — continue post-Bombs",
    ),
    "supers": (
        "scratch/full_start_v1_supers.state",
        "full_start_v1 Spore Spawn Super 0x9B5B items 0x1004 supers≥5 — continue",
    ),
    "hj": (
        "scratch/full_start_v1_hj.state",
        "full_start_v1 Hi Jump 0xA9E5 ~(68,156) items 0x1104 — continue post-HJ",
    ),
    "resume": (
        "scratch/full_start_v1_big_pink.state",
        "full_start_v1 farthest pin — Big Pink enter f49424 items 0x1004",
    ),
    "varia": (
        "scratch/full_start_v1_varia.state",
        "full_start_v1 Varia pickup 0xA6E2 items 0x1105 (Morph+Bombs+HiJump+Varia)",
    ),
    "varia-end": (
        "scratch/full_start_v1_varia_end.state",
        "full_start_v1 F5 end in Varia room ~(120,136) items 0x1105",
    ),
    "bubble-human": (
        "scratch/full_start_v1_bubble.state",
        "full_start_v1 Bubble Mountain end 0xACB3 items 0x1105 — continue Norfair",
    ),
    "bubble-save": (
        "scratch/bubble_save.state",
        "Bubble Save 0xB0DD ~(96,152) items 0x1105 — leave RIGHT → runway WJ climb",
    ),
    "full-start-bubble-save": (
        "scratch/full_start_v1_bubble_save.state",
        "full_start_v1 Bubble Save durable seam (distinct pin from bubble-save)",
    ),
    "bat-cave": (
        "scratch/full_start_v1_bat.state",
        "Bat Cave 0xB07A ~(55,395) items 0x1105 beams 0x1004 — post Bubble climb",
    ),
    "wave": (
        "scratch/full_start_v1_wave.state",
        "Wave Room 0xADDE ~(171,123) items 0x3105 beams 0x1005 — continue to Ice",
    ),
    "alpha-pb": (
        "scratch/full_start_v1_alpha_pb.state",
        "Alpha PB Room 0xA3AE ~(341,171) items 0x3105 beams 0x1007 pb=5 — post collect",
    ),
    # --- cathedral / business pure chain tips ---
    "cathedral": (
        "scratch/post_cathedral_entrance_to_cathedral_pure.state",
        "Cathedral left lip (CATH-02 pure successor)",
    ),
    "cathedral-entrance": (
        "scratch/post_business_to_cathedral_entrance_pure.state",
        "Cathedral Entrance left lip (CATH-01 pure successor)",
    ),
    "rising-tide": (
        "scratch/post_cathedral_to_rising_tide_pure.state",
        "Rising Tide left entry (CATH-03 pure successor)",
    ),
    "bubble": (
        "scratch/post_rising_tide_to_bubble_pure.state",
        "Bubble Mountain entry (CATH-04 pure source)",
    ),
    "business": (
        "scratch/post_business_continuous.state",
        "Business Center continuous tip",
    ),
    # --- post-BT / charge / spazer detours ---
    "parlor": (
        "scratch/post_torizo_parlor_continuous.state",
        "Post-Bomb-Torizo Parlor at Flyway door (~968,651) — left climb demo",
    ),
    "big-pink": (
        "dev_b1_bigpink_main_controller.state",
        "Big Pink main shaft (~746,1465) post-supers — Charge collect+return",
    ),
    "below-spazer": (
        "scratch/post_below_spazer_with_charge_continuous.state",
        "Below Spazer left entry (~49,395) Charge continuous — often corrupt for human",
    ),
    "below-spazer-no-charge": (
        "scratch/post_below_spazer_for_spazer_pure.state",
        "Below Spazer left entry beams 0x0000 — power-only practice",
    ),
    "post-spazer": (
        "scratch/post_spazer_collect_pure.state",
        "Spazer Room post-collect ~(171,171) beams 0x1004 — return + drop",
    ),
    "post-spazer-return": (
        "scratch/post_spazer_return_pure.state",
        "Below Spazer top handoff ~(380,155) beams 0x1004 — clean top→mid only",
    ),
    "warehouse-spazer": (
        "scratch/post_warehouse_with_spazer_continuous.state",
        "Warehouse 0xA6A1 beams 0x1004 Charge+Spazer — post mainline K2.2",
    ),
    # --- speed / ice / red / double chamber ---
    "speed-hall": (
        "scratch/post_speed_hall_pre_speed_with_spazer.state",
        "Speed Hall 0xACF0 pre-Speed, beams 0x1004 Charge+Spazer (no Speed bit)",
    ),
    "speed": (
        "scratch/post_speed_collected.state",
        "Speed Booster Room post-collect ~(169,123) items 0x3105 — Wave/Ice/Moat human",
    ),
    "ice": (
        "scratch/post_ice_snake_to_ice_pure.state",
        "Ice Beam Room 0xA890 post-collect ~(187,120) beams 0x1007 — K5 return human",
    ),
    "red-bottom": (
        "scratch/post_ice_bat_to_red_pure.state",
        "Red Tower 0xA253 bottom ~(216,2443) items 0x3105 — pure Bat→Red dual pin",
    ),
    "double-chamber": (
        "scratch/post_single_to_double_chamber_continuous_like.state",
        "Double Chamber leave ~(39,139) Spazer cont-like beams 0x1004 — missile free+Wave",
    ),
    "dc-pure": (
        "scratch/post_single_to_double_chamber_pure.state",
        "Double Chamber leave pure predecessor (often beams 0 — not Spazer mainline)",
    ),
    "dc-post-missiles": (
        "scratch/dev_dc_post_missiles.state",
        "Double Chamber past-gate post-missile pin — runway/Super only practice",
    ),
    # --- K6 moat → west ocean → wrecked ship → phantoon ---
    "pre-moat": (
        "scratch/post_kihunter_pre_moat_spark.state",
        "Kihunter 0x948C pre-spark pin — bot: west_ocean_spark chain-ws; or free-record",
    ),
    "moat-end": (
        "scratch/alpha_pb_to_moat_human_end.state",
        "Moat 0x95FF human Alpha-PB end — bot chain-ws (leave+open+clear+spark→WS)",
    ),
    "west-ocean": (
        "scratch/post_moat_west_ocean_spark.state",
        "West Ocean 0x93FE post-Moat spark ~(49,1163) items 0x3105 — optional human WO",
    ),
    "ws-entrance": (
        "scratch/post_west_ocean_ws_spark.state",
        "WS Entrance 0xCA08 post over-ocean/chain-ws ~(57,139) gs=8 — Phantoon ship human",
    ),
    "phantoon": (
        "scratch/full_start_v1_phantoon_mid.state",
        "Phantoon 0xCD13 mid-fight ~2k into fight phant~2200 full HP/ammo — living tape pin",
    ),
    "phantoon-entry": (
        "scratch/full_start_v1_phantoon.state",
        "Phantoon 0xCD13 room enter full HP/ammo (fresh fight from door)",
    ),
    "post-phantoon": (
        "scratch/post_phantoon_defeated.state",
        "Phantoon room 0xCD13 after defeat ~(177,187) boss_ws bit0 — Gravity path",
    ),
    # --- gravity / grapple / maridia / late game ---
    "gravity": (
        "scratch/full_start_v1_gravity.state",
        "Gravity Suit Room 0xCE40 post-collect items 0x3125 — continue leave/return",
    ),
    "post-gravity": (
        "scratch/post_gravity_caterpillar.state",
        "Caterpillar 0xA322 ~(70,1419) Gravity items 0x3125 — Grapple/Maridia human",
    ),
    "grapple": (
        "scratch/full_start_v1_grapple.state",
        "Grapple Beam Room 0xAC2B post-collect items 0x7125 — leave / Crocomire return",
    ),
    "post-grapple": (
        "scratch/post_grapple_beam_human.state",
        "Post-Grapple Tutorial 1 0xAC00 ~(236,121) items 0x7125 — Maridia free-record",
    ),
    "main-street": (
        "scratch/full_start_v1_main_street.state",
        "full_start_v1 Main Street 0xCFC9 ~(394,1979) items 0x7125 — continue Maridia",
    ),
    "plasma-beam": (
        "scratch/full_start_v1_plasma.state",
        "full_start_v1 Plasma Room 0xD2AA ~(395,635) items 0x7325 beams 0x100F — continue post-Plasma",
    ),
    "golden-torizo": (
        "scratch/full_start_v1_golden_torizo.state",
        "full_start_v1 Golden Torizo 0xB283 left door ~(39,139) items 0x7325 beams 0x100F — room enter",
    ),
    "metal-pirates": (
        "scratch/full_start_v1_metal_pirates.state",
        "full_start_v1 Metal Pirates 0xB62B right door ~(725,171) items 0x732F beams 0x100F — Screw clear",
    ),
    "post-space-jump": (
        "scratch/post_space_jump.state",
        "Space Jump Room 0xD9AA ~(85,155) items 0x7325 — next segment after SJ",
    ),
    "post-draygon": (
        "scratch/post_draygon_precious.state",
        "Precious 0xD78F ~(55,651) Draygon+SJ items 0x7325 — Maridia exit / next",
    ),
    "post-space-jump-precious": (
        "scratch/post_space_jump_precious.state",
        "Precious 0xD78F first return after SJ collect (earlier than F5 end)",
    ),
    "post-spring-ball": (
        "scratch/post_spring_ball.state",
        "Spring Ball Room 0xD6D0 ~(379,362) items 0x7327 — after Shaktool",
    ),
    "main-hall": (
        "scratch/post_ln_main_hall.state",
        "LN Main Hall 0xB236 ~(1152,648) items 0x7327 beams 0x100F — Ridley/GT",
    ),
    "ln-elev-save": (
        "scratch/post_ln_elevator_save.state",
        "LN Elevator Save 0xB1BB ~(200,139) items 0x7327 — before Main Hall",
    ),
    "post-screw": (
        "scratch/post_screw_attack.state",
        "Screw Attack 0xB6C1 ~(171,667) items 0x732F — after collect",
    ),
    "post-ridley": (
        "scratch/full_start_v1_ridley.state",
        "full_start_v1 Ridley Tank 0xB698 ~(220,185) items 0x732F Ridley bit — post fight + tank",
    ),
    "post-ridley-farming": (
        "scratch/post_ridley_farming.state",
        "LN Farming 0xB37A ~(50,142) after leaving Ridley — exit path",
    ),
    "post-bosses": (
        "scratch/post_bosses_landing_site.state",
        "Landing Site 0x91F8 ~(1152,1088) items 0x732F all bosses — G4/Tourian",
    ),
}

# alias → canonical name (not path)
_ALIASES: dict[str, str] = {
    "post-morph": "morph",
    "morph-ball": "morph",
    "full-start-morph": "morph",
    "bombs": "bomb",
    "post-bomb": "bomb",
    "post-bombs": "bomb",
    "bomb-torizo": "bomb",
    "full-start-bomb": "bomb",
    "super": "supers",
    "super-missile": "supers",
    "spore-super": "supers",
    "post-supers": "supers",
    "full-start-supers": "supers",
    "hijump": "hj",
    "hi-jump": "hj",
    "full-start-pink": "resume",
    "post-varia": "varia",
    "varia-pickup": "varia",
    "full-start-bubble": "bubble-human",
    "bubble-save-room": "bubble-save",
    "bat": "bat-cave",
    "post-bat": "bat-cave",
    "post-bubble-bat": "bat-cave",
    "full-start-bat": "bat-cave",
    "post-wave": "wave",
    "wave-beam": "wave",
    "wave-collect": "wave",
    "full-start-wave": "wave",
    "alpha-power-bomb": "alpha-pb",
    "post-alpha-pb": "alpha-pb",
    "full-start-alpha-pb": "alpha-pb",
    "post-torizo": "parlor",
    "charge": "big-pink",
    "charge-to-spazer": "big-pink",
    "spazer": "below-spazer",
    "early-spazer": "below-spazer",
    "post-spazer-collect": "post-spazer",
    "spazer-return": "post-spazer-return",
    "warehouse-with-spazer": "warehouse-spazer",
    "pre-speed": "speed-hall",
    "pre-speed-spazer": "speed-hall",
    "post-speed": "speed",
    "speed-collected": "speed",
    "post-ice": "ice",
    "ice-collect": "ice",
    "red-tower": "red-bottom",
    "post-ice-red": "red-bottom",
    "dc": "double-chamber",
    "dc-cont": "double-chamber",
    "kihunter-pre-moat": "pre-moat",
    "pre-moat-spark": "pre-moat",
    "alpha-pb-moat-end": "moat-end",
    "post-moat": "west-ocean",
    "post-moat-spark": "west-ocean",
    "moat-spark": "west-ocean",
    "post-west-ocean": "ws-entrance",
    "post-wo-ws": "ws-entrance",
    "post-ws-spark": "ws-entrance",
    "wrecked-ship": "ws-entrance",
    "phantoon-mid": "phantoon",
    "phantoon-fight": "phantoon",
    "full-start-phantoon": "phantoon",
    "phantoon-defeated": "post-phantoon",
    "post-phant": "post-phantoon",
    "gravity-suit": "gravity",
    "gravity-room": "gravity",
    "full-start-gravity": "gravity",
    "post-gravity-caterpillar": "post-gravity",
    "gravity-caterpillar": "post-gravity",
    "maridia-start": "post-gravity",
    "grapple-beam": "grapple",
    "grapple-room": "grapple",
    "full-start-grapple": "grapple",
    "post-grapple-beam": "post-grapple",
    "post-main-street": "main-street",
    "post-grapple-main-street": "main-street",
    "full-start-main-street": "main-street",
    "maridia": "main-street",
    "plasma": "plasma-beam",
    "post-plasma": "plasma-beam",
    "plasma-room": "plasma-beam",
    "full-start-plasma": "plasma-beam",
    "gt": "golden-torizo",
    "gt-entry": "golden-torizo",
    "golden-torizo-entry": "golden-torizo",
    "full-start-gt": "golden-torizo",
    "metal-pirate": "metal-pirates",
    "pirates": "metal-pirates",
    "mp": "metal-pirates",
    "full-start-metal-pirates": "metal-pirates",
    "space-jump": "post-space-jump",
    "post-sj": "post-space-jump",
    "precious": "post-draygon",
    "post-draygon-precious": "post-draygon",
    "spring-ball": "post-spring-ball",
    "ln-main-hall": "main-hall",
    "post-ln-main-hall": "main-hall",
    "lower-norfair": "main-hall",
    "screw-attack": "post-screw",
    "post-screw-attack": "post-screw",
    "ridley-tank": "post-ridley",
    "post-ridley-tank": "post-ridley",
    "full-start-ridley": "post-ridley",
    "full-start-post-ridley": "post-ridley",
    "landing-site-post-bosses": "post-bosses",
    "post-bosses-landing": "post-bosses",
    "post-ridley-landing": "post-bosses",
}


def _build_start_presets() -> dict[str, tuple[str, str]]:
    """Merge canonical pins and aliases into the public START_PRESETS map."""
    out = dict(_CANONICAL)
    for alias, canon in _ALIASES.items():
        if canon not in _CANONICAL:
            raise KeyError(f"alias {alias} → missing canonical {canon}")
        path, _blurb = _CANONICAL[canon]
        out[alias] = (path, f"Alias of {canon}")
    return out


START_PRESETS: dict[str, tuple[str, str]] = _build_start_presets()


def resolve_start_preset(name: str) -> tuple[str, str] | None:
    """Return ``(rel_path, blurb)`` for a preset name, or None if unknown."""
    return START_PRESETS.get(name)
