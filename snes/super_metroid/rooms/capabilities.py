"""Canonical Super Metroid capability spelling tables.

Source-format lock/item names map into tokens accepted by
:func:`retro_harness.adventure.normalize_capability`. Do not add ad-hoc aliases
outside this module or ``retro_harness.adventure.graph``.
"""

from __future__ import annotations

from retro_harness.adventure.graph import normalize_capability

# Re-export the single shared spelling function under the historical name.
normalize_ability = normalize_capability

DOOR_REQUIREMENTS = {
    "red": "missiles",
    "green": "super_missiles",
    "yellow": "power_bombs",
}

LOCK_REQUIREMENTS = {
    "Bombs": "bombs",
    "Morph": "morph_ball",
    "Missile": "missiles",
    "f_DefeatedBombTorizo": "bomb_torizo_defeated",
    "f_DefeatedBotwoon": "botwoon_defeated",
    "f_DefeatedCeresRidley": "ceres_ridley_cleared",
    "f_DefeatedCrocomire": "crocomire_defeated",
    "f_DefeatedDraygon": "draygon_defeated",
    "f_DefeatedGoldenTorizo": "golden_torizo_defeated",
    "f_DefeatedKraid": "kraid_defeated",
    "f_DefeatedMotherBrain": "mother_brain_defeated",
    "f_DefeatedPhantoon": "phantoon_defeated",
    "f_DefeatedRidley": "ridley_defeated",
    "f_DefeatedSporeSpawn": "spore_spawn_defeated",
    "f_ZebesSetAblaze": "mother_brain_defeated",
}

ITEM_CAPABILITIES = {
    "bomb": "bombs",
    "charge beam": "charge_beam",
    "grapple beam": "grapple_beam",
    "gravity suit": "gravity_suit",
    "hi-jump boots": "hi_jump",
    "ice beam": "ice_beam",
    "missile": "missiles",
    "morph ball": "morph_ball",
    "plasma beam": "plasma_beam",
    "power bomb": "power_bombs",
    "screw attack": "screw_attack",
    "space jump": "space_jump",
    "spazer": "spazer",
    "speed booster": "speed_booster",
    "spring ball": "spring_ball",
    "super missile": "super_missiles",
    "varia suit": "varia_suit",
    "wave beam": "wave_beam",
    "x-ray scope": "xray_scope",
}

# Historical private names used by room_graph internals.
_DOOR_REQUIREMENTS = DOOR_REQUIREMENTS
_LOCK_REQUIREMENTS = LOCK_REQUIREMENTS
_ITEM_CAPABILITIES = ITEM_CAPABILITIES
