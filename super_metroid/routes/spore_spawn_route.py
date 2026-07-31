"""Editor-backed post-Bomb-Torizo route specification."""

from __future__ import annotations

from adventure_common.graph import RouteLeg, RoutePatch
from super_metroid.map_planning import sm_route_patch

POST_TORIZO_CAPABILITIES = frozenset(
    {
        "morph_ball",
        "bombs",
        "missiles",
    }
)

POST_TORIZO_ROUTE_PATCHES = (
    sm_route_patch(
        0x9AD9,
        0x9CB3,
        "Right",
        frozenset({"missiles"}),
        door_cap_color="red",
        support=(
            "Missing from SMEDIT nav_graph; supported by exported inverse "
            "0x9CB3->0x9AD9 and the editor practice-route door record."
        ),
    ),
)

POST_TORIZO_TO_SPORE_SPAWN = (
    RouteLeg(
        "parlor_to_terminator",
        0x92FD,
        0x990D,
        frozenset({"bombs"}),
        goal="Bomb through upper Parlor and exit left.",
    ),
    RouteLeg(
        "terminator",
        0x990D,
        0x99BD,
        acquires=frozenset({"terminator_energy_tank"}),
        goal="Collect the Energy Tank naturally and continue left.",
    ),
    RouteLeg(
        "green_pirates",
        0x99BD,
        0x9969,
        goal="Descend Green Pirates Shaft into Lower Mushrooms.",
    ),
    RouteLeg(
        "lower_mushrooms",
        0x9969,
        0x9938,
        goal="Cross Lower Mushrooms to the Green Brinstar elevator.",
    ),
    RouteLeg(
        "green_elevator",
        0x9938,
        0x9AD9,
        goal="Ride down to Green Brinstar Main Shaft.",
    ),
    RouteLeg(
        "dachora_entry",
        0x9AD9,
        0x9CB3,
        frozenset({"missiles"}),
        goal="Open the lower-right red door into Dachora Room.",
    ),
    RouteLeg(
        "dachora",
        0x9CB3,
        0x9D19,
        goal="Traverse Dachora Room into Big Pink.",
    ),
    RouteLeg(
        "big_pink",
        0x9D19,
        0x9D9C,
        frozenset({"missiles"}),
        goal="Open the right red door toward Spore Spawn.",
    ),
    RouteLeg(
        "spore_kihunters",
        0x9D9C,
        0x9DC7,
        goal="Clear the Kihunters to open the forward gray boss-room lock.",
        constraints=(
            "Kill the Kihunters to clear the two gray locks. The forward "
            "Kihunter-to-boss edge has no ammo gate in the editor export; "
            "the green Super gate is on the reverse boss-to-Kihunter edge.",
        ),
    ),
    RouteLeg(
        "spore_spawn",
        0x9DC7,
        0x9B5B,
        frozenset({"missiles"}),
        frozenset({"spore_spawn_defeated"}),
        goal=(
            "Defeat Spore Spawn with naturally unlocked Missiles and reach "
            "the first playable post-boss room."
        ),
    ),
)

# Keep RoutePatch in the module namespace for type checkers / re-exports.
__all__ = [
    "POST_TORIZO_CAPABILITIES",
    "POST_TORIZO_ROUTE_PATCHES",
    "POST_TORIZO_TO_SPORE_SPAWN",
    "RouteLeg",
    "RoutePatch",
]
