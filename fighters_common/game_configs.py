"""
Per-game configuration for all supported fighting games.

Each config specifies ROM paths, game IDs, RAM addresses, health ranges,
and menu navigation sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fighters_common.fighting_env import MK_FIGHTING_ACTIONS


@dataclass
class FightingGameConfig:
    """Configuration for a single fighting game."""
    game_id: str                    # stable-retro game name (e.g., "StreetFighterIITurbo-Snes")
    display_name: str               # Human-readable name
    game_dir_name: str              # Folder name under retro_rl/
    max_health: int = 176           # Maximum health value in RAM
    health_key: str = "health"      # Key in info dict for P1 health
    enemy_health_key: str = "enemy_health"  # Key in info dict for P2 health
    timer_key: str = "timer"        # Key in info dict for round timer
    round_length_frames: int = 5400 # ~90 seconds at 60fps
    rounds_to_win: int = 2          # Rounds needed to win a match
    # For games where data.json can't map high WRAM addresses (>= 0x2000),
    # provide direct RAM reads: {"info_key": ram_offset}
    ram_overrides: Dict[str, int] = field(default_factory=dict)
    # Menu navigation: list of (button_index, hold_frames) pairs to get from
    # title screen to a fight. -1 = wait frames with no input.
    menu_sequence: List[tuple] = field(default_factory=list)
    # Frames to wait after menu sequence before saving state
    menu_settle_frames: int = 120
    # Per-game action space (None = use default SF2-style FIGHTING_ACTIONS)
    actions: Optional[List] = None


# Button indices
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)


SF2_TURBO = FightingGameConfig(
    game_id="StreetFighterIITurbo-Snes",
    display_name="Street Fighter II Turbo - Hyper Fighting",
    game_dir_name="street_fighter_ii",
    max_health=176,
    menu_sequence=[
        # Title screen -> press Start
        (-1, 120),          # Wait for title
        (_START, 10),       # Press start
        (-1, 60),           # Wait for mode select
        (_START, 10),       # Select 1P mode
        (-1, 60),           # Wait for character select
        # Ryu is default, just confirm
        (_START, 10),       # Confirm character
        (-1, 30),           # Wait for difficulty
        (_START, 10),       # Confirm difficulty
        (-1, 180),          # Wait for fight to load
    ],
    menu_settle_frames=60,
)

SUPER_SF2 = FightingGameConfig(
    game_id="SuperStreetFighterII-Snes",
    display_name="Super Street Fighter II - The New Challengers",
    game_dir_name="super_street_fighter_ii",
    max_health=176,
    menu_sequence=[
        (-1, 120),
        (_START, 10),
        (-1, 60),
        (_START, 10),       # 1P mode
        (-1, 60),
        (_START, 10),       # Confirm character (Ryu default)
        (-1, 30),
        (_START, 10),       # Confirm difficulty
        (-1, 180),
    ],
    menu_settle_frames=60,
)

MK1 = FightingGameConfig(
    game_id="MortalKombat-Snes",
    display_name="Mortal Kombat",
    game_dir_name="mortal_kombat",
    max_health=161,         # 0xA1
    actions=MK_FIGHTING_ACTIONS,
    menu_sequence=[
        # MK1 SNES boot: Acclaim/Sculptured logos then title screen
        (-1, 1800),         # Wait through boot logos (~30 sec)
        (_START, 15),       # Title screen -> Tournament mode
        (-1, 120),          # Wait for char select
        # Char select: START alone doesn't confirm. Need button mashing sequence.
        (_START, 15), (-1, 120),
        (_B, 20), (-1, 60),
        (_A, 20), (-1, 60),
        (_X, 20), (-1, 60),
        (_L, 20), (-1, 60),
        (_R, 20), (-1, 60),
        (_Y, 20), (-1, 60),
        # VS screen / fight loading
        (_START, 15), (-1, 120),
    ],
    menu_settle_frames=60,
)

MK2 = FightingGameConfig(
    game_id="MortalKombatII-Snes",
    display_name="Mortal Kombat II",
    game_dir_name="mortal_kombat_ii",
    max_health=161,         # 0xA1
    actions=MK_FIGHTING_ACTIONS,
    # Health in high WRAM (>= 0x2000) — needs DirectRAMReader
    # P1=WRAM 0x2EFC (get_ram 0x4EFD), P2=WRAM 0x30AA (get_ram 0x50AB)
    # Gap = 0x1AE (430 bytes) — player structs are far apart
    ram_overrides={
        "health": 0x4EFD,          # P1 health (WRAM 0x2EFC + 0x2001 offset)
        "enemy_health": 0x50AB,    # P2 health (WRAM 0x30AA + 0x2001 offset)
    },
    menu_sequence=[
        # MK2 SNES has very long boot: Sculptured Software, Acclaim, MK2 intro logos
        (-1, 900),          # Wait through logos (~15 seconds)
        (_START, 15),       # Skip remaining logos
        (-1, 120),
        (_START, 15),
        (-1, 120),
        (_START, 15),
        (-1, 120),
        (_START, 15),
        (-1, 120),
        (_START, 15),       # Should be at title screen
        (-1, 120),
        (_START, 20),       # Title -> press START
        (-1, 120),
        (_START, 20),       # Mode select (if any)
        (-1, 240),
        (_Y, 20),           # Character select -> HP to pick
        (-1, 120),
        (_Y, 20),           # Difficulty select
        (-1, 60),
        # Battle plan + story + bio + VS screens need START mashing
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        (_START, 8), (-1, 15),
        # Wait for fight to load
        (-1, 1800),         # 30 seconds for story/bio/VS/fight intro
    ],
    menu_settle_frames=300,
)


GAME_REGISTRY: Dict[str, FightingGameConfig] = {
    "StreetFighterIITurbo-Snes": SF2_TURBO,
    "SuperStreetFighterII-Snes": SUPER_SF2,
    "MortalKombat-Snes": MK1,
    "MortalKombatII-Snes": MK2,
    # Short aliases
    "sf2": SF2_TURBO,
    "ssf2": SUPER_SF2,
    "mk1": MK1,
    "mk2": MK2,
}


def get_game_config(game_or_alias: str) -> FightingGameConfig:
    """Look up a game config by ID or alias."""
    key = game_or_alias.lower() if game_or_alias.lower() in GAME_REGISTRY else game_or_alias
    if key not in GAME_REGISTRY:
        available = [k for k in GAME_REGISTRY if not k.islower()]
        raise KeyError(f"Unknown game '{game_or_alias}'. Available: {available}")
    return GAME_REGISTRY[key]
