"""Declarative level configuration for side-scroller speedrun optimization.

Follows the same registry pattern as fighters_common/game_configs.py:
define a LevelConfig dataclass, register it with aliases, look it up
via get_level_config().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from retro_harness.ram_state import RAMSchema


@dataclass
class PlatformerRAM:
    """RAM address layout for a platformer game.

    Each field is an (address, type_str) tuple compatible with RAMSchema.
    Set to None for addresses the game doesn't have.
    """

    camera_x: tuple[int, str] | None = None
    camera_y: tuple[int, str] | None = None
    player_x: tuple[int, str] | None = None
    player_y: tuple[int, str] | None = None
    lives: tuple[int, str] | None = None
    level_id: tuple[int, str] | None = None
    timer_frames: tuple[int, str] | None = None
    timer_minutes: tuple[int, str] | None = None
    extras: dict[str, tuple[int, str]] = field(default_factory=dict)

    def to_schema(self) -> RAMSchema:
        """Convert to a RAMSchema for batch reads."""
        addresses: dict[str, tuple[int, str]] = {}
        for name in (
            "camera_x", "camera_y", "player_x", "player_y",
            "lives", "level_id", "timer_frames", "timer_minutes",
        ):
            val = getattr(self, name)
            if val is not None:
                addresses[name] = val
        addresses.update(self.extras)
        return RAMSchema(addresses)


@dataclass
class LevelConfig:
    """Complete configuration for optimizing a single level.

    This is the platformer equivalent of FightingGameConfig. One instance
    per level; multiple levels can share the same PlatformerRAM (all DKC
    levels use the same RAM layout, just different start states and IDs).
    """

    # Identity
    level_id: str                       # unique key, e.g. "dkc_winkys_walkway"
    display_name: str                   # human-readable, e.g. "Winky's Walkway"
    game_name: str                      # stable-retro game ID, e.g. "DonkeyKongCountry-Snes"
    game_dir_name: str                  # folder under retro_rl/, e.g. "donkey_kong_country"

    # Environment setup
    start_state: str                    # .state file name (without extension)
    ram: PlatformerRAM                  # RAM address layout
    target_level_id: int                # expected level_id RAM value at start

    # Progress tracking
    progress_axis: Literal["camera_x", "camera_y", "player_x", "player_y", "composite", "waypoints"] = "camera_x"
    progress_direction: Literal[1, -1] = 1
    backtrack_tolerance: float = 0.0    # for maze levels: regression allowed before stall
    waypoints: list[tuple[float, float]] = field(default_factory=list)

    # Level ID aliases: other level_id values that are part of the same level
    # (e.g. SMB 1-2 underground area has a different internal level_id).
    # These are NOT treated as sub-levels and progress is tracked normally.
    level_id_aliases: list[int] = field(default_factory=list)

    # Death detection
    death_signals: list[str] = field(default_factory=lambda: ["lives_drop", "camera_reset"])
    camera_reset_threshold: float = 100.0

    # Completion detection
    completion_signal: Literal["level_id_change", "ram_flag"] = "level_id_change"
    completion_min_progress: float = 0.0  # min progress before level_id_change counts as completion
    completion_level_ids: list[int] = field(default_factory=list)  # if set, only these level_ids trigger completion
    completion_exclude_ids: list[int] = field(default_factory=list)  # blacklist: these level_ids never count as completion

    # Action space (None = use DEFAULT_PLATFORMER_ACTIONS)
    action_table: list[list[int]] | None = None

    # Fitness weights
    progress_weight: float = 10.0
    death_penalty: float = 1000.0
    completion_bonus: float = 100000.0
    time_bonus_weight: float = 1.0

    # GA defaults (overridable per run)
    max_stall_frames: int = 300
    population_size: int = 50
    num_generations: int = 200

    # Computed values: derive virtual RAM fields from raw reads.
    # e.g. {"player_x": lambda v: v["x_page"]*256 + v["x_offset"]}
    # Applied after every RAM read so evaluator/runner/selftest see them.
    computed_values: dict[str, Callable[[dict[str, int]], int]] = field(default_factory=dict)

    # BK2 button mapping (BK2 hardware order → env logical order)
    # Default: SNES standard reversed mapping
    bk2_to_env: list[int] = field(default_factory=lambda: [11 - i for i in range(12)])

    @property
    def game_dir(self) -> Path:
        """Absolute path to the game directory."""
        return Path(__file__).parent.parent / self.game_dir_name

    @property
    def runs_dir(self) -> Path:
        """Default output directory for optimization runs, organized per-level."""
        return self.game_dir / "optimizer" / "runs" / self.level_id

    @property
    def ram_schema(self) -> RAMSchema:
        """Get the RAMSchema for this level's RAM layout."""
        return self.ram.to_schema()

    def apply_computed(self, values: dict[str, int]) -> dict[str, int]:
        """Apply computed_values transforms to raw RAM values (in-place)."""
        for key, fn in self.computed_values.items():
            values[key] = fn(values)
        return values


# -- Registry ----------------------------------------------------------------

LEVEL_REGISTRY: dict[str, LevelConfig] = {}


def register_level(config: LevelConfig, *aliases: str) -> None:
    """Register a level config with its level_id and optional short aliases."""
    LEVEL_REGISTRY[config.level_id] = config
    for alias in aliases:
        LEVEL_REGISTRY[alias] = config


def get_level_config(level_or_alias: str) -> LevelConfig:
    """Look up a level config by ID or alias."""
    key = level_or_alias.lower() if level_or_alias.lower() in LEVEL_REGISTRY else level_or_alias
    if key not in LEVEL_REGISTRY:
        available = sorted(set(c.level_id for c in LEVEL_REGISTRY.values()))
        raise KeyError(f"Unknown level '{level_or_alias}'. Available: {available}")
    return LEVEL_REGISTRY[key]


def list_levels() -> list[LevelConfig]:
    """Return deduplicated list of all registered levels."""
    seen: set[str] = set()
    result: list[LevelConfig] = []
    for config in LEVEL_REGISTRY.values():
        if config.level_id not in seen:
            seen.add(config.level_id)
            result.append(config)
    return result
