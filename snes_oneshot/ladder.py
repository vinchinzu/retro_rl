"""Ladder status tracker for the top-10 oneshot SNES games."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class LadderStatus(Enum):
    """Per-game completion status on the oneshot ladder."""

    NOT_STARTED = auto()
    SCAFFOLDED = auto()
    BOOT_WORKS = auto()
    INPUT_BLOCKED = auto()
    SEGMENTS = auto()
    ENDING_SEGMENTED = auto()
    FULL_GAME = auto()


@dataclass(frozen=True)
class LadderEntry:
    """One game on the recommended oneshot ladder."""

    rank: int
    slug: str
    title: str
    rom_zip: str
    integration: str
    tier: int
    status: LadderStatus = LadderStatus.NOT_STARTED
    video: str | None = None


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_ROM_DIR = REPO_ROOT / "roms" / "Super Nintendo"

LADDER: tuple[LadderEntry, ...] = (
    LadderEntry(
        1,
        "great_waldo_search",
        "The Great Waldo Search",
        "Great Waldo Search, The.zip",
        "GreatWaldoSearch-Snes",
        tier=0,
        status=LadderStatus.ENDING_SEGMENTED,
    ),
    LadderEntry(
        2,
        "final_fight",
        "Final Fight",
        "Final Fight.zip",
        "FinalFight-Snes",
        tier=1,
        status=LadderStatus.SEGMENTS,
    ),
    LadderEntry(
        3,
        "tmnt_iv",
        "TMNT IV: Turtles in Time",
        "Teenage Mutant Ninja Turtles IV - Turtles in Time.zip",
        "TMNTIV-Snes",
        tier=1,
        status=LadderStatus.ENDING_SEGMENTED,
        video="tmnt_iv/recordings/tmnt_iv_segmented_completion_showcase.mp4",
    ),
    LadderEntry(
        4,
        "super_double_dragon",
        "Super Double Dragon",
        "Super Double Dragon.zip",
        "SuperDoubleDragon-Snes",
        tier=1,
        status=LadderStatus.SEGMENTS,
    ),
    LadderEntry(
        5,
        "rival_turf",
        "Rival Turf!",
        "Rival Turf!.zip",
        "RivalTurf-Snes",
        tier=1,
        status=LadderStatus.BOOT_WORKS,
    ),
    LadderEntry(
        6,
        "f_zero",
        "F-Zero",
        "F-Zero.zip",
        "FZero-Snes",
        tier=2,
        status=LadderStatus.BOOT_WORKS,
    ),
    LadderEntry(
        7,
        "magical_quest",
        "The Magical Quest Starring Mickey Mouse",
        "Magical Quest starring Mickey Mouse, The.zip",
        "MagicalQuest-Snes",
        tier=3,
        status=LadderStatus.BOOT_WORKS,
    ),
    LadderEntry(
        8,
        "pilotwings",
        "Pilotwings",
        "Pilotwings.zip",
        "Pilotwings-Snes",
        tier=2,
        status=LadderStatus.BOOT_WORKS,
    ),
    LadderEntry(
        9,
        "battle_clash",
        "Battle Clash",
        "Battle Clash.zip",
        "BattleClash-Snes",
        tier=2,
        status=LadderStatus.INPUT_BLOCKED,
    ),
    LadderEntry(
        10,
        "joe_and_mac",
        "Joe & Mac",
        "Joe & Mac - Caveman Ninjas.zip",
        "JoeAndMac-Snes",
        tier=3,
        status=LadderStatus.BOOT_WORKS,
    ),
)


def entry_for(slug: str) -> LadderEntry:
    """Look up a ladder entry by game directory slug."""
    for entry in LADDER:
        if entry.slug == slug:
            return entry
    raise KeyError(f"Unknown ladder slug: {slug}")


def shared_rom_zip(entry: LadderEntry) -> Path:
    """Absolute path to the shared ROM zip for a ladder entry."""
    return SHARED_ROM_DIR / entry.rom_zip
