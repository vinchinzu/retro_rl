"""Menu navigation scripts for Hal's Hole in One Golf.

Verified USA boot flow (stable-retro / snes9x):
  Title (mode select) → B → Players → B → Difficulty → START → Name →
  START → Clubs → (DOWN/RIGHT toward OK) → START → hole intro → Command
  (SHOT / GREEN / HOLE).

VS HAL (two-column title menu: RIGHT from Stroke Play):
  Title → RIGHT → B → Difficulty → START → Name → START → Clubs → …
  Players select is skipped.

Confirm notes:
  Menus often accept B; club select shows ``A,B-Enter`` / ``X,Y-Cancel``.
  Name entry accepts START to finish with the current/empty name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from retro_harness.protocol import (
    ActionResult,
    TaskResult,
    TaskStatus,
    WorldState,
)

from hals_golf.core.actions import named_script


class PlayMode(Enum):
    """Title-menu game modes the bot can bootstrap into."""

    STROKE_PLAY = auto()
    VS_HAL = auto()


class ClubSet(Enum):
    """In-round club calibration set."""

    STANDARD = auto()
    METAL = auto()


class Difficulty(Enum):
    """Title-menu difficulty. Amateur is the verified Bronze path.

    On the difficulty screen Amateur is the top entry (START-only). Pro sits
    one row below, so its bootstrap taps DOWN once before START. TOURNAMENT is
    a reserved stub for the harder harvest tracks.
    """

    AMATEUR = auto()
    PRO = auto()
    TOURNAMENT = auto()


def _difficulty_select(difficulty: Difficulty) -> tuple[tuple[str, int], ...]:
    """Difficulty-screen taps: Amateur confirms directly; Pro/Tournament drop."""
    if difficulty is Difficulty.AMATEUR:
        return (("START", 4),)
    if difficulty is Difficulty.PRO:
        downs = 1
    elif difficulty is Difficulty.TOURNAMENT:
        downs = 2
    else:
        raise ValueError(f"unsupported difficulty: {difficulty!r}")
    return (
        *sum([(("DOWN", 2), ("IDLE", 10)) for _ in range(downs)], ()),
        ("START", 4),
    )


def _club_ok_nav() -> tuple[tuple[str, int], ...]:
    downs = sum([(("DOWN", 2), ("IDLE", 5)) for _ in range(30)], ())
    rights = sum([(("RIGHT", 2), ("IDLE", 5)) for _ in range(10)], ())
    return downs + rights


def _name_taps(button: str, count: int) -> tuple[tuple[str, int], ...]:
    """Move one name-editor cell per tap without triggering key repeat."""
    return sum([((button, 2), ("IDLE", 8)) for _ in range(count)], ())


def _metal_play_name_nav() -> tuple[tuple[str, int], ...]:
    """Enter the built-in ``METAL PLAY`` metal-club password.

    The editor starts on uppercase ``A``. Its third row contains cursor-left,
    cursor-right, and OK after eight punctuation keys. Cursor-right advances
    over an untouched name cell, which supplies the password's space.
    """
    enter = (("B", 3), ("IDLE", 8))
    return (
        *_name_taps("RIGHT", 12),  # A -> M
        *enter,
        *_name_taps("LEFT", 8),  # M -> E
        *enter,
        *_name_taps("RIGHT", 15),  # E -> T
        *enter,
        *_name_taps("LEFT", 19),  # T -> A
        *enter,
        *_name_taps("RIGHT", 11),  # A -> L
        *enter,
        *_name_taps("LEFT", 11),  # L -> A
        *_name_taps("DOWN", 2),  # A -> a -> punctuation row
        *_name_taps("RIGHT", 9),  # punctuation start -> cursor-right
        *enter,  # leave the sixth name cell blank
        *_name_taps("LEFT", 9),
        *_name_taps("UP", 2),  # back to uppercase A
        *_name_taps("RIGHT", 15),  # A -> P
        *enter,
        *_name_taps("LEFT", 4),  # P -> L
        *enter,
        *_name_taps("LEFT", 11),  # L -> A
        *enter,
        *_name_taps("RIGHT", 24),  # A -> Y
        *enter,
        # With all ten cells filled, START resets the keyboard cursor to A;
        # directional input otherwise remains trapped on the completed name.
        ("START", 4),
        ("IDLE", 200),
        *_name_taps("DOWN", 2),
        *_name_taps("RIGHT", 10),  # punctuation start -> OK
        ("B", 3),
    )


def title_to_stroke_play_frames(
    *,
    club_set: ClubSet = ClubSet.STANDARD,
    difficulty: Difficulty = Difficulty.AMATEUR,
) -> list[np.ndarray]:
    """Cold-boot from Title into hole-1 command menu for ``difficulty``."""
    use_metal = club_set is ClubSet.METAL
    return named_script(
        [
            ("IDLE", 60),  # Title state resumes before the mode box is ready
            ("B", 3),
            ("IDLE", 120),  # Mode → Players
            ("B", 3),
            ("IDLE", 120),  # Players → Difficulty
            *_difficulty_select(difficulty),
            ("IDLE", 180),  # Name editor animates in slowly
            *(_metal_play_name_nav() if use_metal else (("START", 4),)),
            (
                "IDLE",
                320 if use_metal else 160,
            ),  # password OK animates more slowly than an empty-name START
            *_club_ok_nav(),
            ("START", 4),
            ("IDLE", 516),  # course flyover reaches its skippable point
            ("B", 3),       # skip Hole 1 flyover
            ("IDLE", 200),  # settle at SHOT/GREEN/HOLE command menu
        ]
    )


def title_to_stroke_play_amateur_frames(
    *,
    club_set: ClubSet = ClubSet.STANDARD,
) -> list[np.ndarray]:
    """Backward-compatible Amateur stroke-play bootstrap."""
    return title_to_stroke_play_frames(
        club_set=club_set,
        difficulty=Difficulty.AMATEUR,
    )


def title_to_vs_hal_frames(
    *,
    club_set: ClubSet = ClubSet.STANDARD,
    difficulty: Difficulty = Difficulty.AMATEUR,
) -> list[np.ndarray]:
    """Title → VS HAL into hole-1 command menu for ``difficulty``.

    The mode box is two columns; VS HAL sits right of Stroke Play.
    """
    use_metal = club_set is ClubSet.METAL
    return named_script(
        [
            ("IDLE", 60),
            ("RIGHT", 2),
            ("IDLE", 10),  # Stroke Play → VS HAL
            ("B", 3),
            ("IDLE", 120),  # → Difficulty (Players skipped)
            *_difficulty_select(difficulty),
            ("IDLE", 180),  # → Name
            *(_metal_play_name_nav() if use_metal else (("START", 4),)),
            (
                "IDLE",
                320 if use_metal else 160,
            ),  # password OK animates more slowly than an empty-name START
            *_club_ok_nav(),
            ("START", 4),
            ("IDLE", 516),
            ("B", 3),
            ("IDLE", 200),
        ]
    )


def title_to_vs_hal_amateur_frames(
    *,
    club_set: ClubSet = ClubSet.STANDARD,
) -> list[np.ndarray]:
    """Backward-compatible Amateur VS HAL bootstrap."""
    return title_to_vs_hal_frames(
        club_set=club_set,
        difficulty=Difficulty.AMATEUR,
    )


def cold_boot_from_none_frames(
    *,
    club_set: ClubSet = ClubSet.STANDARD,
    difficulty: Difficulty = Difficulty.AMATEUR,
) -> list[np.ndarray]:
    """Boot from power-on through Title, then the Title→Hole1 script."""
    return named_script([("IDLE", 550), ("START", 4), ("IDLE", 100)]) + (
        title_to_stroke_play_frames(
            club_set=club_set,
            difficulty=difficulty,
        )
    )


def dismiss_scorecard_frames() -> list[np.ndarray]:
    """Close scorecard / stats overlays."""
    return named_script(
        [
            ("X", 3),
            ("IDLE", 20),
            ("A", 3),
            ("IDLE", 20),
            ("B", 3),
            ("IDLE", 30),
        ]
    )


def bootstrap_frames_for_mode(
    mode: PlayMode,
    *,
    club_set: ClubSet = ClubSet.STANDARD,
    difficulty: Difficulty = Difficulty.AMATEUR,
) -> list[np.ndarray]:
    """Return the Title→Hole1 frame script for ``mode`` / ``difficulty``."""
    if mode is PlayMode.VS_HAL:
        return title_to_vs_hal_frames(
            club_set=club_set,
            difficulty=difficulty,
        )
    return title_to_stroke_play_frames(
        club_set=club_set,
        difficulty=difficulty,
    )


@dataclass
class MenuBootstrapTask:
    """Drive title (or cold boot) into an in-round hole-1 command menu."""

    name: str = "menu_bootstrap"
    play_mode: PlayMode = PlayMode.STROKE_PLAY
    club_set: ClubSet = ClubSet.STANDARD
    difficulty: Difficulty = Difficulty.AMATEUR
    from_cold_boot: bool = False
    frames: list[np.ndarray] = field(default_factory=list)
    _index: int = 0

    def __post_init__(self) -> None:
        if not self.frames:
            if self.from_cold_boot:
                self.frames = cold_boot_from_none_frames(
                    club_set=self.club_set,
                    difficulty=self.difficulty,
                )
            else:
                self.frames = bootstrap_frames_for_mode(
                    self.play_mode,
                    club_set=self.club_set,
                    difficulty=self.difficulty,
                )

    def reset(self, world: WorldState) -> None:
        del world
        self._index = 0

    def can_start(self, world: WorldState) -> bool:
        del world
        return True

    def step(self, world: WorldState) -> TaskResult:
        del world
        if self._index >= len(self.frames):
            return TaskResult(status=TaskStatus.SUCCESS)
        action = self.frames[self._index]
        self._index += 1
        done = self._index >= len(self.frames)
        return TaskResult(
            status=TaskStatus.SUCCESS if done else TaskStatus.RUNNING,
            action=ActionResult(action=action, reason="menu_bootstrap"),
            meta={"index": self._index, "total": len(self.frames)},
        )
