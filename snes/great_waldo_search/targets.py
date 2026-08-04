"""Hardcoded search targets (Scene1–5)."""

from __future__ import annotations

from enum import Enum, auto

from retro_harness.cursor import CursorTarget


class TargetKind(Enum):
    """Semantic role of a click target."""

    SCROLL = auto()
    WALDO = auto()
    CLOCK = auto()
    POINTS_150 = auto()
    UNKNOWN = auto()


class TargetStatus(Enum):
    """How strongly a coordinate has been validated in-emulator."""

    CONFIRMED = auto()
    ASSIST_LANDING = auto()
    VISUAL_GUESS = auto()
    UNCONFIRMED = auto()


# RAM / policy constants (shared across scenes)
CURSOR_X_ADDR = 0x0215
CURSOR_Y_ADDR = 0x0217
SCORE_LO_ADDR = 0x0047
SCORE_HI_ADDR = 0x0048
# Set to 2 after the confirmed +1000 scroll find.
# Does not reliably change again on the Waldo find.
# Scene3/4 start often reads 134 here until the first +1000.
FOUND_FLAG_ADDR = 0x01BD

CONFIRMED_FIND_POINTS = 1000
WALDO_POINTS = 1500
SCENE1_CLEAR_SCORE = 2500  # scroll + Waldo (bonus may push higher)
# Scene2 carries Scene1 score (~2625); scroll + Waldo → ≥5125.
SCENE2_SCROLL_SCORE = 3625
SCENE2_CLEAR_SCORE = 5125
# Scene3 carries Scene2 score (~5450); scroll + Waldo → ≥7850.
SCENE3_SCROLL_SCORE = 6450
SCENE3_CLEAR_SCORE = 7850
# Scene4 carries Scene3 score (~7950); scroll + Waldo → ≥10450.
SCENE4_SCROLL_SCORE = 8950
SCENE4_CLEAR_SCORE = 10450
# Scene5 carries Scene4 score (~10650); scroll + Waldo → ≥15150.
# First find often +3000 (~13650); Waldo bonus → ~18575–18725 ending.
SCENE5_SCROLL_SCORE = 11650
SCENE5_CLEAR_SCORE = 15150

# After Scene1_AfterFind1000: pan RIGHT+Y for this many frames, then click Waldo.
WALDO_PAN_RIGHT_FRAMES = 80
# After Scene2_AfterFind1000: hold P2-A this many frames, then click Waldo.
SCENE2_WALDO_P2A_FRAMES = 500
# Scene3 (favorable RNG): P2-A frames for scroll / Waldo assists.
SCENE3_SCROLL_P2A_FRAMES = 300
SCENE3_WALDO_P2A_FRAMES = 200
# Scene4 (favorable RNG / idle~5f advance): P2-A for scroll + Waldo pan.
SCENE4_SCROLL_P2A_FRAMES = 500
SCENE4_WALDO_P2A_FRAMES = 500
# Scene5 (favorable RNG / idle~5f advance): P2-A for scroll + Waldo.
SCENE5_SCROLL_P2A_FRAMES = 300
SCENE5_WALDO_P2A_FRAMES = 500


SCENE1_TARGETS: tuple[CursorTarget, ...] = (
    CursorTarget(x=32, y=100, deadzone=2, label="p2a_primary_1000"),
    # After pan RIGHT 80 from AfterFind1000 (not the raw P2-A secondary land).
    CursorTarget(x=36, y=28, deadzone=4, label="waldo_pan_right80"),
    # P2-A re-seek after scroll; not a working Waldo click without pan.
    CursorTarget(x=206, y=100, deadzone=4, label="p2a_secondary_landing"),
    CursorTarget(x=98, y=75, deadzone=4, label="visual_carpet_waldo"),
    CursorTarget(x=205, y=125, deadzone=4, label="visual_roof_waldo"),
    CursorTarget(x=88, y=108, deadzone=4, label="visual_dome_scroll"),
    CursorTarget(x=32, y=80, deadzone=3, label="points_150_near_scroll"),
)


SCENE1_TARGET_META: dict[str, tuple[TargetKind, TargetStatus]] = {
    "p2a_primary_1000": (TargetKind.SCROLL, TargetStatus.CONFIRMED),
    "waldo_pan_right80": (TargetKind.WALDO, TargetStatus.CONFIRMED),
    "p2a_secondary_landing": (TargetKind.UNKNOWN, TargetStatus.ASSIST_LANDING),
    "visual_carpet_waldo": (TargetKind.WALDO, TargetStatus.VISUAL_GUESS),
    "visual_roof_waldo": (TargetKind.WALDO, TargetStatus.VISUAL_GUESS),
    "visual_dome_scroll": (TargetKind.SCROLL, TargetStatus.VISUAL_GUESS),
    "points_150_near_scroll": (TargetKind.POINTS_150, TargetStatus.CONFIRMED),
}


SCENE2_TARGETS: tuple[CursorTarget, ...] = (
    CursorTarget(x=224, y=100, deadzone=4, label="scene2_scroll_right"),
    # After P2-A ≥500f from AfterFind1000 (assist pans camera left).
    CursorTarget(x=32, y=120, deadzone=4, label="scene2_waldo_p2a500"),
    CursorTarget(x=206, y=100, deadzone=4, label="scene2_p2a_pre_scroll"),
)


SCENE2_TARGET_META: dict[str, tuple[TargetKind, TargetStatus]] = {
    "scene2_scroll_right": (TargetKind.SCROLL, TargetStatus.CONFIRMED),
    "scene2_waldo_p2a500": (TargetKind.WALDO, TargetStatus.CONFIRMED),
    "scene2_p2a_pre_scroll": (TargetKind.UNKNOWN, TargetStatus.ASSIST_LANDING),
}


SCENE3_TARGETS: tuple[CursorTarget, ...] = (
    # Favorable layout (Scene2_Cleared idle~5f then advance): P2-A → land.
    CursorTarget(x=160, y=100, deadzone=4, label="scene3_scroll_p2a300"),
    # Continuous favorable layout: click 196 (198 misses).
    CursorTarget(x=196, y=100, deadzone=4, label="scene3_waldo_p2a200"),
)


SCENE3_TARGET_META: dict[str, tuple[TargetKind, TargetStatus]] = {
    "scene3_scroll_p2a300": (TargetKind.SCROLL, TargetStatus.CONFIRMED),
    "scene3_waldo_p2a200": (TargetKind.WALDO, TargetStatus.CONFIRMED),
}


SCENE4_TARGETS: tuple[CursorTarget, ...] = (
    # Favorable layout (Scene3_Cleared idle~5f then advance): P2-A → land.
    CursorTarget(x=34, y=100, deadzone=4, label="scene4_scroll_p2a500"),
    # AfterFind1000 → P2-A~500f pans; click below assist land y=100.
    CursorTarget(x=196, y=140, deadzone=4, label="scene4_waldo_p2a500"),
)


SCENE4_TARGET_META: dict[str, tuple[TargetKind, TargetStatus]] = {
    "scene4_scroll_p2a500": (TargetKind.SCROLL, TargetStatus.CONFIRMED),
    "scene4_waldo_p2a500": (TargetKind.WALDO, TargetStatus.CONFIRMED),
}


SCENE5_TARGETS: tuple[CursorTarget, ...] = (
    # Favorable layout (Scene4_Cleared idle~5f then advance): P2-A → land.
    CursorTarget(x=32, y=100, deadzone=4, label="scene5_scroll_p2a300"),
    # AfterFind → P2-A~500f; click above assist land y=100.
    CursorTarget(x=180, y=60, deadzone=4, label="scene5_waldo_p2a500"),
)


SCENE5_TARGET_META: dict[str, tuple[TargetKind, TargetStatus]] = {
    "scene5_scroll_p2a300": (TargetKind.SCROLL, TargetStatus.CONFIRMED),
    "scene5_waldo_p2a500": (TargetKind.WALDO, TargetStatus.CONFIRMED),
}


def score_u16(lo: int, hi: int) -> int:
    """Decode little-endian score bytes."""
    return int(lo) + (int(hi) << 8)


def confirmed_targets() -> list[CursorTarget]:
    """Targets with in-emulator score confirmation (Scene1–5)."""
    out: list[CursorTarget] = []
    tables: tuple[
        tuple[
            tuple[CursorTarget, ...],
            dict[str, tuple[TargetKind, TargetStatus]],
        ],
        ...,
    ] = (
        (SCENE1_TARGETS, SCENE1_TARGET_META),
        (SCENE2_TARGETS, SCENE2_TARGET_META),
        (SCENE3_TARGETS, SCENE3_TARGET_META),
        (SCENE4_TARGETS, SCENE4_TARGET_META),
        (SCENE5_TARGETS, SCENE5_TARGET_META),
    )
    for targets, meta in tables:
        for target in targets:
            kind_status = meta.get(target.label)
            if kind_status is None:
                continue
            _kind, status = kind_status
            if status is TargetStatus.CONFIRMED:
                out.append(target)
    return out
