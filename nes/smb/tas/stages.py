"""Stage table + control/goal predicates for control-relative TAS adapt.

One :class:`StageSpec` per body leg. Probe / export / search / chain code in
``smb.tas.slice`` is driven from this table — do not add per-level clones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from smb.paths import GAME_DIR, MODELS_DIR
from smb.ram import (
    ADDR_WORLD,
    PLAYER_STATE_DYING,
    WORLD_INDEX_4,
    WORLD_INDEX_8,
    reached_ending,
)
from smb.reactive_12 import is_surface_control

# ---------------------------------------------------------------------------
# Paths / verified FM2 indices (HappyLee #1715M, fceumm control-relative)
# ---------------------------------------------------------------------------

DEFAULT_FM2 = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2"
DEFAULT_HL_1_1 = MODELS_DIR / "smb_1_1_happylee_slice.json"
DEFAULT_HL_1_2 = MODELS_DIR / "smb_1_2_happylee_slice.json"
DEFAULT_HL_4_1 = MODELS_DIR / "smb_4_1_happylee_slice.json"
DEFAULT_HL_4_2 = MODELS_DIR / "smb_4_2_happylee_slice.json"
DEFAULT_HL_8_1 = MODELS_DIR / "smb_8_1_happylee_slice.json"
DEFAULT_HL_8_2 = MODELS_DIR / "smb_8_2_happylee_slice.json"

HL_1_1_SETTLE = 2
HL_1_1_NATURAL_SETTLE = 1

HL_1_2_FM2_START = 2109
HL_1_2_W4_FRAMES = 1657

HL_4_1_FM2_START = 3968
HL_4_1_LEAVE_FRAMES = 2062

HL_4_2_FM2_START = 6207
HL_4_2_W8_FRAMES = 1516

HL_8_1_FM2_START = 7930
HL_8_1_LEAVE_FRAMES = 2881
HL_8_1_CTRL_WAIT = 209  # odd wait; even FM2 starts clear

HL_8_2_FM2_START = 10910
HL_8_2_LEAVE_FRAMES = 2209
HL_8_2_CTRL_WAIT = 165

# Pure continuous FM2 8-3 still phase-blocked; stitchless skills leave verified.
HL_8_3_FM2_START: int | None = None
HL_8_3_LEAVE_FRAMES: int | None = 2374
HL_8_3_SKILLS_LEAVE = MODELS_DIR / "smb_8_3_stitchless_skills_leave.json"

NAT_8_3_FOR_HL_START = 15933
NAT_8_3_TO_8_4_CONTROL = 2227

FX_8_4_FM2_START = 15210
FX_8_4_ENDING_FRAMES = 2661
HL_8_4_FM2_START: int | None = 15034
HL_8_4_ENDING_FRAMES: int | None = 2833

# natural_82 RTA baselines (Level1_1-relative exit-detect) for Δ reports
NAT82_TO_W4 = 3884
NAT82_TO_4_1_LEAVE = 6198
NAT82_TO_8_1_ENTRY = 12628
NAT82_TO_8_2_LEAVE = 15779
NAT82_TO_ENDING = 21559


# ---------------------------------------------------------------------------
# Control gates
# ---------------------------------------------------------------------------


def is_4_1_control(snap: Any) -> bool:
    """Controllable 4-1 start after W4 pipe (timer live, low x)."""
    return (
        int(snap.world) == 3
        and int(snap.level) == 0
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and int(snap.timer) > 0
        and int(snap.player_x) < 200
    )


def is_4_2_control(snap: Any) -> bool:
    """Controllable 4-2 surface start after 4-1 castle load.

    Timer is often **0** on the first controllable frame — do not require
    timer > 0.
    """
    return (
        int(snap.world) == 3
        and int(snap.level) == 1
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 80
    )


def is_8_1_control(snap: Any) -> bool:
    """Controllable 8-1 after W8 pipe."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 0
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and int(snap.timer) > 0
        and int(snap.player_x) < 120
    )


def is_8_2_control(snap: Any) -> bool:
    """Controllable 8-2 start after 8-1 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 1
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 120
    )


def is_8_3_control(snap: Any) -> bool:
    """Controllable 8-3 start after 8-2 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 2
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 120
    )


def is_8_4_control(snap: Any) -> bool:
    """Controllable 8-4 start after 8-3 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 3
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 200
    )


# 32-exit: 1-3 control after the 1-2 flag pipe (not 1-2 UG AreaNumber).
CONTROL_X_MAX = 80


def stage_dash(snap: Any) -> int:
    """0-indexed LevelNumber ($075C). Falls back to AreaNumber for test snaps."""
    if hasattr(snap, "dash_level"):
        try:
            return int(snap.dash_level)
        except (TypeError, ValueError):
            pass
    raw = getattr(snap, "level_number", None)
    if raw is not None:
        return int(raw)
    return int(snap.level)


def is_dash_control(snap: Any, world: int, dash: int) -> bool:
    """Controllable spawn at ``(world, LevelNumber)``. Never AreaNumber.

    Same x/timer/ps shape as the 1-3 / 1-4 gates. 1-2 surface still uses
    ``is_surface_control`` (area pointer + y).
    """
    return (
        int(snap.world) == world
        and stage_dash(snap) == dash
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and int(getattr(snap, "timer", 0) or 0) > 0
        and int(snap.player_x) <= CONTROL_X_MAX
        and not bool(getattr(snap, "dying", False))
    )


def is_1_3_control(snap: Any) -> bool:
    """Controllable 1-3 spawn: LevelNumber 2, live timer, low x.

    Uses ``dash_level`` ($075C), never AreaNumber ($0760). 1-2 underground
    flips AreaNumber to 2 while LevelNumber stays 1.
    """
    return is_dash_control(snap, 0, 2)


def is_1_4_control(snap: Any) -> bool:
    """Controllable 1-4 spawn after the 1-3 flagpole (LevelNumber 3)."""
    return is_dash_control(snap, 0, 3)


def is_2_1_control(snap: Any) -> bool:
    """Controllable 2-1 overworld spawn after the 1-4 axe (world 1, dash 0)."""
    return is_dash_control(snap, 1, 0)


def is_ending_axe(snap: Any) -> bool:
    """8-4 axe / Peach: World 8-4 with ``oper_mode=2``."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and stage_dash(snap) == 3
        and int(snap.oper_mode) == 2
    )


def reached_world_8(ram: Any) -> bool:
    """True when warp-zone pipe delivered Mario to World 8."""
    return int(ram[ADDR_WORLD]) == WORLD_INDEX_8


def snap_fingerprint(snap: Any) -> dict[str, int]:
    """Compact control-gate fingerprint for reports."""
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "area_pointer": int(getattr(snap, "area_pointer", -1) or -1),
        "oper_mode": int(snap.oper_mode),
        "player_state": int(snap.player_state),
        "player_x": int(snap.player_x),
        "player_y": int(snap.player_y),
        "timer": int(snap.timer),
        "lives": int(snap.lives),
    }


def is_dead(snap: Any, start_lives: int) -> bool:
    return int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING


# ---------------------------------------------------------------------------
# Goal kinds for generic probe
# ---------------------------------------------------------------------------


class GoalKind(str, Enum):
    """How a body probe decides success (leave)."""

    WORLD = "world"  # ram/snap world == goal_world
    LEVEL = "level"  # (world, level) == (goal_world, goal_level)
    ENDING = "ending"  # reached_ending on 8-4


@dataclass(frozen=True)
class StageSpec:
    """One control-relative TAS body leg."""

    id: str
    control: Callable[[Any], bool]
    goal: GoalKind
    fm2_start: int
    body_frames: int
    seed_name: str
    start_state: str
    target: str
    goal_world: int | None = None
    goal_level: int | None = None
    # Optional underground / area transition to record on SliceProbe.ug
    track_ug: tuple[int, int] | None = None
    search_min: int = 0
    search_max: int = 0
    search_step: int = 1
    max_play: int = 4000
    predecessor: str = ""
    note: str = (
        "Control-relative body. Do not sanitize L+R. "
        "Re-search if predecessor timing/phase changes."
    )
    route_id: str | None = None
    source: str = "HappyLee warps #1715M FM2"

    @property
    def seed_path(self) -> Path:
        return MODELS_DIR / self.seed_name

    @property
    def resolved_route_id(self) -> str:
        return self.route_id or self.seed_name.replace(".json", "")


def goal_hit(
    kind: GoalKind,
    *,
    snap: Any,
    ram: Any,
    key: tuple[int, int],
    goal_world: int | None,
    goal_level: int | None,
    start_lives: int,
) -> bool:
    """True when the body probe should stop as a successful leave."""
    if kind is GoalKind.WORLD:
        return goal_world is not None and int(snap.world) == goal_world
    if kind is GoalKind.LEVEL:
        return (
            goal_world is not None
            and goal_level is not None
            and key == (goal_world, goal_level)
        )
    if kind is GoalKind.ENDING:
        return reached_ending(ram, start_lives=start_lives)
    return False


# ---------------------------------------------------------------------------
# HappyLee warps stage table
# ---------------------------------------------------------------------------

STAGE_1_2 = StageSpec(
    id="1-2",
    control=is_surface_control,
    goal=GoalKind.WORLD,
    goal_world=WORLD_INDEX_4,
    track_ug=(0, 2),
    fm2_start=HL_1_2_FM2_START,
    body_frames=HL_1_2_W4_FRAMES,
    search_min=2080,
    search_max=2140,
    search_step=1,
    max_play=2200,
    seed_name="smb_1_2_happylee_slice.json",
    start_state="1-2_surface_control_after_happylee_1_1",
    target="world_4_entry",
    predecessor="smb_1_1_happylee_slice Level1_1 settle=2; idle to is_surface_control",
    note=(
        "Control-relative 1-2 W4 warp. Do not sanitize L+R. "
        "Odd FM2 starts after odd ctrl_wait; re-search if 1-1 body changes."
    ),
    source="HappyLee warps #1715M FM2 (natural HL 1-1 predecessor)",
)

STAGE_4_1 = StageSpec(
    id="4-1",
    control=is_4_1_control,
    goal=GoalKind.LEVEL,
    goal_world=3,
    goal_level=1,
    fm2_start=HL_4_1_FM2_START,
    body_frames=HL_4_1_LEAVE_FRAMES,
    search_min=3880,
    search_max=4020,
    search_step=2,
    max_play=2800,
    seed_name="smb_4_1_happylee_slice.json",
    start_state="4-1_control_after_happylee_w4",
    target="4_2_load",
    predecessor=(
        "HL 1-1 + surface + HL 1-2 W4 + idle to is_4_1_control "
        "(even ctrl_wait → even FM2 start)"
    ),
    note="Control-relative 4-1. Do not sanitize L+R. Re-search if W4 predecessor timing changes.",
    source="HappyLee warps #1715M FM2 (HL W4 predecessor)",
)

STAGE_4_2 = StageSpec(
    id="4-2",
    control=is_4_2_control,
    goal=GoalKind.WORLD,
    goal_world=WORLD_INDEX_8,
    track_ug=(3, 2),
    fm2_start=HL_4_2_FM2_START,
    body_frames=HL_4_2_W8_FRAMES,
    search_min=6100,
    search_max=6250,
    search_step=2,
    max_play=4000,
    seed_name="smb_4_2_happylee_slice.json",
    start_state="4-2_control_after_happylee_4_1",
    target="world_8_entry",
    predecessor=(
        "HL 4-1 body + idle to is_4_2_control "
        "(odd ctrl_wait → odd FM2 start; timer often 0 at gate)"
    ),
    note="Control-relative 4-2 → W8 warp. Do not sanitize L+R. Gate does not require timer>0.",
    source="HappyLee warps #1715M FM2 (HL 4-1 predecessor)",
)

STAGE_8_1 = StageSpec(
    id="8-1",
    control=is_8_1_control,
    goal=GoalKind.LEVEL,
    goal_world=WORLD_INDEX_8,
    goal_level=1,
    fm2_start=HL_8_1_FM2_START,
    body_frames=HL_8_1_LEAVE_FRAMES,
    search_min=7900,
    search_max=8000,
    search_step=1,
    max_play=3500,
    seed_name="smb_8_1_happylee_slice.json",
    start_state="8-1_control_after_happylee_w8",
    target="8_2_load",
    predecessor="HL chain to W8 + idle to is_8_1_control (wait≈209 odd; even FM2)",
    note="Control-relative World 8 HL body. Do not sanitize L+R.",
    source="HappyLee warps #1715M FM2 (HL W8 predecessor)",
)

STAGE_8_2 = StageSpec(
    id="8-2",
    control=is_8_2_control,
    goal=GoalKind.LEVEL,
    goal_world=WORLD_INDEX_8,
    goal_level=2,
    fm2_start=HL_8_2_FM2_START,
    body_frames=HL_8_2_LEAVE_FRAMES,
    search_min=10850,
    search_max=11000,
    search_step=1,
    max_play=3500,
    seed_name="smb_8_2_happylee_slice.json",
    start_state="8-2_control_after_happylee_8_1",
    target="8_3_load",
    predecessor="HL 8-1 + idle to is_8_2_control (wait≈165)",
    note="Control-relative World 8 HL body. Do not sanitize L+R.",
    source="HappyLee warps #1715M FM2 (HL W8 predecessor)",
)

STAGE_8_3 = StageSpec(
    id="8-3",
    control=is_8_3_control,
    goal=GoalKind.LEVEL,
    goal_world=WORLD_INDEX_8,
    goal_level=3,
    fm2_start=0,  # open — pure continuous still phase-blocked
    body_frames=HL_8_3_LEAVE_FRAMES or 0,
    search_min=13000,
    search_max=13600,
    search_step=1,
    max_play=3500,
    seed_name="smb_8_3_happylee_slice.json",
    start_state="8-3_control_after_happylee_8_2",
    target="8_4_load",
    predecessor="HL 8-2 + idle to is_8_3_control",
    note="Pure continuous FM2 8-3 phase-blocked on fceumm; skills leave is separate seed.",
)

STAGE_8_4 = StageSpec(
    id="8-4",
    control=is_8_4_control,
    goal=GoalKind.ENDING,
    fm2_start=HL_8_4_FM2_START or 0,
    body_frames=HL_8_4_ENDING_FRAMES or 0,
    search_min=14900,
    search_max=15300,
    search_step=1,
    max_play=6000,
    seed_name="smb_8_4_happylee_slice.json",
    start_state="8-4_control_after_8_3",
    target="ending_axe",
    predecessor="8-3 leave + idle to is_8_4_control",
    note="Prefer flamexx slice after natural 8-3 bridge for hybrid; pure HL open.",
)

STAGES: dict[str, StageSpec] = {
    s.id: s
    for s in (
        STAGE_1_2,
        STAGE_4_1,
        STAGE_4_2,
        STAGE_8_1,
        STAGE_8_2,
        STAGE_8_3,
        STAGE_8_4,
    )
}


def get_stage(stage_id: str) -> StageSpec:
    """Lookup by id (``1-2``, ``4-1``, …). Raises ``KeyError`` if unknown."""
    key = stage_id.strip().lower().replace("_", "-")
    if key not in STAGES:
        raise KeyError(f"unknown stage {stage_id!r}; known: {sorted(STAGES)}")
    return STAGES[key]
