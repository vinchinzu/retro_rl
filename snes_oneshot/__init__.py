"""Shared helpers for scripted one-shot SNES game completion agents."""

from snes_oneshot.actions import (
    ActionBuilder,
    buttons,
    buttons_multi,
    idle_action,
    idle_action_multi,
)
from snes_oneshot.cursor import (
    CursorPose,
    CursorTarget,
    at_target,
    plan_cursor_path,
    step_toward_target,
)
from snes_oneshot.behavior import (
    ActionNode,
    BehaviorNode,
    Condition,
    NodeStatus,
    Selector,
    Sequence,
    TickResult,
)
from snes_oneshot.combat import (
    AttackCadence,
    align_vertical_action,
    build_segment_tree,
    fight_nearest_action,
    walk_right_action,
)
from snes_oneshot.game_state import EnemyState, GameMode, GameState, ProjectileState
from snes_oneshot.ladder import LADDER, LadderEntry, LadderStatus, entry_for
from snes_oneshot.primitives import (
    ControllerPrimitive,
    FrameAction,
    PrimitiveKind,
    attack,
    mash_start,
    walk_down,
    walk_left,
    walk_right,
    walk_up,
)
from snes_oneshot.ram_diff import RamDelta, diff_changed, snapshot
from snes_oneshot.rom_setup import setup_game_rom
from snes_oneshot.segment_runner import (
    SegmentOutcome,
    SegmentTracker,
    configure_headless,
    is_screen_clear,
)
from snes_oneshot.watchdog import StuckDetector, WatchdogEvent

__all__ = [
    "ActionBuilder",
    "ActionNode",
    "AttackCadence",
    "BehaviorNode",
    "Condition",
    "ControllerPrimitive",
    "CursorPose",
    "CursorTarget",
    "EnemyState",
    "FrameAction",
    "GameMode",
    "GameState",
    "LADDER",
    "LadderEntry",
    "LadderStatus",
    "NodeStatus",
    "PrimitiveKind",
    "ProjectileState",
    "RamDelta",
    "SegmentOutcome",
    "SegmentTracker",
    "Selector",
    "Sequence",
    "StuckDetector",
    "TickResult",
    "WatchdogEvent",
    "align_vertical_action",
    "at_target",
    "attack",
    "build_segment_tree",
    "buttons",
    "buttons_multi",
    "configure_headless",
    "diff_changed",
    "entry_for",
    "fight_nearest_action",
    "idle_action",
    "idle_action_multi",
    "is_screen_clear",
    "mash_start",
    "plan_cursor_path",
    "setup_game_rom",
    "snapshot",
    "step_toward_target",
    "walk_down",
    "walk_left",
    "walk_right",
    "walk_right_action",
    "walk_up",
]
