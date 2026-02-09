"""
Shared harness for retro gaming RL projects.

Provides common abstractions for SNES emulation via stable-retro:
- Controls: Keyboard and controller input handling
- Protocol: Task interfaces for composable behaviors
- Env: Environment setup utilities
- RAM State: Declarative RAM reading
- Splits: Segment timing / speedrun splits
- Play Session: Generic pygame play loop
- Bot Runner: Task-based autopilot framework
"""

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_DOWN,
    SNES_L,
    SNES_LEFT,
    SNES_R,
    SNES_RIGHT,
    SNES_SELECT,
    SNES_START,
    SNES_UP,
    SNES_X,
    SNES_Y,
    CONTROLLER_MAP,
    controller_action,
    keyboard_action,
    sanitize_action,
    sanitize_action_multi,
    sanitize_action_offset,
    init_controller,
    init_controllers,
)

from retro_harness.protocol import (
    TaskStatus,
    WorldState,
    ActionResult,
    TaskResult,
    Task,
)

from retro_harness.env import (
    add_custom_integrations,
    make_env,
    get_available_states,
    save_state,
)
from retro_harness.recordings import (
    ensure_gzip_state,
    append_jsonl,
    iter_jsonl,
    find_latest_recording,
    find_latest_recording_from_manifest,
)
from retro_harness.recorder import (
    SavePointSet,
    RecordingSession,
    LabeledRecorder,
    list_labeled_states,
)
from retro_harness.ram_state import (
    RAMSchema,
    RAMWatcher,
    read_u8,
    read_u16,
    read_u16_be,
    read_s8,
    read_s16,
)
from retro_harness.splits import (
    SplitTracker,
    SplitResult,
)
from retro_harness.bot_runner import (
    BotRunner,
    TaskSequencer,
    TaskRepeater,
)
# PlaySession imported lazily (depends on pygame)

__all__ = [
    # Controls
    "SNES_A", "SNES_B", "SNES_DOWN", "SNES_L", "SNES_LEFT",
    "SNES_R", "SNES_RIGHT", "SNES_SELECT", "SNES_START",
    "SNES_UP", "SNES_X", "SNES_Y", "CONTROLLER_MAP",
    "controller_action", "keyboard_action", "sanitize_action", "sanitize_action_multi",
    "sanitize_action_offset", "init_controller",
    "init_controllers",
    # Protocol
    "TaskStatus", "WorldState", "ActionResult", "TaskResult", "Task",
    # Env
    "add_custom_integrations", "make_env", "get_available_states", "save_state",
    # Recordings/logging
    "ensure_gzip_state", "append_jsonl", "iter_jsonl",
    "find_latest_recording", "find_latest_recording_from_manifest",
    # Labeled recorder
    "SavePointSet", "RecordingSession", "LabeledRecorder", "list_labeled_states",
    # RAM state
    "RAMSchema", "RAMWatcher", "read_u8", "read_u16", "read_u16_be", "read_s8", "read_s16",
    # Splits
    "SplitTracker", "SplitResult",
    # Bot runner
    "BotRunner", "TaskSequencer", "TaskRepeater",
]
