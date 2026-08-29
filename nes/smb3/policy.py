"""World 1 stage table and action-index replay for Super Mario Bros. 3 (NES).

Composer: ``STAGES`` rows. Action indices match ``SMB3_ACTIONS``:

0 NOTHING, 1 RIGHT, 2 RIGHT+B (run), 3 RIGHT+B+A (run+jump),
4 RIGHT+A, 5 A, 6 LEFT, 7 LEFT+B, 8 LEFT+B+A, 9 LEFT+A, 10 DOWN.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from retro_harness.platformer.actions import action_index_to_buttons
from smb3.platformer_levels import SMB3_ACTIONS
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

POLICY_DIR = Path(__file__).resolve().parent / "policies"
DEFAULT_LEVEL1_POLICY = POLICY_DIR / "level1_1.json"
DEFAULT_LEVEL2_POLICY = POLICY_DIR / "level1_2.json"


def load_action_indices(path: Path | str | None = None) -> list[int]:
    """Load a saved action-index sequence from JSON."""
    policy_path = Path(path) if path is not None else DEFAULT_LEVEL1_POLICY
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    actions = data.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"no actions in {policy_path}")
    return [int(a) for a in actions]


def action_index_to_nes(index: int) -> list[int]:
    """Map a policy action index to a 9-button NES vector."""
    buttons = action_index_to_buttons(index, SMB3_ACTIONS)
    return list(buttons[:9])


@dataclass(frozen=True)
class StageSpec:
    """One World 1 segment: map-enter script, in-level tape, checkpoints."""

    id: str
    policy_file: Path
    start_state: str
    after_state: str
    enter: Callable[[], list[FrameAction]] | None
    completion_min_progress: float
    recordings_prefix: str


@dataclass
class Level1Policy:
    """Replay a fixed action-index tape (World 1-1 or 1-2)."""

    actions: list[int]
    frame: int = 0

    @classmethod
    def from_file(cls, path: Path | str | None = None) -> Level1Policy:
        return cls(actions=load_action_indices(path))

    def reset(self) -> None:
        self.frame = 0

    def tick(self) -> FrameAction:
        """Next frame of input; idles after the sequence ends."""
        if self.frame >= len(self.actions):
            self.frame += 1
            return FrameAction(nes_idle_action(), "policy_done")
        index = self.actions[self.frame]
        self.frame += 1
        return FrameAction(action_index_to_nes(index), f"a{index}")

    def __len__(self) -> int:
        return len(self.actions)


def enter_level1_script() -> list[FrameAction]:
    """Map path from boot ready pose onto World 1-1, then confirm entry.

    Boot lands on a path node; World 1-1 is one RIGHT and one UP away.
    """
    frames: list[FrameAction] = []
    for _ in range(50):
        frames.append(FrameAction(nes_action("RIGHT"), "map_right"))
    for _ in range(25):
        frames.append(FrameAction(nes_idle_action(), "map_settle"))
    for _ in range(50):
        frames.append(FrameAction(nes_action("UP"), "map_up"))
    for _ in range(25):
        frames.append(FrameAction(nes_idle_action(), "map_settle"))
    for _ in range(3):
        for _ in range(2):
            frames.append(FrameAction(nes_action("A"), "map_enter"))
        for _ in range(8):
            frames.append(FrameAction(nes_idle_action(), "map_enter_wait"))
    return frames


def enter_level2_script() -> list[FrameAction]:
    """Map path from post-1-1 control onto World 1-2, then confirm entry.

    AfterLevel1 sits on completed 1-1. One RIGHT is a path T-junction
    (not enterable); a second RIGHT lands on the 1-2 panel (tile $04).
    Caller must wait for Map_Operation $0D before playing this script.
    """
    frames: list[FrameAction] = []
    for _ in range(8):
        frames.append(FrameAction(nes_action("RIGHT"), "map_right"))
    for _ in range(45):
        frames.append(FrameAction(nes_idle_action(), "map_settle"))
    for _ in range(8):
        frames.append(FrameAction(nes_action("RIGHT"), "map_right"))
    for _ in range(50):
        frames.append(FrameAction(nes_idle_action(), "map_settle"))
    for _ in range(3):
        for _ in range(2):
            frames.append(FrameAction(nes_action("A"), "map_enter"))
        for _ in range(8):
            frames.append(FrameAction(nes_idle_action(), "map_enter_wait"))
    return frames


STAGES: dict[str, StageSpec] = {
    "1-1": StageSpec(
        id="1-1",
        policy_file=DEFAULT_LEVEL1_POLICY,
        start_state="Level1_1",
        after_state="AfterLevel1",
        enter=enter_level1_script,
        completion_min_progress=1500.0,
        recordings_prefix="level1",
    ),
    "1-2": StageSpec(
        id="1-2",
        policy_file=DEFAULT_LEVEL2_POLICY,
        start_state="Level1_2",
        after_state="AfterLevel2",
        enter=enter_level2_script,
        completion_min_progress=1500.0,
        recordings_prefix="level2",
    ),
}
