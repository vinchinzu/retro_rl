"""World 1-1 clear policy for Super Mario Bros. 3 (NES).

Action indices match ``platformer_common.levels.smb3.SMB3_ACTIONS``:

0 NOTHING, 1 RIGHT, 2 RIGHT+B (run), 3 RIGHT+B+A (run+jump),
4 RIGHT+A, 5 A, 6 LEFT, 7 LEFT+B, 8 LEFT+B+A, 9 LEFT+A, 10 DOWN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platformer_common.actions import action_index_to_buttons
from platformer_common.levels.smb3 import SMB3_ACTIONS
from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction

POLICY_DIR = Path(__file__).resolve().parent / "policies"
DEFAULT_LEVEL1_POLICY = POLICY_DIR / "level1_1.json"


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


@dataclass
class Level1Policy:
    """Replay a fixed action sequence for World 1-1."""

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
