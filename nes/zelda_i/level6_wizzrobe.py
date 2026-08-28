"""Level 6 wizzrobe backstep combat for rooms 0x7a and 0x78.

Sword misses when overlapping at the door; controllers retreat when stuck
too close without a kill, then re-engage. Specs live in ``level6_dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.dungeon import GenericDungeonRoomController
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = (
    "Level6EastKeyController",
    "Level6WestWizzrobeController",
    "make_east_key_controller",
    "make_west_wizzrobe_controller",
)


@dataclass
class Level6EastKeyController(GenericDungeonRoomController):
    """Generic room clear + wizzrobe backstep when overlapping without kills.

    Live: sword swings at distance 0 on the west door miss forever; retreat
    when stuck too close, then re-engage from a short offset.
    """

    last_progress_frame: int = 0
    prev_live_count: int = -1
    backstep_frames: int = 0

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        self.combat_frames += 1
        n_live = len(live)
        if self.prev_live_count < 0:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
        elif n_live < self.prev_live_count:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
            self.backstep_frames = 0
            self.notes.append(f"kill_to_{n_live}_f{self.frames}")

        if not live:
            return self._patrol(snap)

        nearest = min(
            live,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        stuck_close = (
            dist < 16 and (self.frames - self.last_progress_frame) > 100
        )
        if stuck_close or self.backstep_frames > 0:
            if self.backstep_frames <= 0:
                self.backstep_frames = 24
                self.notes.append(f"backstep_f{self.frames}_d{dist}")
            self.backstep_frames -= 1
            if self.backstep_frames == 0:
                # Allow a fresh engage window after retreat.
                self.last_progress_frame = self.frames
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                direction = "LEFT" if dx >= 0 else "RIGHT"
            else:
                direction = "UP" if dy >= 0 else "DOWN"
            # Prefer center when pinned on a door edge.
            if snap.link_x < 40:
                direction = "RIGHT"
            elif snap.link_x > 200:
                direction = "LEFT"
            return FrameAction(nes_action(direction), "wizzrobe_backstep")

        if dist < self.spec.combat.engage_distance:
            return self._engage(snap, nearest)
        return self._patrol(snap)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["last_progress_frame"] = self.last_progress_frame
        base["prev_live_count"] = self.prev_live_count
        return base


def make_east_key_controller() -> Level6EastKeyController:
    """Factory: GenericDungeonRoomController subclass bound to ROOM_7A_SPEC."""
    from zelda_i.level6_dungeon import ROOM_7A_SPEC

    return Level6EastKeyController(spec=ROOM_7A_SPEC)


@dataclass
class Level6WestWizzrobeController(Level6EastKeyController):
    """Same wizzrobe backstep combat as east key, bound to ROOM_78_SPEC."""


def make_west_wizzrobe_controller() -> Level6WestWizzrobeController:
    """Factory: backstep combat controller for 0x78 west wizzrobes."""
    from zelda_i.level6_dungeon import ROOM_78_SPEC

    return Level6WestWizzrobeController(spec=ROOM_78_SPEC)
