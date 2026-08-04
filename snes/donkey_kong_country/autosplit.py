from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


def read_u16(ram: Iterable[int], offset: int) -> Optional[int]:
    try:
        if offset < 0 or offset + 1 >= len(ram):  # type: ignore[arg-type]
            return None
        return int(ram[offset]) | (int(ram[offset + 1]) << 8)  # type: ignore[index]
    except TypeError:
        data = list(ram)
        if offset < 0 or offset + 1 >= len(data):
            return None
        return int(data[offset]) | (int(data[offset + 1]) << 8)


def read_level_timer_frames(
    ram: Iterable[int],
    *,
    frames_offset: int,
    minutes_offset: int,
    fps: int = 60,
) -> Optional[int]:
    frames = read_u16(ram, frames_offset)
    minutes = read_u16(ram, minutes_offset)
    if frames is None and minutes is None:
        return None
    frames = frames or 0
    minutes = minutes or 0
    return minutes * fps * 60 + frames


@dataclass
class LevelStartDetector:
    min_moving_frames: int = 2
    last_total_frames: Optional[int] = None
    moving_frames: int = 0
    movement_seen: bool = False

    def reset(self) -> None:
        self.last_total_frames = None
        self.moving_frames = 0
        self.movement_seen = False

    def update(self, total_frames: Optional[int], moved_this_frame: bool) -> bool:
        if total_frames is None:
            self.moving_frames = 0
            self.last_total_frames = None
            self.movement_seen = False
            return False
        if moved_this_frame:
            self.movement_seen = True
        if self.last_total_frames is not None and total_frames != self.last_total_frames:
            self.moving_frames += 1
        else:
            self.moving_frames = 0
        self.last_total_frames = total_frames
        return total_frames > 0 and self.moving_frames >= self.min_moving_frames and self.movement_seen
