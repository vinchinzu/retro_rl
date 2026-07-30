"""Provisional game-over rule: 3× standard room time ends the run.

Until stronger stuck/progress metrics exist, a bot that dwells in one room
longer than ``ROOM_TIMEOUT_MULTIPLIER`` times that room's baseline (or a
global default baseline) is treated as game over so sessions stop cleanly.

This module is game-agnostic: feed room keys + dwell frames each tick.
Vanilla Super Metroid room timing can supply baselines later
(``super_metroid.room_timer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from smz3.paths import ROOM_TIMEOUT_MULTIPLIER

# Fallback when no per-room baseline is known (5 minutes @ 60 fps).
DEFAULT_BASELINE_FRAMES = 5 * 60 * 60


class TimeoutReason(str, Enum):
    ROOM_DWELL = "room_dwell_exceeded_multiplier"
    SESSION_CAP = "session_frame_cap"


@dataclass(frozen=True)
class RoomBaseline:
    """Expected / standard frames for a room (emulator frames)."""

    room_key: str
    standard_frames: int
    source: str = "default"
    notes: str = ""

    def limit_frames(self, multiplier: float = ROOM_TIMEOUT_MULTIPLIER) -> int:
        return max(1, int(self.standard_frames * multiplier))

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_key": self.room_key,
            "standard_frames": self.standard_frames,
            "source": self.source,
            "notes": self.notes,
            "timeout_multiplier": ROOM_TIMEOUT_MULTIPLIER,
            "limit_frames": self.limit_frames(),
        }


@dataclass(frozen=True)
class TimeoutEvent:
    """A game-over timeout decision."""

    frame: int
    room_key: str
    dwell_frames: int
    limit_frames: int
    standard_frames: int
    multiplier: float
    reason: TimeoutReason = TimeoutReason.ROOM_DWELL

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "room_key": self.room_key,
            "dwell_frames": self.dwell_frames,
            "limit_frames": self.limit_frames,
            "standard_frames": self.standard_frames,
            "multiplier": self.multiplier,
            "reason": self.reason.value,
            "game_over": True,
        }


@dataclass
class RoomTimeoutWatchdog:
    """Track dwell time per room and fire when 3× baseline is exceeded.

    Parameters
    ----------
    baselines:
        Map of room_key → standard frame count (or :class:`RoomBaseline`).
    multiplier:
        Default ``ROOM_TIMEOUT_MULTIPLIER`` (3.0).
    default_baseline_frames:
        Used when the current room has no baseline entry.
    """

    baselines: dict[str, RoomBaseline] = field(default_factory=dict)
    multiplier: float = ROOM_TIMEOUT_MULTIPLIER
    default_baseline_frames: int = DEFAULT_BASELINE_FRAMES
    _current_room: str | None = field(default=None, repr=False)
    _entry_frame: int | None = field(default=None, repr=False)
    _game_over: TimeoutEvent | None = field(default=None, repr=False)
    events: list[TimeoutEvent] = field(default_factory=list)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, int | Mapping[str, Any]],
        *,
        multiplier: float = ROOM_TIMEOUT_MULTIPLIER,
        default_baseline_frames: int = DEFAULT_BASELINE_FRAMES,
        source: str = "mapping",
    ) -> RoomTimeoutWatchdog:
        baselines: dict[str, RoomBaseline] = {}
        for key, value in raw.items():
            if isinstance(value, Mapping):
                baselines[str(key)] = RoomBaseline(
                    room_key=str(key),
                    standard_frames=int(value["standard_frames"]),
                    source=str(value.get("source", source)),
                    notes=str(value.get("notes", "")),
                )
            else:
                baselines[str(key)] = RoomBaseline(
                    room_key=str(key),
                    standard_frames=int(value),
                    source=source,
                )
        return cls(
            baselines=baselines,
            multiplier=multiplier,
            default_baseline_frames=default_baseline_frames,
        )

    @property
    def is_game_over(self) -> bool:
        return self._game_over is not None

    @property
    def game_over_event(self) -> TimeoutEvent | None:
        return self._game_over

    def baseline_for(self, room_key: str) -> RoomBaseline:
        if room_key in self.baselines:
            return self.baselines[room_key]
        return RoomBaseline(
            room_key=room_key,
            standard_frames=self.default_baseline_frames,
            source="default_fallback",
        )

    def set_baseline(
        self,
        room_key: str,
        standard_frames: int,
        *,
        source: str = "manual",
        notes: str = "",
    ) -> RoomBaseline:
        bl = RoomBaseline(
            room_key=room_key,
            standard_frames=int(standard_frames),
            source=source,
            notes=notes,
        )
        self.baselines[room_key] = bl
        return bl

    def reset(self) -> None:
        self._current_room = None
        self._entry_frame = None
        self._game_over = None
        self.events.clear()

    def observe(
        self,
        *,
        frame: int,
        room_key: str,
        settled: bool = True,
    ) -> TimeoutEvent | None:
        """Ingest one frame. Return a TimeoutEvent if game-over just triggered.

        When ``settled`` is False (door transition / load), dwell does not
        accumulate toward the limit (time freezes for timeout purposes).
        """
        if self._game_over is not None:
            return None
        if not settled or not room_key:
            return None

        if self._current_room != room_key:
            self._current_room = room_key
            self._entry_frame = frame
            return None

        assert self._entry_frame is not None
        dwell = frame - self._entry_frame
        bl = self.baseline_for(room_key)
        limit = bl.limit_frames(self.multiplier)
        if dwell > limit:
            event = TimeoutEvent(
                frame=frame,
                room_key=room_key,
                dwell_frames=dwell,
                limit_frames=limit,
                standard_frames=bl.standard_frames,
                multiplier=self.multiplier,
                reason=TimeoutReason.ROOM_DWELL,
            )
            self._game_over = event
            self.events.append(event)
            return event
        return None

    def dwell_frames(self, frame: int) -> int | None:
        if self._entry_frame is None:
            return None
        return frame - self._entry_frame

    def report(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "default_baseline_frames": self.default_baseline_frames,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "game_over": self._game_over.to_dict() if self._game_over else None,
            "events": [e.to_dict() for e in self.events],
            "current_room": self._current_room,
            "entry_frame": self._entry_frame,
        }
