"""In-game clock and location timeline.

Harvest's calendar RAM (hour/minute) is first-class: probes, tests, and the
day planner share one clock type, hour-by-hour location marks, and the
12:00 HaveLunch stand. Play-session frames stay the speed source of truth
(60 fps); the in-game clock is where the farmer actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from harvest.core.ram_catalog import read_ram_value
from harvest.maps.map_config import get_map_name


SEGMENT_FPS = 60.0
LUNCH_HOUR = 12
LUNCH_MINUTE = 0


@dataclass(frozen=True, order=True)
class ClockTime:
    """In-game hour:minute from RAM ``0x11F1C`` / ``0x11F1D``."""

    hour: int
    minute: int = 0

    def __post_init__(self) -> None:
        if not (0 <= int(self.hour) <= 23):
            raise ValueError("hour must be 0..23")
        if not (0 <= int(self.minute) <= 59):
            raise ValueError("minute must be 0..59")

    @property
    def minutes(self) -> int:
        return int(self.hour) * 60 + int(self.minute)

    def __str__(self) -> str:
        return f"{int(self.hour):02d}:{int(self.minute):02d}"

    def to_dict(self) -> dict:
        return {"hour": int(self.hour), "minute": int(self.minute), "clock": str(self)}

    def minutes_until(self, other: "ClockTime") -> int:
        return int(other.minutes) - int(self.minutes)


LUNCH_TIME = ClockTime(LUNCH_HOUR, LUNCH_MINUTE)


def clock_from_ram(ram) -> ClockTime:
    return ClockTime(int(read_ram_value(ram, "hour")), int(read_ram_value(ram, "minute")))


def clock_from_mapping(row: Mapping) -> Optional[ClockTime]:
    if "hour" not in row:
        return None
    return ClockTime(int(row.get("hour", 0)), int(row.get("minute", 0)))


def format_segment_time(frames: int | None) -> dict:
    """Frame split used by corridor benches. 60 fps is the play clock."""
    if frames is None:
        return {"frames": None, "seconds": None, "clock": None}
    n = max(0, int(frames))
    seconds = n / SEGMENT_FPS
    minutes = int(seconds // 60)
    return {
        "frames": n,
        "seconds": round(seconds, 3),
        "clock": f"{minutes:02d}:{seconds % 60:05.2f}",
    }


def compare_frame_benches(before: int | None, after: int | None) -> dict:
    """Negative ``delta_frames`` means the after row is faster."""
    if before is None or after is None:
        return {
            "before": format_segment_time(before),
            "after": format_segment_time(after),
            "delta_frames": None,
            "faster": False,
        }
    delta = int(after) - int(before)
    return {
        "before": format_segment_time(before),
        "after": format_segment_time(after),
        "delta_frames": delta,
        "faster": delta < 0,
    }


@dataclass(frozen=True)
class LocationMark:
    """One RAM sample: play frame + in-game clock + pixel stand."""

    frame: int
    clock: ClockTime
    tilemap: int
    x: int
    y: int
    map_name: str = ""
    stamina: int | None = None
    held_item: int = 0
    phase: str = ""

    @property
    def tile(self) -> tuple[int, int]:
        return (int(self.x) // 16, int(self.y) // 16)

    def to_dict(self) -> dict:
        row = {
            "frame": int(self.frame),
            "hour": int(self.clock.hour),
            "minute": int(self.clock.minute),
            "clock": str(self.clock),
            "tilemap": int(self.tilemap),
            "tilemap_hex": f"0x{int(self.tilemap):02X}",
            "map": self.map_name or get_map_name(int(self.tilemap)),
            "x": int(self.x),
            "y": int(self.y),
            "tx": self.tile[0],
            "ty": self.tile[1],
            "held_item": int(self.held_item),
            "phase": self.phase,
        }
        if self.stamina is not None:
            row["stamina"] = int(self.stamina)
        return row


def mark_from_mapping(row: Mapping) -> LocationMark:
    clock = clock_from_mapping(row) or ClockTime(0, 0)
    tilemap = int(row.get("tilemap", -1))
    return LocationMark(
        frame=int(row.get("frame", 0)),
        clock=clock,
        tilemap=tilemap,
        x=int(row.get("x", 0)),
        y=int(row.get("y", 0)),
        map_name=str(row.get("map") or get_map_name(tilemap)),
        stamina=int(row["stamina"]) if row.get("stamina") is not None else None,
        held_item=int(row.get("held_item", 0)),
        phase=str(row.get("phase") or ""),
    )


def mark_from_ram(ram, frame: int, *, phase: str = "") -> LocationMark:
    from harvest.tasks.nav import get_pos_from_ram

    pos = get_pos_from_ram(ram)
    tilemap = int(read_ram_value(ram, "tilemap"))
    stamina = None
    try:
        stamina = int(read_ram_value(ram, "stamina"))
    except Exception:
        stamina = None
    held = 0
    try:
        held = int(read_ram_value(ram, "held_item"))
    except Exception:
        held = 0
    return LocationMark(
        frame=int(frame),
        clock=clock_from_ram(ram),
        tilemap=tilemap,
        x=int(pos.x),
        y=int(pos.y),
        map_name=get_map_name(tilemap),
        stamina=stamina,
        held_item=held,
        phase=phase,
    )


def _heading(dx: int, dy: int) -> Optional[str]:
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def path_waste(
    samples: Sequence[Mapping | LocationMark],
    *,
    stasis_min_frames: int = 45,
    exclude_phases: Sequence[str] = ("wait_lunch",),
) -> dict:
    """Count wall-hug stasis and heading changes on a time-ordered trace.

    ``stasis_windows`` are ≥``stasis_min_frames`` with no pixel motion (the
    harvest-route corner-hug signal). ``turns`` count heading changes between
    successive moved samples — extra 90° zigzags on an otherwise straight
    corridor.
    """
    skip = {str(phase) for phase in exclude_phases}
    marks = [
        sample if isinstance(sample, LocationMark) else mark_from_mapping(sample)
        for sample in samples
    ]
    if skip:
        marks = [mark for mark in marks if mark.phase not in skip]
    if len(marks) < 2:
        return {
            "stasis_windows": [],
            "stasis_frames": 0,
            "turns": 0,
            "moves": 0,
        }

    windows: list[dict] = []
    stasis_start: Optional[LocationMark] = None
    last = marks[0]
    for mark in marks[1:]:
        moved = mark.x != last.x or mark.y != last.y or mark.tilemap != last.tilemap
        if not moved:
            if stasis_start is None:
                stasis_start = last
        elif stasis_start is not None:
            length = last.frame - stasis_start.frame
            if length >= stasis_min_frames:
                windows.append(
                    {
                        "start": stasis_start.frame,
                        "end": last.frame,
                        "length": length,
                        "tilemap": stasis_start.tilemap,
                        "pixel": [stasis_start.x, stasis_start.y],
                        "tile": list(stasis_start.tile),
                        "clock": str(stasis_start.clock),
                        "map": stasis_start.map_name,
                    }
                )
            stasis_start = None
        last = mark
    if stasis_start is not None:
        length = last.frame - stasis_start.frame
        if length >= stasis_min_frames:
            windows.append(
                {
                    "start": stasis_start.frame,
                    "end": last.frame,
                    "length": length,
                    "tilemap": stasis_start.tilemap,
                    "pixel": [stasis_start.x, stasis_start.y],
                    "tile": list(stasis_start.tile),
                    "clock": str(stasis_start.clock),
                    "map": stasis_start.map_name,
                }
            )

    turns = 0
    moves = 0
    prev_heading: Optional[str] = None
    prev = marks[0]
    for mark in marks[1:]:
        # Tile steps, not every sub-pixel jog — that's the corridor-turn signal.
        dx = mark.tile[0] - prev.tile[0]
        dy = mark.tile[1] - prev.tile[1]
        heading = _heading(dx, dy)
        if heading is None:
            prev = mark
            continue
        moves += 1
        if prev_heading is not None and heading != prev_heading:
            turns += 1
        prev_heading = heading
        prev = mark

    return {
        "stasis_windows": windows,
        "stasis_frames": int(sum(int(w["length"]) for w in windows)),
        "turns": turns,
        "moves": moves,
    }


@dataclass
class ClockTimeline:
    """Hour-by-hour (and minute-fine) location log from RAM samples."""

    samples: list[LocationMark] = field(default_factory=list)

    @classmethod
    def from_samples(cls, rows: Iterable[Mapping | LocationMark]) -> "ClockTimeline":
        marks = [
            row if isinstance(row, LocationMark) else mark_from_mapping(row)
            for row in rows
        ]
        marks.sort(key=lambda mark: (mark.frame, mark.clock.minutes))
        return cls(samples=marks)

    def observe(self, mark: LocationMark) -> None:
        self.samples.append(mark)

    @property
    def start(self) -> Optional[LocationMark]:
        return self.samples[0] if self.samples else None

    @property
    def end(self) -> Optional[LocationMark]:
        return self.samples[-1] if self.samples else None

    def hour_marks(self) -> list[LocationMark]:
        """First sample of the run, then the first sample of each later hour."""
        seen: set[int] = set()
        marks: list[LocationMark] = []
        for mark in self.samples:
            if not marks:
                marks.append(mark)
                seen.add(mark.clock.hour)
                continue
            if mark.clock.hour not in seen:
                marks.append(mark)
                seen.add(mark.clock.hour)
        return marks

    def minute_marks(self) -> list[LocationMark]:
        """First sample of each distinct in-game minute (clock resolution)."""
        seen: set[tuple[int, int]] = set()
        marks: list[LocationMark] = []
        for mark in self.samples:
            key = (mark.clock.hour, mark.clock.minute)
            if key in seen:
                continue
            seen.add(key)
            marks.append(mark)
        return marks

    def lunch_mark(self) -> Optional[LocationMark]:
        """First sample at or after 12:00 (HaveLunch pulse)."""
        for mark in self.samples:
            if mark.clock >= LUNCH_TIME:
                return mark
        return None

    def mark_at_or_after(self, clock: ClockTime) -> Optional[LocationMark]:
        for mark in self.samples:
            if mark.clock >= clock:
                return mark
        return None

    def waste(self, *, stasis_min_frames: int = 45) -> dict:
        return path_waste(self.samples, stasis_min_frames=stasis_min_frames)

    def to_dict(self) -> dict:
        start = self.start
        end = self.end
        lunch = self.lunch_mark()
        frames = None
        if start is not None and end is not None:
            frames = max(0, end.frame - start.frame)
        game_minutes = None
        if start is not None and end is not None:
            game_minutes = end.clock.minutes - start.clock.minutes
        return {
            "start": start.to_dict() if start else None,
            "end": end.to_dict() if end else None,
            "frames": frames,
            "play": format_segment_time(frames),
            "game_minutes": game_minutes,
            "hour_marks": [mark.to_dict() for mark in self.hour_marks()],
            "minute_marks": [mark.to_dict() for mark in self.minute_marks()],
            "lunch": lunch.to_dict() if lunch else None,
            "waste": self.waste(),
        }


# Live Y1_Inside_House → grape ship + 12:00 lunch
# (snes/harvest/recordings/mountain_segments_clock.json). previous_frames is
# the prior ship bench so Δ stays first-class.
BERRY_SHIP_BENCH = {
    "frames": 3154,
    "previous_frames": 3224,
    "start_clock": "06:08",
    "end_clock": "10:10",
    "mountain_entry_to_grape": 966,
    "grape_to_mountain_exit": 410,
    "pick_keep": 293,
    "lunch_clock": "12:00",
    "lunch_map": "farm",
    "lunch_pixel": (135, 456),
    "hour_locations": (
        (6, "house", 128, 200),
        (7, "path", 137, 10),
        (8, "mountain_spring", 505, 694),
        (9, "mountain_spring", 457, 584),
        (10, "path", 244, 118),
        (11, "farm", 135, 456),
        (12, "farm", 135, 456),
    ),
}


__all__ = [
    "BERRY_SHIP_BENCH",
    "ClockTime",
    "ClockTimeline",
    "LUNCH_HOUR",
    "LUNCH_MINUTE",
    "LUNCH_TIME",
    "LocationMark",
    "SEGMENT_FPS",
    "clock_from_mapping",
    "clock_from_ram",
    "compare_frame_benches",
    "format_segment_time",
    "mark_from_mapping",
    "mark_from_ram",
    "path_waste",
]
