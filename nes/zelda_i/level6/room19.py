"""Level 6 settle stages: idle at a room's entry mouth and census the spawn.

The room hops (0x29 -> 0x19 -> 0x09) are ``level6.path`` lattice walks; the
old 0x18 east hop, 0x19 Map pickup and 0x19 KEY-UP controllers went with the
Gleeok route (the 0x28 bomb wall reaches 0x29 directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.ids import INVULN_MOVER_OBJECT_TYPE
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "SETTLE_19_IDLE_FRAMES",
    "SETTLE_19_MAX_FRAMES",
    "Level6Settle19Controller",
    "make_settle_09_controller",
    "make_settle_39_controller",
    "make_settle_3a_controller",
]


SETTLE_19_IDLE_FRAMES = 160
SETTLE_19_SAMPLE_PERIOD = 12
SETTLE_19_MAX_FRAMES = 400
_CENSUS_SKIP_TYPES = frozenset({0, INVULN_MOVER_OBJECT_TYPE})


def _live_census_objects(snap: ZeldaSnapshot) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for obj in snap.objects:
        type_id = int(obj.type_id)
        if obj.slot == 0 or type_id == 0:
            continue
        rows.append(
            {
                "slot": int(obj.slot),
                "type": type_id,
                "x": int(obj.x),
                "y": int(obj.y),
                "hp": int(obj.hp),
            }
        )
    return rows


@dataclass
class Level6Settle19Controller:
    """Idle at leftover. Do not walk into beams / wizzrobes."""

    spec_id: str = "level6_settle_0x19"
    room: int = LEVEL6_MAP_ROOM
    idle_frames: int = SETTLE_19_IDLE_FRAMES
    sample_period: int = SETTLE_19_SAMPLE_PERIOD
    max_frames: int = SETTLE_19_MAX_FRAMES
    frames: int = 0
    idle_in_room: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    type_histogram: dict[str, int] = field(default_factory=dict)
    leftover: dict[str, int] = field(default_factory=dict)
    policy: str = "IDLE at 0x19 west mouth; census spawn; do not walk"

    def _record(self, snap: ZeldaSnapshot, *, force: bool = False) -> None:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "room_item_id": int(snap.room_item_id),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        live = _live_census_objects(snap)
        counts: dict[int, int] = {}
        for row in live:
            type_id = int(row["type"])
            if type_id in _CENSUS_SKIP_TYPES:
                continue
            counts[type_id] = counts.get(type_id, 0) + 1
        for type_id, n in counts.items():
            key = f"0x{type_id:02x}"
            prev = self.type_histogram.get(key, 0)
            if n > prev:
                self.type_histogram[key] = n
        if force or self.frames <= 2 or self.frames % self.sample_period == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "objects": live,
                    "cur_opened_doors": int(snap.cur_opened_doors),
                    "open_doorway_mask": int(snap.open_doorway_mask),
                    "room_item_id": int(snap.room_item_id),
                    "map": int(snap.map),
                }
            )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            self._record(snap, force=True)
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            self._record(snap, force=True)
            return FrameAction(nes_idle_action(), "link_death")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return FrameAction(nes_idle_action(), "left_level")
        if snap.screen != self.room:
            self.failed = True
            self.notes.append(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), f"left_0x{self.room:02x}")

        self.idle_in_room += 1
        self._record(snap, force=self.idle_in_room >= self.idle_frames)
        if self.idle_in_room >= self.idle_frames:
            self.success = True
            hist = ",".join(
                f"{k}x{n}" for k, n in sorted(self.type_histogram.items())
            )
            self.notes.append(
                f"settled_{self.room:02x}_{snap.link_x}_{snap.link_y}_{hist}"
            )
            return FrameAction(nes_idle_action(), "settled")
        return FrameAction(nes_idle_action(), "spawn_idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "idle_in_room": self.idle_in_room,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": self.policy,
            "type_histogram": dict(self.type_histogram),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
        }


def make_settle_09_controller() -> Level6Settle19Controller:
    """Idle ~160f in play 0x09 south mouth. Do not walk into wizzrobes."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x09",
        room=LEVEL6_ROD_WIZZ_ROOM,
        policy="IDLE at 0x09 south mouth; census spawn; do not walk",
    )


def make_settle_39_controller() -> Level6Settle19Controller:
    """Idle ~160f in dark 0x39. Census types; do not invent Gohma."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x39",
        room=LEVEL6_DARK_39_ROOM,
        policy="IDLE at 0x39 north mouth; census spawn; no candle/Gohma",
    )


def make_settle_3a_controller() -> Level6Settle19Controller:
    """Idle ~160f in play 0x3A. Census types; do not push the block."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x3a",
        room=LEVEL6_BLOCK_3A_ROOM,
        policy="IDLE at 0x3A west mouth; census spawn; do not push",
    )
