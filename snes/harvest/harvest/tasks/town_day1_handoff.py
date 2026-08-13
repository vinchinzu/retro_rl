"""Spring D1 town handoff — six talks, truck leave, shed pickups, sleep → D2.

Precomputed from ``docs/town_day1_recon.md`` and verified stands in
``tasks/town_day1_rest.json`` (2026-08-01). Controller-only; no RAM writes.

Completion mask is ``d1_town_event_mask`` ``0x3F`` before the truck leave
response. Flower owner + livestock gift the watering can and grass seed bag
onto the shed shelf (already present from new-game init as row2 ``0x88``);
this handoff also *picks them up* into the 2-slot carry pair after return.

Implementation split (LOC budget):
  - ``town_day1_tasks`` — mask bits, talk/walk/assert helpers
  - ``town_day1_build`` — ``build_day1_handoff_tasks`` sequence assembly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from retro_harness import Task, TaskResult, TaskStatus, WorldState

from harvest.core.carry import seed_item_id, tool_in_carry_pair
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import Tool
from harvest.tasks.town_day1_build import build_day1_handoff_tasks
from harvest.tasks.town_day1_tasks import (
    BIT_ANN,
    BIT_EVE,
    TARGET_MASK,
    SequenceTask,
    read_mask,
)


@dataclass
class TownDay1HandoffTask(Task):
    """Full precomputed D1 town handoff for natural-entry automation."""

    name: str = "town_day1_handoff"
    include_sleep: bool = True
    require_full_mask: bool = True
    pick_starter_tools: bool = True
    # None = auto: require shed when house_size==0 (clean power-on / Gate B).
    require_starter_tools: Optional[bool] = None
    timeout: int = 90_000

    _inner: Optional[SequenceTask] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _require_starter_tools_effective: bool = field(default=False, init=False)
    _house_size_at_start: int = field(default=-1, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        house_size = int(read_ram_value(world.ram, "house_size", raw=True))
        self._house_size_at_start = house_size
        if self.require_starter_tools is None:
            # Clean starter house is size 0; upgraded fixtures (AnnEve) are 2.
            require_shed = house_size == 0
        else:
            require_shed = bool(self.require_starter_tools)
        self._require_starter_tools_effective = require_shed
        # Rest recording only when Ann|Eve bits already set (AnnEve oracle).
        # Clean power-on/Town_Gate (mask 0) must compose pure routes (rr-bhr).
        start_mask = read_mask(world.ram)
        ann_eve_ready = (start_mask & (BIT_ANN | BIT_EVE)) == (BIT_ANN | BIT_EVE)
        use_rest = ann_eve_ready and not require_shed
        self._inner = build_day1_handoff_tasks(
            include_sleep=self.include_sleep,
            require_full_mask=self.require_full_mask,
            pick_starter_tools=self.pick_starter_tools,
            require_starter_tools=require_shed,
            use_rest_recording=use_rest,
        )
        self._inner.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._inner) if self._inner else None
        phase = ""
        if self._inner is not None:
            phase = self._inner.progress_snapshot().phase_text or ""
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=phase,
            step_count=self._step_count,
            child=child,
        )

    def summary(self, world: WorldState) -> dict:
        mask = read_mask(world.ram)
        sel = int(read_ram_value(world.ram, "tool_selected"))
        back = int(read_ram_value(world.ram, "tool_backpack"))
        return {
            "mask": mask,
            "mask_hex": f"0x{mask:02X}",
            "mask_complete": (mask & TARGET_MASK) == TARGET_MASK,
            "day": int(read_ram_value(world.ram, "day")),
            "season": int(read_ram_value(world.ram, "season")),
            "hour": int(read_ram_value(world.ram, "hour")),
            "tilemap": int(read_ram_value(world.ram, "tilemap")),
            "x": int(read_ram_value(world.ram, "player_x")),
            "y": int(read_ram_value(world.ram, "player_y")),
            "tool_selected": sel,
            "tool_backpack": back,
            "has_watering_can": tool_in_carry_pair(world.ram, int(Tool.WATERING_CAN)),
            "has_grass_seeds": tool_in_carry_pair(world.ram, seed_item_id("grass")),
            "grass_seeds_stock": int(read_ram_value(world.ram, "grass_seeds")),
            "house_size": int(read_ram_value(world.ram, "house_size", raw=True)),
            "house_size_at_start": self._house_size_at_start,
            "require_starter_tools": self._require_starter_tools_effective,
            "frames": self._step_count,
            "phase": (
                self._inner.progress_snapshot().phase_text if self._inner is not None else ""
            ),
        }

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._inner is None:
            self.reset(world)
        assert self._inner is not None
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"handoff timeout mask=0x{read_mask(world.ram):02X}",
            )
        return self._inner.step(world)


__all__ = [
    "TARGET_MASK",
    "TownDay1HandoffTask",
    "build_day1_handoff_tasks",
    "read_mask",
]
