"""Autonomous farm bot: day plan, crop, grass, and clearing modes."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from retro_harness import TaskStatus, WorldState
from harvest.paths import TASKS_DIR as PROJECT_TASKS_DIR
from harvest.planner.day_plan import (
    DayPlannerPolicy,
    DayPlanTask,
    MultiDayPlannerTask,
    PHASE_SEQUENCES,
    PhaseSpec,
    auto_day_plan_decision,
    is_rainy_weather,
    ram_has_waterable_crops,
    read_world_date,
    read_world_day_time,
)
from harvest.planner.local_llm import build_local_llm_plan_advisor_from_env
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.runtime.bot_input import env_flag
from harvest.runtime.power_on import PowerOnStartTask
from harvest.tasks.crop_planter import CropWaterTask, DEFAULT_CROP_BOUNDS
from harvest.tasks.farm_clear_task import FarmClearTask
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    DebrisType,
)
from harvest.tasks.fence_flow import FenceClearLoopTask
from harvest.tasks.grass_planter import (
    DEFAULT_BOUNDS as GRASS_DEFAULT_BOUNDS,
    DEFAULT_NO_GO_RECTS as GRASS_DEFAULT_NO_GO,
    GrassPlantTask,
)
from harvest.tasks.town_day1_handoff import TownDay1HandoffTask

TASKS_DIR = os.fspath(PROJECT_TASKS_DIR)

class AutoClearBot:
    """Farm clearing bot."""

    def __init__(
        self,
        priority: Optional[List[DebrisType]] = None,
        clear_fences_first: Optional[bool] = None,
        clear_fences_only: bool = False,
        grass_enabled: bool = False,
        till_only: bool = False,
        grass_bounds: Optional[tuple] = None,
        grass_no_go: Optional[List[tuple]] = None,
        grass_seed_hack: bool = False,
        crop_enabled: bool = False,
        crop_seed_type: str = "potato",
        day_plan_enabled: bool = False,
        day_plan_sequence: Optional[str] = None,
        auto_day_plan_state_name: Optional[str] = None,
        multi_day_until_day: Optional[int] = None,
        multi_day_until_season: Optional[int] = None,
        multi_day_count: Optional[int] = None,
        eve_target_hearts: int = 10,
        power_on: bool = False,
        d1_handoff: Optional[bool] = None,
    ):
        self.clear_task = FarmClearTask(
            priority=priority,
            tasks_dir=TASKS_DIR,
            fetch_tools=not (day_plan_enabled or crop_enabled),
        )
        self.clearer = self.clear_task.clearer
        self.clear_task_started = False
        self.clear_task_done = False

        self.env = None
        self.enabled = False
        self.disable_reason: Optional[str] = None
        self.frame_count = 0
        self.initial_tilemap: Optional[int] = None
        self.map_locked = False
        self.map_hist: Dict[int, int] = {}
        self.fence_task = FenceClearLoopTask(max_fences=None)
        self.fence_task_started = False
        self.fence_task_done = False
        if clear_fences_first is None:
            clear_fences_first = not os.getenv("SKIP_FENCE_TOSS", "").lower() in (
                "1",
                "true",
                "yes",
            )
        self.fence_task_enabled = clear_fences_first
        self.fence_task_only = clear_fences_only

        # Grass planting
        self.grass_enabled = grass_enabled
        self.grass_seed_hack = grass_seed_hack or grass_enabled
        self.grass_task = GrassPlantTask(
            bounds=grass_bounds or GRASS_DEFAULT_BOUNDS,
            no_go_rects=(
                grass_no_go
                if grass_no_go is not None
                else list(GRASS_DEFAULT_NO_GO)
            ),
            till_only=till_only,
        )
        self.grass_task_started = False
        self.grass_task_done = False

        # Crop planting + watering
        self.crop_enabled = crop_enabled
        self.crop_seed_type = crop_seed_type
        self.crop_seed_hack = env_flag("ALLOW_CROP_RAM_SHORTCUTS")
        self.crop_task = CropWaterTask(
            seed_type=crop_seed_type,
            bounds=DEFAULT_CROP_BOUNDS,
        )
        self.crop_task_started = False
        self.crop_task_done = False

        # Day plan mode
        self.day_plan_enabled = day_plan_enabled
        self.day_plan_auto = day_plan_enabled and day_plan_sequence is None
        self.auto_day_plan_state_name = auto_day_plan_state_name
        self.day_plan_sequence_name = day_plan_sequence
        self.multi_day_until_day = multi_day_until_day
        self.multi_day_until_season = multi_day_until_season
        self.multi_day_count = multi_day_count
        self.eve_target_hearts = eve_target_hearts
        self.day_plan_policy = DayPlannerPolicy()
        self.day_plan_advisor = build_local_llm_plan_advisor_from_env()
        self.day_plan_decision = None
        self.day_plan_task = self._build_day_plan_task(day_plan_sequence)
        self.day_plan_started = False
        self.day_plan_done = False
        self._day_plan_start_date: Optional[tuple[int, int]] = None
        self._pending_auto_day_plan_rebuild = False
        self._pending_auto_day_plan_settle_frames = 0

        # Clean power-on bootstrap (title → new diary → Spring D1 town).
        # Mirrors harvest.scripts.run_to_day2 --power-on for live viewing.
        self.power_on_enabled = bool(power_on)
        self.power_on_task = PowerOnStartTask() if self.power_on_enabled else None
        self.power_on_started = False
        self.power_on_done = not self.power_on_enabled
        # Default: after power-on, run D1 town talks + shed + sleep → D2.
        if d1_handoff is None:
            d1_handoff = self.power_on_enabled
        self.d1_handoff_enabled = bool(d1_handoff)
        self.d1_handoff_task = (
            TownDay1HandoffTask(
                include_sleep=True,
                pick_starter_tools=True,
                require_starter_tools=None,  # auto: house_size==0 → grass+can
            )
            if self.d1_handoff_enabled
            else None
        )
        self.d1_handoff_started = False
        self.d1_handoff_done = not self.d1_handoff_enabled

        # Skip startup tasks in day plan mode
        if day_plan_enabled:
            self.grass_seed_hack = False
            self.fence_task_enabled = False
            self.fence_task_done = True

        # Skip fence startup in crop mode
        if crop_enabled:
            self.fence_task_enabled = False
            self.fence_task_done = True

    def set_env(self, env):
        self.env = env

    def _resolve_crop_seed_type(self, ram: Optional[np.ndarray]) -> str:
        """Prefer calendar-aware seed selection over a stale potato default."""
        from harvest.planner.day_plan_status import resolve_seed_type_from_ram

        if self.crop_seed_type and self.crop_seed_type != "potato":
            return self.crop_seed_type
        if ram is None:
            return self.crop_seed_type
        resolved = resolve_seed_type_from_ram(ram)
        return resolved or self.crop_seed_type

    def _sync_seasonal_day_plan_context(self, ram: Optional[np.ndarray]) -> None:
        """Refresh seed type and planting policy from the live calendar."""
        from harvest.planner.day_phase_types import day_planner_policy_for_season
        from harvest.planner.day_plan_status import read_world_date

        if ram is None:
            return
        self.crop_seed_type = self._resolve_crop_seed_type(ram)
        season, _day = read_world_date(ram)
        self.day_plan_policy = day_planner_policy_for_season(
            season,
            self.day_plan_policy,
        )

    def _build_day_plan_task(self, sequence_name: Optional[str]):
        ram = self.env.get_ram() if self.env is not None else None
        self._sync_seasonal_day_plan_context(ram)

        if self.multi_day_count is not None or self.multi_day_until_day is not None:
            target_days = self.multi_day_count
            return MultiDayPlannerTask(
                seed_type=self.crop_seed_type,
                until_day=self.multi_day_until_day or 30,
                until_season=self.multi_day_until_season or 0,
                target_days=target_days,
                max_days=target_days or 40,
                policy=self.day_plan_policy,
                plan_advisor=self.day_plan_advisor,
            )
        # Manual override: use a named sequence from PHASE_SEQUENCES
        if sequence_name == "eve_loop":
            phase_seq = [
                PhaseSpec(
                    "EVE_TALK_LOOP",
                    "eve_talk_loop",
                    {
                        "task_name": "talk_eve_loop",
                        "target_hearts": self.eve_target_hearts,
                    },
                )
            ]
        elif sequence_name:
            phase_seq = PHASE_SEQUENCES.get(sequence_name)
        else:
            # Dynamic builder: inspect state and assemble phases
            decision = auto_day_plan_decision(
                state_name=self.auto_day_plan_state_name,
                ram=ram,
                policy=self.day_plan_policy,
                advisor=self.day_plan_advisor,
            )
            self.day_plan_decision = decision
            phase_seq = list(decision.phases)
            if decision.deferred:
                summary = ", ".join(
                    f"{item.phase}({item.reason})" for item in decision.deferred
                )
                print(f"[BOT] Day plan deferred for later: {summary}")
        return DayPlanTask(
            seed_type=self.crop_seed_type,
            phase_sequence=phase_seq,
            state_name=self.auto_day_plan_state_name,
            policy=self.day_plan_policy,
        )

    def _configure_day_plan(self, sequence_name: Optional[str]) -> None:
        self.day_plan_sequence_name = sequence_name
        self.day_plan_task = self._build_day_plan_task(sequence_name)
        self.day_plan_started = False
        self.day_plan_done = False
        self._day_plan_start_date = None
        self._pending_auto_day_plan_rebuild = False
        self._pending_auto_day_plan_settle_frames = 0
        self.disable_reason = None

    def _multi_day_enabled(self) -> bool:
        return self.multi_day_until_day is not None or self.multi_day_count is not None

    def _should_rebuild_auto_day_plan(self, ram: np.ndarray) -> bool:
        if not self.day_plan_auto or self._multi_day_enabled():
            return False
        if self._day_plan_start_date is None:
            return False
        return read_world_date(ram) != self._day_plan_start_date

    def _auto_day_plan_rebuild_ready(self, ram: np.ndarray) -> bool:
        _day, hour, _minute = read_world_day_time(ram)
        scene = classify_scene_from_ram(ram)
        if not morning_scene_ready(scene, hour):
            self._pending_auto_day_plan_settle_frames = 0
            return False
        self._pending_auto_day_plan_settle_frames += 1
        return self._pending_auto_day_plan_settle_frames >= 10

    def _sync_day_plan_seed_item(self, ram: np.ndarray) -> np.ndarray:
        """Do not RAM-edit crop seeds in core day-plan logic.

        Seed retrieval belongs to EnsureCropSeedsTask, which visits the shed and
        only succeeds if stored seed stock exists. The crop task then cycles the
        real carry pair normally.
        """
        return ram

    def _auto_day_plan_should_rebuild_on_enable(self, ram: np.ndarray) -> bool:
        if not self.day_plan_auto or self._multi_day_enabled():
            return False
        if is_rainy_weather(ram):
            return False
        return ram_has_waterable_crops(ram, state_name=self.auto_day_plan_state_name)

    def prepare_for_enable(self) -> None:
        """Refresh active autonomous work from live RAM before bot control starts."""
        self.disable_reason = None
        if self.env is None:
            return
        ram = self.env.get_ram()
        if self.crop_enabled and self.crop_task_started and not self.crop_task_done:
            world = WorldState(frame=self.frame_count, ram=ram, info={}, obs=None)
            resume = getattr(self.crop_task, "resume_after_hotswap", None)
            if callable(resume):
                resume(world)
        if not self.day_plan_enabled:
            return
        if self._auto_day_plan_should_rebuild_on_enable(ram):
            print("[BOT] Day plan: hot-swap re-scan found dry crops; rebuilding")
            self._configure_day_plan(None)
            return
        if self.day_plan_started and not self.day_plan_done:
            world = WorldState(frame=self.frame_count, ram=ram, info={}, obs=None)
            resume = getattr(self.day_plan_task, "resume_after_hotswap", None)
            if callable(resume):
                resume(world)
            return
        if not self.day_plan_auto:
            return
        # When auto mode is active, always rebuild from current state
        if (
            self.day_plan_done
            or not self.day_plan_started
        ):
            self._configure_day_plan(None)

    def disable(self, reason: str):
        self.disable_reason = reason
        self.enabled = False
        print(f"[BOT] Disabled: {reason}")

    def force_end_day(self, reason: str, world: WorldState) -> bool:
        """Ask a multi-day planner to abandon daytime work and try to sleep."""
        if not self.day_plan_enabled:
            return False
        force = getattr(self.day_plan_task, "force_end_day", None)
        if not callable(force):
            return False
        if not force(world, reason):
            return False
        self.day_plan_started = True
        self.day_plan_done = False
        self.enabled = True
        self.disable_reason = None
        return True

    def get_goal_text(self) -> str:
        if self.power_on_enabled and not self.power_on_done:
            if self.power_on_started and self.power_on_task is not None:
                return f"Goal: {self.power_on_task.phase_text}"
            return "Goal: power-on (waiting)"
        if self.d1_handoff_enabled and not self.d1_handoff_done:
            if self.d1_handoff_started and self.d1_handoff_task is not None:
                snap = self.d1_handoff_task.progress_snapshot()
                phase = snap.phase_text or "running"
                return f"Goal: D1 handoff {phase}"
            return "Goal: D1 handoff (waiting)"
        if self.day_plan_enabled and not self.day_plan_done:
            if self.day_plan_started:
                return f"Goal: day plan {self.day_plan_task.phase_text} ({self.day_plan_task.progress_text})"
            return "Goal: day plan (waiting)"
        if self.fence_task_enabled and not self.fence_task_done:
            return "Goal: clear fences"
        if self.crop_enabled and not self.crop_task_done:
            if self.crop_task_started:
                return f"Goal: crop {self.crop_task.phase_text} ({self.crop_task.progress_text})"
            return "Goal: crop (waiting)"
        if self.grass_enabled and not self.grass_task_done:
            if self.grass_task_started:
                return f"Goal: grass {self.grass_task.phase_text} ({self.grass_task.progress_text})"
            return "Goal: grass (waiting)"
        if not self.clearer.startup_done:
            idx = self.clearer.startup_index
            if idx < len(self.clearer.startup_tasks):
                return f"Goal: {self.clearer.startup_tasks[idx].get('name', 'startup')}"
            return "Goal: startup"
        if self.clearer.current_target:
            t = self.clearer.current_target
            return f"Goal: {t.debris_type.name} at {t.tile}"
        return f"Goal: {self.clearer.state}"

    def get_action(self, game_state: GameState, obs: np.ndarray) -> np.ndarray:
        if not self.enabled or self.env is None:
            return np.zeros(12, dtype=np.int32)

        self.frame_count += 1
        ram = self.env.get_ram()
        world = WorldState(frame=self.frame_count, ram=ram, info={}, obs=obs)

        # Clean power-on: title → START → new diary → Spring D1 town gate.
        if self.power_on_enabled and not self.power_on_done and self.power_on_task is not None:
            if not self.power_on_started:
                self.power_on_task.reset(world)
                self.power_on_started = True
                print("[BOT] Power-on: title -> START -> new diary -> Spring D1")
            result = self.power_on_task.step(world)
            if result.action is not None:
                return result.action.action
            if result.status == TaskStatus.SUCCESS:
                print(
                    f"[BOT] Power-on complete ({self.power_on_task.progress_text}): "
                    f"{self.power_on_task.phase_text}"
                )
                self.power_on_done = True
                # Rebuild day plan from live Spring D1 RAM once controllable.
                if self.day_plan_enabled and not self.day_plan_started:
                    self._configure_day_plan(self.day_plan_sequence_name)
            elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or result.status.value
                print(f"[BOT] Power-on stopped ({reason})")
                self.power_on_done = True
                self.disable(f"Power-on stopped ({reason})")
            return np.zeros(12, dtype=np.int32)

        # Gate B / rr-5in: six talks + truck + outdoor intro + shed + sleep → D2.
        if self.d1_handoff_enabled and not self.d1_handoff_done and self.d1_handoff_task is not None:
            if not self.d1_handoff_started:
                if self.d1_handoff_task.can_start(world):
                    self.d1_handoff_task.reset(world)
                    self.d1_handoff_started = True
                    print("[BOT] D1 handoff: talks + truck + shed + sleep -> D2")
                else:
                    print("[BOT] D1 handoff: cannot start (skipping)")
                    self.d1_handoff_done = True
            if self.d1_handoff_started and not self.d1_handoff_done:
                result = self.d1_handoff_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    phase = self.d1_handoff_task.progress_snapshot().phase_text or "done"
                    print(f"[BOT] D1 handoff complete ({phase})")
                    self.d1_handoff_done = True
                    # Multi-day / auto plan must re-scan from D2 morning RAM.
                    if self.day_plan_enabled and not self.day_plan_started:
                        self._configure_day_plan(self.day_plan_sequence_name)
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] D1 handoff stopped ({reason})")
                    self.d1_handoff_done = True
                    self.disable(f"D1 handoff stopped ({reason})")
                return np.zeros(12, dtype=np.int32)

        if self.fence_task_enabled and not self.fence_task_done:
            if not self.fence_task_started:
                if self.fence_task.can_start(world):
                    self.fence_task.reset(world)
                    self.fence_task_started = True
                    print("[BOT] Fence clear: start")
                else:
                    self.fence_task_done = True
                    print("[BOT] Fence clear: missing recording")

            if self.fence_task_started and not self.fence_task_done:
                result = self.fence_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Fence clear: complete ({self.fence_task.cleared_count})")
                    self.fence_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Fence clear: stopped ({reason})")
                    self.fence_task_done = True
                if self.fence_task_only and self.fence_task_done:
                    self.disable("Fence-only complete")
                return np.zeros(12, dtype=np.int32)
        elif self.fence_task_only:
            self.disable("Fence-only complete")
            return np.zeros(12, dtype=np.int32)

        # Day plan mode: runs before map lock since it traverses multiple tilemaps
        if self.day_plan_enabled and self._pending_auto_day_plan_rebuild:
            if self._auto_day_plan_rebuild_ready(ram):
                print("[BOT] Day plan: new day settled; re-scan")
                self._configure_day_plan(None)
            return np.zeros(12, dtype=np.int32)

        if self.day_plan_enabled and not self.day_plan_done:
            if not self.day_plan_started:
                if self.day_plan_task.can_start(world):
                    self.day_plan_task.reset(world)
                    self.day_plan_started = True
                    self._day_plan_start_date = read_world_date(ram)
                    print("[BOT] Day plan: start")
                else:
                    self.day_plan_done = True
                    print("[BOT] Day plan: cannot start")

            if self.day_plan_started and not self.day_plan_done:
                ram = self._sync_day_plan_seed_item(ram)
                world = WorldState(frame=self.frame_count, ram=ram, info={}, obs=obs)
                result = self.day_plan_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Day plan: complete ({self.day_plan_task.progress_text})")
                    if self._should_rebuild_auto_day_plan(ram):
                        print("[BOT] Day plan: new day detected; waiting for morning")
                        self.day_plan_started = False
                        self.day_plan_done = False
                        self._pending_auto_day_plan_rebuild = True
                        self._pending_auto_day_plan_settle_frames = 0
                    else:
                        self.day_plan_done = True
                        self.disable(f"Day plan complete ({self.day_plan_task.progress_text})")
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Day plan: stopped ({reason})")
                    self.day_plan_done = True
                    self.disable(f"Day plan stopped ({reason})")
                return np.zeros(12, dtype=np.int32)

        # Lock to initial map after warmup
        tilemap = ram[ADDR_TILEMAP] if ADDR_TILEMAP < len(ram) else 0
        if not self.map_locked:
            self.map_hist[tilemap] = self.map_hist.get(tilemap, 0) + 1
            if self.frame_count >= 180:
                nonzero = {k: v for k, v in self.map_hist.items() if k != 0}
                self.initial_tilemap = max((nonzero or self.map_hist).items(), key=lambda kv: kv[1])[0]
                self.map_locked = True
                print(f"[BOT] Map locked: 0x{self.initial_tilemap:02X}")

        if self.map_locked and tilemap != self.initial_tilemap:
            self.disable(f"Map changed to 0x{tilemap:02X}")
            return np.zeros(12, dtype=np.int32)

        # Crop mode: detect plots, plant + water
        if self.crop_enabled and not self.crop_task_done:
            if not self.crop_task_started:
                if self.crop_task.can_start(world):
                    self.crop_task.reset(world)
                    self.crop_task_started = True
                    print("[BOT] Crop task: start")
                else:
                    self.crop_task_done = True
                    print("[BOT] Crop task: cannot start")

            if self.crop_task_started and not self.crop_task_done:
                result = self.crop_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Crop task: complete ({self.crop_task.progress_text})")
                    self.crop_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Crop task: stopped ({reason})")
                    self.crop_task_done = True
                if self.crop_task_done:
                    self.disable(f"Crop complete ({self.crop_task.progress_text})")
                return np.zeros(12, dtype=np.int32)

        # If grass mode, run grass task instead of (or after) clearing
        if self.grass_enabled and not self.grass_task_done:
            if not self.grass_task_started:
                if self.grass_task.can_start(world):
                    self.grass_task.reset(world)
                    self.grass_task_started = True
                    print("[BOT] Grass planter: start")
                else:
                    self.grass_task_done = True
                    print("[BOT] Grass planter: cannot start")

            if self.grass_task_started and not self.grass_task_done:
                result = self.grass_task.step(world)
                if result.action is not None:
                    return result.action.action
                if result.status == TaskStatus.SUCCESS:
                    print(f"[BOT] Grass planter: complete ({self.grass_task.progress_text})")
                    self.grass_task_done = True
                elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    reason = result.reason or result.status.value
                    print(f"[BOT] Grass planter: stopped ({reason})")
                    self.grass_task_done = True
                if self.grass_task_done:
                    self.disable(f"Grass complete ({self.grass_task.progress_text})")
                return np.zeros(12, dtype=np.int32)

        if not self.clear_task_done:
            if not self.clear_task_started:
                if self.clear_task.can_start(world):
                    self.clear_task.reset(world)
                    self.clear_task_started = True
                    print("[BOT] Farm clear: start")
                else:
                    self.clear_task_done = True
                    self.disable("Farm clear: nothing to clear")
                    return np.zeros(12, dtype=np.int32)

            result = self.clear_task.step(world)
            if result.action is not None:
                return result.action.action
            if result.status == TaskStatus.SUCCESS:
                print(
                    f"[BOT] Farm clear: complete ({self.clear_task.progress_text})"
                )
                self.clear_task_done = True
                reason_suffix = f" {result.reason}" if result.reason else ""
                self.disable(
                    f"Complete ({self.clearer.cleared_count} cleared{reason_suffix})"
                )
            elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or result.status.value
                print(f"[BOT] Farm clear: stopped ({reason})")
                self.clear_task_done = True
                self.disable(f"Farm clear stopped ({reason})")
            return np.zeros(12, dtype=np.int32)

        self.disable(f"Complete ({self.clearer.cleared_count} cleared)")
        return np.zeros(12, dtype=np.int32)

