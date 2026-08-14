"""Day-plan orchestrators (single-day and multi-day).

``MultiDayPlannerTask`` lives in :mod:`harvest.planner.multi_day_planner` and
is re-exported here for a stable import path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.recovery import RecoveryTask
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.tasks.nav import make_action
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    Tool,
)
from harvest.planner.day_phase_types import PhaseKind, PhaseSpec, SKIP_MAP_LOCK_KINDS
from harvest.planner.day_plan_phases import (
    DayPlannerPolicy,
    DAY1_PHASES,
    EXIT_TO_FARM_PHASE,
    GO_HOME_TRIGGER_PHASES,
    GO_TO_SLEEP_PHASE,
    OPTIONAL_MONEY_PHASES,
    RETURN_HOME_PHASE,
    build_outdoor_day_phases_from_ram,
)
from harvest.planner.day_plan_decision import DeferredPlan
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    is_farm_tilemap,
    read_world_day_time,
)
from harvest.planner.day_plan_tasks import (
    EnsureCarryToolTask,
    ExitToFarmTask,
)
from harvest.planner.day_task_factory import DayTaskFactory


@dataclass
class PhaseSchedule:
    """Immutable planned sequence plus the active runtime sequence."""

    planned: tuple[PhaseSpec, ...]
    active: list[PhaseSpec]

    @classmethod
    def from_phases(cls, phases: Sequence[PhaseSpec]) -> PhaseSchedule:
        planned = tuple(phases)
        return cls(planned=planned, active=list(planned))

    @classmethod
    def from_sequence(
        cls,
        phase_sequence: Optional[List[PhaseSpec]],
        default: List[PhaseSpec],
    ) -> PhaseSchedule:
        phases = phase_sequence if phase_sequence is not None else default
        return cls.from_phases(phases)

    def current_at(self, index: int) -> Optional[PhaseSpec]:
        if index < len(self.active):
            return self.active[index]
        return None

    def splice_at(self, index: int, replacement: Sequence[PhaseSpec]) -> None:
        self.active = self.active[:index] + list(replacement) + self.active[index + 1:]

    def append(self, phases: Sequence[PhaseSpec]) -> None:
        self.active.extend(phases)

    def has_end_day_phases(self) -> bool:
        return any(
            phase.phase in {"RETURN_HOME", "GO_TO_SLEEP"} for phase in self.active
        )


@dataclass
class DayPlanTask(Task):
    """Orchestrator: steps through phase sequence, creating sub-tasks on demand."""

    name: str = "day_plan"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    phase_sequence: Optional[List[PhaseSpec]] = None
    state_name: Optional[str] = None
    policy: DayPlannerPolicy = field(default_factory=DayPlannerPolicy)

    _schedule: PhaseSchedule = field(
        default_factory=lambda: PhaseSchedule.from_phases([]),
        init=False,
    )
    _phase_index: int = field(default=0, init=False)
    _current_task: Optional[Task] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _skip_map_lock: bool = field(default=False, init=False)
    _end_day_appended: bool = field(default=False, init=False)
    _ready_to_go_home: bool = field(default=False, init=False)
    _recovery_task: Optional[Task] = field(default=None, init=False)
    _recovering_spec: Optional[PhaseSpec] = field(default=None, init=False)
    _recovery_original_reason: str = field(default="", init=False)
    _recovery_attempted_phases: set[tuple[int, str]] = field(default_factory=set, init=False)
    _deferred_plans: list[DeferredPlan] = field(default_factory=list, init=False)
    _phase_results: list[dict[str, object]] = field(default_factory=list, init=False)
    _task_factory: DayTaskFactory = field(init=False, repr=False)

    def __post_init__(self):
        self._reset_phase_lists()
        self._reset_task_factory()

    def reset(self, world: WorldState) -> None:
        self._reset_phase_lists()
        self._reset_task_factory()
        self._phase_index = 0
        self._current_task = None
        self._step_count = 0
        self._skip_map_lock = False
        self._end_day_appended = False
        self._ready_to_go_home = False
        self._recovery_task = None
        self._recovering_spec = None
        self._recovery_original_reason = ""
        self._recovery_attempted_phases.clear()
        self._deferred_plans.clear()
        self._phase_results.clear()

    def _reset_phase_lists(self) -> None:
        self._schedule = PhaseSchedule.from_sequence(self.phase_sequence, DAY1_PHASES)

    def _reset_task_factory(self) -> None:
        """Start each day with one shared builder context.

        ``DayTaskFactory`` owns the per-frame ``WorldContext`` cache.  Keeping
        it for the lifetime of a day lets phase builders share observations
        while reset deliberately drops any cache from the previous morning.
        """
        self._task_factory = DayTaskFactory(
            seed_type=self.seed_type,
            tasks_dir=self.tasks_dir,
            state_name=self.state_name,
            policy=self.policy,
        )

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def skip_map_lock(self) -> bool:
        """True when a recorded task is active (may change tilemap)."""
        return self._skip_map_lock

    @property
    def phase_text(self) -> str:
        current = self._schedule.current_at(self._phase_index)
        return current.phase if current is not None else "DONE"

    @property
    def progress_text(self) -> str:
        return f"phase={self._phase_index + 1}/{len(self._schedule.active)} step={self._step_count}"

    @property
    def deferred_plans(self) -> tuple[DeferredPlan, ...]:
        return tuple(self._deferred_plans)

    @property
    def phase_results(self) -> tuple[dict[str, object], ...]:
        """Per-phase outcomes for the current day (success / skipped / failed)."""
        return tuple(dict(row) for row in self._phase_results)

    def _record_phase_result(
        self,
        spec: PhaseSpec | None,
        status: str,
        reason: str = "",
    ) -> None:
        if spec is None:
            return
        row: dict[str, object] = {
            "phase": spec.phase,
            "kind": getattr(spec.kind, "value", str(spec.kind)),
            "status": status,
            "reason": reason,
            "step": int(self._step_count),
        }
        # rr-53g: surface harvest ship counts in the day journal when present.
        task = self._current_task
        if task is not None and hasattr(task, "shipped_count"):
            try:
                row["shipped_count"] = int(getattr(task, "shipped_count"))
                row["harvested_count"] = int(getattr(task, "harvested_count", 0))
            except Exception:
                pass
        self._phase_results.append(row)

    @property
    def phases(self) -> tuple[PhaseSpec, ...]:
        return self._schedule.planned

    @property
    def runtime_phases(self) -> tuple[PhaseSpec, ...]:
        return tuple(self._schedule.active)

    @property
    def first_phase(self) -> Optional[PhaseSpec]:
        return self._schedule.planned[0] if self._schedule.planned else None

    @property
    def current_phase(self) -> Optional[PhaseSpec]:
        return self._schedule.current_at(self._phase_index)

    @property
    def current_task(self) -> Optional[Task]:
        return self._recovery_task or self._current_task

    @property
    def phase_index(self) -> int:
        return self._phase_index

    @property
    def step_count(self) -> int:
        return self._step_count

    def progress_snapshot(self) -> ProgressSnapshot:
        child = self.current_task
        child_snap = task_progress_snapshot(child) if child is not None else None
        return ProgressSnapshot(
            task_name=self.__class__.__name__,
            phase_text=self.phase_text,
            phase_index=self.phase_index,
            step_count=self.step_count,
            child=child_snap,
        )

    def _make_task(self, spec: PhaseSpec, world: WorldState) -> Optional[Task]:
        return self._task_factory.make_task(spec, world)

    def resume_after_hotswap(self, world: WorldState) -> None:
        task = self._recovery_task or self._current_task
        if task is None:
            return
        resume = getattr(task, "resume_after_hotswap", None)
        if callable(resume):
            resume(world)

    def _mark_ready_to_go_home(self, source: str) -> None:
        if self._ready_to_go_home:
            return
        self._ready_to_go_home = True
        print(f"[DAY_PLAN] Ready to go home (from {source})")

    def _ensure_end_day_phases(self) -> None:
        """Append return-home/sleep once when the go-home flag is set."""
        if self._end_day_appended or not self.policy.include_end_day:
            return
        if self._schedule.has_end_day_phases():
            self._end_day_appended = True
            return
        self._schedule.append([RETURN_HOME_PHASE, GO_TO_SLEEP_PHASE])
        self._end_day_appended = True
        print("[DAY_PLAN] Appending end-day route after go-home flag")

    def _splice_plant_after_shop(self, world: WorldState) -> None:
        """rr-20w.1: outdoor plan expands at 06:08 before the bag exists."""
        from harvest.planner.day_plan_phases import pocket_plant_phases
        from harvest.planner.world_probe import WorldProbe

        remaining = [phase.phase for phase in self._schedule.active[self._phase_index + 1 :]]
        if any(name in remaining for name in ("CROP_ESTABLISH", "CLEAR_PLOT")):
            return
        if not self.policy.include_planting:
            return
        probe = WorldProbe.from_inputs(ram=world.ram, state_name=self.state_name)
        _day, hour, _minute = probe.day_time()
        if hour >= self.policy.late_water_hour:
            return
        if not probe.has_seasonal_plantable_seeds():
            return
        planted = [
            PhaseSpec(
                phase.phase,
                phase.kind,
                dict(phase.params),
                failure_policy="optional",
                contract=phase.contract,
            )
            for phase in pocket_plant_phases()
        ]
        # Replace the leftover whole-farm CLEAR — pocket clear is the plant path.
        tail = [
            phase
            for phase in self._schedule.active[self._phase_index + 1 :]
            if phase.phase != "CLEAR_FIELD"
        ]
        self._schedule.active = (
            self._schedule.active[: self._phase_index + 1] + planted + tail
        )
        names = ", ".join(phase.phase for phase in planted)
        print(f"[DAY_PLAN] Spliced post-shop pocket plant: {names}")

    def _advance(self, world: WorldState, reason: str) -> None:
        """Move to next phase after real work success."""
        current = self._schedule.current_at(self._phase_index)
        phase_name = current.phase if current is not None else "?"
        print(f"[DAY_PLAN] {phase_name} -> {reason}")
        self._record_phase_result(current, "success", reason)
        if current is not None and current.phase == "BUY_SEEDS":
            self._splice_plant_after_shop(world)
        if current is not None and current.phase in GO_HOME_TRIGGER_PHASES:
            self._mark_ready_to_go_home(current.phase)
            self._ensure_end_day_phases()
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False

    def _advance_no_work(self, world: WorldState, reason: str) -> None:
        """Advance after intentional no-op SUCCESS (journal status=no_work)."""
        current = self._schedule.current_at(self._phase_index)
        phase_name = current.phase if current is not None else "?"
        print(f"[DAY_PLAN] {phase_name} -> no_work ({reason})")
        self._record_phase_result(current, "no_work", reason)
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False

    def _expand_dynamic_phase(self, spec: PhaseSpec, world: WorldState) -> bool:
        if spec.kind != PhaseKind.DYNAMIC_OUTDOOR_PLAN:
            return False
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if not is_farm_tilemap(tilemap):
            self._schedule.splice_at(
                self._phase_index,
                [EXIT_TO_FARM_PHASE, spec],
            )
            print("[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN deferred until farm tilemap")
            return True

        expanded = build_outdoor_day_phases_from_ram(world.ram, policy=self.policy, state_name=self.state_name)
        replacement = expanded if expanded else []
        self._schedule.splice_at(self._phase_index, replacement)
        if expanded:
            names = ", ".join(phase.phase for phase in expanded)
            print(f"[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN expanded to: {names}")
        else:
            print("[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN expanded to no-op")
        return True

    def _optional_route_group(self, phase_name: str) -> frozenset[str]:
        """Return the contiguous optional-route group for a failed phase.

        Berry forage, seed shop, and chicken sale are independent money routes.
        Failing one must not cascade-skip the others (e.g. bush thrash must not
        cancel BUY_SEEDS when the wallet still covers potato).
        """
        from harvest.planner.day_phase_catalog import (
            OPTIONAL_BERRY_PHASES,
            OPTIONAL_CHICKEN_SALE_PHASES,
            OPTIONAL_SHOP_PHASES,
        )

        if phase_name in OPTIONAL_BERRY_PHASES:
            return OPTIONAL_BERRY_PHASES
        if phase_name in OPTIONAL_SHOP_PHASES:
            return OPTIONAL_SHOP_PHASES
        if phase_name in OPTIONAL_CHICKEN_SALE_PHASES:
            return OPTIONAL_CHICKEN_SALE_PHASES
        return OPTIONAL_MONEY_PHASES

    def _skip_optional_money_route(self, reason: str, *, group: frozenset[str] | None = None) -> None:
        """Skip a contiguous optional money sub-route after a cutoff or route miss."""
        route_group = group
        if route_group is None and self._phase_index < len(self._schedule.active):
            route_group = self._optional_route_group(
                self._schedule.active[self._phase_index].phase
            )
        if route_group is None:
            route_group = OPTIONAL_MONEY_PHASES
        skipped: List[str] = []
        while (
            self._phase_index < len(self._schedule.active)
            and self._schedule.active[self._phase_index].phase in route_group
        ):
            skipped_spec = self._schedule.active[self._phase_index]
            skipped.append(skipped_spec.phase)
            self._record_deferred_phase(skipped_spec, reason)
            self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False
        if skipped:
            print(f"[DAY_PLAN] Skipping optional route after failure ({reason}): {', '.join(skipped)}")

    def _record_deferred_phase(self, spec: PhaseSpec, reason: str, *, retry: str = "tomorrow") -> None:
        deferred = DeferredPlan.from_phase(spec, reason, retry=retry)
        key = (deferred.phase, deferred.reason, deferred.retry)
        if any((item.phase, item.reason, item.retry) == key for item in self._deferred_plans):
            return
        self._deferred_plans.append(deferred)
        print(f"[DAY_PLAN] Deferred {deferred.phase} until {retry}: {reason}")

    def _skip_failed_phase(self, spec: PhaseSpec, reason: str) -> None:
        """Skip an optional failed phase and avoid optional berry/shop follow-up after harvest trouble."""
        print(f"[DAY_PLAN] Skipping failed phase {spec.phase}: {reason}")
        self._record_phase_result(spec, "skipped", reason)
        self._record_deferred_phase(spec, reason)
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False
        if spec.kind == PhaseKind.HARVEST:
            self._skip_optional_money_route(reason)

    def _failure_policy(self, spec: PhaseSpec) -> str:
        """Return the phase failure policy, keeping legacy money phases optional."""
        if spec.phase in OPTIONAL_MONEY_PHASES:
            return "optional"
        return getattr(spec, "failure_policy", "required") or "required"

    def _recovery_phase_key(self, spec: PhaseSpec) -> tuple[int, str]:
        return self._phase_index, spec.phase

    def _make_recovery_task(
        self,
        spec: PhaseSpec,
        status: TaskStatus,
        reason: str,
        world: WorldState,
    ) -> Task:
        if spec.phase == "CROP_WATER" and reason == "watering can not in carry pair":
            return EnsureCarryToolTask(
                name="recover_crop_water_watering_can",
                tool_id=int(Tool.WATERING_CAN),
                tasks_dir=self.tasks_dir,
            )
        return RecoveryTask(
            name=f"recover_{spec.phase.lower()}",
            route_to_target_factory=lambda: ExitToFarmTask(tasks_dir=self.tasks_dir),
        )

    def _start_recovery(
        self,
        spec: PhaseSpec,
        status: TaskStatus,
        reason: str,
        world: WorldState,
    ) -> TaskResult:
        key = self._recovery_phase_key(spec)
        self._recovery_attempted_phases.add(key)
        self._current_task = None
        self._skip_map_lock = False
        self._recovering_spec = spec
        self._recovery_original_reason = reason
        self._recovery_task = self._make_recovery_task(spec, status, reason, world)
        self._recovery_task.reset(world)
        print(f"[DAY_PLAN] Recovering before aborting required phase {spec.phase}: {reason}")
        return self._step_recovery(world)

    def _clear_recovery(self) -> None:
        self._recovery_task = None
        self._recovering_spec = None
        self._recovery_original_reason = ""

    def _phase_target_satisfied_after_recovery(self, spec: PhaseSpec, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if spec.kind != PhaseKind.DIRECTIONAL_TRANSITION:
            return False

        target = spec.params.get("target_tilemap")
        if target is not None:
            target = int(target)
            if tilemap == target:
                return True
            if is_farm_tilemap(tilemap) and is_farm_tilemap(target):
                return True

        for candidate in spec.params.get("target_tilemaps") or ():
            candidate = int(candidate)
            if tilemap == candidate:
                return True
            if is_farm_tilemap(tilemap) and is_farm_tilemap(candidate):
                return True
        return False

    def _step_recovery(self, world: WorldState) -> TaskResult:
        if self._recovery_task is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="recovery task missing")

        spec = self._recovering_spec
        phase_name = spec.phase if spec is not None else "unknown"
        result = self._recovery_task.step(world)
        if result.status == TaskStatus.RUNNING:
            if result.action is not None:
                return result
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=result.reason)

        original_reason = self._recovery_original_reason or "unknown"
        if result.status == TaskStatus.SUCCESS:
            print(f"[DAY_PLAN] Recovery complete for {phase_name}: {result.reason or 'success'}")
            if spec is not None and self._phase_target_satisfied_after_recovery(spec, world):
                self._phase_index += 1
            self._clear_recovery()
            self._current_task = None
            self._skip_map_lock = False
            return self.step(world)

        self._clear_recovery()
        return TaskResult(
            status=result.status,
            reason=(
                f"required phase {phase_name} failed after recovery: "
                f"{original_reason}; recovery {result.status.value}: {result.reason or 'unknown'}"
            ),
        )

    def _handle_failed_phase(
        self,
        spec: PhaseSpec,
        status: TaskStatus,
        reason: str,
        world: WorldState,
    ) -> TaskResult:
        policy = self._failure_policy(spec)
        if policy in {"optional", "opportunistic"}:
            if spec.phase in OPTIONAL_MONEY_PHASES:
                self._skip_optional_money_route(
                    reason, group=self._optional_route_group(spec.phase)
                )
            else:
                self._skip_failed_phase(spec, reason)
            return self.step(world)
        if reason != "no task":
            key = self._recovery_phase_key(spec)
            if key not in self._recovery_attempted_phases:
                return self._start_recovery(spec, status, reason, world)
            return TaskResult(
                status=status,
                reason=f"required phase {spec.phase} failed after recovery: {reason}",
            )
        return TaskResult(
            status=status,
            reason=f"required phase {spec.phase} failed: {reason}",
        )

    def _append_late_end_day_if_needed(self, world: WorldState) -> bool:
        """Append return-home/sleep for late clock or explicit go-home flag."""
        if self._end_day_appended or not self.policy.include_end_day:
            return False
        if self._schedule.has_end_day_phases():
            self._end_day_appended = True
            return False
        _day, hour, _minute = read_world_day_time(world.ram)
        if not self._ready_to_go_home and hour < self.policy.late_water_hour:
            return False
        self._schedule.append([RETURN_HOME_PHASE, GO_TO_SLEEP_PHASE])
        self._end_day_appended = True
        reason = "go-home flag" if self._ready_to_go_home else "late clock"
        print(f"[DAY_PLAN] Appending end-day route ({reason})")
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1

        if self._recovery_task is not None:
            return self._step_recovery(world)

        # All phases complete
        if self._phase_index >= len(self._schedule.active):
            if self._append_late_end_day_if_needed(world):
                return self.step(world)
            return TaskResult(status=TaskStatus.SUCCESS, reason="day plan complete")

        spec = self._schedule.active[self._phase_index]
        if self._expand_dynamic_phase(spec, world):
            return self.step(world)

        # Create sub-task if needed
        if self._current_task is None:
            task = self._make_task(spec, world)
            if task is None:
                reason = "no task"
                print(f"[DAY_PLAN] Phase {spec.phase} unavailable: {reason}")
                return self._handle_failed_phase(spec, TaskStatus.FAILURE, reason, world)
            task.reset(world)
            self._current_task = task
            self._skip_map_lock = (
                isinstance(spec.kind, PhaseKind) and spec.kind in SKIP_MAP_LOCK_KINDS
            )
            print(
                f"[DAY_PLAN] Starting phase {self._phase_index + 1}/{len(self._schedule.active)}: "
                f"{spec.phase} ({spec.kind})"
            )

        # Step the sub-task
        result = self._current_task.step(world)

        if result.status == TaskStatus.SUCCESS:
            reason = result.reason or "SUCCESS"
            # Crop no-ops (no dry plots / no seeds) must not count as crop work.
            try:
                from harvest.tasks.water_refill import is_no_work_reason

                if is_no_work_reason(reason):
                    self._advance_no_work(world, reason)
                    return self.step(world)
            except Exception:
                pass
            self._advance(world, reason if reason != "SUCCESS" else "SUCCESS")
            return self.step(world)
        elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            reason = result.reason or "unknown"
            print(f"[DAY_PLAN] Phase {spec.phase} {result.status.value.upper()}: {reason}")
            return self._handle_failed_phase(spec, result.status, reason, world)

        # Pass through RUNNING action
        if result.action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


# Stable re-export for runtime importers.
from harvest.planner.multi_day_planner import MultiDayPlannerTask  # noqa: E402

__all__ = ["PhaseSchedule", "DayPlanTask", "MultiDayPlannerTask"]
