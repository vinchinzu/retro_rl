"""Day-plan orchestrators (single-day and multi-day)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.recovery import RecoveryTask
from harvest.core.scene import (
    classify_scene_from_ram,
    morning_scene_ready,
    scene_indicates_ending,
)
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.tasks.farm_clearer import ADDR_TILEMAP, Point, Tool, make_action
from harvest.tasks.primitives import dismiss_dialogue_result
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
from harvest.planner.day_plan_decision import (
    DayPlanAdvisor,
    DayPlanDecision,
    DeferredPlan,
    auto_day_plan_decision,
)
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    is_farm_tilemap,
    is_house_tilemap,
    read_world_date,
    read_world_day_time,
)
from harvest.planner.day_plan_tasks import (
    EnsureCarryToolTask,
    ExitToFarmTask,
    GoToSleepTask,
    HOUSE_SLEEP_TRANSITION_TILEMAP,
    ReturnHomeTask,
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

    def __post_init__(self):
        self._reset_phase_lists()

    def reset(self, world: WorldState) -> None:
        self._reset_phase_lists()
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
        self._phase_results.append(
            {
                "phase": spec.phase,
                "kind": getattr(spec.kind, "value", str(spec.kind)),
                "status": status,
                "reason": reason,
                "step": int(self._step_count),
            }
        )

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
        return DayTaskFactory(
            seed_type=self.seed_type,
            tasks_dir=self.tasks_dir,
            state_name=self.state_name,
        ).make_task(spec, world)

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

    def _advance(self, world: WorldState, reason: str) -> None:
        """Move to next phase."""
        current = self._schedule.current_at(self._phase_index)
        phase_name = current.phase if current is not None else "?"
        print(f"[DAY_PLAN] {phase_name} -> {reason}")
        self._record_phase_result(current, "success", reason)
        if current is not None and current.phase in GO_HOME_TRIGGER_PHASES:
            self._mark_ready_to_go_home(current.phase)
            self._ensure_end_day_phases()
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

    def _skip_optional_money_route(self, reason: str) -> None:
        """Skip a contiguous optional shop/berry route after a cutoff or route miss."""
        skipped: List[str] = []
        while (
            self._phase_index < len(self._schedule.active)
            and self._schedule.active[self._phase_index].phase in OPTIONAL_MONEY_PHASES
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
                self._skip_optional_money_route(reason)
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
            self._advance(world, "SUCCESS")
            return self.step(world)
        elif result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            reason = result.reason or "unknown"
            print(f"[DAY_PLAN] Phase {spec.phase} {result.status.value.upper()}: {reason}")
            return self._handle_failed_phase(spec, result.status, reason, world)

        # Pass through RUNNING action
        if result.action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class MultiDayPlannerTask(Task):
    """Run day plans repeatedly, sleeping between days until a target date."""

    name: str = "multi_day_planner"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    until_season: int = 0
    until_day: int = 30
    target_days: Optional[int] = None
    max_days: int = 40
    morning_settle_frames: int = 10
    policy: DayPlannerPolicy = field(default_factory=DayPlannerPolicy)
    plan_advisor: Optional[DayPlanAdvisor] = None

    _phase: str = field(default="plan_day", init=False)
    _current_task: Optional[Task] = field(default=None, init=False)
    _days_completed: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _settle_count: int = field(default=0, init=False)
    _active_day: Tuple[int, int] = field(default=(0, 0), init=False)
    _last_day_decision: Optional[DayPlanDecision] = field(default=None, init=False)
    _last_day_phase_results: list[dict[str, object]] = field(default_factory=list, init=False)
    _last_day_deferred: list[dict[str, object]] = field(default_factory=list, init=False)
    _day_failures: list[dict[str, object]] = field(default_factory=list, init=False)
    _day_journal: list[dict[str, object]] = field(default_factory=list, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "plan_day"
        self._current_task = None
        self._days_completed = 0
        self._step_count = 0
        self._settle_count = 0
        self._active_day = read_world_date(world.ram)
        self._last_day_decision = None
        self._last_day_phase_results.clear()
        self._last_day_deferred.clear()
        self._day_failures.clear()
        self._day_journal.clear()

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        return self._phase.upper()

    @property
    def progress_text(self) -> str:
        season, day = self._active_day
        target = self.target_days if self.target_days is not None else self.max_days
        return f"date={season}:{day} completed={self._days_completed}/{target}"

    @property
    def last_day_decision(self) -> Optional[DayPlanDecision]:
        return self._last_day_decision

    @property
    def day_failures(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(item) for item in self._day_failures)

    @property
    def day_journal(self) -> tuple[dict[str, object], ...]:
        """One row per completed overnight with phase outcomes when available."""
        return tuple(dict(item) for item in self._day_journal)

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    @property
    def step_count(self) -> int:
        return self._step_count

    def progress_snapshot(self) -> ProgressSnapshot:
        child = self.current_task
        child_snap = task_progress_snapshot(child) if child is not None else None
        return ProgressSnapshot(
            task_name=self.__class__.__name__,
            phase_text=self.phase_text,
            step_count=self.step_count,
            details=(("days_completed", self._days_completed),),
            child=child_snap,
        )

    def _target_reached(self, ram: np.ndarray) -> bool:
        if self.target_days is not None:
            return self._days_completed >= self.target_days
        current = read_world_date(ram)
        return current > (self.until_season, self.until_day)

    def _build_day_task(self, world: WorldState) -> DayPlanTask:
        from harvest.planner.day_phase_types import day_planner_policy_for_season
        from harvest.planner.day_plan_status import resolve_seed_type_from_ram

        season, day = read_world_date(world.ram)
        day_policy = day_planner_policy_for_season(
            season,
            replace(self.policy, include_end_day=False),
        )
        resolved_seed = resolve_seed_type_from_ram(world.ram) or self.seed_type
        decision = auto_day_plan_decision(
            ram=world.ram,
            policy=day_policy,
            advisor=self.plan_advisor,
        )
        self._last_day_decision = decision
        phase_names = ", ".join(phase.phase for phase in decision.phases) or "none"
        if decision.deferred:
            deferred = "; ".join(
                f"{item.phase}:{item.reason}" for item in decision.deferred
            )
            print(
                f"[MULTI_DAY] Plan {season}:{day} seed={resolved_seed} "
                f"phases={phase_names} deferred={deferred}"
            )
        else:
            print(
                f"[MULTI_DAY] Plan {season}:{day} seed={resolved_seed} "
                f"phases={phase_names}"
            )
        return DayPlanTask(
            seed_type=resolved_seed,
            tasks_dir=self.tasks_dir,
            phase_sequence=list(decision.phases),
            policy=day_policy,
        )

    def _build_return_home_task(self) -> ReturnHomeTask:
        return ReturnHomeTask(tasks_dir=self.tasks_dir)

    def _build_sleep_task(self) -> GoToSleepTask:
        return GoToSleepTask(tasks_dir=self.tasks_dir)

    def _activate(self, phase: str, task: Task, world: WorldState) -> None:
        self._phase = phase
        self._current_task = task
        self._active_day = read_world_date(world.ram)
        print(f"[MULTI_DAY] Start {phase} date={self._active_day[0]}:{self._active_day[1]}")
        task.reset(world)

    def _record_day_failure(self, world: WorldState, status: TaskStatus | str, reason: str) -> None:
        season, day = read_world_date(world.ram)
        status_text = status.value if isinstance(status, TaskStatus) else str(status)
        row: dict[str, object] = {
            "season": int(season),
            "day": int(day),
            "phase": self._phase,
            "status": status_text,
            "reason": reason,
            "step": int(self._step_count),
        }
        self._day_failures.append(row)
        print(
            "[MULTI_DAY] Failed day checkpoint "
            f"date={season}:{day} phase={self._phase} status={status_text}: {reason}"
        )

    def _force_return_home_after_failure(
        self,
        world: WorldState,
        status: TaskStatus | str,
        reason: str,
    ) -> TaskResult:
        self._record_day_failure(world, status, reason)
        self._phase = "return_home"
        self._current_task = None
        self._active_day = read_world_date(world.ram)
        print("[MULTI_DAY] Forcing return_home after failed day work")
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def force_end_day(self, world: WorldState, reason: str) -> bool:
        """External watchdog hook: abandon current daytime work and try to sleep."""
        if self._target_reached(world.ram):
            return False
        if self._phase == "sleep":
            return False
        self._record_day_failure(world, "watchdog", reason)
        self._phase = "return_home"
        self._current_task = None
        self._active_day = read_world_date(world.ram)
        print(f"[MULTI_DAY] Watchdog forced return_home: {reason}")
        return True

    def _capture_day_plan_outcomes(self) -> None:
        """Snapshot DayPlanTask phase results before switching to return/sleep."""
        child = self._current_task
        if not isinstance(child, DayPlanTask):
            return
        self._last_day_phase_results = list(child.phase_results)
        self._last_day_deferred = [
            {
                "phase": item.phase,
                "reason": item.reason,
                "retry": item.retry,
            }
            for item in child.deferred_plans
        ]

    def _journal_day_complete(self, world: WorldState, *, sleep_reason: str) -> None:
        """Record phase outcomes for the day that just ended."""
        season, day = self._active_day
        end_season, end_day = read_world_date(world.ram)
        money = 0
        try:
            from harvest.core.ram_catalog import read_ram_value

            money = int(read_ram_value(world.ram, "money"))
        except Exception:
            money = 0
        # Prefer snapshot taken when plan_day finished; fall back to live task.
        self._capture_day_plan_outcomes()
        phase_results = list(self._last_day_phase_results)
        deferred = list(self._last_day_deferred)
        planned = []
        if self._last_day_decision is not None:
            planned = [p.phase for p in self._last_day_decision.phases]
        row = {
            "plan_season": int(season),
            "plan_day": int(day),
            "end_season": int(end_season),
            "end_day": int(end_day),
            "overnights_completed": int(self._days_completed + 1),
            "sleep_reason": sleep_reason,
            "money": money,
            "planned_phases": planned,
            "phase_results": phase_results,
            "deferred": deferred,
            "step": int(self._step_count),
        }
        self._day_journal.append(row)
        succeeded = [r["phase"] for r in phase_results if r.get("status") == "success"]
        skipped = [r["phase"] for r in phase_results if r.get("status") == "skipped"]
        print(
            f"[MULTI_DAY] Day journal {season}:{day} -> {end_season}:{end_day} "
            f"ok={succeeded or ['(none)']} skipped={skipped or ['(none)']} money={money}"
        )

    def _finish_sleep(self, world: WorldState, reason: str = "day advanced") -> TaskResult:
        self._journal_day_complete(world, sleep_reason=reason)
        self._days_completed += 1
        season, day = read_world_date(world.ram)
        print(
            f"[MULTI_DAY] Completed sleep {self._days_completed}: "
            f"now date={season}:{day} ({reason})"
        )
        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene) or reason == "ending reached":
            return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")
        # Always settle the morning after sleep so the final overnight also
        # ends on a stable house/farm scene (M3 acceptance), not mid-fade.
        self._phase = "settle_morning"
        self._current_task = None
        self._settle_count = 0
        self._active_day = read_world_date(world.ram)
        if self._target_reached(world.ram):
            print("[MULTI_DAY] Settling morning scene before target success")
        else:
            print("[MULTI_DAY] Settling morning scene before next day plan")
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _step_settle_morning(self, world: WorldState) -> TaskResult:
        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene):
            return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")
        _day, hour, _minute = read_world_day_time(world.ram)
        if morning_scene_ready(scene, hour):
            self._settle_count += 1
            if self._settle_count >= self.morning_settle_frames:
                self._active_day = read_world_date(world.ram)
                print(f"[MULTI_DAY] Morning settled: {scene.summary()}")
                if self._target_reached(world.ram):
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        reason="target date reached",
                    )
                self._phase = "plan_day"
                self._current_task = None
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"morning settle {self._settle_count}/{self.morning_settle_frames}",
            )

        self._settle_count = 0
        if scene.needs_input_dismiss:
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason=f"morning {scene.mode.value}",
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"waiting for morning: {scene.summary()}",
        )

    def _observe_running_side_effects(
        self,
        world: WorldState,
        result: TaskResult,
    ) -> TaskResult:
        """Catch overnight/ending transitions while a child task is still RUNNING."""
        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene):
            if self._phase == "sleep":
                return self._finish_sleep(world, "ending reached")
            return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")
        if self._phase in {"plan_day", "return_home"} and read_world_date(world.ram) != self._active_day:
            if self._phase == "plan_day":
                self._capture_day_plan_outcomes()
            self._record_day_failure(
                world,
                "overnight",
                f"day advanced during {self._phase}: {result.reason or 'running'}",
            )
            return self._finish_sleep(world, f"advanced during {self._phase}")
        if (
            self._phase == "return_home"
            and int(world.ram[ADDR_TILEMAP]) == HOUSE_SLEEP_TRANSITION_TILEMAP
        ):
            self._phase = "sleep"
            self._current_task = None
            print("[MULTI_DAY] Return home entered sleep transition; switching to sleep")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        return result

    def _start_phase(self, world: WorldState) -> TaskResult:
        if self._target_reached(world.ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason="target date reached")
        if self._days_completed >= self.max_days:
            return TaskResult(status=TaskStatus.FAILURE, reason="max_days reached")

        if self._phase == "plan_day":
            self._activate("plan_day", self._build_day_task(world), world)
        elif self._phase == "return_home":
            self._activate("return_home", self._build_return_home_task(), world)
        elif self._phase == "sleep":
            self._activate("sleep", self._build_sleep_task(), world)
        else:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"unknown planner phase {self._phase}",
            )
        return self._current_task.step(world)

    def _handle_result(self, world: WorldState, result: TaskResult) -> TaskResult:
        if result.status == TaskStatus.RUNNING:
            return self._observe_running_side_effects(world, result)
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            reason = result.reason or "unknown"
            scene = classify_scene_from_ram(world.ram)
            if scene_indicates_ending(scene) or "ending" in reason:
                return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")
            if self._phase == "plan_day":
                self._capture_day_plan_outcomes()
                return self._force_return_home_after_failure(world, result.status, reason)
            if self._phase == "return_home":
                tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
                if read_world_date(world.ram) != self._active_day:
                    self._record_day_failure(
                        world,
                        result.status,
                        f"return_home became overnight transition: {reason}",
                    )
                    return self._finish_sleep(world, "advanced during return_home")
                if tilemap == HOUSE_SLEEP_TRANSITION_TILEMAP:
                    self._record_day_failure(
                        world,
                        result.status,
                        f"return_home entered sleep transition: {reason}",
                    )
                    self._phase = "sleep"
                    self._current_task = None
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
                if is_house_tilemap(tilemap):
                    self._record_day_failure(
                        world,
                        result.status,
                        f"return_home failed in house: {reason}",
                    )
                    self._phase = "sleep"
                    self._current_task = None
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
            return TaskResult(
                status=result.status,
                reason=f"{self._phase} failed: {reason}",
            )

        if self._phase == "plan_day":
            self._capture_day_plan_outcomes()
            self._phase = "return_home"
            self._current_task = None
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if self._phase == "return_home":
            self._phase = "sleep"
            self._current_task = None
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if self._phase == "sleep":
            reason = result.reason or "day advanced"
            return self._finish_sleep(world, reason)

        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"unknown planner phase {self._phase}",
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._phase == "settle_morning":
            return self._step_settle_morning(world)
        if self._current_task is None:
            return self._handle_result(world, self._start_phase(world))

        return self._handle_result(world, self._current_task.step(world))
