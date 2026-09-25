"""Day-plan orchestrators (single-day and multi-day).

``MultiDayPlannerTask`` lives in :mod:`harvest.planner.multi_day_planner` and
is re-exported here for a stable import path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.recovery import RecoveryTask
from harvest.core.task_progress import (
    GOAL_STALL_FRAMES,
    MOTION_STALL_FRAMES,
    ProgressSnapshot,
    stalled,
    task_progress_snapshot,
)
from harvest.core.world_context import WorldContext
from harvest.tasks.nav import make_action
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    Tool,
)
from harvest.tasks.water_refill import is_no_work_reason
from harvest.planner.day_phase_types import (
    ACQUIRE_TOOL_KINDS,
    PhaseKind,
    PhaseSpec,
    SKIP_MAP_LOCK_KINDS,
    tool_tags_from_ram,
)
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
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.planner.day_plan_tasks import (
    EnsureCarryToolTask,
    ExitToFarmTask,
)
from harvest.core.shipping_credit import shipping_scene_needs_dismiss
from harvest.tasks.primitives import dismiss_dialogue_result


# Carry tags a mid-day phase cannot go and get for itself. Tools come off a
# shed shelf via EnsureCarryToolTask; a seed bag needs the shop.
_UNFETCHABLE_TOOL_TAGS = frozenset({"seed"})
# One external step used to recurse. Cap the loop so a repeated D2 child
# cannot spin the stack or the frame.
_STEP_BUDGET = 128


def _tag_d2_spec(spec: PhaseSpec) -> PhaseSpec:
    """Copy a live D2 child so catalog singletons stay untagged."""
    params = dict(spec.params)
    params["d2_expanded"] = True
    return PhaseSpec(
        spec.phase,
        spec.kind,
        params,
        failure_policy=spec.failure_policy,
        contract=spec.contract,
    )


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
    _map_lock_exits: set = field(default_factory=set, init=False)
    _extra_establishes: int = field(default=0, init=False)
    _end_day_appended: bool = field(default=False, init=False)
    _ready_to_go_home: bool = field(default=False, init=False)
    _recovery_task: Optional[Task] = field(default=None, init=False)
    _recovering_spec: Optional[PhaseSpec] = field(default=None, init=False)
    _recovery_original_reason: str = field(default="", init=False)
    _recovery_attempted_phases: set[tuple[int, str]] = field(default_factory=set, init=False)
    _deferred_plans: list[DeferredPlan] = field(default_factory=list, init=False)
    _phase_results: list[dict[str, object]] = field(default_factory=list, init=False)
    _world_context: WorldContext = field(init=False, repr=False)
    _build_ctx: TaskBuildContext = field(init=False, repr=False)
    _d2_last_phase: str = field(default="", init=False)
    _d2_plot_attempted: bool = field(default=False, init=False)
    _d2_prev: object = field(default=None, init=False)
    _d2_unobs: int = field(default=0, init=False)
    _d2_fails: dict = field(default_factory=dict, init=False)
    _d2_motion_seen: object = field(default=None, init=False)
    _d2_motion_at: int = field(default=0, init=False)
    _d2_goal_key: object = field(default=None, init=False)
    _d2_goal_at: int = field(default=0, init=False)
    _d2_status: object = field(default=None, init=False)

    def __post_init__(self):
        self._reset_phase_lists()
        self._reset_build_context()
        self._reset_d2_cursor()

    def reset(self, world: WorldState) -> None:
        self._reset_phase_lists()
        self._reset_build_context()
        self._reset_d2_cursor()
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
        self._map_lock_exits.clear()
        self._extra_establishes = 0
        self._deferred_plans.clear()
        self._phase_results.clear()

    def _reset_phase_lists(self) -> None:
        self._schedule = PhaseSchedule.from_sequence(self.phase_sequence, DAY1_PHASES)

    def _reset_build_context(self) -> None:
        """One WorldContext per day. Reset drops the previous morning's cache."""
        self._world_context = WorldContext()
        self._build_ctx = TaskBuildContext(
            seed_type=self.seed_type,
            tasks_dir=self.tasks_dir,
            state_name=self.state_name,
            policy=self.policy,
            world_context=self._world_context,
        )

    def _reset_d2_cursor(self) -> None:
        self._d2_last_phase = ""
        self._d2_plot_attempted = False
        self._d2_prev = None
        self._d2_unobs = 0
        self._d2_fails = {}
        self._d2_motion_seen = None
        self._d2_motion_at = 0
        self._d2_goal_key = None
        self._d2_goal_at = 0
        self._d2_status = None

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
        world: WorldState | None = None,
        extra: dict | None = None,
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
        # D2 completion must prove the bin deposit itself, rather than infer
        # it later from a 5pm dialog or money RAM.  MountainGrapeShipTask
        # retains the before/after accumulator values until this row is made.
        if (
            world is not None
            and spec.phase == "MOUNTAIN_BERRY"
            and status == "success"
            and int(getattr(task, "shipped_count", 0) or 0) > 0
        ):
            from harvest.core.ram_catalog import read_ram_value

            row["shipping_deposit"] = {
                "season": int(read_ram_value(world.ram, "season") or 0),
                "day": int(read_ram_value(world.ram, "day") or 0),
                "hour": int(read_ram_value(world.ram, "hour") or 0),
                "minute": int(read_ram_value(world.ram, "minute") or 0),
                "shipping_money_before": int(getattr(task, "_shipping_before", 0)),
                "shipping_money_after": int(getattr(task, "_shipping_after", 0)),
            }
        if extra:
            row.update(extra)
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
        self._world_context.bind(world)
        return build_phase_task(self._build_ctx, spec, world)

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
        from harvest.planner.d2_work import d2_post_shop_work_phases, leftover_already_queued
        from harvest.planner.world_probe import WorldProbe

        remaining = [phase.phase for phase in self._schedule.active[self._phase_index + 1 :]]
        if leftover_already_queued(remaining) or any(
            name in remaining for name in ("CROP_ESTABLISH", "CLEAR_PLOT", "D2_FARM_CLEAR")
        ):
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
                failure_policy=phase.failure_policy,
                contract=phase.contract,
            )
            for phase in d2_post_shop_work_phases()
        ]
        # Replace the leftover whole-farm CLEAR — pocket + quota smash is the path.
        tail = [
            phase
            for phase in self._schedule.active[self._phase_index + 1 :]
            if phase.phase != "CLEAR_FIELD"
        ]
        self._schedule.active = (
            self._schedule.active[: self._phase_index + 1] + planted + tail
        )
        names = ", ".join(phase.phase for phase in planted)
        print(f"[DAY_PLAN] Spliced post-shop D2 work: {names}")

    def _splice_second_establish(self, world: WorldState) -> None:
        """Replant the other pocket ring the same day when a bag is left.

        A harvest day empties both rings at once, but the establish pipeline
        resolves exactly one ring target, so the second one idled until the
        next shop trip. With BUY_SEEDS now carrying a bag per waiting ring,
        the limit should be seed in the pocket, not the phase table.
        """
        from harvest.core.ram_catalog import read_ram_value
        from harvest.maps.farm_pond import (
            POCKET_PLANT_CENTERS,
            pocket_plant_targets,
        )
        from harvest.planner.day_phase_catalog import CROP_ESTABLISH_PHASE

        if self._extra_establishes >= len(POCKET_PLANT_CENTERS) - 1:
            return
        if not self.policy.include_planting:
            return
        remaining = [phase.phase for phase in self._schedule.active[self._phase_index + 1 :]]
        if "CROP_ESTABLISH" in remaining:
            return
        try:
            wanted = len(pocket_plant_targets(world.ram))
            bags = int(read_ram_value(world.ram, "potato_seeds") or 0)
        except Exception:
            return
        if wanted <= 0 or bags <= 0:
            return
        self._extra_establishes += 1
        self._schedule.active = (
            self._schedule.active[: self._phase_index + 1]
            + [CROP_ESTABLISH_PHASE]
            + self._schedule.active[self._phase_index + 1 :]
        )
        print(
            f"[DAY_PLAN] Spliced another CROP_ESTABLISH: "
            f"{wanted} ring(s) still want seed, {bags} bag(s) in the pocket"
        )

    def _advance(self, world: WorldState, reason: str) -> None:
        """Move to next phase after real work success."""
        current = self._schedule.current_at(self._phase_index)
        phase_name = current.phase if current is not None else "?"
        print(f"[DAY_PLAN] {phase_name} -> {reason}")
        self._record_phase_result(current, "success", reason, world)
        if current is not None and current.phase == "BUY_SEEDS":
            self._splice_plant_after_shop(world)
        if current is not None and current.phase == "CROP_ESTABLISH":
            self._splice_second_establish(world)
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
        self._record_phase_result(current, "no_work", reason, world)
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
            OPTIONAL_COW_PURCHASE_PHASES,
            OPTIONAL_SHOP_PHASES,
        )

        if phase_name in OPTIONAL_BERRY_PHASES:
            return OPTIONAL_BERRY_PHASES
        if phase_name in OPTIONAL_SHOP_PHASES:
            return OPTIONAL_SHOP_PHASES
        if phase_name in OPTIONAL_CHICKEN_SALE_PHASES:
            return OPTIONAL_CHICKEN_SALE_PHASES
        if phase_name in OPTIONAL_COW_PURCHASE_PHASES:
            return OPTIONAL_COW_PURCHASE_PHASES
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

    def _phase_map_mismatch(self, spec: PhaseSpec, world: WorldState) -> Optional[str]:
        """Fail-fast farm-only phases when the grape run left us on 0x10/0x0C."""
        contract = getattr(spec, "contract", None)
        required = tuple(getattr(contract, "required_maps", ()) or ())
        if not required:
            return None
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        allowed = {int(m) for m in required}
        if 0x00 in allowed and is_farm_tilemap(tilemap):
            return None
        if tilemap in allowed:
            return None
        need = ",".join(f"0x{m:02X}" for m in required)
        return f"map_mismatch:have=0x{tilemap:02X}:need={need}"

    def _try_map_lock_exit(self, spec: PhaseSpec, world: WorldState, reason: str) -> bool:
        """Walk back out to the farm instead of map-locking the rest of the day.

        run12 D8/D9: CLEAR_FIELD wandered indoors mid-phase, so NAV_CROP,
        HARVEST_ROUTE and both berry phases all map-locked on 0x15 and the day
        earned nothing. The farmhouse is one EXIT_TO_FARM away; spend that
        rather than forfeit every farm phase behind it. Once per phase per day.
        """
        from harvest.planner.day_plan import is_house_tilemap

        contract = getattr(spec, "contract", None)
        required = {int(m) for m in (getattr(contract, "required_maps", ()) or ())}
        if 0x00 not in required:
            return False
        if spec.phase in self._map_lock_exits:
            return False
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if not is_house_tilemap(tilemap):
            return False
        self._map_lock_exits.add(spec.phase)
        print(
            f"[DAY_PLAN] Phase {spec.phase} map lock ({reason}); "
            f"exiting to farm and retrying once"
        )
        self._schedule.splice_at(self._phase_index, [EXIT_TO_FARM_PHASE, spec])
        self._current_task = None
        return True

    def _phase_tool_lock(self, spec: PhaseSpec, world: WorldState) -> Optional[str]:
        """No-work a phase whose required carry items are simply not held.

        CROP_ESTABLISH hoes the whole ring before ``select_carry_0x07`` finds
        the seed bag was never bought (run13 D10: BUY_SEEDS blew its cutoff,
        then establish burned the afternoon and failed anyway).

        Only the seed bag is gated. Tools are recoverable in place — a missing
        watering can already routes to ``EnsureCarryToolTask`` via
        ``_make_recovery_task``, and skipping instead of fetching would be
        strictly worse. A bag is not: it needs a shop trip, which is its own
        phase behind its own window, so a missing one at this point is a
        settled fact for the day rather than a transient.
        """
        # ENSURE_* phases declare the tool they go and fetch, so for them a
        # missing tag is the reason to run, not a reason to skip.
        if isinstance(spec.kind, PhaseKind) and spec.kind in ACQUIRE_TOOL_KINDS:
            return None
        contract = getattr(spec, "contract", None)
        required = tuple(getattr(contract, "required_tools", ()) or ())
        if not required:
            return None
        have = set(tool_tags_from_ram(world.ram))
        missing = [
            t
            for t in required
            if str(t).lower() in _UNFETCHABLE_TOOL_TAGS
            and str(t).lower() not in have
        ]
        if not missing:
            return None
        return "no_work:missing_tool:" + ",".join(str(t) for t in missing)

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
    ) -> TaskResult | None:
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

    def _step_recovery(self, world: WorldState) -> TaskResult | None:
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
            return None

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
    ) -> TaskResult | None:
        """None means step() should keep walking the schedule."""
        policy = self._failure_policy(spec)
        if policy in {"optional", "opportunistic"}:
            if spec.phase in OPTIONAL_MONEY_PHASES:
                self._skip_optional_money_route(
                    reason, group=self._optional_route_group(spec.phase)
                )
            else:
                self._skip_failed_phase(spec, reason)
            return None
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

    def _d2_controls(self, spec: PhaseSpec) -> tuple[str, str, bool]:
        params = spec.params or {}
        chunk = params.get("chunk") or "all"
        if not isinstance(chunk, str):
            chunk = "all"
        return (
            str(params.get("section") or "all"),
            chunk,
            bool(params.get("include_spa", True)),
        )

    def _d2_marker(self) -> PhaseSpec | None:
        for phase in self._schedule.active[self._phase_index :]:
            if phase.phase == "D2_FARM_CLEAR":
                return phase
        return None

    def _d2_idle(self, reason: str, world: WorldState) -> TaskResult:
        from harvest.tasks.farm_clear_quota import yard_load_action

        if reason == "stale_farm_map":
            action = yard_load_action(world.ram)
        else:
            action = make_action()
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action),
            reason=reason,
        )

    def _d2_blocked(self, spec: PhaseSpec, reason: str, *, watchdog: str = "") -> TaskResult:
        extra: dict[str, object] = {}
        chunk = (spec.params or {}).get("chunk")
        if chunk:
            extra["chunk"] = chunk
        if watchdog:
            extra["watchdog"] = watchdog
        self._record_phase_result(spec, "blocked", reason, extra=extra or None)
        return TaskResult(status=TaskStatus.BLOCKED, reason=reason)

    def _d2_motion_key(self, world: WorldState):
        """Liveness while navigating. A planted tool swing is not motion."""
        from harvest.tasks.nav import get_pos_from_ram

        child = self._current_task
        if child is None:
            return None
        snapshot = task_progress_snapshot(child)
        phase = (snapshot.phase_text if snapshot is not None else "").lower()
        name = str(getattr(child, "name", "")).lower()
        navigating = phase in {"navigate", "navigating", "navigation"}
        navigating = navigating or name == "nav" or name.startswith("nav_")
        if not navigating:
            return None
        details = dict(snapshot.details) if snapshot is not None else {}
        pos = get_pos_from_ram(world.ram)
        return (
            (pos.x, pos.y),
            details.get("target", getattr(child, "_target_tile", None)),
            details.get("approach", getattr(child, "_approach_tile", None)),
        )

    def _d2_stall(self, spec: PhaseSpec, world: WorldState, status) -> TaskResult | None:
        """Motion and goal stalls for expanded D2 work. The spa stands still."""
        if spec.phase == "HOT_SPRING_STAMINA":
            self._d2_motion_at = self._step_count
            self._d2_goal_at = self._step_count
            self._d2_motion_seen = None
            return None
        if spec.phase != "D2_FARM_CLEAR":
            motion = self._d2_motion_key(world)
            if motion is None or motion != self._d2_motion_seen:
                self._d2_motion_seen = motion
                self._d2_motion_at = self._step_count
            elif stalled(self._d2_motion_at, self._step_count, MOTION_STALL_FRAMES):
                return self._d2_blocked(
                    spec, "navigation motion stall", watchdog="navigation_motion_stall"
                )
        from harvest.core.carry import backpack_tool, selected_tool

        goal = (
            status.weeds,
            status.fences,
            status.stones,
            status.large_rocks,
            status.stumps,
            status.planted,
            status.wet,
            status.stamina.current,
            int(selected_tool(world.ram)),
            int(backpack_tool(world.ram)),
        )
        if goal != self._d2_goal_key:
            self._d2_goal_key = goal
            self._d2_goal_at = self._step_count
        elif stalled(self._d2_goal_at, self._step_count, GOAL_STALL_FRAMES):
            return self._d2_blocked(spec, "goal stall", watchdog="goal_stall")
        return None

    def _expand_d2_marker(self, spec: PhaseSpec, world: WorldState) -> TaskResult | None:
        """Splice the next live child, or drop the marker when the section settles."""
        from harvest.planner.d2_work import (
            D2FarmOutcome,
            expand_d2_marker,
            observe_d2_farm,
        )

        status = observe_d2_farm(world.ram, self._phase_results)
        self._d2_status = status
        stalled_result = self._d2_stall(spec, world, status)
        if stalled_result is not None:
            return stalled_result
        if status.outcome == D2FarmOutcome.TEMPORARILY_UNOBSERVABLE:
            self._d2_unobs += 1
            if self._d2_unobs >= GOAL_STALL_FRAMES:
                return self._d2_blocked(spec, "stale_farm_map")
            return self._d2_idle(status.reason or "temporarily_unobservable", world)
        self._d2_unobs = 0
        section, chunk, include_spa = self._d2_controls(spec)
        kind, nxt = expand_d2_marker(
            status,
            section=section,
            chunk=chunk,
            include_spa=include_spa,
            last_phase=self._d2_last_phase,
            plot_attempted=self._d2_plot_attempted,
            previous=self._d2_prev,  # type: ignore[arg-type]
        )
        if kind == "start" and nxt is not None:
            self._current_task = None
            self._schedule.splice_at(self._phase_index, [_tag_d2_spec(nxt), spec])
            return None
        if kind == "complete":
            self._d2_prev = None
            self._current_task = None
            self._schedule.splice_at(self._phase_index, [])
            return None
        self._d2_prev = status
        reason = "settle" if kind == "settle" else "waiting verification"
        return self._d2_idle(reason, world)

    def _insert_d2_after(self, specs: list[PhaseSpec]) -> None:
        nxt = self._phase_index + 1
        active = self._schedule.active
        if nxt < len(active) and active[nxt].phase == "D2_FARM_CLEAR":
            self._schedule.splice_at(nxt, [*specs, active[nxt]])
            return
        active[nxt:nxt] = list(specs)

    def _leave_d2_phase(
        self,
        spec: PhaseSpec,
        status: str,
        reason: str,
        world: WorldState,
    ) -> None:
        print(f"[DAY_PLAN] {spec.phase} -> {status} ({reason})")
        self._record_phase_result(spec, status, reason, world)
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False

    def _finish_d2_child(self, spec: PhaseSpec, result: TaskResult, world: WorldState) -> TaskResult:
        from harvest.planner.day_phase_stamina import full_restore_spa_phase
        from harvest.planner.d2_work import (
            _SPA_RETRY_PHASES,
            _section_done,
            leftover_chain_decision,
            observe_d2_farm,
        )

        status = observe_d2_farm(world.ram, self._phase_results)
        self._d2_status = status
        reason = result.reason or ""
        if spec.phase == "CLEAR_PLOT" and result.status == TaskStatus.SUCCESS:
            self._d2_plot_attempted = True
        self._d2_last_phase = spec.phase
        if spec.phase == "HOT_SPRING_STAMINA" and result.status != TaskStatus.SUCCESS:
            text = f"spa failed: {reason or result.status.value}"
            self._record_phase_result(spec, result.status.value, text, world)
            self._current_task = None
            return TaskResult(status=TaskStatus.BLOCKED, reason=text)
        marker = self._d2_marker()
        if marker is None:
            section, chunk, include_spa = "all", "all", True
        else:
            section, chunk, include_spa = self._d2_controls(marker)
        remaining: list[str] = []
        if spec.phase in _SPA_RETRY_PHASES and not _section_done(status, section, chunk):
            remaining = ["CLEAR_ROCKS", "CLEAR_STUMPS"]
        decision = leftover_chain_decision(
            spec.phase,
            result.status,
            reason,
            status.stamina,
            remaining,
            include_spa=include_spa,
        )
        if decision == "spa_retry":
            self._insert_d2_after([_tag_d2_spec(full_restore_spa_phase()), spec])
        elif decision == "insert_spa":
            self._insert_d2_after([_tag_d2_spec(full_restore_spa_phase())])
        if decision in {"spa_retry", "insert_spa", "continue"} or result.status == TaskStatus.SUCCESS:
            recorded = "success" if result.status == TaskStatus.SUCCESS else result.status.value
            self._leave_d2_phase(spec, recorded, reason or recorded, world)
            idle = "queued" if decision in {"spa_retry", "insert_spa"} else "advance"
            return self._d2_idle(idle, world)
        chunk_name = (spec.params or {}).get("chunk")
        key = (spec.phase, chunk_name)
        self._d2_fails[key] = self._d2_fails.get(key, 0) + 1
        self._record_phase_result(spec, result.status.value, reason, world)
        self._current_task = None
        if chunk_name and self._d2_fails[key] >= 2:
            text = (
                f"required chunk failed {self._d2_fails[key]} times: "
                f"{spec.phase} chunk={chunk_name}; {reason or result.status.value}"
            )
            self._record_phase_result(
                spec,
                "blocked",
                text,
                world,
                extra={"chunk": chunk_name, "watchdog": "required_chunk_failure"},
            )
            return TaskResult(status=TaskStatus.BLOCKED, reason=text)
        return TaskResult(
            status=TaskStatus.BLOCKED,
            reason=f"blocked: {reason or result.status.value}",
        )

    def _boot_phase_task(self, spec: PhaseSpec, world: WorldState) -> TaskResult | None:
        """Start spec. None means the schedule changed and step should continue."""
        map_reason = self._phase_map_mismatch(spec, world)
        if map_reason is not None:
            if self._try_map_lock_exit(spec, world, map_reason):
                return None
            print(f"[DAY_PLAN] Phase {spec.phase} map lock: {map_reason}")
            return self._handle_failed_phase(spec, TaskStatus.FAILURE, map_reason, world)
        tool_reason = self._phase_tool_lock(spec, world)
        if tool_reason is not None:
            print(f"[DAY_PLAN] Phase {spec.phase} tool lock: {tool_reason}")
            if spec.params.get("d2_expanded"):
                self._d2_last_phase = spec.phase
            self._advance_no_work(world, tool_reason)
            if spec.params.get("d2_expanded"):
                return self._d2_idle("advance", world)
            return None
        task = self._make_task(spec, world)
        if task is None:
            reason = "no task"
            print(f"[DAY_PLAN] Phase {spec.phase} unavailable: {reason}")
            if spec.params.get("d2_expanded"):
                self._record_phase_result(spec, "failure", reason, world)
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"required phase {spec.phase} failed: {reason}",
                )
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
        return task  # type: ignore[return-value]

    def step(self, world: WorldState) -> TaskResult:
        expanded_d2 = False
        for _ in range(_STEP_BUDGET):
            self._step_count += 1
            if shipping_scene_needs_dismiss(world.ram):
                # 5pm shipper box (any phase). Child tasks must not hold A.
                return dismiss_dialogue_result(
                    self._step_count,
                    buttons=("a",),
                    pulse_every=2,
                    reason="shipping scene",
                )
            if self._recovery_task is not None:
                recovered = self._step_recovery(world)
                if recovered is None:
                    continue
                return recovered
            if self._phase_index >= len(self._schedule.active):
                if self._append_late_end_day_if_needed(world):
                    continue
                return TaskResult(status=TaskStatus.SUCCESS, reason="day plan complete")

            spec = self._schedule.active[self._phase_index]
            if self._expand_dynamic_phase(spec, world):
                continue
            if spec.phase == "D2_FARM_CLEAR":
                if expanded_d2:
                    return self._d2_idle("advance", world)
                expanded_d2 = True
                held = self._expand_d2_marker(spec, world)
                if held is None:
                    continue
                return held

            if self._current_task is None:
                booted = self._boot_phase_task(spec, world)
                if booted is None:
                    continue
                if isinstance(booted, TaskResult):
                    return booted

            if spec.params.get("d2_expanded"):
                from harvest.planner.d2_work import observe_d2_farm

                status = observe_d2_farm(world.ram, self._phase_results)
                self._d2_status = status
                stalled_result = self._d2_stall(spec, world, status)
                if stalled_result is not None:
                    return stalled_result

            result = self._current_task.step(world)
            if result.status == TaskStatus.SUCCESS:
                reason = result.reason or "SUCCESS"
                if is_no_work_reason(reason):
                    if spec.params.get("d2_expanded"):
                        self._d2_last_phase = spec.phase
                    self._advance_no_work(world, reason)
                    if spec.params.get("d2_expanded"):
                        return self._d2_idle("advance", world)
                    continue
                if spec.params.get("d2_expanded"):
                    return self._finish_d2_child(spec, result, world)
                self._advance(world, reason if reason != "SUCCESS" else "SUCCESS")
                continue
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or "unknown"
                print(f"[DAY_PLAN] Phase {spec.phase} {result.status.value.upper()}: {reason}")
                if spec.params.get("d2_expanded"):
                    return self._finish_d2_child(spec, result, world)
                failed = self._handle_failed_phase(spec, result.status, reason, world)
                if failed is None:
                    continue
                return failed
            if result.action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=result.action)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason="day plan advance",
        )


# Stable re-export for runtime importers.
from harvest.planner.multi_day_planner import MultiDayPlannerTask  # noqa: E402

__all__ = ["PhaseSchedule", "DayPlanTask", "MultiDayPlannerTask"]
