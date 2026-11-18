from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.recovery import PhaseFaultInjector, RecoveryTask
from harvest.core.task_progress import ( GOAL_STALL_FRAMES, MOTION_STALL_FRAMES, ProgressSnapshot, stalled, task_progress_snapshot, work_snapshot, work_receipt, )
from harvest.core.world_context import WorldContext
from harvest.tasks.nav import make_action
from harvest.core.tile_catalog import ( ADDR_TILEMAP, Tool, )
from harvest.tasks.water_refill import is_no_work_reason
from harvest.planner.day_phase_types import ( ACQUIRE_TOOL_KINDS, PhaseKind, PhaseSpec, SKIP_MAP_LOCK_KINDS, phase_backstop_frames, tool_tags_from_ram, )
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
from harvest.planner.day_plan_status import ( TASKS_DIR, is_farm_tilemap, read_world_day_time, )
from harvest.planner.day_phase_chicken import splice_coop_reentry
from harvest.planner.day_phase_cow import MAX_BARN_RESETS, splice_barn_reentry, splice_barn_reset; from harvest.planner.day_phase_stamina import splice_water_retry
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.planner.day_plan_tasks import ( EnsureCarryToolTask, ExitToFarmTask, )
from harvest.core.scene import SceneGuard, unowned_scene_press
from harvest.tasks.primitives import dismiss_dialogue_result
_UNFETCHABLE_TOOL_TAGS = frozenset({"seed"})
_STEP_BUDGET = 128
_PHASE_WATCHDOG_KINDS = (PhaseKind.CROP, PhaseKind.HARVEST)
_PHASE_WATCHDOG_FRAMES = 24_000
def _phase_watchdog_frames(spec: PhaseSpec) -> int:
    explicit = spec.params.get("watchdog_frames")
    if explicit is not None:
        return int(explicit)
    if spec.params.get("d2_expanded"):
        return 0
    if isinstance(spec.kind, PhaseKind) and spec.kind in _PHASE_WATCHDOG_KINDS:
        return _PHASE_WATCHDOG_FRAMES
    return phase_backstop_frames(spec)
_ENV_FAULTS: list = []
def _env_fault_injector() -> Optional[PhaseFaultInjector]:
    if not _ENV_FAULTS:
        _ENV_FAULTS.append(PhaseFaultInjector.from_env())
    return _ENV_FAULTS[0]
def _tag_d2_spec(spec: PhaseSpec) -> PhaseSpec:
    params = dict(spec.params)
    params["d2_expanded"] = True
    return PhaseSpec( spec.phase, spec.kind, params, failure_policy=spec.failure_policy, contract=spec.contract, )
COMPOSER_SEAMS = ("social", "family", "cover")
_SEAM_ROWS: dict[str, tuple[PhaseSpec, ...]] = {name: () for name in COMPOSER_SEAMS}
def register_composer_seam(name: str, phases: Sequence[PhaseSpec] | None = None) -> None:
    if name not in _SEAM_ROWS:
        raise KeyError(name)
    _SEAM_ROWS[name] = tuple(phases or ())
def composer_seam_phases() -> tuple[PhaseSpec, ...]:
    return tuple(phase for name in COMPOSER_SEAMS for phase in _SEAM_ROWS[name])
def _with_composer_seams(phases: Sequence[PhaseSpec]) -> Sequence[PhaseSpec]:
    extra = composer_seam_phases()
    if not extra:
        return phases
    seq = list(phases)
    at = next((i for i, phase in enumerate(seq) if phase.phase in {"RETURN_HOME", "GO_TO_SLEEP"}), len(seq))
    return seq[:at] + list(extra) + seq[at:]
@dataclass
class PhaseSchedule:

    planned: tuple[PhaseSpec, ...]
    active: list[PhaseSpec]
    @classmethod
    def from_phases(cls, phases: Sequence[PhaseSpec]) -> PhaseSchedule:
        planned = tuple(phases)
        return cls(planned=planned, active=list(planned))
    @classmethod
    def from_sequence( cls, phase_sequence: Optional[List[PhaseSpec]], default: List[PhaseSpec], ) -> PhaseSchedule:
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
        return any( phase.phase in {"RETURN_HOME", "GO_TO_SLEEP"} for phase in self.active )

TOOL_STAMINA_PHASES = frozenset({"CLEAR_FIELD", "TILL_FIELD", "PLANT_PASTURE", "COVER_FIELD", "COVER_TILL", "CROP_WATER", "CUT_FODDER"})
SKIP_AFTER_RECOVERY_PHASES = frozenset({"CROP_ESTABLISH", "ENSURE_ANIMAL_TOOLS"})  # a ring that will not plant must not cost the day's watering; a missed brush or milker must not cost the herd its feed (y1a Winter D13: a cow died a week later)

@dataclass
class DayPlanTask(Task):

    name: str = "day_plan"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    phase_sequence: Optional[List[PhaseSpec]] = None
    state_name: Optional[str] = None
    policy: DayPlannerPolicy = field(default_factory=DayPlannerPolicy)
    fault_injector: Optional[PhaseFaultInjector] = None
    arbitrate: bool = False
    _fault_key: object = field(default=None, init=False)
    _fault_mode: Optional[str] = field(default=None, init=False)
    _schedule: PhaseSchedule = field( default_factory=lambda: PhaseSchedule.from_phases([]), init=False, )
    _phase_index: int = field(default=0, init=False)
    _current_task: Optional[Task] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _phase_started_at: int = field(default=0, init=False)
    _phase_start_frame: int | None = field(default=None, init=False)
    _intent_before: dict = field(default_factory=dict, init=False)
    _skip_map_lock: bool = field(default=False, init=False)
    _map_lock_exits: set = field(default_factory=set, init=False)
    _extra_establishes: int = field(default=0, init=False)
    _spa_refreshes: int = field(default=0, init=False)
    _barn_resets: int = field(default=0, init=False)
    _refreshed_phases: set[str] = field(default_factory=set, init=False)
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
        self._berry_pickup = None
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
        self._spa_refreshes = 0
        self._barn_resets = 0
        self._refreshed_phases.clear()
        self._deferred_plans.clear()
        self._phase_results.clear()
        self._intent_before.clear()
        self._phase_start_frame = None
    def _reset_phase_lists(self) -> None:
        phases = self.phase_sequence
        if composer_seam_phases():
            phases = _with_composer_seams(DAY1_PHASES if phases is None else phases)
        self._schedule = PhaseSchedule.from_sequence(phases, DAY1_PHASES)
    def _reset_build_context(self) -> None:
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
        end_frame = getattr(world, "frame", None)
        if end_frame is not None and self._phase_start_frame is not None:
            row.update(start_frame=self._phase_start_frame, end_frame=int(end_frame),
                       frames=max(0, int(end_frame) - self._phase_start_frame))
        if spec.params.get("intent_id"):
            from harvest.planner.campaign_composer import intent_ram_snapshot, verify_intent_receipt
            after = intent_ram_snapshot(world.ram) if world is not None else {}
            verified = bool(after and self._intent_before and
                            verify_intent_receipt(spec, self._intent_before, after))
            row.update(intent_id=spec.params["intent_id"], receipt={
                "before": dict(self._intent_before), "after": after, "verified": verified})
        task = self._current_task
        purchase = getattr(task, "purchase_receipt", None)
        if purchase:
            row["purchase_receipt"] = dict(purchase)
        for key in ("shipped_count", "harvested_count", "planted_count", "watered_count", "skipped_water", "skipped_count"):
            if task is not None and hasattr(task, key):
                row[key] = int(getattr(task, key))
        if getattr(task, "ship_log", None):  # per-crop pickup-to-ship frames (harvest_task.SHIP_LOG_SCHEMA)
            row["ship_log"] = [dict(rec) for rec in task.ship_log]
        if world is not None and self._phase_start_frame is not None:
            row["work"] = work_receipt(getattr(self, "_work_before", {}), world.ram, task)
        if (
            world is not None
            and spec.kind == "mountain_berry"
            and status == "success"
            and int(getattr(task, "shipped_count", 0) or 0) > 0
        ):
            from harvest.core.ram_catalog import read_ram_value
            row["shipping_deposit"] = {
                "season": int(read_ram_value(world.ram, "season") or 0),
                "day": int(read_ram_value(world.ram, "day") or 0),
                "hour": int(read_ram_value(world.ram, "hour") or 0),
                "minute": int(read_ram_value(world.ram, "minute") or 0),
                "shipping_money_before": int(getattr(task, "phase_shipping_before", getattr(task, "_shipping_before", 0))),
                "shipping_money_after": int(getattr(task, "_shipping_after", 0)),
            }
        if extra:
            row.update(extra)
        self._phase_results.append(row)
        self._phase_start_frame = None
    @property
    def phases(self) -> tuple[PhaseSpec, ...]:
        return self._schedule.planned
    @property
    def runtime_phases(self) -> tuple[PhaseSpec, ...]:
        return tuple(self._schedule.active)
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
        if spec.kind == PhaseKind.CHICKEN_SALE_REQUEST and self._build_ctx.chicken_sale_baseline is None:
            from harvest.core.ram_catalog import read_ram_value
            # Capture before the request: farm re-entry can pay and take the hen
            # before the event phase starts. Batch requests share this baseline.
            baseline = (int(read_ram_value(world.ram, "num_chickens")), int(read_ram_value(world.ram, "money")))
            self._build_ctx = replace(self._build_ctx, chicken_sale_baseline=baseline)
        task = build_phase_task(self._build_ctx, spec, world)
        if spec.kind == PhaseKind.CHICKEN_SALE_EVENT:
            self._build_ctx = replace(self._build_ctx, chicken_sale_baseline=None)
        return task
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
        if self._end_day_appended or not self.policy.include_end_day:
            return
        if self._schedule.has_end_day_phases():
            self._end_day_appended = True
            return
        self._schedule.append([RETURN_HOME_PHASE, GO_TO_SLEEP_PHASE])
        self._end_day_appended = True
        print("[DAY_PLAN] Appending end-day route after go-home flag")
    def _splice_plant_after_shop(self, world: WorldState) -> None:
        from harvest.planner.d2_work import d2_post_shop_work_phases, leftover_already_queued
        from harvest.planner.world_probe import WorldProbe
        remaining = [phase.phase for phase in self._schedule.active[self._phase_index + 1 :]]
        if leftover_already_queued(remaining) or any( name in remaining for name in ("CROP_ESTABLISH", "CLEAR_PLOT", "D2_FARM_CLEAR") ) or not self.policy.include_planting:
            return
        probe = WorldProbe.from_inputs(ram=world.ram, state_name=self.state_name)
        _day, hour, _minute = probe.day_time()
        if hour >= self.policy.late_water_hour or not probe.has_seasonal_plantable_seeds():
            return
        planted = [
            PhaseSpec( phase.phase, phase.kind, dict(phase.params), failure_policy=phase.failure_policy, contract=phase.contract, )
            for phase in d2_post_shop_work_phases()
        ]
        tail = [ phase for phase in self._schedule.active[self._phase_index + 1 :] if phase.phase != "CLEAR_FIELD" ]
        self._schedule.active = ( self._schedule.active[: self._phase_index + 1] + planted + tail )
        names = ", ".join(phase.phase for phase in planted)
        print(f"[DAY_PLAN] Spliced post-shop D2 work: {names}")
    def _splice_second_establish(self, world: WorldState) -> None:
        from harvest.core.ram_catalog import read_ram_value
        from harvest.maps.farm_pond import ( active_plot_centers, pocket_plant_targets, ram_year, )
        from harvest.planner.day_phase_catalog import CROP_ESTABLISH_PHASE
        if self._extra_establishes >= len(active_plot_centers(ram_year(world.ram), int(read_ram_value(world.ram, "season")))) - 1 or not self.policy.include_planting:
            return
        if "CROP_ESTABLISH" in [phase.phase for phase in self._schedule.active[self._phase_index + 1 :]]:
            return
        try:
            wanted = len(pocket_plant_targets(world.ram))
            bags = int(read_ram_value(world.ram, f"{self.seed_type}_seeds") or 0)
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
        print( f"[DAY_PLAN] Spliced another CROP_ESTABLISH: " f"{wanted} ring(s) still want seed, {bags} bag(s) in the pocket" )
    def _stamina_empty(self, ram: np.ndarray) -> bool:
        from harvest.core.stamina import _read_u8
        try:
            return _read_u8(ram, "max_stamina") > 0 and _read_u8(ram, "stamina") <= 4
        except Exception:
            return False
    def _try_stamina_spa_refresh(self, spec: PhaseSpec, world: WorldState, reason: str = "", result_reason: str = "") -> bool:
        from harvest.planner.day_phase_stamina import full_restore_spa_phase, refresh_marks, spa_refresh_allowed
        if spec.params.get("stop_hour") is not None:
            return False  # bounded morning work must leave time for the dated festival
        if not self.policy.include_spa or not spa_refresh_allowed(spec.phase, self._spa_refreshes, self._refreshed_phases, result_reason):
            return False
        self._schedule.active.insert(self._phase_index, full_restore_spa_phase())
        self._spa_refreshes += 1
        self._refreshed_phases.update(refresh_marks(spec.phase, self._refreshed_phases, result_reason))
        self._current_task = None
        print(f"[DAY_PLAN] Stamina low/empty for {spec.phase} ({reason}); spliced hot spring spa refresh")
        return True
    def _advance(self, world: WorldState, reason: str) -> None:
        current = self._schedule.current_at(self._phase_index)
        phase_name = current.phase if current is not None else "?"
        print(f"[DAY_PLAN] {phase_name} -> {reason}")
        self._record_phase_result(current, "success", reason, world)
        if current is not None and "stamina_low" in str(reason or "") and current.phase in TOOL_STAMINA_PHASES:
            if self._try_stamina_spa_refresh(current, world, "stamina_low_advance", str(reason)):
                return
        if current is not None and current.phase == "BUY_SEEDS":
            self._splice_plant_after_shop(world)
        if current is not None and current.phase == "CROP_ESTABLISH":
            self._splice_second_establish(world)
        if current is not None and current.phase in GO_HOME_TRIGGER_PHASES:
            self._mark_ready_to_go_home(current.phase)
            self._ensure_end_day_phases()
        if current is not None and current.params.get("ends_day"):  # the day rolled inside the phase
            self._schedule.active = self._schedule.active[: self._phase_index + 1]
            self._end_day_appended = True
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False
    def _advance_no_work(self, world: WorldState, reason: str) -> None:
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
            self._schedule.splice_at( self._phase_index, [EXIT_TO_FARM_PHASE, spec], )
            print("[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN deferred until farm tilemap")
            return True
        expanded = build_outdoor_day_phases_from_ram(world.ram, policy=self.policy, state_name=self.state_name)
        if composer_seam_phases():
            expanded = _with_composer_seams(expanded if expanded else [])
        replacement = expanded if expanded else []
        if self.arbitrate:
            from harvest.planner.campaign_composer import apply_block_order
            replacement = apply_block_order(replacement)
        self._schedule.splice_at(self._phase_index, replacement)
        if expanded:
            names = ", ".join(phase.phase for phase in expanded)
            print(f"[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN expanded to: {names}")
        else:
            print("[DAY_PLAN] DYNAMIC_OUTDOOR_PLAN expanded to no-op")
        return True
    def _optional_route_group(self, phase_name: str) -> frozenset[str]:
        from harvest.planner.day_phase_catalog import (
            OPTIONAL_BERRY_PHASES,
            OPTIONAL_CHICKEN_SALE_PHASES,
            OPTIONAL_WIFE_GIFT_PHASES,
            OPTIONAL_COW_PURCHASE_PHASES,
            OPTIONAL_SHOP_PHASES,
        )
        if phase_name in OPTIONAL_BERRY_PHASES:
            return OPTIONAL_BERRY_PHASES
        if phase_name in OPTIONAL_SHOP_PHASES:
            return OPTIONAL_SHOP_PHASES
        if phase_name in OPTIONAL_CHICKEN_SALE_PHASES:
            return OPTIONAL_CHICKEN_SALE_PHASES
        if phase_name in OPTIONAL_WIFE_GIFT_PHASES:
            return OPTIONAL_WIFE_GIFT_PHASES
        if phase_name in OPTIONAL_COW_PURCHASE_PHASES:
            return OPTIONAL_COW_PURCHASE_PHASES
        return OPTIONAL_MONEY_PHASES
    def _skip_optional_money_route(self, reason: str, *, group: frozenset[str] | None = None) -> None:
        route_group = group
        if route_group is None and self._phase_index < len(self._schedule.active):
            route_group = self._optional_route_group( self._schedule.active[self._phase_index].phase )
        if route_group is None:
            route_group = OPTIONAL_MONEY_PHASES
        skipped: List[str] = []
        while self._phase_index < len(self._schedule.active):
            current_phase = self._schedule.active[self._phase_index].phase
            if current_phase in route_group:
                skipped_spec = self._schedule.active[self._phase_index]
                skipped.append(skipped_spec.phase)
                self._record_deferred_phase(skipped_spec, reason)
                self._phase_index += 1
                continue
            if (
                current_phase == "EXIT_TO_FARM"
                and self._phase_index + 1 < len(self._schedule.active)
                and self._schedule.active[self._phase_index + 1].phase in route_group
            ):
                skipped_spec = self._schedule.active[self._phase_index]
                skipped.append(skipped_spec.phase)
                self._phase_index += 1
                continue
            break
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
    def _skip_failed_phase(self, spec: PhaseSpec, reason: str, world: WorldState | None = None) -> None:
        print(f"[DAY_PLAN] Skipping failed phase {spec.phase}: {reason}")
        self._record_phase_result(spec, "skipped", reason, world)
        self._record_deferred_phase(spec, reason)
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False
        if spec.kind == PhaseKind.HARVEST:
            self._skip_optional_money_route(reason)
    def _failure_policy(self, spec: PhaseSpec) -> str:
        policy = getattr(spec, "failure_policy", "required") or "required"
        if policy != "required":
            return policy
        if spec.phase in {"ENSURE_ANIMAL_TOOLS", "ENSURE_BRUSH", "NAV_TO_BARN", "ENTER_BARN", "COW_CHORES", "EXIT_BARN"}:
            return "required"
        if spec.phase in OPTIONAL_MONEY_PHASES:
            return "optional"
        return policy
    def _phase_map_mismatch(self, spec: PhaseSpec, world: WorldState) -> Optional[str]:
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
        from harvest.planner.day_plan import is_house_tilemap
        contract = getattr(spec, "contract", None)
        required = {int(m) for m in (getattr(contract, "required_maps", ()) or ())}
        if 0x00 not in required:
            return False
        if spec.phase in self._map_lock_exits:
            return False
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if not (is_house_tilemap(tilemap) or tilemap in {0x04, 0x05, 0x0C, 0x0F, 0x1C, 0x22, 0x23, 0x26, 0x27, 0x28}):
            return False
        self._map_lock_exits.add(spec.phase)
        print( f"[DAY_PLAN] Phase {spec.phase} map lock ({reason}); " f"exiting to farm and retrying once" )
        self._schedule.splice_at(self._phase_index, [EXIT_TO_FARM_PHASE, spec])
        self._current_task = None
        return True
    def _phase_tool_lock(self, spec: PhaseSpec, world: WorldState) -> Optional[str]:
        if isinstance(spec.kind, PhaseKind) and spec.kind in ACQUIRE_TOOL_KINDS:
            return None
        contract = getattr(spec, "contract", None)
        required = tuple(getattr(contract, "required_tools", ()) or ())
        if not required:
            return None
        have = set(tool_tags_from_ram(world.ram))
        missing = [ t for t in required if str(t).lower() in _UNFETCHABLE_TOOL_TAGS and str(t).lower() not in have ]
        if not missing:
            return None
        return "no_work:missing_tool:" + ",".join(str(t) for t in missing)
    def _recovery_phase_key(self, spec: PhaseSpec) -> tuple[int, str]:
        return self._phase_index, spec.phase
    def _make_recovery_task( self, spec: PhaseSpec, status: TaskStatus, reason: str, world: WorldState, ) -> Task:
        if spec.phase == "CROP_WATER" and reason == "watering can not in carry pair":
            return EnsureCarryToolTask( name="recover_crop_water_watering_can", tool_id=int(Tool.WATERING_CAN), tasks_dir=self.tasks_dir, )
        return RecoveryTask(
            name=f"recover_{spec.phase.lower()}",
            route_to_target_factory=lambda: ExitToFarmTask(tasks_dir=self.tasks_dir),
        )
    def _start_recovery( self, spec: PhaseSpec, status: TaskStatus, reason: str, world: WorldState, ) -> TaskResult | None:
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
            if spec is not None:
                args = (self._schedule, self._phase_index, spec, int(world.ram[ADDR_TILEMAP]), self._map_lock_exits)
                splice_coop_reentry(*args) or splice_barn_reentry(*args)
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
    def _handle_failed_phase( self, spec: PhaseSpec, status: TaskStatus, reason: str, world: WorldState, ) -> TaskResult | None:
        if spec.phase == "COW_CHORES" and self._barn_resets < MAX_BARN_RESETS and splice_barn_reset(self._schedule, self._phase_index, spec, reason):
            self._barn_resets, self._current_task = self._barn_resets + 1, None
            self._record_phase_result(spec, "retry", reason, world)
            return None
        if spec.phase == "COW_CHORES" and self._barn_resets >= MAX_BARN_RESETS:
            return self._skip_failed_phase(spec, reason, world)  # never lose the fields to one unlucky cow
        if "stamina_low" in str(reason or "") and spec.phase in TOOL_STAMINA_PHASES:
            if self._try_stamina_spa_refresh(spec, world, "stamina_low_fail"):
                return None
        if splice_water_retry(self._schedule, self._phase_index, spec, reason, world.ram, self._map_lock_exits): self._current_task = None; self._record_phase_result(spec, "retry", reason, world); return None
        policy = self._failure_policy(spec)
        if policy in {"optional", "opportunistic"}:
            if spec.phase in OPTIONAL_MONEY_PHASES:
                self._skip_optional_money_route( reason, group=self._optional_route_group(spec.phase) )
            else:
                self._skip_failed_phase(spec, reason, world)
            return None
        self._record_phase_result(spec, status.value, reason, world)
        if reason != "no task":
            key = self._recovery_phase_key(spec)
            if key not in self._recovery_attempted_phases:
                return self._start_recovery(spec, status, reason, world)
            if spec.phase in SKIP_AFTER_RECOVERY_PHASES: return self._skip_failed_phase(spec, f"failed after recovery: {reason}", world)
            return TaskResult( status=status, reason=f"required phase {spec.phase} failed after recovery: {reason}", )
        return TaskResult( status=status, reason=f"required phase {spec.phase} failed: {reason}", )
    def _append_late_end_day_if_needed(self, world: WorldState) -> bool:
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
        return ( str(params.get("section") or "all"), chunk, bool(params.get("include_spa", True)), )
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
        return TaskResult( status=TaskStatus.RUNNING, action=ActionResult(action), reason=reason, )
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
                return self._d2_blocked( spec, "navigation motion stall", watchdog="navigation_motion_stall" )
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
        from harvest.planner.d2_work import ( D2FarmOutcome, expand_d2_marker, observe_d2_farm, )
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
    def _leave_d2_phase( self, spec: PhaseSpec, status: str, reason: str, world: WorldState, ) -> None:
        print(f"[DAY_PLAN] {spec.phase} -> {status} ({reason})")
        self._record_phase_result(spec, status, reason, world)
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False
    def _finish_d2_child(self, spec: PhaseSpec, result: TaskResult, world: WorldState) -> TaskResult:
        from harvest.planner.day_phase_stamina import full_restore_spa_phase
        from harvest.planner.d2_work import ( _SPA_RETRY_PHASES, _section_done, leftover_chain_decision, observe_d2_farm, )
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
        decision = leftover_chain_decision( spec.phase, result.status, reason, status.stamina, remaining, include_spa=include_spa, )
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
            self._record_phase_result( spec, "blocked", text, world, extra={"chunk": chunk_name, "watchdog": "required_chunk_failure"}, )
            return TaskResult(status=TaskStatus.BLOCKED, reason=text)
        return TaskResult( status=TaskStatus.BLOCKED, reason=f"blocked: {reason or result.status.value}", )
    def _boot_phase_task(self, spec: PhaseSpec, world: WorldState) -> TaskResult | None:
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
                return TaskResult( status=TaskStatus.FAILURE, reason=f"required phase {spec.phase} failed: {reason}", )
            return self._handle_failed_phase(spec, TaskStatus.FAILURE, reason, world)
        if spec.phase in TOOL_STAMINA_PHASES - {"CROP_WATER"} and self._stamina_empty(world.ram):  # watering budgets itself
            if self._try_stamina_spa_refresh(spec, world, "empty_before_start"):
                return None
        task.reset(world)
        self._current_task = task
        self._phase_started_at = self._step_count
        self._phase_start_frame = getattr(world, "frame", None)
        self._work_before = work_snapshot(world.ram, census=True)
        if spec.params.get("intent_id"):
            from harvest.planner.campaign_composer import intent_ram_snapshot
            self._intent_before = intent_ram_snapshot(world.ram)
        self._skip_map_lock = ( isinstance(spec.kind, PhaseKind) and spec.kind in SKIP_MAP_LOCK_KINDS )
        print( f"[DAY_PLAN] Starting phase {self._phase_index + 1}/{len(self._schedule.active)}: " f"{spec.phase} ({spec.kind})" )
        return task  # type: ignore[return-value]
    def step(self, world: WorldState) -> TaskResult:
        from harvest.tasks.power_berry import PowerBerryTask, collect_spawned_berry
        # A scheduled berry phase owns its berry (the Kappa's cutscene sprite looks like a stray one to this hook).
        if not isinstance(self._current_task, PowerBerryTask):
            berry = collect_spawned_berry(self, world)
            if berry is not None:
                return berry
        expanded_d2 = False
        for _ in range(_STEP_BUDGET):
            self._step_count += 1
            guard = getattr(self, "_scene_guard", None)
            if guard is None:
                guard = self._scene_guard = SceneGuard()
            task = self._current_task
            registered = getattr(task, "purchase_already_registered", None)
            bought = bool(task is not None and registered is not None and registered(world.ram))
            choice = None
            if not bought:
                choice = unowned_scene_press( world.ram, guard, self._step_count, child_pressed=True )
            if choice is not None:
                buttons, reason = choice
                return dismiss_dialogue_result( self._step_count, buttons=buttons, pulse_every=2, reason=reason, )
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
            injector = self.fault_injector or _env_fault_injector()
            if injector is not None and not spec.params.get("d2_expanded"):
                key = (self._phase_index, spec.phase)
                if self._fault_key != key:
                    self._fault_key = key
                    self._fault_mode = injector.decide(spec.kind, spec.phase)
                    if self._fault_mode == "fail":
                        failed = self._handle_failed_phase( spec, TaskStatus.FAILURE, "injected fault", world )
                        if failed is None:
                            continue
                        return failed
                if injector.jitter_wait(key, spec.kind, ram=world.ram):
                    return TaskResult( status=TaskStatus.RUNNING, action=ActionResult(make_action()) )
                if self._fault_mode == "stall":
                    budget = _phase_watchdog_frames(spec)
                    if budget and self._step_count - self._phase_started_at > budget:
                        failed = self._handle_failed_phase( spec, TaskStatus.FAILURE, "injected stall watchdog", world )
                        if failed is None:
                            continue
                        return failed
                    return TaskResult( status=TaskStatus.RUNNING, action=ActionResult(make_action()) )
            budget = _phase_watchdog_frames(spec)
            if budget and self._step_count - self._phase_started_at > budget:
                reason = f"phase watchdog: {spec.phase} exceeded {budget} frames"
                print(f"[DAY_PLAN] Phase {spec.phase} FAILURE: {reason}")
                failed = self._handle_failed_phase(spec, TaskStatus.FAILURE, reason, world)
                if failed is None:
                    continue
                return failed
            result = self._current_task.step(world)
            if result.status == TaskStatus.SUCCESS:
                reason = result.reason or "SUCCESS"
                if spec.params.get("intent_id"):
                    from harvest.planner.campaign_composer import intent_ram_snapshot, verify_intent_receipt
                    if not verify_intent_receipt(spec, self._intent_before, intent_ram_snapshot(world.ram)):
                        failed = self._handle_failed_phase(spec, TaskStatus.FAILURE,
                                                          "RAM receipt did not satisfy milestone", world)
                        if failed is None:
                            continue
                        return failed
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
        return TaskResult( status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="day plan advance", )

from harvest.planner.multi_day_planner import MultiDayPlannerTask  # noqa: E402

__all__ = [
    "COMPOSER_SEAMS",
    "DayPlanTask",
    "MultiDayPlannerTask",
    "PhaseSchedule",
    "composer_seam_phases",
    "register_composer_seam",
]
