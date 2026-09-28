"""Multi-day day-plan orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.scene import (
    classify_scene_from_ram,
    morning_scene_ready,
    scene_indicates_ending,
)
from harvest.core.shipping_credit import SHIPPING_SCENE_HOUR
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.day_plan_decision import (
    DayPlanAdvisor,
    DayPlanDecision,
    auto_day_plan_decision,
)
from harvest.planner.day_plan_phases import DayPlannerPolicy
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    is_farm_tilemap,
    is_house_tilemap,
    read_world_date,
    read_world_day_time,
)
from harvest.planner.day_plan_tasks import (
    FarmShippingWaitTask,
    GoToSleepTask,
    HOUSE_SLEEP_TRANSITION_TILEMAP,
    ReturnHomeTask,
)
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import dismiss_dialogue_result


@dataclass
class MultiDayPlannerTask(Task):
    """Run day plans repeatedly, sleeping between days until a target date."""

    name: str = "multi_day_planner"
    seed_type: str = "potato"
    crop_rotation: Optional[dict[int, str]] = None
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
    _return_home_retries: int = field(default=0, init=False)
    _was_storm_yesterday: bool = field(default=False, init=False)
    # A stranded return_home is the dominant terminal killer of long runs. Give
    # it a few fresh attempts (nav state is rebuilt each time) before failing.
    max_return_home_retries: int = 3

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
        self._was_storm_yesterday = False

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

    def _build_day_task(self, world: WorldState):
        from harvest.planner.day_phase_types import day_planner_policy_for_season
        from harvest.planner.day_plan_orchestrator import DayPlanTask
        from harvest.planner.day_plan_status import resolve_seed_type_from_ram

        season, day = read_world_date(world.ram)
        day_policy = day_planner_policy_for_season(
            season,
            replace(self.policy, include_end_day=False),
        )
        # Determine target seed for the season
        target_seed = None
        if self.crop_rotation and season in self.crop_rotation:
            target_seed = self.crop_rotation[season]
        elif season == 0 and self.seed_type:
            target_seed = self.seed_type
        else:
            from harvest.planner.day_phase_calendar import default_crop_for_season

            target_seed = default_crop_for_season(season) or self.seed_type

        # The campaign's configured crop wins while it is still plantable this
        # date — ram resolution otherwise drifts to turnip after a few potato
        # ships (its ranch-master "remaining" ranks higher), diluting the
        # potato-max goal and thrashing the pocket rings between crops.
        resolved_seed = resolve_seed_type_from_ram(world.ram) or target_seed
        if target_seed and resolved_seed != target_seed:
            try:
                from harvest.planner.crop_planner import resolve_seed_type_for_date

                if (
                    resolve_seed_type_for_date(
                        season, day, inventory={target_seed: 1}
                    )
                    == target_seed
                ):
                    resolved_seed = target_seed
            except Exception:
                pass

        from harvest.planner.world_probe import WorldProbe

        probe = WorldProbe.from_inputs(ram=world.ram)
        is_storm = probe.is_storm()
        is_festival = probe.is_festival()
        festival_name = probe.festival_name()

        decision = auto_day_plan_decision(
            ram=world.ram,
            policy=day_policy,
            advisor=self.plan_advisor,
            is_storm=is_storm,
            is_post_storm=self._was_storm_yesterday,
            is_festival=is_festival,
            festival_name=festival_name,
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

    def _continue_d2_clear(self, world: WorldState) -> TaskResult | None:
        """Keep --stop-after-d2-clear from idling after an unfinished day plan."""
        from harvest.planner.day_plan_orchestrator import DayPlanTask
        from harvest.planner.d2_work import observe_d2_farm

        current = self._current_task
        if isinstance(current, DayPlanTask):
            planned = current.phases
            if len(planned) == 1 and planned[0].phase == "D2_FARM_CLEAR":
                return None
        status = observe_d2_farm(world.ram, self._last_day_phase_results)
        if not status.farm_map_loaded or status.is_complete:
            return None
        self._activate("plan_day", self._build_d2_clear_task(), world)
        plan = self._current_task
        assert isinstance(plan, DayPlanTask)
        plan._phase_results.extend(dict(row) for row in self._last_day_phase_results)
        return plan.step(world)

    def _build_d2_clear_task(self):
        from harvest.planner.day_plan_orchestrator import DayPlanTask
        from harvest.planner.d2_work import d2_farm_clear_phase

        return DayPlanTask(
            tasks_dir=self.tasks_dir,
            phase_sequence=[d2_farm_clear_phase()],
            policy=self.policy,
        )

    def _build_farm_shipping_wait_task(self) -> FarmShippingWaitTask:
        return FarmShippingWaitTask(
            name="wait_farm_shipping",
            target_hour=SHIPPING_SCENE_HOUR,
            timeout=25000,
        )

    def _needs_farm_shipping_wait(self, world: WorldState) -> bool:
        """True when bin has goods and farm 5pm ShippingScene has not fired yet.

        Day09 path (rr-53g / rr-y8n): stay on farm through hour 17 so
        ShippingScene dialogue runs before return-home/sleep. Wallet credit is
        still NightReset ``AddMoney``; this wait is the farm-side window.
        """
        try:
            from harvest.tasks.harvest_task import read_shipping_money

            shipping_money = int(read_shipping_money(world.ram))
        except Exception:
            shipping_money = 0
        if shipping_money <= 0:
            # Also honor journaled harvest ships if RAM already zeroed (edge).
            shipped = 0
            for row in self._last_day_phase_results:
                if str(row.get("phase")) == "HARVEST_ROUTE":
                    try:
                        shipped += int(row.get("shipped_count") or 0)
                    except Exception:
                        pass
            if shipped <= 0:
                return False
        _day, hour, _minute = read_world_day_time(world.ram)
        if hour >= SHIPPING_SCENE_HOUR:
            return False
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return is_farm_tilemap(tilemap)

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
        from harvest.planner.day_plan_orchestrator import DayPlanTask

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
        shipping_money = 0
        try:
            from harvest.core.ram_catalog import read_ram_value
            from harvest.tasks.harvest_task import read_shipping_money

            money = int(read_ram_value(world.ram, "money"))
            shipping_money = int(read_shipping_money(world.ram))
        except Exception:
            money = 0
            shipping_money = 0
        # Prefer snapshot taken when plan_day finished; fall back to live task.
        self._capture_day_plan_outcomes()
        phase_results = list(self._last_day_phase_results)
        # Decision-time omissions (for example rainy-day watering or a closed
        # shop) used to be printed but disappeared from continuous-run
        # journals.  Keep them alongside runtime failures so the next planning
        # review sees the complete reason a task was deferred.
        planned_deferred = (
            [item.to_jsonable() for item in self._last_day_decision.deferred]
            if self._last_day_decision is not None
            else []
        )
        deferred = _merge_deferred_rows(planned_deferred, self._last_day_deferred)
        planned = []
        if self._last_day_decision is not None:
            planned = [p.phase for p in self._last_day_decision.phases]
        shipped_total = 0
        harvested_total = 0
        establish_planted = 0
        for pr in phase_results:
            if str(pr.get("phase")) == "HARVEST_ROUTE":
                try:
                    shipped_total += int(pr.get("shipped_count") or 0)
                    harvested_total += int(pr.get("harvested_count") or 0)
                except Exception:
                    pass
            if str(pr.get("phase")) == "CROP_ESTABLISH" and str(pr.get("status")) == "success":
                reason = str(pr.get("reason") or "")
                # reason like planted=6 watered=0
                if "planted=" in reason:
                    try:
                        establish_planted += int(reason.split("planted=")[1].split()[0])
                    except Exception:
                        establish_planted += 1
        is_storm_day = bool(self._last_day_decision and self._last_day_decision.facts.is_storm)
        is_post_storm_day = bool(self._last_day_decision and self._last_day_decision.facts.is_post_storm)
        is_fest_day = bool(self._last_day_decision and self._last_day_decision.facts.is_festival)
        fest_name = self._last_day_decision.facts.festival_name if self._last_day_decision else None
        row = {
            "plan_season": int(season),
            "plan_day": int(day),
            "end_season": int(end_season),
            "end_day": int(end_day),
            "is_storm": is_storm_day,
            "is_post_storm": is_post_storm_day,
            "is_festival": is_fest_day,
            "festival_name": fest_name,
            "overnights_completed": int(self._days_completed + 1),
            "sleep_reason": sleep_reason,
            "money": money,
            "shipping_money": shipping_money,
            "shipped_count": shipped_total,
            "harvested_count": harvested_total,
            "establish_planted": establish_planted,
            "planned_phases": planned,
            "phase_results": phase_results,
            "deferred": deferred,
            "step": int(self._step_count),
        }
        self._day_journal.append(row)
        self._was_storm_yesterday = is_storm_day
        succeeded = [r["phase"] for r in phase_results if r.get("status") == "success"]
        skipped = [r["phase"] for r in phase_results if r.get("status") == "skipped"]
        no_work = [r["phase"] for r in phase_results if r.get("status") == "no_work"]
        print(
            f"[MULTI_DAY] Day journal {season}:{day} -> {end_season}:{end_day} "
            f"ok={succeeded or ['(none)']} skipped={skipped or ['(none)']} "
            f"no_work={no_work or ['(none)']} money={money}"
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
                self._return_home_retries = 0
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
        if self._phase in {"plan_day", "wait_shipping", "return_home"} and read_world_date(
            world.ram
        ) != self._active_day:
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
        elif self._phase == "wait_shipping":
            self._activate(
                "wait_shipping", self._build_farm_shipping_wait_task(), world
            )
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
                if not self.policy.include_end_day:
                    return TaskResult(status=result.status, reason=reason)
                return self._force_return_home_after_failure(world, result.status, reason)
            if self._phase == "wait_shipping":
                # Optional window — go home even if 5pm wait fails.
                self._record_day_failure(world, result.status, f"wait_shipping: {reason}")
                self._phase = "return_home"
                self._current_task = None
                print("[MULTI_DAY] Farm shipping wait failed; forcing return_home")
                return TaskResult(
                    status=TaskStatus.RUNNING, action=ActionResult(make_action())
                )
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
                if self._return_home_retries < self.max_return_home_retries:
                    self._return_home_retries += 1
                    self._current_task = None
                    print(
                        f"[MULTI_DAY] return_home outdoor failure "
                        f"({self._return_home_retries}/{self.max_return_home_retries}); "
                        f"rebuilding: {reason}"
                    )
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
            if self._needs_farm_shipping_wait(world):
                self._phase = "wait_shipping"
                self._current_task = None
                print(
                    "[MULTI_DAY] Day work done with shipping bin goods; "
                    "staying on farm for 5pm ShippingScene (Day09 path)"
                )
            elif not self.policy.include_end_day:
                continued = self._continue_d2_clear(world)
                if continued is not None:
                    return continued
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=result.reason or "day work complete",
                )
            else:
                self._phase = "return_home"
                self._current_task = None
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if self._phase == "wait_shipping":
            # Journal as a synthetic phase result for Gate A evidence.
            self._last_day_phase_results.append(
                {
                    "phase": "WAIT_FARM_SHIPPING",
                    "kind": "farm_shipping_wait",
                    "status": "success",
                    "reason": result.reason or "SUCCESS",
                    "step": int(self._step_count),
                }
            )
            print(f"[MULTI_DAY] Farm shipping wait done: {result.reason or 'success'}")
            if not self.policy.include_end_day:
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=result.reason or "shipping wait done",
                )
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


def _merge_deferred_rows(
    planned: Sequence[dict[str, object]],
    runtime: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge journal deferrals without duplicating the same intention.

    Planner and runtime rows have intentionally different detail levels, so
    preserve the richer planning row when both report the same phase/reason.
    """
    merged: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in (*planned, *runtime):
        clean = dict(row)
        key = (
            str(clean.get("phase", "")),
            str(clean.get("reason", "")),
            str(clean.get("retry", "tomorrow")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(clean)
    return merged
