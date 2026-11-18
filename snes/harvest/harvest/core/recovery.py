"""Scene-level recovery task for autonomous planner failures."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.scene import Scene, SceneLocation, SceneMode, classify_scene_from_ram
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import dismiss_dialogue_result


TaskFactory = Callable[[], Task]

FAR_RECOVERY_TIMEOUT = 6000
_FAR_LOCATIONS = frozenset({SceneLocation.MOUNTAIN, SceneLocation.PATH, SceneLocation.TOWN, SceneLocation.SHOP})


@dataclass
class RecoveryTask(Task):
    """Recover from transient scene states before retrying a planner phase."""

    name: str = "recovery"
    target_location: SceneLocation | str | None = SceneLocation.FARM
    stable_frames: int = 8
    timeout: int = 900
    cutscene_mash_limit: int = 240
    route_to_target_factory: Optional[TaskFactory] = None
    dismiss_buttons: Sequence[str] = ("b", "a")

    _budget: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _stable_count: int = field(default=0, init=False)
    _cutscene_mash_count: int = field(default=0, init=False)
    _route_task: Optional[Task] = field(default=None, init=False)
    _route_started_from: str = field(default="", init=False)
    _last_scene: str = field(default="", init=False)

    def reset(self, world: WorldState) -> None:
        # A route home from the mountain, path or town is several map walks
        # (~3,000 frames); 900 timed out one map short of the farm (Y1 Summer D14).
        far = classify_scene_from_ram(world.ram).location in _FAR_LOCATIONS
        self._budget = max(self.timeout, FAR_RECOVERY_TIMEOUT) if far else self.timeout
        self._step_count = 0
        self._stable_count = 0
        self._cutscene_mash_count = 0
        self._route_task = None
        self._route_started_from = ""
        self._last_scene = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        scene = classify_scene_from_ram(world.ram)
        self._last_scene = scene.summary()

        if self._target_matches(scene):
            self._stable_count += 1
            self._cutscene_mash_count = 0
            if self._stable_count >= self.stable_frames:
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"recovered at {scene.summary()}")
            return self._idle(f"stabilizing {scene.summary()}")

        self._stable_count = 0
        if self._step_count > (self._budget or self.timeout):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"recovery timeout, last scene {self._last_scene}",
            )

        if scene.mode in {SceneMode.UNKNOWN_TILEMAP, SceneMode.INVALID_COORDINATES, SceneMode.ENDING_CREDITS}:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"unrecoverable scene {scene.summary()}")

        if scene.needs_input_dismiss:
            if scene.mode == SceneMode.CUTSCENE_EVENT:
                self._cutscene_mash_count += 1
                if self._cutscene_mash_count > self.cutscene_mash_limit:
                    return TaskResult(
                        status=TaskStatus.BLOCKED,
                        reason=f"cutscene did not clear: {scene.summary()}",
                    )
            else:
                self._cutscene_mash_count = 0
            return dismiss_dialogue_result(
                self._step_count,
                buttons=self.dismiss_buttons,
                pulse_every=1,
                reason=f"recovering {scene.mode.value}",
            )

        self._cutscene_mash_count = 0
        if scene.mode in {SceneMode.MAP_TRANSITION, SceneMode.SLEEP_WAKE_TRANSITION}:
            return self._idle(f"waiting for {scene.summary()}")

        if scene.mode != SceneMode.NORMAL:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"unhandled scene {scene.summary()}")

        return self._route_to_target(world, scene)

    def _target_matches(self, scene: Scene) -> bool:
        if scene.mode != SceneMode.NORMAL:
            return False
        if self.target_location is None:
            return True
        return scene.location.value == _enum_text(self.target_location)

    def _route_to_target(self, world: WorldState, scene: Scene) -> TaskResult:
        if self.route_to_target_factory is None:
            return TaskResult(
                status=TaskStatus.BLOCKED,
                reason=f"normal scene is not target location: {scene.summary()}",
            )
        if self._route_task is None:
            self._route_task = self.route_to_target_factory()
            self._route_task.reset(world)
            self._route_started_from = scene.summary()

        result = self._route_task.step(world)
        if result.status == TaskStatus.RUNNING:
            if result.action is not None:
                return result
            return self._idle(result.reason or "recovery route running")

        if result.status == TaskStatus.SUCCESS:
            scene_after = classify_scene_from_ram(world.ram)
            if self._target_matches(scene_after):
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"route recovered from {self._route_started_from}: {result.reason or scene_after.summary()}",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "recovery route completed before target scene: "
                    f"{scene_after.summary()} ({result.reason or 'no route reason'})"
                ),
            )

        return TaskResult(
            status=result.status,
            reason=f"recovery route {result.status.value}: {result.reason or 'unknown'}",
        )

    def _idle(self, reason: str) -> TaskResult:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=reason)


def _enum_text(value: SceneLocation | str) -> str:
    return value.value if isinstance(value, SceneLocation) else str(value)


# ROM truth (HM-Decomp bank_82 UpdateTime, NightReset): the clock caps at 18:00
# and nothing ever forces the day to end. Only the bed (time_running bit 1,
# bank_81 $81D8xx) reaches NightReset, so a farmer that never gets home stays
# outdoors forever. The floor below is therefore the only thing that ends a
# bad day; the ROM will not do it for us.
# It is a last resort, not a schedule: work after 18:00 costs no game time (the
# hour is frozen), and CROP/HARVEST phases have their own 24k watchdog. At
# 24k/3k it cut watering on 8 of 28 spring days ($4,630 vs $5,540, D3 pin);
# at 48k/15k it never trips there and spring ends at $5,150 (rr-bu62.16).
# Owner, 2026-11-02: no per-day time cap. A day may run an hour of play (216,000
# frames at 60 fps) if the work is real; only a stall, glitch or clip ends it, and
# those are caught by the per-phase watchdogs and ``MAX_RESCUES_PER_DAY``. The 48k
# floor cut the packed ring lattice's planting and watering days short (28-32 trips
# in two Day-3-pin runs against 5 for nine rings). Override with
# ``HARVEST_DAY_FRAME_BUDGET`` / ``HARVEST_DAY_CLOCK_GRACE`` (0 / -1 disable).
DAY_WORK_FRAME_BUDGET = 0
DAY_CLOCK_HOUR = 18
DAY_CLOCK_GRACE_FRAMES = -1
MAX_RESCUES_PER_DAY = 8
# Phases that renew the 18:00 grace while they run (the hard budget still
# applies): the Eve talk loop, and watering with its hot-spring refill, which a
# field larger than one stamina bar finishes after dark.
LATE_WORK_PHASES = frozenset({"EVE_TALK_LOOP", "CROP_WATER", "HOT_SPRING_STAMINA"})


@dataclass
class DayFloor:
    """Per-day frame budget and clock deadline for the work window.

    Pure logic: the planner feeds it the step count and the game hour and gets
    back a reason string when the day's remaining phases must be abandoned.
    ``bad_days`` counts consecutive days that hit the floor, for reporting.
    """

    frame_budget: int = DAY_WORK_FRAME_BUDGET
    clock_hour: int = DAY_CLOCK_HOUR
    clock_grace: int = DAY_CLOCK_GRACE_FRAMES
    max_rescues: int = MAX_RESCUES_PER_DAY
    bad_days: int = 0
    total_bad_days: int = 0
    rescues: int = 0
    _started_at: int = field(default=0, init=False)
    _late_since: Optional[int] = field(default=None, init=False)
    _tripped: bool = field(default=False, init=False)
    _bad: bool = field(default=False, init=False)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "DayFloor":
        env = os.environ if env is None else env
        floor = cls()
        raw = env.get("HARVEST_DAY_FRAME_BUDGET")
        if raw:
            floor.frame_budget = int(raw)
        raw = env.get("HARVEST_DAY_CLOCK_GRACE")
        if raw:
            floor.clock_grace = int(raw)
        return floor

    def begin_day(self, step: int) -> None:
        self._started_at = int(step)
        self._late_since = None
        self._tripped = False
        self._bad = False
        self.rescues = 0

    def check(self, step: int, hour: int, *, allow_late_work: bool = False) -> Optional[str]:
        """Evening skills may renew clock grace, never the hard frame budget."""
        if self._tripped:
            return None
        elapsed = int(step) - self._started_at
        if self.frame_budget > 0 and elapsed > self.frame_budget:
            return self._trip(f"day frame budget {self.frame_budget} exceeded")
        if hour >= self.clock_hour:
            if self._late_since is None or allow_late_work:
                self._late_since = int(step)
            if self.clock_grace >= 0 and step - self._late_since > self.clock_grace:
                return self._trip(
                    f"clock at {hour}:00 for {self.clock_grace} frames"
                )
        return None

    def _trip(self, reason: str) -> str:
        self._tripped = True
        return reason

    def end_day(self, *, clean: bool) -> None:
        """Roll the consecutive-bad-day counter (reporting only)."""
        if clean and not self._tripped and not self._bad and self.rescues == 0:
            self.bad_days = 0
        else:
            self.bad_days += 1
            self.total_bad_days += 1

    def note_bad(self) -> None:
        """Mark today bad (a failure was recorded) without aborting.

        This must not disarm :meth:`check`: ``y3a`` recorded an optional ``CLEAR_FIELD``
        failure, the floor then never tripped again, and a stall at 17:00 hung the
        planner for 1.6M frames (Y3 Spring D20).
        """
        self._bad = True

    def next_rescue(self) -> bool:
        """Spend one rescue. False when the farmer is provably wedged."""
        self.rescues += 1
        return self.rescues <= self.max_rescues


_FAULT_MODES = ("fail", "stall")


@dataclass
class PhaseFaultInjector:
    """Test hook: force chosen phase kinds to fail or stall, or jitter phase starts.

    ``HARVEST_FAULT_PHASES="crop:fail,harvest:stall,nav:fail"`` (kind names, as
    in ``PhaseKind``); ``HARVEST_FAULT_RATE`` (0..1, default 1) fails only that
    share of phase starts, drawn from ``HARVEST_FAULT_SEED``. A ``*`` kind hits
    every phase except the end-of-day route (return_home / sleep).
    """

    faults: dict[str, str] = field(default_factory=dict)
    rate: float = 1.0
    seed: int = 0
    jitter: int = 0
    # Jitter only phases of this 0-based game year on (``HARVEST_PHASE_JITTER_FROM_YEAR``): a power-on run keeps Years 1-2
    # exactly as the deterministic campaign plays them and seeds only the Year 3 draws.
    jitter_from_year: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _jitter_rng: random.Random = field(init=False, repr=False)
    _jitter_key: object = field(default=None, init=False, repr=False)
    _jitter_left: int = field(default=0, init=False, repr=False)
    fired: list[tuple[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._jitter_rng = random.Random(self.seed * 7919 + 13)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["PhaseFaultInjector"]:
        env = os.environ if env is None else env
        raw = env.get("HARVEST_FAULT_PHASES", "").strip()
        faults: dict[str, str] = {}
        for item in raw.split(",") if raw else ():
            kind, _, mode = item.strip().partition(":")
            mode = mode or "fail"
            if kind and mode in _FAULT_MODES:
                faults[kind.lower()] = mode
        jitter = int(env.get("HARVEST_PHASE_JITTER", "0") or 0)
        if not faults and jitter <= 0:
            return None
        return cls(
            faults=faults,
            rate=float(env.get("HARVEST_FAULT_RATE", "1") or 1),
            seed=int(env.get("HARVEST_FAULT_SEED", "0") or 0),
            jitter=max(0, jitter),
            jitter_from_year=int(env.get("HARVEST_PHASE_JITTER_FROM_YEAR", "0") or 0),
        )

    def jitter_wait(self, key: object, kind: object, year: Optional[int] = None, ram=None) -> bool:
        """True while idling before a new phase starts (0..``jitter`` frames, seeded).

        ``HARVEST_PHASE_JITTER=<max frames>`` shifts sprite, animal and ROM RNG
        timing run to run without any RAM write, so a repeatability sweep sees
        different draws; end-of-day phases are left alone.
        """
        if self.jitter <= 0 or str(getattr(kind, "value", kind)).lower() in {"return_home", "sleep"}:
            return False
        if year is None and ram is not None and self.jitter_from_year:
            from harvest.maps.farm_pond import ram_year

            year = ram_year(ram)
        if year is not None and int(year) < self.jitter_from_year:
            return False
        if self._jitter_key != key:
            self._jitter_key, self._jitter_left = key, self._jitter_rng.randint(0, self.jitter)
        if self._jitter_left <= 0:
            return False
        self._jitter_left -= 1
        return True

    def decide(self, kind: object, phase: str) -> Optional[str]:
        """Fault mode for a phase start, or None. Never touches end-of-day."""
        name = str(getattr(kind, "value", kind)).lower()
        if name in {"return_home", "sleep"}:
            return None
        mode = self.faults.get(name) or self.faults.get("*")
        if mode is None:
            return None
        if self._rng.random() >= self.rate:
            return None
        self.fired.append((phase, mode))
        print(f"[FAULT] injecting {mode} into phase {phase} ({name})")
        return mode


__all__ = [
    "DAY_WORK_FRAME_BUDGET",
    "DayFloor",
    "PhaseFaultInjector",
    "RecoveryTask",
]
