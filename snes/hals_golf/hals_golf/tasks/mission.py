"""End-to-end stroke-play / VS HAL mission that can survive human takeover."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retro_harness.mission_control import MissionSnapshot
from retro_harness.protocol import (
    ActionResult,
    TaskResult,
    TaskStatus,
    WorldState,
)

from hals_golf.core.actions import idle, named_script
from hals_golf.core.ram import (
    WRAM_HOLE_INDEX,
    read_hole_number,
    read_opponent_strokes,
    read_rest_distance,
    read_stroke_count,
    read_u8,
)
from hals_golf.core.recovery import RecoveryController
from hals_golf.core.scene import classify_scene
from hals_golf.tasks.menus import (
    ClubSet,
    Difficulty,
    MenuBootstrapTask,
    PlayMode,
)
from hals_golf.tasks.mission_play import (
    COMMAND_STALL_FRAMES,
    FUTILE_SHOT_LIMIT,
    InRoundPlay,
    MissionPhase,
    OPPONENT_WAIT_LIMIT,
)
from hals_golf.tasks.profile import MissionProfile
from hals_golf.tasks.routes.tables import AMATEUR_PARS, VS_HAL_MATCH_HOLES
from hals_golf.tasks.scorecard import ScorecardBook
from hals_golf.tasks.shot import PuttTask, ShotTask
from hals_golf.tasks.shot_policy import DeterministicRoutePolicy, RoutePolicy

__all__ = [
    "COMMAND_STALL_FRAMES",
    "FUTILE_SHOT_LIMIT",
    "MissionPhase",
    "OPPONENT_WAIT_LIMIT",
    "StrokePlayMission",
]


@dataclass
class StrokePlayMission(InRoundPlay):
    """Beat stroke play or VS HAL by chaining menu bootstrap + repeated shots.

    Human takeover (``~`` / L+R+SELECT) keeps this mission hot. On resume we
    run recovery dismissals, then rebuild the current shot from the command
    menu instead of replaying the whole bootstrap.

    VS HAL reuses the Amateur hole plans. Hal auto-plays between holes; while
    the command panel is absent we idle in ``WAIT_OPPONENT`` instead of mashing
    SHOT. Match outcome is derived from per-hole stroke comparisons.
    """

    name: str = "stroke_play"
    play_mode: PlayMode = PlayMode.STROKE_PLAY
    club_set: ClubSet = ClubSet.STANDARD
    difficulty: Difficulty = Difficulty.AMATEUR
    max_holes: int = 18
    max_strokes_per_hole: int = 24
    skip_bootstrap: bool = False
    power_delay: int = 42
    impact_delay: int = 26
    route_policy: RoutePolicy = field(default_factory=DeterministicRoutePolicy)
    _phase: MissionPhase = MissionPhase.BOOTSTRAP
    _bootstrap: MenuBootstrapTask = field(default_factory=MenuBootstrapTask)
    _shot: ShotTask | PuttTask | None = None
    _recovery: RecoveryController = field(default_factory=RecoveryController)
    _dismiss_queue: list[np.ndarray] = field(default_factory=list)
    _holes_completed: int = 0
    _strokes_this_hole: int = 0
    _peak_strokes_this_hole: int = 0
    _last_hole: int = -1
    _transition_frames: int = 0
    _transition_taps: int = 0
    _last_putt_rest: int = -1
    _putt_retries: int = 0
    _card: ScorecardBook = field(default_factory=ScorecardBook)
    _stall_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
    _stall_frames: int = 0
    _stall_nudges: int = 0
    _futile_shots: int = 0
    _shot_start_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
    _last_rest: int = -1
    _awaiting_stroke_reset: bool = False
    _opponent_wait_frames: int = 0
    _no_command_frames: int = 0
    _waiting_after_hole_out: bool = False

    def __post_init__(self) -> None:
        if self.play_mode is PlayMode.VS_HAL and self.name == "stroke_play":
            self.name = "vs_hal"

    @property
    def profile(self) -> MissionProfile:
        return MissionProfile(
            play_mode=self.play_mode,
            club_set=self.club_set,
            difficulty=self.difficulty,
            max_holes=self.max_holes,
        )

    @property
    def metal_clubs(self) -> bool:
        """Compatibility alias for metal club calibration."""
        return self.club_set is ClubSet.METAL

    @property
    def _hole_scores(self) -> list[int]:
        return self._card.holes

    def reset(self, world: WorldState) -> None:
        self._phase = (
            MissionPhase.PLAY_HOLE if self.skip_bootstrap else MissionPhase.BOOTSTRAP
        )
        self._bootstrap = MenuBootstrapTask(
            play_mode=self.play_mode,
            club_set=self.club_set,
            difficulty=self.difficulty,
        )
        self._bootstrap.reset(world)
        self._shot = None
        self._recovery = RecoveryController()
        self._dismiss_queue = []
        self._holes_completed = 0
        # Mid-hole skip-bootstrap fixtures seed from RAM so stroke-index plans
        # match. Title / cold-boot RAM is garbage until Hole 1, so keep zero
        # until bootstrap finishes (see ``_step_bootstrap``).
        strokes = (
            max(0, read_stroke_count(world.ram, world.info))
            if self.skip_bootstrap
            else 0
        )
        self._strokes_this_hole = strokes
        self._peak_strokes_this_hole = strokes
        hole = read_hole_number(world.ram, world.info)
        self._last_hole = hole if 1 <= hole <= self.max_holes else -1
        self._transition_frames = 0
        self._transition_taps = 0
        self._last_putt_rest = -1
        self._putt_retries = 0
        self._card = ScorecardBook()
        self._stall_key = (-1, -1, -1, -1)
        self._stall_frames = 0
        self._stall_nudges = 0
        self._futile_shots = 0
        self._shot_start_key = (-1, -1, -1, -1)
        self._last_rest = read_rest_distance(world.ram, world.info)
        self._awaiting_stroke_reset = False
        self._opponent_wait_frames = 0
        self._no_command_frames = 0
        self._waiting_after_hole_out = False
        if self.skip_bootstrap:
            self._start_shot(world)

    def can_start(self, world: WorldState) -> bool:
        del world
        return True

    def on_human_takeover(self) -> None:
        """Pause autonomous issuance; mission state stays intact."""
        return None

    def on_autopilot_resume(self) -> None:
        """Dismiss chord side-effects and re-sync to the command menu."""
        self._recovery.start(reason="autopilot_resume")
        self._dismiss_queue = []
        self._shot = None
        if self._phase == MissionPhase.BOOTSTRAP:
            return
        self._phase = MissionPhase.PLAY_HOLE

    def mission_status(self) -> MissionSnapshot:
        hole = self._last_hole if self._last_hole > 0 else "?"
        phase = self._phase.name
        if self._shot is not None:
            shot_phase = getattr(
                self._shot._phase,
                "name",
                str(self._shot._phase),
            )
            phase = f"{phase}/{shot_phase}"
        total = self._card.total
        if self.play_mode is PlayMode.VS_HAL:
            objective = (
                f"hole={hole} strokes={self._strokes_this_hole} "
                f"done={self._holes_completed}/{self.max_holes} "
                f"match={self._card.holes_won}-"
                f"{self._card.holes_lost}-{self._card.holes_tied}"
            )
        else:
            objective = (
                f"hole={hole} strokes={self._strokes_this_hole} "
                f"done={self._holes_completed}/{self.max_holes} "
                f"total={total}"
            )
        return MissionSnapshot(
            mission_id=self.name,
            phase=phase,
            objective=objective,
        )

    def progress_snapshot(self) -> dict[str, int | str]:
        """Stall-watchdog friendly progress tuple."""
        return {
            "phase": self._phase.name,
            "hole": int(self._last_hole),
            "strokes": int(self._strokes_this_hole),
            "holes_completed": int(self._holes_completed),
            "rest": int(self._last_rest),
            "total": int(self._card.total),
            "stall": int(self._stall_frames),
            "holes_won": int(self._card.holes_won),
            "holes_lost": int(self._card.holes_lost),
            "holes_tied": int(self._card.holes_tied),
            "mode": self.play_mode.name,
        }

    def scorecard(self) -> dict[str, int | list[int]]:
        """Return recorded scores, pars, round total, and per-hole regressions."""
        numbers = list(self._card.hole_numbers)
        pars = [AMATEUR_PARS[number - 1] for number in numbers]
        return self._card.as_dict(pars)

    def match_lead(self) -> int:
        """Return holes up (positive) or down (negative) versus Hal."""
        return self._card.match_lead

    def _record_hole_score(self, world: WorldState) -> None:
        """Append the just-finished hole score.

        The red byte previously labeled as a cumulative score is the aiming
        offset. The peak per-hole stroke byte is the reliable source of truth,
        so the round total is the sum of the recorded hole scores.

        In VS HAL, ``0x10A3`` often holds Hal's strokes for the hole at the
        boundary; compare it to our peak to update the match tally.
        """
        scored = self._peak_strokes_this_hole
        if scored <= 0 and self._strokes_this_hole > 0:
            scored = self._strokes_this_hole
        opponent = None
        if self.play_mode is PlayMode.VS_HAL:
            opponent = read_opponent_strokes(world.ram)
        self._card.record(scored, self._last_hole, opponent=opponent)
        self._peak_strokes_this_hole = 0
        self._stall_key = (-1, -1, -1, -1)
        self._stall_frames = 0
        self._stall_nudges = 0
        self._futile_shots = 0

    def step(self, world: WorldState) -> TaskResult:
        """Thin dispatcher: sync tracking, then delegate to a phase helper."""
        decision = classify_scene(world.ram, info=world.info, obs=world.obs)
        hole = read_hole_number(world.ram, world.info)
        strokes = read_stroke_count(world.ram, world.info)
        rest = read_rest_distance(world.ram, world.info)
        self._last_rest = rest
        self._sync_stroke_tracking(world, hole, strokes)

        # The menu bootstrap is a verified frame script. Pre-round RAM and
        # scene heuristics must not pre-empt it.
        if self._phase == MissionPhase.BOOTSTRAP:
            return self._step_bootstrap(world)

        self._maybe_advance_hole(world, hole, strokes, rest)

        completion = self._check_completion(world)
        if completion is not None:
            return completion

        recovery_action = self._recovery.step(decision)
        if recovery_action is not None:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=recovery_action, reason="recovery"),
                meta=self.progress_snapshot(),
            )

        if self._dismiss_queue:
            action = self._dismiss_queue.pop(0)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=action, reason="dismiss"),
                meta=self.progress_snapshot(),
            )

        if self._phase == MissionPhase.NEXT_HOLE:
            return self._step_next_hole(world, strokes, rest)

        if self._phase == MissionPhase.WAIT_OPPONENT:
            return self._step_wait_opponent(world, rest)

        if self._phase == MissionPhase.PLAY_HOLE:
            return self._step_play_hole(world, hole, strokes, rest)

        if self._phase == MissionPhase.COMPLETE:
            return TaskResult(status=TaskStatus.SUCCESS, reason="complete")

        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"unhandled_phase:{self._phase.name}",
        )

    def _sync_stroke_tracking(
        self,
        world: WorldState,
        hole: int,
        strokes: int,
    ) -> None:
        """Track the peak per-hole stroke byte before any phase transition."""
        if 1 <= hole <= self.max_holes and hole == self._last_hole:
            if self._awaiting_stroke_reset:
                if strokes == 0:
                    self._awaiting_stroke_reset = False
            elif self.profile.is_vs_hal:
                # Prefer our issued-shot count; Hal can transiently own 0x10A1.
                self._peak_strokes_this_hole = max(
                    self._peak_strokes_this_hole,
                    self._strokes_this_hole,
                )
            else:
                self._peak_strokes_this_hole = max(
                    self._peak_strokes_this_hole,
                    strokes,
                )
        elif (
            self._last_hole == self.max_holes
            and self._hole_index(world) >= self.max_holes
        ):
            # The final putt advances the raw index to 18 immediately. Its
            # incremented stroke byte still belongs to Hole 18.
            if self.profile.is_vs_hal:
                self._peak_strokes_this_hole = max(
                    self._peak_strokes_this_hole,
                    self._strokes_this_hole,
                )
            else:
                self._peak_strokes_this_hole = max(
                    self._peak_strokes_this_hole,
                    strokes,
                )

    def _hole_index(self, world: WorldState) -> int:
        if "hole_index" in world.info:
            return int(world.info["hole_index"])
        return read_u8(world.ram, WRAM_HOLE_INDEX)

    def _step_bootstrap(self, world: WorldState) -> TaskResult:
        """Advance the verified menu bootstrap frame script."""
        result = self._bootstrap.step(world)
        if result.status == TaskStatus.SUCCESS:
            self._phase = MissionPhase.PLAY_HOLE
            hole = read_hole_number(world.ram, world.info)
            if 1 <= hole <= self.max_holes:
                self._last_hole = hole
            # Pre-round RAM contains stale bytes from the title state.
            # Establish score baselines only after reaching Hole 1.
            self._card = ScorecardBook()
            self._peak_strokes_this_hole = max(
                0, read_stroke_count(world.ram, world.info)
            )
            self._strokes_this_hole = 0
            self._awaiting_stroke_reset = False
            self._start_shot(world)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=result.action or ActionResult(action=idle()),
            meta=self.progress_snapshot(),
        )

    def _maybe_advance_hole(
        self,
        world: WorldState,
        hole: int,
        strokes: int,
        rest: int,
    ) -> None:
        """Detect a new hole and reset per-hole tracking / transition state."""
        if not (1 <= hole <= self.max_holes):
            return
        if hole == self._last_hole:
            return
        if 1 <= self._last_hole <= self.max_holes:
            self._holes_completed += 1
            self._record_hole_score(world)
        self._last_hole = hole
        self._strokes_this_hole = 0
        # Hole index advances before the old stroke byte always clears. Start
        # the new hole at zero so the prior score does not contaminate this
        # hole's peak.
        self._peak_strokes_this_hole = 0
        self._awaiting_stroke_reset = strokes != 0
        self._waiting_after_hole_out = False
        self._shot = None
        self._phase = MissionPhase.NEXT_HOLE
        self._transition_frames = 0
        self._transition_taps = 0
        self._last_putt_rest = -1
        self._putt_retries = 0
        if self.profile.is_vs_hal and rest < 10:
            # Only burst on residual rest=0/1 green cameras. Doing this on a
            # live tee (rest=300+) keeps the command panel closed forever.
            burst: list[np.ndarray] = []
            for _ in range(50):
                burst.extend(named_script([("B", 3), ("IDLE", 1)]))
            self._dismiss_queue = burst

    def _check_completion(self, world: WorldState) -> TaskResult | None:
        """Return a terminal result when the course / match has finished."""
        is_vs_hal = self.profile.is_vs_hal
        # VS HAL ends on Hole 12; waiting for the stroke-play Hole 18 boundary
        # leaves a won match idling forever on its result screen.
        completion_holes = self.max_holes
        if is_vs_hal:
            completion_holes = min(completion_holes, VS_HAL_MATCH_HOLES)
        finished_course = self._holes_completed >= completion_holes or (
            self._last_hole == completion_holes
            and self._hole_index(world) >= completion_holes
        )
        # Match play can end early once the remaining holes cannot change the
        # lead (dormie / clinched).
        remaining = completion_holes - self._holes_completed
        match_clinched = (
            is_vs_hal
            and self._holes_completed > 0
            and remaining >= 0
            and abs(self.match_lead()) > remaining
        )
        if not (finished_course or match_clinched):
            return None
        if (
            self._last_hole == completion_holes
            and len(self._card.holes) < completion_holes
            and finished_course
        ):
            self._holes_completed = max(
                self._holes_completed,
                completion_holes,
            )
            self._record_hole_score(world)
        card = self.scorecard()
        if is_vs_hal:
            lead = self.match_lead()
            record = (
                f"{self._card.holes_won}-"
                f"{self._card.holes_lost}-"
                f"{self._card.holes_tied}"
            )
            if lead > 0:
                self._phase = MissionPhase.COMPLETE
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"match_won lead={lead} record={record} holes={card['holes']}",
                    meta={**self.progress_snapshot(), **card},
                )
            if lead < 0 and (finished_course or match_clinched):
                self._phase = MissionPhase.FAILED
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"match_lost lead={lead} record={record} holes={card['holes']}",
                    meta={**self.progress_snapshot(), **card},
                )
            if finished_course and lead == 0:
                self._phase = MissionPhase.FAILED
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"match_tied record={record} holes={card['holes']}",
                    meta={**self.progress_snapshot(), **card},
                )
        self._phase = MissionPhase.COMPLETE
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                f"course_complete total={card['total']} "
                f"holes={card['holes']}"
            ),
            meta={**self.progress_snapshot(), **card},
        )
