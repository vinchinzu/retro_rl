"""End-to-end stroke-play / VS HAL mission that can survive human takeover."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

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
    LIE_PUTTING_GREEN,
    WRAM_HOLE_INDEX,
    WRAM_LIE_TYPE,
    WRAM_OPPONENT_STROKE_COUNT,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
    read_hole_number,
    read_u16_le,
    read_u8,
)
from hals_golf.core.recovery import RecoveryController
from hals_golf.core.scene import classify_scene, is_command_screen
from hals_golf.tasks.menus import (
    ClubSet,
    Difficulty,
    MenuBootstrapTask,
    PlayMode,
    dismiss_scorecard_frames,
)
from hals_golf.tasks.profile import MissionProfile
from hals_golf.tasks.routes.tables import AMATEUR_PARS, VS_HAL_MATCH_HOLES
from hals_golf.tasks.shot import PuttTask, ShotTask
from hals_golf.tasks.shot_policy import (
    DeterministicRoutePolicy,
    RoutePolicy,
    ShotSituation,
)

# Cumulative command-menu frames on an unchanged lie before nudging aim/power.
COMMAND_STALL_FRAMES = 900
# Shot successes that leave hole/stroke/rest/lie unchanged before nudging.
FUTILE_SHOT_LIMIT = 3
# VS HAL: max frames to idle for Hal before forcing a resume attempt.
OPPONENT_WAIT_LIMIT = 720


def _hole_number(world: WorldState) -> int:
    return read_hole_number(world.ram, world.info)


def _stroke_count(world: WorldState) -> int:
    if "stroke_count" in world.info:
        return int(world.info["stroke_count"])
    return read_u8(world.ram, WRAM_STROKE_COUNT)


def _rest_distance(world: WorldState) -> int:
    if "rest_distance" in world.info:
        return int(world.info["rest_distance"])
    return read_u16_le(world.ram, WRAM_REST_DISTANCE)


def _hole_index(world: WorldState) -> int:
    if "hole_index" in world.info:
        return int(world.info["hole_index"])
    return read_u8(world.ram, WRAM_HOLE_INDEX)


def _lie_type(world: WorldState) -> int:
    if "lie_type" in world.info:
        return int(world.info["lie_type"])
    return read_u8(world.ram, WRAM_LIE_TYPE)


def _opponent_strokes(world: WorldState) -> int:
    return read_u8(world.ram, WRAM_OPPONENT_STROKE_COUNT)


class MissionPhase(Enum):
    BOOTSTRAP = auto()
    PLAY_HOLE = auto()
    WAIT_OPPONENT = auto()
    DISMISS_UI = auto()
    NEXT_HOLE = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class StrokePlayMission:
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
    _idle_frames: int = 0
    _transition_frames: int = 0
    _transition_taps: int = 0
    _last_putt_rest: int = -1
    _putt_retries: int = 0
    _recorded_total: int = 0
    _hole_scores: list[int] = field(default_factory=list)
    _hole_score_numbers: list[int] = field(default_factory=list)
    _opponent_hole_scores: list[int] = field(default_factory=list)
    _holes_won: int = 0
    _holes_lost: int = 0
    _holes_tied: int = 0
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
        self._strokes_this_hole = (
            max(0, _stroke_count(world)) if self.skip_bootstrap else 0
        )
        self._peak_strokes_this_hole = (
            max(0, _stroke_count(world)) if self.skip_bootstrap else 0
        )
        hole = _hole_number(world)
        self._last_hole = hole if 1 <= hole <= self.max_holes else -1
        self._idle_frames = 0
        self._transition_frames = 0
        self._transition_taps = 0
        self._last_putt_rest = -1
        self._putt_retries = 0
        self._recorded_total = 0
        self._hole_scores = []
        self._hole_score_numbers = []
        self._opponent_hole_scores = []
        self._holes_won = 0
        self._holes_lost = 0
        self._holes_tied = 0
        self._stall_key = (-1, -1, -1, -1)
        self._stall_frames = 0
        self._stall_nudges = 0
        self._futile_shots = 0
        self._shot_start_key = (-1, -1, -1, -1)
        self._last_rest = _rest_distance(world)
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
        total = self._recorded_total
        if self.play_mode is PlayMode.VS_HAL:
            objective = (
                f"hole={hole} strokes={self._strokes_this_hole} "
                f"done={self._holes_completed}/{self.max_holes} "
                f"match={self._holes_won}-{self._holes_lost}-{self._holes_tied}"
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
            "total": int(self._recorded_total),
            "stall": int(self._stall_frames),
            "holes_won": int(self._holes_won),
            "holes_lost": int(self._holes_lost),
            "holes_tied": int(self._holes_tied),
            "mode": self.play_mode.name,
        }

    def scorecard(self) -> dict[str, int | list[int]]:
        """Return recorded scores, pars, round total, and per-hole regressions."""
        numbers = list(self._hole_score_numbers)
        pars = [AMATEUR_PARS[number - 1] for number in numbers]
        over_par = [
            number
            for number, score, par in zip(numbers, self._hole_scores, pars)
            if score > par
        ]
        card: dict[str, int | list[int]] = {
            "holes": list(self._hole_scores),
            "hole_numbers": numbers,
            "pars": pars,
            "total": int(self._recorded_total),
            "to_par": int(sum(self._hole_scores) - sum(pars)),
            "over_par_holes": over_par,
            "holes_completed": int(len(self._hole_scores)),
            "holes_won": int(self._holes_won),
            "holes_lost": int(self._holes_lost),
            "holes_tied": int(self._holes_tied),
            "opponent_holes": list(self._opponent_hole_scores),
        }
        return card

    def match_lead(self) -> int:
        """Return holes up (positive) or down (negative) versus Hal."""
        return int(self._holes_won - self._holes_lost)

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
        if scored > 0:
            self._hole_scores.append(scored)
            self._hole_score_numbers.append(self._last_hole)
            if self.play_mode is PlayMode.VS_HAL:
                opponent = _opponent_strokes(world)
                if opponent <= 0:
                    opponent = scored
                self._opponent_hole_scores.append(opponent)
                if scored < opponent:
                    self._holes_won += 1
                elif scored > opponent:
                    self._holes_lost += 1
                else:
                    self._holes_tied += 1
        self._recorded_total = sum(self._hole_scores)
        self._peak_strokes_this_hole = 0
        self._stall_key = (-1, -1, -1, -1)
        self._stall_frames = 0
        self._stall_nudges = 0
        self._futile_shots = 0

    def step(self, world: WorldState) -> TaskResult:
        """Thin dispatcher: sync tracking, then delegate to a phase helper."""
        decision = classify_scene(world.ram, info=world.info, obs=world.obs)
        hole = _hole_number(world)
        strokes = _stroke_count(world)
        rest = _rest_distance(world)
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
            and _hole_index(world) >= self.max_holes
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

    def _step_bootstrap(self, world: WorldState) -> TaskResult:
        """Advance the verified menu bootstrap frame script."""
        result = self._bootstrap.step(world)
        if result.status == TaskStatus.SUCCESS:
            self._phase = MissionPhase.PLAY_HOLE
            hole = _hole_number(world)
            if 1 <= hole <= self.max_holes:
                self._last_hole = hole
            # Pre-round RAM contains stale bytes from the title state.
            # Establish score baselines only after reaching Hole 1.
            self._recorded_total = 0
            self._peak_strokes_this_hole = max(0, _stroke_count(world))
            self._strokes_this_hole = 0
            self._hole_scores = []
            self._hole_score_numbers = []
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
            and _hole_index(world) >= completion_holes
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
            and len(self._hole_scores) < completion_holes
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
            if lead > 0:
                self._phase = MissionPhase.COMPLETE
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=(
                        f"match_won lead={lead} "
                        f"record={self._holes_won}-"
                        f"{self._holes_lost}-{self._holes_tied} "
                        f"holes={card['holes']}"
                    ),
                    meta={**self.progress_snapshot(), **card},
                )
            if lead < 0 and (finished_course or match_clinched):
                self._phase = MissionPhase.FAILED
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"match_lost lead={lead} "
                        f"record={self._holes_won}-"
                        f"{self._holes_lost}-{self._holes_tied} "
                        f"holes={card['holes']}"
                    ),
                    meta={**self.progress_snapshot(), **card},
                )
            if finished_course and lead == 0:
                self._phase = MissionPhase.FAILED
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"match_tied "
                        f"record={self._holes_won}-"
                        f"{self._holes_lost}-{self._holes_tied} "
                        f"holes={card['holes']}"
                    ),
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

    def _step_next_hole(
        self,
        world: WorldState,
        strokes: int,
        rest: int,
    ) -> TaskResult:
        """Wait for the next tee / command panel, then start the shot."""
        self._transition_frames += 1
        lie = _lie_type(world)
        # Live tee: wait for the command panel. The old 90-frame fallback
        # started swings during the hole intro and produced phantom "success"
        # completes with REST unchanged (H2/H3 blow-ups after H1).
        tee_ready = (
            strokes == 0
            and rest >= 150
            and is_command_screen(world.obs)
            and (
                self._transition_frames >= 30
                or not self.profile.is_vs_hal
            )
        )
        if self.profile.is_vs_hal:
            tee_ready = strokes == 0 and rest >= 150 and (
                is_command_screen(world.obs)
                or self._transition_frames >= 90
            )
        short_ready = (
            is_command_screen(world.obs)
            and strokes == 0
            and 10 < rest < 150
            and lie in (1, 2, 3, 6)
            and self._transition_frames >= 180
        )
        if tee_ready or short_ready:
            self._phase = MissionPhase.PLAY_HOLE
            self._transition_frames = 0
            self._start_shot(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason="next_hole_ready"),
                meta=self.progress_snapshot(),
            )
        # Residual rest=0/1: Hal green camera — keep dismissing.
        # Live tees (rest>=150) idle; do not mash B (closes SHOT panel).
        if rest < 10:
            tap_period = 60 if self.profile.is_vs_hal else 360
            if (
                self._transition_frames >= tap_period
                and self._transition_frames % tap_period < 3
            ):
                self._transition_taps += 1
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(
                        action=self._confirm_action(),
                        reason="next_hole",
                    ),
                    meta=self.progress_snapshot(),
                )
        elif (
            not self.profile.is_vs_hal
            and self._transition_frames >= 360
            and self._transition_frames % 360 < 3
        ):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(
                    action=self._confirm_action(),
                    reason="next_hole",
                ),
                meta=self.progress_snapshot(),
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action=idle(), reason="next_hole_wait"),
            meta=self.progress_snapshot(),
        )

    def _step_wait_opponent(
        self,
        world: WorldState,
        rest: int,
    ) -> TaskResult:
        """Idle through Hal's turn until our command panel returns."""
        # Kept for resume/tests: treat as a soft idle that returns to play as
        # soon as the command panel or a sane lie is available.
        self._opponent_wait_frames += 1
        if self._waiting_after_hole_out:
            # Our ball is already in the cup.  The command panel now belongs to
            # Hal, even though it is visually identical to ours; never issue
            # another shot until the hole index moves.
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason="opponent_finishing"),
                meta=self.progress_snapshot(),
            )
        if 0 < rest < 1000 and (
            is_command_screen(world.obs)
            or self._opponent_wait_frames >= OPPONENT_WAIT_LIMIT
        ):
            self._phase = MissionPhase.PLAY_HOLE
            self._opponent_wait_frames = 0
            if is_command_screen(world.obs):
                self._start_shot(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason="our_turn"),
                meta=self.progress_snapshot(),
            )
        if (
            not is_command_screen(world.obs)
            and self._opponent_wait_frames % 180 < 3
        ):
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(
                    action=self._confirm_action(),
                    reason="wait_opponent_skip",
                ),
                meta=self.progress_snapshot(),
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action=idle(), reason="wait_opponent"),
            meta=self.progress_snapshot(),
        )

    def _step_play_hole(
        self,
        world: WorldState,
        hole: int,
        strokes: int,
        rest: int,
    ) -> TaskResult:
        """Issue and settle one shot / putt on the current hole."""
        if (
            self._strokes_this_hole >= self.max_strokes_per_hole
            and self._shot is None
            and not self._dismiss_queue
        ):
            # Soft fail-forward once, then keep playing with nudged plans. Do
            # not re-enter every frame or we spin on dismiss forever.
            self._dismiss_queue = dismiss_scorecard_frames()
            self._stall_nudges += 1
            self._strokes_this_hole = max(0, self.max_strokes_per_hole - 3)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(
                    action=self._dismiss_queue.pop(0), reason="stroke_cap"
                ),
                meta=self.progress_snapshot(),
            )

        stall_action = self._maybe_break_command_stall(world, hole, strokes, rest)
        if stall_action is not None:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=stall_action, reason="stall_nudge"),
                meta=self.progress_snapshot(),
            )

        # VS HAL: never mash SHOT while the command panel is absent (Hal
        # animations / flyovers). Idle only; NEXT_HOLE covers post-hole Hal.
        if (
            self.profile.is_vs_hal
            and self._shot is None
            and not is_command_screen(world.obs)
        ):
            self._no_command_frames += 1
            # After a long quiet stretch, force a SHOT open — the panel
            # heuristic sometimes misses VS HAL HUD variants.
            if self._no_command_frames >= 900 and 30 < rest < 1000:
                self._no_command_frames = 0
                self._start_shot(world)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(action=idle(), reason="force_shot"),
                    meta=self.progress_snapshot(),
                )
            if (
                self._no_command_frames >= 240
                and self._no_command_frames % 240 < 3
            ):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(
                        action=self._confirm_action(),
                        reason="vs_hal_skip",
                    ),
                    meta=self.progress_snapshot(),
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action=idle(), reason="await_command"),
                meta=self.progress_snapshot(),
            )

        if self._shot is None:
            self._no_command_frames = 0
            self._start_shot(world)

        assert self._shot is not None
        result = self._shot.step(world)
        if result.status == TaskStatus.FAILURE:
            # Shot timed out without REST moving (common on VS HAL when a
            # phantom stroke byte advances). Nudge and retry.
            self._futile_shots += 1
            if self._futile_shots >= FUTILE_SHOT_LIMIT:
                self._futile_shots = 0
                self._stall_nudges += 1
                self._recovery.start(reason="rest_unchanged")
            self._shot = None
            self._no_command_frames = 0
            if is_command_screen(world.obs):
                self._start_shot(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=result.action or ActionResult(action=idle()),
                meta=self.progress_snapshot(),
            )
        if result.status == TaskStatus.SUCCESS:
            # Key on rest/lie only — VS HAL stroke bytes are unreliable.
            rest_key = (hole, rest, _lie_type(world))
            start_rest_key = (
                self._shot_start_key[0],
                self._shot_start_key[2],
                self._shot_start_key[3],
            )
            ball_moved = rest_key != start_rest_key
            # Phantom completes must not advance the plan index. This used to
            # be VS HAL-only; stroke-play hole transitions hit the same trap.
            if ball_moved:
                self._strokes_this_hole += 1
            if not ball_moved:
                self._futile_shots += 1
            else:
                self._futile_shots = 0
            if self._futile_shots >= FUTILE_SHOT_LIMIT:
                self._futile_shots = 0
                self._stall_nudges += 1
                self._recovery.start(reason="futile_shots")
            self._shot = None
            self._no_command_frames = 0
            if rest == 0:
                # Hole-out (including stroke-play HIO): do not open SHOT on the
                # scorecard / flyover. VS HAL waits for Hal; stroke play waits
                # for the next tee command panel.
                if self.profile.is_vs_hal:
                    self._phase = MissionPhase.WAIT_OPPONENT
                    self._opponent_wait_frames = 0
                else:
                    self._phase = MissionPhase.NEXT_HOLE
                    self._transition_frames = 0
                self._waiting_after_hole_out = True
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=result.action or ActionResult(action=idle()),
                    meta=self.progress_snapshot(),
                )
            if self.profile.is_vs_hal and not is_command_screen(world.obs):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=result.action or ActionResult(action=idle()),
                    meta=self.progress_snapshot(),
                )
            self._start_shot(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=result.action or ActionResult(action=idle()),
                meta=self.progress_snapshot(),
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=result.action,
            meta=self.progress_snapshot(),
        )

    def _maybe_break_command_stall(
        self,
        world: WorldState,
        hole: int,
        strokes: int,
        rest: int,
    ) -> np.ndarray | None:
        """Nudge aim/power when the same lie keeps returning to the command menu."""
        key = (hole, strokes, rest, _lie_type(world))
        if key != self._stall_key:
            self._stall_key = key
            self._stall_frames = 0
            self._futile_shots = 0
            return None
        if not is_command_screen(world.obs):
            return None
        self._stall_frames += 1
        if self._stall_frames < COMMAND_STALL_FRAMES:
            return None
        self._stall_frames = 0
        self._stall_nudges += 1
        self._recovery.start(reason="command_stall")
        self._shot = None
        self._start_shot(world)
        return idle()

    def _start_shot(self, world: WorldState) -> None:
        rest = _rest_distance(world)
        lie = _lie_type(world)
        hole = _hole_number(world)
        self._shot_start_key = (
            hole,
            _stroke_count(world),
            rest,
            lie,
        )
        profile = self.profile
        # Prefer our successful-shot counter for plan lookup. The on-screen
        # stroke byte advances on swings that never move the ball (futile
        # WAIT_FLIGHT retries), which skipped H18's 3W and fired the SW from
        # the tee corridor after the H13-birdie timeline.
        strokes = self._strokes_this_hole
        situation = ShotSituation(
            hole=hole,
            strokes=strokes,
            rest=rest,
            lie=lie,
            stall_nudges=self._stall_nudges,
            putt_retries=self._putt_retries,
            last_putt_rest=self._last_putt_rest,
            default_power=self.power_delay,
        )
        if rest > 0 and lie == LIE_PUTTING_GREEN:
            intent = self.route_policy.plan_putt(situation, profile)
            self._putt_retries = intent.putt_retries
            self._last_putt_rest = intent.last_putt_rest
            self._shot = PuttTask(
                power_delay=intent.power,
                flight_wait=1200,
                require_rest_change=intent.require_rest_change,
                complete_on_rest_zero=intent.complete_on_rest_zero,
            )
        else:
            self._last_putt_rest = -1
            self._putt_retries = 0
            intent = self.route_policy.plan_shot(situation, profile)
            self._shot = ShotTask(
                power_delay=intent.power,
                impact_delay=self.impact_delay,
                flight_wait=1400,
                aim_steps=intent.aim,
                club_downs=intent.club_downs,
                require_rest_change=intent.require_rest_change,
                complete_on_rest_zero=intent.complete_on_rest_zero,
            )
        self._shot.reset(world)

    @staticmethod
    def _confirm_action() -> np.ndarray:
        from hals_golf.core.actions import CONFIRM, press

        return press(CONFIRM)
