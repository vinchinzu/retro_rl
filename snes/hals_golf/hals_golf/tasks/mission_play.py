"""In-round phase helpers mixed into ``StrokePlayMission``."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from retro_harness.protocol import (
    ActionResult,
    TaskResult,
    TaskStatus,
    WorldState,
)

from hals_golf.core.actions import idle
from hals_golf.core.ram import (
    LIE_PUTTING_GREEN,
    read_hole_number,
    read_lie_type,
    read_rest_distance,
    read_stroke_count,
)
from hals_golf.core.scene import is_command_screen
from hals_golf.tasks.menus import dismiss_scorecard_frames
from hals_golf.tasks.shot import PuttTask, ShotTask
from hals_golf.tasks.shot_policy import ShotSituation

# Cumulative command-menu frames on an unchanged lie before nudging aim/power.
COMMAND_STALL_FRAMES = 900
# Shot successes that leave hole/stroke/rest/lie unchanged before nudging.
FUTILE_SHOT_LIMIT = 3
# VS HAL: max frames to idle for Hal before forcing a resume attempt.
OPPONENT_WAIT_LIMIT = 720


class MissionPhase(Enum):
    BOOTSTRAP = auto()
    PLAY_HOLE = auto()
    WAIT_OPPONENT = auto()
    DISMISS_UI = auto()
    NEXT_HOLE = auto()
    COMPLETE = auto()
    FAILED = auto()


class InRoundPlay:
    """Phase methods mixed into ``StrokePlayMission`` (no extra fields)."""

    def _step_next_hole(
        self,
        world: WorldState,
        strokes: int,
        rest: int,
    ) -> TaskResult:
        """Wait for the next tee / command panel, then start the shot."""
        self._transition_frames += 1
        lie = read_lie_type(world.ram, world.info)
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
            rest_key = (hole, rest, read_lie_type(world.ram, world.info))
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
        key = (hole, strokes, rest, read_lie_type(world.ram, world.info))
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
        rest = read_rest_distance(world.ram, world.info)
        lie = read_lie_type(world.ram, world.info)
        hole = read_hole_number(world.ram, world.info)
        self._shot_start_key = (
            hole,
            read_stroke_count(world.ram, world.info),
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
