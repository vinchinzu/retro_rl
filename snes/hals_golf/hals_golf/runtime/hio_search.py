"""Evaluate hole-in-one tee candidates from a fixed save state.

The search loop is deliberately outside ``StrokePlayMission``: reload the
state, fire one ``ShotTask`` per ``ShotIntent``, and score the settled REST.
Default clears keep ``DeterministicRoutePolicy``; this module is the HIO
exploration entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from retro_harness.env import make_env
from retro_harness.protocol import TaskStatus, WorldState

from hals_golf.core.actions import idle
from hals_golf.core.ram import (
    WRAM_HOLE_INDEX,
    WRAM_LIE_TYPE,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
    read_hole_number,
    read_u16_le,
    read_u8,
)
from hals_golf.paths import GAME, PROJECT_DIR
from hals_golf.tasks.menus import ClubSet, Difficulty, PlayMode
from hals_golf.tasks.profile import MissionProfile
from hals_golf.tasks.shot import ShotTask
from hals_golf.tasks.shot_policy import (
    HoleInOneSearchPolicy,
    SearchSpec,
    ShotIntent,
    ShotSituation,
)


@dataclass(frozen=True)
class CandidateResult:
    """Outcome of one tee-shot candidate after ball flight settles."""

    index: int
    intent: ShotIntent
    start_rest: int
    end_rest: int
    end_strokes: int
    frames: int
    status: str

    @property
    def hole_in_one(self) -> bool:
        return self.end_rest == 0 and self.end_strokes <= 1

    @property
    def rest_delta(self) -> int:
        return self.start_rest - self.end_rest


def situation_from_ram(ram: np.ndarray, *, default_power: int = 42) -> ShotSituation:
    """Build a planner situation from a WRAM snapshot."""
    return ShotSituation(
        hole=read_hole_number(ram),
        strokes=int(read_u8(ram, WRAM_STROKE_COUNT)),
        rest=int(read_u16_le(ram, WRAM_REST_DISTANCE)),
        lie=int(read_u8(ram, WRAM_LIE_TYPE)),
        default_power=default_power,
    )


def _world_from_env(env: object, frame: int, obs: np.ndarray) -> WorldState:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    return WorldState(
        frame=frame,
        ram=ram,
        info={
            "hole_index": int(read_u8(ram, WRAM_HOLE_INDEX)),
            "stroke_count": int(read_u8(ram, WRAM_STROKE_COUNT)),
            "rest_distance": int(read_u16_le(ram, WRAM_REST_DISTANCE)),
            "lie_type": int(read_u8(ram, WRAM_LIE_TYPE)),
        },
        obs=obs,
    )


def _run_intent(
    env: object,
    intent: ShotIntent,
    *,
    impact_delay: int,
    max_frames: int,
) -> CandidateResult:
    obs, _info = env.reset()
    world = _world_from_env(env, 0, obs)
    start_rest = int(world.info["rest_distance"])
    task = ShotTask(
        power_delay=intent.power,
        impact_delay=impact_delay,
        flight_wait=1400,
        aim_steps=intent.aim,
        club_downs=intent.club_downs,
        require_rest_change=intent.require_rest_change,
        complete_on_rest_zero=True,
    )
    task.reset(world)
    status = TaskStatus.RUNNING
    frames = 0
    for frame in range(1, max_frames + 1):
        result = task.step(world)
        status = result.status
        action = result.action.action if result.action is not None else idle()
        obs, _reward, terminated, truncated, _info = env.step(action)
        world = _world_from_env(env, frame, obs)
        frames = frame
        if status != TaskStatus.RUNNING or terminated or truncated:
            break
    end_rest = int(world.info["rest_distance"])
    end_strokes = int(world.info["stroke_count"])
    return CandidateResult(
        index=-1,
        intent=intent,
        start_rest=start_rest,
        end_rest=end_rest,
        end_strokes=end_strokes,
        frames=frames,
        status=status.value,
    )


def search_tee_candidates(
    *,
    state: str,
    play_mode: PlayMode = PlayMode.STROKE_PLAY,
    club_set: ClubSet = ClubSet.STANDARD,
    difficulty: Difficulty = Difficulty.AMATEUR,
    impact_delay: int = 26,
    max_frames_per_candidate: int = 2500,
    max_candidates: int | None = None,
    spec: SearchSpec | None = None,
) -> tuple[ShotSituation, list[CandidateResult]]:
    """Reload ``state`` for each HIO neighborhood candidate and score REST.

    Returns the opening situation plus per-candidate results ordered the same
    as ``HoleInOneSearchPolicy.candidates`` (deterministic base first).
    """
    import stable_retro as retro

    from hals_golf.runtime.retro_setup import register_golf_integration

    register_golf_integration(retro, quiet=True)
    profile = MissionProfile(
        play_mode=play_mode,
        club_set=club_set,
        difficulty=difficulty,
    )
    search_spec = spec or SearchSpec()
    if max_candidates is not None:
        search_spec = SearchSpec(
            power_deltas=search_spec.power_deltas,
            aim_deltas=search_spec.aim_deltas,
            club_deltas=search_spec.club_deltas,
            max_candidates=max_candidates,
            power_min=search_spec.power_min,
            power_max=search_spec.power_max,
        )
    policy = HoleInOneSearchPolicy(spec=search_spec)

    env = make_env(
        game=GAME,
        state=state,
        game_dir=PROJECT_DIR,
        render_mode="rgb_array",
    )
    results: list[CandidateResult] = []
    try:
        obs, _info = env.reset()
        world = _world_from_env(env, 0, obs)
        situation = situation_from_ram(world.ram)
        intents = list(policy.candidates(situation, profile))
        for index, intent in enumerate(intents):
            outcome = _run_intent(
                env,
                intent,
                impact_delay=impact_delay,
                max_frames=max_frames_per_candidate,
            )
            results.append(
                CandidateResult(
                    index=index,
                    intent=outcome.intent,
                    start_rest=outcome.start_rest,
                    end_rest=outcome.end_rest,
                    end_strokes=outcome.end_strokes,
                    frames=outcome.frames,
                    status=outcome.status,
                )
            )
    finally:
        env.close()
    return situation, results
