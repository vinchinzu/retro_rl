"""Short-horizon, frame-costed search for reactive-policy handoffs.

The adapter starts from the *exact* live emulator state (including subpixels,
velocity, momentum, pose, and enemy phase), branches a small set of timed
button pulses, and restores the original state before returning.  It is used
for both different door-entry kinematics and human → autopilot handoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np

from super_metroid.ram import SuperMetroidState, parse_env_state
from super_metroid.reactive_policy import (
    ACTION_SIZE,
    ReactivePolicyRunner,
    ReferenceSample,
)

StateT = TypeVar("StateT")
TokenT = TypeVar("TokenT")


def _action(value: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(v) for v in value)
    if len(out) != ACTION_SIZE or any(v not in (0, 1) for v in out):
        raise ValueError("adapter actions must be SNES-12 binary arrays")
    return out


@dataclass(frozen=True)
class TimedAction:
    buttons: tuple[int, ...]
    frames: int
    label: str = ""

    def __post_init__(self) -> None:
        _action(self.buttons)
        if self.frames <= 0:
            raise ValueError("TimedAction.frames must be positive")


@dataclass(frozen=True)
class AdapterSearchConfig:
    beam_width: int = 12
    max_depth: int = 5
    frame_penalty: float = 0.35


@dataclass(frozen=True)
class AdapterPlan:
    frames: tuple[tuple[int, ...], ...]
    score_before: float
    score_after: float
    expanded: int
    reached: bool

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass
class _Node(Generic[StateT, TokenT]):
    token: TokenT
    state: StateT
    frames: tuple[tuple[int, ...], ...]
    score: float


def beam_search_adapter(
    initial_token: TokenT,
    initial_state: StateT,
    *,
    actions: Sequence[TimedAction],
    expand: Callable[[TokenT, TimedAction], tuple[TokenT, StateT]],
    score: Callable[[StateT], float],
    reached: Callable[[StateT, float], bool],
    state_key: Callable[[StateT], tuple[Any, ...]],
    config: AdapterSearchConfig = AdapterSearchConfig(),
) -> AdapterPlan:
    """Generic deterministic beam search, independently unit-testable."""
    if not actions:
        raise ValueError("adapter search needs at least one timed action")
    score_before = float(score(initial_state))
    if reached(initial_state, score_before):
        return AdapterPlan((), score_before, score_before, 0, True)

    root = _Node(initial_token, initial_state, (), score_before)
    beam = [root]
    best = root
    best_goal: _Node[StateT, TokenT] | None = None
    expanded = 0

    for _depth in range(max(1, config.max_depth)):
        children: list[_Node[StateT, TokenT]] = []
        for node in beam:
            for timed in actions:
                token, child_state = expand(node.token, timed)
                child_frames = node.frames + (timed.buttons,) * timed.frames
                child_score = float(score(child_state))
                child = _Node(token, child_state, child_frames, child_score)
                children.append(child)
                expanded += 1
                if (child.score, len(child.frames)) < (best.score, len(best.frames)):
                    best = child
                if reached(child_state, child_score):
                    if best_goal is None or (len(child.frames), child.score) < (
                        len(best_goal.frames),
                        best_goal.score,
                    ):
                        best_goal = child

        if best_goal is not None:
            best = best_goal
            break

        # Quantized state de-duplication prevents identical idle / held branches
        # from consuming the whole beam.
        unique: dict[tuple[Any, ...], _Node[StateT, TokenT]] = {}
        for child in children:
            key = state_key(child.state)
            incumbent = unique.get(key)
            rank = child.score + config.frame_penalty * len(child.frames)
            if incumbent is None or rank < (
                incumbent.score + config.frame_penalty * len(incumbent.frames)
            ):
                unique[key] = child
        beam = sorted(
            unique.values(),
            key=lambda node: (
                node.score + config.frame_penalty * len(node.frames),
                len(node.frames),
            ),
        )[: max(1, config.beam_width)]
        if not beam:
            break

    return AdapterPlan(
        frames=best.frames,
        score_before=score_before,
        score_after=best.score,
        expanded=expanded,
        reached=best_goal is not None,
    )


def _buttons(*indices: int) -> tuple[int, ...]:
    out = [0] * ACTION_SIZE
    for index in indices:
        out[index] = 1
    return tuple(out)


def adapter_action_library(
    target: ReferenceSample,
    nearby_actions: Sequence[Sequence[int]] = (),
    *,
    pulse_lengths: Sequence[int] = (1, 2, 4),
) -> tuple[TimedAction, ...]:
    """Reference-aware movement pulses for exact frame/timing search."""
    # SNES order: B,Y,Select,Start,Up,Down,Left,Right,A,X,L,R.
    base = [
        tuple(target.action),
        _buttons(),
        _buttons(6),
        _buttons(7),
        _buttons(6, 8),
        _buttons(7, 8),
        _buttons(0, 6, 8),
        _buttons(0, 7, 8),
        _buttons(8),
        _buttons(9),
    ]
    base.extend(_action(value) for value in nearby_actions)
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in base:
        action = _action(value)
        if action not in seen:
            seen.add(action)
            unique.append(action)
    return tuple(
        TimedAction(action, int(frames), label="rejoin")
        for frames in pulse_lengths
        if int(frames) > 0
        for action in unique
    )


def _state_key(state: SuperMetroidState) -> tuple[int, ...]:
    return (
        int(state.room_id),
        int(state.samus_x) // 4,
        int(state.samus_y) // 4,
        int(state.velocity_x),
        int(state.velocity_y),
        int(state.momentum_x),
        int(state.pose),
        int(state.facing),
        int(state.movement_type),
    )


def search_live_adapter(
    env: Any,
    runner: ReactivePolicyRunner,
    *,
    config: AdapterSearchConfig = AdapterSearchConfig(),
) -> AdapterPlan:
    """Search from live emulator state and restore it before returning."""
    original = env.em.get_state()
    initial = parse_env_state(env, mode="nav")
    target = runner.project_global(initial)
    trajectory = runner.variant.trajectories[target.trajectory_index]
    lo = max(0, target.sample_index - 8)
    hi = min(len(trajectory.samples), target.sample_index + 16)
    nearby = [trajectory.samples[i].action for i in range(lo, hi)]
    actions = adapter_action_library(target.sample, nearby)

    def expand(token: bytes, timed: TimedAction) -> tuple[bytes, SuperMetroidState]:
        env.em.set_state(token)
        raw = np.asarray(timed.buttons, dtype=np.int8)
        state = parse_env_state(env, mode="nav")
        for _ in range(timed.frames):
            env.step(raw)
            state = parse_env_state(env, mode="nav")
        return env.em.get_state(), state

    def score(state: SuperMetroidState) -> float:
        if int(state.room_id) != int(initial.room_id):
            # Leaving through an arbitrary door is not an adapter success.
            return 1_000_000.0
        return runner.project_global(state).score

    try:
        return beam_search_adapter(
            original,
            initial,
            actions=actions,
            expand=expand,
            score=score,
            reached=lambda _state, value: value <= runner.variant.rejoin_threshold,
            state_key=_state_key,
            config=config,
        )
    finally:
        env.em.set_state(original)


__all__ = [
    "AdapterPlan",
    "AdapterSearchConfig",
    "TimedAction",
    "adapter_action_library",
    "beam_search_adapter",
    "search_live_adapter",
]
