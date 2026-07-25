"""Reusable timed inputs for title screens, menus, and deterministic macros."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, NamedTuple

from retro_harness.actions import SNES_ACTION_SIZE, idle_action, snes_action
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX, SNES_BUTTON_NAMES
from retro_harness.runtime import reset_env, step_env

NOOP_TOKENS = frozenset({"IDLE", "WAIT", "NOOP", "NONE"})


class InputStep(NamedTuple):
    """Hold indexed buttons, then release them for a number of frames."""

    buttons: list[int]
    hold_frames: int
    wait_frames: int


@dataclass(frozen=True)
class FrameAction:
    """One controller frame plus a short diagnostic label."""

    action: list[int]
    reason: str = ""


class PrimitiveKind(Enum):
    """Common labels for fixed-duration controller primitives."""

    IDLE = auto()
    WALK_RIGHT = auto()
    WALK_LEFT = auto()
    WALK_UP = auto()
    WALK_DOWN = auto()
    JUMP = auto()
    JUMP_RIGHT = auto()
    ATTACK = auto()
    START = auto()
    CONFIRM = auto()


@dataclass(frozen=True)
class ControllerPrimitive:
    """Hold a named button combination for a fixed number of frames."""

    kind: PrimitiveKind
    duration: int
    hold: tuple[str, ...] = ()
    reason: str = ""

    def frames(self) -> Iterator[FrameAction]:
        if self.duration < 0:
            raise ValueError("duration must be >= 0")
        action = snes_action(*self.hold)
        reason = self.reason or self.kind.name.lower()
        for _ in range(self.duration):
            yield FrameAction(action=list(action), reason=reason)


@dataclass(frozen=True)
class StartupPlan:
    """Declarative cold-boot/title/menu path to a game's first playable screen."""

    steps: tuple[InputStep, ...]
    name: str = "startup"

    @classmethod
    def title_menu(
        cls,
        *menu_moves: str,
        initial_wait: int = 60,
        start_button: str = "START",
        start_hold: int = 2,
        start_wait: int = 90,
        move_hold: int = 2,
        move_wait: int = 12,
        confirm_button: str | None = "A",
        confirm_hold: int = 2,
        confirm_wait: int = 180,
        name: str = "title_menu",
    ) -> StartupPlan:
        """Build the common wait → START → menu moves → confirm sequence."""

        steps: list[InputStep] = []
        if initial_wait:
            steps.append(input_step(hold=0, wait=initial_wait))
        steps.append(input_step(start_button, hold=start_hold, wait=start_wait))
        steps.extend(
            input_step(move, hold=move_hold, wait=move_wait) for move in menu_moves
        )
        if confirm_button:
            steps.append(
                input_step(
                    confirm_button,
                    hold=confirm_hold,
                    wait=confirm_wait,
                )
            )
        return cls(tuple(steps), name=name)

    @classmethod
    def parse(cls, script: str, *, name: str = "startup") -> StartupPlan:
        """Build a plan from compact ``BUTTONS:hold:wait`` tokens."""

        return cls(tuple(parse_input_script(script)), name=name)


@dataclass(frozen=True)
class ScriptResult:
    """Outcome of a startup or deterministic input script."""

    frames: int
    completed: bool
    ready: bool
    terminated: bool
    truncated: bool
    observation: Any = None
    info: dict[str, Any] | None = None


def _button_index(name: str) -> int | None:
    key = name.strip().upper()
    if key in NOOP_TOKENS:
        return None
    if key not in SNES_BUTTON_NAME_TO_INDEX:
        valid = ", ".join((*SNES_BUTTON_NAMES, *sorted(NOOP_TOKENS)))
        raise ValueError(f"Unknown button {name!r}. Valid: {valid}")
    return SNES_BUTTON_NAME_TO_INDEX[key]


def input_step(*button_names: str, hold: int = 1, wait: int = 0) -> InputStep:
    """Create an indexed timed step from canonical button names."""

    if hold < 0 or wait < 0:
        raise ValueError("hold and wait frames must be >= 0")
    indices = [
        index for name in button_names if (index := _button_index(name)) is not None
    ]
    return InputStep(indices, hold, wait)


def parse_input_script(script: str) -> list[InputStep]:
    """Parse ``BUTTON[+BUTTON]:hold:wait`` tokens into reusable input steps."""

    steps: list[InputStep] = []
    for token in script.strip().split():
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid input step {token!r}: expected BUTTONS:hold:wait"
            )
        button_text, hold_text, wait_text = parts
        names = (
            () if button_text.upper() in NOOP_TOKENS else tuple(button_text.split("+"))
        )
        steps.append(input_step(*names, hold=int(hold_text), wait=int(wait_text)))
    return steps


def iter_input_steps(
    steps: Iterable[InputStep],
    *,
    action_size: int = SNES_ACTION_SIZE,
) -> Iterator[FrameAction]:
    """Expand timed steps into per-frame actions without allocating the trace."""

    for step in steps:
        if step.hold_frames < 0 or step.wait_frames < 0:
            raise ValueError("hold and wait frames must be >= 0")
        held = [0] * action_size
        for index in step.buttons:
            if not 0 <= index < action_size:
                raise ValueError(
                    f"button index {index} outside action size {action_size}"
                )
            held[index] = 1
        names = (
            "+".join(
                SNES_BUTTON_NAMES[index]
                for index in step.buttons
                if index < len(SNES_BUTTON_NAMES)
            )
            or "idle"
        )
        for _ in range(step.hold_frames):
            yield FrameAction(list(held), names.lower())
        for _ in range(step.wait_frames):
            yield FrameAction(idle_action(action_size=action_size), "wait")


def repeat_action(
    action: Sequence[int], frames: int, *, dtype: Any | None = None
) -> list[Any]:
    """Copy one action for a fixed number of frames."""

    count = max(0, frames)
    if dtype is None:
        return [list(action) for _ in range(count)]

    import numpy as np

    return [np.asarray(action, dtype=dtype).copy() for _ in range(count)]


def press_button_sequence(
    button: str,
    *,
    face: str | None = None,
    face_frames: int = 0,
    pre_press_settle_frames: int = 0,
    hold_frames: int = 1,
    settle_frames: int = 0,
    hold_face_with_button: bool = False,
    action_size: int = SNES_ACTION_SIZE,
    dtype: Any | None = None,
) -> list[Any]:
    """Build a face, settle, press, release sequence for interactions/menus."""

    actions: list[Any] = []
    if face:
        actions.extend(
            snes_action(face, action_size=action_size, dtype=dtype)
            for _ in range(max(0, face_frames))
        )
    actions.extend(
        idle_action(action_size=action_size, dtype=dtype)
        for _ in range(max(0, pre_press_settle_frames))
    )
    held = (face, button) if face and hold_face_with_button else (button,)
    actions.extend(
        snes_action(*held, action_size=action_size, dtype=dtype)
        for _ in range(max(0, hold_frames))
    )
    actions.extend(
        idle_action(action_size=action_size, dtype=dtype)
        for _ in range(max(0, settle_frames))
    )
    return actions


def run_input_steps(
    env: Any,
    steps: Iterable[InputStep],
    *,
    is_ready: Callable[[Any, dict[str, Any]], bool] | None = None,
    action_size: int = SNES_ACTION_SIZE,
    max_frames: int | None = None,
) -> ScriptResult:
    """Run a deterministic input script, optionally stopping on a ready predicate."""

    frames = 0
    observation: Any = None
    info: dict[str, Any] = {}
    if is_ready is not None and is_ready(env, info):
        return ScriptResult(0, False, True, False, False, observation, info)

    for frame in iter_input_steps(steps, action_size=action_size):
        if max_frames is not None and frames >= max_frames:
            return ScriptResult(frames, False, False, False, False, observation, info)
        observation, _reward, terminated, truncated, info = step_env(env, frame.action)
        frames += 1
        ready = is_ready is not None and is_ready(env, info)
        if ready or terminated or truncated:
            return ScriptResult(
                frames,
                False,
                ready,
                terminated,
                truncated,
                observation,
                info,
            )
    return ScriptResult(
        frames,
        True,
        bool(is_ready is not None and is_ready(env, info)),
        False,
        False,
        observation,
        info,
    )


def run_startup(
    env: Any,
    plan: StartupPlan,
    *,
    is_ready: Callable[[Any, dict[str, Any]], bool] | None = None,
    reset: bool = True,
    max_cycles: int = 1,
    max_frames: int | None = None,
    action_size: int = SNES_ACTION_SIZE,
) -> ScriptResult:
    """Reset and run a title/menu plan, repeating only when readiness is known."""

    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")
    observation: Any = None
    info: dict[str, Any] = {}
    if reset:
        observation, info = reset_env(env)
        if is_ready is not None and is_ready(env, info):
            return ScriptResult(0, False, True, False, False, observation, info)

    total_frames = 0
    for _cycle in range(max_cycles):
        remaining = None if max_frames is None else max(0, max_frames - total_frames)
        result = run_input_steps(
            env,
            plan.steps,
            is_ready=is_ready,
            action_size=action_size,
            max_frames=remaining,
        )
        total_frames += result.frames
        result = ScriptResult(
            total_frames,
            result.completed,
            result.ready,
            result.terminated,
            result.truncated,
            result.observation,
            result.info,
        )
        if result.ready or result.terminated or result.truncated:
            return result
        if is_ready is None or (max_frames is not None and total_frames >= max_frames):
            return result
    return result


def walk_right(duration: int) -> ControllerPrimitive:
    return ControllerPrimitive(
        PrimitiveKind.WALK_RIGHT, duration, ("RIGHT",), "walk_right"
    )


def walk_left(duration: int) -> ControllerPrimitive:
    return ControllerPrimitive(
        PrimitiveKind.WALK_LEFT, duration, ("LEFT",), "walk_left"
    )


def walk_up(duration: int) -> ControllerPrimitive:
    return ControllerPrimitive(PrimitiveKind.WALK_UP, duration, ("UP",), "walk_up")


def walk_down(duration: int) -> ControllerPrimitive:
    return ControllerPrimitive(
        PrimitiveKind.WALK_DOWN, duration, ("DOWN",), "walk_down"
    )


def attack(duration: int = 4, button: str = "Y") -> ControllerPrimitive:
    return ControllerPrimitive(PrimitiveKind.ATTACK, duration, (button,), "attack")


def mash_start(pulses: int = 3, hold: int = 8, gap: int = 20) -> list[FrameAction]:
    """Produce START pulses used by title screens and simple menu advances."""

    steps = [input_step("START", hold=hold, wait=gap) for _ in range(pulses)]
    return list(iter_input_steps(steps))


__all__ = [
    "ControllerPrimitive",
    "FrameAction",
    "InputStep",
    "NOOP_TOKENS",
    "PrimitiveKind",
    "ScriptResult",
    "StartupPlan",
    "attack",
    "input_step",
    "iter_input_steps",
    "mash_start",
    "parse_input_script",
    "press_button_sequence",
    "repeat_action",
    "run_input_steps",
    "run_startup",
    "walk_down",
    "walk_left",
    "walk_right",
    "walk_up",
]
