"""Shared SNES hold / Z3 control-loop primitives for SMZ3 segments.

Route modules own geometry and phase policy; this module owns the boring
frame-stepping and “become controllable” seams so death/text/hold-up handling
does not fork per leg.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from retro_harness.snes import idle_action, snes_action
from smz3.ram import ComboSnapshot, snapshot_env

Z3_MODULE_DEATH = 0x12
Z3_MODULE_TEXT = 0x0E
Z3_CONTROLLABLE_MODULES = frozenset({0x07, 0x09})


def hold(
    env: Any,
    buttons: tuple[str, ...] | list[str] | None,
    n: int,
    *,
    frame: int,
) -> int:
    """Step *n* frames with optional buttons; return updated frame counter."""
    action = (
        snes_action(*buttons, dtype=np.int8) if buttons else idle_action(dtype=np.int8)
    )
    for _ in range(max(0, n)):
        env.step(action)
        frame += 1
    return frame


def is_z3_dead(snap: ComboSnapshot) -> bool:
    return snap.z3_module == Z3_MODULE_DEATH


def wait_z3_control(
    env: Any,
    *,
    start_frame: int = 0,
    max_frames: int = 400,
    clear_hold_up: bool = False,
    hold_up_check: Any | None = None,
) -> tuple[int, ComboSnapshot]:
    """Idle / mash text until module $07/$09 sub 0 (optionally clear hold-up).

    Parameters
    ----------
    clear_hold_up:
        When True, also require inventory hold-up clear. Uses *hold_up_check*
        if provided (callable ``env -> bool`` True when holding item), else
        only module/submodule.
    hold_up_check:
        Optional ``callable(env) -> bool``; True means still holding item up.
    """
    frame = start_frame
    for i in range(max(0, max_frames)):
        snap = snapshot_env(env, frame=frame)
        holding = bool(hold_up_check(env)) if hold_up_check is not None else False
        if is_z3_dead(snap):
            return frame, snap
        if (
            snap.z3_controllable
            and snap.z3_module in Z3_CONTROLLABLE_MODULES
            and (not clear_hold_up or not holding)
        ):
            return frame, snap
        if holding:
            frame = hold(env, ("LEFT",), 1, frame=frame)
        elif snap.z3_module == Z3_MODULE_TEXT:
            btn = ("B",) if (i // 4) % 2 == 0 else ("A",)
            frame = hold(env, btn, 2, frame=frame)
        else:
            frame = hold(env, None, 1, frame=frame)
    return frame, snapshot_env(env, frame=frame)


def go_xy(
    env: Any,
    frame: int,
    tx: int,
    ty: int,
    *,
    tol: int = 2,
    max_steps: int = 250,
    step_frames: int = 2,
    clear_hold_up: bool = False,
    hold_up_check: Any | None = None,
) -> int:
    """Walk toward absolute (tx, ty) with dual-axis D-pad; return frame."""
    for _ in range(max_steps):
        snap = snapshot_env(env, frame=frame)
        if not snap.z3_controllable:
            frame, snap = wait_z3_control(
                env,
                start_frame=frame,
                max_frames=80,
                clear_hold_up=clear_hold_up,
                hold_up_check=hold_up_check,
            )
        dx = tx - snap.z3_link_x
        dy = ty - snap.z3_link_y
        if abs(dx) <= tol and abs(dy) <= tol:
            return frame
        buttons: list[str] = []
        if abs(dx) > tol:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > tol:
            buttons.append("DOWN" if dy > 0 else "UP")
        frame = hold(
            env,
            tuple(buttons[:2]) if buttons else None,
            step_frames,
            frame=frame,
        )
    return frame
