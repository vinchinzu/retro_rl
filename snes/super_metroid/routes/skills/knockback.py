"""Room-agnostic knockback pose checks and open-loop escape helpers.

K4 Cathedral / Rising Tide (and similar contact corridors) share pose-137/138
handling: either idle-hold through the stun or spin-escape with a short run
prefix. Frame budgets stay caller-owned — this module does not retune hops.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, select_weapon
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession


def is_knockback(
    state: SuperMetroidState, *, poses: frozenset[int] = POSE_KNOCKBACK
) -> bool:
    """True when Samus is in knockback poses (default 137 / 138)."""
    return int(state.pose) in poses


def hold_through_knockback(
    session: ControllerSession,
    frames: int,
    *,
    label: str,
    reason: str = "kb",
) -> SuperMetroidState:
    """Cathedral-style idle clear: hold neutral through knockback frames.

    Use when contact is brief and planting is safe (floor cross / mid climb).
    Never use this under continuous enemy contact with unlimited-energy assist
    — prefer :func:`escape_knockback_spin`.
    """
    return hold(session, frames, reason=f"{label}_{reason}")


def escape_knockback_spin(
    session: ControllerSession,
    *,
    prefer_dir: str = "RIGHT",
    run_frames: int = 6,
    spin_frames: int = 20,
    label: str,
    run_with: Sequence[str] = ("B",),
    spin_with: Sequence[str] = ("B", "A"),
    run_reason: str = "kb_run",
    spin_reason: str = "kb_spin",
    stop_room_id: int | None = None,
    break_on_motion_clear: bool = False,
    ensure_beam: bool = False,
    motion_clear_px: int = 2,
) -> SuperMetroidState:
    """Spin-escape out of knockback poses 137/138.

    Open-loop: ``run_frames`` of ``prefer_dir`` + ``run_with``, then
    ``spin_frames`` of ``prefer_dir`` + ``spin_with``. Defaults match Rising
    Tide (RIGHT+B then RIGHT+B+A). Cathedral uses ``spin_frames=24``,
    ``run_with=("B", "X")``, ``ensure_beam=True``, and motion-clear break.

    Returns the last stepped state (or current if both budgets are zero).
    """
    start_x = int(session.state.samus_x)
    if ensure_beam and int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    run_btns = (prefer_dir, *run_with)
    spin_btns = (prefer_dir, *spin_with)
    st = session.state

    for _ in range(run_frames):
        st = hold(session, 1, *run_btns, reason=f"{label}_{run_reason}")
        if stop_room_id is not None and int(st.room_id) == stop_room_id:
            return st

    for _ in range(spin_frames):
        st = hold(session, 1, *spin_btns, reason=f"{label}_{spin_reason}")
        if stop_room_id is not None and int(st.room_id) == stop_room_id:
            return st
        if break_on_motion_clear and (
            not is_knockback(st)
            and abs(int(st.samus_x) - start_x) > motion_clear_px
        ):
            break

    return session.state


def escape_kb(
    session: ControllerSession,
    label: str,
    prefer: str,
    *,
    stop_room_id: int | None = None,
    run_frames: int = 6,
    spin_frames: int = 18,
) -> SuperMetroidState:
    """Corridor knockback escape with shared Wave-shaped defaults.

    Thin wrapper around :func:`escape_knockback_spin` used by multi-hop room
    policies (Wave branch, etc.). Only ``stop_room_id`` / direction typically
    vary per hop — do not reintroduce private ``_escape_kb_*`` triples.

    Defaults (``run_frames=6``, ``spin_frames=18``) match the Wave Bubble →
    Double → Wave corridor escapes. Callers with different knobs (Cathedral
    beam thrash, Moat, speed halls) should keep calling
    :func:`escape_knockback_spin` directly.
    """
    return escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=run_frames,
        spin_frames=spin_frames,
        label=label,
        stop_room_id=stop_room_id,
    )


__all__ = [
    "is_knockback",
    "hold_through_knockback",
    "escape_knockback_spin",
    "escape_kb",
]
