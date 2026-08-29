"""Shared combat primitives for boss strategies.

Compose strategies from these helpers instead of new ad-hoc frame loops.
Extracted from patterns in Kraid, Bomb Torizo, and Spore Spawn.

See ``docs/BOSS_PIPELINE.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import select_weapon, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

T = TypeVar("T")


# ---------------------------------------------------------------------------
# One-frame action builders (pure; no session)
# ---------------------------------------------------------------------------


def lane_hold_action(
    samus_x: int,
    *,
    min_x: int,
    max_x: int,
    face: str = "RIGHT",
    dash: bool = True,
) -> tuple[str, ...]:
    """Stay inside ``[min_x, max_x]``; face ``face`` when in lane.

    Outside the window, walk back in (with optional dash). Off-map / wrap
    values (> 60_000) yield an empty action.
    """
    if samus_x > 60_000:
        return ()
    if samus_x > max_x:
        names = ["LEFT"]
        if dash:
            names.append("B")
        return tuple(names)
    if samus_x < min_x:
        names = ["RIGHT"]
        if dash:
            names.append("B")
        return tuple(names)
    return (face,) if face else ()


def spray_action(
    frame_index: int,
    *,
    face: str = "RIGHT",
    fire_period: int = 12,
    fire_hold_frames: int = 6,
    jump_period: int = 50,
    jump_hold_frames: int = 10,
    dash_when_not_jumping: bool = True,
    fire_button: str = "X",
) -> tuple[str, ...]:
    """Periodic fire + jump spray facing a direction (Kraid-style)."""
    names: list[str] = []
    if face:
        names.append(face)
    if jump_period > 0 and frame_index % jump_period < jump_hold_frames:
        names.append("A")
    if fire_period > 0 and frame_index % fire_period < fire_hold_frames:
        names.append(fire_button)
    if dash_when_not_jumping and "A" not in names:
        names.append("B")
    # Preserve order, drop duplicates.
    return tuple(dict.fromkeys(names))


def face_toward_action(
    samus_x: int,
    target_x: int,
    *,
    fire: bool = False,
    jump: bool = False,
    dash: bool = False,
    fire_button: str = "X",
) -> tuple[str, ...]:
    """Face the target X; optionally fire / jump / dash this frame."""
    names: list[str] = []
    if target_x >= samus_x:
        names.append("RIGHT")
    else:
        names.append("LEFT")
    if jump:
        names.append("A")
    if fire:
        names.append(fire_button)
    if dash:
        names.append("B")
    return tuple(names)


def range_kite_action(
    samus_x: int,
    enemy_x: int,
    *,
    min_range: int = 70,
    max_range: int = 120,
    jump_range: int = 100,
    frame_index: int = 0,
    jump_period: int = 50,
    jump_hold_frames: int = 18,
    fire_period: int = 2,
    fire_button: str = "X",
) -> tuple[str, ...]:
    """Torizo-style range kite: hold distance band, face, periodic fire/jump."""
    dx = enemy_x - samus_x
    dist = abs(dx)
    face = "RIGHT" if dx >= 0 else "LEFT"
    away = "LEFT" if dx >= 0 else "RIGHT"
    toward = face

    names: list[str] = []
    if dist < min_range:
        names.append(away)
    elif dist > max_range:
        names.append(toward)
    else:
        names.append(face)

    if dist <= jump_range and jump_period > 0:
        if frame_index % jump_period < jump_hold_frames:
            names.append("A")
    if fire_period > 0 and frame_index % fire_period == 0:
        names.append(fire_button)
    return tuple(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def ensure_weapon(session: ControllerSession, weapon: int) -> None:
    """Select weapon index if capacity allows (no-op when already selected)."""
    if session.state.selected_item == weapon:
        return
    # Capacity gates for ammo weapons (0=beam always ok).
    if weapon == 1 and session.state.max_missiles <= 0:
        return
    if weapon == 2 and session.state.max_super_missiles <= 0:
        return
    if weapon == 3 and session.state.max_power_bombs <= 0:
        return
    select_weapon(session, weapon)


def settle_standing(
    session: ControllerSession,
    *,
    min_y: int | None = None,
    bad_poses: frozenset[int] = frozenset({81, 164}),
    max_frames: int = 60,
    reason: str = "combat_settle",
) -> SuperMetroidState:
    """Idle until Samus is not in mid-air entry poses (doorway land)."""
    for _ in range(max_frames):
        st = session.state
        y_ok = min_y is None or st.samus_y >= min_y
        if y_ok and st.pose not in bad_poses:
            return st
        hold(session, 1, reason=reason)
    return session.state


def lane_hold_window(
    session: ControllerSession,
    *,
    min_x: int,
    max_x: int,
    hold_frames: int,
    face: str = "RIGHT",
    dash: bool = True,
    recovery_frames: int = 0,
    settle_min_y: int | None = None,
    settle_bad_poses: frozenset[int] = frozenset({81, 164}),
    reason: str = "lane_hold_window",
) -> SuperMetroidState:
    """Hold a horizontal position lane for a duration, then optional recovery settle.

    Session-level **window** companion to the one-frame :func:`lane_hold_action`.
    Each hold frame recomputes buttons from the current ``session.state.samus_x``
    so Samus walks/dashes back into ``[min_x, max_x]`` and faces ``face`` while
    inside the band. When ``recovery_frames > 0``, finishes with
    :func:`settle_standing` (idle until standing / not mid-air entry poses).

    Call signature for future ``BossStrategy`` composition::

        lane_hold_window(
            session,
            min_x=50,
            max_x=260,
            hold_frames=90,
            face="RIGHT",
            dash=True,
            recovery_frames=30,
            settle_min_y=390,          # optional floor gate (Kraid-style)
            settle_bad_poses=frozenset({81, 164}),
            reason="boss_lane",
        )

    Parameters
    ----------
    session:
        Active controller session (already in the boss room / arena).
    min_x, max_x:
        Inclusive position band (same semantics as :func:`lane_hold_action`).
    hold_frames:
        Number of frames to re-apply lane hold (``N`` in the card recipe).
    face:
        Direction held while inside the band (default ``\"RIGHT\"``).
    dash:
        Include ``B`` when walking back into the band.
    recovery_frames:
        Max frames for post-window :func:`settle_standing` (``M``). ``0`` skips
        recovery entirely.
    settle_min_y, settle_bad_poses:
        Forwarded to :func:`settle_standing` when recovering.
    reason:
        Hold reason prefix; recovery uses ``f\"{reason}_recover\"``.

    Returns
    -------
    SuperMetroidState
        Session state after the hold window (and optional recovery).
    """
    for _ in range(max(0, hold_frames)):
        buttons = lane_hold_action(
            session.state.samus_x,
            min_x=min_x,
            max_x=max_x,
            face=face,
            dash=dash,
        )
        if buttons:
            hold(session, 1, *buttons, reason=reason)
        else:
            hold(session, 1, reason=reason)

    if recovery_frames > 0:
        return settle_standing(
            session,
            min_y=settle_min_y,
            bad_poses=settle_bad_poses,
            max_frames=recovery_frames,
            reason=f"{reason}_recover",
        )
    return session.state


def wait_predicate(
    session: ControllerSession,
    pred: Callable[[SuperMetroidState], bool],
    *,
    timeout: int,
    reason: str = "combat_wait",
    hold_buttons: tuple[str, ...] = (),
) -> SuperMetroidState:
    """Hold (optionally with buttons) until ``pred`` or raise TimeoutError."""
    for _ in range(timeout):
        if pred(session.state):
            return session.state
        if hold_buttons:
            hold(session, 1, *hold_buttons, reason=reason)
        else:
            hold(session, 1, reason=reason)
    raise TimeoutError(f"{reason} timed out after {timeout} frames")


def wait_enemy_hp_zero(
    session: ControllerSession,
    *,
    timeout: int = 15_000,
    reason: str = "wait_enemy_hp_zero",
    hold_buttons: tuple[str, ...] = (),
) -> SuperMetroidState:
    """Wait until enemy0 HP is 0."""
    return wait_predicate(
        session,
        lambda s: s.enemy0_hp == 0,
        timeout=timeout,
        reason=reason,
        hold_buttons=hold_buttons,
    )


def wait_room(
    session: ControllerSession,
    room_id: int,
    *,
    timeout: int = 600,
    reason: str = "wait_room",
    hold_buttons: tuple[str, ...] = ("RIGHT",),
) -> SuperMetroidState:
    """Walk/hold until ``session`` is in ``room_id``."""
    return wait_predicate(
        session,
        lambda s: s.room_id == room_id,
        timeout=timeout,
        reason=reason,
        hold_buttons=hold_buttons,
    )


def push_horizontal_door(
    session: ControllerSession,
    *,
    direction: str,
    target_room: int,
    timeout: int = 900,
    dash: bool = True,
    reason: str = "push_door",
) -> SuperMetroidState:
    """Hold left/right (optional dash) until room changes to ``target_room``."""
    buttons = [direction]
    if dash:
        buttons.append("B")
    return wait_room(
        session,
        target_room,
        timeout=timeout,
        reason=reason,
        hold_buttons=tuple(buttons),
    )


def hold_for(
    session: ControllerSession,
    frames: int,
    *buttons: str,
    reason: str = "combat_hold",
) -> SuperMetroidState:
    """Thin wrapper around ``hold`` for composition readability."""
    return hold(session, frames, *buttons, reason=reason)


# ---------------------------------------------------------------------------
# Phase machine
# ---------------------------------------------------------------------------


class PhaseResult(Enum):
    """Outcome of one phase tick / run."""

    CONTINUE = auto()
    ADVANCE = auto()
    DONE = auto()
    FAIL = auto()


@dataclass
class PhaseMachine:
    """Simple ordered phase runner for multi-phase bosses.

    Usage::

        machine = PhaseMachine(["activate", "fight", "exit"])
        while not machine.done:
            # ... play current phase ...
            machine.advance()  # or machine.fail("reason")
    """

    phases: list[str]
    index: int = 0
    failed: str | None = None

    @property
    def current(self) -> str | None:
        if self.failed is not None:
            return None
        if self.index >= len(self.phases):
            return None
        return self.phases[self.index]

    @property
    def done(self) -> bool:
        return self.failed is None and self.index >= len(self.phases)

    @property
    def ok(self) -> bool:
        return self.failed is None and self.done

    def advance(self) -> None:
        if self.failed is not None:
            return
        self.index += 1

    def fail(self, reason: str) -> None:
        self.failed = reason

    def to_dict(self) -> dict[str, object]:
        return {
            "phases": list(self.phases),
            "index": self.index,
            "current": self.current,
            "done": self.done,
            "failed": self.failed,
        }


# ---------------------------------------------------------------------------
# Enemy-projectile pickups (bank $86; not boss-specific)
# ---------------------------------------------------------------------------

# 18 slots × 2 bytes. $1997 is the header pointer (not a 1-byte type).
# Pickups are projectile $F337; kind is the instruction list at $1B47.
N_ENEMY_PROJECTILES = 18
ADDR_PROJ_ID = 0x1997
ADDR_PROJ_X = 0x1A4B
ADDR_PROJ_Y = 0x1A93
ADDR_PROJ_ILIST = 0x1B47
PICKUP_PROJ_ID = 0xF337
ILIST_SMALL_ENERGY = 0xED8D
ILIST_BIG_ENERGY = 0xEDA3
ILIST_MISSILES = 0xEDB9
PICKUP_SMALL_ENERGY = 0x16
PICKUP_BIG_ENERGY = 0x17
PICKUP_MISSILE = 0x18
_ILIST_TO_KIND = {
    ILIST_SMALL_ENERGY: PICKUP_SMALL_ENERGY,
    ILIST_BIG_ENERGY: PICKUP_BIG_ENERGY,
    ILIST_MISSILES: PICKUP_MISSILE,
}


@dataclass(frozen=True)
class Pickup:
    slot: int
    kind: int
    x: int
    y: int


def _read_u16(ram: Any, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def list_pickups(env: Any) -> tuple[Pickup, ...]:
    """Live enemy-projectile pickups (energy / missiles). Empty if no RAM."""
    if env is None:
        return ()
    try:
        ram = env.get_ram()
    except Exception:
        return ()
    need = ADDR_PROJ_ILIST + N_ENEMY_PROJECTILES * 2
    if ram is None or len(ram) < need:
        return ()
    found: list[Pickup] = []
    for slot in range(N_ENEMY_PROJECTILES):
        header = _read_u16(ram, ADDR_PROJ_ID + slot * 2)
        if header == PICKUP_PROJ_ID:
            kind = _ILIST_TO_KIND.get(_read_u16(ram, ADDR_PROJ_ILIST + slot * 2), 0)
        else:
            kind = header & 0xFF
        if kind not in (PICKUP_SMALL_ENERGY, PICKUP_BIG_ENERGY, PICKUP_MISSILE):
            continue
        x = _read_u16(ram, ADDR_PROJ_X + slot * 2)
        y = _read_u16(ram, ADDR_PROJ_Y + slot * 2)
        if x == 0 and y == 0:
            continue
        found.append(Pickup(slot=slot, kind=kind, x=x, y=y))
    return tuple(found)


# ---------------------------------------------------------------------------
# Re-exports for strategy modules
# ---------------------------------------------------------------------------

__all__ = [
    "ADDR_PROJ_ID",
    "ADDR_PROJ_ILIST",
    "ADDR_PROJ_X",
    "ADDR_PROJ_Y",
    "ILIST_BIG_ENERGY",
    "ILIST_MISSILES",
    "ILIST_SMALL_ENERGY",
    "N_ENEMY_PROJECTILES",
    "PICKUP_BIG_ENERGY",
    "PICKUP_MISSILE",
    "PICKUP_PROJ_ID",
    "PICKUP_SMALL_ENERGY",
    "PhaseMachine",
    "PhaseResult",
    "Pickup",
    "ensure_weapon",
    "face_toward_action",
    "hold_for",
    "lane_hold_action",
    "lane_hold_window",
    "list_pickups",
    "push_horizontal_door",
    "range_kite_action",
    "select_weapon",
    "settle_standing",
    "spray_action",
    "unmorph",
    "wait_enemy_hp_zero",
    "wait_predicate",
    "wait_room",
]
