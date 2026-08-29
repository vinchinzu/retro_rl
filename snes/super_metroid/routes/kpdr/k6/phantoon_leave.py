"""Phantoon loot pickup + left-door exit to WS Basement.

After the kill (HP 0 + ``$D82B`` bit 0), grab remaining flame drops then
jump-left into the door. Floor-hug LEFT at x≤40 is wall knockback (p138);
the door slot is the enter height (~y 124). Measured from
``scratch/post_phantoon_poweron.state`` (37,187) p1: jump LEFT+A →
``0xCC6F`` settle ~(1240,139) p10 gs=8.

https://wiki.supermetroid.run/Phantoon%27s_Room
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    is_morph,
    require_room,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k6.phantoon_fight import phantoon_boss_bit_set
from super_metroid.routes.kpdr.room_ids import ROOM_PHANTOON, ROOM_WS_BASEMENT
from super_metroid.routes.runtime import ControllerSession, RouteSession, Split, hold

# Left-corner floor vs door slot. Jump LEFT+A once x is in this band.
DOOR_X_MAX = 70
SWEEP_X = 160
# Wall knockback from hugging LEFT on the floor (live leave probe).
WALL_HURT = frozenset({83, 84, 109, 137, 138, 143, 158, 159, 160})
_SETTLE = 240

__all__ = [
    "DOOR_X_MAX",
    "SWEEP_X",
    "loot_walk_action",
    "door_jump_action",
    "play_phantoon_loot_exit",
    "require_phantoon_left",
]


def loot_walk_action(
    samus_x: int, target_x: int | None, *, swept: bool = False
) -> tuple[str, ...]:
    """Walk to a pickup x, else one right-sweep, else empty (go to the door)."""
    if target_x is not None:
        if int(target_x) < int(samus_x) - 6:
            return ("LEFT", "B")
        if int(target_x) > int(samus_x) + 6:
            return ("RIGHT", "B")
        return ()
    if not swept and int(samus_x) < SWEEP_X:
        return ("RIGHT", "B")
    return ()


def door_jump_action(samus_x: int, pose: int, frame: int) -> tuple[str, ...]:
    """Jump-left into the door. Do not floor-hug LEFT (p138)."""
    if int(pose) in WALL_HURT:
        return ()
    if int(samus_x) > DOOR_X_MAX:
        return ("LEFT", "B")
    phase = int(frame) % 36
    if phase < 20:
        return ("LEFT", "A")
    if phase < 24:
        return ("LEFT", "X")
    return ()


def require_phantoon_left(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    """SpineHop ``after``: basement settle with Phantoon bit still set."""
    del splits, result
    st = session.state
    if int(st.room_id) != ROOM_WS_BASEMENT:
        raise RuntimeError(
            "phantoon_loot_exit: expected WS Basement 0x"
            f"{ROOM_WS_BASEMENT:04X}, got {st}"
        )
    if not phantoon_boss_bit_set(session):
        raise RuntimeError(
            "phantoon_loot_exit: Wrecked Ship $D82B bit 0 not set: "
            f"{st}"
        )


def play_phantoon_loot_exit(session: ControllerSession) -> SuperMetroidState:
    """Grab remaining drops in ``0xCD13``, jump-left into WS Basement."""
    label = "phantoon_loot_exit"
    if (
        int(session.state.room_id) == ROOM_WS_BASEMENT
        and int(session.state.game_state) == 8
        and int(session.state.door_transition) == 0
    ):
        return session.state
    require_room(session, ROOM_PHANTOON, label)
    if not phantoon_boss_bit_set(session):
        raise RuntimeError(f"{label}: Phantoon not defeated: {session.state}")
    if is_morph(int(session.state.pose)):
        try:
            unmorph(session)
        except Exception:
            pass
    _collect_loot(session, label)
    _jump_left_door(session, label)
    return wait_ordinary_room(
        session, ROOM_WS_BASEMENT, settle_frames=_SETTLE, label=label
    )


def _collect_loot(session: ControllerSession, label: str) -> None:
    from super_metroid.combat.primitives import list_pickups

    swept = False
    for _ in range(360):
        st = session.state
        if int(st.room_id) != ROOM_PHANTOON:
            return
        if int(st.pose) in WALL_HURT:
            hold(session, 1, reason=f"{label}_loot_hurt")
            continue
        if is_morph(int(st.pose)):
            try:
                unmorph(session)
            except Exception:
                hold(session, 1, reason=f"{label}_loot_unmorph")
            continue
        drops = list_pickups(getattr(session, "env", None))
        target = None
        if drops:
            target = min(drops, key=lambda p: abs(int(p.x) - int(st.samus_x))).x
        elif int(st.samus_x) >= SWEEP_X:
            swept = True
        names = loot_walk_action(int(st.samus_x), target, swept=swept)
        if not names:
            return
        hold(session, 1, *names, reason=f"{label}_loot")


def _jump_left_door(session: ControllerSession, label: str) -> None:
    for frame in range(480):
        st = session.state
        if int(st.room_id) == ROOM_WS_BASEMENT:
            return
        if int(st.room_id) != ROOM_PHANTOON:
            return
        names = door_jump_action(int(st.samus_x), int(st.pose), frame)
        if names:
            hold(session, 1, *names, reason=f"{label}_door")
        else:
            hold(session, 1, reason=f"{label}_hurt")
    if int(session.state.room_id) != ROOM_WS_BASEMENT:
        raise TimeoutError(f"{label}: left door missed: {session.state}")
