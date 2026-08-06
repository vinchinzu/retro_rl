"""Ceres Magnet Stairs + Falling Tile reverse (WRAM-reactive escape)."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_arm_pump_step,
    _ceres_clear_knockback,
    _ceres_enemy_near,
    _ceres_is_knockback,
)
from super_metroid.routes.kpdr.ceres.geometry import _CERES_MAGNET_EXIT_Y
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
)
from super_metroid.routes.runtime import ActionSpan, RouteSession


def _ceres_magnet_reached_falling(state) -> bool:
    return int(state.room_id) == ROOM_CERES_FALLING and int(state.game_state) == 8


def _ceres_magnet_step(
    session: RouteSession,
    names: tuple[str, ...],
    reason: str,
) -> bool:
    """One magnet frame. Returns True if Falling ordinary reached."""
    st = session.state
    if _ceres_magnet_reached_falling(st):
        return True
    if int(st.room_id) == ROOM_CERES_FALLING and int(st.game_state) in (9, 11):
        session.step(buttons("LEFT"), "ceres_magnet_exit_trans")
        return _ceres_magnet_reached_falling(session.state)
    if int(st.room_id) != ROOM_CERES_MAGNET:
        return False
    if _ceres_is_knockback(st):
        # Full spin-escape — single LEFT never leaves pose 137/138.
        _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
        return _ceres_magnet_reached_falling(session.state)
    # Abort RIGHT only on the east door lip (Scientist). Stair chain briefly
    # visits x~150–180 mid-climb — do not cut that short.
    if "RIGHT" in names and int(st.samus_x) > 220:
        session.step(buttons("LEFT", "A"), "ceres_magnet_abort_east")
        return False
    session.step(buttons(*names) if names else idle_action(), reason)
    return _ceres_magnet_reached_falling(session.state)


def _ceres_reactive_magnet_escape(session: RouteSession) -> None:
    """Magnet Stairs escape — WRAM-gated climb + left exit.

    Reverse arm-pump plants mid/high Magnet. Geometry: seat left (~x37),
    stair chain RIGHT+A then LEFT+A to exit height (~y139), arm-pump left into
    Falling. Every frame aborts on Falling ordinary or east-door x; not a
    blind full-escape restore.
    """
    # LEFT through door settle (idle can drop off upper ledges).
    for _ in range(220):
        st = session.state
        if st.room_id == ROOM_CERES_MAGNET and st.game_state == 8:
            break
        if _ceres_magnet_reached_falling(st):
            return
        session.step(buttons("LEFT"), "ceres_magnet_door")
    else:
        raise TimeoutError(f"ceres magnet ordinary missed: {session.state}")

    if _ceres_is_knockback(session.state):
        _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")

    # If already on exit band, skip climb and run out.
    if int(session.state.samus_y) <= _CERES_MAGNET_EXIT_Y:
        for i in range(360):
            st = session.state
            if _ceres_magnet_reached_falling(st):
                return
            if st.room_id != ROOM_CERES_MAGNET and st.room_id != ROOM_CERES_FALLING:
                break
            if _ceres_is_knockback(st):
                _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
                continue
            if _ceres_enemy_near(st, dx=40, dy=30):
                session.step(buttons("LEFT", "A"), "ceres_magnet_exit_hop")
            else:
                _ceres_arm_pump_step(
                    session, "LEFT", i, "ceres_magnet_exit", force_pump=True
                )
        if session.state.room_id != ROOM_CERES_FALLING:
            raise TimeoutError(f"ceres magnet high-exit missed Falling: {session.state}")
        return

    # Approach left (working continuous pin: ~214f from high/mid entry) until
    # seated near x≤45 or already at exit height. WRAM-gated, not a restore.
    for _ in range(280):
        st = session.state
        if _ceres_magnet_reached_falling(st):
            return
        if int(st.room_id) != ROOM_CERES_MAGNET:
            break
        if int(st.samus_y) <= _CERES_MAGNET_EXIT_Y:
            break
        if int(st.samus_x) <= 45:
            break
        if _ceres_magnet_step(session, ("LEFT",), "ceres_magnet_to_seat"):
            return

    # Up to 2 y-interruptible stair chains if first does not reach exit band.
    for _attempt in range(2):
        if int(session.state.samus_y) <= _CERES_MAGNET_EXIT_Y:
            break
        if int(session.state.room_id) != ROOM_CERES_MAGNET:
            break
        plan: list[tuple[tuple[str, ...], int]] = [
            ((), 20),
            (("A",), 16),
            (("RIGHT", "A"), 124),
            (("LEFT", "A"), 60),
        ]
        for names, frames in plan:
            for _ in range(frames):
                st = session.state
                if _ceres_magnet_reached_falling(st):
                    return
                if int(st.room_id) != ROOM_CERES_MAGNET:
                    break
                if int(st.samus_y) <= _CERES_MAGNET_EXIT_Y and "RIGHT" in names:
                    break
                if _ceres_magnet_step(session, names, "ceres_magnet_climb"):
                    return

    # Exit left until Falling (arm-pump when y is high; hop if enemy near).
    for i in range(400):
        st = session.state
        if _ceres_magnet_reached_falling(st):
            return
        if st.room_id != ROOM_CERES_MAGNET and st.room_id != ROOM_CERES_FALLING:
            break
        if st.game_state in (9, 11) and st.room_id == ROOM_CERES_FALLING:
            session.step(buttons("LEFT"), "ceres_magnet_exit_trans")
            continue
        if _ceres_is_knockback(st):
            _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
            continue
        # Mid platform: need height still — hop up rather than wall-walk.
        if int(st.samus_y) > _CERES_MAGNET_EXIT_Y:
            session.step(buttons("LEFT", "A"), "ceres_magnet_up_hop")
            continue
        if _ceres_enemy_near(st, dx=40, dy=30):
            session.step(buttons("LEFT", "A"), "ceres_magnet_exit_hop")
        else:
            _ceres_arm_pump_step(
                session, "LEFT", i, "ceres_magnet_exit", force_pump=True
            )

    if session.state.room_id != ROOM_CERES_FALLING:
        raise TimeoutError(f"ceres magnet exit missed Falling: {session.state}")


def _ceres_reactive_falling(session: RouteSession) -> None:
    """Falling Tile reverse → elev door. WRAM: room, x progress, KB, enemy0.

    Falling→elev remaps to **high** elev (y≈139). Prefer steady LEFT (product
    walk) — arm-pump + long i-frame thrash desyncs elev entry momentum.
    """
    if session.state.room_id != ROOM_CERES_FALLING:
        raise RuntimeError(f"expected Falling after magnet: {session.state}")
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_FALLING and s.game_state == 8,
        timeout=120,
        reason="ceres_falling_door",
    )
    # Product-shaped: short hop then LEFT walk into elev door.
    session.span(ActionSpan(("LEFT", "A"), 40, "ceres_falling_entry"))
    last_x = int(session.state.samus_x)
    stagnant = 0
    for i in range(500):
        st = session.state
        if st.room_id == ROOM_CERES_ELEVATOR:
            return
        if st.game_state in (9, 11):
            session.step(buttons("LEFT"), "ceres_falling_door_trans")
            continue
        x = int(st.samus_x)
        if x < last_x - 1:
            stagnant = 0
            last_x = x
        else:
            stagnant += 1
        if _ceres_is_knockback(st):
            # One spin-escape, then resume walk — no multi-second i-frame loop.
            _ceres_clear_knockback(session, "LEFT", reason="ceres_falling")
            last_x = int(session.state.samus_x)
            stagnant = 0
            continue
        if _ceres_enemy_near(st, dx=40, dy=32):
            session.step(buttons("LEFT", "A"), "ceres_falling_enemy_hop")
            continue
        if stagnant > 14:
            for _ in range(6):
                session.step(buttons("LEFT", "A"), "ceres_falling_leap")
            stagnant = 0
            last_x = int(session.state.samus_x)
            continue
        # Steady LEFT (matches product falling leave pose/speed better than pump).
        session.step(buttons("LEFT"), "ceres_falling_walk")
    raise TimeoutError(f"falling missed elev: {session.state}")


__all__ = [
    "_ceres_magnet_reached_falling",
    "_ceres_magnet_step",
    "_ceres_reactive_magnet_escape",
    "_ceres_reactive_falling",
]
