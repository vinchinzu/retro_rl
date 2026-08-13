"""Classic L↔R arm-pump helpers and knockback recovery for Ceres.

YT reference (Kentroid TFsGVxQReMw chunk ``k0_ceres``) lives at
``policies/early_game/ceres_kentroid_spans.json`` + gitignored
``refs/yt_reference/.../chunks/k0_ceres/``. Absolute Input Display replay
desyncs on elevator lag / magnet geometry — do **not** restore fixed product
open-loop when a later leg breaks. Speed every section; re-solve tails with
room / pose / y / knockback reads (same idea as K4 knockback skills).

Classic arm-pump: dir+B with L↔R angle spam (``runway_dash`` period-2).
"""

from __future__ import annotations

from retro_harness.actions import buttons
from super_metroid.ram import GS_ORDINARY
from super_metroid.routes.kpdr.ceres.geometry import _CERES_ARM_PUMP_PERIOD
from super_metroid.routes.runtime import ActionSpan, RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import shoulder_pump_button


def _arm_pump_dash_spans(
    direction: str,
    frames: int,
    reason: str,
    *,
    period: int = _CERES_ARM_PUMP_PERIOD,
) -> list[ActionSpan]:
    """Expand ``dir+B`` into classic L↔R arm-pump (``runway_dash`` pattern)."""
    period = max(1, period)
    out: list[ActionSpan] = []
    i = 0
    while i < frames:
        ang = shoulder_pump_button(i, period)
        chunk = min(period, frames - i)
        out.append(ActionSpan((direction, "B", ang), chunk, reason))
        i += chunk
    return out


def _ceres_enemy_near(state, *, dx: int = 48, dy: int = 40) -> bool:
    """True when enemy0 is alive and within a contact box of Samus."""
    if int(state.enemy0_hp) <= 0:
        return False
    ex, ey = int(state.enemy0_x), int(state.enemy0_y)
    # Dead/despawned slots often park at 0 or far off-screen.
    if ex == 0 and ey == 0:
        return False
    sx, sy = int(state.samus_x), int(state.samus_y)
    return abs(ex - sx) <= dx and abs(ey - sy) <= dy


def _ceres_arm_pump_step(
    session: RouteSession,
    direction: str,
    i: int,
    reason: str,
    *,
    period: int = _CERES_ARM_PUMP_PERIOD,
    jump: bool = False,
    force_pump: bool = False,
) -> None:
    """One frame of classic L↔R arm-pump (optional A).

    Only angle-spam when already running (speed_flag or momentum). Bare dir+B
    while accelerating — shoulder L/R alone can freeze Samus in aim poses
    (e.g. 207). ``direction`` is D-pad LEFT/RIGHT.
    """
    period = max(1, period)
    st = session.state
    running = force_pump or int(st.speed_flag) != 0 or abs(int(st.momentum_x)) >= 1
    if running and not jump:
        names: tuple[str, ...] = (direction, "B", shoulder_pump_button(i, period))
    elif jump and running:
        names = (direction, "B", shoulder_pump_button(i, period), "A")
    elif jump:
        names = (direction, "B", "A")
    else:
        names = (direction, "B")
    session.step(buttons(*names), reason)


def _ceres_clear_knockback(
    session: RouteSession,
    direction: str,
    *,
    reason: str,
    max_frames: int = 40,
) -> None:
    """Spin-escape knockback using WRAM pose (no fixed open-loop restore)."""
    for i in range(max_frames):
        if not is_knockback(session.state):
            return
        # Short run then spin in travel direction.
        if i < 6:
            session.step(buttons(direction, "B"), f"{reason}_kb_run")
        else:
            session.step(buttons(direction, "B", "A"), f"{reason}_kb_spin")


def _ceres_arm_pump_until(
    session: RouteSession,
    direction: str,
    *,
    reason: str,
    max_frames: int,
    done,
    period: int = _CERES_ARM_PUMP_PERIOD,
    recover_knockback: bool = True,
    jump_every: int | None = None,
    stuck_jump_after: int | None = 45,
) -> int:
    """Classic L↔R arm-pump until ``done(state)``; WRAM-reactive on KB / stuck.

    - Knockback (pose 137/138): spin-escape then resume.
    - Stuck (x not advancing): brief A pulse (magnet stairs / ledges).
    - ``jump_every``: optional periodic hop (elevator / magnet bands).

    Returns frames consumed. Raises TimeoutError on miss.
    """
    period = max(1, period)
    if done(session.state):
        return 0
    last_x = int(session.state.samus_x)
    stagnant = 0
    i = 0
    while i < max_frames:
        st = session.state
        if done(st):
            return i
        if recover_knockback and is_knockback(st):
            _ceres_clear_knockback(session, direction, reason=reason)
            last_x = int(session.state.samus_x)
            stagnant = 0
            i += 1
            continue
        jump = False
        if jump_every is not None and jump_every > 0 and (i % jump_every) == 0:
            jump = True
        x = int(st.samus_x)
        # Progress in travel direction?
        progressed = (x < last_x - 1) if direction == "LEFT" else (x > last_x + 1)
        if progressed:
            stagnant = 0
            last_x = x
        else:
            stagnant += 1
            if stuck_jump_after is not None and stagnant >= stuck_jump_after:
                jump = True
                stagnant = 0
                last_x = x
        _ceres_arm_pump_step(
            session, direction, i, reason, period=period, jump=jump
        )
        i += 1
    if done(session.state):
        return i
    raise TimeoutError(
        f"{reason} arm-pump timed out after {max_frames}f: {session.state}"
    )


def _ceres_wait_ordinary(
    session: RouteSession, room_id: int, *, reason: str, timeout: int = 200
) -> None:
    session.wait_until(
        lambda s: s.room_id == room_id and s.game_state == GS_ORDINARY,
        timeout=timeout,
        reason=reason,
    )


__all__ = [
    "_arm_pump_dash_spans",
    "_ceres_enemy_near",
    "_ceres_arm_pump_step",
    "_ceres_clear_knockback",
    "_ceres_arm_pump_until",
    "_ceres_wait_ordinary",
]
