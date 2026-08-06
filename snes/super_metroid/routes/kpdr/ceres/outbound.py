"""Ceres outbound (elev→Ridley) and escape (Ridley→Landing) play callables."""

from __future__ import annotations

from retro_harness.actions import idle_action
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _arm_pump_dash_spans,
    _ceres_arm_pump_until,
    _ceres_wait_ordinary,
)
from super_metroid.routes.kpdr.ceres.elev_escape import _ceres_reactive_elev_climb
from super_metroid.routes.kpdr.ceres.magnet import (
    _ceres_reactive_falling,
    _ceres_reactive_magnet_escape,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_LANDING_SITE,
)
from super_metroid.routes.runtime import ActionSpan, RouteSession


def _ceres_outbound_to_scientist_spans() -> list[ActionSpan]:
    """Elevator → Scientist open-loop with classic L↔R on every RIGHT+B dash.

    Elevator shaft + Falling Tile + Magnet Stairs are geometry-heavy (product
    trace: KB fall, upper-path Falling, magnet y 139→395). Keep that pin but
    inject classic arm-pump on run holds. Scientist → Ridley is room-gated
    arm-pump in :func:`play_ceres_outbound_to_ridley` (the real length cut).
    """
    out: list[ActionSpan] = []
    # (buttons, frames, arm_pump on dir+B)
    raw: list[tuple[tuple[str, ...], int, bool]] = [
        (("RIGHT", "A"), 24, False),
        (("RIGHT",), 120, False),
        (("LEFT",), 120, False),
        (("RIGHT", "B"), 240, True),
        ((), 60, False),
        (("RIGHT",), 24, False),
        (("RIGHT", "B"), 24, True),
        (("RIGHT", "B", "A"), 24, False),
        (("RIGHT", "A"), 24, False),
        (("RIGHT",), 24, False),
        (("RIGHT",), 24, False),
        (("RIGHT",), 24, False),
        (("RIGHT",), 24, False),
        (("RIGHT", "B"), 24, True),
        ((), 12, False),
        (("RIGHT",), 24, False),
        ((), 140, False),
        (("RIGHT",), 160, False),
        (("LEFT",), 120, False),
        (("RIGHT", "B"), 96, True),
        # Product had idle 120 here; cut — room-gate finishes Scientist→Ridley.
        ((), 24, False),
    ]
    for names, frames, pump in raw:
        if pump:
            direction = "LEFT" if "LEFT" in names else "RIGHT"
            out.extend(
                _arm_pump_dash_spans(direction, frames, "ceres_outbound")
            )
        else:
            out.append(ActionSpan(names, frames, "ceres_outbound"))
    return out


def _ceres_escape_spans() -> list[ActionSpan]:
    """Legacy full open-loop escape (tests only — product uses reactive play_)."""
    raw = (
        (("LEFT", "A"), 40, "ceres_ridley_exit"),
        (("LEFT",), 1000, "ceres_reverse_rooms"),
        ((), 20, "ceres_magnet_phase_align"),
        (("A",), 16, "ceres_magnet_climb"),
        (("RIGHT", "A"), 124, "ceres_magnet_climb"),
        (("LEFT", "A"), 60, "ceres_magnet_climb"),
        (("LEFT",), 320, "ceres_magnet_exit"),
        (("LEFT", "A"), 40, "ceres_falling_room"),
        (("LEFT",), 380, "ceres_falling_room"),
        (("LEFT", "A"), 70, "ceres_elevator_lower_ledge"),
    )
    return [ActionSpan(names, frames, reason) for names, frames, reason in raw]


def play_ceres_outbound_to_ridley(session: RouteSession) -> None:
    """Ceres elevator → Ridley + natural countdown (classic L↔R arm-pump).

    Elev→Scientist keeps product geometry with arm-pump injected on RIGHT+B
    holds. Scientist→Flat→Ridley is room-gated arm-pump (drops ~600f of product
    idle+dash pad). Escape re-solves reactively — never restore product reverse.
    """
    session.spans(_ceres_outbound_to_scientist_spans())
    # Scientist → Flat → Ridley: classic L↔R room-gated run.
    if session.state.room_id != ROOM_CERES_RIDLEY:
        _ceres_arm_pump_until(
            session,
            "RIGHT",
            reason="ceres_out_flat_band",
            max_frames=900,
            done=lambda s: s.room_id == ROOM_CERES_RIDLEY,
            stuck_jump_after=40,
        )
    _ceres_wait_ordinary(
        session, ROOM_CERES_RIDLEY, reason="ceres_ridley_door", timeout=200
    )
    # Natural Ridley damage + countdown (energy assist suspended on Ceres).
    session.wait_until(
        lambda state: state.room_id == ROOM_CERES_RIDLEY
        and state.timer_type == 3
        and state.health <= 27,
        timeout=6_000,
        reason="ceres_ridley_natural_countdown",
    )


def play_ceres_escape_to_landing(session: RouteSession) -> None:
    """Ceres reverse + elev → Zebes Landing (arm-pump + WRAM-reactive).

    Speed every reverse room with classic L↔R. When a faster prefix shifts
    entry kinematics, re-solve magnet / falling / elev from room, y, pose,
    knockback, and enemy0 — never restore product open-loop budgets.
    """
    # Leave Ridley left (jump clear of platform).
    session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
    # Flat + Scientist: arm-pump reverse into Magnet.
    if session.state.room_id != ROOM_CERES_MAGNET:
        _ceres_arm_pump_until(
            session,
            "LEFT",
            reason="ceres_reverse_arm_pump",
            max_frames=700,
            done=lambda s: s.room_id == ROOM_CERES_MAGNET,
        )
    _ceres_reactive_magnet_escape(session)
    _ceres_reactive_falling(session)
    _ceres_reactive_elev_climb(session)

    session.wait_until(
        lambda state: state.room_id == ROOM_LANDING_SITE and state.game_state == 8,
        timeout=3_000,
        reason="zebes_landing_transition",
    )
    stable = 0
    for _ in range(1_200):
        if session.state.samus_y == 1088:
            stable += 1
            if stable >= 30:
                break
        else:
            stable = 0
        session.step(idle_action(), "zebes_ship_final_settle")
    else:
        raise TimeoutError(f"Zebes ship never reached final settle: {session.state}")


__all__ = [
    "_ceres_outbound_to_scientist_spans",
    "_ceres_escape_spans",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
]
