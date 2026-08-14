"""Ceres outbound (elev→Ridley) and escape (Ridley→Landing) play callables."""

from __future__ import annotations

from retro_harness.actions import idle_action
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _arm_pump_dash_spans,
    _ceres_arm_pump_until,
    _ceres_wait_ordinary,
)
from super_metroid.routes.skills.knockback import is_knockback
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


# (buttons, frames, arm_pump on dir+B). Rooms 1–3 slice this table.
# 0:5 elevator→falling, 5:18 falling→magnet, 18:21 magnet→scientist.
CERES_OUTBOUND_TO_SCIENTIST_RAW: list[tuple[tuple[str, ...], int, bool]] = [
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


def expand_ceres_outbound_spans(
    raw: list[tuple[tuple[str, ...], int, bool]] | None = None,
    *,
    reason: str = "ceres_outbound",
) -> list[ActionSpan]:
    """Expand the scientist-prefix raw table, including arm-pump dashes."""
    out: list[ActionSpan] = []
    for names, frames, pump in raw or CERES_OUTBOUND_TO_SCIENTIST_RAW:
        if pump:
            direction = "LEFT" if "LEFT" in names else "RIGHT"
            out.extend(_arm_pump_dash_spans(direction, frames, reason))
        else:
            out.append(ActionSpan(names, frames, reason))
    return out


def _ceres_outbound_to_scientist_spans() -> list[ActionSpan]:
    """Elevator → Scientist open-loop with classic L↔R on every RIGHT+B dash."""
    return expand_ceres_outbound_spans()


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


def play_ceres_to_ridley_door(session: RouteSession) -> None:
    """Ceres elevator → Ridley room ordinary settle (no fight)."""
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


def play_ceres_outbound_to_ridley(session: RouteSession) -> None:
    """Ceres elevator → Ridley + countdown (classic L↔R arm-pump).

    Elev→Scientist keeps product geometry with arm-pump injected on RIGHT+B
    holds. Scientist→Flat→Ridley is room-gated arm-pump (drops ~600f of product
    idle+dash pad). Fight body is tail-tank :func:`play_ceres_ridley_fight`.
    Escape re-solves magnet / falling / elev reactively.
    """
    play_ceres_to_ridley_door(session)
    # Late import: combat.__init__ → progression → early_spine → this module.
    from super_metroid.combat.ceres_ridley import (
        CeresRidleyStrategy,
        play_ceres_ridley_fight,
        require_ceres_ridley_countdown,
    )

    # Energy assist is already suspended on Ceres.
    evidence = play_ceres_ridley_fight(session, strategy=CeresRidleyStrategy())
    require_ceres_ridley_countdown(evidence)
    # Tail-tank often ends in KB (pose 137/138). Escape LEFT+A needs standing.
    for _ in range(40):
        if not is_knockback(session.state):
            break
        session.step(idle_action(), "ceres_ridley_settle")


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
    "CERES_OUTBOUND_TO_SCIENTIST_RAW",
    "expand_ceres_outbound_spans",
    "_ceres_outbound_to_scientist_spans",
    "_ceres_escape_spans",
    "play_ceres_to_ridley_door",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
]
