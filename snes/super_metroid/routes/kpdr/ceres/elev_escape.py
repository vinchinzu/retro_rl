"""Ceres elevator shaft climb after Falling → ship leave (WRAM-reactive)."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_clear_knockback,
    _ceres_is_knockback,
)
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_LEDGE_POSE,
    _CERES_ELEV_LEDGE_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
    _CERES_KB_POSES,
    _CERES_WALL_LATCH,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.runtime import ActionSpan, RouteSession


def _ceres_shaft_spans() -> list[ActionSpan]:
    """Elevator shaft climb — still open-loop; re-pin reactively if entry y shifts."""
    raw = (
        (("LEFT", "A"), 70),
        ((), 7),
        (("RIGHT", "A"), 67),
        ((), 1),
        (("LEFT", "A"), 67),
        ((), 19),
        (("RIGHT", "A"), 66),
        ((), 8),
        (("RIGHT", "A"), 86),
        (("LEFT", "A"), 83),
        (("RIGHT", "A"), 72),
        ((), 3),
        (("LEFT", "A"), 38),
        (("LEFT",), 25),
    )
    return [ActionSpan(names, frames, "ceres_elevator_climb") for names, frames in raw]


def _ceres_elev_ship_band(state) -> bool:
    """Grounded on ship pad (product leave ~x145 y75 pose 2/10 → gs 32)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == 8
        and int(state.samus_y) <= _CERES_ELEV_SHIP_Y
        and abs(int(state.velocity_y)) <= 1
    )


def _ceres_elev_leaving(state) -> bool:
    """Ceres success / ship cutscene or already left elev."""
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return True
    # 32–34 Ceres success; 6–7 load; 9/11 transition.
    return int(state.game_state) in (6, 7, 9, 11, 32, 33, 34)


def _ceres_on_elev_ledge(state) -> bool:
    """Planted on the mid-shaft ledge (product pin y=571)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == 8
        and abs(int(state.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6
        and int(state.velocity_y) == 0
        and int(state.pose) not in _CERES_KB_POSES
        and int(state.samus_x) < 200
    )


def _ceres_reactive_elev_climb(session: RouteSession) -> None:
    """Elev after Falling → ship leave. WRAM-reactive bottom→ledge→shaft.

    Door transition **remaps** to the bottom floor (y≈651). Mid-transition
    coords can still read y≈139 from Falling — that is not a high-entry path.
    Product: settle bottom → LEFT+A 70 → ledge y=571 ~x108 → shaft → y≈75.
    """
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    # Hold LEFT through door settle so we land walking on the bottom floor.
    # (gs=11 remaps y 139→651; LEFT+A mid-remap wastes the jump budget.)
    for _ in range(160):
        st = session.state
        if (
            st.room_id == ROOM_CERES_ELEVATOR
            and st.game_state == 8
            and int(st.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
        ):
            break
        session.step(buttons("LEFT"), "ceres_elev_entry")

    # Bottom plant then arc to mid ledge (product LEFT+A 70). Require gs=8.
    if (
        int(session.state.game_state) == 8
        and int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 30
    ):
        for _ in range(4):
            session.step(idle_action(), "ceres_elev_bottom_plant")
        session.span(ActionSpan(("LEFT", "A"), 70, "ceres_elev_ledge_jump"))
    elif not _ceres_on_elev_ledge(session.state):
        # Still mid-transition or mid-air — LEFT+A toward ledge band.
        for _ in range(100):
            if _ceres_on_elev_ledge(session.state):
                break
            if int(session.state.samus_y) <= _CERES_ELEV_SHIP_Y:
                break
            if (
                int(session.state.game_state) == 8
                and int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
            ):
                session.span(ActionSpan(("LEFT", "A"), 70, "ceres_elev_ledge_jump"))
                break
            session.step(buttons("LEFT", "A"), "ceres_elev_to_ledge")

    # Settle on ledge y=571.
    for _ in range(60):
        st = session.state
        if (
            int(st.samus_y) == _CERES_ELEV_LEDGE_Y
            and int(st.pose) == _CERES_ELEV_LEDGE_POSE
            and int(st.velocity_y) == 0
        ):
            break
        if _ceres_on_elev_ledge(st):
            session.step(idle_action(), "ceres_elev_ledge_settle")
            continue
        # Missed ledge — short recover hop.
        if int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 20:
            session.step(buttons("LEFT", "A"), "ceres_elev_ledge_recover")
        else:
            session.step(idle_action(), "ceres_elev_ledge_settle")

    # Exact product pin: y=571 pose=2 (wait, do not invent a different seat).
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR
        and int(s.samus_y) == _CERES_ELEV_LEDGE_Y
        and int(s.pose) == _CERES_ELEV_LEDGE_POSE,
        timeout=120,
        reason="ceres_lower_ledge_settle",
    )

    # Product s0 (LEFT+A 70) from the green path *stays on the ledge*
    # (ends x≈45 y=571 pose 138). Our arm-pump reverse lands with different
    # x_sub so the same LEFT+A jumps off (y→488) and desyncs the whole shaft.
    # Re-solve s0: walk LEFT on the ledge to the left seat (~x45), then product
    # s1+ climb chain. WRAM-gate: stay y≈571 until x is low.
    for _ in range(90):
        st = session.state
        if int(st.samus_x) <= 50 and abs(int(st.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6:
            break
        if int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 12:
            # Fell off — rejump to ledge.
            session.step(buttons("LEFT", "A"), "ceres_elev_reseat")
            continue
        if _ceres_is_knockback(st):
            # Product s0 often ends pose 138 on the ledge — hold through briefly.
            session.step(idle_action(), "ceres_elev_ledge_kb")
            continue
        # Prefer ground walk; brief A only if stagnant.
        session.step(buttons("LEFT"), "ceres_elev_ledge_walk")

    # Product s1 idle (hold through KB pose 138 on left seat).
    for _ in range(8):
        if _ceres_is_knockback(session.state):
            session.step(idle_action(), "ceres_elev_ledge_kb")
        else:
            session.step(idle_action(), "ceres_elev_gap")

    # Product shaft s2–s10 with debris-phase search (rr-14u).
    # Identical ledge pin under TAS vs legacy boot; only absolute frame differs.
    # Probe: idle 0 dual-green on legacy; idle 14 clears TAS boot elev.
    # Re-seat between phases — do not thrash open-loop hops.
    if session.state.room_id == ROOM_CERES_ELEVATOR:
        climbed = _ceres_product_shaft_with_phase(session)
        if climbed or int(session.state.samus_y) < _CERES_ELEV_LEDGE_Y - 50:
            # SM-CERES-ELEV-TOP: right-wall pose 137 → LEFT+A → ship pad.
            _ceres_elev_top_to_ship(session)

    if _ceres_elev_leaving(session.state) or session.state.room_id != ROOM_CERES_ELEVATOR:
        return

    # Fallback reactive hops only if top residual missed ship pad.
    best_y = int(session.state.samus_y)
    stagnant = 0
    side = "RIGHT"
    hop_i = 0
    for i in range(1_000):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR:
            return
        if _ceres_elev_leaving(st):
            session.step(idle_action(), "ceres_elev_ship_seq")
            if st.game_state in (6, 7, 32, 33, 34):
                # Hold through Ceres-success cutscene into Zebes load.
                continue
            continue
        if st.game_state in (26, 36):
            raise TimeoutError(f"ceres elev death during climb: {st}")
        if st.game_state != 8:
            session.step(idle_action(), "ceres_elev_wait_gs")
            continue

        y = int(st.samus_y)
        x = int(st.samus_x)
        pose = int(st.pose)
        if y < best_y - 2:
            best_y = y
            stagnant = 0
        else:
            stagnant += 1

        if _ceres_elev_ship_band(st):
            # Walk toward product pad x then idle for gs 32.
            if x > _CERES_ELEV_SHIP_X + 8:
                session.step(buttons("LEFT"), "ceres_elev_ship")
            elif x < _CERES_ELEV_SHIP_X - 12:
                session.step(buttons("RIGHT"), "ceres_elev_ship")
            else:
                session.step(idle_action(), "ceres_elev_ship_wait")
            continue

        if _ceres_is_knockback(st) and y <= _CERES_ELEV_TOP_Y + 20:
            # Top residual retry: plant KB then LEFT+A boost.
            for _ in range(3):
                if not _ceres_is_knockback(session.state):
                    break
                session.step(idle_action(), "ceres_elev_top_kb")
            session.step(buttons("LEFT", "A"), "ceres_elev_top_boost")
            continue

        if _ceres_is_knockback(st):
            for _ in range(6):
                if not _ceres_is_knockback(session.state):
                    break
                session.step(idle_action(), "ceres_elev_kb")
            if _ceres_is_knockback(session.state):
                _ceres_clear_knockback(
                    session, "LEFT" if x > 128 else "RIGHT", reason="ceres_elev"
                )
            continue

        if y >= _CERES_ELEV_BOTTOM_Y - 15 and abs(int(st.velocity_y)) <= 1:
            session.step(buttons("LEFT", "A"), "ceres_elev_rejump")
            side = "LEFT"
            hop_i = 0
            continue

        if pose == _CERES_WALL_LATCH:
            flip = "RIGHT" if x < 120 else "LEFT"
            session.step(buttons(flip, "A"), "ceres_elev_wj_latch")
            side = flip
            stagnant = 0
            continue

        # High band: re-seek right wall KB then boost (same residual).
        if 100 < y < 250 and abs(int(st.velocity_y)) <= 1:
            if int(st.pose) in (39, 40, 41, 42):
                session.step(buttons("UP"), "ceres_elev_uncrouch")
                continue
            if x < _CERES_ELEV_TOP_X - 2 and not _ceres_is_knockback(st):
                session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")
                continue
            session.step(buttons("LEFT", "A"), "ceres_elev_top_boost")
            continue

        hop_i += 1
        if hop_i > 65 or stagnant > 20:
            side = "RIGHT" if side == "LEFT" else "LEFT"
            hop_i = 0
            stagnant = 0
            session.step(idle_action(), "ceres_elev_flip")
            continue
        session.step(buttons(side, "A"), "ceres_elev_hop")

    if session.state.room_id == ROOM_CERES_ELEVATOR and not _ceres_elev_leaving(
        session.state
    ):
        raise TimeoutError(
            f"ceres elev climb timed out best_y={best_y}: {session.state}"
        )


# Debris-phase idles before product shaft (rr-14u probe under TAS vs legacy).
_CERES_SHAFT_PHASE_IDLES = (0, 14)


def _ceres_product_shaft_once(session: RouteSession) -> int:
    """Product s2–s10 climb. Returns best (lowest) y reached."""
    best_y = int(session.state.samus_y)
    for names, frames in (
        (("RIGHT", "A"), 67),
        ((), 1),
        (("LEFT", "A"), 67),
        ((), 19),
        (("RIGHT", "A"), 66),
        ((), 8),
        (("RIGHT", "A"), 86),
        (("LEFT", "A"), 83),
        (("RIGHT", "A"), 72),
        ((), 3),
    ):
        for _ in range(frames):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR:
                return best_y
            if _ceres_elev_leaving(st) or _ceres_elev_ship_band(st):
                session.step(idle_action(), "ceres_elev_ship_seq")
                return best_y
            y = int(st.samus_y)
            if y < best_y:
                best_y = y
            session.step(
                buttons(*names) if names else idle_action(), "ceres_elev_shaft"
            )
    return best_y


def _ceres_reseat_left_seat(session: RouteSession) -> None:
    """Re-plant left ledge seat after a failed phase (no thrash)."""
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        return
    # If on floor, product ledge hop.
    if int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 30:
        for _ in range(4):
            session.step(idle_action(), "ceres_elev_bottom_plant")
        for _ in range(70):
            if abs(int(session.state.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6:
                break
            if _ceres_is_knockback(session.state):
                break
            session.step(buttons("LEFT", "A"), "ceres_elev_ledge_jump")
        for _ in range(20):
            if (
                abs(int(session.state.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6
                and abs(int(session.state.velocity_y)) <= 1
            ):
                break
            session.step(idle_action(), "ceres_elev_ledge_settle")
    for _ in range(90):
        st = session.state
        if int(st.samus_x) <= 50 and abs(int(st.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6:
            break
        if int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 12:
            session.step(buttons("LEFT", "A"), "ceres_elev_reseat")
            continue
        if _ceres_is_knockback(st) and int(st.samus_x) <= 55:
            break
        if _ceres_is_knockback(st):
            session.step(idle_action(), "ceres_elev_ledge_kb")
            continue
        session.step(buttons("LEFT"), "ceres_elev_ledge_walk")
    for _ in range(6):
        session.step(idle_action(), "ceres_elev_gap")


def _ceres_product_shaft_with_phase(session: RouteSession) -> bool:
    """Try product shaft at phase idles 0 then 14. True if top band reached.

    Ledge pin WRAM matches across boots; debris jets are frame-phased.
    """
    for i, idle in enumerate(_CERES_SHAFT_PHASE_IDLES):
        if session.state.room_id != ROOM_CERES_ELEVATOR:
            return True
        if _ceres_elev_leaving(session.state) or _ceres_elev_ship_band(session.state):
            return True
        if int(session.state.samus_y) <= _CERES_ELEV_TOP_Y + 40:
            return True

        if i > 0:
            _ceres_reseat_left_seat(session)

        for _ in range(idle):
            if _ceres_elev_leaving(session.state):
                return True
            session.step(idle_action(), "ceres_elev_phase_align")

        best_y = _ceres_product_shaft_once(session)
        if _ceres_elev_leaving(session.state) or _ceres_elev_ship_band(session.state):
            return True
        cur_y = int(session.state.samus_y)
        # Product success: currently in top residual band.
        if cur_y <= _CERES_ELEV_TOP_Y + 50 or best_y <= _CERES_ELEV_TOP_Y + 20:
            return True
        # Phase miss — next idle after re-seat.
    return int(session.state.samus_y) <= _CERES_ELEV_TOP_Y + 80


def _ceres_elev_top_to_ship(session: RouteSession) -> None:
    """From s10 land (~y171) force right-wall pose 137 then LEFT+A to ship pad.

    Product (open-loop s10 tail): land x211 y171 pose 9 → idle → pose 137 →
    LEFT+A 38 peaks ~y65 → LEFT 25 walks to x≈145 y75 → Ceres success (gs 32).
    Arm-pump reverse s10 lands short (~x189 pose 9): walk RIGHT into the wall
    for the same KB, then the product tail. Do **not** RIGHT+A early — that
    jumps without KB and misses the height transfer.
    """
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        return
    if _ceres_elev_leaving(session.state) or _ceres_elev_ship_band(session.state):
        return

    y = int(session.state.samus_y)
    if not (100 < y < 280):
        return

    # Uncrouch if needed (pose 39–42 blocks ordinary walk).
    if int(session.state.pose) in (39, 40, 41, 42):
        for _ in range(10):
            if int(session.state.pose) not in (39, 40, 41, 42):
                break
            session.step(buttons("UP"), "ceres_elev_uncrouch")

    # Walk into right wall until pose 137/138 (or x hits product contact).
    for _ in range(40):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        if _ceres_is_knockback(st):
            break
        if int(st.samus_y) > _CERES_ELEV_TOP_Y + 30:
            # Fell off top band — abort to fallback hops.
            return
        # Ground walk only; A would jump and skip wall-contact KB.
        session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")

    # Product idle 3 on wall → stable pose 137.
    for _ in range(4):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        if not _ceres_is_knockback(st) and int(st.samus_x) >= _CERES_ELEV_TOP_X - 2:
            # At wall without KB yet — one more idle frame often flips 9→137.
            session.step(idle_action(), "ceres_elev_top_kb")
            continue
        if _ceres_is_knockback(st):
            session.step(idle_action(), "ceres_elev_top_kb")
        else:
            break

    if not _ceres_is_knockback(session.state):
        # Missed wall KB — leave to fallback hops (do not freestyle boost).
        return

    # Product s12–s13 from the same KB pin: LEFT+A 38 peaks ~y65, LEFT 25
    # walks through pad x≈145 y75 pose 10 — Ceres success (gs 32) fires mid-walk.
    for _ in range(38):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        session.step(buttons("LEFT", "A"), "ceres_elev_top_boost")

    for _ in range(25):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        session.step(buttons("LEFT"), "ceres_elev_top_walk")

    # If still ordinary on pad, nudge across the product leave x (≈143–145).
    # Product fires gs 32 while still in walk pose 10 with momentum — pure idle
    # at x≥148 pose 2 never starts the cutscene.
    for i in range(80):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        if not _ceres_elev_ship_band(st):
            return
        x = int(st.samus_x)
        # Oscillate through the pad center so the ship PLM / door triggers.
        if i % 20 < 12:
            session.step(buttons("LEFT"), "ceres_elev_ship")
        elif i % 20 < 16:
            session.step(idle_action(), "ceres_elev_ship_wait")
        else:
            session.step(buttons("RIGHT"), "ceres_elev_ship")


__all__ = [
    "_ceres_shaft_spans",
    "_ceres_elev_ship_band",
    "_ceres_elev_leaving",
    "_ceres_on_elev_ledge",
    "_ceres_reactive_elev_climb",
    "_ceres_product_shaft_once",
    "_ceres_product_shaft_with_phase",
    "_ceres_elev_top_to_ship",
]
