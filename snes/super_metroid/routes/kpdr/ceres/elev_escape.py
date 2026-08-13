"""Ceres elevator shaft climb after Falling → ship leave (WRAM-reactive).

Inbound settle waits for ordinary control (gs==8) at the bottom remap
(y≈651). Mid-transition y≈139 / gs 9/11 is not a high entry and is not
leave. Climb is kinematic takeoff windows (shared ``PlatformHop``), then
right-wall KB → LEFT+A → pad walk through x≈145 until gs 32.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GS_CERES_LEAVE, GS_DEAD, GS_ORDINARY
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_ELEV_HOPS,
    _CERES_ARM_PUMP_PERIOD,
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_LEDGE_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.runtime import ActionSpan, RouteSession
from super_metroid.routes.skills.geometry import (
    CROUCH_POSES,
    GUN_JUMP_POSES,
    LAND_POSES,
    POSE_KNOCKBACK,
    SPIN_POSES,
    STAND_LOCOMOTION_POSES,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import (
    PlatformHop,
    TakeoffWindow,
    approach_window,
    hop_for_y,
    next_hop_above,
    should_release_over,
    spin_jump,
    walk_toward_x,
)

_SHAFT_RELEASE = 2
_SHAFT_BUDGET = 1_400


@dataclass
class CeresShaftClimb:
    """Kinematic spin-hops for the Ceres elevator shaft.

    Approach arm-pumps L↔R until the takeoff window (x, x_sub, momentum,
    facing). ``dir+A`` without B is a gun-jump and never latches.
    """

    hops: tuple[PlatformHop, ...] = CERES_ELEV_HOPS
    side: str = "RIGHT"
    pump_i: int = 0
    kb_i: int = 0
    last_ground_y: int = 700
    releasing: bool = False
    release_i: int = 0
    release_frames: int = _SHAFT_RELEASE

    def _spin(self) -> tuple[str, ...]:
        return spin_jump(self.side)

    def _approach(self, state, hop: PlatformHop) -> tuple[str, ...]:
        self.side = hop.side
        names, self.pump_i = approach_window(
            state, hop, pump_i=self.pump_i, period=_CERES_ARM_PUMP_PERIOD
        )
        return names

    def action(self, state, *, knockback: bool = False) -> tuple[str, ...]:
        """One-frame shaft input. Mutates hold/release so callers stay pure-ish."""
        if _ceres_elev_leaving(state) or _ceres_elev_ship_band(state):
            return ()
        if int(getattr(state, "game_state", GS_ORDINARY)) != GS_ORDINARY:
            return ()

        x = int(state.samus_x)
        y = int(state.samus_y)
        pose = int(state.pose)
        vy = int(state.velocity_y)
        hop = hop_for_y(y, self.hops)
        nxt = next_hop_above(self.last_ground_y, self.hops)
        planted = pose in STAND_LOCOMOTION_POSES or pose in LAND_POSES
        grounded = planted and abs(vy) <= 1

        if pose in CROUCH_POSES:
            return ("UP",)

        if y >= _CERES_ELEV_BOTTOM_Y - 15 and x < 90 and grounded:
            self.side = "RIGHT"
            return ("RIGHT", "A")

        if knockback or pose in POSE_KNOCKBACK:
            self.releasing = False
            self.release_i = 0
            self.kb_i += 1
            self.side = "RIGHT" if x < 128 else "LEFT"
            if self.kb_i < 6:
                return ()
            return (self.side, "B")
        self.kb_i = 0

        if pose == POSE_WALL_LATCH:
            self.releasing = True
            self.release_i = 0
            return ()

        if self.releasing:
            self.release_i += 1
            if self.release_i < self.release_frames:
                return ()
            self.releasing = False
            self.side = "RIGHT" if self.side == "LEFT" else "LEFT"
            return self._spin()

        if pose in GUN_JUMP_POSES:
            if abs(vy) <= 1 and y >= _CERES_ELEV_LEDGE_Y - 15:
                return (self.side, "B")
            return ()

        if grounded and y > _CERES_ELEV_TOP_Y + 40:
            if abs(y - self.last_ground_y) > 10:
                self.pump_i = 0
            self.last_ground_y = y
            recipe = hop or PlatformHop(
                y,
                max(0, x - 40),
                x + 40,
                TakeoffWindow((max(0, x - 20), x + 20), self.side),
            )
            self.side = recipe.side
            if pose in LAND_POSES:
                interior = "RIGHT" if x < (recipe.x_lo + recipe.x_hi) // 2 else "LEFT"
                return (interior,)
            if recipe.ready(state) or recipe.at_ledge_end(x):
                return self._spin()
            return self._approach(state, recipe)

        in_air = pose in SPIN_POSES or (not planted and abs(vy) > 1)
        if in_air:
            recipe = hop_for_y(self.last_ground_y, self.hops)
            release_vy = recipe.takeoff.release_vy if recipe is not None else 0
            if should_release_over(state, nxt, release_vy=release_vy):
                return ()
            if recipe is not None:
                self.side = recipe.side
            return self._spin()

        # Turn / gun-rise / other grounded-ish poses: keep approaching.
        if hop is not None:
            return self._approach(state, hop)
        return (self.side, "B")


def climb_ceres_shaft_action(
    state,
    climb: CeresShaftClimb | None = None,
    *,
    knockback: bool = False,
) -> tuple[str, ...]:
    """Pure-ish one-frame Ceres shaft action (tests + live climb)."""
    machine = climb if climb is not None else CeresShaftClimb()
    return machine.action(state, knockback=knockback)


def _ceres_elev_ship_band(state) -> bool:
    """Grounded on ship pad (product leave ~x145 y75 pose 2/10 → gs 32)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and int(state.samus_y) <= _CERES_ELEV_SHIP_Y
        and abs(int(state.velocity_y)) <= 1
    )


def ship_pad_action(state) -> tuple[str, ...]:
    """Walk through the Ceres pad x that starts gs 32."""
    return walk_toward_x(int(state.samus_x), _CERES_ELEV_SHIP_X)


def _ceres_elev_leaving(state) -> bool:
    """True ship leave: left the elevator, or Ceres success / Zebes load.

    Inbound Falling→elev door (gs 9/11, often fake y≈139) is not leave.
    """
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return True
    return int(state.game_state) in GS_CERES_LEAVE


def _ceres_elev_top_seat(state) -> bool:
    """s10 land / right-wall KB — the only shaft→ship handoff."""
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return False
    if int(state.game_state) != GS_ORDINARY:
        return False
    if abs(int(state.samus_y) - _CERES_ELEV_TOP_Y) > 16:
        return False
    x = int(state.samus_x)
    if int(state.pose) in POSE_KNOCKBACK and x >= _CERES_ELEV_TOP_X - 30:
        return True
    return x >= _CERES_ELEV_TOP_X - 20


def _ceres_on_elev_ledge(state) -> bool:
    """Planted on the mid-shaft ledge (product pin y=571)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6
        and int(state.velocity_y) == 0
        and int(state.pose) not in POSE_KNOCKBACK
        and int(state.samus_x) < 200
    )


def _ceres_reactive_elev_climb(session: RouteSession) -> None:
    """Elev after Falling → ship leave. WRAM-reactive bottom→ledge→shaft.

    Door transition remaps to the bottom floor (y≈651). Mid-transition coords
    can still read y≈139 from Falling — that is not a high-entry path.
    """
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    for _ in range(160):
        st = session.state
        if _ceres_elev_leaving(st):
            return
        if (
            st.room_id == ROOM_CERES_ELEVATOR
            and st.game_state == GS_ORDINARY
            and int(st.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
        ):
            break
        session.step(buttons("LEFT"), "ceres_elev_entry")

    if (
        int(session.state.game_state) == 8
        and int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 30
    ):
        for _ in range(4):
            session.step(idle_action(), "ceres_elev_bottom_plant")
        session.span(ActionSpan(("LEFT", "A"), 70, "ceres_elev_ledge_jump"))

    for _ in range(50):
        st = session.state
        if _ceres_elev_leaving(st) or _ceres_on_elev_ledge(st):
            break
        if int(st.samus_y) <= _CERES_ELEV_SHIP_Y:
            break
        if int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 20:
            session.step(buttons("LEFT", "A"), "ceres_elev_ledge_recover")
        else:
            session.step(idle_action(), "ceres_elev_ledge_settle")

    # Walk to the left seat, then idle KB so the shaft can runup + spin-jump.
    for _ in range(70):
        st = session.state
        if _ceres_elev_leaving(st):
            return
        on_ledge = abs(int(st.samus_y) - _CERES_ELEV_LEDGE_Y) <= 8
        if int(st.samus_x) <= 55 and on_ledge and not is_knockback(st):
            break
        if int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 16:
            session.step(buttons("LEFT", "A"), "ceres_elev_reseat")
            continue
        if is_knockback(st):
            session.step(idle_action(), "ceres_elev_ledge_kb")
            continue
        session.step(buttons("LEFT"), "ceres_elev_ledge_walk")

    _trace_point(session, "left_seat")
    _ceres_reactive_shaft(session)
    if session.state.room_id == ROOM_CERES_ELEVATOR:
        _ceres_elev_top_to_ship(session)


def _trace_point(session: RouteSession, label: str) -> None:
    trace = getattr(session, "ceres_shaft_trace", None)
    if trace is None:
        return
    st = session.state
    trace.append(
        {
            "i": -1,
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "kb": int(is_knockback(st)),
            "side": label,
            "hold_i": 0,
            "rel": 0,
            "act": [],
            "best_y": int(st.samus_y),
        }
    )


def _ceres_reactive_shaft(session: RouteSession) -> None:
    """Spin-jump until the s10 seat, ship pad, real leave, or timeout."""
    climb = CeresShaftClimb()
    best_y = int(session.state.samus_y)
    trace = getattr(session, "ceres_shaft_trace", None)
    for i in range(_SHAFT_BUDGET):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR:
            return
        if _ceres_elev_leaving(st):
            return
        if _ceres_elev_top_seat(st) or _ceres_elev_ship_band(st):
            return
        if st.game_state in GS_DEAD:
            raise TimeoutError(f"ceres elev death during climb: {st}")
        if st.game_state != GS_ORDINARY:
            session.step(idle_action(), "ceres_elev_wait_gs")
            continue
        y = int(st.samus_y)
        if y < best_y:
            best_y = y
        names = climb.action(st, knockback=is_knockback(st))
        if trace is not None and i % 15 == 0:
            trace.append(
                {
                    "i": i,
                    "x": int(st.samus_x),
                    "y": int(st.samus_y),
                    "pose": int(st.pose),
                    "kb": int(is_knockback(st)),
                    "side": climb.side,
                    "hold_i": climb.pump_i,
                    "rel": int(climb.releasing),
                    "act": list(names),
                    "best_y": best_y,
                }
            )
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_shaft",
        )
    st = session.state
    if st.room_id == ROOM_CERES_ELEVATOR and not _ceres_elev_leaving(st):
        if not (_ceres_elev_top_seat(st) or _ceres_elev_ship_band(st)):
            raise TimeoutError(
                f"ceres elev climb timed out best_y={best_y}: {st}"
            )


def _ceres_elev_top_to_ship(session: RouteSession) -> None:
    """From s10 land (~y171) force right-wall pose 137 then LEFT+A to ship pad.

    Product (open-loop s10 tail): land x211 y171 pose 9 → idle → pose 137 →
    LEFT+A 38 peaks ~y65 → LEFT 25 walks to x≈145 y75 → Ceres success (gs 32).
    Already on the pad still walks through ``_CERES_ELEV_SHIP_X`` until gs 32.
    """
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        return
    if _ceres_elev_leaving(session.state):
        return

    if int(session.state.pose) in CROUCH_POSES:
        for _ in range(10):
            if int(session.state.pose) not in CROUCH_POSES:
                break
            session.step(buttons("UP"), "ceres_elev_uncrouch")

    if not _ceres_elev_ship_band(session.state) and int(session.state.samus_y) < 280:
        for _ in range(40):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if is_knockback(st):
                break
            session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")

        for _ in range(4):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if not is_knockback(st) and int(st.samus_x) >= _CERES_ELEV_TOP_X - 2:
                session.step(idle_action(), "ceres_elev_top_kb")
                continue
            if is_knockback(st):
                session.step(idle_action(), "ceres_elev_top_kb")
            else:
                break

        if is_knockback(session.state):
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

    for _ in range(80):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        names = ship_pad_action(st)
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_ship",
        )

    if (
        session.state.room_id == ROOM_CERES_ELEVATOR
        and not _ceres_elev_leaving(session.state)
    ):
        raise TimeoutError(f"ceres elev ship leave failed: {session.state}")


__all__ = [
    "CeresShaftClimb",
    "climb_ceres_shaft_action",
    "ship_pad_action",
    "_ceres_elev_ship_band",
    "_ceres_elev_leaving",
    "_ceres_elev_top_seat",
    "_ceres_on_elev_ledge",
    "_ceres_reactive_elev_climb",
    "_ceres_elev_top_to_ship",
]
