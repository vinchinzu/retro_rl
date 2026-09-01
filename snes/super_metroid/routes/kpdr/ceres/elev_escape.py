"""Ceres elevator shaft climb after Falling → ship leave (WRAM-reactive).

Inbound settle waits for ordinary control (gs==8). A spin jump through the
previous door preserves a y≈628 fast phase. TAS air-turns LEFT and latches
pose 132 at x≈204 y≈597, left of the Ceres-door overlay ($E23F at 224/232).
Steam jets ($E1FF) hide/show via $0F88 bit 2; shown steam knockbacks and is
absorbed as a d-boost, not idled. Ship handoff remains right-wall KB → LEFT+A.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.env import write_state_bytes
from super_metroid.combat.enemies import CERES_DOOR_ID, list_enemies
from super_metroid.combat.enemies.species import enemy_overlaps, steam_is_burning
from super_metroid.ram import GS_CERES_LEAVE, GS_ORDINARY
from super_metroid.routes.controller_common import POSE_WALL_LATCH, is_wall_latch
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
from super_metroid.routes.skills.walljump import (
    PreciseWallJumpTiming,
    precise_walljump_once,
)
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
_POSE_WALL_LATCH_LEFT = 131
# TAS air-turns LEFT and latches 132 at x≈204 y≈597, left of the Ceres-door
# overlay ($E23F at 224/232). This pin is already falling (vy=+4 at y=637);
# LEFT-into dumps the pit. RIGHT-into still misses the overlay (trace
# ceres_door_overlay) and seats 571. Steam ($E1FF) is a different slot.
_CERES_FAST_ENTRY_WALLJUMP = PreciseWallJumpTiming(
    into="RIGHT",
    away="LEFT",
    coast_frames=10,
    into_frames=12,
    release_frames=2,
    jump_frames=36,
)
# Sniq 100% lsnes: 2-frame plant at (163, 475) pose 165/10, then LEFT+A off
# the seat, left-wall latch pose 131, land (156, 363). RIGHT runup from this
# plant hits x=211 y=651 p137.
_CERES_475_TO_363_WALLJUMP = PreciseWallJumpTiming(
    into="LEFT",
    away="RIGHT",
    coast_frames=0,
    into_frames=16,
    release_frames=2,
    jump_frames=24,
)
# Missed y=628 WJ plants 475 at x≈123. Steam ($E1FF) in the hop box used
# to wait 32f/36f; idle 0–28 dumped y=651 p137. Absorb the burn instead.
_CERES_475_LOW_X = 140
_CERES_475_DEBRIS_IDLE = 0
_CERES_363_HIGH_X = 175
_CERES_363_LOW_X_IDLE = 0
_CERES_363_HIGH_X_IDLE = 0
_CERES_STEAM_DBOOST_FRAMES = 24


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

        if pose == POSE_WALL_LATCH or pose == _POSE_WALL_LATCH_LEFT:
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
    """Planted on the mid-shaft ledge (y=571)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - _CERES_ELEV_LEDGE_Y) <= 6
        and int(state.velocity_y) == 0
        and int(state.pose) not in POSE_KNOCKBACK
        and int(state.samus_x) < 200
    )


def _ceres_seat_ledge(session: RouteSession) -> None:
    """Bottom floor → mid-shaft ledge when the pose-25 door phase was missed."""
    if (
        int(session.state.game_state) == GS_ORDINARY
        and int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 30
    ):
        for _ in range(4):
            session.step(idle_action(), "ceres_elev_bottom_plant")
        session.span(ActionSpan(("LEFT", "A"), 70, "ceres_elev_ledge_jump"))

    for _ in range(400):
        st = session.state
        if _ceres_elev_leaving(st) or _ceres_on_elev_ledge(st):
            return
        if int(st.samus_y) <= _CERES_ELEV_SHIP_Y:
            return
        if st.game_state != GS_ORDINARY:
            session.step(idle_action(), "ceres_elev_ledge_wait_gs")
            continue
        if is_knockback(st):
            session.step(idle_action(), "ceres_elev_ledge_kb")
            continue
        x = int(st.samus_x)
        y = int(st.samus_y)
        if x < 90 and y >= _CERES_ELEV_BOTTOM_Y - 20:
            session.step(buttons("RIGHT", "A"), "ceres_elev_pit")
            continue
        if y > _CERES_ELEV_LEDGE_Y + 16:
            session.step(buttons("LEFT", "A"), "ceres_elev_ledge_recover")
            continue
        session.step(idle_action(), "ceres_elev_ledge_settle")


def _ceres_elev_entry_action(state) -> tuple[str, ...] | None:
    """Inputs until the shaft may start. ``None`` means the window is ready.

    Hold LEFT only while the Falling door is still settling. Ordinary y≈628
    spin is the precise-WJ phase — walking LEFT there leaves x=216 and dumps
    the well. A floor remap (y≈651) is a failed product entry.
    """
    if _ceres_elev_leaving(state):
        return None
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return ()
    if int(state.game_state) != GS_ORDINARY:
        # Door-transition inputs are not movement frames. Preserve the
        # predecessor's pose-25 rise; the first ordinary frame owns the WJ.
        return ()
    if _ceres_fast_entry_window(state):
        return None
    if int(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
        return None
    return ()


def _ceres_reactive_elev_climb(session: RouteSession) -> None:
    """Elev after Falling → ship leave through one deterministic climb.

    The predecessor's late door jump remaps to y≈628 with its spin phase intact.
    The policy attempts the measured direct plant, then continues from the
    physical ledge actually reached. Both are states in this one controller;
    there is no legacy tape or alternate route policy.
    """
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    _trace_point(session, "elev_door")
    for _ in range(160):
        names = _ceres_elev_entry_action(session.state)
        if names is None:
            break
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_entry",
        )
    _trace_point(session, "elev_entry_ready")

    if not _ceres_try_fast_elev_climb(session):
        raise TimeoutError(f"ceres direct elevator climb missed: {session.state}")
    _ceres_elev_top_to_ship(session)
    if session.state.room_id == ROOM_CERES_ELEVATOR and not _ceres_elev_leaving(
        session.state
    ):
        _ceres_elev_top_to_ship(session)


def _ceres_planted_at(state, target_y: int, *, slack: int = 18) -> bool:
    return (
        int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - target_y) <= slack
        and abs(int(state.velocity_y)) <= 1
        and (
            int(state.pose) in STAND_LOCOMOTION_POSES
            or int(state.pose) in LAND_POSES
        )
    )


def _ceres_at_checkpoint(state, target_y: int, *, slack: int = 18) -> bool:
    """A grounded checkpoint, including debris knockback pose 137/138."""
    return (
        int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - target_y) <= slack
        and abs(int(state.velocity_y)) <= 1
        and (
            int(state.pose) in STAND_LOCOMOTION_POSES
            or int(state.pose) in LAND_POSES
            or int(state.pose) in POSE_KNOCKBACK
        )
    )


def _ceres_fast_entry_window(state) -> bool:
    """Natural door-jump phase from which the y=475 wall jump is repeatable."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and 210 <= int(state.samus_x) <= 220
        and 624 <= int(state.samus_y) <= 641
        and int(state.pose) in SPIN_POSES
    )


def _ceres_any_wall_latch(state) -> bool:
    """Right-wall latch 132 or left-wall latch 131 (TAS 475→363 contact)."""
    return int(state.pose) in (POSE_WALL_LATCH, _POSE_WALL_LATCH_LEFT)


def _ceres_entry_to_475(session: RouteSession) -> bool:
    """Two measured wall jumps from the natural door phase to y=475."""
    if not _ceres_fast_entry_window(session.state):
        return False

    # Reach the lower edge of $E23F, release A for one frame, then wall-jump
    # at x≈208/y620. This predecessor is lower than Sniq's native entry, so
    # the first wall jump needs a 40-frame hold before the reverse.
    for names, frames in (
        (("LEFT", "A"), 1),
        (("A",), 1),
        (("LEFT", "A"), 1),
        (("A",), 3),
        (("LEFT",), 1),
    ):
        session.span(ActionSpan(names, frames, "ceres_elev_direct_entry"))

    first_latch = False
    for _ in range(40):
        session.step(buttons("LEFT", "A"), "ceres_elev_direct_first_wj")
        first_latch = first_latch or int(session.state.pose) == POSE_WALL_LATCH
    if not first_latch:
        return False

    # Unlatch at x≈165/y524, reverse for one frame, and catch the left wall.
    # The second jump clears the underside and plants x≈164/y475 naturally.
    tail = (
        (("RIGHT", "A"), 1),
        (("RIGHT",), 1),
        (("RIGHT", "A"), 1),
        (("LEFT", "A"), 7),
        (("DOWN", "RIGHT", "A"), 1),
        (("LEFT",), 1),
        ((), 1),
        (("LEFT",), 1),
        (("LEFT", "A"), 3),
    )
    second_latch = False
    for names, frames in tail:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_direct_second_wj",
            )
            second_latch = second_latch or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 475, slack=4):
                return second_latch
    return False


def _ceres_475_to_363(session: RouteSession) -> bool:
    """Left-wall jump from the planted 475 seat onto y=363.

    TAS (lsnes sniq_100): plant x=163 pose 165/10, LEFT+A, pose 131 at
    y≈404, land x=156 y=363. Do not RIGHT-run from this seat.
    """
    if not _ceres_planted_at(session.state, 475, slack=4):
        return False
    spans = (
        (("LEFT", "A"), 3),
        (("A",), 2),
        (("RIGHT", "A"), 1),
        (("LEFT", "A"), 1),
        (("RIGHT", "A"), 1),
        (("A",), 9),
        (("RIGHT", "A"), 1),
        (("A",), 1),
        (("LEFT", "RIGHT"), 1),
        (("LEFT", "RIGHT", "A"), 1),
        (("A",), 5),
        (("A", "X"), 1),
        (("DOWN", "A"), 1),
        (("RIGHT",), 1),
        ((), 1),
        (("RIGHT",), 1),
    )
    latched = False
    for names, frames in spans:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_475_363",
            )
            latched = latched or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 363, slack=4):
                return latched
    return False


def _ceres_try_fast_elev_climb(session: RouteSession) -> bool:
    """Wall jump to y=475, 475→363 left WJ, then checkpoint to the top seat."""
    if not _ceres_fast_entry_window(session.state):
        return False
    if _ceres_door_blocks_wj(session):
        _trace_point(session, "ceres_door_overlay")
    if not _ceres_entry_to_475(session):
        _trace_point(session, "fast_walljump_miss")
        return False
    _trace_point(session, "fast_walljump_475")
    if _ceres_475_to_363(session):
        _trace_point(session, "fast_walljump_363")
    else:
        _trace_point(session, "fast_475_363_miss")
        if not any(
            _ceres_at_checkpoint(session.state, y)
            for y in (571, 475, 363, 267, 171)
        ):
            return False
    if _ceres_elev_leaving(session.state) or _ceres_elev_ship_band(session.state):
        _trace_point(session, "fast_shaft_leave")
        return True
    if _ceres_elev_top_seat(session.state):
        _trace_point(session, "fast_shaft_top")
        return True
    return _ceres_checkpoint_shaft(session)


def _ceres_checkpoint_hop(
    session: RouteSession,
    *,
    side: str,
    runup: int,
    target_y: int,
    start_x: int | None = None,
) -> bool:
    """Replay one emulator-swept platform hop and require its natural land."""
    for _ in range(16):
        pose = int(session.state.pose)
        if pose in STAND_LOCOMOTION_POSES or pose in LAND_POSES:
            break
        if pose in CROUCH_POSES:
            session.step(buttons("UP"), "ceres_elev_checkpoint_uncrouch")
        else:
            session.step(buttons("LEFT"), "ceres_elev_checkpoint_stand")
    if start_x is not None:
        for _ in range(40):
            names = walk_toward_x(int(session.state.samus_x), start_x, slack=4)
            if not names:
                break
            session.step(buttons(*names), "ceres_elev_checkpoint_position")
    for _ in range(runup):
        session.step(buttons(side, "B"), "ceres_elev_checkpoint_runup")
    for _ in range(40):
        session.step(buttons(side, "B", "A"), "ceres_elev_checkpoint_jump")
        if int(session.state.knockback_timer) > 0:
            boost_side = "LEFT" if side == "RIGHT" else "RIGHT"
            _ceres_steam_dboost(session, boost_side)
            break
    for _ in range(220):
        st = session.state
        if _ceres_elev_leaving(st) or _ceres_elev_ship_band(st):
            return True
        if is_knockback(st):
            _ceres_steam_dboost(session, side)
            st = session.state
            if _ceres_planted_at(st, target_y):
                return True
        if _ceres_planted_at(st, target_y):
            return True
        if any(
            _ceres_at_checkpoint(st, y)
            for y in (571, 475, 363, 267, 171)
        ):
            capture = os.environ.get("SM_CERES_CAPTURE_363")
            if capture and _ceres_at_checkpoint(st, 363) and not Path(capture).exists():
                write_state_bytes(Path(capture), session.env.em.get_state())
            return target_y == min(
                (y for y in (571, 475, 363, 267, 171) if _ceres_at_checkpoint(st, y)),
                key=lambda y: abs(int(st.samus_y) - y),
            )
        # Holding the hop side into the land walks off 475 into the well
        # (x=211 y=651). Coast lets the 40f RIGHT+B+A plant (180,363) p1.
        session.step(idle_action(), "ceres_elev_checkpoint_land")
    return _ceres_planted_at(session.state, target_y)


def _ceres_475_phase_idle(x: int) -> int:
    """No debris idle. Steam burns are absorbed as a d-boost."""
    del x
    return _CERES_475_DEBRIS_IDLE


def _ceres_363_phase_idle(x: int) -> int:
    """No debris idle. Steam burns are absorbed as a d-boost."""
    del x
    return _CERES_363_HIGH_X_IDLE


def _ceres_door_blocks_wj(session: RouteSession) -> bool:
    """True when a Ceres-door overlay occupies the right-wall WJ contact."""
    st = session.state
    sx, sy = int(st.samus_x), int(st.samus_y)
    for enemy in list_enemies(session):
        if int(enemy.enemy_id) != CERES_DOOR_ID:
            continue
        if enemy_overlaps(enemy, sx, sy, samus_r=16):
            return True
    return False


def _ceres_steam_in_path(session: RouteSession, x: int, y: int) -> bool:
    """True when shown Ceres steam overlaps the hop box."""
    return any(
        steam_is_burning(enemy) and enemy_overlaps(enemy, x, y, samus_r=24)
        for enemy in list_enemies(session)
    )


def _ceres_steam_dboost(session: RouteSession, side: str) -> bool:
    """Hold A+dir through steam knockback. True if still on a checkpoint."""
    for _ in range(_CERES_STEAM_DBOOST_FRAMES):
        st = session.state
        if (
            int(st.knockback_timer) <= 0
            and int(st.movement_type) not in (10, 25)
            and not is_knockback(st)
        ):
            break
        session.step(buttons(side, "A"), "ceres_elev_steam_dboost")
    st = session.state
    return any(
        _ceres_at_checkpoint(st, y) for y in (571, 475, 363, 267, 171)
    )


def _ceres_checkpoint_shaft(session: RouteSession) -> bool:
    """Natural checkpoint chain with debris-safe restart from lower seats."""
    recipes = {
        571: ("RIGHT", 0, 475, None),
        # Fast plant at x=163 uses LEFT WJ (`_ceres_475_to_363`). A plant at
        # x≈122 uses this RIGHT takeoff window.
        475: ("RIGHT", 4, 363, None),
        363: ("LEFT", 0, 267, None),
        267: ("RIGHT", 0, 171, 131),
    }
    history: list[str] = []
    for _ in range(16):
        if _ceres_at_checkpoint(session.state, _CERES_ELEV_TOP_Y):
            return True
        seat_y = next(
            (y for y in recipes if _ceres_at_checkpoint(session.state, y)),
            None,
        )
        if seat_y is None:
            if int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
                history.append(f"floor:{int(session.state.samus_y)}")
                _ceres_seat_ledge(session)
                continue
            raise TimeoutError(
                f"ceres elev lost checkpoint history={history}: {session.state}"
            )
        side, runup, target_y, start_x = recipes[seat_y]
        if seat_y == 363 and int(session.state.samus_x) < 100:
            # A live steam absorb can plant the same shelf at x≈87/y379.
            # Launch inward from that left seat; LEFT drives into the wall and
            # loops pose 137/165 without gaining a platform.
            side, runup = "RIGHT", 4
        if seat_y == 475:
            phase_idle = _ceres_475_phase_idle(int(session.state.samus_x))
            for _ in range(phase_idle):
                session.step(idle_action(), "ceres_elev_475_debris")
        if seat_y == 363:
            capture = os.environ.get("SM_CERES_CAPTURE_363")
            if capture and not Path(capture).exists():
                write_state_bytes(Path(capture), session.env.em.get_state())
            phase_idle = _ceres_363_phase_idle(int(session.state.samus_x))
            for _ in range(phase_idle):
                session.step(idle_action(), "ceres_elev_debris_phase")
        landed = _ceres_checkpoint_hop(
            session,
            side=side,
            runup=runup,
            target_y=target_y,
            start_x=start_x,
        )
        history.append(
            f"{seat_y}->{target_y}:{int(landed)}@"
            f"{int(session.state.samus_x)},{int(session.state.samus_y)},p{int(session.state.pose)}"
        )
        if not landed:
            _trace_point(session, f"checkpoint_miss_{target_y}")
        else:
            _trace_point(session, f"checkpoint_{target_y}")
    raise TimeoutError(
        f"ceres elev checkpoint retries exhausted history={history}: {session.state}"
    )


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

    # The natural checkpoint hop lands x≈203 in pose 166.  A fixed 12f LEFT
    # plants ordinary locomotion without sliding off; shorter normalization
    # reaches the wall but produces a weak boost that falls back to y267.
    if int(session.state.pose) in LAND_POSES:
        for _ in range(12):
            session.step(buttons("LEFT"), "ceres_elev_top_stand")

    for _ in range(4):
        if int(session.state.pose) in STAND_LOCOMOTION_POSES:
            break
        session.step(buttons("LEFT"), "ceres_elev_top_stand")

    if not _ceres_elev_ship_band(session.state) and int(session.state.samus_y) < 280:
        for _ in range(40):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if is_knockback(st):
                break
            session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")

        for _ in range(3):
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

            for _ in range(80):
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
    "_ceres_fast_entry_window",
    "_ceres_try_fast_elev_climb",
    "_ceres_475_to_363",
    "_ceres_475_phase_idle",
    "_ceres_363_phase_idle",
    "_ceres_door_blocks_wj",
    "_ceres_steam_dboost",
    "_ceres_steam_in_path",
    "_ceres_any_wall_latch",
    "_ceres_on_elev_ledge",
    "_ceres_seat_ledge",
    "_ceres_reactive_elev_climb",
    "_ceres_elev_top_to_ship",
]
