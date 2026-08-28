"""Powered Wrecked Ship Basement → Main Shaft (rr-kw8t hop 1).

Phantoon leave pin is the right door ``0xCC6F`` ~(1240,139). Ship is on.
Morph-roll LEFT through the tunnel (bomb blocks respawn; X while morph).
Unmorph. Ice-until-dead Atomics (``0xE9FF``) via charge-release seat
(not tap-X from x=879), stay east of the Workrobot (``0xE8FF``),
spin-LEFT from x~750 onto ~(657,91), then tap-shot the blue ceiling
hatch into Main Shaft ``0xCAF6``. Map station LEFT is dead — skip.
Coverns (``0xEA3F``) can tank. Do not re-enter Phantoon.

https://wiki.supermetroid.run/Basement
"""

from __future__ import annotations

from super_metroid.combat.enemies import (
    ATOMIC_ID,
    COVERN_ID,
    WORKROBOT_ID,
    Intent,
    choose,
    list_enemies,
)
from super_metroid.hop_glance import LeaveMiss, raise_leave_miss
from super_metroid.leave_specs import WS_BASEMENT_TO_MAIN
from super_metroid.ram import FACING_LEFT, SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
)
from super_metroid.routes.kpdr.k6.phantoon_fight import phantoon_boss_bit_set
from super_metroid.routes.kpdr.k6.ws_basement_ice import basement_overlay_targets
from super_metroid.routes.kpdr.k6.ws_ceiling_door import (
    ceiling_door_action,
    play_ceiling_door,
    settle_ceiling_dest,
    tap_up_action,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_PHANTOON,
    ROOM_WS_BASEMENT,
    ROOM_WS_MAIN,
    ROOM_WS_MAP,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.charge_shot import session_beam_charge
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)
from super_metroid.takeoff import spin_jump, walk_toward_x

WEAPON_BEAM = 0

# Pin is the Phantoon door. Tunnel clear x=900 LEFT. Hatch on ~(657,91).
# Under-hatch floor is occupied — takeoff just east of the platform wall
# (~728). A standstill jump from x~820 never reaches the lip (lands ~787);
# a running jump from x~720 hits the wall. 750–780 is the lip landing
# that sat (717,163) p48. Hold A in air (variable jump height).
WS_BASEMENT_FLOOR_Y = 170
WS_BASEMENT_TUNNEL_LIP_X = 1140
WS_BASEMENT_TUNNEL_CLEAR_X = 900
WS_BASEMENT_HATCH_X_MIN = 630
WS_BASEMENT_HATCH_X_MAX = 690
WS_BASEMENT_PLATFORM_X = 657
# Standing center on the mid platform is ~163. Floor is ~187. 175 is the
# approach band (hatch_jump). Walk only once landed ~(≤168, vy≈0) — releasing
# A at 175 cuts the jump 7px short of the lip (leftover 737,170 p82).
WS_BASEMENT_PLATFORM_Y = 175
WS_BASEMENT_SEAT_Y = 168
WS_BASEMENT_TAKEOFF_X_MIN = 750
WS_BASEMENT_TAKEOFF_X_MAX = 780
# Turnaround at the 750 lip coasts a few px west. Stay in takeoff.
WS_BASEMENT_TAKEOFF_LATCH_SLACK = 16
WS_BASEMENT_MAP_X = 80
_HATCH_OVERLAY = Intent(
    engage=frozenset({ATOMIC_ID}),
    absorb=frozenset({COVERN_ID}),
    avoid=frozenset({WORKROBOT_ID}),
)
_AIR_POSES = frozenset({19, 20, 21, 25, 26, 47, 48, 81, 82, 83, 84})
_TURNING = 14
_HATCH_SLACK = 8
_BOMB_CYCLES = 8
_SETTLE = 200
_DROP_BUDGET = 240
_TUNNEL_ROLL = 160
_RUN_BUDGET = 2400


def ws_basement_main_settled(state: SuperMetroidState) -> bool:
    """Ordinary Main Shaft handoff: room ``0xCAF6`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_basement_hatch_seat(state: SuperMetroidState) -> bool:
    """Standing on the mid platform under the blue ceiling hatch.

    Floor y~187 and gun-jump peaks do not count — those are under the
    platform, not on it.
    """
    pose = int(state.pose)
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and abs(int(state.samus_x) - WS_BASEMENT_PLATFORM_X) <= 16
        and int(state.samus_y) <= WS_BASEMENT_PLATFORM_Y
        and pose in (1, 2, 9, 10)
        and abs(int(state.velocity_y)) <= 1
    )


def hatch_jump_action(samus_x: int, samus_y: int, pose: int, frame: int) -> tuple[str, ...]:
    """Open the blue ceiling hatch from the seat, then jump through.

    Charge-release UP from the platform (y~163) for ``_DOOR_SHOOT_FRAMES``
    before A. Jumping first lands on the hatch lip ~(662,91) p4 with the
    door still closed. Lip: charge-release then A. Never L.
    """
    names = ceiling_door_action(
        samus_x,
        samus_y,
        pose,
        frame,
        seat_x=WS_BASEMENT_PLATFORM_X,
        lip_y=140,
        shaft_y=80,
        slack=_HATCH_SLACK,
        hold_charge=False,
    )
    if names is not None:
        return names
    x = int(samus_x)
    if x < WS_BASEMENT_PLATFORM_X - _HATCH_SLACK:
        return ("RIGHT", "B")
    if x > WS_BASEMENT_PLATFORM_X + _HATCH_SLACK:
        return ("LEFT", "B")
    return tap_up_action(frame, hold_charge=True)


def hatch_mount_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    velocity_y: int,
    facing: int = FACING_LEFT,
    movement_type: int = 0,
) -> tuple[str, ...]:
    """Floor: x~750–780 spin-LEFT. On-platform: walk to ~x=657.

    Drift below 750 during the LEFT turn must not resume RIGHT. Airborne
    holds A (variable jump height).
    """
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    pose_i = int(pose)
    facing_i = int(facing)
    turning = int(movement_type) == _TURNING
    if y <= WS_BASEMENT_SEAT_Y and abs(int(velocity_y)) <= 1:
        return walk_toward_x(x, WS_BASEMENT_PLATFORM_X, slack=8)
    if pose_i in _AIR_POSES:
        # Hold A: SM jump height is variable; releasing A is a short hop
        # that never clears the lip (leftover 819,181 p82).
        return spin_jump("LEFT") if facing_i == FACING_LEFT else ()
    if x > WS_BASEMENT_TAKEOFF_X_MAX:
        return ("LEFT", "B")
    in_band = x >= WS_BASEMENT_TAKEOFF_X_MIN
    latched = x >= WS_BASEMENT_TAKEOFF_X_MIN - WS_BASEMENT_TAKEOFF_LATCH_SLACK and (
        facing_i == FACING_LEFT or turning
    )
    if in_band or latched:
        if facing_i != FACING_LEFT or turning:
            return ("LEFT",)
        return spin_jump("LEFT")
    return ("RIGHT", "B")


def _guard(session: ControllerSession, label: str) -> None:
    room = int(session.state.room_id)
    if room == ROOM_WS_MAIN:
        return
    if room == ROOM_WS_MAP:
        raise TimeoutError(f"{label}: entered map 0xCCCB: {session.state}")
    if room == ROOM_PHANTOON:
        raise TimeoutError(f"{label}: back into Phantoon 0xCD13: {session.state}")
    if room != ROOM_WS_BASEMENT:
        raise TimeoutError(
            f"{label}: left Basement into 0x{room:04X}: {session.state}"
        )
    if int(session.state.samus_x) < WS_BASEMENT_MAP_X:
        raise TimeoutError(f"{label}: walked into left map door: {session.state}")


def _kb(session: ControllerSession, label: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir="LEFT",
        run_frames=6,
        spin_frames=24,
        label=label,
        stop_room_id=ROOM_WS_MAIN,
    )


def _drop_to_tunnel_floor(session: ControllerSession, label: str) -> None:
    """LEFT off the Phantoon-door ledge onto the morph-tunnel floor."""
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return
    if is_morph(int(session.state.pose)):
        return
    hold(session, 8, "LEFT", reason=f"{label}_turn")
    for _ in range(_DROP_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_MAIN:
            return
        if is_knockback(st):
            _kb(session, f"{label}_drop_kb")
            continue
        y = int(st.samus_y)
        x = int(st.samus_x)
        if (
            y >= WS_BASEMENT_FLOOR_Y
            and x <= WS_BASEMENT_TUNNEL_LIP_X + 20
            and int(st.velocity_y) == 0
            and not is_morph(int(st.pose))
        ):
            return
        hold(session, 1, "LEFT", "B", reason=f"{label}_drop")
    # Lip stall is OK — morph next. Only fail if still on the door ledge.
    if int(session.state.samus_y) < WS_BASEMENT_FLOOR_Y - 20:
        raise TimeoutError(f"{label}: did not drop to tunnel floor: {session.state}")


def _bomb_tunnel_left(session: ControllerSession, label: str) -> None:
    """Morph-roll LEFT; bomb (X) at the tunnel stall; clear past x=900."""
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return
    if int(session.state.samus_x) <= WS_BASEMENT_TUNNEL_CLEAR_X:
        return
    ensure_morph(session)
    for _cycle in range(_BOMB_CYCLES):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_MAIN:
            return
        if int(st.samus_x) <= WS_BASEMENT_TUNNEL_CLEAR_X:
            return
        prev = int(st.samus_x)
        stall = 0
        for _ in range(_TUNNEL_ROLL):
            st = session.state
            _guard(session, label)
            if int(st.room_id) == ROOM_WS_MAIN:
                return
            if int(st.samus_x) <= WS_BASEMENT_TUNNEL_CLEAR_X:
                return
            if is_knockback(st):
                _kb(session, f"{label}_roll_kb")
                stall = 0
                continue
            if not is_morph(int(st.pose)):
                ensure_morph(session)
            hold(session, 1, "LEFT", reason=f"{label}_roll")
            x = int(session.state.samus_x)
            stall = stall + 1 if abs(x - prev) < 2 else 0
            prev = x
            if stall >= 16:
                hold(session, 3, "X", reason=f"{label}_bomb")
                hold(session, 80, reason=f"{label}_boom")
                break
        else:
            if int(session.state.samus_x) <= WS_BASEMENT_TUNNEL_CLEAR_X:
                return
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return
    if int(session.state.samus_x) > WS_BASEMENT_TUNNEL_CLEAR_X:
        raise TimeoutError(
            f"{label}: morph tunnel did not clear left: {session.state}"
        )


def _run_to_hatch(session: ControllerSession, label: str) -> None:
    """Unmorph. Ice keepaway (charge-release seat). Takeoff from x~750."""
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return
    if is_morph(int(session.state.pose)):
        unmorph(session)
    select_weapon(session, WEAPON_BEAM)
    for _frame in range(_RUN_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_MAIN:
            return
        if at_ws_basement_hatch_seat(st):
            return
        if is_knockback(st):
            _kb(session, f"{label}_run_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            hold(session, 8, "UP", reason=f"{label}_unmorph")
            continue
        enemies = basement_overlay_targets(
            int(st.samus_x), int(st.samus_y), list_enemies(session)
        )
        choice = choose(
            int(st.samus_x),
            int(st.samus_y),
            int(st.facing),
            enemies,
            _HATCH_OVERLAY,
            movement_type=int(st.movement_type),
            charge=session_beam_charge(session),
            velocity_y=int(st.velocity_y),
            takeoff_x_min=WS_BASEMENT_TAKEOFF_X_MIN,
            clamp_solids=True,
        )
        if choice.buttons is not None:
            stance = choice.stance.name.lower()
            if choice.buttons:
                hold(session, 1, *choice.buttons, reason=f"{label}_{stance}")
            else:
                hold(session, 1, reason=f"{label}_{stance}_wait")
            continue
        names = hatch_mount_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.velocity_y),
            int(st.facing),
            int(st.movement_type),
        )
        if names:
            hold(session, 1, *names, reason=f"{label}_mount")
        else:
            hold(session, 1, reason=f"{label}_wait")
    if int(session.state.room_id) != ROOM_WS_MAIN and not at_ws_basement_hatch_seat(
        session.state
    ):
        raise TimeoutError(f"{label}: did not reach ceiling hatch: {session.state}")


def _jump_up_hatch(session: ControllerSession, label: str) -> None:
    """Shoot then jump UP through the blue ceiling door."""
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return
    select_weapon(session, WEAPON_BEAM)
    play_ceiling_door(
        session,
        label=label,
        dest_room=ROOM_WS_MAIN,
        lip_y=WS_BASEMENT_PLATFORM_Y,
        remount=lambda st: hatch_mount_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.velocity_y),
            int(st.facing),
            int(st.movement_type),
        ),
        door_action=lambda st, i: hatch_jump_action(
            int(st.samus_x), int(st.samus_y), int(st.pose), i
        ),
        guard=_guard,
        on_knockback=_kb,
    )


def play_ws_basement_to_main(session: ControllerSession) -> SuperMetroidState:
    """Powered basement return. Morph-roll LEFT, jump UP into Main Shaft.

    Pin: ``scratch/post_phantoon_leave.state`` ``0xCC6F`` ~(1240,139) p10
    gs=8, ``$D82B`` bit 0. Bomb the morph-tunnel obstruction (X while morph).
    Ice-until-dead Atomics from a charge-release seat (east of the
    Workrobot), takeoff x~750–780, tap-shot the blue ceiling hatch on the mid
    platform ~x=657. Lands ordinary ``gs=8`` in ``0xCAF6``.
    """
    label = "ws_basement_to_main"
    try:
        if ws_basement_main_settled(session.state):
            return session.state
        require_room(session, ROOM_WS_BASEMENT, label)
        if not phantoon_boss_bit_set(session):
            raise RuntimeError(f"{label}: Phantoon not defeated: {session.state}")
        _drop_to_tunnel_floor(session, label)
        _bomb_tunnel_left(session, label)
        _run_to_hatch(session, label)
        _jump_up_hatch(session, label)
        return settle_ceiling_dest(
            session, ROOM_WS_MAIN, label=label, settle_frames=_SETTLE
        )
    except LeaveMiss:
        raise
    except Exception as exc:
        raise_leave_miss(
            session.state,
            "ws_basement_to_main",
            WS_BASEMENT_TO_MAIN,
            room_label="Wrecked Ship Main Shaft",
            to_room=ROOM_WS_MAIN,
            exc=exc,
        )
        raise  # unreachable; keeps type checkers happy


__all__ = [
    "ATOMIC_ID",
    "ROOM_WS_MAP",
    "WEAPON_BEAM",
    "WORKROBOT_ID",
    "WS_BASEMENT_FLOOR_Y",
    "WS_BASEMENT_HATCH_X_MAX",
    "WS_BASEMENT_HATCH_X_MIN",
    "WS_BASEMENT_MAP_X",
    "WS_BASEMENT_PLATFORM_X",
    "WS_BASEMENT_PLATFORM_Y",
    "WS_BASEMENT_TAKEOFF_X_MAX",
    "WS_BASEMENT_TAKEOFF_X_MIN",
    "WS_BASEMENT_TUNNEL_CLEAR_X",
    "WS_BASEMENT_TUNNEL_LIP_X",
    "at_ws_basement_hatch_seat",
    "hatch_jump_action",
    "hatch_mount_action",
    "play_ws_basement_to_main",
    "ws_basement_main_settled",
]
