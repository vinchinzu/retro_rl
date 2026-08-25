"""Powered Wrecked Ship Basement → Main Shaft (rr-kw8t hop 1).

Phantoon leave pin is the right door ``0xCC6F`` ~(1240,139). Ship is on.
Morph-roll LEFT through the tunnel (bomb blocks respawn; X while morph).
Unmorph. Ice-freeze Atomics (``0xE9FF``), stay east of the Workrobot
(``0xE8FF``), spin-LEFT from x≳720 onto ~(657,91), then tap-shot the blue
ceiling hatch into Main Shaft ``0xCAF6``. Map station LEFT is dead — skip.
Coverns (``0xEA3F``) can tank. Do not re-enter Phantoon.

https://wiki.supermetroid.run/Basement
"""

from __future__ import annotations

from typing import NamedTuple

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k6.phantoon_fight import phantoon_boss_bit_set
from super_metroid.routes.kpdr.room_ids import ROOM_PHANTOON, ROOM_WS_BASEMENT, ROOM_WS_MAIN
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)
from super_metroid.takeoff import spin_jump, walk_toward_x

ROOM_WS_MAP = 0xCCCB
WEAPON_BEAM = 0

# Pin is the Phantoon door. Tunnel clear x=900 LEFT. Hatch on ~(657,91).
# Under-hatch floor is occupied — takeoff east of the Workrobot (x≳720).
WS_BASEMENT_FLOOR_Y = 170
WS_BASEMENT_TUNNEL_LIP_X = 1140
WS_BASEMENT_TUNNEL_CLEAR_X = 900
WS_BASEMENT_HATCH_X_MIN = 630
WS_BASEMENT_HATCH_X_MAX = 690
WS_BASEMENT_PLATFORM_X = 657
WS_BASEMENT_PLATFORM_Y = 130
WS_BASEMENT_TAKEOFF_X_MIN = 720
WS_BASEMENT_TAKEOFF_X_MAX = 780
WS_BASEMENT_MAP_X = 80
ATOMIC_ID = 0xE9FF
WORKROBOT_ID = 0xE8FF
_ENEMY_BASE = 0x0F78
_ENEMY_STRIDE = 0x40
_MAX_ENEMY_SLOTS = 8
_ICE_KEEPAWAY_PX = 400
_ATOMIC_PATH_X_MIN = 400
_ATOMIC_PATH_X_MAX = 1100
_ATOMIC_OVERLAP_PX = 24
_ATOMIC_AIM_UP_DY = -40
_ICE_TAP_FRAMES = 2
_ICE_RELEASE_FRAMES = 6
_ROBOT_GAP_PX = 48
_MOVEMENT_STUN = 14
_BOMB_CYCLES = 8
_SETTLE = 200
_DROP_BUDGET = 240
_TUNNEL_ROLL = 160
_RUN_BUDGET = 960
_DOOR_BUDGET = 480


def ws_basement_main_settled(state: SuperMetroidState) -> bool:
    """Ordinary Main Shaft handoff: room ``0xCAF6`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_basement_hatch_seat(state: SuperMetroidState) -> bool:
    """True under the blue ceiling hatch of powered Basement."""
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and WS_BASEMENT_HATCH_X_MIN <= int(state.samus_x) <= WS_BASEMENT_HATCH_X_MAX
        and int(state.samus_y) <= WS_BASEMENT_PLATFORM_Y
    )


def hatch_jump_action(samus_x: int, samus_y: int, pose: int, frame: int) -> tuple[str, ...]:
    """Jump-up through the blue ceiling hatch. Tap X (Charge is on). Never L."""
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    if x < WS_BASEMENT_HATCH_X_MIN:
        return ("RIGHT", "B")
    if x > WS_BASEMENT_HATCH_X_MAX:
        return ("LEFT", "B")
    # Charge beam: hold X charges. Tap 2f then jump. Door is the mid platform ~x=657.
    phase = int(frame) % 40
    if y > WS_BASEMENT_PLATFORM_Y:
        return ("UP", "A") if phase >= 4 else ("UP", "X")
    if phase < 2:
        return ("UP", "X")
    if phase < 8:
        return ("UP",)
    if phase < 28:
        return ("UP", "A")
    return ("UP",)


class BasementEnemy(NamedTuple):
    """One enemy slot from ``$0F78 + i*0x40`` (id/x/y/hp/freeze at +0x26)."""

    slot: int
    enemy_id: int
    x: int
    y: int
    hp: int
    freeze_timer: int


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def list_basement_enemies(session: ControllerSession) -> tuple[BasementEnemy, ...]:
    """Scan slots 0–7. Empty when the session has no ``env.get_ram``."""
    env = getattr(session, "env", None)
    get_ram = getattr(env, "get_ram", None) if env is not None else None
    if get_ram is None:
        return ()
    ram = get_ram()
    out: list[BasementEnemy] = []
    for slot in range(_MAX_ENEMY_SLOTS):
        base = _ENEMY_BASE + slot * _ENEMY_STRIDE
        enemy_id = _u16(ram, base)
        if enemy_id == 0:
            continue
        hp = _u16(ram, base + 0x14)
        if hp <= 0:
            continue
        x = _u16(ram, base + 0x02)
        y = _u16(ram, base + 0x06)
        if x >= 0xFE00 or y >= 0xFE00:
            continue
        out.append(
            BasementEnemy(
                slot=slot,
                enemy_id=enemy_id,
                x=x,
                y=y,
                hp=hp,
                freeze_timer=_u16(ram, base + 0x26),
            )
        )
    return tuple(out)


def ice_keepaway_action(
    samus_x: int,
    samus_y: int,
    facing: int,
    enemies: tuple[BasementEnemy, ...],
) -> tuple[str, ...] | None:
    """Shoot path Atomics until hp=0. None = none left to kill.

    Frozen is not dead. Aim UP when the blob is above the beam line.
    Caller tap-releases X so Charge actually fires.
    """
    atomics = [
        enemy
        for enemy in enemies
        if int(enemy.enemy_id) == ATOMIC_ID
        and int(enemy.hp) > 0
        and _ATOMIC_PATH_X_MIN <= int(enemy.x) <= _ATOMIC_PATH_X_MAX
    ]
    if not atomics:
        return None
    sx, sy = int(samus_x), int(samus_y)
    nearest = min(
        atomics, key=lambda enemy: (int(enemy.x) - sx) ** 2 + (int(enemy.y) - sy) ** 2
    )
    dx = int(nearest.x) - sx
    dy = int(nearest.y) - sy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > _ICE_KEEPAWAY_PX:
        return None
    if dy <= _ATOMIC_AIM_UP_DY and abs(dx) > 12:
        return ("LEFT",) if dx < 0 else ("RIGHT",)
    if dx < 0 and int(facing) != FACING_LEFT:
        return ("LEFT",)
    if dx > 0 and int(facing) != FACING_RIGHT:
        return ("RIGHT",)
    if dy <= _ATOMIC_AIM_UP_DY:
        return ("UP", "X")
    return ("X",)


def workrobot_avoid_action(
    samus_x: int,
    samus_y: int,
    enemies: tuple[BasementEnemy, ...],
) -> tuple[str, ...] | None:
    """Do not walk into Workrobot ``0xE8FF``. None = path is clear.

    Empty tuple = idle (let the robot walk). Under-hatch is occupied —
    flee east to the takeoff band.
    """
    robots = [
        enemy
        for enemy in enemies
        if int(enemy.enemy_id) == WORKROBOT_ID
        and int(enemy.hp) > 0
        and abs(int(enemy.y) - int(samus_y)) < 50
    ]
    if not robots:
        return None
    sx = int(samus_x)
    nearest = min(robots, key=lambda enemy: abs(int(enemy.x) - sx))
    gap = int(nearest.x) - sx
    if abs(gap) >= _ROBOT_GAP_PX:
        return None
    if sx < WS_BASEMENT_TAKEOFF_X_MIN:
        if gap > 16:
            return ("RIGHT", "B")
        if gap > 0:
            return ()
        return ("RIGHT", "B")
    return ()


def hatch_mount_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    velocity_y: int,
) -> tuple[str, ...]:
    """Floor → ~(657,91) from x≳720 spin-LEFT. Never L. Under-hatch is occupied."""
    del velocity_y
    if int(pose) in (137, 138):
        return ()
    x = int(samus_x)
    y = int(samus_y)
    if y <= WS_BASEMENT_PLATFORM_Y:
        return walk_toward_x(x, WS_BASEMENT_PLATFORM_X, slack=8)
    if x > WS_BASEMENT_TAKEOFF_X_MAX:
        return ("LEFT", "B")
    if x >= WS_BASEMENT_TAKEOFF_X_MIN:
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
    """Unmorph. Ice keepaway. Takeoff east of the robot onto ~(657,91)."""
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
        if is_morph(int(st.pose)) or int(st.pose) in (37, 38):
            unmorph(session)
            hold(session, 8, "UP", reason=f"{label}_unmorph")
            continue
        enemies = list_basement_enemies(session)
        keepaway = ice_keepaway_action(
            int(st.samus_x), int(st.samus_y), int(st.facing), enemies
        )
        if keepaway is not None:
            if "X" in keepaway:
                hold(session, _ICE_TAP_FRAMES, *keepaway, reason=f"{label}_ice")
                hold(session, _ICE_RELEASE_FRAMES, reason=f"{label}_ice_shot")
            elif keepaway:
                hold(session, 1, *keepaway, reason=f"{label}_ice")
            else:
                hold(session, 1, reason=f"{label}_ice_wait")
            continue
        if int(getattr(st, "movement_type", 0) or 0) == _MOVEMENT_STUN:
            hold(session, _ICE_TAP_FRAMES, "X", reason=f"{label}_ice_stun")
            hold(session, _ICE_RELEASE_FRAMES, reason=f"{label}_ice_shot")
            continue
        avoid = workrobot_avoid_action(int(st.samus_x), int(st.samus_y), enemies)
        if avoid is not None:
            if avoid:
                hold(session, 1, *avoid, reason=f"{label}_robot")
            else:
                hold(session, 1, reason=f"{label}_robot_wait")
            continue
        names = hatch_mount_action(
            int(st.samus_x), int(st.samus_y), int(st.pose), int(st.velocity_y)
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
    for frame in range(_DOOR_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_MAIN:
            return
        if is_knockback(st):
            _kb(session, f"{label}_door_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        names = hatch_jump_action(
            int(st.samus_x), int(st.samus_y), int(st.pose), frame
        )
        if names:
            hold(session, 1, *names, reason=f"{label}_door")
        else:
            hold(session, 1, reason=f"{label}_hurt")
    if int(session.state.room_id) != ROOM_WS_MAIN:
        raise TimeoutError(f"{label}: ceiling hatch missed: {session.state}")


def play_ws_basement_to_main(session: ControllerSession) -> SuperMetroidState:
    """Powered basement return. Morph-roll LEFT, jump UP into Main Shaft.

    Pin: ``scratch/post_phantoon_leave.state`` ``0xCC6F`` ~(1240,139) p10
    gs=8, ``$D82B`` bit 0. Bomb the morph-tunnel obstruction (X while morph).
    Ice-freeze Atomics, takeoff east of the Workrobot (x≳720), tap-shot the
    blue ceiling hatch on the mid platform ~x=657. Lands ordinary ``gs=8``
    in ``0xCAF6``.
    """
    label = "ws_basement_to_main"
    if ws_basement_main_settled(session.state):
        return session.state
    require_room(session, ROOM_WS_BASEMENT, label)
    if not phantoon_boss_bit_set(session):
        raise RuntimeError(f"{label}: Phantoon not defeated: {session.state}")
    _drop_to_tunnel_floor(session, label)
    _bomb_tunnel_left(session, label)
    _run_to_hatch(session, label)
    _jump_up_hatch(session, label)
    return wait_ordinary_room(
        session, ROOM_WS_MAIN, settle_frames=_SETTLE, label=label
    )


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
    "BasementEnemy",
    "at_ws_basement_hatch_seat",
    "hatch_jump_action",
    "hatch_mount_action",
    "ice_keepaway_action",
    "list_basement_enemies",
    "play_ws_basement_to_main",
    "workrobot_avoid_action",
    "ws_basement_main_settled",
]
