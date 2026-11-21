"""Restore-free RAM-driven Skills for Tourian Metroid Rooms 1–4 and Rinka Shaft.

Replaces the slow open-loop Metroid Room overrides (rows 316–319, 9,501f) and
Lower Tourian Save Room detour (rows 328–330, 1,027f) with fast, zero-restore
RAM controllers that:
  - Freeze each Metroid with Ice+Wave+Plasma, kill with 1 Super (or 5 rapid
    Missiles when topping up Supers toward max), and sweep the 6-item drop
    shower so Samus leaves Metroid Room 4 with full Supers (15/15) and Reserve
    (100/100) across shifted RNG.
  - Drop directly down Rinka Shaft to Mother Brain's red door (1 Super),
    bypassing the Lower Tourian Save Room.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.ram import (
    ADDR_HEALTH,
    ADDR_MAX_HEALTH,
    ADDR_MAX_RESERVE_HEALTH,
    ADDR_MAX_SUPER_MISSILES,
    ADDR_MISSILES,
    ADDR_RESERVE_HEALTH,
    ADDR_SELECTED_ITEM,
    ADDR_SUPER_MISSILES,
    SuperMetroidState,
    read_wram_u16,
)
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.runtime import ControllerSession

ROOM_METROID_1 = 0xDAE1
ROOM_METROID_2 = 0xDB31
ROOM_METROID_3 = 0xDB7D
ROOM_METROID_4 = 0xDBCD
ROOM_HOPPER = 0xDC19
ROOM_BIG_BOY = 0xDCB1
ROOM_SEAWEED = 0xDCFF
ROOM_RINKA_SHAFT = 0xDDF3
ROOM_MOTHER_BRAIN = 0xDD58

_SPECIES_METROID = 0xDD7F
_ENEMY_BASE = 0x0F78
_ENEMY_STRIDE = 0x40
_ENEMY_SLOTS = 16

_MORPH_POSES = frozenset({
    0x001D, 0x001E, 0x001F, 0x0031, 0x0032, 0x0041,
    0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E,
})
_SPIN_POSES = frozenset({
    0x0019, 0x001A, 0x001B, 0x001C,
    0x0081, 0x0082, 0x0083, 0x0084,
})


@dataclass(frozen=True)
class _Metroid:
    slot: int
    x: int
    y: int
    hp: int
    frz: int


def _live_metroids(env: object) -> list[_Metroid]:
    out: list[_Metroid] = []
    for i in range(_ENEMY_SLOTS):
        b = _ENEMY_BASE + i * _ENEMY_STRIDE
        if read_wram_u16(env, b) == _SPECIES_METROID:
            hp = read_wram_u16(env, b + 0x14)
            if hp > 0:
                out.append(
                    _Metroid(
                        slot=i,
                        x=read_wram_u16(env, b + 2),
                        y=read_wram_u16(env, b + 6),
                        hp=hp,
                        frz=read_wram_u16(env, b + 0x26),
                    )
                )
    return out


def _metroid_slot(env: object, slot: int) -> _Metroid | None:
    b = _ENEMY_BASE + slot * _ENEMY_STRIDE
    if read_wram_u16(env, b) != _SPECIES_METROID:
        return None
    hp = read_wram_u16(env, b + 0x14)
    if hp <= 0:
        return None
    return _Metroid(
        slot=slot,
        x=read_wram_u16(env, b + 2),
        y=read_wram_u16(env, b + 6),
        hp=hp,
        frz=read_wram_u16(env, b + 0x26),
    )


def _fast_select(session: ControllerSession, target: int, label: str) -> None:
    env = session.env
    cur = read_wram_u16(env, ADDR_SELECTED_ITEM)
    if cur == target:
        return
    if target == 0:
        hold(session, 1, "Y", reason=f"{label}_sel0")
        hold(session, 1, reason=f"{label}_sel0_s")
        if read_wram_u16(env, ADDR_SELECTED_ITEM) == 0:
            return
    elif cur > target:
        hold(session, 1, "Y", reason=f"{label}_sel_reset")
        hold(session, 1, reason=f"{label}_sel_reset_s")
    for _ in range(10):
        if read_wram_u16(env, ADDR_SELECTED_ITEM) == target:
            return
        hold(session, 1, "SELECT", reason=f"{label}_sel")
        hold(session, 1, reason=f"{label}_sel_s")
    select_weapon(session, target)


def _unmorph_if_needed(session: ControllerSession, label: str) -> None:
    for _ in range(6):
        if session.state.pose not in _MORPH_POSES:
            return
        hold(session, 2, "UP", reason=f"{label}_unmorph")
        hold(session, 2, reason=f"{label}_unmorph_s")


def _bomb_off_latch(session: ControllerSession, label: str) -> None:
    """Morph and blast a latched Metroid off (1 Power Bomb if available, else morph bombs)."""
    env = session.env
    use_pb = read_wram_u16(env, 0x09CE) > 0
    _fast_select(session, 3 if use_pb else 0, label)
    for _ in range(6):
        if session.state.pose in _MORPH_POSES:
            break
        hold(session, 2, "DOWN", reason=f"{label}_latch_morph")
        hold(session, 2, reason=f"{label}_latch_morph_s")
    if use_pb:
        hold(session, 2, "X", reason=f"{label}_latch_pb")
        for bf in range(55):
            hold(session, 1, reason=f"{label}_latch_pb_wait")
            if bf >= 20:
                s = session.state
                if not any(
                    om.frz == 0 and abs(om.x - s.samus_x) < 28 and abs(om.y - s.samus_y) < 34
                    for om in _live_metroids(env)
                ):
                    break
    else:
        for bf in range(135):
            if bf in (0, 12, 24, 65, 77, 89):
                hold(session, 1, "X", reason=f"{label}_latch_bomb")
            else:
                hold(session, 1, reason=f"{label}_latch_wait")
            if bf >= 50:
                s = session.state
                if not any(
                    om.frz == 0 and abs(om.x - s.samus_x) < 28 and abs(om.y - s.samus_y) < 34
                    for om in _live_metroids(env)
                ):
                    break
    _unmorph_if_needed(session, label)
    _fast_select(session, 0, label)


def _aim_buttons(sx: int, sy: int, tx: int, ty: int) -> tuple[str, ...]:
    dx = tx - sx
    dy = ty - (sy - 8)
    face = "RIGHT" if dx >= 0 else "LEFT"
    if abs(dx) <= 18 and dy < -10:
        return ("UP",)
    if dy < -18:
        return (face, "R")
    if dy > 18:
        return (face, "L")
    return (face,)


def _line_of_fire(sx: int, sy: int, tx: int, ty: int) -> tuple[str, ...] | None:
    """Return exact aim buttons if (tx, ty) lies on a cardinal or 45-deg ray."""
    dx = tx - sx
    dy = ty - (sy - 8)
    adx = abs(dx)
    ady = abs(dy)
    face = "RIGHT" if dx >= 0 else "LEFT"
    if 12 <= adx <= 155 and ady <= 15:
        return (face,)
    if adx <= 14 and -155 <= dy <= -10:
        return ("UP",)
    if 16 <= adx <= 145 and dy < -12 and abs(adx - ady) <= 14:
        return (face, "R")
    if 16 <= adx <= 145 and dy > 12 and abs(adx - ady) <= 14:
        return (face, "L")
    return None


def _ground_shot_target_x(sx: int, sy: int, mx: int, my: int, room_x_max: int) -> int:
    """Pick the closest ground x from which UP, straight, or 45-deg R hits (mx, my)."""
    dy = my - (sy - 8)
    if abs(dy) <= 15:
        # Straight horizontal shot: stand ~60px away on Samus's side.
        side = 1 if sx >= mx else -1
        return max(32, min(room_x_max, mx + side * 60))
    if dy < -15:
        # Above Samus: compare straight-UP (x=mx) vs 45-deg diag-up (x = mx +/- |dy|).
        cands = [mx, mx + abs(dy), mx - abs(dy)]
        valid = [max(32, min(room_x_max, cx)) for cx in cands]
        return min(valid, key=lambda cx: abs(cx - sx))
    # Below Samus: stand at diag-down distance or drop down.
    side = 1 if sx >= mx else -1
    return max(32, min(room_x_max, mx + side * max(36, abs(dy))))


def _fly_step(
    session: ControllerSession,
    tx: int,
    ty: int,
    frame: int,
    stuck: int,
    *,
    dash: bool = True,
    shoot_door: str | None = None,
    label: str = "fly",
) -> SuperMetroidState:
    s = session.state
    btns: list[str] = []
    dx = tx - s.samus_x
    if dx < -5:
        btns.append("LEFT")
    elif dx > 5:
        btns.append("RIGHT")
    if dash and ("LEFT" in btns or "RIGHT" in btns):
        btns.append("B")

    need_up = s.samus_y > ty + 6 or (stuck >= 8 and (stuck % 22) < 12)
    if need_up:
        if s.vertical_direction == 1 or (
            frame % 2 == 0 and (s.vertical_direction == 0 or s.velocity_y >= 2)
        ):
            btns.append("A")
            if "LEFT" not in btns and "RIGHT" not in btns:
                btns.append("LEFT" if s.samus_x >= tx else "RIGHT")

    if shoot_door and abs(s.samus_y - ty) <= 14 and frame % 8 == 3:
        if "A" in btns:
            btns.remove("A")
        if shoot_door == "DOWN":
            btns.append("DOWN")
        elif shoot_door in ("LEFT", "RIGHT") and shoot_door not in btns:
            btns = [b for b in btns if b not in ("LEFT", "RIGHT")]
            btns.append(shoot_door)
        btns.append("X")

    return hold(session, 1, *btns, reason=label)


def _fly_to(
    session: ControllerSession,
    waypoints: list[tuple[int, int]],
    *,
    target_room: int | None = None,
    shoot_door: str | None = None,
    max_frames: int = 600,
    label: str = "fly",
) -> SuperMetroidState:
    wp_i = 0
    prev_x = -1
    stuck = 0
    for f in range(max_frames):
        s = session.state
        if target_room is not None and s.room_id == target_room:
            return wait_ordinary_room(session, target_room, settle_frames=320, label=label)
        if s.game_state != 8:
            hold(session, 1, reason=f"{label}_wait_gs8")
            continue
        tx, ty = waypoints[wp_i]
        if abs(s.samus_x - tx) <= 18 and abs(s.samus_y - ty) <= 22:
            if wp_i + 1 < len(waypoints):
                wp_i += 1
                tx, ty = waypoints[wp_i]
                stuck = 0
            elif target_room is None:
                return s
        stuck = stuck + 1 if (s.samus_x == prev_x and abs(s.samus_x - tx) > 8) else 0
        prev_x = s.samus_x
        door = shoot_door if wp_i == len(waypoints) - 1 else None
        _fly_step(session, tx, ty, f, stuck, dash=True, shoot_door=door, label=label)
    s = session.state
    raise TimeoutError(
        f"{label}: timed out at wp {wp_i} ({waypoints[wp_i]}) "
        f"room=0x{s.room_id:04X} xy=({s.samus_x},{s.samus_y})"
    )


def _needs_drops(env: object) -> bool:
    s_cur = read_wram_u16(env, ADDR_SUPER_MISSILES)
    s_max = read_wram_u16(env, ADDR_MAX_SUPER_MISSILES)
    r_cur = read_wram_u16(env, ADDR_RESERVE_HEALTH)
    r_max = read_wram_u16(env, ADDR_MAX_RESERVE_HEALTH)
    e_cur = read_wram_u16(env, ADDR_HEALTH)
    e_max = read_wram_u16(env, ADDR_MAX_HEALTH)
    return s_cur < s_max or r_cur < r_max or e_cur < e_max - 20


def _sweep_drops(session: ControllerSession, kx: int, ky: int, label: str, frames: int = 34) -> None:
    """Sweep through a freshly killed Metroid's drop shower if stock isn't full."""
    env = session.env
    if not _needs_drops(env):
        return
    # Drops spawn ~18 frames after kill right around (kx, ky); fly across (kx, ky+10).
    prev_x = -1
    stuck = 0
    for f in range(frames):
        if not _needs_drops(env):
            return
        s = session.state
        tx = kx + (10 if (f // 10) % 2 == 0 else -10)
        ty = ky + 10
        stuck = stuck + 1 if (s.samus_x == prev_x and abs(s.samus_x - tx) > 8) else 0
        prev_x = s.samus_x
        _fly_step(session, tx, ty, f, stuck, dash=True, label=f"{label}_drop")


def _clear_metroids(
    session: ControllerSession,
    *,
    room_id: int,
    prefer_supers: bool,
    vertical_order: bool,
    room_x_max: int,
    label: str,
) -> None:
    """Clear every Metroid in the current room with zero save/restore.

    Crowd-control + ground-first freeze-and-blast:
      - If latched by an unfrozen Metroid, immediately morph + bomb it off.
      - Before firing Missiles/Supers, freeze ANY nearby unfrozen Metroid
        (<165px) with Ice+Wave+Plasma so nothing can swarm/latch during shots.
      - Align horizontally on the ground (UP / straight / 45-deg R) for fast,
        clean 1-Super or 5-Missile kills, then sweep drops if stock isn't full.
    """
    env = session.env
    _unmorph_if_needed(session, label)

    while True:
        mets = _live_metroids(env)
        if not mets:
            break
        s = session.state
        if vertical_order:
            target = min(mets, key=lambda m: (m.y // 64, abs(m.x - s.samus_x) + abs(m.y - s.samus_y)))
        else:
            target = min(mets, key=lambda m: abs(m.x - s.samus_x) + abs(m.y - s.samus_y))
        slot = target.slot
        last_kx, last_ky = target.x, target.y

        latch_f = 0
        prev_x = -1
        stuck = 0
        miss_streak = 0
        ground_y = -1
        ground_f = 0
        drop_prev_x = -1
        drop_wall_f = 0
        drop_dir = "LEFT" if s.samus_x > 128 else "RIGHT"

        for f in range(900):
            m = _metroid_slot(env, slot)
            if m is None:
                break
            last_kx, last_ky = m.x, m.y
            s = session.state
            if s.room_id != room_id:
                return

            all_mets = _live_metroids(env)

            # 1) Latch recovery: if ANY unfrozen Metroid sits on Samus, morph + bomb it off!
            latched = any(
                om.frz == 0 and abs(om.x - s.samus_x) < 28 and abs(om.y - s.samus_y) < 34
                for om in all_mets
            )
            if latched:
                latch_f += 1
                if latch_f >= 3:
                    _bomb_off_latch(session, label)
                    latch_f = 0
                    continue
            else:
                latch_f = 0

            # 2) Crowd control: freeze ANY dangerously close level unfrozen Metroid!
            threat = None
            for om in all_mets:
                if om.slot != slot and om.frz < 30:
                    if abs(om.x - s.samus_x) < 92 and abs(om.y - s.samus_y) < 42:
                        threat = om
                        break
            if threat is not None:
                _fast_select(session, 0, label)
                aim_t = _aim_buttons(s.samus_x, s.samus_y, threat.x, threat.y)
                if f % 3 == 0:
                    hold(session, 1, *aim_t, "X", reason=f"{label}_freeze_threat")
                else:
                    away = "LEFT" if threat.x >= s.samus_x else "RIGHT"
                    hold(session, 1, away, "B", reason=f"{label}_kite_threat")
                continue

            dx = m.x - s.samus_x
            dy = m.y - s.samus_y

            # 3) Vertical rooms: drop off an upper ledge when target is well below Samus.
            if vertical_order and dy > 36 and s.velocity_y == 0 and s.vertical_direction == 0:
                if abs(s.samus_y - ground_y) <= 4:
                    ground_f += 1
                    drop_wall_f = drop_wall_f + 1 if s.samus_x == drop_prev_x else 0
                    if drop_wall_f >= 5 or ground_f % 85 == 0:
                        drop_dir = "RIGHT" if drop_dir == "LEFT" else "LEFT"
                        drop_wall_f = 0
                else:
                    ground_y = s.samus_y
                    ground_f = 1
                    drop_wall_f = 0
                    drop_dir = "LEFT" if s.samus_x > 128 else "RIGHT"
                drop_prev_x = s.samus_x
                hold(session, 1, drop_dir, "B", reason=f"{label}_drop_ledge")
                continue
            else:
                ground_f = 0
                drop_wall_f = 0

            # 4) Target not frozen (or about to thaw): close in and freeze with Ice+Wave+Plasma!
            if m.frz < 50:
                _fast_select(session, 0, label)
                stuck = stuck + 1 if (s.samus_x == prev_x and abs(dx) > 48) else 0
                prev_x = s.samus_x

                in_cone = abs(dx) <= 185 and abs(dy) <= 115
                if in_cone and (f % 3 == 0 or (abs(dx) < 90 and abs(dy) < 65 and f % 2 == 0)):
                    aim = _aim_buttons(s.samus_x, s.samus_y, m.x, m.y)
                    if len(aim) == 1 and aim[0] in ("LEFT", "RIGHT") and abs(dx) > 95 and s.velocity_y == 0:
                        hold(session, 1, aim[0], "B", "X", reason=f"{label}_run_freeze")
                    else:
                        hold(session, 1, *aim, "X", reason=f"{label}_freeze")
                    continue

                # Back off if Metroid gets too close (<75px) while unfrozen, else approach to 95px.
                if abs(dx) < 75 and abs(dy) < 55:
                    away = "LEFT" if dx >= 0 else "RIGHT"
                    hold(session, 1, away, "B", reason=f"{label}_kite")
                    continue
                tx = max(32, min(room_x_max, m.x + (95 if s.samus_x >= m.x else -95)))
                ty = s.samus_y if (not vertical_order and stuck < 8 and abs(dy) < 75) else (m.y + 4)
                _fly_step(session, tx, ty, f, stuck, dash=True, label=f"{label}_approach")
                continue

            # 5) Target is frozen! Pick Super (1 hit) when stocked near max, else Missiles (5 hits)
            # so natural Metroid Super drops (+2S) reliably top Supers up to 15/15 for Mother Brain.
            s_cur = read_wram_u16(env, ADDR_SUPER_MISSILES)
            s_max = read_wram_u16(env, ADDR_MAX_SUPER_MISSILES)
            m_cur = read_wram_u16(env, ADDR_MISSILES)
            use_super = (s_cur >= s_max) or (prefer_supers and s_cur >= 14) or (m_cur < 8 and s_cur > 0)
            weapon = 2 if use_super else 1
            _fast_select(session, weapon, label)

            ray = _line_of_fire(s.samus_x, s.samus_y, m.x, m.y)
            stable_to_fire = (s.velocity_y == 0) or (abs(dx) <= 68 and abs(dy) <= 14)
            if ray is not None and stable_to_fire and miss_streak == 0:
                hp0 = m.hp
                if s.pose in _SPIN_POSES:
                    hold(session, 1, *ray, "X", reason=f"{label}_break_spin")
                hold(session, 1, *ray, reason=f"{label}_aim")
                hold(session, 1, *ray, "X", reason=f"{label}_fire")
                if weapon == 2:
                    hit = False
                    for _ in range(14):
                        hold(session, 1, *ray, reason=f"{label}_super_wait")
                        mm = _metroid_slot(env, slot)
                        if mm is None or mm.hp < hp0:
                            hit = True
                            break
                    miss_streak = 0 if hit else 1
                else:
                    hit_any = False
                    for shot_i in range(5):
                        mm = _metroid_slot(env, slot)
                        if mm is None or mm.frz < 28:
                            break
                        # Abort rapid-fire immediately if another unfrozen Metroid closes in!
                        s_mid = session.state
                        if any(
                            om.slot != slot and om.frz == 0
                            and abs(om.x - s_mid.samus_x) < 75 and abs(om.y - s_mid.samus_y) < 55
                            for om in _live_metroids(env)
                        ):
                            break
                        if shot_i > 0:
                            hold(session, 1, *ray, "X", reason=f"{label}_missile_fire")
                        for _ in range(7):
                            hold(session, 1, *ray, reason=f"{label}_missile_wait")
                            mm2 = _metroid_slot(env, slot)
                            if mm2 is None:
                                hit_any = True
                                break
                            if mm2.hp < hp0:
                                hit_any = True
                                hp0 = mm2.hp
                        if _metroid_slot(env, slot) is None:
                            break
                    miss_streak = 0 if hit_any else 1
                continue

            # 6) Blocked by a pillar/step or not on a clean ray: reposition!
            if miss_streak >= 1:
                side = 1 if s.samus_x >= m.x else -1
                if abs(s.samus_x - m.x) < 38:
                    side = -side
                tx = max(32, min(room_x_max, m.x + side * 42))
                ty = max(64, m.y - 6)
                for mf in range(22):
                    mm = _metroid_slot(env, slot)
                    if mm is None or mm.frz < 35:
                        break
                    s2 = session.state
                    if abs(s2.samus_x - tx) <= 12 and abs(s2.samus_y - ty) <= 12:
                        break
                    _fly_step(session, tx, ty, mf, 0, dash=True, label=f"{label}_realign")
                miss_streak = 0
            else:
                tx = _ground_shot_target_x(s.samus_x, s.samus_y, m.x, m.y, room_x_max)
                ty = s.samus_y if (not vertical_order and m.y <= s.samus_y + 12) else (m.y + 4)
                stuck = stuck + 1 if (s.samus_x == prev_x and abs(s.samus_x - tx) > 8) else 0
                prev_x = s.samus_x
                _fly_step(session, tx, ty, f, stuck, dash=True, label=f"{label}_align")
        else:
            raise TimeoutError(f"{label}: failed to kill Metroid slot={slot} in 0x{room_id:04X}")

        # Sweep drops right over the kill spot if we still need Supers, Reserve, or Energy.
        _fast_select(session, 0, label)
        _sweep_drops(session, last_kx, last_ky, label, frames=46)

    _fast_select(session, 0, label)


def play_metroid_room_1(session: ControllerSession) -> SuperMetroidState:
    """Metroid Room 1 (0xDAE1 -> 0xDB31, row 316): 4 Metroids, exit left."""
    label = "metroid_room_1"
    require_room(session, ROOM_METROID_1, label)
    _clear_metroids(
        session,
        room_id=ROOM_METROID_1,
        prefer_supers=True,
        vertical_order=False,
        room_x_max=1485,
        label=label,
    )
    return _fly_to(
        session,
        [(100, 139), (16, 139)],
        target_room=ROOM_METROID_2,
        shoot_door="LEFT",
        max_frames=600,
        label=label,
    )


def play_metroid_room_2(session: ControllerSession) -> SuperMetroidState:
    """Metroid Room 2 (0xDB31 -> 0xDB7D, row 317): 2 Metroids, exit bottom-right."""
    label = "metroid_room_2"
    require_room(session, ROOM_METROID_2, label)
    _clear_metroids(
        session,
        room_id=ROOM_METROID_2,
        prefer_supers=True,
        vertical_order=True,
        room_x_max=224,
        label=label,
    )
    # If on the bottom pit floor right under the door ledge (x > 160, y > 400), step left first.
    for _ in range(45):
        s = session.state
        if s.samus_y <= 400 or s.samus_x <= 155:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_clear_step")
    return _fly_to(
        session,
        [(145, 370), (205, 385), (236, 395)],
        target_room=ROOM_METROID_3,
        shoot_door="RIGHT",
        max_frames=500,
        label=label,
    )


def play_metroid_room_3(session: ControllerSession) -> SuperMetroidState:
    """Metroid Room 3 (0xDB7D -> 0xDBCD, row 318): 3 Metroids, exit right."""
    label = "metroid_room_3"
    require_room(session, ROOM_METROID_3, label)
    _clear_metroids(
        session,
        room_id=ROOM_METROID_3,
        prefer_supers=False,
        vertical_order=False,
        room_x_max=1495,
        label=label,
    )
    return _fly_to(
        session,
        [(1420, 139), (1516, 139)],
        target_room=ROOM_METROID_4,
        shoot_door="RIGHT",
        max_frames=600,
        label=label,
    )


def play_metroid_room_4(session: ControllerSession) -> SuperMetroidState:
    """Metroid Room 4 (0xDBCD -> 0xDC19, row 319): 3 Metroids, exit floor hatch."""
    label = "metroid_room_4"
    require_room(session, ROOM_METROID_4, label)
    _clear_metroids(
        session,
        room_id=ROOM_METROID_4,
        prefer_supers=False,
        vertical_order=True,
        room_x_max=224,
        label=label,
    )
    # Drop off any upper ledge down to the bottom floor (y >= 425) before aligning on the hatch.
    drop_dir = "RIGHT" if session.state.samus_x < 128 else "LEFT"
    prev_x = -1
    wall_f = 0
    for _ in range(300):
        s = session.state
        if s.samus_y >= 425 and s.velocity_y == 0:
            break
        if s.velocity_y == 0 and s.vertical_direction == 0:
            wall_f = wall_f + 1 if s.samus_x == prev_x else 0
            if wall_f >= 5:
                drop_dir = "RIGHT" if drop_dir == "LEFT" else "LEFT"
                wall_f = 0
            prev_x = s.samus_x
            hold(session, 1, drop_dir, "B", reason=f"{label}_hatch_drop")
        else:
            wall_f = 0
            hold(session, 1, reason=f"{label}_hatch_fall")
    _fly_to(session, [(104, 435)], max_frames=200, label=f"{label}_hatch_pos")
    _unmorph_if_needed(session, label)
    for f in range(300):
        s = session.state
        if s.room_id == ROOM_HOPPER:
            return wait_ordinary_room(session, ROOM_HOPPER, settle_frames=320, label=label)
        if s.pose in _MORPH_POSES:
            _unmorph_if_needed(session, label)
            continue
        if s.samus_x < 104:
            hold(session, 1, "RIGHT", reason=f"{label}_hatch_r")
        elif s.samus_x > 118 or (s.pose == 28 and s.vertical_direction == 2 and s.samus_x > 110):
            hold(session, 1, "LEFT", reason=f"{label}_hatch_l")
        elif f % 28 < 6:
            hold(session, 1, "A", reason=f"{label}_hatch_hop")
        elif f % 28 < 10:
            hold(session, 1, "DOWN", reason=f"{label}_hatch_aim")
        elif f % 28 == 10:
            hold(session, 1, "DOWN", "X", reason=f"{label}_hatch_shoot")
        else:
            hold(session, 1, reason=f"{label}_hatch_wait")
    raise TimeoutError(f"{label}: failed to open floor hatch to 0x{ROOM_HOPPER:04X}: {session.state}")


def play_metroid_rooms_all(session: ControllerSession) -> SuperMetroidState:
    """Clear all four Metroid Rooms (rows 316–319: 0xDAE1 -> 0xDC19) restore-free."""
    play_metroid_room_1(session)
    play_metroid_room_2(session)
    play_metroid_room_3(session)
    return play_metroid_room_4(session)


def play_big_boy_room(session: ControllerSession) -> SuperMetroidState:
    """Big Boy Room (0xDCB1 -> 0xDCFF, rows 322–323) restore-free Skill.

    Works from ANY entry energy (unlike the fixed-length human tape whose
    drain duration shifts with entry HP) and cuts ~50s of wandering/idle.
    Enters on the right at (984, 142), flies left to trigger the Baby Metroid
    cutscene, waits until the Baby flies off-screen left (enemy0_x > 60000),
    then flies over the x=309 pillar and x=101 step into Seaweed Room (0xDCFF).
    """
    label = "big_boy_room"
    require_room(session, ROOM_BIG_BOY, label)
    env = session.env
    _fast_select(session, 0, label)

    prev_x = -1
    stuck = 0
    for f in range(4800):
        s = session.state
        if s.room_id == ROOM_SEAWEED:
            return wait_ordinary_room(session, ROOM_SEAWEED, settle_frames=320, label=label)
        if s.game_state != 8:
            hold(session, 1, reason=f"{label}_wait_gs8")
            continue

        baby_x = read_wram_u16(env, 0x0F7A)
        # Baby Metroid flies off-screen left (wraps to ~65454) when cutscene ends!
        if s.health <= 1 and baby_x > 60000 and s.pose != 233:
            # First cross past the mid-room pillar (x=309) onto the left floor (x <= 150).
            if s.samus_x > 165:
                if 290 <= s.samus_x <= 355 and s.samus_y > 140:
                    hold(session, 10, "RIGHT", "B", reason=f"{label}_clear_pillar")
                    # A running spin-jump clears the pillar from its lower ledge.
                    hold(session, 14, "LEFT", "B", "A", reason=f"{label}_takeoff")
                _fly_to(session, [(250, 125), (150, 155)], max_frames=400, label=f"{label}_to_left")
            # Break the shot blocks guarding the left exit staircase (x=101,y=187 -> x=85,y=171 -> x=57,y=139 -> door at x=16,y=108)!
            for ef in range(400):
                s2 = session.state
                if s2.room_id == ROOM_SEAWEED:
                    return wait_ordinary_room(session, ROOM_SEAWEED, settle_frames=320, label=label)
                if s2.samus_x <= 72 and s2.samus_y <= 148:
                    # On the upper door ledge (y<=145): shoot LEFT to open, jump LEFT+A up into (20,108)!
                    if ef % 20 == 2:
                        hold(session, 1, "LEFT", "X", reason=f"{label}_door_shoot")
                    elif ef % 20 == 5:
                        hold(session, 1, "DOWN", "X", reason=f"{label}_door_crouch_shoot")
                    elif ef % 20 < 17:
                        hold(session, 1, "LEFT", "A", reason=f"{label}_door_jump")
                    else:
                        hold(session, 1, "LEFT", reason=f"{label}_door_enter")
                elif s2.samus_x > 94:
                    # Approach (101, 187), shoot LEFT+R to clear the first shot block, then hop to (85, 171)!
                    if s2.samus_x > 106:
                        hold(session, 1, "LEFT", "B", reason=f"{label}_to_step1")
                    elif ef % 24 < 6:
                        hold(session, 1, "LEFT", "R", "X", reason=f"{label}_break_blk1")
                    elif ef % 24 < 18:
                        hold(session, 1, "LEFT", "A", reason=f"{label}_hop_step1")
                    else:
                        hold(session, 1, "LEFT", reason=f"{label}_land_step1")
                elif s2.samus_y > 148:
                    # On (85, 171): shoot UP to clear the overhead shot block, then hop left onto (57, 139)!
                    if ef % 26 < 6:
                        hold(session, 1, "UP", "X", reason=f"{label}_break_blk2")
                    elif ef % 26 < 18:
                        hold(session, 1, "LEFT", "A", reason=f"{label}_hop_step2")
                    else:
                        hold(session, 1, "LEFT", reason=f"{label}_land_step2")
                else:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_run_door")
            raise TimeoutError(f"{label}: failed to enter doorway corridor: {session.state}")

        stuck = stuck + 1 if s.samus_x == prev_x else 0
        prev_x = s.samus_x
        if s.pose == 233 or (s.health <= 1 and baby_x <= 60000):
            # Cutscene active (draining / Baby realizing / flying away): idle cleanly.
            hold(session, 1, reason=f"{label}_cutscene")
        elif stuck >= 8:
            # Hop cleanly over the x=821 ledge to trigger Baby Metroid immediately!
            if (f % 20) < 10:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_unstick_r") if stuck < 14 else hold(session, 1, "LEFT", "B", "A", reason=f"{label}_hop_step")
            else:
                hold(session, 1, "LEFT", "B", reason=f"{label}_hop_air")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_run")
    raise TimeoutError(f"{label}: failed to reach Seaweed Room 0xDCFF: {session.state}")


def play_rinka_shaft_direct(session: ControllerSession) -> SuperMetroidState:
    """Rinka Shaft (0xDDF3 -> 0xDD58, rows 328–330) direct drop skipping Save Room.

    Enters top-left at (39, 139), drops down the shaft to the bottom-left red
    door (x<=76, y=651), opens it with 1 Super Missile, and enters Mother Brain's
    Room (0xDD58).
    """
    label = "rinka_shaft_direct"
    require_room(session, ROOM_RINKA_SHAFT, label)

    # Drop down Rinka Shaft using the exact ledge gaps:
    # (39,139) -> RIGHT to x>=70 (y=171) -> LEFT to x<=48 (falls to y=395)
    # -> RIGHT to x>=82 (falls straight to bottom floor y=651).
    for _ in range(360):
        s = session.state
        if s.samus_y >= 635 and s.velocity_y == 0:
            break
        if s.samus_y < 165:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_drop_top")
        elif s.samus_y < 380:
            if s.samus_x > 48:
                hold(session, 1, "LEFT", "B", reason=f"{label}_drop_upper_l")
            else:
                hold(session, 1, reason=f"{label}_drop_upper_fall")
        elif s.samus_y < 430:
            if s.samus_x < 78:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_drop_lower_r")
            elif s.samus_x > 88:
                hold(session, 1, "LEFT", "B", reason=f"{label}_drop_lower_l")
            else:
                hold(session, 1, reason=f"{label}_drop_lower_fall")
        elif s.samus_y < 520:
            if s.samus_x > 84:
                hold(session, 1, "LEFT", "B", reason=f"{label}_drop_mid_l")
            else:
                hold(session, 1, reason=f"{label}_drop_mid_fall")
        else:
            if s.samus_x > 74:
                hold(session, 1, "LEFT", "B", reason=f"{label}_drop_bot_l")
            else:
                hold(session, 1, reason=f"{label}_drop_bot")
    else:
        raise TimeoutError(f"{label}: failed to reach bottom of Rinka Shaft: {session.state}")

    # Move onto the bottom-left red door pad (x <= 76) and open with 1 Super.
    _fast_select(session, 2, label)
    for _ in range(80):
        s = session.state
        if s.samus_x <= 76:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_align_door")
    hold(session, 4, "LEFT", reason=f"{label}_face_door")
    hold(session, 1, "LEFT", "X", reason=f"{label}_super_door")
    _fast_select(session, 0, label)

    for f in range(240):
        s = session.state
        if s.room_id == ROOM_MOTHER_BRAIN:
            return wait_ordinary_room(session, ROOM_MOTHER_BRAIN, settle_frames=320, label=label)
        if f % 12 == 8:
            hold(session, 1, "LEFT", "B", "X", reason=f"{label}_enter_shot")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_enter")
    raise TimeoutError(f"{label}: failed to enter Mother Brain Room 0xDD58: {session.state}")


__all__ = [
    "play_big_boy_room",
    "play_metroid_room_1",
    "play_metroid_room_2",
    "play_metroid_room_3",
    "play_metroid_room_4",
    "play_metroid_rooms_all",
    "play_rinka_shaft_direct",
]
