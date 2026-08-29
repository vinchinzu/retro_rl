"""Continuous L4 residual: natural-key PostLadder → map → Gleeok → TF 0x08.

rr-05fz assisted dual-green; **rr-vdnc Clean dual-green** (no ``--infinite-life``).
Skip-compass route leaves keys≥1 so KEY-UP needs **no recon poke**.
Default start ``Level4Room31PostLadderNaturalKey`` (ladder=1, keys≥1).

Phases (live IDs only)::

    map_21 no-poke → Level4Map
    BOMB_UP 0x21 → 0x11
    RIGHT → 0x12 clear Vires → push 0x68 LEFT → PATH_12_TO_GLEEOK → 0x13
    Gleeok south-stand melee → HC → UP 0x03 → tf&0x08

Not full-game Clean STATUS. Examples::

    # Assisted first-pass
    uv run python nes/zelda_i/scripts/run_level4_continuous_tf.py \\
        --infinite-life --trials 2 --save-state --tag l4_05fz_cont_tf

    # Clean dual (rr-vdnc)
    uv run python nes/zelda_i/scripts/run_level4_continuous_tf.py \\
        --trials 2 --tag l4_vdnc_clean_cont_tf

    # Map-only residual (already dual-green natural key):
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment map_21 \\
        --from-state Level4Room31PostLadderNaturalKey \\
        --infinite-life --no-key-poke --trials 2
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Any

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import ensure_bomb, idle, room_fields
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level4_boss_combat import Level4GleeokFightController, level4_tf08
from zelda_i.level4_dungeon import (
    BOMB_21_NORTH_FACE,
    BOMB_21_NORTH_STAND,
    GEL_OBJECT_TYPE,
    ROOM_L4_GLEEOK_13,
    ROOM_L4_MAP_21,
    ROOM_L4_MID_11,
    ROOM_L4_VIRES_12,
    level4_gleeok_enter_success,
    level4_map_success,
    level4_room_12_cleared,
)
from zelda_i.level4_maze_path import PATH_12_TO_GLEEOK
from zelda_i.level4_overworld import LEVEL4
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts import run_level4_rooms as r4


def _ensure_bomb_selected(env) -> None:
    ensure_bomb(env)

def _thrash_room(env, assist, total: list[int], room: int, max_frames: int = 10000) -> dict:
    patrol = tuple(
        (x, y)
        for y in (93, 109, 125, 141, 157, 173, 189)
        for x in (40, 72, 104, 136, 168, 200)
    )
    spec = DungeonRoomSpec(
        spec_id=f"l4_cont_thrash_{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("LEFT", ((16, 141), (48, 141))),
        enemy_types=(GEL_OBJECT_TYPE, 0x12, 0x1C, 0x1B, 0x13, 0x14, 0x17, 0x35),
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=56,
            attack_phase=4,
            engage_attack_period=4,
            engage_attack_hold=2,
            patrol_attack_period=8,
            patrol_attack_hold=2,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=LEVEL4,
    )
    ctrl = GenericDungeonRoomController(spec)
    ctrl.phase = DungeonPhase.FIGHT
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death"}
        if snap.screen != room:
            env.step(nes_action("LEFT" if snap.link_x > 120 else "RIGHT"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            continue
        fa = ctrl.step(snap)
        env.step(fa.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        if ctrl.success:
            break
    return {"ok": True, "frames": ctrl.frames, "success": bool(ctrl.success)}

def _bfs_to_xy(env, sx: int, sy: int, *, hold: int = 4, quant: int = 4) -> bool:
    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    start = s0.screen
    if abs(s0.link_x - sx) + abs(s0.link_y - sy) <= 8:
        return True
    st0 = em.get_state()
    start_c = (s0.link_x // quant * quant, s0.link_y // quant * quant)
    cs = {start_c: st0}
    parent: dict = {start_c: None}
    q: deque = deque([start_c])
    seen = {start_c}
    found = None
    exp = 0
    while q and exp < 12000 and found is None:
        cur = q.popleft()
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            exp += 1
            em.set_state(cs[cur])
            for _ in range(hold):
                env.step(nes_action(d))
            s = read_snapshot(env.get_ram())
            if s.screen != start or s.mode not in (PLAY_MODE, 5):
                continue
            nc = (s.link_x // quant * quant, s.link_y // quant * quant)
            if nc in seen:
                continue
            if abs(s.link_x - cur[0]) + abs(s.link_y - cur[1]) < 2:
                continue
            seen.add(nc)
            cs[nc] = em.get_state()
            parent[nc] = (cur, d)
            q.append(nc)
            if abs(s.link_x - sx) + abs(s.link_y - sy) <= 8:
                found = nc
                break
    if found is None:
        em.set_state(st0)
        return False
    em.set_state(cs[found])
    return True

def _bomb_north_21(env, assist, total: list[int]) -> dict[str, Any]:
    """BFS to bomb stand (thrash only if needed) → bomb UP → 0x11.

    Clean residual (rr-zavx): map pickup already expands much of the maze;
    thrash-first burns hearts and continuous Gleeok needs enter health ≥~107
    (rr-gjey lab). Gel thrash does not reliably drop hearts.
    """
    sx, sy = BOMB_21_NORTH_STAND
    thr: dict[str, Any] | None = None
    ok_stand = _bfs_to_xy(env, sx, sy)
    if not ok_stand:
        thr = _thrash_room(env, assist, total, ROOM_L4_MAP_21, max_frames=5000)
        ok_stand = _bfs_to_xy(env, sx, sy)
    if not ok_stand:
        # second thrash expand + retry
        thr2 = _thrash_room(env, assist, total, ROOM_L4_MAP_21, max_frames=5000)
        thr = {"first": thr, "second": thr2}
        ok_stand = _bfs_to_xy(env, sx, sy)
    if not ok_stand:
        return {"ok": False, "error": "bomb_stand_unreachable", "thrash": thr}
    # fine align
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - sx) <= 4 and abs(snap.link_y - sy) <= 4:
            break
        if abs(snap.link_x - sx) > 4:
            d = "RIGHT" if snap.link_x < sx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < sy else "UP"
        env.step(nes_action(d))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    _ensure_bomb_selected(env)
    face = BOMB_21_NORTH_FACE
    bombs0 = read_snapshot(env.get_ram()).bombs
    for _ in range(6):
        env.step(nes_action(face))
        total[0] += 1
    env.step(nes_action(face, "B"))
    total[0] += 1
    for _ in range(100):
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    start_sc = ROOM_L4_MAP_21
    for _ in range(320):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start_sc and snap.mode in (PLAY_MODE, 4, 6, 7):
            break
        if snap.transitioning or snap.mode in (4, 6, 7):
            env.step(nes_action(face))
        else:
            env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning:
            break
        env.step(nes_idle_action())
        total[0] += 1
    snap = read_snapshot(env.get_ram())
    return {
        "ok": snap.screen == ROOM_L4_MID_11,
        "dest": f"0x{snap.screen:02x}",
        "bombs0": bombs0,
        "bombs1": snap.bombs,
        "thrash": thr,
        "xy": [snap.link_x, snap.link_y],
    }

def _follow(env, path: list[str] | tuple[str, ...], hold: int, assist, total, dest: int | None = None) -> bool:
    for d in path:
        for _ in range(hold):
            snap = read_snapshot(env.get_ram())
            if dest is not None and snap.screen == dest and snap.mode == PLAY_MODE and not snap.transitioning:
                return True
            env.step(nes_action(d))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            snap = read_snapshot(env.get_ram())
            if snap.transitioning or snap.mode in (4, 6, 7, 16):
                for _ in range(40):
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
    for _ in range(50):
        snap = read_snapshot(env.get_ram())
        if dest is None or (snap.screen == dest and snap.mode == PLAY_MODE and not snap.transitioning):
            if dest is None or snap.screen == dest:
                return dest is None or snap.screen == dest
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    return dest is None or snap.screen == dest

def _bfs_exit(env, dest: int, assist, total, hold: int = 4, max_exp: int = 12000) -> list[str] | None:
    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())

    def cell(x: int, y: int) -> tuple[int, int]:
        return (x // 4 * 4, y // 4 * 4)

    st = cell(s0.link_x, s0.link_y)
    cs = {st: em.get_state()}
    parent: dict = {st: None}
    q: deque = deque([st])
    seen = {st}
    exp = 0
    while q and exp < max_exp:
        cur = q.popleft()
        for d in ("RIGHT", "UP", "DOWN", "LEFT"):
            exp += 1
            em.set_state(cs[cur])
            for _ in range(hold):
                env.step(nes_action(d))
            s2 = read_snapshot(env.get_ram())
            if s2.transitioning or s2.mode in (4, 6, 7, 16):
                for _ in range(350):
                    env.step(nes_idle_action())
                s2 = read_snapshot(env.get_ram())
            if s2.screen == dest and s2.mode == PLAY_MODE:
                path: list[str] = []
                n = cur
                while n is not None and parent[n] is not None:
                    pp, pd = parent[n]
                    path.append(pd)
                    n = pp
                path.reverse()
                path.append(d)
                em.set_state(cs[st])
                return path
            if s2.screen != s0.screen or s2.mode != PLAY_MODE:
                continue
            nc = cell(s2.link_x, s2.link_y)
            if nc in seen:
                continue
            if abs(s2.link_x - cur[0]) + abs(s2.link_y - cur[1]) < 2:
                continue
            seen.add(nc)
            cs[nc] = em.get_state()
            parent[nc] = (cur, d)
            q.append(nc)
    em.set_state(cs[st])
    return None

def _push_block_12(env, assist, total) -> dict[str, Any]:
    """Push block 0x68 LEFT on 0x12: stand~(112,144) via (96,144)→(80,144)."""
    # Walk to stand near block
    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L4_VIRES_12:
            return {"ok": False, "error": f"left_12_0x{snap.screen:02x}"}
        tx, ty = 112, 144
        if abs(snap.link_x - tx) <= 4 and abs(snap.link_y - ty) <= 4:
            break
        if abs(snap.link_x - tx) > 4:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < ty else "UP"
        env.step(nes_action(d))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    # Approach from east of block then hold LEFT
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if snap.link_x > 100:
            env.step(nes_action("LEFT"))
        elif abs(snap.link_y - 144) > 4:
            env.step(nes_action("DOWN" if snap.link_y < 144 else "UP"))
        else:
            break
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    doors0 = read_snapshot(env.get_ram()).cur_opened_doors
    for _ in range(180):
        env.step(nes_action("LEFT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        snap = read_snapshot(env.get_ram())
        if (snap.cur_opened_doors & 0x01) or snap.cur_opened_doors > doors0:
            return {
                "ok": True,
                "doors": snap.cur_opened_doors,
                "xy": [snap.link_x, snap.link_y],
            }
    snap = read_snapshot(env.get_ram())
    # Accept R bit or doors raw>=3
    ok = bool(snap.cur_opened_doors & 0x01) or snap.cur_opened_doors >= 3
    return {
        "ok": ok,
        "doors": snap.cur_opened_doors,
        "xy": [snap.link_x, snap.link_y],
        "error": None if ok else "push_no_right_bit",
    }

def run_from_map_to_tf(
    env,
    assist,
    total: list[int],
    *,
    tag: str,
    trial_i: int,
) -> dict[str, Any]:
    """Level4Map play-ready → Gleeok → TF."""
    report: dict[str, Any] = {"ok": False, "phases": {}}
    snap = read_snapshot(env.get_ram())
    if not level4_map_success(env.get_ram()):
        report["error"] = "need_map"
        return report
    if snap.screen != ROOM_L4_MAP_21:
        # try BFS/enter map room not required if already have map bit
        path = _bfs_exit(env, ROOM_L4_MAP_21, assist, total)
        if path:
            _follow(env, path, 4, assist, total, ROOM_L4_MAP_21)
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L4_MAP_21:
            report["error"] = f"not_on_map_room_0x{snap.screen:02x}"
            return report

    # BOMB_UP 0x21 → 0x11 (maze: thrash + BFS to stand first)
    bomb = _bomb_north_21(env, assist, total)
    report["phases"]["bomb21"] = bomb
    if not bomb.get("ok"):
        report["error"] = bomb.get("error") or f"bomb21_dest_{bomb.get('dest')}"
        return report
    idle(env, assist, total, 30)
    snap = read_snapshot(env.get_ram())
    if snap.screen != ROOM_L4_MID_11:
        report["error"] = f"bomb21_dest_0x{snap.screen:02x}"
        return report

    # BOMB_RIGHT 0x11 → 0x12 (live stand ~(192,141); not free door).
    # Prefer BFS before thrash to preserve Clean hearts (rr-zavx).
    sx12, sy12 = 192, 141
    thr11: dict[str, Any] | None = None
    ok_stand = _bfs_to_xy(env, sx12, sy12)
    if not ok_stand:
        thr11 = _thrash_room(env, assist, total, ROOM_L4_MID_11, max_frames=6000)
        ok_stand = _bfs_to_xy(env, sx12, sy12)
    if not ok_stand:
        thr11 = {
            "a": thr11,
            "b": _thrash_room(env, assist, total, ROOM_L4_MID_11, max_frames=6000),
        }
        ok_stand = _bfs_to_xy(env, sx12, sy12)
    if not ok_stand:
        report["error"] = "bomb12_stand_unreachable"
        report["phases"]["to_12"] = {"thrash": thr11}
        return report
    for _ in range(60):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - sx12) <= 4 and abs(snap.link_y - sy12) <= 4:
            break
        if abs(snap.link_x - sx12) > 4:
            d = "RIGHT" if snap.link_x < sx12 else "LEFT"
        else:
            d = "DOWN" if snap.link_y < sy12 else "UP"
        env.step(nes_action(d))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    _ensure_bomb_selected(env)
    face = "RIGHT"
    for _ in range(6):
        env.step(nes_action(face))
        total[0] += 1
    env.step(nes_action(face, "B"))
    total[0] += 1
    for _ in range(100):
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(320):
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L4_VIRES_12:
            break
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(60):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning:
            break
        env.step(nes_idle_action())
        total[0] += 1
    snap = read_snapshot(env.get_ram())
    report["phases"]["to_12"] = {
        "ok": snap.screen == ROOM_L4_VIRES_12,
        "dest": f"0x{snap.screen:02x}",
        "thrash": thr11,
        "stand": [sx12, sy12],
    }
    if snap.screen != ROOM_L4_VIRES_12:
        report["error"] = f"no_0x12_got_0x{snap.screen:02x}"
        return report
    idle(env, assist, total, 20)

    # Clear 0x12 Vires (TYPE presence; ignore block 0x68). Cap thrash so
    # Clean hearts survive for Gleeok (south-stand needs ~108+ health).
    idle(env, assist, total, 40)
    thr12 = _thrash_room(env, assist, total, ROOM_L4_VIRES_12, max_frames=12000)
    # require no type 0x12/0x1c for a settle window
    clear_ok = False
    dead_streak = 0
    for _ in range(200):
        snap = read_snapshot(env.get_ram())
        live = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id in (0x12, 0x1C)
        ]
        if live:
            dead_streak = 0
            # keep thrashing nearest; brief backstep if overlapping
            tgt = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(tgt.x - snap.link_x) + abs(tgt.y - snap.link_y)
            dx, dy = tgt.x - snap.link_x, tgt.y - snap.link_y
            if dist < 12 and assist is None:
                # Clean: step away one beat then slash (contact damage piles).
                d = (
                    ("LEFT" if dx > 0 else "RIGHT")
                    if abs(dx) >= abs(dy)
                    else ("UP" if dy > 0 else "DOWN")
                )
                env.step(nes_action(d))
            else:
                d = (
                    ("RIGHT" if dx > 0 else "LEFT")
                    if abs(dx) >= abs(dy)
                    else ("DOWN" if dy > 0 else "UP")
                )
                env.step(nes_action(d, "A"))
        else:
            dead_streak += 1
            env.step(nes_idle_action())
            if dead_streak >= 25:
                clear_ok = True
                break
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    report["phases"]["clear12"] = {
        "ok": clear_ok,
        "thrash": thr12,
        "dead_streak": dead_streak,
        "predicate": level4_room_12_cleared(env.get_ram()),
        "health_after": read_snapshot(env.get_ram()).health,
    }
    if not clear_ok:
        report["error"] = "clear12_failed"
        return report
    idle(env, assist, total, 30)

    # Push block 0x68 LEFT: stand east of block ~(112,144), hold LEFT → (80,144)
    # Do not overshoot west door (x→0).
    for _ in range(500):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L4_VIRES_12:
            report["error"] = f"left_12_during_push_0x{snap.screen:02x}"
            return report
        tx, ty = 112, 144
        if abs(snap.link_x - tx) <= 3 and abs(snap.link_y - ty) <= 3:
            break
        if abs(snap.link_y - ty) > 3:
            d = "DOWN" if snap.link_y < ty else "UP"
        else:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        env.step(nes_action(d))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    doors0 = read_snapshot(env.get_ram()).cur_opened_doors
    for _ in range(90):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L4_VIRES_12 or snap.link_x < 40:
            break
        env.step(nes_action("LEFT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        if snap.cur_opened_doors & 0x01:
            break
    # if still L-only, try from (96,144)
    snap = read_snapshot(env.get_ram())
    if not (snap.cur_opened_doors & 0x01):
        for _ in range(200):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_x - 96) <= 3 and abs(snap.link_y - 144) <= 3:
                break
            if abs(snap.link_y - 144) > 3:
                d = "DOWN" if snap.link_y < 144 else "UP"
            else:
                d = "RIGHT" if snap.link_x < 96 else "LEFT"
            env.step(nes_action(d))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
        for _ in range(100):
            snap = read_snapshot(env.get_ram())
            if snap.link_x < 48:
                break
            env.step(nes_action("LEFT"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            if snap.cur_opened_doors & 0x01:
                break
    snap = read_snapshot(env.get_ram())
    push = {
        "ok": bool(snap.cur_opened_doors & 0x01) or snap.cur_opened_doors >= 3,
        "doors": snap.cur_opened_doors,
        "doors0": doors0,
        "xy": [snap.link_x, snap.link_y],
    }
    report["phases"]["push12"] = push
    if not push.get("ok"):
        report["error"] = "push12_failed"
        return report
    idle(env, assist, total, 15)

    # Nudge off west door before maze path (push often ends west-of-block).
    for _ in range(40):
        snap = read_snapshot(env.get_ram())
        if snap.link_x >= 96:
            break
        env.step(nes_action("RIGHT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    # Align mid door band y≈141 — PATH + free RIGHT scroll need center band
    # (post-clear thrash can leave y≈149 and stick on east wall, rr-zavx).
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_y - 141) <= 4:
            break
        env.step(nes_action("DOWN" if snap.link_y < 141 else "UP"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    # PATH_12_TO_GLEEOK hold4
    path_ok = _follow(
        env, list(PATH_12_TO_GLEEOK), 4, assist, total, ROOM_L4_GLEEOK_13
    )
    bfs_path = None
    if not path_ok:
        # settle scroll / retry hold RIGHT at y141
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if abs(snap.link_y - 141) > 4:
                env.step(nes_action("DOWN" if snap.link_y < 141 else "UP"))
            else:
                env.step(nes_action("RIGHT"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_GLEEOK_13 and snap.mode == PLAY_MODE:
                path_ok = True
                break
    if not path_ok and not level4_gleeok_enter_success(env.get_ram()):
        # Live BFS exit 0x12 → 0x13 (maze; pose-stable after push).
        bfs_path = _bfs_exit(env, ROOM_L4_GLEEOK_13, assist, total, hold=4)
        if bfs_path:
            path_ok = _follow(
                env, bfs_path, 4, assist, total, ROOM_L4_GLEEOK_13
            )
        if not path_ok:
            for _ in range(200):
                snap = read_snapshot(env.get_ram())
                if snap.screen == ROOM_L4_GLEEOK_13 and snap.mode == PLAY_MODE:
                    path_ok = True
                    break
                if abs(snap.link_y - 141) > 4:
                    env.step(nes_action("DOWN" if snap.link_y < 141 else "UP"))
                else:
                    env.step(nes_action("RIGHT"))
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
    # Scroll settle only — do not idle 40f on Gleeok vestibule (fireballs
    # fill the room and double approach cost; rr-gjey Clean health floor).
    for _ in range(12):
        snap = read_snapshot(env.get_ram())
        if (
            snap.screen == ROOM_L4_GLEEOK_13
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            break
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    report["phases"]["enter_gleeok"] = {
        "ok": level4_gleeok_enter_success(env.get_ram()),
        "path_ok": path_ok,
        "bfs_path_len": len(bfs_path) if bfs_path else None,
        "final": room_fields(read_snapshot(env.get_ram()), env.get_ram()),
        "health": read_snapshot(env.get_ram()).health,
    }
    if not level4_gleeok_enter_success(env.get_ram()):
        snap = read_snapshot(env.get_ram())
        report["error"] = f"gleeok_enter_0x{snap.screen:02x}"
        return report

    fight = Level4GleeokFightController(tag=f"{tag}_t{trial_i}")
    fr = fight.run(env, assist, total)
    report["phases"]["fight"] = {
        "ok": fr.get("ok"),
        "tf08": fr.get("tf08"),
        "frames": fr.get("frames"),
        "error": fr.get("error"),
        "notes": fr.get("notes"),
        "final": fr.get("final"),
    }
    report["ok"] = bool(fr.get("ok") and (fr.get("tf08") or level4_tf08(env.get_ram())))
    report["tf08"] = bool(level4_tf08(env.get_ram()))
    report["final"] = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    return report

def run_once(
    *,
    start_state: str,
    infinite_life: bool,
    save_checkpoint: bool,
    tag: str,
    trial_i: int,
    from_map: bool,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "bead": "rr-vdnc" if not infinite_life else "rr-05fz",
        "start_state": start_state,
        "track": "assisted" if infinite_life else "clean",
        "trial": trial_i,
        "tag": tag,
        "natural_entry": False,
        "key_poke": False,
    }
    # Phase 1: map from natural-key post-ladder (rooms runner; no poke).
    # Compose dual-env: map segment dual already proven; then map→TF dual.
    map_state = start_state
    if not from_map:
        # Save Level4Map every trial so dual gleeok phase reloads *this*
        # trial's map (not a stale prior checkpoint).
        map_rep = r4.run_once(
            segment="map_21",
            start_state=start_state,
            infinite_life=infinite_life,
            save_checkpoint=save_checkpoint,
            tag=f"{tag}_map_t{trial_i}",
            allow_key_poke=False,
        )
        report["map"] = {
            "ok": map_rep.get("ok"),
            "frames": map_rep.get("frames"),
            "error": map_rep.get("error"),
            "key": (map_rep.get("controllers") or {}).get("map_21_key"),
            "final": map_rep.get("final"),
            "checkpoint": map_rep.get("checkpoint"),
        }
        if not map_rep.get("ok"):
            report["error"] = f"map_failed:{map_rep.get('error')}"
            return report
        poke = (map_rep.get("controllers") or {}).get("map_21_key") or {}
        if poke.get("recon_poke"):
            report["error"] = "unexpected_key_poke"
            report["key_poke"] = True
            return report
        map_state = "Level4Map"

    env = make_env(GAME, map_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    try:
        env.reset()
        idle(env, assist, total, 10)
        if int(read_u8(env.get_ram(), ADDR_LADDER)) <= 0:
            report["error"] = "no_ladder"
            return report
        g = run_from_map_to_tf(env, assist, total, tag=tag, trial_i=trial_i)
        report["gleeok_path"] = g
        report["ok"] = bool(g.get("ok") and g.get("tf08"))
        report["tf08"] = bool(g.get("tf08") or level4_tf08(env.get_ram()))
        report["total_frames"] = total[0] + int((report.get("map") or {}).get("frames") or 0)
        report["final"] = g.get("final") or room_fields(
            read_snapshot(env.get_ram()), env.get_ram()
        )
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final.png")
        if save_checkpoint and report["ok"] and trial_i == 0:
            path = save_state(env, GAME_DIR, GAME, "Level4Complete")
            report["checkpoint"] = str(path)
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{map_state}.state"
                ),
                request={
                    "bead": report["bead"],
                    "segment": "continuous_map_gleeok_tf",
                    "track": report["track"],
                    "natural_entry": False,
                    "key_poke": False,
                },
                selected_trial={
                    "ok": True,
                    "tf08": True,
                    "final": report["final"],
                },
                natural_entry=False,
            )
        return report
    finally:
        env.close()

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-state",
        default="Level4Room31PostLadderNaturalKey",
        help="Natural-key post-ladder (default) or Level4Map with --from-map",
    )
    p.add_argument(
        "--from-map",
        action="store_true",
        help="Start at Level4Map (skip map_21; still continuous map→TF)",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l4_05fz_cont_tf")
    args = p.parse_args()

    trials: list[dict[str, Any]] = []
    for i in range(args.trials):
        print(f"=== trial {i} ===", flush=True)
        r = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoint=args.save_state,
            tag=args.tag,
            trial_i=i,
            from_map=args.from_map,
        )
        print(
            "RESULT",
            {
                "ok": r.get("ok"),
                "tf08": r.get("tf08"),
                "error": r.get("error")
                or (r.get("gleeok_path") or {}).get("error"),
                "map_ok": (r.get("map") or {}).get("ok"),
                "frames": r.get("total_frames"),
                "key_poke": r.get("key_poke"),
            },
            flush=True,
        )
        trials.append(r)

    dual = all(t.get("ok") and t.get("tf08") for t in trials) and len(trials) >= 2
    out = {
        "bead": "rr-vdnc" if not args.infinite_life else "rr-05fz",
        "segment": "continuous_natural_key_map_gleeok_tf",
        "from": args.from_state,
        "from_map": args.from_map,
        "dual_green": dual,
        "ok": dual or (len(trials) == 1 and trials[0].get("ok")),
        "track": "assisted" if args.infinite_life else "clean",
        "key_poke": any(t.get("key_poke") for t in trials),
        "trials": trials,
        "tag": args.tag,
    }
    path = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(path, out)
    print(f"wrote {path} dual={dual} ok={out['ok']}", flush=True)
    return 0 if out["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
