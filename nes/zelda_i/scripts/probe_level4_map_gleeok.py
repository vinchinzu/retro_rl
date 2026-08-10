"""Live recon from Level4Map (0x21) toward Gleeok / TF 0x08 (rr-rvae).

Assisted first-pass. 0x21 is a dark maze pocket after map pickup — naive door
pushes fail. Uses gel thrash + state-saving BFS to discover free exits, then
expands west into 0x20 and north into 0x10 (Manhandla side path). Dense bomb
stands on each visited room. Documents LIVE edges; not Clean STATUS.

Examples::

    uv run python nes/zelda_i/scripts/probe_level4_map_gleeok.py \\
        --infinite-life --from-state Level4Map --tag l4_rvae_gleeok
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level4_dungeon import (
    GEL_OBJECT_TYPE,
    LEVEL4_MAP_BIT,
    ROOM_L4_MAP_21,
    ROOM_L4_WATER_NORTH_20,
)
from zelda_i.level4_overworld import LEVEL4
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_COMPASS,
    ADDR_LADDER,
    ADDR_MAP,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

DOOR_TARGETS = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 96),
    "DOWN": (120, 189),
}

BOMB_STANDS: list[tuple[str, int, int]] = [
    ("UP", 120, 105),
    ("UP", 96, 105),
    ("UP", 144, 105),
    ("UP", 80, 105),
    ("UP", 160, 105),
    ("DOWN", 120, 189),
    ("DOWN", 96, 189),
    ("DOWN", 144, 189),
    ("LEFT", 48, 141),
    ("LEFT", 48, 125),
    ("LEFT", 48, 157),
    ("RIGHT", 192, 141),
    ("RIGHT", 192, 125),
    ("RIGHT", 192, 157),
    ("UP", 120, 117),
    ("LEFT", 64, 141),
    ("RIGHT", 176, 141),
]

PUSH_CENTERS = [
    (120, 141),
    (96, 141),
    (144, 141),
    (80, 141),
    (160, 141),
    (112, 141),
    (128, 141),
    (120, 125),
    (120, 157),
]

# Known source-hypothesized destinations relative to map branch.
HYP_DESTS = {
    0x21: [0x20, 0x11, 0x22, 0x31, 0x10],
    0x20: [0x10, 0x21, 0x30, 0x00, 0x11],
    0x10: [0x00, 0x11, 0x20, 0x01],
}


def _door_bits(raw: int) -> dict:
    return {
        "R": bool(raw & DOOR_RIGHT),
        "L": bool(raw & DOOR_LEFT),
        "D": bool(raw & DOOR_DOWN),
        "U": bool(raw & DOOR_UP),
        "raw": int(raw),
    }


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12):
            continue
        if o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
            }
        )
    return out


def _sample(snap: ZeldaSnapshot, ram, *, event: str = "sample") -> dict:
    objs = _objs(snap)
    types = Counter(o["type"] for o in objs)
    return {
        "event": event,
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "health": int(snap.health),
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "cur_opened_doors": snap.cur_opened_doors,
        "doors": _door_bits(snap.cur_opened_doors),
        "open_doorway_mask": snap.open_doorway_mask,
        "objects": objs,
        "type_counts": {f"0x{k:02x}": v for k, v in types.items()},
        "type_names": {f"0x{k:02x}": object_name(k) for k in types},
        "inv": {
            "ladder": int(read_u8(ram, ADDR_LADDER)),
            "map": int(read_u8(ram, ADDR_MAP)),
            "map_l4": bool(int(read_u8(ram, ADDR_MAP)) & LEVEL4_MAP_BIT),
            "compass": int(read_u8(ram, ADDR_COMPASS)),
            "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
            "tf_l4": bool(int(read_u8(ram, ADDR_TRIFORCE)) & 0x08),
        },
    }


def _ensure_bomb_selected(env) -> None:
    ram = env.get_ram()
    if read_u8(ram, ADDR_SELECTED_ITEM) == B_ITEM_BOMB:
        return
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "set_byte"):
            mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            return
    except Exception:
        pass
    try:
        env.unwrapped.data.set_value("selected_item", B_ITEM_BOMB)
    except Exception:
        pass


def _poke_bombs(env, bombs: int = 8) -> list[str]:
    notes = []
    try:
        env.unwrapped.data.set_value("bombs", int(bombs) & 0xFF)
        notes.append(f"bombs={bombs}")
    except Exception as exc:
        notes.append(f"bomb_poke_fail={exc!r}")
    _ensure_bomb_selected(env)
    return notes


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _goto_xy(snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 6):
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def _push_door(snap: ZeldaSnapshot, direction: str):
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - ty) > 4:
            return nes_action("DOWN" if snap.link_y < ty else "UP")
        return nes_action(direction)
    if abs(snap.link_x - tx) > 6:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT")
    return nes_action(direction)


def _settle_play(env, *, max_f: int = 400) -> ZeldaSnapshot:
    for _ in range(max_f):
        env.step(nes_idle_action())
        s = read_snapshot(env.get_ram())
        if s.mode in (PLAY_MODE, 5) and not s.transitioning:
            return s
    return read_snapshot(env.get_ram())


def _thrash_gels(env, assist, *, room: int, max_frames: int = 12000) -> dict:
    """Partial gel thrash expands maze walls (same idea as map_21)."""
    patrol = tuple(
        (x, y)
        for y in (93, 109, 125, 141, 157, 173, 189)
        for x in (40, 72, 104, 136, 168, 200)
    )
    spec = DungeonRoomSpec(
        spec_id=f"l4_probe_gels_{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("LEFT", ((16, 141), (48, 141))),
        enemy_types=(GEL_OBJECT_TYPE, 0x12, 0x1C, 0x1B, 0x13, 0x14, 0x17),
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
    for f in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if assist is not None:
            assist.apply_env(env, frame=f)
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": f}
        if snap.screen != room:
            env.step(nes_action("LEFT" if snap.link_x > 120 else "RIGHT"))
            continue
        if ctrl.success:
            break
        fa = ctrl.step(snap)
        env.step(fa.action)
    snap = read_snapshot(env.get_ram())
    return {
        "ok": bool(ctrl.success),
        "frames": ctrl.frames,
        "success": bool(ctrl.success),
        "final": _sample(snap, env.get_ram(), event="post_thrash"),
    }


def _bfs_discover_exits(
    env,
    *,
    hold: int = 4,
    max_exp: int = 20000,
    quant: int = 4,
) -> dict:
    """State-saving BFS: find free room transitions from current play room."""
    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    start = s0.screen
    if s0.mode not in (PLAY_MODE, 5) or s0.level != LEVEL4:
        return {
            "ok": False,
            "error": f"bad_start mode={s0.mode} lv={s0.level} sc=0x{s0.screen:02x}",
            "exits": {},
            "cells": 0,
        }

    st0 = em.get_state()
    start_c = (s0.link_x // quant * quant, s0.link_y // quant * quant)
    cell_state = {start_c: st0}
    parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        start_c: None
    }
    q: deque[tuple[int, int]] = deque([start_c])
    seen = {start_c}
    exits: dict[str, dict] = {}  # dest_hex -> meta
    exp = 0
    bbox = [s0.link_x, s0.link_y, s0.link_x, s0.link_y]

    while q and exp < max_exp:
        cur = q.popleft()
        for d in ("LEFT", "RIGHT", "UP", "DOWN"):
            exp += 1
            em.set_state(cell_state[cur])
            for _ in range(hold):
                env.step(nes_action(d))
            s = read_snapshot(env.get_ram())
            if s.mode == 17:
                continue
            # Left room / scrolling?
            if s.screen != start or s.mode in (4, 6, 7, 10, 16) or s.transitioning:
                for _ in range(400):
                    env.step(nes_idle_action())
                    s2 = read_snapshot(env.get_ram())
                    if s2.mode in (PLAY_MODE, 5) and not s2.transitioning:
                        break
                s2 = read_snapshot(env.get_ram())
                if (
                    s2.level == LEVEL4
                    and s2.screen != start
                    and s2.mode in (PLAY_MODE, 5)
                ):
                    dest = f"0x{s2.screen:02x}"
                    # reconstruct path
                    path: list[str] = []
                    n: tuple[int, int] | None = cur
                    while n is not None and parent[n] is not None:
                        prev, pd = parent[n]
                        path.append(pd)
                        n = prev
                    path.reverse()
                    path.append(d)
                    if dest not in exits or len(path) < len(exits[dest]["path"]):
                        exits[dest] = {
                            "dir_last": d,
                            "path": path,
                            "hold": hold,
                            "path_len": len(path),
                            "end_xy": [s2.link_x, s2.link_y],
                            "end_sample": _sample(
                                s2, env.get_ram(), event=f"exit_{dest}"
                            ),
                        }
                continue
            # still in room
            nx, ny = s.link_x // quant * quant, s.link_y // quant * quant
            if (nx, ny) in seen:
                continue
            if abs(s.link_x - cur[0]) + abs(s.link_y - cur[1]) < 2:
                continue
            seen.add((nx, ny))
            cell_state[(nx, ny)] = em.get_state()
            parent[(nx, ny)] = (cur, d)
            q.append((nx, ny))
            bbox[0] = min(bbox[0], s.link_x)
            bbox[1] = min(bbox[1], s.link_y)
            bbox[2] = max(bbox[2], s.link_x)
            bbox[3] = max(bbox[3], s.link_y)

    em.set_state(st0)
    _idle(env, 3)
    return {
        "ok": bool(exits),
        "start_sc": f"0x{start:02x}",
        "start_xy": [s0.link_x, s0.link_y],
        "hold": hold,
        "quant": quant,
        "exp": exp,
        "cells": len(seen),
        "bbox": bbox,
        "exits": {
            k: {kk: vv for kk, vv in v.items() if kk != "end_sample"}
            | {"end_sample": v.get("end_sample")}
            for k, v in exits.items()
        },
    }


def _follow_path(env, path: list[str], *, hold: int, assist, dest: int | None) -> bool:
    start = read_snapshot(env.get_ram()).screen
    for i, d in enumerate(path):
        for _ in range(hold):
            env.step(nes_action(d))
            if assist is not None:
                assist.apply_env(env, frame=i)
            s = read_snapshot(env.get_ram())
            if s.screen != start or s.mode in (4, 6, 7, 10, 16) or s.transitioning:
                s2 = _settle_play(env)
                if dest is None:
                    return s2.screen != start and s2.mode in (PLAY_MODE, 5)
                return s2.screen == dest and s2.mode in (PLAY_MODE, 5)
    s2 = _settle_play(env, max_f=80)
    if dest is None:
        return s2.screen != start
    return s2.screen == dest and s2.mode in (PLAY_MODE, 5)


def _try_bomb(env, face: str, sx: int, sy: int, *, wait_blast: int = 100) -> dict:
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    bombs_before = snap0.bombs
    for _ in range(500):
        snap = read_snapshot(env.get_ram())
        if snap.mode not in (PLAY_MODE, 5):
            env.step(nes_idle_action())
            continue
        act, at = _goto_xy(snap, sx, sy, tol=4)
        env.step(act)
        if at:
            break
    # if still far, BFS-less skip
    snap = read_snapshot(env.get_ram())
    dist = abs(snap.link_x - sx) + abs(snap.link_y - sy)
    if dist > 24:
        return {
            "ok": False,
            "skipped": True,
            "reason": f"stand_unreachable dist={dist} at={[snap.link_x, snap.link_y]}",
            "face": face,
            "stand": [sx, sy],
            "start_sc": f"0x{start_sc:02x}",
        }

    _ensure_bomb_selected(env)
    for _ in range(6):
        env.step(nes_action(face))
    env.step(nes_action(face, "B"))
    _idle(env, 2)
    snap = read_snapshot(env.get_ram())
    bomb_consumed = snap.bombs < bombs_before
    for _ in range(wait_blast):
        env.step(nes_action(face) if face in ("UP", "DOWN") else nes_idle_action())

    reached = False
    for _ in range(320):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start_sc and snap.mode in (PLAY_MODE, 4, 6, 7):
            reached = True
            break
        if snap.transitioning or snap.mode in (4, 6, 7):
            env.step(nes_action(face))
        else:
            env.step(_push_door(snap, face))
    if reached:
        s2 = _settle_play(env)
    else:
        s2 = read_snapshot(env.get_ram())
    ok = reached and s2.screen != start_sc
    result = {
        "ok": ok,
        "face": face,
        "stand": [sx, sy],
        "bomb_consumed": bomb_consumed,
        "bombs_before": bombs_before,
        "bombs_after": int(s2.bombs),
        "start_sc": f"0x{start_sc:02x}",
        "end_sc": f"0x{s2.screen:02x}",
        "end_xy": [s2.link_x, s2.link_y],
        "end_sample": _sample(s2, env.get_ram(), event="bomb_end") if ok else None,
    }
    if ok:
        opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[face]
        for _ in range(400):
            snap = read_snapshot(env.get_ram())
            if snap.mode in (PLAY_MODE, 5) and snap.screen == start_sc:
                break
            env.step(nes_action(opp))
        _settle_play(env, max_f=100)
    return result


def _bfs_to_stand(env, sx: int, sy: int, *, hold: int = 4, quant: int = 4) -> bool:
    """Move toward stand with BFS when maze-blocked."""
    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    start = s0.screen
    goal = (sx // quant * quant, sy // quant * quant)
    st0 = em.get_state()
    start_c = (s0.link_x // quant * quant, s0.link_y // quant * quant)
    if abs(s0.link_x - sx) + abs(s0.link_y - sy) <= 6:
        return True
    cs = {start_c: st0}
    parent: dict = {start_c: None}
    q = deque([start_c])
    seen = {start_c}
    found = None
    exp = 0
    while q and exp < 8000 and found is None:
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
        _idle(env, 2)
        return False
    em.set_state(cs[found])
    _idle(env, 2)
    return True


def _push_blocks(env, centers: list[tuple[int, int]]) -> list[dict]:
    results = []
    snap0 = read_snapshot(env.get_ram())
    start = snap0.screen
    for cx, cy in centers:
        _bfs_to_stand(env, cx, cy)
        for d in ("UP", "RIGHT", "DOWN", "LEFT"):
            for _ in range(16):
                env.step(nes_action(d))
        snap = read_snapshot(env.get_ram())
        results.append(
            {
                "center": [cx, cy],
                "sc": f"0x{snap.screen:02x}",
                "mode": snap.mode,
                "room_changed": snap.screen != start,
                "xy": [snap.link_x, snap.link_y],
                "doors_after": snap.cur_opened_doors,
            }
        )
        if snap.screen != start:
            break
    return results


def _probe_room(
    env,
    assist,
    *,
    room: int,
    tag: str,
    phase: str,
    bomb_limit: int | None,
    do_push: bool,
    thrash: bool,
) -> dict:
    snap = read_snapshot(env.get_ram())
    if snap.screen != room or snap.level != LEVEL4:
        return {
            "phase": phase,
            "ok": False,
            "error": f"not_on_room sc=0x{snap.screen:02x} lv={snap.level}",
        }

    entry = _sample(snap, env.get_ram(), event=f"{phase}_entry")
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"{tag}_{phase}_entry.png")

    thrash_meta = None
    if thrash:
        thrash_meta = _thrash_gels(env, assist, room=room, max_frames=10000)
        save_rgb_png(
            env.step(nes_idle_action())[0],
            RECORDINGS_DIR / f"{tag}_{phase}_post_thrash.png",
        )

    # BFS free exits with multiple holds
    bfs_runs = []
    all_exits: dict[str, dict] = {}
    for hold in (4, 6, 3, 2):
        meta = _bfs_discover_exits(env, hold=hold, max_exp=18000, quant=4)
        bfs_runs.append(
            {
                "hold": hold,
                "ok": meta.get("ok"),
                "cells": meta.get("cells"),
                "exp": meta.get("exp"),
                "exits": list((meta.get("exits") or {}).keys()),
                "bbox": meta.get("bbox"),
            }
        )
        for dest, info in (meta.get("exits") or {}).items():
            if dest not in all_exits or info["path_len"] < all_exits[dest]["path_len"]:
                all_exits[dest] = info
        if all_exits:
            break  # enough once we have any free exit set

    # Also try hyp dests with dedicated BFS (path-replay style) if missing
    for dest in HYP_DESTS.get(room, []):
        dh = f"0x{dest:02x}"
        if dh in all_exits:
            continue
        # cheap targeted BFS
        em = env.unwrapped.em
        st0 = em.get_state()
        s0 = read_snapshot(env.get_ram())
        found_path = None
        hold = 4
        quant = 4
        q = deque([(s0.link_x // quant * quant, s0.link_y // quant * quant, ())])
        seen = {q[0][:2]}
        exp = 0
        while q and exp < 12000 and found_path is None:
            x, y, path = q.popleft()
            exp += 1
            if len(path) > 80:
                continue
            for d in ("LEFT", "RIGHT", "UP", "DOWN"):
                em.set_state(st0)
                for pd in path:
                    for _ in range(hold):
                        env.step(nes_action(pd))
                for _ in range(hold):
                    env.step(nes_action(d))
                s = read_snapshot(env.get_ram())
                if s.mode == 17:
                    continue
                if s.screen != room or s.mode in (4, 6, 7) or s.transitioning:
                    for _ in range(400):
                        env.step(nes_idle_action())
                        s2 = read_snapshot(env.get_ram())
                        if s2.mode in (PLAY_MODE, 5) and not s2.transitioning:
                            break
                    s2 = read_snapshot(env.get_ram())
                    if s2.screen == dest and s2.mode in (PLAY_MODE, 5):
                        found_path = list(path) + [d]
                        break
                    continue
                nx, ny = s.link_x // quant * quant, s.link_y // quant * quant
                if (nx, ny) in seen:
                    continue
                if abs(s.link_x - x) + abs(s.link_y - y) < 2:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, path + (d,)))
        em.set_state(st0)
        _idle(env, 2)
        if found_path is not None:
            all_exits[dh] = {
                "dir_last": found_path[-1],
                "path": found_path,
                "hold": hold,
                "path_len": len(found_path),
                "targeted": True,
            }

    # Bomb stands (after thrash / partial BFS coverage)
    bomb_tests = []
    stands = BOMB_STANDS if bomb_limit is None else BOMB_STANDS[:bomb_limit]
    # restore play on room
    snap = read_snapshot(env.get_ram())
    if snap.screen != room:
        # cannot bomb
        bomb_tests.append({"ok": False, "error": f"left_room_0x{snap.screen:02x}"})
    else:
        if snap.bombs < 4:
            _poke_bombs(env, 8)
        for face, sx, sy in stands:
            snap = read_snapshot(env.get_ram())
            if snap.screen != room:
                break
            if snap.bombs <= 0:
                _poke_bombs(env, 8)
            # try move to stand via BFS then bomb
            _bfs_to_stand(env, sx, sy)
            bt = _try_bomb(env, face, sx, sy)
            bomb_tests.append({k: v for k, v in bt.items() if k != "end_sample"})
            if bt.get("ok"):
                dest = bt["end_sc"]
                if dest not in all_exits:
                    all_exits[dest] = {
                        "dir_last": face,
                        "path": [face],
                        "hold": 1,
                        "path_len": 1,
                        "kind": "bomb",
                        "stand": [sx, sy],
                    }
                save_rgb_png(
                    env.step(nes_idle_action())[0],
                    RECORDINGS_DIR
                    / f"{tag}_{phase}_bomb_{face}_{sx}_{sy}_{dest}.png",
                )

    push_tests = []
    if do_push and read_snapshot(env.get_ram()).screen == room:
        push_tests = _push_blocks(env, PUSH_CENTERS)
        for pt in push_tests:
            if pt.get("room_changed"):
                dest = pt["sc"]
                if dest not in all_exits:
                    all_exits[dest] = {
                        "dir_last": "PUSH",
                        "path": [],
                        "hold": 0,
                        "path_len": 0,
                        "kind": "push",
                        "center": pt["center"],
                    }

    final = _sample(read_snapshot(env.get_ram()), env.get_ram(), event=f"{phase}_final")
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"{tag}_{phase}_final.png")

    live_edges = []
    for dest, info in all_exits.items():
        live_edges.append(
            {
                "kind": info.get("kind", "door_bfs"),
                "from": f"0x{room:02x}",
                "to": dest,
                "dir": info.get("dir_last"),
                "path_len": info.get("path_len"),
                "hold": info.get("hold"),
                "stand": info.get("stand"),
            }
        )

    return {
        "phase": phase,
        "ok": True,
        "room": f"0x{room:02x}",
        "entry": entry,
        "thrash": thrash_meta,
        "bfs_runs": bfs_runs,
        "exits": {
            k: {kk: vv for kk, vv in v.items() if kk != "end_sample"}
            for k, v in all_exits.items()
        },
        "bomb_tests": bomb_tests,
        "push_tests": push_tests,
        "live_edges": live_edges,
        "final": final,
        # keep paths for navigation
        "_exit_paths": all_exits,
    }


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    tag: str,
    bomb_limit: int | None,
    do_push: bool,
    expand_depth: int,
    thrash: bool,
    save_checkpoint: bool,
    poke_bombs: bool,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        snap = read_snapshot(env.get_ram())
        entry0 = _sample(snap, env.get_ram(), event="t0")
        if not (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_MAP_21
            and (int(read_u8(env.get_ram(), ADDR_MAP)) & LEVEL4_MAP_BIT)
        ):
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_not_map.png")
            return {
                "ok": False,
                "bead": "rr-rvae",
                "track": track,
                "error": "not_on_Level4Map_0x21_with_map_bit",
                "entry": entry0,
            }

        poke_notes = _poke_bombs(env, 8) if poke_bombs else []
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t0.png")

        rooms: dict[str, Any] = {}
        edges: list[dict] = []
        exit_paths: dict[tuple[int, int], dict] = {}  # (from,to) -> path info
        visited: set[int] = set()
        queue: deque[tuple[int, int]] = deque([(ROOM_L4_MAP_21, 0)])

        while queue:
            room, depth = queue.popleft()
            if room in visited or depth > expand_depth:
                continue

            snap = read_snapshot(env.get_ram())
            if snap.screen != room:
                # Try navigate via known path
                navigated = False
                key = (snap.screen, room)
                if key in exit_paths:
                    info = exit_paths[key]
                    navigated = _follow_path(
                        env,
                        info["path"],
                        hold=int(info.get("hold") or 4),
                        assist=assist,
                        dest=room,
                    )
                if not navigated:
                    # reverse edge
                    for (a, b), info in list(exit_paths.items()):
                        if b == snap.screen and a == room:
                            # no reverse path stored — skip
                            pass
                    rooms[f"0x{room:02x}"] = {
                        "ok": False,
                        "error": f"unreachable_from_0x{snap.screen:02x}",
                        "phase": f"d{depth}",
                    }
                    continue

            visited.add(room)
            phase = f"r{room:02x}_d{depth}"
            report = _probe_room(
                env,
                assist,
                room=room,
                tag=tag,
                phase=phase,
                bomb_limit=bomb_limit,
                do_push=do_push,
                thrash=thrash,
            )
            paths = report.pop("_exit_paths", {})
            rooms[f"0x{room:02x}"] = report
            for dest_h, info in paths.items():
                dest = int(dest_h, 16)
                edges.append(
                    {
                        "kind": info.get("kind", "door_bfs"),
                        "from": f"0x{room:02x}",
                        "to": dest_h,
                        "dir": info.get("dir_last"),
                        "path_len": info.get("path_len"),
                        "hold": info.get("hold"),
                        "stand": info.get("stand"),
                    }
                )
                exit_paths[(room, dest)] = info
                if dest not in visited and depth + 1 <= expand_depth:
                    queue.append((dest, depth + 1))

            # Prefer expand west first for Gleeok path (map is side room).
            # Reorder is already natural via LEFT-first BFS dirs.

            # Navigate into next unvisited if still on current and depth allows
            snap = read_snapshot(env.get_ram())
            if snap.screen == room:
                # pick next queued dest that has path
                for dest, d2 in list(queue):
                    if dest in visited or d2 != depth + 1:
                        continue
                    if (room, dest) in exit_paths:
                        info = exit_paths[(room, dest)]
                        ok_nav = _follow_path(
                            env,
                            info["path"],
                            hold=int(info.get("hold") or 4),
                            assist=assist,
                            dest=dest,
                        )
                        if ok_nav:
                            save_rgb_png(
                                env.step(nes_idle_action())[0],
                                RECORDINGS_DIR / f"{tag}_enter_0x{dest:02x}.png",
                            )
                        break

        boss_hits = []
        for sc, rep in rooms.items():
            for key in ("entry", "final"):
                sample = rep.get(key) or {}
                for o in sample.get("objects") or []:
                    t = o.get("type", 0)
                    if t in (0x3C, 0x3D, 0x32, 0x41) or (
                        isinstance(t, int) and t >= 0x30 and o.get("hp", 0) >= 64
                        and t not in (0x5F, 0x2B, 0x68)
                    ):
                        boss_hits.append({"room": sc, "obj": o, "sample": key})
            # also scan exit samples
            for dest, info in (rep.get("exits") or {}).items():
                pass

        # Extra: if we reached 0x20 and have path, note prior map2 Manhandla fact
        known = {
            "0x20_UP": "0x10 Manhandla (type 0x3c) — prior l4_rvae_map2 recon",
            "map_room": "0x21 ADDR_MAP|0x08",
        }

        tf_l4 = bool(int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & 0x08)
        checkpoint = None
        if save_checkpoint:
            snap = read_snapshot(env.get_ram())
            if snap.level == LEVEL4:
                name = f"Level4MapExpand_0x{snap.screen:02x}"
                path = save_state(env, GAME_DIR, GAME, name)
                checkpoint = str(path)
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR
                    / "custom_integrations"
                    / GAME
                    / f"{start_state}.state",
                    request={
                        "bead": "rr-rvae",
                        "segment": "l4_map_gleeok_probe",
                        "track": track,
                        "intervention_class": "survival" if infinite_life else "clean",
                    },
                    selected_trial={
                        "final": _sample(snap, env.get_ram(), event="checkpoint"),
                        "live_edges": edges,
                        "visited": [f"0x{r:02x}" for r in sorted(visited)],
                    },
                )

        return {
            "ok": bool(edges) or bool(visited),
            "bead": "rr-rvae",
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "entry": entry0,
            "poke_notes": poke_notes,
            "expand_depth": expand_depth,
            "visited": [f"0x{r:02x}" for r in sorted(visited)],
            "live_edges": edges,
            "live_edge_map": {
                f"{e['from']}-{e.get('dir')}": e["to"] for e in edges
            },
            "rooms": rooms,
            "boss_hits": boss_hits,
            "known_prior": known,
            "tf_l4": tf_l4,
            "checkpoint": checkpoint,
            "final": _sample(
                read_snapshot(env.get_ram()), env.get_ram(), event="final"
            ),
            "assist": assist.report() if assist else None,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level4Map")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--tag", default="l4_rvae_gleeok")
    parser.add_argument("--bomb-limit", type=int, default=8)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--expand-depth", type=int, default=2)
    parser.add_argument("--no-thrash", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--no-poke-bombs", action="store_true")
    args = parser.parse_args(argv)

    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        tag=args.tag,
        bomb_limit=args.bomb_limit,
        do_push=not args.no_push,
        expand_depth=args.expand_depth,
        thrash=not args.no_thrash,
        save_checkpoint=args.save_state,
        poke_bombs=not args.no_poke_bombs,
    )
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print(
        f"ok={report.get('ok')} visited={report.get('visited')} "
        f"edges={len(report.get('live_edges') or [])} "
        f"boss_hits={len(report.get('boss_hits') or [])} "
        f"tf_l4={report.get('tf_l4')} wrote={out}"
    )
    for e in report.get("live_edges") or []:
        print(
            f"  EDGE {e.get('kind')} {e.get('from')} -{e.get('dir')}-> {e.get('to')} "
            f"plen={e.get('path_len')} stand={e.get('stand')}"
        )
    for sc, rep in (report.get("rooms") or {}).items():
        print(
            f"  ROOM {sc} ok={rep.get('ok')} exits={list((rep.get('exits') or {}).keys())} "
            f"err={rep.get('error')}"
        )
    for b in report.get("boss_hits") or []:
        o = b.get("obj") or {}
        print(
            f"  BOSS? {b.get('room')} type=0x{o.get('type', 0):02x} "
            f"{o.get('type_name')} hp={o.get('hp')}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
