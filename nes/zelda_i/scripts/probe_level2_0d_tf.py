"""Assisted recon: collect triforce bit 0x02 in L2 room 0x0d after Dodongo.

Geometry discovery only (rr-n5i residual). Not Clean STATUS.

Live facts (2026-08-07):
  Boss 0x0e type 0x32 → bomb-mouth → HC; post-kill doors LEFT only → 0x0d.
  Room 0x0d diamond floor is a **maze open from the SOUTH** (not solid seal).
  Collect at ~(128, 149): waypoints (208,141)→(208,189)→(128,189)→(128,149).
  Green north-band sprite is NOT the collect hitbox. Walkthrough "east of boss"
  is wrong for live (RIGHT sealed).

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_0d_tf.py --policy-only
    uv run python nes/zelda_i/scripts/probe_level2_0d_tf.py --from-state Level2_0D_PostBoss --policy-only
    uv run python nes/zelda_i/scripts/probe_level2_0d_tf.py --infinite-life --from-state Level2_0E
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

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
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# selected-item address lives in inventory block (same as other runners)
ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

ROOM_0E = 0x0E
ROOM_0D = 0x0D
DODONGO_TYPE = 0x32
FACE_E, FACE_W, FACE_S, FACE_N = 0x01, 0x02, 0x04, 0x08

# LIVE 2026-08-07: south-band maze into pedestal (mode 18, tf|0x02).
TF_WAYPOINTS = [
    (208, 141),  # east free column at door height (LEFT from x≈224 first)
    (208, 189),  # south-east free
    (128, 189),  # south band under pedestal
    (128, 149),  # triforce collect hitbox
]

# Visual north-band green item (from l2_0d_map.png / blocks_end) — pixel guesses
# refined by live object dump + approach trials. (NOT the collect hitbox.)
NORTH_ITEM_GUESSES = [
    (160, 93),
    (152, 93),
    (168, 93),
    (144, 93),
    (176, 93),
    (136, 93),
    (184, 93),
    (160, 88),
    (152, 88),
    (168, 88),
    (160, 101),
    (152, 101),
    (148, 96),
    (156, 96),
    (164, 96),
    (172, 96),
    (120, 93),
    (128, 93),
    (112, 93),
    (120, 101),
    (120, 109),
    (120, 125),
    (120, 141),  # classic center pedestal (likely solid)
    (104, 93),
    (96, 93),
    (80, 93),
    (64, 93),
    (48, 93),
]

# Diamond / black-block push centers (room-relative pixel coords)
PUSH_CENTERS = [
    (120, 141),
    (104, 141),
    (136, 141),
    (120, 125),
    (120, 157),
    (88, 141),
    (152, 141),
    (104, 125),
    (136, 125),
    (104, 157),
    (136, 157),
    (88, 125),
    (152, 125),
    (72, 141),
    (168, 141),
    (120, 109),
    (120, 173),
    # black decorative blocks visible mid-room in screenshots
    (96, 157),
    (144, 157),
    (96, 125),
    (144, 125),
    (112, 141),
    (128, 141),
]

BOMB_STANDS = [
    (120, 109, "UP"),
    (120, 173, "DOWN"),
    (64, 141, "LEFT"),
    (176, 141, "RIGHT"),
    (120, 125, "UP"),
    (96, 141, "LEFT"),
    (144, 141, "RIGHT"),
    (160, 109, "UP"),
    (80, 109, "UP"),
    (120, 157, "DOWN"),
    (88, 157, "LEFT"),
    (152, 157, "RIGHT"),
]


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _ensure_bomb(env) -> None:
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


def _poke_bombs(env, n: int = 16) -> str:
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        return f"poke_fail={exc!r}"


def _goto(snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 4):
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def _tf02(ram) -> bool:
    return bool(read_u8(ram, ADDR_TRIFORCE) & 0x02)


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out = []
    for o in snap.objects:
        if not (0 <= o.slot <= 12):
            continue
        if o.type_id in (0, 0xFF) and o.hp == 0 and o.x == 0 and o.y == 0:
            continue
        out.append(
            {
                "slot": o.slot,
                "type": f"0x{o.type_id:02x}",
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
                "facing": o.facing,
                "state": o.state,
            }
        )
    return out


def _sample(snap: ZeldaSnapshot, ram, *, event: str) -> dict:
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    ]
    return {
        "event": event,
        "mode": snap.mode,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "facing": snap.facing,
        "doors": snap.cur_opened_doors,
        "mask": snap.open_doorway_mask,
        "all_dead": snap.room_all_dead,
        "room_item": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "live": len(live),
        "types": {
            f"0x{k:02x}": v
            for k, v in Counter(o.type_id for o in live).items()
        },
        "objects": _objs(snap),
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf02": _tf02(ram),
        "bombs": snap.bombs,
        "keys": snap.keys,
        "health": snap.health,
    }


def _enter_left(env, dest: int, *, budget: int = 900) -> bool:
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("LEFT")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("LEFT"))
    return read_snapshot(env.get_ram()).screen == dest


def _mouth_target(dodo) -> tuple[int, int, str]:
    f = int(dodo.facing)
    if f & FACE_E:
        return dodo.x + 12, dodo.y, "LEFT"
    if f & FACE_W:
        return dodo.x - 12, dodo.y, "RIGHT"
    if f & FACE_S:
        return dodo.x, dodo.y + 12, "UP"
    if f & FACE_N:
        return dodo.x, dodo.y - 12, "DOWN"
    return dodo.x, dodo.y, "UP"


def _fight_dodongo(env, assist, *, max_frames: int = 14000) -> dict:
    log = []
    bombs_used = 0
    place_cd = 0
    last_hp = None
    hits = 0
    _poke_bombs(env, 16)
    _ensure_bomb(env)
    f = 0
    for f in range(max_frames):
        if assist is not None and f % 15 == 0:
            assist.apply_env(env, frame=f)
        s = read_snapshot(env.get_ram())
        if s.bombs < 2 and assist is not None:
            _poke_bombs(env, 12)
            _ensure_bomb(env)
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if _tf02(env.get_ram()):
            log.append(_sample(s, env.get_ram(), event="tf_mid_fight"))
            break
        dodos = [
            o
            for o in s.objects
            if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10
        ]
        living = [o for o in dodos if o.hp > 0]
        if not living and not dodos and s.room_all_dead >= 20:
            log.append(_sample(s, env.get_ram(), event="dodongo_dead"))
            break
        if not living:
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[f // 20 % 4], "A"))
            if f > 200 and s.room_all_dead >= 20 and not dodos:
                log.append(_sample(s, env.get_ram(), event="dodongo_dead_settle"))
                break
            continue
        d = living[0]
        if last_hp is not None and d.hp < last_hp:
            hits += 1
        last_hp = d.hp
        tx, ty, face = _mouth_target(d)
        tx = max(48, min(192, tx))
        ty = max(105, min(185, ty))
        dist = abs(s.link_x - d.x) + abs(s.link_y - d.y)
        at_mouth = abs(s.link_x - tx) <= 12 and abs(s.link_y - ty) <= 12
        if place_cd > 0:
            place_cd -= 1
            if place_cd > 50:
                retreat = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(
                    face, "DOWN"
                )
                env.step(nes_action(retreat))
            elif place_cd > 20:
                env.step(nes_action(face, "A"))
            else:
                env.step(nes_idle_action())
            continue
        if (at_mouth or dist <= 24) and s.bombs > 0:
            _ensure_bomb(env)
            if dist > 14:
                act, _ = _goto(s, d.x, d.y, tol=8)
                env.step(act)
                continue
            env.step(nes_action(face))
            env.step(nes_action(face, "B"))
            bombs_used += 1
            place_cd = 95
            continue
        act, _ = _goto(s, tx, ty, tol=6)
        env.step(act)
    s = read_snapshot(env.get_ram())
    alive = [o for o in s.objects if o.type_id == DODONGO_TYPE and o.hp > 0]
    return {
        "success": len(alive) == 0,
        "frames": f + 1,
        "bombs_used_est": bombs_used,
        "hits_est": hits,
        "final": _sample(s, env.get_ram(), event="fight_end"),
        "log_tail": log[-10:],
    }


def _collect_heart_and_exit_left(env, assist, *, budget: int = 2500) -> dict:
    """Touch HC center on 0x0e then LEFT → 0x0d."""
    heart = False
    log = []
    for f in range(budget):
        if assist is not None and f % 20 == 0:
            assist.apply_env(env, frame=10000 + f)
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_0D and s.mode == PLAY_MODE:
            log.append(_sample(s, env.get_ram(), event="entered_0d"))
            return {"ok": True, "frames": f + 1, "log": log, "final": log[-1]}
        if s.mode != PLAY_MODE:
            env.step(
                nes_action("LEFT")
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        if s.screen != ROOM_0E:
            env.step(nes_idle_action())
            continue
        if not heart and f < 350:
            act, at = _goto(s, 120, 141, tol=8)
            env.step(act)
            if at:
                heart = True
            continue
        # Prefer open LEFT bit; else brute LEFT at y=141
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("LEFT"))
        if f % 80 == 0:
            log.append(_sample(s, env.get_ram(), event=f"exit_f{f}"))
    s = read_snapshot(env.get_ram())
    return {
        "ok": s.screen == ROOM_0D,
        "frames": budget,
        "log": log,
        "final": _sample(s, env.get_ram(), event="exit_fail"),
    }


def _walk_to(env, tx: int, ty: int, *, budget: int = 400, tol: int = 3) -> dict:
    """Naive goto with stuck-wiggle; returns reachability + final xy."""
    last = (-1, -1)
    stuck = 0
    for i in range(budget):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if abs(s.link_x - tx) <= tol and abs(s.link_y - ty) <= tol:
            return {
                "reach": True,
                "frames": i + 1,
                "xy": [s.link_x, s.link_y],
                "target": [tx, ty],
            }
        xy = (s.link_x, s.link_y)
        if xy == last:
            stuck += 1
        else:
            stuck, last = 0, xy
        if stuck > 12:
            env.step(nes_action(("DOWN", "LEFT", "UP", "RIGHT", "DOWN")[stuck % 5]))
            continue
        act, _ = _goto(s, tx, ty, tol=tol)
        env.step(act)
    s = read_snapshot(env.get_ram())
    return {
        "reach": False,
        "frames": budget,
        "xy": [s.link_x, s.link_y],
        "target": [tx, ty],
    }


def _free_cell_bfs(env, *, step: int = 4, budget_per: int = 180) -> dict:
    """Explore walkable cells from current position via BFS of targets.

    Strategy: from each visited cell, try stepping ±step in 4 dirs by holding
    that direction for up to ~step frames; record reachable (link_x, link_y)
    quantized to step grid. Also tries explicit walk_to each candidate.
    """
    s0 = read_snapshot(env.get_ram())
    if s0.screen != ROOM_0D:
        return {"error": f"not_on_0d sc=0x{s0.screen:02x}", "cells": []}

    start = (s0.link_x, s0.link_y)
    # Quantize start
    q0 = (start[0] // step * step, start[1] // step * step)
    visited: set[tuple[int, int]] = set()
    reached_raw: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    queue.append(q0)
    visited.add(q0)
    trials = 0
    max_trials = 900

    # Also do a free-walk: hold each dir from start to map walls
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        # reload is expensive — instead walk and record, then try return
        pass

    def record_pos():
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_0D or s.mode != PLAY_MODE:
            return None
        raw = (s.link_x, s.link_y)
        reached_raw.add(raw)
        q = (s.link_x // step * step, s.link_y // step * step)
        return q

    # Phase A: radial free-walk from start — hold each dir long enough to hit wall
    for d in ("UP", "DOWN", "LEFT", "RIGHT", "UP", "LEFT", "DOWN", "RIGHT"):
        for _ in range(80):
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE:
                env.step(nes_idle_action())
                continue
            if s.screen != ROOM_0D:
                # try return
                env.step(nes_action("RIGHT" if d == "LEFT" else "LEFT" if d == "RIGHT" else "DOWN" if d == "UP" else "UP"))
                continue
            env.step(nes_action(d))
            q = record_pos()
            if q and q not in visited:
                visited.add(q)
                queue.append(q)

    # Phase B: from start, try walk_to grid of candidates around free band
    # expand envelope from known free cells
    for _round in range(6):
        if trials >= max_trials:
            break
        frontier = list(visited)
        for qx, qy in frontier:
            for dx, dy in ((0, -step), (0, step), (-step, 0), (step, 0),
                           (-step, -step), (step, -step), (-step, step), (step, step)):
                if trials >= max_trials:
                    break
                tx, ty = qx + dx, qy + dy
                # room bounds rough
                if not (24 <= tx <= 232 and 72 <= ty <= 210):
                    continue
                tq = (tx // step * step, ty // step * step)
                if tq in visited:
                    continue
                trials += 1
                r = _walk_to(env, tx, ty, budget=budget_per, tol=max(2, step // 2))
                q = record_pos()
                if r["reach"] or (q and abs(r["xy"][0] - tx) <= step and abs(r["xy"][1] - ty) <= step):
                    visited.add(tq)
                    if q:
                        visited.add(q)
                # if we left room, try re-enter
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM_0D:
                    for _ in range(200):
                        s = read_snapshot(env.get_ram())
                        if s.screen == ROOM_0D and s.mode == PLAY_MODE:
                            break
                        if s.mode != PLAY_MODE:
                            env.step(nes_action("RIGHT") if s.transitioning else nes_idle_action())
                        else:
                            env.step(nes_action("RIGHT"))  # from 0e re-enter? left door of 0d is from 0e LEFT so re-enter from right of 0d
                    # after leaving to 0e, re-enter 0d left
                    if read_snapshot(env.get_ram()).screen == ROOM_0E:
                        _enter_left(env, ROOM_0D, budget=500)

    cells = sorted(visited)
    raw = sorted(reached_raw)
    xs = [c[0] for c in cells] or [0]
    ys = [c[1] for c in cells] or [0]
    return {
        "step": step,
        "n_quantized": len(cells),
        "n_raw": len(raw),
        "x_range": [min(xs), max(xs)],
        "y_range": [min(ys), max(ys)],
        "cells": [[x, y] for x, y in cells],
        "raw_sample": [[x, y] for x, y in raw[:80]],
        "trials": trials,
        "start": list(start),
        "summary": (
            f"step={step} quantized={len(cells)} raw≈{len(raw)} "
            f"x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]"
        ),
    }


def _hold_path(env, moves: list[tuple[str, int]]) -> None:
    for d, n in moves:
        for _ in range(n):
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE:
                env.step(nes_idle_action())
            else:
                env.step(nes_action(d))


def _approach_and_collect(env, targets: list[tuple[int, int]], *, tag: str) -> list[dict]:
    """Walk each target; idle 30 frames; check TF bit."""
    results = []
    for tx, ty in targets:
        r = _walk_to(env, tx, ty, budget=500, tol=2)
        # micro-nudge + idle (pickup frames)
        for d in ("LEFT", "RIGHT", "UP", "DOWN", ""):
            for _ in range(8):
                if d:
                    env.step(nes_action(d))
                else:
                    env.step(nes_idle_action())
            s = read_snapshot(env.get_ram())
            if _tf02(env.get_ram()) or s.mode == 18:
                snap = _sample(s, env.get_ram(), event="tf_got")
                results.append(
                    {
                        "target": [tx, ty],
                        "reach": r,
                        "tf02": True,
                        "mode": s.mode,
                        "final": snap,
                        "policy": f"walk_to({tx},{ty})+nudge",
                    }
                )
                return results
        s = read_snapshot(env.get_ram())
        results.append(
            {
                "target": [tx, ty],
                "reach": r,
                "tf02": False,
                "mode": s.mode,
                "xy": [s.link_x, s.link_y],
                "room_item": s.room_item_id,
            }
        )
    return results


def _push_trials(env, centers: list[tuple[int, int]], *, hold: int = 90) -> list[dict]:
    out = []
    s0 = read_snapshot(env.get_ram())
    doors0 = s0.cur_opened_doors
    sc0 = s0.screen
    for cx, cy in centers:
        r = _walk_to(env, cx, cy, budget=350, tol=4)
        for d in ("UP", "RIGHT", "DOWN", "LEFT"):
            for _ in range(hold // 4):
                env.step(nes_action(d))
            s = read_snapshot(env.get_ram())
            entry = {
                "stand": [cx, cy],
                "dir": d,
                "reach_stand": r["reach"],
                "xy": [s.link_x, s.link_y],
                "doors_before": doors0,
                "doors_after": s.cur_opened_doors,
                "doors_changed": s.cur_opened_doors != doors0,
                "left_room": s.screen != sc0,
                "tf02": _tf02(env.get_ram()),
                "mode": s.mode,
                "room_item": s.room_item_id,
            }
            out.append(entry)
            if entry["tf02"] or entry["doors_changed"] or entry["left_room"]:
                return out
            # if left room, re-enter
            if s.screen != sc0:
                if s.screen == ROOM_0E:
                    _enter_left(env, ROOM_0D, budget=600)
                return out
    return out


def _bomb_trials(env, stands: list[tuple[int, int, str]], *, wait: int = 100) -> list[dict]:
    out = []
    _poke_bombs(env, 16)
    _ensure_bomb(env)
    s0 = read_snapshot(env.get_ram())
    doors0 = s0.cur_opened_doors
    sc0 = s0.screen
    for sx, sy, face in stands:
        r = _walk_to(env, sx, sy, budget=350, tol=4)
        if not r["reach"]:
            out.append(
                {
                    "stand": [sx, sy],
                    "face": face,
                    "reach": False,
                    "xy": r["xy"],
                }
            )
            continue
        _ensure_bomb(env)
        for _ in range(4):
            env.step(nes_action(face))
        env.step(nes_action(face, "B"))
        for _ in range(wait):
            env.step(nes_idle_action())
        # walk into blast area / potential hole
        for d in (face, "UP", "DOWN", "LEFT", "RIGHT"):
            for _ in range(20):
                env.step(nes_action(d))
            s = read_snapshot(env.get_ram())
            if _tf02(env.get_ram()) or s.mode == 18:
                out.append(
                    {
                        "stand": [sx, sy],
                        "face": face,
                        "reach": True,
                        "tf02": True,
                        "mode": s.mode,
                        "xy": [s.link_x, s.link_y],
                        "doors": s.cur_opened_doors,
                    }
                )
                return out
        s = read_snapshot(env.get_ram())
        out.append(
            {
                "stand": [sx, sy],
                "face": face,
                "reach": True,
                "tf02": False,
                "xy": [s.link_x, s.link_y],
                "doors_before": doors0,
                "doors_after": s.cur_opened_doors,
                "doors_changed": s.cur_opened_doors != doors0,
                "left_room": s.screen != sc0,
                "mode": s.mode,
                "room_item": s.room_item_id,
            }
        )
        if s.screen != sc0:
            if s.screen == ROOM_0E:
                _enter_left(env, ROOM_0D, budget=600)
            break
    return out


def _north_band_sweep(env) -> list[dict]:
    """From east door: UP to north wall, then LEFT across north free strip.

    Screenshots show green TF on north band; prior free-cell maps had y≈88
    corridor with a center gap hypothesis — sweep every few px.
    """
    results = []
    # Reset toward east entry alcove first
    for _ in range(120):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if s.link_x >= 200:
            break
        env.step(nes_action("RIGHT"))
    # UP to north
    for _ in range(100):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if s.link_y <= 90:
            break
        env.step(nes_action("UP"))
    s = read_snapshot(env.get_ram())
    results.append({"event": "north_band_start", "xy": [s.link_x, s.link_y]})

    # Sweep LEFT holding, sample every few frames
    for i in range(200):
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        if _tf02(env.get_ram()) or s.mode == 18:
            results.append(
                {
                    "event": "tf_during_north_sweep",
                    "i": i,
                    "xy": [s.link_x, s.link_y],
                    "mode": s.mode,
                    "tf": int(read_u8(env.get_ram(), ADDR_TRIFORCE)),
                }
            )
            return results
        env.step(nes_action("LEFT"))
        if i % 8 == 0:
            s = read_snapshot(env.get_ram())
            results.append(
                {
                    "event": "north_sample",
                    "i": i,
                    "xy": [s.link_x, s.link_y],
                    "room_item": s.room_item_id,
                }
            )
    # reverse RIGHT sweep with slight DOWN/UP
    for y_nudge in ("", "DOWN", "UP"):
        for i in range(200):
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE:
                env.step(nes_idle_action())
                continue
            if _tf02(env.get_ram()) or s.mode == 18:
                results.append(
                    {
                        "event": "tf_during_north_resweep",
                        "nudge": y_nudge,
                        "i": i,
                        "xy": [s.link_x, s.link_y],
                        "mode": s.mode,
                    }
                )
                return results
            if y_nudge and i % 15 == 0:
                env.step(nes_action(y_nudge))
            else:
                env.step(nes_action("RIGHT"))
    s = read_snapshot(env.get_ram())
    results.append(
        {
            "event": "north_sweep_end",
            "xy": [s.link_x, s.link_y],
            "tf02": _tf02(env.get_ram()),
        }
    )
    return results


def _perimeter_full(env) -> list[dict]:
    """Walk perimeter clockwise: E wall N→S, S wall E→W, W wall S→N, N wall W→E."""
    path = [
        ("RIGHT", 80),
        ("UP", 120),
        ("LEFT", 200),
        ("DOWN", 140),
        ("RIGHT", 200),
        ("UP", 80),
        ("LEFT", 100),
        ("DOWN", 40),
        ("RIGHT", 40),
        ("UP", 40),
        ("LEFT", 40),
        ("DOWN", 40),
    ]
    log = []
    for d, n in path:
        for i in range(n):
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE:
                env.step(nes_idle_action())
                continue
            if _tf02(env.get_ram()) or s.mode == 18:
                log.append(
                    {
                        "event": "tf_perimeter",
                        "dir": d,
                        "i": i,
                        "xy": [s.link_x, s.link_y],
                        "mode": s.mode,
                    }
                )
                return log
            env.step(nes_action(d))
        s = read_snapshot(env.get_ram())
        log.append({"event": "peri_corner", "dir": d, "xy": [s.link_x, s.link_y]})
    return log


def _south_band_tf_policy(env, *, budget_per_wp: int = 500) -> dict:
    """Known LIVE path: east column → south band → (128,149)."""
    log = []
    for tx, ty in TF_WAYPOINTS:
        r = _walk_to(env, tx, ty, budget=budget_per_wp, tol=3)
        s = read_snapshot(env.get_ram())
        entry = {
            "wp": [tx, ty],
            "reach": r["reach"],
            "xy": [s.link_x, s.link_y],
            "tf": int(read_u8(env.get_ram(), ADDR_TRIFORCE)),
            "tf02": _tf02(env.get_ram()),
            "mode": s.mode,
        }
        log.append(entry)
        for _ in range(12):
            env.step(nes_idle_action())
            if _tf02(env.get_ram()):
                break
        if _tf02(env.get_ram()) or s.mode == 18:
            break
    if not _tf02(env.get_ram()):
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            for _ in range(4):
                env.step(nes_action(d))
                if _tf02(env.get_ram()):
                    break
            if _tf02(env.get_ram()):
                break
    for _ in range(40):
        env.step(nes_idle_action())
        if _tf02(env.get_ram()) and read_snapshot(env.get_ram()).mode == 18:
            break
    s = read_snapshot(env.get_ram())
    return {
        "ok": _tf02(env.get_ram()),
        "tf": int(read_u8(env.get_ram(), ADDR_TRIFORCE)),
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "waypoints": [list(w) for w in TF_WAYPOINTS],
        "log": log,
        "kind": "south_band_waypoints",
    }


def run_probe(
    *,
    start_state: str = "Level2_0E",
    infinite_life: bool = True,
    step: int = 4,
    tag: str = "l2_0d_tf_reach",
    save_mid: bool = False,
    policy_only: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    negatives: list[dict] = []
    winning_policy = None
    free_summary = ""
    free_cells: dict = {}
    objects_at_entry: list = []
    timeline: list = []

    try:
        env.reset()
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        _poke_bombs(env, 16)
        _ensure_bomb(env)

        s = read_snapshot(env.get_ram())
        timeline.append(_sample(s, env.get_ram(), event="boot"))

        # --- Phase 0: ensure post-kill on 0x0e with LEFT open ---
        if s.screen == ROOM_0E:
            dodos = [
                o
                for o in s.objects
                if o.type_id == DODONGO_TYPE and 1 <= o.slot <= 10 and o.hp > 0
            ]
            if dodos or (
                any(o.type_id == DODONGO_TYPE for o in s.objects)
                and s.room_all_dead < 20
            ):
                fight = _fight_dodongo(env, assist)
                timeline.append({"event": "fight", **{k: v for k, v in fight.items() if k != "log_tail"}})
                if not fight["success"]:
                    return _pack(
                        result="FAIL",
                        reason="dodongo_alive",
                        timeline=timeline,
                        free_summary="n/a",
                        objects=[],
                        negatives=[{"phase": "fight", "detail": fight}],
                        winning_policy=None,
                        tag=tag,
                        env=env,
                    )
            # collect heart + exit LEFT
            ex = _collect_heart_and_exit_left(env, assist)
            timeline.append({"event": "exit_to_0d", "ok": ex["ok"], "frames": ex["frames"]})
            timeline.extend(ex.get("log") or [])
            if not ex["ok"]:
                # maybe already can walk left without HC
                if not _enter_left(env, ROOM_0D, budget=700):
                    return _pack(
                        result="FAIL",
                        reason="no_enter_0d",
                        timeline=timeline,
                        free_summary="n/a",
                        objects=[],
                        negatives=[{"phase": "exit", "detail": ex}],
                        winning_policy=None,
                        tag=tag,
                        env=env,
                    )
        elif s.screen == ROOM_0D:
            pass
        else:
            return _pack(
                result="FAIL",
                reason=f"bad_start_sc=0x{s.screen:02x}",
                timeline=timeline,
                free_summary="n/a",
                objects=[],
                negatives=[],
                winning_policy=None,
                tag=tag,
                env=env,
            )

        # Settle 0x0d
        for _ in range(40):
            if assist is not None:
                assist.apply_env(env, frame=20000)
            env.step(nes_idle_action())
        s = read_snapshot(env.get_ram())
        entry = _sample(s, env.get_ram(), event="0d_entry")
        timeline.append(entry)
        objects_at_entry = entry["objects"]
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entry.png")

        if _tf02(env.get_ram()):
            winning_policy = {"kind": "already_had_tf02"}
            return _pack(
                result="LIVE",
                reason="tf02_at_entry",
                timeline=timeline,
                free_summary="n/a",
                objects=objects_at_entry,
                negatives=[],
                winning_policy=winning_policy,
                tag=tag,
                env=env,
            )

        if save_mid:
            save_state(env, GAME_DIR, GAME, "Level2_0D_PostBoss")

        # --- Phase 0b: known LIVE south-band policy (try first) ---
        pol = _south_band_tf_policy(env)
        timeline.append({"event": "south_band_policy", **{k: v for k, v in pol.items() if k != "log"}})
        timeline.extend(pol.get("log") or [])
        if pol["ok"]:
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_LIVE.png")
            winning_policy = {
                "kind": "south_band_waypoints",
                "waypoints": pol["waypoints"],
                "tol": 3,
                "collect_xy": pol["xy"],
                "mode": pol["mode"],
                "tf": pol["tf"],
                "log": pol["log"],
            }
            return _pack(
                result="LIVE",
                reason="south_band_policy",
                timeline=timeline,
                free_summary=(
                    "policy-only" if policy_only else "skipped_full_map_after_LIVE_policy"
                ),
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=None,
                tag=tag,
                env=env,
                next_encode_hint=(
                    "LIVE: encode TF_WAYPOINTS into run_level2_dodongo._collect_and_tf. "
                    "Stop on tf&0x02 / mode 18. Checkpoint Level2_0D_PostBoss."
                ),
            )
        negatives.append({"trial": "south_band_policy", "detail": pol, "tf02": False})
        if policy_only:
            return _pack(
                result="FAIL",
                reason="policy_only_miss",
                timeline=timeline,
                free_summary="policy_only",
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=None,
                tag=tag,
                env=env,
                next_encode_hint="South-band policy failed; re-run full recon without --policy-only.",
            )

        # --- Phase 1: free-cell BFS ---
        free_cells = _free_cell_bfs(env, step=step, budget_per=160)
        free_summary = free_cells.get("summary", "")
        timeline.append({"event": "free_cells", "summary": free_summary, "n": free_cells.get("n_quantized")})
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_bfs.png")

        # --- Phase 2: north-band sweep (screenshot hypothesis) ---
        north = _north_band_sweep(env)
        timeline.append({"event": "north_band", "n": len(north), "tail": north[-5:]})
        if any(e.get("event", "").startswith("tf_") for e in north):
            winning_policy = {
                "kind": "north_band_sweep",
                "detail": north,
                "notes": "UP from east door then LEFT along y≈88–93 free strip",
            }
            return _pack(
                result="LIVE",
                reason="north_band",
                timeline=timeline,
                free_summary=free_summary,
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=free_cells,
                tag=tag,
                env=env,
            )
        negatives.append({"trial": "north_band_sweep", "samples": north[-12:], "tf02": False})
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_north.png")

        # --- Phase 3: approach guessed item coords + free cells ---
        # Build targets: free cells near north + explicit guesses + all free cells sample
        cells = free_cells.get("cells") or []
        northish = [(x, y) for x, y in cells if y <= 110]
        targets = list(NORTH_ITEM_GUESSES)
        for c in northish:
            if tuple(c) not in targets:
                targets.append(tuple(c))
        # also every free cell (might step on room item)
        for c in cells:
            t = tuple(c)
            if t not in targets:
                targets.append(t)

        approach = _approach_and_collect(env, targets[:80], tag=tag)
        timeline.append(
            {
                "event": "approach",
                "n": len(approach),
                "any_tf": any(a.get("tf02") for a in approach),
            }
        )
        if any(a.get("tf02") for a in approach):
            win = next(a for a in approach if a.get("tf02"))
            winning_policy = {
                "kind": "walk_to_target",
                "target": win["target"],
                "detail": win,
            }
            return _pack(
                result="LIVE",
                reason="approach",
                timeline=timeline,
                free_summary=free_summary,
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=free_cells,
                tag=tag,
                env=env,
            )
        negatives.append(
            {
                "trial": "approach_targets",
                "n": len(approach),
                "reached": sum(1 for a in approach if a.get("reach", {}).get("reach")),
                "sample": approach[:15] + approach[-5:],
            }
        )
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_approach.png")

        # --- Phase 4: perimeter walk ---
        peri = _perimeter_full(env)
        timeline.append({"event": "perimeter", "tail": peri[-6:]})
        if any(e.get("event") == "tf_perimeter" for e in peri):
            winning_policy = {"kind": "perimeter_walk", "detail": peri}
            return _pack(
                result="LIVE",
                reason="perimeter",
                timeline=timeline,
                free_summary=free_summary,
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=free_cells,
                tag=tag,
                env=env,
            )
        negatives.append({"trial": "perimeter", "log": peri, "tf02": False})

        # --- Phase 5: push blocks ---
        # Re-center toward east so we can reach stands
        for _ in range(100):
            s = read_snapshot(env.get_ram())
            if s.screen != ROOM_0D:
                break
            if s.link_x >= 190:
                break
            env.step(nes_action("RIGHT") if s.mode == PLAY_MODE else nes_idle_action())
        pushes = _push_trials(env, PUSH_CENTERS, hold=100)
        timeline.append(
            {
                "event": "pushes",
                "n": len(pushes),
                "any_change": any(
                    p.get("doors_changed") or p.get("tf02") or p.get("left_room")
                    for p in pushes
                ),
            }
        )
        if any(p.get("tf02") for p in pushes):
            win = next(p for p in pushes if p.get("tf02"))
            winning_policy = {"kind": "push_block", "detail": win}
            return _pack(
                result="LIVE",
                reason="push",
                timeline=timeline,
                free_summary=free_summary,
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=free_cells,
                tag=tag,
                env=env,
            )
        negatives.append(
            {
                "trial": "push_blocks",
                "n": len(pushes),
                "doors_changed": [p for p in pushes if p.get("doors_changed")],
                "left_room": [p for p in pushes if p.get("left_room")],
                "sample": pushes[:20],
            }
        )
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_push.png")

        # --- Phase 6: bombs ---
        bombs = _bomb_trials(env, BOMB_STANDS, wait=100)
        timeline.append(
            {
                "event": "bombs",
                "n": len(bombs),
                "any_tf": any(b.get("tf02") for b in bombs),
            }
        )
        if any(b.get("tf02") for b in bombs):
            win = next(b for b in bombs if b.get("tf02"))
            winning_policy = {"kind": "bomb", "detail": win}
            return _pack(
                result="LIVE",
                reason="bomb",
                timeline=timeline,
                free_summary=free_summary,
                objects=objects_at_entry,
                negatives=negatives,
                winning_policy=winning_policy,
                free_cells=free_cells,
                tag=tag,
                env=env,
            )
        negatives.append({"trial": "bombs", "results": bombs})
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_bomb.png")

        # --- Phase 7: object re-dump + sword free cells after bombs/pushes ---
        free2 = _free_cell_bfs(env, step=step, budget_per=120)
        timeline.append({"event": "free_cells_post", "summary": free2.get("summary")})
        s = read_snapshot(env.get_ram())
        final = _sample(s, env.get_ram(), event="final")
        timeline.append(final)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_end.png")

        # Hypothesis for encode
        cells = free_cells.get("cells") or []
        north_cells = [c for c in cells if c[1] <= 110]
        west_cells = [c for c in cells if c[0] <= 80]
        center_reach = any(
            abs(c[0] - 120) <= 16 and abs(c[1] - 141) <= 16 for c in cells
        )
        hint = (
            "Room 0x0d free geometry is perimeter-only (diamond solids block center). "
            f"North free cells={len(north_cells)}, west={len(west_cells)}, "
            f"center(120,141) reachable={center_reach}. "
            "Green TF sprite is on north band in screenshots; systematic walk-to + "
            "push + bomb did not set tf&0x02. Next: verify whether RoomItemId 0x1B "
            "spawns as object with x/y after a latent timer/mode, try sword-touch, "
            "or check if correct TF room is elsewhere (UP from 0x0e bomb residual / "
            "different post-boss door). Re-check HC collected (containers) before exit."
        )

        return _pack(
            result="PARTIAL" if free_cells.get("n_quantized", 0) > 0 else "FAIL",
            reason="no_tf_collect",
            timeline=timeline,
            free_summary=free_summary,
            objects=objects_at_entry,
            negatives=negatives,
            winning_policy=None,
            free_cells=free_cells,
            free_cells_post=free2,
            next_encode_hint=hint,
            tag=tag,
            env=env,
        )
    finally:
        env.close()


def _pack(
    *,
    result: str,
    reason: str,
    timeline: list,
    free_summary: str,
    objects: list,
    negatives: list,
    winning_policy,
    tag: str,
    env,
    free_cells: dict | None = None,
    free_cells_post: dict | None = None,
    next_encode_hint: str = "",
) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf = int(read_u8(ram, ADDR_TRIFORCE))
    out = {
        "result": result if not (tf & 0x02) else "LIVE",
        "triforce_bit_0x02": bool(tf & 0x02),
        "room": f"0x{snap.screen:02x}",
        "reason": reason,
        "free_cells_summary": free_summary,
        "free_cells": free_cells,
        "free_cells_post": free_cells_post,
        "objects": objects,
        "objects_final": _objs(snap),
        "winning_policy": winning_policy,
        "negatives": negatives,
        "next_encode_hint": next_encode_hint
        or (
            "Encode north-band walk if LIVE; else expand bomb/push residual map."
        ),
        "final": _sample(snap, ram, event="pack_final"),
        "timeline": timeline,
        "bead": "rr-n5i",
        "track": "assisted",
        "evidence": [
            f"recordings/{tag}.json",
            f"recordings/{tag}_entry.png",
            f"recordings/{tag}_end.png",
        ],
    }
    if tf & 0x02:
        out["result"] = "LIVE"
        out["next_encode_hint"] = (
            "LIVE path found — promote policy into run_level2_dodongo "
            f"_collect_and_tf / new Level2TriforceController. policy={winning_policy!r}"
        )
    write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
    # Also write the required deliverable name
    write_json_report(RECORDINGS_DIR / "l2_0d_tf_reach.json", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2_0E")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--tag", default="l2_0d_tf_reach")
    p.add_argument("--save-mid", action="store_true")
    p.add_argument(
        "--policy-only",
        action="store_true",
        help="Only run south-band TF waypoints (no full BFS/push/bomb map)",
    )
    args = p.parse_args()
    infinite = not args.no_infinite_life
    start = args.from_state
    if args.policy_only and start == "Level2_0E":
        # Prefer post-boss TF room when available
        ck = GAME_DIR / "custom_integrations" / GAME / "Level2_0D_PostBoss.state"
        if ck.exists():
            start = "Level2_0D_PostBoss"
    out = run_probe(
        start_state=start,
        infinite_life=infinite,
        step=args.step,
        tag=args.tag,
        save_mid=args.save_mid,
        policy_only=args.policy_only,
    )
    print(
        json.dumps(
            {
                "result": out["result"],
                "tf02": out["triforce_bit_0x02"],
                "room": out["room"],
                "reason": out.get("reason"),
                "free": out.get("free_cells_summary"),
                "policy": out.get("winning_policy"),
                "hint": out.get("next_encode_hint"),
                "neg_n": len(out.get("negatives") or []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
