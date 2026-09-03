"""Recon: fixture -> 0x6B LEFT -> 0x6A LEFT -> 0x69, then carefully map 0x69's
NORTH wall (precise x-sweep on the y=93 band: open-door notch vs solid) and
bomb-sweep the WEST wall + any solid NORTH span.

Source: 0x69 == GORIYA_BOMB_HUB -- bomb LEFT (KEESE_TRAPS) and/or bomb UP
(DODONGOS_UPGRADE) is the candle-path branch after the Stalfos-key dead-end.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room69_branch.py --tag 69_branch_v2
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.ops import ensure_bomb, poke_bombs
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_69, ROOM_6A, ROOM_6B = 0x69, 0x6A, 0x6B
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if a:
        a.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "screen_int": int(s.screen),
        "mode": int(s.mode), "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
    }


def _west_traverse(env, a, room, next_room, f, budget=2200):
    """W exit mirror of Room6AEastController."""
    TOP_Y, DOOR_Y = 93, 141
    EAST_COL, WEST_COL = 200, 36
    phase = "off_mouth"
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) == next_room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        if int(s.screen) not in (room, next_room):
            return f, False
        if s.transitioning:
            _step(env, a, "LEFT", f); f += 1; continue
        x, y = int(s.link_x), int(s.link_y)
        if phase == "off_mouth":
            btn = "LEFT" if x > EAST_COL + 8 else "UP"
            if x <= EAST_COL + 8:
                phase = "rise"
        elif phase == "rise":
            btn = "UP" if y > TOP_Y + 4 else "LEFT"
            if y <= TOP_Y + 4:
                phase = "cross"
        elif phase == "cross":
            if x > WEST_COL + 4:
                btn = "LEFT" if abs(y - TOP_Y) <= 6 else ("UP" if y > TOP_Y else "DOWN")
            else:
                phase = "drop"; btn = "DOWN"
        elif phase == "drop":
            btn = ("DOWN" if y < DOOR_Y else "UP") if abs(y - DOOR_Y) > 4 else "LEFT"
            if abs(y - DOOR_Y) <= 4:
                phase = "push"
        else:
            btn = "LEFT" if abs(y - DOOR_Y) <= 6 else ("UP" if y > DOOR_Y else "DOWN")
        _step(env, a, btn, f); f += 1
    return f, False


def _reach(env, a, room, tx, ty, f, budget=240):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if (tx is None or abs(x - tx) <= 2) and (ty is None or abs(y - ty) <= 3):
            return f, True
        if ty is not None and abs(y - ty) > 3 and (tx is None or abs(x - tx) <= 10):
            _step(env, a, "UP" if y > ty else "DOWN", f)
        elif tx is not None and abs(x - tx) > 2:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        elif ty is not None and abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        else:
            return f, True
        f += 1
    return f, False


def _hold(env, a, room, btn, f, budget=60):
    """Hold btn; return (frame, transitioned_screen_or_None, end_xy, tiles)."""
    tiles = []
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, int(s.screen), [int(s.link_x), int(s.link_y)], tiles
        tiles.append(int(s.colliding_tile))
        _step(env, a, btn, f); f += 1
    e = _s(env)
    return f, None, [int(e.link_x), int(e.link_y)], tiles


def _bomb_push(env, a, room, sx, sy, face, f):
    f, _ = _reach(env, a, room, sx, sy, f)
    for _ in range(4):
        ensure_bomb(env); _step(env, a, face, f); f += 1
    ensure_bomb(env)
    env.step(nes_action(face, "B"))
    if a:
        a.apply_env(env, frame=f)
    f += 1
    for _ in range(7):
        _step(env, a, OPP[face], f); f += 1
    for _ in range(110):
        _step(env, a, None, f); f += 1
    for _ in range(150):
        s = _s(env)
        if int(s.screen) != room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, int(s.screen), [int(s.link_x), int(s.link_y)]
        x = int(s.link_x)
        b = face
        if face in ("UP", "DOWN") and abs(x - sx) > 3:
            b = "LEFT" if x > sx else "RIGHT"
        _step(env, a, b, f); f += 1
    return f, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="69_branch_v2")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        f, ok = _west_traverse(env, a, ROOM_6B, ROOM_6A, f)
        if ok:
            for _ in range(30):
                _step(env, a, None, f); f += 1
            f, ok = _west_traverse(env, a, ROOM_6A, ROOM_69, f)
        out["reached_69"] = ok
        if not ok:
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL 69", out["end"]); return
        for _ in range(120):
            _step(env, a, None, f); f += 1
        out["at_69"] = _glance(env)
        print("0x69", out["at_69"])

        # NORTH wall: precise x-sweep on the y=93 band
        north = []
        for tx in range(104, 157, 2):
            f, _ = _reach(env, a, ROOM_69, tx, 93, f)
            f, trans, xy, tiles = _hold(env, a, ROOM_69, "UP", f, budget=45)
            rec = {"x": tx, "end_xy": xy, "min_tile": min(tiles) if tiles else None,
                   "screen": f"0x{trans:02x}" if trans else "0x69"}
            north.append(rec); print("N", rec)
            if trans is not None:
                for _ in range(140):
                    _step(env, a, None, f); f += 1
                out["north_dest"] = _glance(env)
                print("  NORTH DEST", out["north_dest"])
                break
            f, _ = _reach(env, a, ROOM_69, tx, 141, f, budget=90)
        out["north_scan"] = north

        # WEST wall bomb-sweep (if no north door found)
        wsweep = []
        if "north_dest" not in out:
            for by in (141, 109, 93, 125, 77):
                poke_bombs(env, 8)
                if int(_s(env).screen) != ROOM_69:
                    break
                f, trans, xy = _bomb_push(env, a, ROOM_69, 44, by, "LEFT", f)
                rec = {"y": by, "dest": f"0x{trans:02x}" if trans else None, "at": xy}
                wsweep.append(rec); print("bombW", rec)
                if trans is not None:
                    for _ in range(140):
                        _step(env, a, None, f); f += 1
                    out["west_dest"] = _glance(env)
                    break
                f, _ = _reach(env, a, ROOM_69, 120, 141, f)
        out["west_bomb_sweep"] = wsweep

        # NORTH wall bomb-sweep (if still nothing)
        nsweep = []
        if "north_dest" not in out and "west_dest" not in out:
            for bx in (120, 128, 112, 136, 104, 144):
                poke_bombs(env, 8)
                if int(_s(env).screen) != ROOM_69:
                    break
                f, trans, xy = _bomb_push(env, a, ROOM_69, bx, 97, "UP", f)
                rec = {"x": bx, "dest": f"0x{trans:02x}" if trans else None, "at": xy}
                nsweep.append(rec); print("bombN", rec)
                if trans is not None:
                    for _ in range(140):
                        _step(env, a, None, f); f += 1
                    out["north_bomb_dest"] = _glance(env)
                    break
                f, _ = _reach(env, a, ROOM_69, 120, 141, f)
        out["north_bomb_sweep"] = nsweep

        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
