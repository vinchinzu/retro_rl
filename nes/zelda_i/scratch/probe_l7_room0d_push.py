"""Recon: 0x0D TIP_OF_NOSE — wallmasters + mid-right 0x68 stairs.

Pin: Level7Interior0DReconFixture — L7 play 0x0D (32,141) W mouth,
wallmaster 0x27, 0x68, bombs 6, candle 2. Do not get grabbed.

Graph: push mid-right block (not the 0x1A left block), stairs DOWN to
NOSE_CELLAR, far-side stairs to PRE_BOSS.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \\
        nes/zelda_i/scratch/probe_l7_room0d_push.py --tag 0d_map_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint, is_off_wall
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.ops import ensure_bomb
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level9.stairs import PUSHABLE_BLOCK
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM = 0x0D
CELLAR_MODES = {9, 10, 11}
WALLMASTER = 0x27
# Plus-corner 0x27 never move / take no sword (statues, not the 5 spawners).
STATUE_XY = frozenset({(128, 125), (128, 157), (160, 125), (160, 157)})
# Stay off west-mouth grab x=32 and east-wall grab x=208.
# x=32 ANY y grabs (0d_map_v4: bubble shove 58,117 → 32,117 → 0x79).
INLAND_X = (64, 192)
INLAND_Y = (109, 173)
# West of the plus first (y=141 gap), then west-wall lure off the door row.
LURE = (
    (96, 141),
    (64, 117),
    (96, 117),
    (64, 165),
    (96, 165),
    (80, 141),
)
LURE_LINGER = 90
HOME = (96, 141)


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    if btn is None:
        env.step(nes_idle_action())
    elif isinstance(btn, tuple):
        env.step(nes_action(*btn))
    else:
        env.step(nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _blocks(s):
    return [
        {"slot": int(o.slot), "x": int(o.x), "y": int(o.y), "hp": int(o.hp)}
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) == PUSHABLE_BLOCK
    ]


def _wallmasters(s):
    return [
        o
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) == WALLMASTER
    ]


def _inland_override(x: int, y: int, btn):
    """Force off the grab columns. x=32 any y grabs; resist bubble shove."""
    if x < 64:
        return "RIGHT"
    if x >= 200:
        return "LEFT"
    if y <= 96:
        return "DOWN"
    if y >= 189:
        return "UP"
    if btn == "LEFT" and x <= INLAND_X[0]:
        return "RIGHT"
    if btn == "RIGHT" and x >= INLAND_X[1]:
        return "LEFT"
    return btn


def _live_spawners(s, last_pos: dict[int, tuple[int, int]]):
    """Killable wallmasters: moved off statue cells and off the wall park."""
    live = []
    parked = []
    for o in _wallmasters(s):
        xy = (int(o.x), int(o.y))
        slot = int(o.slot)
        prev = last_pos.get(slot)
        last_pos[slot] = xy
        if int(o.hp) <= 0:
            continue
        if xy in STATUE_XY and (prev is None or prev == xy):
            continue
        # South-wall crawlers sit at x=12..16 (is_off_wall needs x>16).
        in_floor = 12 <= int(o.x) <= 200 and 80 <= int(o.y) <= 200
        if int(o.state) != 0 and in_floor:
            live.append(o)
            continue
        if not is_off_wall(o):
            parked.append(o)
            continue
        live.append(o)
    return live, parked


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "level": int(s.level),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "objects": [
            {
                "slot": int(o.slot),
                "type": f"0x{int(o.type_id):02x}",
                "hp": int(o.hp),
                "state": int(o.state),
                "facing": int(o.facing),
                "xy": [int(o.x), int(o.y)],
            }
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        ],
        "blocks": _blocks(s),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "sword": int(s.sword),
        "triforce": int(s.triforce),
    }


def _reach(env, a, tx, ty, f, budget=360):
    last = None
    stuck = 0
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False, [int(s.link_x), int(s.link_y)]
        if int(s.mode) in CELLAR_MODES:
            return f, True, [int(s.link_x), int(s.link_y)]
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True, [x, y]
        xy = (x, y)
        if xy == last:
            stuck += 1
            if stuck >= 50:
                return f, False, [x, y]
        else:
            stuck = 0
            last = xy
        if abs(y - ty) > 3:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        _step(env, a, btn, f)
        f += 1
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


def _grid() -> OccupancyGrid:
    """Inland of grab columns. Pre-block plus statues and the 0x68."""
    g = OccupancyGrid(xmin=48, xmax=200, ymin=93, ymax=189)
    for cx, cy in STATUE_XY:
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                g.blocked.add((cx + dx, cy + dy))
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            g.blocked.add((192 + dx, 144 + dy))
    return g


def _occ(env, a, tx, ty, f, budget=1800, tag="", walker=None):
    walker = walker or OccupancyWalker(goal=(tx, ty), grid=_grid())
    walker.goal = (tx, ty)
    walker.path = None
    stood = 0
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False, [int(s.link_x), int(s.link_y)], walker.misses
        if int(s.mode) in CELLAR_MODES:
            return f, True, [int(s.link_x), int(s.link_y)], walker.misses
        xy = (int(s.link_x), int(s.link_y))
        walker.observe(xy)
        if abs(xy[0] - tx) <= 3 and abs(xy[1] - ty) <= 3:
            return f, True, list(xy), walker.misses
        direction = walker.next_dir(xy)
        if direction is None:
            stood += 1
            if stood >= 20:
                return f, False, list(xy), walker.misses
            _step(env, a, None, f)
        else:
            stood = 0
            _step(env, a, direction, f)
        f += 1
        if tag and i > 0 and i % 250 == 0:
            save_rgb_png(
                env.render(),
                RECORDINGS_DIR / f"{tag}_occ_{i}_{xy[0]}_{xy[1]}.png",
            )
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)], walker.misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0d_map_v1")
    ap.add_argument("--from-state", default="Level7Interior0DReconFixture")
    ap.add_argument("--push", default="")
    ap.add_argument("--which", default="right", choices=["left", "right"])
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--stairs", action="store_true")
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--clip", default="")
    ap.add_argument("--bomb-center", action="store_true")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "push": args.push, "which": args.which}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        if args.map:
            # East pocket (176,141) is boxed north at y=117 tile 179.
            # Cut west through the plus gap at y=141 first.
            f, ok, xy = _reach(env, a, 96, 141, f, budget=400)
            print("WEST GAP", ok, xy, "tile", int(_s(env).colliding_tile))
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_westgap.png"
            )
            targets = (
                (96, 165),
                (96, 189),
                (64, 189),
                (64, 165),
                (64, 141),
                (56, 117),
                (80, 117),
                (176, 189),
                (176, 165),
            )
            hops = []
            for ti, (tx, ty) in enumerate(targets):
                f, ok, xy = _reach(env, a, tx, ty, f, budget=500)
                rec = {
                    "target": [tx, ty],
                    "ok": ok,
                    "xy": xy,
                    "tile": int(_s(env).colliding_tile),
                    "screen": f"0x{int(_s(env).screen):02x}",
                    "wm": [
                        {
                            "slot": int(o.slot),
                            "hp": int(o.hp),
                            "state": int(o.state),
                            "xy": [int(o.x), int(o.y)],
                            "off": is_off_wall(o),
                        }
                        for o in _wallmasters(_s(env))
                    ],
                }
                print("MAP WP", rec)
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_wp{ti}_{xy[0]}_{xy[1]}.png",
                )
                if int(_s(env).screen) != ROOM:
                    rec["grabbed"] = _glance(env)
                    hops.append(rec)
                    break
                for linger_i in range(180):
                    _step(env, a, None, f)
                    f += 1
                    if linger_i % 60 == 0:
                        wm = [
                            {
                                "slot": int(o.slot),
                                "hp": int(o.hp),
                                "state": int(o.state),
                                "xy": [int(o.x), int(o.y)],
                            }
                            for o in _wallmasters(_s(env))
                        ]
                        print("LINGER", ti, linger_i, _glance(env)["xy"], "wm", wm)
                    if int(_s(env).screen) != ROOM:
                        rec["grabbed"] = _glance(env)
                        print("GRABBED", rec["grabbed"])
                        break
                hops.append(rec)
                if int(_s(env).screen) != ROOM:
                    break
            out["map"] = hops
            out["end_map"] = _glance(env)
            print("MAP END", out["end_map"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_map.png")
        if args.stairs:
            # Plus-center north to y=127, then east along that band.
            targets = (
                (152, 141),
                (152, 127),
                (176, 127),
                (192, 127),
                (200, 127),
                (200, 117),
                (200, 109),
                (200, 96),
                (192, 117),
                (176, 117),
            )
            for ti, (tx, ty) in enumerate(targets):
                f, ok, xy = _reach(env, a, tx, ty, f, budget=500)
                s = _s(env)
                rec = {
                    "target": [tx, ty],
                    "ok": ok,
                    "xy": xy,
                    "tile": int(s.colliding_tile),
                    "mode": int(s.mode),
                    "screen": f"0x{int(s.screen):02x}",
                }
                print("STAIRS", rec)
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_st{ti}_{xy[0]}_{xy[1]}.png",
                )
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR", out["cellar"])
                    break
                for _ in range(20):
                    _step(env, a, None, f)
                    f += 1
                    if int(_s(env).mode) in CELLAR_MODES:
                        out["cellar"] = _glance(env)
                        print("CELLAR", out["cellar"])
                        break
                if out.get("cellar"):
                    break
        if args.clear:
            for _ in range(120):
                s = _s(env)
                if int(s.screen) != ROOM or int(s.link_x) >= 120:
                    break
                _step(env, a, "RIGHT", f)
                f += 1
            # Nudge west wall (spawn) then retreat inland (slash). x=32 grabs.
            nudges = ((52, 117), (48, 117))
            nudge_i = 0
            phase = "to_nudge"
            phase_n = 0
            to_stuck = 0
            to_xy: tuple[int, int] | None = None
            last_pos: dict[int, tuple[int, int]] = {}
            kills = 0
            seen_hp: dict[int, int] = {}
            last_kills = 0
            for i in range(18000):
                s = _s(env)
                if int(s.screen) != ROOM:
                    out["grabbed"] = _glance(env)
                    print("GRABBED", out["grabbed"])
                    break
                for o in _wallmasters(s):
                    slot = int(o.slot)
                    hp = int(o.hp)
                    prev_hp = seen_hp.get(slot)
                    if prev_hp is not None and hp < prev_hp:
                        print("HIT", slot, prev_hp, "->", hp, "xy",
                              [int(o.x), int(o.y)], "f", f)
                    if prev_hp is not None and prev_hp > 0 and hp <= 0:
                        kills += 1
                        print("KILL", slot, "kills", kills, "f", f)
                    seen_hp[slot] = hp
                gone = [
                    slot for slot, hp in list(seen_hp.items())
                    if hp > 0 and slot not in {
                        int(o.slot) for o in _wallmasters(s)
                    }
                ]
                for slot in gone:
                    kills += 1
                    seen_hp[slot] = 0
                    print("DESPAWN", slot, "kills", kills, "f", f)
                just_killed = kills > last_kills
                if just_killed:
                    last_kills = kills
                    phase = "retreat"
                    phase_n = 0
                    print("PEEL AFTER KILL", kills,
                          [int(s.link_x), int(s.link_y)])
                live, parked = _live_spawners(s, last_pos)
                incoming = [o for o in parked if int(o.state) != 0]
                x, y = int(s.link_x), int(s.link_y)
                if int(s.room_all_dead) != 0 and not live and not incoming:
                    print("ROOM_ALL_DEAD", i, [x, y], "kills", kills)
                    break
                btn: str | tuple[str, ...] | None
                if x < 44:
                    btn = "RIGHT"
                    phase = "retreat"
                    phase_n = 0
                elif live and not just_killed:
                    phase = "fight"
                    tgt = nearest_enemy(s.link_x, s.link_y, live)
                    dx = int(tgt.x) - int(s.link_x)
                    dy = int(tgt.y) - int(s.link_y)
                    stand_x = max(48, int(tgt.x) + 18)
                    if x < 48:
                        btn = "RIGHT"
                    elif abs(y - int(tgt.y)) > 10:
                        btn = "DOWN" if y < int(tgt.y) else "UP"
                    elif abs(x - stand_x) > 3:
                        btn = "RIGHT" if x < stand_x else "LEFT"
                    else:
                        face = (
                            "LEFT" if dx < 0 else "RIGHT" if dx > 0 else (
                                "DOWN" if dy > 0 else "UP"
                            )
                        )
                        btn = (face, "A") if (i % 8) < 5 else face
                    if i % 30 == 0:
                        print("FIGHT", i, [x, y], "tgt",
                              [int(tgt.x), int(tgt.y)], "hp", int(tgt.hp))
                elif incoming:
                    phase = "retreat"
                    if x < HOME[0] - 4:
                        btn = "RIGHT"
                    elif abs(y - HOME[1]) > 4:
                        btn = "UP" if y > HOME[1] else "DOWN"
                    else:
                        btn = ("LEFT", "A") if (i % 8) < 5 else "LEFT"
                    if i % 20 == 0:
                        print("INCOMING", i, [x, y], phase,
                              [(int(o.slot), int(o.x), int(o.y), int(o.state))
                               for o in incoming])
                elif phase == "to_nudge":
                    tx, ty = nudges[nudge_i % len(nudges)]
                    if abs(x - tx) <= 6 and abs(y - ty) <= 6:
                        phase = "nudge"
                        phase_n = 0
                        btn = "LEFT"
                        print("NUDGE", i, [x, y], "target", [tx, ty])
                    elif abs(y - ty) > 4:
                        btn = "UP" if y > ty else "DOWN"
                    else:
                        btn = "LEFT" if x > tx else "RIGHT"
                    if (x, y) == to_xy:
                        to_stuck += 1
                        if to_stuck >= 50:
                            print("NUDGE STUCK", [x, y], "skip")
                            nudge_i += 1
                            to_stuck = 0
                    else:
                        to_stuck = 0
                        to_xy = (x, y)
                elif phase == "nudge":
                    phase_n += 1
                    if x > 42:
                        btn = "LEFT"
                    elif x < 40:
                        btn = "RIGHT"
                    else:
                        btn = ("LEFT", "A") if (phase_n % 8) < 5 else "LEFT"
                    if phase_n >= 72 or x <= 40:
                        phase = "retreat"
                        phase_n = 0
                        print("RETREAT", i, [x, y])
                else:
                    phase_n += 1
                    if abs(x - HOME[0]) > 4:
                        btn = "RIGHT" if x < HOME[0] else "LEFT"
                    elif abs(y - HOME[1]) > 4:
                        btn = "UP" if y > HOME[1] else "DOWN"
                    else:
                        btn = None
                    if phase_n >= 40:
                        nudge_i += 1
                        phase = "to_nudge"
                        phase_n = 0
                if x < 44:
                    btn = "RIGHT"
                elif phase not in ("nudge", "to_nudge", "fight"):
                    if isinstance(btn, tuple):
                        face = _inland_override(x, y, btn[0])
                        btn = (face, "A") if face == btn[0] else face
                    else:
                        btn = _inland_override(x, y, btn)
                _step(env, a, btn, f)
                f += 1
                if i > 0 and i % 400 == 0:
                    wm = [
                        {
                            "slot": int(o.slot),
                            "hp": int(o.hp),
                            "st": int(o.state),
                            "xy": [int(o.x), int(o.y)],
                        }
                        for o in _wallmasters(s)
                    ]
                    print(
                        "CLEAR", i, [x, y], "phase", phase, phase_n,
                        "live", len(live), "in", len(incoming),
                        "dead", int(s.room_all_dead), "kills", kills,
                        "wm", wm,
                    )
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_clear_{i}.png",
                    )
            out["kills"] = kills
            out["after_clear"] = _glance(env)
            print("AFTER CLEAR", out["after_clear"], "kills", kills)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_clear.png")

        if args.bomb_center:
            f, ok, xy = _reach(env, a, 144, 141, f)
            print("BOMB STAND", ok, xy)
            ensure_bomb(env)
            env.step(nes_action("B"))
            a.apply_env(env, frame=f)
            f += 1
            for _ in range(8):
                _step(env, a, "LEFT", f)
                f += 1
            for _ in range(120):
                _step(env, a, None, f)
                f += 1
            out["after_bomb"] = _glance(env)
            print("AFTER BOMB", out["after_bomb"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_bomb.png")
        blocks = _blocks(_s(env))
        out["blocks"] = blocks
        print("BLOCKS", blocks)
        if args.push and blocks:
            pick = max(blocks, key=lambda b: b["x"]) if args.which == "right" else min(
                blocks, key=lambda b: b["x"]
            )
            out["target_block"] = pick
            print("TARGET BLOCK", pick)
            face = args.push
            if face == "UP":
                stand = (int(pick["x"]), int(pick["y"]) + 16)
            elif face == "DOWN":
                stand = (int(pick["x"]), int(pick["y"]) - 16)
            elif face == "LEFT":
                stand = (int(pick["x"]) + 16, int(pick["y"]))
            else:
                stand = (int(pick["x"]) - 16, int(pick["y"]))
            if args.occ:
                f, ok, xy, misses = _occ(
                    env, a, stand[0], stand[1], f, tag=args.tag
                )
                print("OCC", ok, xy, "misses", misses, "tile",
                      int(_s(env).colliding_tile))
            else:
                if face == "UP":
                    approach = ((32, 189), (184, 189), (184, 162), stand)
                elif face == "RIGHT":
                    approach = ((160, 141), (176, 144), stand)
                elif face == "DOWN":
                    approach = ((32, 125), (176, 125), (192, 128), stand)
                else:
                    approach = ((32, 189), (208, 189), (208, 141), stand)
                for wx, wy in approach:
                    f, ok, xy = _reach(env, a, wx, wy, f)
                    print("WP", [wx, wy], ok, xy, "tile",
                          int(_s(env).colliding_tile))
                    if int(_s(env).screen) != ROOM:
                        break
            out["at_stand"] = {"ok": ok, "xy": xy, "glance": _glance(env)}
            print("STAND", out["at_stand"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stand.png")
            by = int(pick["y"])
            for _ in range(40):
                s = _s(env)
                if abs(int(s.link_y) - by) <= 1:
                    break
                _step(env, a, "DOWN" if int(s.link_y) < by else "UP", f)
                f += 1
            print("ALIGNED", _glance(env)["xy"], "block", _blocks(_s(env)))
            if args.clip:
                clip_btn = tuple(
                    p.strip() for p in args.clip.upper().split("+") if p.strip()
                )
                for i in range(80):
                    s = _s(env)
                    if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                        break
                    bys = _blocks(s)
                    if bys and (
                        bys[0]["x"] != int(pick["x"]) or bys[0]["y"] != int(pick["y"])
                    ):
                        print("MOVED", bys, "f", f)
                        break
                    _step(env, a, clip_btn, f)
                    f += 1
                    if i % 20 == 0:
                        print("CLIP", _glance(env)["xy"], "tile",
                              int(_s(env).colliding_tile), "blocks", bys)
            ox, oy = int(pick["x"]), int(pick["y"])
            moved = False
            for i in range(80):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES:
                    break
                if int(s.screen) != ROOM:
                    break
                bys = _blocks(s)
                if bys and (
                    abs(int(bys[0]["x"]) - ox) >= 16
                    or abs(int(bys[0]["y"]) - oy) >= 16
                ):
                    print("MOVED", bys, "f", f, "link", _glance(env)["xy"])
                    moved = True
                    break
                _step(env, a, face, f)
                f += 1
                if i % 20 == 0:
                    print("PUSHING", _glance(env)["xy"], "blocks", bys)
            out["after_push"] = _glance(env)
            print("AFTER PUSH", out["after_push"], "moved", moved)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")
            for tx, ty in (
                (ox, oy),
                (ox, 141),
                (ox - 16, oy),
                (176, 144),
                (176, 141),
            ):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                if int(_s(env).screen) != ROOM:
                    break
                f, _, xy = _reach(env, a, tx, ty, f, budget=200)
                rec = {
                    "target": [tx, ty],
                    "xy": xy,
                    "mode": int(_s(env).mode),
                    "screen": f"0x{int(_s(env).screen):02x}",
                }
                print("STAIR HUNT", rec)
                if int(_s(env).mode) in CELLAR_MODES:
                    break
            if int(_s(env).mode) in CELLAR_MODES:
                out["cellar"] = _glance(env)
                print("CELLAR", out["cellar"])
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png"
                )
                for _ in range(200):
                    s = _s(env)
                    if int(s.mode) == PLAY_MODE and int(s.screen) != ROOM:
                        break
                    _step(env, a, None, f)
                    f += 1
                out["cellar_settled"] = _glance(env)
                print("CELLAR SETTLED", out["cellar_settled"])

        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        if args.save_fixture and int(_s(env).screen) not in {0x79}:
            path = save_state(env, GAME_DIR, GAME, args.save_fixture)
            src = state_path(GAME_DIR, GAME, args.from_state)
            write_state_provenance(
                path,
                source_state_path=src if src.exists() else None,
                request={
                    "bead": "rr-8t4.3",
                    "phase": "level7_tip_of_nose_recon",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "development_only": True,
                    "fixture_writes": [],
                    "notes": [
                        f"Derived from {args.from_state}: 0x0D {args.which} "
                        f"0x68 push {args.push}. No Candle/TF/door writes.",
                        "UnlimitedHealthAssist traversal aid only.",
                    ],
                },
                selected_trial={
                    "ok": True,
                    "state": compact_snapshot(_s(env)),
                    "glance": end,
                },
                natural_entry=False,
            )
            out["saved_fixture"] = str(path)
            print("saved", path)
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
