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
CELLAR_MODES = {9, 10, 11, 16}
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


def _reach(env, a, tx, ty, f, budget=360, tol=3):
    last = None
    stuck = 0
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False, [int(s.link_x), int(s.link_y)]
        if int(s.mode) in CELLAR_MODES:
            return f, True, [int(s.link_x), int(s.link_y)]
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return f, True, [x, y]
        xy = (x, y)
        if xy == last:
            stuck += 1
            if stuck >= 50:
                return f, False, [x, y]
        else:
            stuck = 0
            last = xy
        if abs(y - ty) > tol:
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
    ap.add_argument("--bomb-stair", action="store_true")
    ap.add_argument("--occ", action="store_true")
    ap.add_argument("--clip", default="")
    ap.add_argument("--bomb-center", action="store_true")
    ap.add_argument("--ne", action="store_true")
    ap.add_argument("--dump-tiles", action="store_true")
    ap.add_argument("--race", action="store_true")
    ap.add_argument("--east-col", action="store_true")
    ap.add_argument("--poke-warp", action="store_true")
    ap.add_argument("--north-band", action="store_true")
    ap.add_argument("--north-after-push", action="store_true")
    ap.add_argument("--west-north", action="store_true")
    ap.add_argument("--wn-push", action="store_true")
    ap.add_argument("--wn-bomb", action="store_true")
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
        if args.west_north:
            # West wall corridor: 0x2b residuals patrol x=32 y~93-123, so the
            # west column may connect the south floor to the y=93 top band and
            # the NE staircase (tiles 0x70-0x73 at x~192 y<=101).
            wp = [
                (48, 141), (44, 125), (44, 109), (44, 101), (44, 93),
                (64, 93), (96, 93), (128, 93), (160, 93), (176, 93),
                (184, 93), (192, 93),
            ]
            hops = []
            for tx, ty in wp:
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    break
                f, ok, xy = _reach(env, a, tx, ty, f, budget=400, tol=1)
                rec = {
                    "target": [tx, ty], "ok": ok, "xy": xy,
                    "tile": int(_s(env).colliding_tile),
                    "mode": int(_s(env).mode),
                    "screen": f"0x{int(_s(env).screen):02x}",
                }
                print("WN WP", rec)
                hops.append(rec)
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_wn_{tx}_{ty}.png"
                )
                if args.wn_push and ok and abs(xy[0] - 192) <= 3 and xy[1] <= 96:
                    # at the west face of the (192,144) block? no — try pushing
                    # RIGHT here in case the staircase needs the block pushed
                    # from the top band.
                    for _ in range(20):
                        _step(env, a, "RIGHT", f)
                        f += 1
                if args.wn_bomb and ok and xy[1] >= 105 and xy[1] <= 130:
                    for _ in range(8):
                        _step(env, a, "UP", f)
                        f += 1
                    ensure_bomb(env)
                    env.step(nes_action("B"))
                    a.apply_env(env, frame=f)
                    f += 1
                    for _ in range(8):
                        _step(env, a, "DOWN", f)
                        f += 1
                    for _ in range(90):
                        _step(env, a, None, f)
                        f += 1
                    print("WN BOMB AFTER", [tx, ty], _glance(env)["xy"])
                    f, ok, xy = _reach(env, a, tx, ty - 16, f, budget=200, tol=2)
                    print("WN BOMB WALK", ok, xy, "mode", int(_s(env).mode))
                if ok and abs(xy[0] - tx) <= 2 and abs(xy[1] - ty) <= 2 and ty <= 96:
                    for btn in ("UP", "RIGHT", None, "UP", None):
                        _step(env, a, btn, f)
                        f += 1
                        if int(_s(env).mode) in CELLAR_MODES:
                            print("WN CELLAR", [tx, ty], btn, _glance(env))
                            break
                    for _ in range(30):
                        if int(_s(env).mode) in CELLAR_MODES:
                            break
                        _step(env, a, None, f)
                        f += 1
                if int(_s(env).mode) in CELLAR_MODES:
                    break
            out["wn_hops"] = hops
            if int(_s(env).mode) in CELLAR_MODES:
                out["cellar"] = _glance(env)
                print("WN CELLAR REACHED", out["cellar"])
                for _ in range(240):
                    s = _s(env)
                    if int(s.mode) == PLAY_MODE and int(s.screen) != ROOM:
                        break
                    _step(env, a, None, f)
                    f += 1
                out["cellar_settled"] = _glance(env)
                print("WN CELLAR SETTLED", out["cellar_settled"])
            out["after_wn"] = _glance(env)
            print("AFTER WN", out["after_wn"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_wn_final.png")
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
            # CheckWarps: x multiple of 16, y = 16k+13 (93/109/125/141).
            # Plus-center and door-row first; NE hole is blocked by the
            # parked 0x68 at (208,96).
            targets = (
                (176, 141),
                (192, 141),
                (192, 136),
            )
            for ti, (tx, ty) in enumerate(targets):
                f, ok, xy = _reach(env, a, tx, ty, f, budget=500, tol=0)
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
                idle = 60 if (xy[0] % 16 == 0) else 20
                for _ in range(idle):
                    _step(env, a, None, f)
                    f += 1
                    if int(_s(env).mode) in CELLAR_MODES:
                        out["cellar"] = _glance(env)
                        print("CELLAR IDLE", _glance(env))
                        break
                if out.get("cellar"):
                    break
                # Door-clip toward the visible NE hole.
                if ok and tx >= 176:
                    for i in range(24):
                        _step(env, a, ("RIGHT", "UP"), f)
                        f += 1
                        ss = _s(env)
                        if int(ss.mode) in CELLAR_MODES:
                            out["cellar"] = _glance(env)
                            print("CELLAR CLIP", _glance(env))
                            break
                        if i % 8 == 0:
                            print("CLIP RU", _glance(env)["xy"],
                                  "tile", int(ss.colliding_tile))
                    if out.get("cellar"):
                        break
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_clip_{ti}.png",
                    )
        if args.bomb_stair:
            # From post-push NE pocket: bomb the wall between (176,125) and
            # the visible stair hole.
            spots = (
                ((176, 125), "RIGHT"),
                ((176, 117), "RIGHT"),
                ((192, 136), "UP"),
                ((192, 141), "RIGHT"),
            )
            for (tx, ty), face in spots:
                f, ok, xy = _reach(env, a, tx, ty, f, budget=400, tol=2)
                print("BOMB SPOT", [tx, ty], face, ok, xy)
                for _ in range(8):
                    _step(env, a, face, f)
                    f += 1
                ensure_bomb(env)
                env.step(nes_action("B"))
                a.apply_env(env, frame=f)
                f += 1
                retreat = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}
                for _ in range(12):
                    _step(env, a, retreat[face], f)
                    f += 1
                for _ in range(90):
                    _step(env, a, None, f)
                    f += 1
                print("AFTER BOMB", face, _glance(env))
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_bomb_{face}_{xy[0]}_{xy[1]}.png",
                )
                f, ok2, xy2 = _reach(env, a, tx + (16 if face == "RIGHT" else 0),
                                     ty - (16 if face == "UP" else 0),
                                     f, budget=200, tol=2)
                print("POST BOMB WALK", ok2, xy2, "mode", int(_s(env).mode),
                      "screen", f"0x{int(_s(env).screen):02x}")
                if int(_s(env).mode) in CELLAR_MODES:
                    out["cellar"] = _glance(env)
                    print("CELLAR", out["cellar"])
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
                    approach = ((96, 141), (176, 141), (176, 157), (184, 157), stand)
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
            started = False
            for i in range(120):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES:
                    break
                if int(s.screen) != ROOM:
                    break
                bys = _blocks(s)
                bx = int(bys[0]["x"]) if bys else ox
                byy = int(bys[0]["y"]) if bys else oy
                dxb, dyb = bx - ox, byy - oy
                if i < 40 or dxb or dyb:
                    print(
                        "PUSH", i, "link", _glance(env)["xy"],
                        "block", [bx, byy], "d", [dxb, dyb],
                    )
                # One tile is 16px. Release as soon as the slide starts so
                # Link does not walk with the block into a second push.
                if (not started) and (dxb != 0 or dyb != 0):
                    started = True
                    print("STARTED", [bx, byy], "f", f)
                    if args.race:
                        print("RACE BREAK", [bx, byy], "link", _glance(env)["xy"])
                        break
                if started and abs(dxb) >= 16 and dyb == 0:
                    print("TILE", [bx, byy], "f", f, "link", _glance(env)["xy"])
                    moved = True
                    break
                if started and dyb != 0:
                    print("Y SLIDE", [bx, byy], "f", f)
                    moved = True
                    break
                btn = None if started else face
                _step(env, a, btn, f)
                f += 1
            # Let a started slide finish without more RIGHT.
            # --race peels UP the x=176 column during the 32f slide.
            if not args.race:
                for _ in range(24):
                    bys = _blocks(_s(env))
                    if not bys:
                        break
                    bx, byy = int(bys[0]["x"]), int(bys[0]["y"])
                    if abs(bx - ox) >= 16 and byy == oy:
                        moved = True
                        print("SETTLED", [bx, byy])
                        break
                    _step(env, a, None, f)
                    f += 1
            out["after_push"] = _glance(env)
            print("AFTER PUSH", out["after_push"], "moved", moved)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")
            # CheckWarps UW: X multiple of $10, y often $10k+$D (141).
            # Stairs are visible NE after the 16px slide (block parks 208,96).
            hunt_targets = () if args.race else (
                (192, 141),
                (192, 136),
                (192, 125),
                (200, 141),
                (200, 125),
                (200, 109),
                (176, 125),
                (176, 117),
                (208, 141),
                (ox, oy),
            )
            for tx, ty in hunt_targets:
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                if int(_s(env).screen) != ROOM:
                    break
                f, ok, xy = _reach(env, a, tx, ty, f, budget=240, tol=0)
                rec = {
                    "target": [tx, ty],
                    "ok": ok,
                    "xy": xy,
                    "tile": int(_s(env).colliding_tile),
                    "mode": int(_s(env).mode),
                    "screen": f"0x{int(_s(env).screen):02x}",
                }
                print("STAIR HUNT", rec)
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_hunt_{tx}_{ty}.png",
                )
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                if ok and xy[0] % 16 == 0:
                    for _ in range(40):
                        if int(_s(env).mode) in CELLAR_MODES:
                            print("CELLAR IDLE", _glance(env))
                            break
                        _step(env, a, None, f)
                        f += 1
                for btn in (None, "UP", "DOWN", "LEFT", "RIGHT"):
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                    _step(env, a, btn, f)
                    f += 1
                    if int(_s(env).mode) in CELLAR_MODES:
                        print("CELLAR NUDGE", btn, _glance(env))
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

            if (
                args.north_after_push
                and int(_s(env).mode) not in CELLAR_MODES
                and int(_s(env).screen) == ROOM
            ):
                from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y

                mem = env.unwrapped.data.memory
                print("NAP BLOCK", _blocks(_s(env)))
                # 1) fine tile sweep of the N / NE quadrant post-push, via
                #    position poke (recon only) to spot a new stair tile or
                #    a walkable notch the RIGHT push may have opened.
                sweep: list[dict] = []
                for py in range(0x55, 0x86, 4) if not args.race else ():
                    row = []
                    for px in range(0x80, 0xC1, 8):
                        mem.assign(int(ADDR_LINK_X), "|u1", int(px) & 0xFF)
                        mem.assign(int(ADDR_LINK_Y), "|u1", int(py) & 0xFF)
                        env.step(nes_idle_action())
                        ss = _s(env)
                        row.append((int(ss.link_x), int(ss.link_y),
                                    int(ss.colliding_tile), int(ss.mode),
                                    f"0x{int(ss.screen):02x}"))
                        if int(ss.mode) in CELLAR_MODES or int(ss.screen) != ROOM:
                            out["cellar"] = _glance(env)
                            print("CELLAR SWEEP", out["cellar"])
                            break
                    print("NAP ROW", py, row)
                    sweep.append(row)
                    if out.get("cellar"):
                        break
                out["nap_sweep"] = sweep
                # restore Link to a safe floor cell before real walking
                if not out.get("cellar"):
                    mem.assign(int(ADDR_LINK_X), "|u1", 176)
                    mem.assign(int(ADDR_LINK_Y), "|u1", 141)
                    env.step(nes_idle_action())
                # 2) real walk-in attempts on the y=93 band, west->east, and
                #    the x=176-184 gap UP.
                nap_route = (
                    (176, 141), (176, 125), (176, 117), (180, 117),
                    (176, 109), (176, 101), (176, 93),
                    (184, 93), (192, 93), (200, 93),
                )
                for tx, ty in nap_route:
                    if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                        break
                    f, ok, xy = _reach(env, a, tx, ty, f, budget=300, tol=1)
                    rec = {
                        "target": [tx, ty], "ok": ok, "xy": xy,
                        "tile": int(_s(env).colliding_tile),
                        "mode": int(_s(env).mode),
                        "screen": f"0x{int(_s(env).screen):02x}",
                    }
                    print("NAP WP", rec)
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_nap_{tx}_{ty}.png",
                    )
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                    # at each y=93 cell, press UP then RIGHT then idle
                    if ok and ty <= 96:
                        for btn in ("UP", "UP", "RIGHT", None, "UP", None):
                            _step(env, a, btn, f)
                            f += 1
                            if int(_s(env).mode) in CELLAR_MODES:
                                print("NAP CELLAR", [tx, ty], btn, _glance(env))
                                break
                        for _ in range(30):
                            if int(_s(env).mode) in CELLAR_MODES:
                                break
                            _step(env, a, None, f)
                            f += 1
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                # 3) straight-UP pushes from the y=101/109 rows at x=192/184
                if int(_s(env).mode) not in CELLAR_MODES:
                    for sx, sy in ((192, 109), (184, 109), (192, 101), (200, 109)):
                        if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                            break
                        f, ok, xy = _reach(env, a, sx, sy, f, budget=260, tol=1)
                        print("NAP UPPUSH STAND", [sx, sy], ok, xy,
                              "tile", int(_s(env).colliding_tile))
                        for i in range(24):
                            s = _s(env)
                            if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                                break
                            _step(env, a, "UP", f)
                            f += 1
                            if i % 6 == 0:
                                print("NAP UPPUSH", [sx, sy], i, _glance(env)["xy"],
                                      "tile", int(s.colliding_tile))
                        save_rgb_png(
                            env.render(),
                            RECORDINGS_DIR / f"{args.tag}_nap_up_{sx}_{sy}.png",
                        )
                        if int(_s(env).mode) in CELLAR_MODES:
                            break
                if int(_s(env).mode) in CELLAR_MODES:
                    out["cellar"] = _glance(env)
                    print("NAP CELLAR FINAL", out["cellar"])
                    for _ in range(200):
                        s = _s(env)
                        if int(s.mode) == PLAY_MODE and int(s.screen) != ROOM:
                            break
                        _step(env, a, None, f)
                        f += 1
                    out["cellar_settled"] = _glance(env)
                    print("NAP CELLAR SETTLED", out["cellar_settled"])
                out["after_north_after_push"] = _glance(env)
                print("AFTER NAP", out["after_north_after_push"])

        if args.race and int(_s(env).mode) not in CELLAR_MODES:
            # After a just-started RIGHT slide, run UP the x=176 column
            # toward the NE hole before slot11 snaps to (208,96).
            print("RACE START", _glance(env)["xy"], "block", _blocks(_s(env)))
            for i in range(80):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR RACE", out["cellar"])
                    break
                x, y = int(s.link_x), int(s.link_y)
                btn: str | tuple[str, ...]
                # North-arm east edge is (176,117). RIGHT+UP toward
                # CheckWarp (208,93) while the 0x68 is still on y=144.
                if y > 125:
                    btn = "UP"
                elif x < 208:
                    btn = ("RIGHT", "UP")
                else:
                    btn = "UP"
                _step(env, a, btn, f)
                f += 1
                if i % 4 == 0:
                    print(
                        "RACE", i, _glance(env)["xy"],
                        "tile", int(_s(env).colliding_tile),
                        "block", _blocks(_s(env)),
                        "mode", int(s.mode),
                    )
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_race.png")
            out["after_race"] = _glance(env)
            print("AFTER RACE", out["after_race"])

        if args.ne and int(_s(env).mode) not in CELLAR_MODES:
            # North-arm east edge sits one tile west of the visible hole.
            for wx, wy in ((160, 141), (160, 117), (176, 117)):
                f, ok, xy = _reach(env, a, wx, wy, f, budget=500, tol=1)
                print(
                    "NE WP", [wx, wy], ok, xy,
                    "tile", int(_s(env).colliding_tile),
                    "mode", int(_s(env).mode),
                )
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    break
            out["ne_stand"] = _glance(env)
            print("NE STAND", out["ne_stand"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_ne_stand.png")
            holds: tuple[tuple[str | tuple[str, ...], int], ...] = (
                ("RIGHT", 48),
                ("UP", 48),
                (("RIGHT", "UP"), 48),
                (("RIGHT", "DOWN"), 24),
                ("DOWN", 16),
            )
            for hi, (hbtn, n) in enumerate(holds):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                f, ok, xy = _reach(env, a, 176, 117, f, budget=240, tol=1)
                print("NE RESET", hi, ok, xy, "tile", int(_s(env).colliding_tile))
                last = None
                for i in range(n):
                    s = _s(env)
                    if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                        out["cellar"] = _glance(env)
                        print("CELLAR NE", hbtn, out["cellar"])
                        break
                    xy = [int(s.link_x), int(s.link_y)]
                    tile = int(s.colliding_tile)
                    if xy != last or i % 8 == 0:
                        print("NE HOLD", hbtn, i, xy, "tile", tile)
                        last = xy
                    _step(env, a, hbtn, f)
                    f += 1
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_ne_{hi}.png",
                )
                if out.get("cellar"):
                    break
            if int(_s(env).mode) not in CELLAR_MODES:
                for tx, ty in ((176, 109), (176, 93), (160, 93), (192, 117)):
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                    f, ok, xy = _reach(env, a, tx, ty, f, budget=240, tol=0)
                    rec = {
                        "target": [tx, ty],
                        "ok": ok,
                        "xy": xy,
                        "tile": int(_s(env).colliding_tile),
                        "mode": int(_s(env).mode),
                    }
                    print("NE POSE", rec)
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_pose_{tx}_{ty}.png",
                    )
                    if ok:
                        for _ in range(40):
                            if int(_s(env).mode) in CELLAR_MODES:
                                out["cellar"] = _glance(env)
                                print("CELLAR POSE", out["cellar"])
                                break
                            _step(env, a, None, f)
                            f += 1
                    if out.get("cellar"):
                        break
            out["after_ne"] = _glance(env)
            print("AFTER NE", out["after_ne"])

        if args.north_band and int(_s(env).mode) not in CELLAR_MODES:
            # Plus north-arm center, then UP onto the y=93 band the
            # bubbles patrol, then RIGHT toward CheckWarp (208,93).
            for wx, wy in ((160, 141), (160, 117), (144, 117), (128, 117)):
                f, ok, xy = _reach(env, a, wx, wy, f, budget=400, tol=1)
                print(
                    "NB WP", [wx, wy], ok, xy,
                    "tile", int(_s(env).colliding_tile),
                )
            f, ok, xy = _reach(env, a, 160, 117, f, budget=240, tol=1)
            print("NB ARM", ok, xy, "tile", int(_s(env).colliding_tile))
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_nb_arm.png"
            )
            last = None
            for i in range(80):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR NB UP", out["cellar"])
                    break
                y = int(s.link_y)
                x = int(s.link_x)
                if y <= 93:
                    break
                _step(env, a, "UP", f)
                f += 1
                xy = [x, y]
                if xy != last or i % 8 == 0:
                    print("NB UP", i, xy, "tile", int(s.colliding_tile))
                    last = xy
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_nb_up.png"
            )
            for cx in (128, 144, 160, 176):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                f, ok, xy = _reach(env, a, cx, 117, f, budget=200, tol=1)
                print("NB CLIP STAND", cx, ok, xy, "tile", int(_s(env).colliding_tile))
                for clip in (("UP", "RIGHT"), ("UP", "LEFT"), ("RIGHT", "UP")):
                    for i in range(12):
                        s = _s(env)
                        if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                            out["cellar"] = _glance(env)
                            print("CELLAR CLIP", clip, out["cellar"])
                            break
                        _step(env, a, clip, f)
                        f += 1
                        if i % 4 == 0:
                            print(
                                "NB CLIP", cx, clip, i,
                                _glance(env)["xy"],
                                "tile", int(s.colliding_tile),
                            )
                    if out.get("cellar"):
                        break
                    f, ok, xy = _reach(env, a, cx, 117, f, budget=80, tol=1)
                if out.get("cellar"):
                    break
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_nb_clip.png"
            )
            last = None
            for i in range(160):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR NB", out["cellar"])
                    break
                x, y = int(s.link_x), int(s.link_y)
                if x >= 208 and y <= 93:
                    _step(env, a, None, f)
                elif y > 93:
                    _step(env, a, "UP", f)
                else:
                    _step(env, a, "RIGHT", f)
                f += 1
                xy = [x, y]
                if xy != last or i % 8 == 0:
                    print(
                        "NB RIGHT", i, xy,
                        "tile", int(s.colliding_tile),
                        "mode", int(s.mode),
                    )
                    last = xy
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_nb_right.png"
            )
            out["after_north_band"] = _glance(env)
            print("AFTER NB", out["after_north_band"])

        if args.east_col and int(_s(env).mode) not in CELLAR_MODES:
            # L6 pattern: east column x=208 UP onto (208,93). After
            # room_all_dead the east wall is not a wallmaster grab.
            for wx, wy in ((176, 141), (192, 141), (200, 141), (208, 141)):
                f, ok, xy = _reach(env, a, wx, wy, f, budget=400, tol=1)
                print(
                    "EC WP", [wx, wy], ok, xy,
                    "tile", int(_s(env).colliding_tile),
                    "screen", f"0x{int(_s(env).screen):02x}",
                    "mode", int(_s(env).mode),
                )
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_ec_{wx}_{wy}.png",
                )
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR EC", out["cellar"])
                    break
            if int(_s(env).mode) not in CELLAR_MODES and int(_s(env).screen) == ROOM:
                for i in range(120):
                    s = _s(env)
                    if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                        out["cellar"] = _glance(env)
                        print("CELLAR EC UP", out["cellar"])
                        break
                    x, y = int(s.link_x), int(s.link_y)
                    if x < 208:
                        btn: str | tuple[str, ...] = "RIGHT"
                    else:
                        btn = "UP"
                    _step(env, a, btn, f)
                    f += 1
                    if i % 8 == 0:
                        print(
                            "EC UP", i, [x, y],
                            "tile", int(s.colliding_tile),
                            "mode", int(s.mode),
                        )
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_ec_up.png"
                )
            out["after_east_col"] = _glance(env)
            print("AFTER EC", out["after_east_col"])

        if args.poke_warp and int(_s(env).mode) not in CELLAR_MODES:
            from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y

            mem = env.unwrapped.data.memory
            poses = (
                (192, 93),
                (176, 93),
                (192, 96),
                (208, 85),
                (216, 85),
                (224, 85),
                (192, 85),
                (208, 93),
            )
            for px, py in poses:
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                mem.assign(int(ADDR_LINK_X), "|u1", int(px) & 0xFF)
                mem.assign(int(ADDR_LINK_Y), "|u1", int(py) & 0xFF)
                env.step(nes_idle_action())
                print("POKE AT", [px, py], _glance(env)["xy"],
                      "tile", int(_s(env).colliding_tile),
                      "mode", int(_s(env).mode),
                      "screen", f"0x{int(_s(env).screen):02x}")
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_poke_{px}_{py}.png",
                )
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR POKE AT", out["cellar"])
                    break
                for btn in ("LEFT", "RIGHT", "UP", "DOWN", "RIGHT", "UP"):
                    _step(env, a, btn, f)
                    f += 1
                    ss = _s(env)
                    print(
                        "POKE STEP", [px, py], btn,
                        _glance(env)["xy"],
                        "tile", int(ss.colliding_tile),
                        "mode", int(ss.mode),
                        "screen", f"0x{int(ss.screen):02x}",
                    )
                    if int(ss.mode) in CELLAR_MODES or int(ss.screen) != ROOM:
                        out["cellar"] = _glance(env)
                        print("CELLAR POKE STEP", out["cellar"])
                        break
                if out.get("cellar"):
                    break
            print("POKE", _glance(env))
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_poke.png"
            )
            # CheckWarps wants a step onto the pixel, not an idle spawn.
            for btn in ("LEFT", "RIGHT", "UP", "RIGHT", "UP"):
                _step(env, a, btn, f)
                f += 1
                print("POKE STEP", btn, _glance(env))
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR POKE STEP", out["cellar"])
                    break
            for i in range(120):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    out["cellar"] = _glance(env)
                    print("CELLAR POKE", out["cellar"])
                    break
                x, y = int(s.link_x), int(s.link_y)
                if x < 208:
                    btn = "RIGHT"
                elif y > 93:
                    btn = "UP"
                else:
                    btn = "UP"
                _step(env, a, btn, f)
                f += 1
                if i % 10 == 0:
                    print("POKE HOLD", i, _glance(env))
            out["after_poke"] = _glance(env)
            print("AFTER POKE", out["after_poke"])
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_poke_settled.png"
            )

        if args.dump_tiles and int(_s(env).mode) not in CELLAR_MODES:
            from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y

            hits = []
            ne = []
            mem = env.unwrapped.data.memory
            for y in range(0x4D, 0x90, 1):
                row = []
                for x in range(0xB0, 0xE1, 1):
                    mem.assign(int(ADDR_LINK_X), "|u1", int(x) & 0xFF)
                    mem.assign(int(ADDR_LINK_Y), "|u1", int(y) & 0xFF)
                    env.step(nes_idle_action())
                    s = _s(env)
                    rec = {
                        "x": int(s.link_x),
                        "y": int(s.link_y),
                        "tile": int(s.colliding_tile),
                        "mode": int(s.mode),
                        "screen": f"0x{int(s.screen):02x}",
                    }
                    row.append(rec)
                    if (
                        0x70 <= int(s.colliding_tile) <= 0x73
                        or int(s.mode) in CELLAR_MODES
                        or int(s.screen) != ROOM
                    ):
                        hits.append(rec)
                        print("TILE HIT", rec)
                    if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                        out["cellar"] = _glance(env)
                        print("CELLAR DUMP", out["cellar"])
                        break
                ne.append(row)
                if out.get("cellar"):
                    break
                if y in (93, 96, 101, 109, 117) or y % 8 == 0:
                    print(
                        "NE ROW", y,
                        [(r["x"], r["tile"]) for r in row if r["x"] % 8 == 0 or r["x"] in (192, 200, 208)],
                    )
            out["ne_dump"] = [
                {"x": r["x"], "y": r["y"], "tile": r["tile"]}
                for row in ne for r in row
                if r["tile"] not in (116, 117, 118, 119, 176, 177, 178, 179)
            ]
            print("NE UNUSUAL", out["ne_dump"][:40], "n", len(out["ne_dump"]))
            if out.get("cellar"):
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_dump_cellar.png",
                )
            # keep the coarse dump below for the rest of the room
            for y in range(0x4D, 0xDE, 8):
                if out.get("cellar"):
                    break
                for x in range(0x20, 0xE1, 8):
                    mem.assign(int(ADDR_LINK_X), "|u1", int(x) & 0xFF)
                    mem.assign(int(ADDR_LINK_Y), "|u1", int(y) & 0xFF)
                    env.step(nes_idle_action())
                    s = _s(env)
                    tile = int(s.colliding_tile)
                    rec = {
                        "x": int(s.link_x),
                        "y": int(s.link_y),
                        "tile": tile,
                        "mode": int(s.mode),
                        "screen": f"0x{int(s.screen):02x}",
                    }
                    interesting = (
                        0x70 <= tile <= 0x76
                        or tile == 0x24
                        or int(s.mode) in CELLAR_MODES
                        or int(s.screen) != ROOM
                    )
                    if interesting:
                        hits.append(rec)
                        print("TILE HIT", rec)
                    if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                        out["cellar"] = _glance(env)
                        print("CELLAR DUMP", out["cellar"])
                        save_rgb_png(
                            env.render(),
                            RECORDINGS_DIR / f"{args.tag}_dump_cellar.png",
                        )
                        break
            out["tile_hits"] = hits
            print("TILE HITS", hits)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_dump.png")

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
