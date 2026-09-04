"""Recon: 0x0D TIP_OF_NOSE — reach the NE staircase (dest 0x7b mode 9).

The staircase tiles (0x70-0x73) sit at x~192 y<=101 in a north pocket that
is sealed by a full-width solid band at y~101-133.  A poke to (208,93)
warps to NOSE_CELLAR (screen 0x7b, mode 9, 4x keese 0x1b) but that is not a
walk-on.  This probe tries to open the x=192 column with a bomb on the
y~133 plug, then walk up onto the staircase.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_room0d_ne.py --tag 0d_ne_v1 --push --bomb-y 133
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ops import ensure_bomb
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
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
PUSHABLE_BLOCK = 0x68


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
        {"slot": int(o.slot), "x": int(o.x), "y": int(o.y)}
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) == PUSHABLE_BLOCK
    ]


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "level": int(s.level),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_all_dead": int(s.room_all_dead),
        "blocks": _blocks(s),
        "objects": [
            {"slot": int(o.slot), "type": f"0x{int(o.type_id):02x}",
             "hp": int(o.hp), "xy": [int(o.x), int(o.y)]}
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        ],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "triforce": int(s.triforce),
    }


def _reach(env, a, tx, ty, f, budget=600, tol=2):
    last = None
    stuck = 0
    axis = 0  # 0 = y-first, 1 = x-first (flip when stuck)
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False, [int(s.link_x), int(s.link_y)]
        if int(s.mode) in CELLAR_MODES:
            return f, True, [int(s.link_x), int(s.link_y)]
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return f, True, [x, y]
        if (x, y) == last:
            stuck += 1
            if stuck in (12, 24, 36):
                axis ^= 1  # try the other axis
            if stuck >= 60:
                return f, False, [x, y]
        else:
            stuck = 0
            last = (x, y)
        want_y = abs(y - ty) > tol
        want_x = abs(x - tx) > tol
        if want_y and want_x:
            btn = ("UP" if y > ty else "DOWN") if axis == 0 else (
                "LEFT" if x > tx else "RIGHT")
        elif want_y:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        _step(env, a, btn, f)
        f += 1
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


_FACE_STAND = {
    "RIGHT": lambda bx, by: (bx - 16, by),
    "LEFT": lambda bx, by: (bx + 16, by),
    "UP": lambda bx, by: (bx, by + 16),
    "DOWN": lambda bx, by: (bx, by - 16),
}


def _hold(env, a, btn, n, f, stop_screen=True):
    for _ in range(n):
        s = _s(env)
        if stop_screen and (int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM):
            return f
        _step(env, a, btn, f)
        f += 1
    return f


def _push_right(env, a, f, face="RIGHT"):
    """Push the 0x68 at (192,144) in ``face`` and let the slide settle."""
    bl = _blocks(_s(env))
    if not bl:
        return f, False
    bx, by = bl[0]["x"], bl[0]["y"]
    sx, sy = _FACE_STAND[face](bx, by)
    if face in ("RIGHT", "LEFT"):
        pre = ((160, 141), (sx, by), (sx, by))
        align_axis = "y"
        for wx, wy in pre:
            f, ok, xy = _reach(env, a, wx, wy, f)
            print("  push approach", [wx, wy], ok, xy,
                  "tile", int(_s(env).colliding_tile))
    else:
        # precision squeeze: the y~152-164 corridor east past the x176-188
        # diamond mass (mass bottom ~y148, lower band top ~y164 -> ~16px gap,
        # Link must sit at y~156). Then rise the x=192 column to (192,160).
        f, ok, xy = _reach(env, a, 150, 152, f)
        print("  pu pre-squeeze", ok, xy)
        for _ in range(160):
            s = _s(env)
            if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                break
            x, y = int(s.link_x), int(s.link_y)
            if x >= sx - 2:
                break
            if y < 154:
                btn = ("DOWN", "RIGHT")
            elif y > 158:
                btn = ("UP", "RIGHT")
            else:
                btn = "RIGHT"
            _step(env, a, btn, f)
            f += 1
        print("  pu squeeze end", _glance(env)["xy"], "tile", int(_s(env).colliding_tile))
        f, ok, xy = _reach(env, a, sx, sy, f)
        print("  pu at south face", ok, xy, "tile", int(_s(env).colliding_tile))
        align_axis = "x"
    for _ in range(30):
        s = _s(env)
        if align_axis == "y":
            if abs(int(s.link_y) - by) <= 1:
                break
            _step(env, a, "DOWN" if int(s.link_y) < by else "UP", f)
        else:
            if abs(int(s.link_x) - bx) <= 1:
                break
            _step(env, a, "RIGHT" if int(s.link_x) < bx else "LEFT", f)
        f += 1
    started = False
    for _ in range(120):
        s = _s(env)
        if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
            break
        cb = _blocks(s)
        cbx = cb[0]["x"] if cb else bx
        cby = cb[0]["y"] if cb else by
        if not started and (cbx != bx or cby != by):
            started = True
        if started and (abs(cbx - bx) >= 16 or abs(cby - by) >= 16):
            break
        _step(env, a, None if started else face, f)
        f += 1
    for _ in range(24):
        cb = _blocks(_s(env))
        if not cb:
            break
        if abs(cb[0]["x"] - bx) >= 16 or abs(cb[0]["y"] - by) >= 16:
            break
        _step(env, a, None, f)
        f += 1
    return f, True


def _bomb_up(env, a, f, stand_xy, retreat="DOWN", settle=110):
    tx, ty = stand_xy
    f, ok, xy = _reach(env, a, tx, ty, f, tol=1)
    print("  bomb stand", stand_xy, ok, xy, "tile", int(_s(env).colliding_tile))
    for _ in range(6):
        _step(env, a, "UP", f)
        f += 1
    ensure_bomb(env)
    env.step(nes_action("B"))
    a.apply_env(env, frame=f)
    f += 1
    for _ in range(10):
        _step(env, a, retreat, f)
        f += 1
    for _ in range(settle):
        _step(env, a, None, f)
        f += 1
    return f, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0d_ne_v1")
    ap.add_argument("--from-state", default="Level7Interior0DClearedReconFixture")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-dir", default="RIGHT", choices=["RIGHT", "UP", "DOWN", "LEFT"])
    ap.add_argument("--e-column", action="store_true",
                    help="after push: SE corner then climb x=208 UP to (208,96)")
    ap.add_argument("--bomb-col", default="192",
                    help="comma x list for the bomb-UP column climb")
    ap.add_argument("--bomb-y", default="141,125,109",
                    help="comma y list of stand rows to bomb UP from")
    ap.add_argument("--no-bomb", action="store_true")
    ap.add_argument("--race-col", type=int, default=0,
                    help="race UP this x column during the block slide")
    ap.add_argument("--save-fixture", default="")
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
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        if args.race_col:
            bl = _blocks(_s(env))
            bx, by = bl[0]["x"], bl[0]["y"]
            for wx, wy in ((160, 141), (176, 144), (bx - 16, by)):
                f, ok, xy = _reach(env, a, wx, wy, f)
            for _ in range(30):
                s = _s(env)
                if abs(int(s.link_y) - by) <= 1:
                    break
                _step(env, a, "DOWN" if int(s.link_y) < by else "UP", f)
                f += 1
            # push until the slide starts, then sprint to the race column
            started = False
            for _ in range(120):
                s = _s(env)
                cb = _blocks(s)
                cbx = cb[0]["x"] if cb else bx
                if cbx != bx:
                    started = True
                    break
                _step(env, a, "RIGHT", f)
                f += 1
            print("SLIDE STARTED, racing x=", args.race_col)
            trace = []
            for i in range(90):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    break
                x, y = int(s.link_x), int(s.link_y)
                cb = _blocks(s)
                trace.append([x, y, int(s.colliding_tile),
                              cb[0]["x"] if cb else -1, cb[0]["y"] if cb else -1])
                if abs(x - args.race_col) > 2:
                    _step(env, a, "LEFT" if x > args.race_col else "RIGHT", f)
                else:
                    _step(env, a, "UP", f)
                f += 1
            print("RACE TRACE", trace[::4])
            out["race_trace"] = trace
            out["after_race"] = _glance(env)
            print("AFTER RACE", out["after_race"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_race.png")
        if args.push:
            f, _ = _push_right(env, a, f, face=args.push_dir)
            out["after_push"] = _glance(env)
            print("AFTER PUSH", out["after_push"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")

        if args.e_column and int(_s(env).mode) not in CELLAR_MODES:
            # L9 room30/03 recipe: after the block secret opens, the east-wall
            # column becomes walkable; climb x=208 from y=189 to stand exactly
            # at (208,96) = (0xD0,0x60) -> CheckWarps.
            for wx, wy in ((192, 165), (200, 189), (208, 189), (208, 165),
                           (208, 141), (208, 125), (208, 109), (208, 96)):
                if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
                    break
                f, ok, xy = _reach(env, a, wx, wy, f, budget=320, tol=1)
                g = _glance(env)
                print("ECOL WP", [wx, wy], ok, xy, "tile", g["colliding_tile"],
                      "mode", g["mode"], "screen", g["screen"])
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_ecol_{wx}_{wy}.png"
                )
                if ok and wy <= 100:
                    for btn in ("UP", None, "RIGHT", None, "UP", None, None):
                        _step(env, a, btn, f)
                        f += 1
                        if int(_s(env).mode) in CELLAR_MODES:
                            print("ECOL CELLAR", btn, _glance(env))
                            break
                    for _ in range(40):
                        if int(_s(env).mode) in CELLAR_MODES:
                            break
                        _step(env, a, None, f)
                        f += 1
            out["after_ecol"] = _glance(env)
            print("AFTER ECOL", out["after_ecol"])

        cols = [int(v) for v in args.bomb_col.split(",") if v.strip()]
        rows = [int(v) for v in args.bomb_y.split(",") if v.strip()]
        if not args.no_bomb:
            for cx in cols:
                for ry in rows:
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                    if int(_s(env).screen) != ROOM:
                        break
                    print("BOMB COL", cx, "ROW", ry)
                    f, ok = _bomb_up(env, a, f, (cx, ry))
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_bomb_{cx}_{ry}.png",
                    )
                    # try to walk up the column past the bombed row
                    f, wok, wxy = _reach(env, a, cx, ry - 24, f, budget=240, tol=1)
                    g = _glance(env)
                    print("  UP-WALK", wok, wxy, "tile", g["colliding_tile"],
                          "mode", g["mode"], "screen", g["screen"])
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_upwalk_{cx}_{ry}.png",
                    )
                    if int(_s(env).mode) in CELLAR_MODES:
                        break
                if int(_s(env).mode) in CELLAR_MODES:
                    break

        # climb the column: walk north as far as possible, log tile at each y
        if int(_s(env).mode) not in CELLAR_MODES and int(_s(env).screen) == ROOM:
            cx = cols[0]
            f, ok, xy = _reach(env, a, cx, 141, f, budget=260, tol=2)
            print("CLIMB from", xy)
            climb = []
            last_y = None
            for i in range(200):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    break
                x, y = int(s.link_x), int(s.link_y)
                if abs(x - cx) > 2:
                    _step(env, a, "LEFT" if x > cx else "RIGHT", f)
                else:
                    _step(env, a, "UP", f)
                f += 1
                if y != last_y:
                    rec = [x, y, int(s.colliding_tile)]
                    climb.append(rec)
                    last_y = y
                if y <= 88:
                    break
            print("CLIMB TRACE", climb)
            out["climb"] = climb
            # if we made it into the pocket, walk east to trip the warp
            for i in range(120):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                    break
                _step(env, a, "RIGHT", f)
                f += 1
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_climb.png")

        if int(_s(env).mode) in CELLAR_MODES or int(_s(env).screen) != ROOM:
            out["cellar"] = _glance(env)
            print("CELLAR", out["cellar"])
            for _ in range(240):
                s = _s(env)
                if int(s.mode) == PLAY_MODE and int(s.screen) != ROOM:
                    break
                _step(env, a, None, f)
                f += 1
            out["cellar_settled"] = _glance(env)
            print("CELLAR SETTLED", out["cellar_settled"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png")

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
                    "phase": "level7_nose_cellar_recon",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "development_only": True,
                    "fixture_writes": [],
                    "notes": [
                        f"Derived from {args.from_state}: 0x0D RIGHT push + "
                        "bomb the y~133 plug in the x=192 column, walk up the "
                        "staircase to NOSE_CELLAR. No Candle/TF/door writes.",
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
