"""Recon: NOSE_CELLAR (0x7b) -> PRE_BOSS -> AQUAMENTUS tail (rr-8t4.3).

Starts from Level7Interior0DNoseCellarReconFixture (Step-2 poke fixture).
No writes here -- pure navigation recon. Screenshots every transition.

  --cellar      walk the cellar both ways, find the far-side stairs
  --to-preboss  cellar -> PRE_BOSS, glance + shot
  --to-aqua     ... -> bomb east -> AQUAMENTUS, live census
  --save NAME   save state at the current stop (recon fixture, no writes)

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_nosecellar_tail.py --cellar
"""
import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action, nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.dungeon.ops import ensure_bomb
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (read_snapshot, read_u8, ADDR_LINK_X, ADDR_LINK_Y,
                         ADDR_BOMBS, ADDR_KEYS, ADDR_CANDLE, ADDR_HEALTH,
                         ADDR_HEART_PARTIAL)
from zelda_i.runner import make_assist

FROM = "Level7Interior0DNoseCellarReconFixture"
CELLAR_MODES = {9, 10, 11, 16}


def build(frm=FROM):
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, frm, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    for _ in range(2):
        env.step(nes_idle_action())
    return env, a


def s(env):
    return read_snapshot(env.get_ram())


def g(env):
    st = s(env)
    ram = env.get_ram()
    return {
        "screen": f"0x{int(st.screen):02x}", "mode": int(st.mode),
        "xy": [int(st.link_x), int(st.link_y)], "ct": int(st.colliding_tile),
        "level": int(st.level), "transitioning": bool(st.transitioning),
        "cur_opened_doors": int(st.cur_opened_doors),
        "open_doorway_mask": int(st.open_doorway_mask),
        "room_all_dead": int(st.room_all_dead), "room_item_id": int(st.room_item_id),
        "triforce": int(st.triforce),
        "objs": [{"slot": int(o.slot), "t": f"0x{int(o.type_id):02x}",
                  "name": object_name(int(o.type_id)), "hp": int(o.hp),
                  "state": int(o.state), "xy": [int(o.x), int(o.y)]}
                 for o in st.objects if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)],
        "bombs": int(read_u8(ram, ADDR_BOMBS)), "keys": int(read_u8(ram, ADDR_KEYS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "hc": int(st.heart_containers),
        "health_raw": int(read_u8(ram, ADDR_HEALTH)),
        "heart_partial": int(read_u8(ram, ADDR_HEART_PARTIAL)),
    }


def step(env, a, btn, f=0):
    env.step(nes_idle_action() if btn is None else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    a.apply_env(env, frame=f)


def hold(env, a, btn, n):
    for _ in range(n):
        step(env, a, btn)


def reach(env, a, tx, ty, budget=600, tol=2, stopfn=None):
    last, stuck = None, 0
    for _ in range(budget):
        st = s(env)
        if stopfn and stopfn(st):
            return "stop"
        x, y = int(st.link_x), int(st.link_y)
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return "arrived"
        if (x, y) == last:
            stuck += 1
            if stuck >= 50:
                return "stuck"
        else:
            stuck, last = 0, (x, y)
        if abs(y - ty) > tol:
            step(env, a, "UP" if y > ty else "DOWN")
        else:
            step(env, a, "LEFT" if x > tx else "RIGHT")
    return "budget"


def sweep_cellar(env, a):
    mem = env.unwrapped.data.memory
    ox, oy = int(s(env).link_x), int(s(env).link_y)
    rows = []
    seen = {}
    for py in range(0x50, 0xAC, 4):
        line = []
        for px in range(0x08, 0xF8, 4):
            mem.assign(int(ADDR_LINK_X), "|u1", px)
            mem.assign(int(ADDR_LINK_Y), "|u1", py)
            env.step(nes_idle_action())
            st = s(env)
            t = int(st.colliding_tile)
            if int(st.mode) != 9 or int(st.screen) != 0x7b:
                line.append("W")
            else:
                seen[t] = seen.get(t, 0) + 1
                line.append(f"{t:02x} "[0])  # placeholder
        rows.append(f"y={py:3d} " + "".join(line))
    rows.append("tiles seen: " + json.dumps({f"0x{k:02x}": v for k, v in sorted(seen.items())}))
    mem.assign(int(ADDR_LINK_X), "|u1", ox)
    mem.assign(int(ADDR_LINK_Y), "|u1", oy)
    env.step(nes_idle_action())
    return rows


def walk_cellar2(env, a, tag):
    """Cellar loads only after a long idle. Floor is y=189; climb UP at a
    ladder column at one end. Sweep every x, press UP, watch for mode!=9."""
    ev = []
    ss = int(s(env).screen)
    def gone():
        st = s(env)
        return int(st.mode) != 9 or int(st.screen) != ss
    hold(env, a, None, 400)
    ev.append(("loaded", g(env)))
    reach(env, a, int(s(env).link_x), 189, tol=2, stopfn=lambda st: gone())
    ev.append(("on_floor", g(env)["xy"]))
    # sweep x 208 -> 16, at each x try UP for 20f
    x = 208
    while x >= 16 and not gone():
        reach(env, a, x, 189, tol=3, stopfn=lambda st: gone())
        for _ in range(24):
            if gone():
                break
            step(env, a, "UP")
        if gone():
            ev.append((f"UP@x{x}", g(env)))
            break
        cur = int(s(env).link_x)
        if abs(cur - x) > 6:  # blocked, note & continue
            ev.append((f"blocked@x{x}", [cur, int(s(env).link_y)]))
        x -= 8
    if not gone():
        # try pushing hard into both ends
        for tx, d in ((240, "RIGHT"), (0, "LEFT")):
            reach(env, a, tx, 189, tol=4, stopfn=lambda st: gone())
            for _ in range(40):
                if gone():
                    break
                step(env, a, d)
            if gone():
                ev.append((f"end_{d}", g(env)))
                break
    # ride transition
    for _ in range(300):
        st = s(env)
        if int(st.mode) == 5 and int(st.screen) != ss:
            break
        step(env, a, "UP" if int(st.mode) in (10, 16) else None)
    ev.append(("after", g(env)))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_wc2.png")
    return ev


def walk_cellar(env, a, tag):
    """LoZ cellar: drop to the corridor, try each end for the up-stairs."""
    ev = []
    start_screen = int(s(env).screen)
    # settle
    hold(env, a, None, 20)
    ev.append(("settled", g(env)))
    # go down to corridor
    reach(env, a, int(s(env).link_x), 180, tol=3)
    ev.append(("at_corridor", g(env)))
    # try east end
    r = reach(env, a, 240, 180, tol=3, stopfn=lambda st: int(st.mode) not in CELLAR_MODES or int(st.screen) != start_screen)
    ev.append(("east_end", r, g(env)))
    for _ in range(40):
        st = s(env)
        if int(st.mode) not in CELLAR_MODES or int(st.screen) != start_screen:
            break
        step(env, a, "RIGHT")
    ev.append(("east_push", g(env)))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_cellar_east.png")
    if int(s(env).screen) == start_screen and int(s(env).mode) in CELLAR_MODES:
        # try west end
        reach(env, a, 16, 180, tol=3, stopfn=lambda st: int(st.mode) not in CELLAR_MODES or int(st.screen) != start_screen)
        for _ in range(40):
            st = s(env)
            if int(st.mode) not in CELLAR_MODES or int(st.screen) != start_screen:
                break
            step(env, a, "LEFT")
        ev.append(("west_push", g(env)))
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_cellar_west.png")
    # ride the stair transition
    for _ in range(200):
        st = s(env)
        if int(st.mode) == 5 and int(st.screen) != start_screen:
            break
        step(env, a, "UP" if int(st.mode) in (10, 16) else None)
    ev.append(("after_stairs", g(env)))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_after_stairs.png")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l7tail_v1")
    ap.add_argument("--from-state", default=FROM)
    ap.add_argument("--cellar", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--pit", action="store_true", help="navigate to left pit and step onto it")
    ap.add_argument("--manual", default="", help="comma seq like L:200,U:60,IDLE:120")
    ap.add_argument("--topsweep", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    env, a = build(args.from_state)
    out = {"tag": args.tag, "start": g(env)}
    print("START", json.dumps(out["start"], default=str))

    if args.manual:
        B = {"L": "LEFT", "R": "RIGHT", "U": "UP", "D": "DOWN", "IDLE": None}
        trace = []
        last = None
        for tok in args.manual.split(","):
            k, n = tok.split(":")
            for _ in range(int(n)):
                step(env, a, B[k])
                st = s(env)
                key = (int(st.screen), int(st.mode), int(st.link_x), int(st.link_y))
                if key != last:
                    trace.append({"k": k, **g(env)})
                    last = key
        out["manual"] = trace
        for t in trace:
            print(json.dumps(t, default=str))
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_manual.png")

    if args.pit and False:
        pass

    if getattr(args, "topsweep", False):
        env.close()
        res = []
        for xt in (208, 176, 160, 128, 96, 64, 48, 32):
            env2, a2 = build(args.from_state)
            hold(env2, a2, None, 400)
            reach(env2, a2, xt, 100, tol=3,
                  stopfn=lambda st: int(st.mode) != 9 or int(st.screen) != 0x7b)
            atx = g(env2)["xy"]
            for _ in range(80):
                st = s(env2)
                if int(st.mode) != 9 or int(st.screen) != 0x7b:
                    break
                step(env2, a2, "UP")
            for _ in range(200):
                st = s(env2)
                if int(st.mode) == 5 and int(st.screen) != 0x7b:
                    break
                step(env2, a2, "UP" if int(st.mode) in (10, 16) else None)
            res.append({"x_target": xt, "at": atx, "dest": g(env2)})
            print("TOP", xt, "at", atx, "->", g(env2)["screen"], "m", g(env2)["mode"], g(env2)["xy"])
            save_rgb_png(env2.render(), RECORDINGS_DIR / f"{args.tag}_top{xt}.png")
            env2.close()
        out["topsweep"] = res
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1, default=str))
        return

    if args.sweep:
        out["sweep"] = sweep_cellar(env, a)
        for r in out["sweep"]:
            print(r)

    if args.pit:
        ev = []
        # rise into the corridor then walk the full width both ways, trying
        # to step onto every dark cell.
        hold(env, a, None, 16)
        for tgt in [(120, 150), (40, 150), (40, 177), (40, 165)]:
            r = reach(env, a, *tgt, tol=3,
                      stopfn=lambda st: int(st.mode) != 9 or int(st.screen) != 0x7b)
            ev.append((f"reach_{tgt}", r, g(env)["xy"], g(env)["ct"], g(env)["mode"]))
            for btn in ("DOWN", "LEFT", "UP", "DOWN"):
                for _ in range(12):
                    st = s(env)
                    if int(st.mode) != 9 or int(st.screen) != 0x7b:
                        break
                    step(env, a, btn)
                if int(s(env).mode) != 9 or int(s(env).screen) != 0x7b:
                    break
            if int(s(env).mode) != 9 or int(s(env).screen) != 0x7b:
                break
        # ride transition
        for _ in range(240):
            st = s(env)
            if int(st.mode) == 5 and int(st.screen) != 0x7b:
                break
            step(env, a, "UP" if int(st.mode) in (10, 16) else None)
        ev.append(("after", g(env)))
        out["pit"] = ev
        for e in ev:
            print(json.dumps(e, default=str))
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pit.png")

    if args.cellar:
        out["cellar"] = walk_cellar2(env, a, args.tag)
        for e in out["cellar"]:
            print(json.dumps(e, default=str))

    if args.save and int(s(env).mode) == 5:
        path = save_state(env, GAME_DIR, GAME, args.save)
        src = state_path(GAME_DIR, GAME, args.from_state)
        write_state_provenance(
            path, source_state_path=src if src.exists() else None,
            request={"bead": "rr-8t4.3", "phase": "level7_preboss_recon",
                     "track": "recon_fixture", "route_eligible": False,
                     "fixture_only": True, "natural_entry": False,
                     "development_only": True, "fixture_writes": [],
                     "notes": ["Downstream of the 0x0D poke fixture; no writes in this hop.",
                               "UnlimitedHealthAssist traversal aid only."]},
            selected_trial={"ok": True, "state": compact_snapshot(s(env)), "glance": g(env)},
            natural_entry=False)
        out["saved"] = str(path)
        print("SAVED", path)

    (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print("END", json.dumps(g(env), default=str))
    env.close()


if __name__ == "__main__":
    main()
