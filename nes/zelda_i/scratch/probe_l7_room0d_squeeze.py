"""Recon: 0x0D south-face UP-push squeeze attempt (rr-8t4.3).

Step 1 of the l7-handoff task: frame-perfect attempt to stand the 0x68
south face (192,160) and UP-push per level9/stairs.py room03 recipe.

  --sweep       full colliding_tile map y=0x88..0xBD (below the top sweep)
  --push        run the verified 0x68 RIGHT push first, then sweep
  --squeeze     raw-input attempts to reach (192,160) then UP-push
  --from-east   squeeze entry from the east pocket going down (default west)
  --xoff N      sub-pixel-ish x target offset for the corridor run

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_room0d_squeeze.py --sweep
"""

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action, nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot, ADDR_LINK_X, ADDR_LINK_Y
from zelda_i.runner import make_assist

ROOM = 0x0D
CELLAR_MODES = {9, 10, 11, 16}


def build():
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior0DClearedReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    reset_obs(env)
    for _ in range(2):
        env.step(nes_idle_action())
    return env, a


def s(env):
    return read_snapshot(env.get_ram())


def glance(env):
    st = s(env)
    return {
        "screen": f"0x{int(st.screen):02x}", "mode": int(st.mode),
        "xy": [int(st.link_x), int(st.link_y)],
        "colliding_tile": int(st.colliding_tile),
        "blocks": [(int(o.x), int(o.y), int(o.state))
                   for o in st.objects if int(o.type_id) == 0x68],
        "objs": sorted({f"0x{int(o.type_id):02x}" for o in st.objects
                        if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)}),
    }


def step(env, a, btn, f=0):
    env.step(nes_idle_action() if btn is None else nes_action(*btn)
             if isinstance(btn, tuple) else nes_action(btn))
    a.apply_env(env, frame=f)


def reach(env, a, tx, ty, budget=500, tol=2, yfirst=True):
    last, stuck = None, 0
    for _ in range(budget):
        st = s(env)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            return False
        x, y = int(st.link_x), int(st.link_y)
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return True
        if (x, y) == last:
            stuck += 1
            if stuck >= 40:
                return False
        else:
            stuck, last = 0, (x, y)
        if yfirst and abs(y - ty) > tol:
            btn = "UP" if y > ty else "DOWN"
        elif abs(x - tx) > tol:
            btn = "LEFT" if x > tx else "RIGHT"
        else:
            btn = "UP" if y > ty else "DOWN"
        step(env, a, btn)
    return False


def do_push(env, a):
    """Verified RIGHT push of the 0x68 (192,144)->(208,96) snap."""
    bl = [o for o in s(env).objects if int(o.type_id) == 0x68]
    bx, by = int(bl[0].x), int(bl[0].y)
    for wx, wy in ((160, 141), (176, 144), (bx - 16, by)):
        reach(env, a, wx, wy)
    for _ in range(30):
        st = s(env)
        if abs(int(st.link_y) - by) <= 1:
            break
        step(env, a, "DOWN" if int(st.link_y) < by else "UP")
    started = False
    for _ in range(160):
        st = s(env)
        cb = [o for o in st.objects if int(o.type_id) == 0x68]
        cbx, cby = (int(cb[0].x), int(cb[0].y)) if cb else (bx, by)
        if not started and (cbx != bx or cby != by):
            started = True
        if started and (abs(cbx - bx) >= 16 or cby != by):
            break
        step(env, a, "RIGHT" if not started else None)
    for _ in range(24):
        step(env, a, None)
    return glance(env)


def sweep(env, a, y0=0x88, y1=0xBE, tag=""):
    mem = env.unwrapped.data.memory
    out = []
    for py in range(y0, y1, 2):
        line = []
        for px in range(0x20, 0xD0, 4):
            mem.assign(int(ADDR_LINK_X), "|u1", px)
            mem.assign(int(ADDR_LINK_Y), "|u1", py)
            env.step(nes_idle_action())
            st = s(env)
            t = int(st.colliding_tile)
            if int(st.mode) in CELLAR_MODES or int(st.screen) != ROOM:
                m = "W"
            elif t in (116, 117, 118, 119):
                m = "."
            elif t in (176, 177, 178, 179):
                m = "#"
            elif 112 <= t <= 115:
                m = "S"
            else:
                m = "?"
            line.append(m)
        row = f"y={py:3d} x20 " + "".join(line)
        print(row)
        out.append(row)
    return out


def squeeze(env, a, from_east, xoff, tag):
    """Try to stand (192,160) south face and UP-push."""
    events = []
    target = (192 + xoff, 160)
    if from_east:
        # east pocket (192,141) then straight down x=192
        reach(env, a, 192, 141, yfirst=False)
        events.append(("at_east_pocket", glance(env)["xy"]))
        for _ in range(60):
            st = s(env)
            if int(st.link_y) >= 158 or int(st.screen) != ROOM:
                break
            step(env, a, "DOWN")
        events.append(("east_down_pin", glance(env)["xy"], glance(env)["colliding_tile"]))
    else:
        # west corridor: come along y=156 from the west
        reach(env, a, 96, 165, yfirst=True)
        reach(env, a, 150, 156 if xoff == 0 else 156, yfirst=False)
        events.append(("west_corridor_entry", glance(env)["xy"]))
        for _ in range(80):
            st = s(env)
            x, y = int(st.link_x), int(st.link_y)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                break
            if x >= target[0] - 1:
                break
            # hug y target while pushing east
            if y > 156:
                step(env, a, ("RIGHT", "UP"))
            elif y < 155:
                step(env, a, ("RIGHT", "DOWN"))
            else:
                step(env, a, "RIGHT")
        events.append(("west_corridor_pin", glance(env)["xy"], glance(env)["colliding_tile"]))

    st = s(env)
    x, y = int(st.link_x), int(st.link_y)
    stood_south = abs(x - 192) <= 4 and 156 <= y <= 164
    events.append(("stood_south_face", stood_south, [x, y]))
    if stood_south:
        bl0 = [(int(o.x), int(o.y)) for o in s(env).objects if int(o.type_id) == 0x68]
        for _ in range(90):
            st = s(env)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                break
            step(env, a, "UP")
        bl1 = [(int(o.x), int(o.y)) for o in s(env).objects if int(o.type_id) == 0x68]
        events.append(("up_push", {"block_before": bl0, "block_after": bl1,
                                   "glance": glance(env)}))
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_uppush.png")
    return events


def around_east(env, a, tag):
    """Go around the block on its east side: down x~204, then west to (192,160)."""
    ev = []
    reach(env, a, 200, 141, yfirst=False)
    ev.append(("ne_of_block", glance(env)["xy"]))
    for col in (204, 200, 196):
        reach(env, a, col, 141, yfirst=False)
        for _ in range(40):
            st = s(env)
            if int(st.link_y) >= 162 or int(st.screen) != ROOM:
                break
            step(env, a, "DOWN")
        ev.append((f"down_x{col}", glance(env)["xy"], glance(env)["colliding_tile"]))
        st = s(env)
        if int(st.link_y) >= 156:
            for _ in range(30):
                st = s(env)
                if int(st.link_x) <= 192 or int(st.screen) != ROOM:
                    break
                step(env, a, "LEFT")
            ev.append((f"west_from_x{col}", glance(env)["xy"], glance(env)["colliding_tile"]))
            x, y = int(s(env).link_x), int(s(env).link_y)
            if abs(x - 192) <= 6 and 154 <= y <= 166:
                bl0 = [(int(o.x), int(o.y)) for o in s(env).objects if int(o.type_id) == 0x68]
                for _ in range(90):
                    st = s(env)
                    if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                        break
                    step(env, a, "UP")
                bl1 = [(int(o.x), int(o.y)) for o in s(env).objects if int(o.type_id) == 0x68]
                ev.append(("up_push", {"before": bl0, "after": bl1, "glance": glance(env)}))
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_ae_uppush.png")
                return ev
    return ev


def wiggle_pin(env, a, tag):
    """At the (176,157) east-approach pin, try sub-pixel diagonal wiggle east."""
    ev = []
    reach(env, a, 192, 141, yfirst=False)
    for _ in range(60):
        st = s(env)
        if int(st.link_y) >= 156 or int(st.screen) != ROOM:
            break
        step(env, a, "DOWN")
    ev.append(("pin", glance(env)["xy"], glance(env)["colliding_tile"]))
    seqs = [("RIGHT", "DOWN"), ("RIGHT",), ("DOWN", "RIGHT"), ("RIGHT", "UP")]
    for _ in range(200):
        st = s(env)
        x, y = int(st.link_x), int(st.link_y)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            break
        if x >= 190:
            break
        step(env, a, seqs[_ % len(seqs)])
    ev.append(("after_wiggle", glance(env)["xy"], glance(env)["colliding_tile"]))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0d_sq_v1")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--squeeze", action="store_true")
    ap.add_argument("--from-east", action="store_true")
    ap.add_argument("--around-east", action="store_true")
    ap.add_argument("--wiggle", action="store_true")
    ap.add_argument("--xoff", type=int, default=0)
    args = ap.parse_args()

    env, a = build()
    out = {"tag": args.tag, "start": glance(env)}
    if args.push:
        out["push"] = do_push(env, a)
    if args.sweep:
        out["sweep"] = sweep(env, a, tag=args.tag)
    if args.squeeze:
        out["squeeze"] = squeeze(env, a, args.from_east, args.xoff, args.tag)
    if args.around_east:
        out["around_east"] = around_east(env, a, args.tag)
    if args.wiggle:
        out["wiggle"] = wiggle_pin(env, a, args.tag)
    out["end"] = glance(env)
    print(json.dumps(out, indent=2, default=str))
    env.close()


if __name__ == "__main__":
    main()
