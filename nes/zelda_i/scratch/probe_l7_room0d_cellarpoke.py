"""Recon: 0x0D -> NOSE_CELLAR via RIGHT push + ONE disclosed position poke.

Step 2 of the l7-handoff task. Real action: RIGHT push the 0x68 (reveals
the NE secret staircase). Then EXACTLY ONE disclosed write: poke Link to a
stair cell to trip CheckWarps into NOSE_CELLAR ($EB=0x7b mode 9).

  --scan       after push, scan NE stair cells one poke at a time (recon)
  --poke X Y   after push, single poke to (X,Y) then hold UP; save fixture
  --save NAME  save the cellar-arrival state under this fixture name

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_room0d_cellarpoke.py --scan
"""
import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action, nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot, read_u8, ADDR_LINK_X, ADDR_LINK_Y, ADDR_BOMBS, ADDR_KEYS, ADDR_CANDLE
from zelda_i.runner import make_assist

ROOM = 0x0D
CELLAR_MODES = {9, 10, 11, 16}
FROM = "Level7Interior0DClearedReconFixture"


def build():
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, FROM, GAME_DIR, render_mode="rgb_array")
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
        "level": int(st.level), "eb_screen": int(st.screen),
        "room_all_dead": int(st.room_all_dead),
        "blocks": [(int(o.x), int(o.y), int(o.state)) for o in st.objects if int(o.type_id) == 0x68],
        "objs": [{"t": f"0x{int(o.type_id):02x}", "xy": [int(o.x), int(o.y)], "hp": int(o.hp)}
                 for o in st.objects if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)],
        "bombs": int(read_u8(ram, ADDR_BOMBS)), "keys": int(read_u8(ram, ADDR_KEYS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
    }


def step(env, a, btn, f=0):
    env.step(nes_idle_action() if btn is None else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    a.apply_env(env, frame=f)


def reach(env, a, tx, ty, budget=400, tol=2):
    last, stuck = None, 0
    for _ in range(budget):
        st = s(env)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            return
        x, y = int(st.link_x), int(st.link_y)
        if abs(x - tx) <= tol and abs(y - ty) <= tol:
            return
        if (x, y) == last:
            stuck += 1
            if stuck >= 40:
                return
        else:
            stuck, last = 0, (x, y)
        if abs(y - ty) > tol:
            step(env, a, "UP" if y > ty else "DOWN")
        else:
            step(env, a, "LEFT" if x > tx else "RIGHT")


def push_right(env, a):
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
    return g(env)


def scan(env, a):
    mem = env.unwrapped.data.memory
    hits = []
    for py in range(88, 132, 2):
        for px in range(176, 224, 2):
            if int(s(env).mode) in CELLAR_MODES:
                return hits, (px, py)
            mem.assign(int(ADDR_LINK_X), "|u1", px)
            mem.assign(int(ADDR_LINK_Y), "|u1", py)
            for _ in range(4):
                step(env, a, None)
                st = s(env)
                if int(st.mode) in CELLAR_MODES or int(st.screen) != ROOM:
                    hits.append((px, py, int(st.mode), f"0x{int(st.screen):02x}"))
                    return hits, (px, py)
            for btn in ("UP", "DOWN", "LEFT", "RIGHT"):
                step(env, a, btn)
                st = s(env)
                if int(st.mode) in CELLAR_MODES or int(st.screen) != ROOM:
                    hits.append((px, py, btn, int(st.mode), f"0x{int(st.screen):02x}"))
                    return hits, (px, py)
            # restore for next
    return hits, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0d_cp_v1")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--poke", nargs=2, type=int, metavar=("X", "Y"))
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    env, a = build()
    out = {"tag": args.tag, "start": g(env)}
    out["after_push"] = push_right(env, a)
    print("AFTER PUSH", json.dumps(out["after_push"], default=str))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")

    if args.scan:
        hits, landed = scan(env, a)
        out["scan_hits"] = hits
        out["scan_landed"] = landed
        print("SCAN HITS", hits, "LANDED", landed)
        if int(s(env).mode) in CELLAR_MODES:
            for _ in range(240):
                st = s(env)
                if int(st.mode) == 5 and int(st.screen) != ROOM:
                    break
                step(env, a, None)
            out["cellar_settled"] = g(env)
            print("CELLAR SETTLED", json.dumps(out["cellar_settled"], default=str))
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png")

    if args.poke:
        px, py = args.poke
        mem = env.unwrapped.data.memory
        before = g(env)
        mem.assign(int(ADDR_LINK_X), "|u1", px & 0xFF)
        mem.assign(int(ADDR_LINK_Y), "|u1", py & 0xFF)
        out["poke_write"] = {"addr": ["ADDR_LINK_X", "ADDR_LINK_Y"],
                             "before_xy": before["xy"], "to": [px, py],
                             "from_fixture": FROM}
        for _ in range(4):
            step(env, a, None)
        for i in range(120):
            st = s(env)
            if int(st.mode) in CELLAR_MODES or int(st.screen) != ROOM:
                break
            step(env, a, "UP")
        out["after_poke"] = g(env)
        print("AFTER POKE", json.dumps(out["after_poke"], default=str))
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_poke.png")
        if int(s(env).mode) in CELLAR_MODES:
            for _ in range(300):
                st = s(env)
                if int(st.mode) == 5 and int(st.screen) != ROOM:
                    break
                step(env, a, "UP" if i % 2 else None)
            out["cellar_settled"] = g(env)
            print("CELLAR SETTLED", json.dumps(out["cellar_settled"], default=str))
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png")
            if args.save:
                path = save_state(env, GAME_DIR, GAME, args.save)
                src = state_path(GAME_DIR, GAME, FROM)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.3", "phase": "level7_nose_cellar_recon",
                        "track": "recon_fixture", "route_eligible": False,
                        "fixture_only": True, "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [out["poke_write"]],
                        "notes": [
                            "0x0D walk-on OPEN -- position poke to (%d,%d) stands in "
                            "for the unsolved south-face UP push; see l7-handoff "
                            "rr-8t4.3." % (px, py),
                            "Real action first: 0x68 RIGHT push (reveals NE secret "
                            "staircase). Then ONE disclosed ADDR_LINK_X/Y write.",
                            "No Candle/TF/door/key/health/capacity writes. "
                            "UnlimitedHealthAssist traversal aid only.",
                        ],
                    },
                    selected_trial={"ok": True, "state": compact_snapshot(s(env)),
                                    "glance": out["cellar_settled"]},
                    natural_entry=False,
                )
                out["saved_fixture"] = str(path)
                print("SAVED", path)

    (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print("END", json.dumps(g(env), default=str))
    env.close()


if __name__ == "__main__":
    main()
