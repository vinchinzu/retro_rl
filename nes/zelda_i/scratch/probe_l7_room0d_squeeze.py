"""Recon: 0x0D south-face UP-push squeeze attempt (rr-8t4.3).

Step 1 of the l7-handoff task: frame-perfect attempt to stand the 0x68
south face (192,160) and UP-push per level9/stairs.py room03 recipe.

  --sweep       full colliding_tile map y=0x88..0xBD (below the top sweep)
  --push        run the verified 0x68 RIGHT push first, then sweep
  --squeeze     raw-input attempts to reach (192,160) then UP-push
  --from-east   squeeze entry from the east pocket going down (default west)
  --xoff N      sub-pixel-ish x target offset for the corridor run
  --trial-a     y=141 gap -> (96,165) lure -> RIGHT y=162-165 -> south face
  --trial-b     after RIGHT push, LEFT+UP / RIGHT+UP clip at (192,133) plug
  --plus-corner y=141 gap -> (144,141) -> DOWN between plus-corners x=144
  --north-up    (144,141) UP toward y<=99 between upper plus-corners

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
SOUTH_BAND_Y = 162
SOUTH_BAND_LO = 161
SOUTH_BAND_HI = 164
SOUTH_FACE = (192, 160)
# Written before the first live south-of-gap trial. Halt on miss.
RAM_CLAIM_SOUTH_GAP = (
    "From Level7Interior0DClearedReconFixture play 0x0D (63,149), DOWN to "
    "y=162 (south-of-gap band between unwalkable y136-158 and south wall "
    "y~166), then RIGHT along y=161-164 to x=192. Claim: that band is "
    "walkable floor to the 0x68 south-face column. Miss if y never enters "
    "161-164, or RIGHT pins at x<=176 (old gap pin 176,155-157)."
)
RAM_CLAIM_SOUTH_STRIP = (
    "From pin (63,149) play 0x0D, DOWN/south-strip toward y=189 (L9 room30 "
    "recipe). Claim: y=189 is reachable despite ROM south door WALL (code 1). "
    "Miss if y never reaches 180. One shot; do not loop."
)
RAM_CLAIM_TRIAL_A = (
    "From pin (63,149), walk to (96,141) via the y=141 mid-floor gap "
    "(UP/RIGHT, never DOWN at x=63), then DOWN to (96,165) (known lure), "
    "then RIGHT along y=162-165 toward x=192, then UP to the 0x68 south "
    "face. Miss if (96,165) is not reached, or RIGHT along that y pins at "
    "x<=176, or the south face is still unreachable. If south face stood, "
    "hold UP. Record block rest and whether mode 9 / $EB=0x7B."
)
RAM_CLAIM_TRIAL_B = (
    "After the verified RIGHT push, from east pocket (192,141), one-frame "
    "LEFT+UP then RIGHT+UP at the (192,133) plug. Claim: a diagonal clip "
    "crosses north of the plug onto the x=192-204 column (the only gap in "
    "the y101-115 band). Miss if still boxed at y>=133."
)
INLAND_X_MIN = 64
WAY_GAP = (96, 141)
WAY_LURE = (96, 165)
SOUTH_BAND_A_LO = 162
SOUTH_BAND_A_HI = 165
PLUS_COL_X = 144
PLUS_SOUTH = (144, 165)
RAM_CLAIM_PLUS_CORNER = (
    "From pin (63,149) to (96,141) (known live y=141 gap), RIGHT to "
    "(144,141), DOWN between plus-corners (statues at x=128 and x=160 "
    "y=157) toward y=165. Miss if DOWN pins at y<=157 or x slides off 144. "
    "If (144,165) or y>=160 at x~144 is reached, RIGHT toward x=192 along "
    "that y, then UP to the 0x68 south face. If south face stood, hold UP "
    "and record block rest + whether mode 9 / $EB=0x7B."
)
NORTH_Y = 93
RAM_CLAIM_NORTH_UP = (
    "From pin to (144,141) (known live), UP toward y<=99 between the upper "
    "plus-corners (128,125)/(160,125). Miss if UP pins at y>=117 or cannot "
    "enter y<=99. If the north corridor is reached, RIGHT along y~93 toward "
    "the 0x68 / NE stair pocket (x~196-208). Before any push."
)
RAM_CLAIM_NORTH_UP_176 = (
    "From (176,141) (west of 0x68, not x=32/48), UP toward y<=99. Miss if "
    "UP pins at y>=117. One fallback column after x=144 miss."
)


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
        "level": int(st.level),
        "xy": [int(st.link_x), int(st.link_y)],
        "colliding_tile": int(st.colliding_tile),
        "facing": int(st.facing),
        "doors": int(st.cur_opened_doors),
        "open_doorway_mask": int(st.open_doorway_mask),
        "room_all_dead": int(st.room_all_dead),
        "keys": int(st.keys), "bombs": int(st.bombs),
        "candle": int(st.candle), "ladder": int(st.ladder),
        "triforce": int(st.triforce),
        "blocks": [(int(o.x), int(o.y), int(o.state))
                   for o in st.objects if int(o.type_id) == 0x68],
        "objs": sorted({f"0x{int(o.type_id):02x}" for o in st.objects
                        if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)}),
    }


def shot(env, tag, label):
    path = RECORDINGS_DIR / f"{tag}_{label}.png"
    save_rgb_png(env.render(), path)
    return str(path)


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


def _stuck_walk(env, a, btn, frames, halt=None):
    """Hold one button. Return (frames_used, halt_reason_or_None)."""
    last, stuck = None, 0
    for i in range(frames):
        st = s(env)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            return i + 1, "left_room"
        xy = (int(st.link_x), int(st.link_y))
        if halt is not None:
            reason = halt(st)
            if reason:
                return i + 1, reason
        if xy == last:
            stuck += 1
            if stuck >= 40:
                return i + 1, "stuck"
        else:
            stuck, last = 0, xy
        step(env, a, btn)
    return frames, None


def south_of_gap(env, a, tag):
    """New vector: y=161-164 between gap and south wall, then RIGHT to x=192."""
    ev = []
    claim = RAM_CLAIM_SOUTH_GAP
    ev.append(("ram_claim", claim))
    ev.append(("start", glance(env), shot(env, tag, "sog_start")))

    used, why = _stuck_walk(
        env, a, "DOWN", 80,
        halt=lambda st: "band_y" if int(st.link_y) >= SOUTH_BAND_Y else None,
    )
    g = glance(env)
    ev.append(("after_down", g, used, why, shot(env, tag, "sog_after_down")))
    y = g["xy"][1]
    if not (SOUTH_BAND_LO <= y <= SOUTH_BAND_HI):
        ev.append(("grade", "MISS", "y_never_entered_161_164", g))
        return ev

    last, stuck, used, why = None, 0, 0, None
    for used in range(1, 201):
        st = s(env)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            why = "left_room"
            break
        x, y = int(st.link_x), int(st.link_y)
        if x >= SOUTH_FACE[0] - 2 and SOUTH_BAND_LO <= y <= SOUTH_BAND_HI:
            why = "at_south_col"
            break
        xy = (x, y)
        if xy == last:
            stuck += 1
            if stuck >= 40:
                why = "stuck"
                break
        else:
            stuck, last = 0, xy
        if y > SOUTH_BAND_HI:
            btn = "UP"
        elif y < SOUTH_BAND_LO:
            btn = "DOWN"
        else:
            btn = "RIGHT"
        step(env, a, btn)
    g = glance(env)
    ev.append(("after_right", g, used, why, shot(env, tag, "sog_after_right")))
    x, y = g["xy"]
    if x <= 176:
        ev.append(("grade", "MISS", "right_pin_x_le_176", g))
        return ev
    if not (SOUTH_BAND_LO <= y <= SOUTH_BAND_HI) or x < SOUTH_FACE[0] - 4:
        ev.append(("grade", "MISS", "band_not_at_south_col", g))
        return ev
    ev.append(("grade", "HIT", "south_band_east_to_x192", g))

    used, why = _stuck_walk(
        env, a, "UP", 80,
        halt=lambda st: "south_face" if (
            abs(int(st.link_x) - SOUTH_FACE[0]) <= 4
            and abs(int(st.link_y) - SOUTH_FACE[1]) <= 4
        ) else None,
    )
    g = glance(env)
    ev.append(("after_up_to_face", g, used, why, shot(env, tag, "sog_south_face")))
    x, y = g["xy"]
    stood = abs(x - SOUTH_FACE[0]) <= 4 and 156 <= y <= 164
    if not stood:
        ev.append(("grade_face", "MISS", "not_south_face", g))
        return ev
    bl0 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    used, why = _stuck_walk(
        env, a, "UP", 90,
        halt=lambda st: "pushed" if any(
            int(o.x) != 192 or int(o.y) != 144
            for o in st.objects if int(o.type_id) == 0x68
        ) else None,
    )
    bl1 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    g = glance(env)
    ev.append(("up_push", {"block_before": bl0, "block_after": bl1,
                           "used": used, "why": why, "glance": g,
                           "png": shot(env, tag, "sog_uppush")}))
    return ev


def south_strip(env, a, tag):
    """L9 room30 recipe: y=189 first. 0x0D south is WALL — one shot."""
    ev = []
    ev.append(("ram_claim", RAM_CLAIM_SOUTH_STRIP))
    ev.append(("start", glance(env), shot(env, tag, "ss_start")))
    used, why = _stuck_walk(
        env, a, "DOWN", 120,
        halt=lambda st: "y189" if int(st.link_y) >= 189 else (
            "y180" if int(st.link_y) >= 180 else None
        ),
    )
    g = glance(env)
    ev.append(("after_down", g, used, why, shot(env, tag, "ss_after_down")))
    y = g["xy"][1]
    if y < 180:
        ev.append(("grade", "MISS", "y_never_reached_180", g))
    elif y >= 189:
        ev.append(("grade", "HIT", "y189_reachable", g))
    else:
        ev.append(("grade", "MISS", "y180_but_not_189", g))
    return ev


def trial_a(env, a, tag):
    """y=141 gap → (96,165) lure → RIGHT y=162-165 → south-face UP-push."""
    ev = []
    ev.append(("ram_claim", RAM_CLAIM_TRIAL_A))
    ev.append(("start", glance(env), shot(env, tag, "a_start")))
    frames = 0
    last_xy = None
    stuck = 0

    def inland_btn(x, y, btn):
        if x < INLAND_X_MIN and btn != "RIGHT":
            return "RIGHT"
        if x <= 63 and btn == "DOWN":
            return "RIGHT"
        return btn

    def walk_to(tx, ty, budget, *, never_down_west=False, hug_y=None):
        nonlocal frames, last_xy, stuck
        for _ in range(budget):
            st = s(env)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                return "left_room", glance(env)
            x, y = int(st.link_x), int(st.link_y)
            if abs(x - tx) <= 2 and abs(y - ty) <= 2:
                return "at_target", glance(env)
            xy = (x, y)
            if xy == last_xy:
                stuck += 1
                if stuck > 0 and stuck % 250 == 0:
                    ev.append(("stuck_250", glance(env), frames,
                               shot(env, tag, f"a_stuck_{frames}")))
                if stuck >= 40 and hug_y is None:
                    return "stuck", glance(env)
                if stuck >= 40 and hug_y is not None:
                    return "stuck", glance(env)
            else:
                stuck, last_xy = 0, xy
            if hug_y is not None:
                lo, hi = hug_y
                if y > hi:
                    btn = "UP"
                elif y < lo:
                    btn = "DOWN"
                elif x < tx - 2:
                    btn = "RIGHT"
                elif x > tx + 2:
                    btn = "LEFT"
                else:
                    btn = "UP" if y > ty else "DOWN"
            elif never_down_west and x <= 63:
                if y > ty:
                    btn = "UP"
                elif x < tx:
                    btn = "RIGHT"
                else:
                    btn = "UP" if y > ty else "RIGHT"
            elif abs(y - ty) > 2:
                btn = "UP" if y > ty else "DOWN"
            elif abs(x - tx) > 2:
                btn = "LEFT" if x > tx else "RIGHT"
            else:
                btn = "UP" if y > ty else "DOWN"
            btn = inland_btn(x, y, btn)
            step(env, a, btn, frames)
            frames += 1
        return "budget", glance(env)

    why, g = walk_to(WAY_GAP[0], WAY_GAP[1], 200, never_down_west=True)
    ev.append(("at_96_141", g, frames, why, shot(env, tag, "a_96_141")))
    x, y = g["xy"]
    if abs(x - WAY_GAP[0]) > 4 or abs(y - WAY_GAP[1]) > 4:
        ev.append(("grade", "MISS", "never_96_141", g))
        return ev

    why, g = walk_to(WAY_LURE[0], WAY_LURE[1], 200)
    ev.append(("at_96_165", g, frames, why, shot(env, tag, "a_96_165")))
    x, y = g["xy"]
    if abs(x - WAY_LURE[0]) > 4 or abs(y - WAY_LURE[1]) > 4:
        ev.append(("grade", "MISS", "never_96_165", g))
        return ev
    ev.append(("grade_lure", "HIT", "reached_96_165", g))

    why, g = walk_to(
        SOUTH_FACE[0], 164, 300, hug_y=(SOUTH_BAND_A_LO, SOUTH_BAND_A_HI),
    )
    ev.append(("after_right_band", g, frames, why, shot(env, tag, "a_band_east")))
    x, y = g["xy"]
    if x <= 176:
        ev.append(("grade", "MISS", "right_pin_x_le_176", g))
        return ev
    if not (SOUTH_BAND_A_LO - 2 <= y <= SOUTH_BAND_A_HI + 2) or x < SOUTH_FACE[0] - 6:
        ev.append(("grade", "MISS", "band_not_at_south_col", g))
        return ev
    ev.append(("grade_band", "HIT", "south_band_east_to_x192", g))

    why, g = walk_to(SOUTH_FACE[0], SOUTH_FACE[1], 120)
    ev.append(("south_face", g, frames, why, shot(env, tag, "a_south_face")))
    x, y = g["xy"]
    stood = abs(x - SOUTH_FACE[0]) <= 4 and 156 <= y <= 166
    if not stood:
        ev.append(("grade", "MISS", "south_face_unreachable", g))
        return ev
    ev.append(("grade_face", "HIT", "stood_south_face", g))

    bl0 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    used, why = _stuck_walk(
        env, a, "UP", 120,
        halt=lambda st: "mode9" if int(st.mode) in CELLAR_MODES else (
            "pushed" if any(
                int(o.x) != 192 or int(o.y) != 144
                for o in st.objects if int(o.type_id) == 0x68
            ) else None
        ),
    )
    frames += used
    bl1 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    g = glance(env)
    ev.append(("up_push", {
        "block_before": bl0, "block_after": bl1, "used": used, "why": why,
        "glance": g, "png": shot(env, tag, "a_uppush"), "frames": frames,
    }))
    dest_cellar = int(s(env).mode) in CELLAR_MODES or int(s(env).screen) == 0x7B
    ev.append(("grade_push", "HIT" if dest_cellar else "push_done",
               {"dest_cellar": dest_cellar, "block_rest": bl1, "glance": g}))
    return ev


def trial_b(env, a, tag):
    """After RIGHT push: diagonal clip at (192,133) plug from east pocket."""
    ev = []
    ev.append(("ram_claim", RAM_CLAIM_TRIAL_B))
    ev.append(("start", glance(env), shot(env, tag, "b_start")))
    push_g = do_push(env, a)
    ev.append(("after_push", push_g, shot(env, tag, "b_pushed")))

    def to_plug():
        reach(env, a, 192, 141, yfirst=False)
        _stuck_walk(
            env, a, "UP", 80,
            halt=lambda st: "plug" if int(st.link_y) <= 133 else None,
        )
        return glance(env)

    pocket = glance(env)
    # after push Link is near the west face; go east pocket then UP to plug
    g_plug = to_plug()
    ev.append(("at_plug", g_plug, shot(env, tag, "b_plug")))
    clips = []
    for combo in (("LEFT", "UP"), ("RIGHT", "UP")):
        # re-seat at the plug so each combo is one frame from the same stand
        g_plug = to_plug()
        st0 = glance(env)
        step(env, a, combo)
        st1 = glance(env)
        row = {
            "combo": combo, "before": st0["xy"], "after": st1["xy"],
            "tile": st1["colliding_tile"], "y": st1["xy"][1],
            "mode": st1["mode"], "screen": st1["screen"],
            "plug_stand": g_plug["xy"],
        }
        clips.append(row)
        ev.append((f"clip_{combo[0]}_{combo[1]}", st1, row,
                   shot(env, tag, f"b_clip_{combo[0]}")))
        if int(s(env).mode) in CELLAR_MODES:
            ev.append(("grade", "HIT", "mode9_on_clip", glance(env)))
            return ev
        if int(s(env).link_y) < 133:
            ev.append(("grade", "HIT", "north_of_plug", glance(env)))
            return ev
    g = glance(env)
    if g["xy"][1] >= 133:
        ev.append(("grade", "MISS", "still_boxed_y_ge_133", g))
    else:
        ev.append(("grade", "HIT", "north_of_plug", g))
    ev.append(("clips", clips))
    ev.append(("end", g, shot(env, tag, "b_end")))
    return ev


def trial_plus_corner(env, a, tag):
    """y=141 gap → (144,141) → DOWN between plus-corners x=144."""
    ev = []
    ev.append(("ram_claim", RAM_CLAIM_PLUS_CORNER))
    ev.append(("start", glance(env), shot(env, tag, "pc_start")))
    frames = 0
    last_xy = None
    stuck = 0

    def inland_btn(x, btn):
        if x < INLAND_X_MIN and btn != "RIGHT":
            return "RIGHT"
        if x <= 63 and btn == "DOWN":
            return "RIGHT"
        return btn

    def walk_to(tx, ty, budget, *, never_down_west=False, hug_x=None):
        nonlocal frames, last_xy, stuck
        for _ in range(budget):
            st = s(env)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                return "left_room", glance(env)
            x, y = int(st.link_x), int(st.link_y)
            if abs(x - tx) <= 2 and abs(y - ty) <= 2:
                return "at_target", glance(env)
            xy = (x, y)
            if xy == last_xy:
                stuck += 1
                if stuck > 0 and stuck % 250 == 0:
                    ev.append(("stuck_250", glance(env), frames,
                               shot(env, tag, f"pc_stuck_{frames}")))
                if stuck >= 40:
                    return "stuck", glance(env)
            else:
                stuck, last_xy = 0, xy
            if hug_x is not None:
                if abs(x - hug_x) > 2:
                    btn = "LEFT" if x > hug_x else "RIGHT"
                elif y < ty:
                    btn = "DOWN"
                elif y > ty:
                    btn = "UP"
                else:
                    btn = "DOWN"
            elif never_down_west and x <= 63:
                if y > ty:
                    btn = "UP"
                elif x < tx:
                    btn = "RIGHT"
                else:
                    btn = "UP" if y > ty else "RIGHT"
            elif abs(y - ty) > 2:
                btn = "UP" if y > ty else "DOWN"
            elif abs(x - tx) > 2:
                btn = "LEFT" if x > tx else "RIGHT"
            else:
                btn = "UP" if y > ty else "DOWN"
            btn = inland_btn(x, btn)
            step(env, a, btn, frames)
            frames += 1
        return "budget", glance(env)

    why, g = walk_to(WAY_GAP[0], WAY_GAP[1], 200, never_down_west=True)
    ev.append(("at_96_141", g, frames, why, shot(env, tag, "pc_96_141")))
    x, y = g["xy"]
    if abs(x - WAY_GAP[0]) > 4 or abs(y - WAY_GAP[1]) > 4:
        ev.append(("grade", "MISS", "never_96_141", g))
        return ev

    why, g = walk_to(PLUS_COL_X, WAY_GAP[1], 200)
    ev.append(("at_144_141", g, frames, why, shot(env, tag, "pc_144_141")))
    x, y = g["xy"]
    if abs(x - PLUS_COL_X) > 4 or abs(y - WAY_GAP[1]) > 4:
        ev.append(("grade", "MISS", "never_144_141", g))
        return ev
    ev.append(("grade_col", "HIT", "at_144_141", g))

    why, g = walk_to(PLUS_SOUTH[0], PLUS_SOUTH[1], 200, hug_x=PLUS_COL_X)
    ev.append(("after_down", g, frames, why, shot(env, tag, "pc_after_down")))
    x, y = g["xy"]
    if abs(x - PLUS_COL_X) > 8:
        ev.append(("grade", "MISS", "x_slid_off_144", g))
        return ev
    if y <= 157:
        ev.append(("grade", "MISS", "down_pin_y_le_157", g))
        return ev
    if y < 160:
        ev.append(("grade", "MISS", "down_not_y160", g))
        return ev
    ev.append(("grade_down", "HIT", "plus_corner_south", g))

    last_xy, stuck, why = None, 0, None
    target_y = max(y, 160)
    for _ in range(300):
        st = s(env)
        if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
            why = "left_room"
            break
        x, y = int(st.link_x), int(st.link_y)
        if x >= SOUTH_FACE[0] - 2 and y >= 160:
            why = "at_south_col"
            break
        xy = (x, y)
        if xy == last_xy:
            stuck += 1
            if stuck >= 40:
                why = "stuck"
                break
        else:
            stuck, last_xy = 0, xy
        if y > target_y + 2:
            btn = "UP"
        elif y < target_y - 2:
            btn = "DOWN"
        elif x < SOUTH_FACE[0] - 2:
            btn = "RIGHT"
        else:
            btn = "UP"
        btn = inland_btn(x, btn)
        step(env, a, btn, frames)
        frames += 1
    else:
        why = "budget"
    g = glance(env)
    ev.append(("after_right", g, frames, why, shot(env, tag, "pc_band_east")))
    x, y = g["xy"]
    if x <= 176:
        ev.append(("grade", "MISS", "right_pin_x_le_176", g))
        return ev

    why, g = walk_to(SOUTH_FACE[0], SOUTH_FACE[1], 120)
    ev.append(("south_face", g, frames, why, shot(env, tag, "pc_south_face")))
    x, y = g["xy"]
    stood = abs(x - SOUTH_FACE[0]) <= 4 and 156 <= y <= 166
    if not stood:
        ev.append(("grade", "MISS", "south_face_unreachable", g))
        return ev
    ev.append(("grade_face", "HIT", "stood_south_face", g))

    bl0 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    used, why = _stuck_walk(
        env, a, "UP", 120,
        halt=lambda st: "mode9" if int(st.mode) in CELLAR_MODES else (
            "pushed" if any(
                int(o.x) != 192 or int(o.y) != 144
                for o in st.objects if int(o.type_id) == 0x68
            ) else None
        ),
    )
    frames += used
    bl1 = [(int(o.x), int(o.y), int(o.state))
           for o in s(env).objects if int(o.type_id) == 0x68]
    g = glance(env)
    ev.append(("up_push", {
        "block_before": bl0, "block_after": bl1, "used": used, "why": why,
        "glance": g, "png": shot(env, tag, "pc_uppush"), "frames": frames,
    }))
    dest_cellar = int(s(env).mode) in CELLAR_MODES or int(s(env).screen) == 0x7B
    ev.append(("grade_push", "HIT" if dest_cellar else "push_done",
               {"dest_cellar": dest_cellar, "block_rest": bl1, "glance": g}))
    return ev


def trial_north_up(env, a, tag, *, col=144):
    """UP toward y<=99 at col (144 between upper plus-corners, or 176)."""
    ev = []
    claim = RAM_CLAIM_NORTH_UP if col == 144 else RAM_CLAIM_NORTH_UP_176
    ev.append(("ram_claim", claim))
    ev.append(("start", glance(env), shot(env, tag, "nu_start")))
    frames = 0
    last_xy = None
    stuck = 0

    def inland_btn(x, btn):
        if x < INLAND_X_MIN and btn != "RIGHT":
            return "RIGHT"
        if x <= 63 and btn == "DOWN":
            return "RIGHT"
        return btn

    def walk_to(tx, ty, budget, *, never_down_west=False, hug_x=None):
        nonlocal frames, last_xy, stuck
        last_xy, stuck = None, 0
        for _ in range(budget):
            st = s(env)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                return "left_room", glance(env)
            x, y = int(st.link_x), int(st.link_y)
            if abs(x - tx) <= 2 and abs(y - ty) <= 2:
                return "at_target", glance(env)
            xy = (x, y)
            if xy == last_xy:
                stuck += 1
                if stuck > 0 and stuck % 250 == 0:
                    ev.append(("stuck_250", glance(env), frames,
                               shot(env, tag, f"nu_stuck_{frames}")))
                if stuck >= 40:
                    return "stuck", glance(env)
            else:
                stuck, last_xy = 0, xy
            if hug_x is not None:
                if abs(x - hug_x) > 2:
                    btn = "LEFT" if x > hug_x else "RIGHT"
                elif y > ty:
                    btn = "UP"
                elif y < ty:
                    btn = "DOWN"
                else:
                    btn = "UP"
            elif never_down_west and x <= 63:
                if y > ty:
                    btn = "UP"
                elif x < tx:
                    btn = "RIGHT"
                else:
                    btn = "UP" if y > ty else "RIGHT"
            elif abs(y - ty) > 2:
                btn = "UP" if y > ty else "DOWN"
            elif abs(x - tx) > 2:
                btn = "LEFT" if x > tx else "RIGHT"
            else:
                btn = "UP" if y > ty else "DOWN"
            btn = inland_btn(x, btn)
            step(env, a, btn, frames)
            frames += 1
        return "budget", glance(env)

    why, g = walk_to(WAY_GAP[0], WAY_GAP[1], 200, never_down_west=True)
    ev.append(("at_96_141", g, frames, why, shot(env, tag, "nu_96_141")))
    why, g = walk_to(col, WAY_GAP[1], 200)
    ev.append((f"at_{col}_141", g, frames, why, shot(env, tag, f"nu_{col}_141")))
    x, y = g["xy"]
    if abs(x - col) > 4 or abs(y - WAY_GAP[1]) > 4:
        ev.append(("grade", "MISS", f"never_{col}_141", g))
        return ev

    why, g = walk_to(col, NORTH_Y, 200, hug_x=col)
    ev.append((f"after_up_{col}", g, frames, why,
               shot(env, tag, f"nu_after_up_{col}")))
    x, y = g["xy"]
    if y <= 99:
        ev.append(("grade_up", "HIT", "y_le_99", g))
        last_xy, stuck, why = None, 0, None
        for _ in range(300):
            st = s(env)
            if int(st.screen) != ROOM or int(st.mode) in CELLAR_MODES:
                why = "left_room"
                break
            x, y = int(st.link_x), int(st.link_y)
            if x >= 196 and y <= 99:
                why = "at_ne_pocket"
                break
            xy = (x, y)
            if xy == last_xy:
                stuck += 1
                if stuck >= 40:
                    why = "stuck"
                    break
            else:
                stuck, last_xy = 0, xy
            if y > 99:
                btn = "UP"
            elif y < 90:
                btn = "DOWN"
            else:
                btn = "RIGHT"
            btn = inland_btn(x, btn)
            step(env, a, btn, frames)
            frames += 1
        else:
            why = "budget"
        g = glance(env)
        ev.append(("east_along_north", g, frames, why,
                   shot(env, tag, "nu_north_east")))
        return ev
    ev.append(("grade", "MISS",
               f"up_{col}_pin_y_ge_117" if y >= 117 else f"up_{col}_not_y99", g))
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
    ap.add_argument("--south-of-gap", action="store_true")
    ap.add_argument("--south-strip", action="store_true")
    ap.add_argument("--trial-a", action="store_true")
    ap.add_argument("--trial-b", action="store_true")
    ap.add_argument("--plus-corner", action="store_true")
    ap.add_argument("--north-up", action="store_true")
    ap.add_argument("--north-up-col", type=int, default=144)
    ap.add_argument("--glance", action="store_true")
    ap.add_argument("--xoff", type=int, default=0)
    ap.add_argument("--infinite-life", action="store_true", default=True)
    ap.add_argument("--no-video", action="store_true", default=True)
    args = ap.parse_args()

    env, a = build()
    out = {"tag": args.tag, "start": glance(env),
           "start_png": shot(env, args.tag, "start")}
    if args.glance and not (args.south_of_gap or args.south_strip
                            or args.squeeze or args.sweep or args.push
                            or args.around_east or args.wiggle
                            or args.trial_a or args.trial_b
                            or args.plus_corner or args.north_up):
        out["end"] = glance(env)
        print(json.dumps(out, indent=2, default=str))
        env.close()
        return
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
    if args.south_of_gap:
        out["ram_claim"] = RAM_CLAIM_SOUTH_GAP
        out["south_of_gap"] = south_of_gap(env, a, args.tag)
    if args.south_strip:
        out["ram_claim_south_strip"] = RAM_CLAIM_SOUTH_STRIP
        out["south_strip"] = south_strip(env, a, args.tag)
    if args.trial_a:
        out["ram_claim"] = RAM_CLAIM_TRIAL_A
        out["trial_a"] = trial_a(env, a, args.tag)
    if args.trial_b:
        out["ram_claim"] = RAM_CLAIM_TRIAL_B
        out["trial_b"] = trial_b(env, a, args.tag)
    if args.plus_corner:
        out["ram_claim"] = RAM_CLAIM_PLUS_CORNER
        out["plus_corner"] = trial_plus_corner(env, a, args.tag)
    if args.north_up:
        out["ram_claim"] = (
            RAM_CLAIM_NORTH_UP if args.north_up_col == 144
            else RAM_CLAIM_NORTH_UP_176
        )
        out["north_up"] = trial_north_up(
            env, a, args.tag, col=args.north_up_col,
        )
    out["end"] = glance(env)
    out["end_png"] = shot(env, args.tag, "end")
    print(json.dumps(out, indent=2, default=str))
    path = RECORDINGS_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    env.close()


if __name__ == "__main__":
    main()
