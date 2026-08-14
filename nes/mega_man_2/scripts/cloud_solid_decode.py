"""rr-54ui: decode empty Thunder Chariot object-solid + feet-on-top land.

From AirFanPlatform:
  - pulse-B kill rider 0x3D (known Clean)
  - dump aobject_tsa=$4E0, flag, type, speeds around kill
  - attempt stand from ABOVE (sy < by; feet ≈ by) not body-center sy==by

MM2 Y is sprite/object top-ish; Mega Man ~24px tall → feet_y ≈ sy+24.
Stand if feet land near cloud top (by) while descending and X overlaps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.ram import (
    ADDR_CAMERA_X,
    ADDR_CAMERA_X_SCREEN,
    ADDR_HEALTH,
    ADDR_INVULN_TIMER,
    ADDR_TILE_CENTER,
    ADDR_TILE_FEET,
    ADDR_TILE_OVERLAP,
    camera_progress_x,
    is_fallen,
    player_screen_x,
    player_screen_y,
)
from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report

# Object parallel arrays (slot i)
ADDR_OBJ_PTR = 0x0400
ADDR_OBJ_FLAG = 0x0420
ADDR_OBJ_SCREEN = 0x0440
ADDR_OBJ_X = 0x0460
ADDR_OBJ_X_FRAC = 0x0480
ADDR_OBJ_Y = 0x04A0
ADDR_OBJ_Y_FRAC = 0x04C0
ADDR_OBJ_TSA = 0x04E0
ADDR_OBJ_XSPD = 0x0600
ADDR_OBJ_XSPD_FRAC = 0x0620
ADDR_OBJ_YSPD = 0x0640
ADDR_OBJ_YSPD_FRAC = 0x0660
ADDR_OBJ_FS_LO = 0x0680
ADDR_OBJ_FS_HI = 0x06A0
ADDR_ENEMY_HP = 0x06C0  # slot-aligned: HP for object i often at 0x06C0+i

# Megaman status / extra collision-ish
ADDR_STATUS = 0x002C
ADDR_TILE_FEET_V = ADDR_TILE_FEET

LL_MOVE = 0x3D
LL_BODY = 0x3E
LL_BOLT = 0x3F
LL_TYPES = {LL_MOVE, LL_BODY, LL_BOLT, 6, 118, 0x60}  # + killed/explosion/life
# flags from lsmmega/mm2 constants/flags.asm
F_EXIST = 0x80
F_RIGHT = 0x40
F_INVIS = 0x20
F_APPEAR = 0x10

# approx sprite height for feet geometry
MM_H = 24
CLOUD_TOP_SLACK = 6  # feet near by ± this counts as surface

def snap_all_active(ram) -> list[dict[str, Any]]:
    out = []
    for i in range(32):
        flag = int(ram[ADDR_OBJ_FLAG + i])
        if not (flag & F_EXIST):
            continue
        t = int(ram[ADDR_OBJ_PTR + i])
        out.append(
            {
                "i": i,
                "t": t,
                "fl": flag,
                "fl_bits": f"e={bool(flag&F_EXIST)} r={bool(flag&F_RIGHT)} "
                f"v={bool(flag&F_INVIS)} a={bool(flag&F_APPEAR)}",
                "scr": int(ram[ADDR_OBJ_SCREEN + i]),
                "x": int(ram[ADDR_OBJ_X + i]),
                "y": int(ram[ADDR_OBJ_Y + i]),
                "tsa": int(ram[ADDR_OBJ_TSA + i]),
                "xs": int(ram[ADDR_OBJ_XSPD + i]),
                "xsf": int(ram[ADDR_OBJ_XSPD_FRAC + i]),
                "ys": int(ram[ADDR_OBJ_YSPD + i]),
                "ysf": int(ram[ADDR_OBJ_YSPD_FRAC + i]),
                "fs_lo": int(ram[ADDR_OBJ_FS_LO + i]),
                "fs_hi": int(ram[ADDR_OBJ_FS_HI + i]),
                "hp": int(ram[ADDR_ENEMY_HP + i]),
            }
        )
    return out

def snap_ll(ram) -> list[dict[str, Any]]:
    return [o for o in snap_all_active(ram) if o["t"] in LL_TYPES or o["t"] in (LL_MOVE, LL_BODY)]

def body_of(objs: list[dict]) -> dict | None:
    bs = [o for o in objs if o["t"] == LL_BODY]
    return bs[0] if bs else None

def rider_of(objs: list[dict]) -> dict | None:
    rs = [o for o in objs if o["t"] == LL_MOVE]
    return rs[0] if rs else None

def meta(ram) -> dict[str, Any]:
    sy = player_screen_y(ram)
    return {
        "cam": int(ram[ADDR_CAMERA_X_SCREEN]),
        "cx": int(ram[ADDR_CAMERA_X]),
        "prog": camera_progress_x(ram),
        "sx": player_screen_x(ram),
        "sy": sy,
        "feet_y": sy + MM_H,
        "hp": int(ram[ADDR_HEALTH]),
        "ft": int(ram[ADDR_TILE_FEET]),
        "tc": int(ram[ADDR_TILE_CENTER]),
        "to": int(ram[ADDR_TILE_OVERLAP]),
        "inv": int(ram[ADDR_INVULN_TIMER]),
        "st": int(ram[ADDR_STATUS]),
        "fallen": is_fallen(ram),
    }

def btns(*names: str):
    if not names:
        return nes_idle_action()
    return nes_action(*names)

def run_recipe(params: dict[str, Any], out: Path) -> dict[str, Any]:
    """One Clean trial: camp → jump/shoot → postkill land mode."""
    env = make_env(GAME, "AirFanPlatform", GAME_DIR, render_mode="rgb_array")
    obs, _ = env.reset()

    camp = params["camp"]
    dx_min = params["dx_min"]
    dx_max = params["dx_max"]
    jh = params["jh"]
    hang = params["hang"]
    sp = params["sp"]  # B pulse period
    post = params["post"]  # post-kill air strategy name
    wait_max = params.get("wait_max", 120)
    max_frames = params.get("max_frames", 320)
    label = params["label"]

    phase = "walk"
    phase_t = 0
    jump_f = None
    kill_f = None
    first_ll = None
    rider_dead = False
    prev_rh = 99
    hits: list[dict] = []
    kill_snapshot: dict | None = None
    post_dumps: list[dict] = []
    geom_log: list[dict] = []
    stands: list[dict] = []
    y_hist: list[int] = []
    max_prog = 0
    max_cam = 0
    min_dist = 999.0
    best_near = None
    best_feet_top = None  # best |feet_y - by| while dx small after kill
    left_ground = False
    prev_sy = None

    for f in range(1, max_frames + 1):
        ram = env.get_ram()
        m = meta(ram)
        objs = snap_all_active(ram)
        lls = [o for o in objs if o["t"] in (LL_MOVE, LL_BODY, LL_BOLT, 6, 0x60, 118)]
        b = body_of(lls)
        r = rider_of(lls)
        max_prog = max(max_prog, m["prog"])
        max_cam = max(max_cam, m["cam"])
        if m["ft"] == 0:
            left_ground = True

        # rider HP / kill detect
        rh = r["hp"] if r else None
        if r is not None and rh is not None and rh < prev_rh and prev_rh <= 20:
            hits.append(
                {
                    "f": f,
                    "prev": prev_rh,
                    "rh": rh,
                    "sx": m["sx"],
                    "sy": m["sy"],
                    "bx": b["x"] if b else None,
                    "by": b["y"] if b else None,
                    "rx": r["x"],
                    "ry": r["y"],
                    "r_tsa": r["tsa"],
                    "b_tsa": b["tsa"] if b else None,
                    "r_fl": r["fl"],
                    "b_fl": b["fl"] if b else None,
                }
            )
        if r is not None:
            prev_rh = rh if rh is not None else prev_rh
        elif not rider_dead and first_ll is not None and prev_rh < 20:
            # rider slot gone after damage
            rider_dead = True
            kill_f = f
            kill_snapshot = {
                "f": f,
                "meta": m,
                "body": b,
                "all_ll": lls,
                "hits": list(hits),
            }

        if b and first_ll is None:
            first_ll = f

        dx = dy = feet_dy = None
        if b:
            dx = b["x"] - m["sx"]
            dy = b["y"] - m["sy"]  # >0 ⇒ player above cloud top
            feet_dy = b["y"] - m["feet_y"]  # 0 ⇒ feet on cloud top
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_near = {
                    "f": f,
                    **m,
                    "bx": b["x"],
                    "by": b["y"],
                    "dx": dx,
                    "dy": dy,
                    "feet_dy": feet_dy,
                    "b_tsa": b["tsa"],
                    "b_fl": b["fl"],
                    "rh": rh,
                }
            if rider_dead or (r is None and hits):
                adx = abs(dx)
                af = abs(feet_dy)
                if adx <= 20 and (
                    best_feet_top is None
                    or af < abs(best_feet_top.get("feet_dy", 99))
                    or (af == abs(best_feet_top.get("feet_dy", 99)) and adx < abs(best_feet_top.get("dx", 99)))
                ):
                    best_feet_top = {
                        "f": f,
                        **m,
                        "bx": b["x"],
                        "by": b["y"],
                        "dx": dx,
                        "dy": dy,
                        "feet_dy": feet_dy,
                        "b_tsa": b["tsa"],
                        "b_fl": b["fl"],
                        "b_ys": b["ys"],
                        "b_xs": b["xs"],
                    }

        # stand: post-island, Y flat, feet near cloud top or ft==1
        y_hist.append(m["sy"])
        if len(y_hist) > 10:
            y_hist.pop(0)
        yvar = max(y_hist) - min(y_hist) if len(y_hist) >= 6 else 99
        near_top = (
            b is not None
            and abs(m["sx"] - b["x"]) <= 18
            and abs(m["feet_y"] - b["y"]) <= CLOUD_TOP_SLACK
        )
        body_overlap = (
            b is not None
            and abs(m["sx"] - b["x"]) <= 18
            and abs(m["sy"] - b["y"]) <= 12
        )
        descending = prev_sy is not None and m["sy"] >= prev_sy
        if (
            left_ground
            and m["prog"] > 984
            and not m["fallen"]
            and m["hp"] > 0
            and m["sy"] < 100
            and yvar <= 2
            and len(y_hist) >= 6
            and (m["ft"] == 1 or near_top or (body_overlap and yvar <= 1))
        ):
            stands.append(
                {
                    "f": f,
                    **m,
                    "near_top": near_top,
                    "body_overlap": body_overlap,
                    "yvar": yvar,
                    "dx": dx,
                    "dy": dy,
                    "feet_dy": feet_dy,
                    "b": b,
                    "descending": descending,
                }
            )
        prev_sy = m["sy"]

        # dump around kill ± 20f and every frame in land window
        dump_now = False
        if kill_f and abs(f - kill_f) <= 24:
            dump_now = True
        if hits and f - hits[-1]["f"] <= 4:
            dump_now = True
        if b and dx is not None and abs(dx) <= 30 and m["prog"] > 990:
            dump_now = True
        if dump_now:
            geom_log.append(
                {
                    "f": f,
                    "ph": phase,
                    **m,
                    "dx": dx,
                    "dy": dy,
                    "feet_dy": feet_dy,
                    "rider": r,
                    "body": b,
                    "lls": lls,
                }
            )
            if kill_f and f - kill_f <= 12:
                post_dumps.append(
                    {
                        "f": f,
                        "objs_ll": lls,
                        "all_types": [(o["i"], o["t"], o["fl"], o["tsa"], o["x"], o["y"]) for o in objs],
                    }
                )

        # --- control ---
        buttons: list[str] = []
        shoot = False

        if phase == "walk":
            buttons = ["RIGHT"]
            if m["prog"] >= camp and m["ft"] == 1:
                phase = "camp"
                phase_t = 0
        elif phase == "camp":
            phase_t += 1
            if m["prog"] < camp - 2 and m["ft"] == 1:
                buttons = ["RIGHT"]
            # pulse B while camping so first shots ready
            shoot = (f % sp) == 0
            ready = (
                b is not None
                and dx is not None
                and dx_min <= dx <= dx_max
                and m["ft"] == 1
            )
            if ready or phase_t >= wait_max:
                phase = "jump"
                phase_t = 0
                jump_f = f
            if m["ft"] != 1:
                phase = "jump"
                phase_t = 0
                jump_f = f
        elif phase == "jump":
            phase_t += 1
            buttons = ["RIGHT", "A"]
            shoot = (f % sp) == 0
            if phase_t >= jh or m["ft"] == 0:
                phase = "air"
                phase_t = 0
        elif phase == "air":
            phase_t += 1
            # pre-kill: hang + pulse + approach
            if not rider_dead:
                buttons = ["RIGHT"]
                if phase_t <= hang:
                    buttons.append("A")
                elif phase_t % 3 == 0:
                    buttons.append("A")
                shoot = (f % sp) == 0
                # if already above cloud with good X, stop hang early to fall onto top
                if (
                    b is not None
                    and dx is not None
                    and abs(dx) <= 16
                    and dy is not None
                    and 8 <= dy <= 40
                    and phase_t > hang // 2
                ):
                    # release A to drop toward top
                    buttons = ["RIGHT"] if dx > 2 else (["LEFT"] if dx < -2 else [])
                    shoot = (f % sp) == 0
            else:
                # post-kill land strategies
                buttons, shoot = postkill_control(post, f, phase_t, m, b, dx, dy, feet_dy, sp)
                if m["ft"] == 1 and m["prog"] > 984:
                    phase = "ride"
                    phase_t = 0
                if stands and len(stands) >= 3:
                    phase = "ride"
                    phase_t = 0
        elif phase == "ride":
            phase_t += 1
            buttons = ["RIGHT"]
            if m["ft"] == 0 and phase_t % 2 == 0:
                buttons.append("A")
            if m["ft"] == 1 and phase_t % 30 < 8:
                buttons.append("A")  # hop along chain

        if shoot:
            buttons.append("B")
        # dedupe
        ub: list[str] = []
        for x in buttons:
            if x not in ub:
                ub.append(x)
        action = btns(*ub) if ub else nes_idle_action()
        obs, _, term, trunc, _ = env.step(action)

        m2 = meta(env.get_ram())
        if m2["fallen"] or m2["hp"] <= 0 or term or trunc or m2["cam"] >= 5:
            if stands or m2["cam"] >= 5 or rider_dead:
                try:
                    save_rgb_png(
                        obs,
                        out / f"{label}_end_p{m2['prog']}_c{m2['cam']}_sy{m2['sy']}.png",
                    )
                except Exception:
                    pass
            break

    final = meta(env.get_ram())
    final_objs = snap_all_active(env.get_ram())
    final_ll = [o for o in final_objs if o["t"] in (LL_MOVE, LL_BODY, LL_BOLT, 6, 0x60)]
    saved = None
    if stands and final["hp"] > 0 and not final["fallen"] and final["sy"] < 90:
        try:
            path = save_state(env, GAME_DIR, GAME, f"AirCloudSolid_{label[:36]}")
            saved = path.name
        except Exception as e:
            saved = f"err:{e}"
    env.close()

    # sustained stand ≥4 consecutive
    real = [s for s in stands if jump_f is None or s["f"] > jump_f]
    sustained = 0
    if real:
        run = 1
        best_run = 1
        for i in range(1, len(real)):
            if real[i]["f"] == real[i - 1]["f"] + 1:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 1
        sustained = best_run

    return {
        "label": label,
        "params": {k: v for k, v in params.items() if k != "label"},
        "success_cam5": final["cam"] >= 5 and final["hp"] > 0 and not final["fallen"],
        "n_stand": len(real),
        "stand_sustained": sustained,
        "stand_sample": real[:6],
        "rider_dead": rider_dead,
        "kill_f": kill_f,
        "hits": hits,
        "kill_snapshot": kill_snapshot,
        "best_near": best_near,
        "best_feet_top": best_feet_top,
        "min_dist": round(min_dist, 1) if min_dist < 999 else None,
        "max_prog": max_prog,
        "max_cam": max_cam,
        "final": {**final, "ll": final_ll},
        "saved": saved,
        "geom_near_kill": [g for g in geom_log if kill_f and abs(g["f"] - kill_f) <= 16][:20],
        "post_dumps": post_dumps[:8],
        "geom_feet_close": [
            g
            for g in geom_log
            if g.get("feet_dy") is not None
            and abs(g["feet_dy"]) <= 10
            and g.get("dx") is not None
            and abs(g["dx"]) <= 20
        ][:12],
    }

def postkill_control(
    post: str,
    f: int,
    phase_t: int,
    m: dict,
    b: dict | None,
    dx: int | None,
    dy: int | None,
    feet_dy: int | None,
    sp: int,
) -> tuple[list[str], bool]:
    """Return (buttons, shoot) after rider kill."""
    shoot = False
    buttons: list[str] = []

    # default steer X toward cloud
    def steer() -> list[str]:
        if dx is None:
            return ["RIGHT"]
        if dx > 4:
            return ["RIGHT"]
        if dx < -4:
            return ["LEFT"]
        return []

    if post == "drop":
        # pure drop: no A, steer X — land from above
        buttons = steer()
    elif post == "drop_late_a":
        # drop first 12f then micro A if feet passed top
        buttons = steer()
        if phase_t > 12 and feet_dy is not None and feet_dy < -4:
            buttons.append("A")  # too low — hop
    elif post == "hold_a":
        buttons = steer() + ["A"]
    elif post == "pulse_a":
        buttons = steer()
        if phase_t % 4 < 2:
            buttons.append("A")
    elif post == "hover_then_drop":
        # hold A while feet_dy > 8 (still above), release when approaching top
        buttons = steer()
        if feet_dy is None or feet_dy > 6:
            buttons.append("A")
        # else drop through stand band
    elif post == "feet_band":
        # try to keep feet_y near by: A if feet below top, release if above
        buttons = steer()
        if feet_dy is not None:
            if feet_dy < -2:  # feet below cloud top → jump
                buttons.append("A")
            elif feet_dy > 8:  # still high above → light hang
                if phase_t % 2 == 0:
                    buttons.append("A")
            # else in band: neutral fall
        else:
            buttons.append("A")
    elif post == "nudge_up":
        # short hop if overlapping body center (sy~by) to get above then drop
        buttons = steer()
        if dy is not None and dy <= 4:
            buttons.append("A")
        elif feet_dy is not None and feet_dy < 0:
            buttons.append("A")
    else:
        buttons = steer()
        if phase_t % 2 == 0:
            buttons.append("A")

    # keep pulsing B in case rider not fully dead / second LL
    shoot = (f % sp) == 0
    return buttons, shoot

def main() -> None:
    configure_headless()
    out = RECORDINGS_DIR / "air_post4_cloud_solid"
    out.mkdir(parents=True, exist_ok=True)

    posts = [
        "drop",
        "drop_late_a",
        "hold_a",
        "pulse_a",
        "hover_then_drop",
        "feet_band",
        "nudge_up",
    ]
    recipes: list[dict[str, Any]] = []

    # Core kill recipes known to work (dps_search / RED pin class)
    for camp in (968, 972, 976):
        for dx_min, dx_max in [(20, 50), (30, 70), (40, 80), (50, 90)]:
            for jh in (10, 12, 14):
                for hang in (16, 24, 28, 36):
                    for sp in (3, 4):
                        for post in posts:
                            # prune
                            if hang not in (24, 28) and post not in ("drop", "feet_band", "hover_then_drop"):
                                continue
                            if jh != 12 and post not in ("drop", "feet_band"):
                                continue
                            if camp != 972 and dx_min not in (30, 40) and post != "drop":
                                continue
                            label = (
                                f"c{camp}_dx{dx_min}-{dx_max}_jh{jh}_h{hang}"
                                f"_sp{sp}_{post}"
                            )
                            recipes.append(
                                {
                                    "label": label,
                                    "camp": camp,
                                    "dx_min": dx_min,
                                    "dx_max": dx_max,
                                    "jh": jh,
                                    "hang": hang,
                                    "sp": sp,
                                    "post": post,
                                    "wait_max": 110,
                                    "max_frames": 300,
                                }
                            )

    # Extra: jump earlier with higher hang to stay above cloud for drop
    for camp in (970, 974):
        for dx_min, dx_max in [(50, 100), (60, 110)]:
            for hang in (32, 40, 48):
                for post in ("drop", "hover_then_drop", "feet_band"):
                    label = f"hi_c{camp}_dx{dx_min}-{dx_max}_h{hang}_{post}"
                    recipes.append(
                        {
                            "label": label,
                            "camp": camp,
                            "dx_min": dx_min,
                            "dx_max": dx_max,
                            "jh": 12,
                            "hang": hang,
                            "sp": 3,
                            "post": post,
                            "wait_max": 130,
                            "max_frames": 320,
                        }
                    )

    print(f"recipes {len(recipes)}", flush=True)
    results = []
    any5 = False
    any_stand = False
    best_kill = None
    best_feet = None
    best_stand = None
    decode_samples: list[dict] = []

    for i, p in enumerate(recipes):
        r = run_recipe(p, out)
        results.append(r)
        if r["success_cam5"]:
            any5 = True
            print("CAM5!", r["label"], r["final"], flush=True)
        if r["stand_sustained"] >= 4:
            any_stand = True
            print(
                "STAND!",
                r["label"],
                "sust",
                r["stand_sustained"],
                r["stand_sample"][:2],
                flush=True,
            )
        if r["rider_dead"] and (
            best_kill is None
            or (r["best_feet_top"] and best_kill.get("best_feet_top") is None)
            or (
                r.get("best_feet_top")
                and best_kill.get("best_feet_top")
                and abs(r["best_feet_top"]["feet_dy"])
                < abs(best_kill["best_feet_top"]["feet_dy"])
            )
        ):
            best_kill = r
        if r.get("best_feet_top") and (
            best_feet is None
            or abs(r["best_feet_top"]["feet_dy"]) < abs(best_feet["best_feet_top"]["feet_dy"])
        ):
            best_feet = r
        if r["stand_sustained"] > 0 and (
            best_stand is None or r["stand_sustained"] > best_stand["stand_sustained"]
        ):
            best_stand = r

        # keep a few decode dumps when kill happens
        if r["rider_dead"] and r.get("post_dumps") and len(decode_samples) < 6:
            decode_samples.append(
                {
                    "label": r["label"],
                    "kill_f": r["kill_f"],
                    "hits": r["hits"],
                    "kill_snapshot": r["kill_snapshot"],
                    "post_dumps": r["post_dumps"][:4],
                    "geom_near_kill": r["geom_near_kill"][:8],
                    "best_feet_top": r["best_feet_top"],
                }
            )

        if (i + 1) % 20 == 0 or r["stand_sustained"] > 0 or (
            r["rider_dead"] and r.get("best_feet_top") and abs(r["best_feet_top"]["feet_dy"]) <= 6
        ):
            bft = r.get("best_feet_top")
            print(
                f"[{i+1}/{len(recipes)}] {r['label']} kill={r['rider_dead']} "
                f"hits={len(r['hits'])} p={r['max_prog']} c={r['max_cam']} "
                f"st={r['stand_sustained']} "
                f"feet={bft and (bft['feet_dy'], bft['dx'], bft['sy'], bft['by'])}",
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda r: (
            0 if r["success_cam5"] else 1,
            0 if r["stand_sustained"] >= 4 else 1,
            -r["stand_sustained"],
            abs(r["best_feet_top"]["feet_dy"]) if r.get("best_feet_top") else 99,
            abs(r["best_feet_top"]["dx"]) if r.get("best_feet_top") else 99,
            -r["max_prog"],
        ),
    )
    top = []
    for r in ranked[:25]:
        top.append(
            {
                k: r[k]
                for k in [
                    "label",
                    "success_cam5",
                    "stand_sustained",
                    "n_stand",
                    "rider_dead",
                    "kill_f",
                    "hits",
                    "best_near",
                    "best_feet_top",
                    "max_prog",
                    "max_cam",
                    "min_dist",
                    "final",
                    "params",
                    "stand_sample",
                ]
            }
        )

    summary = {
        "n": len(results),
        "n_kills": sum(1 for r in results if r["rider_dead"]),
        "any_cam5": any5,
        "any_stand": any_stand,
        "best_stand": (
            {
                k: best_stand[k]
                for k in [
                    "label",
                    "stand_sustained",
                    "stand_sample",
                    "best_feet_top",
                    "max_prog",
                    "params",
                    "final",
                ]
            }
            if best_stand
            else None
        ),
        "best_feet_top": (
            {
                k: best_feet[k]
                for k in [
                    "label",
                    "best_feet_top",
                    "hits",
                    "kill_f",
                    "max_prog",
                    "stand_sustained",
                    "params",
                    "geom_feet_close",
                ]
            }
            if best_feet
            else None
        ),
        "decode_samples": decode_samples,
        "top": top,
    }
    write_json_report(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k not in ("top", "decode_samples")}, indent=2))
    print("DECODE sample count", len(decode_samples), flush=True)
    if decode_samples:
        s0 = decode_samples[0]
        print("KILL sample label", s0["label"], "hits", s0["hits"], flush=True)
        print("post_dumps0", json.dumps(s0["post_dumps"][:2], indent=2)[:2000], flush=True)
    print("TOP5:", flush=True)
    for t in top[:5]:
        print(
            t["label"],
            "kill",
            t["rider_dead"],
            "st",
            t["stand_sustained"],
            "feet",
            t.get("best_feet_top"),
            flush=True,
        )

if __name__ == "__main__":
    main()
