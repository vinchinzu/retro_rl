"""rr-54ui: screen-align land + body-AI solid arm probe.

Focus (not re-gridding closed dead-ends):
1. Land window only when cam and body share screen (cam>=4 / body scr=4).
2. Target cloud TOP band (oam top ≈ by-16; feet_dy ≈ +8..+20), not body center.
3. Deep-dump status $2C, yspeeds, body tsa/flag/child around contact.
4. Diagnostic (not Clean): fall-from-above poke + optional flag $08 poke.

Disasm notes (lsmmega/mm2 + PRG scan 2026-08-10):
- Body 0x3E AI in bank14 (14_19): spawns rider 0x3D via $F159; rider follows
  body at by-0x14. No CMP solid-arm on rider death in AI — body stays 0x3E.
- Flag bit $08 set/cleared in body state machine (AI phase), not appearing_block.
- Appearing-block solid uses objects_appearing_block=$10 — never set on empty cloud.
- Full-PRG: only 4x CMP #$3E (AI/timer), 0x CMP #$3D — no type whitelist solid.
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

ADDR_OBJ_PTR = 0x0400
ADDR_OBJ_FLAG = 0x0420
ADDR_OBJ_SCREEN = 0x0440
ADDR_OBJ_X = 0x0460
ADDR_OBJ_Y = 0x04A0
ADDR_OBJ_TSA = 0x04E0
ADDR_OBJ_XSPD = 0x0600
ADDR_OBJ_XSPD_FRAC = 0x0620
ADDR_OBJ_YSPD = 0x0640
ADDR_OBJ_YSPD_FRAC = 0x0660
ADDR_OBJ_FS_LO = 0x0680
ADDR_OBJ_FS_HI = 0x06A0
ADDR_ENEMY_HP = 0x06C0
ADDR_STATUS = 0x002C
ADDR_CHILD = 0x0110  # body AI child slot index (disasm)
ADDR_PARENT = 0x0120

LL_MOVE = 0x3D
LL_BODY = 0x3E
LL_BOLT = 0x3F
F_EXIST = 0x80
F_RIGHT = 0x40
F_INVIS = 0x20
F_APPEAR = 0x10
F_AI08 = 0x08  # body AI phase bit from 14_19 ORA #$08

MM_H = 24
# oamcoord_3e spans y=-16..+16 around body y → top ≈ by-16
CLOUD_TOP_OFF = 16
WRAM_STATE_BASE = 93  # fceumm state blob WRAM0 offset (verified)

def btns(*names: str):
    if not names:
        return nes_idle_action()
    return nes_action(*names)

def snap_obj(ram, i: int) -> dict[str, Any]:
    flag = int(ram[ADDR_OBJ_FLAG + i])
    return {
        "i": i,
        "t": int(ram[ADDR_OBJ_PTR + i]),
        "fl": flag,
        "fl_bits": (
            f"e={bool(flag & F_EXIST)} r={bool(flag & F_RIGHT)} "
            f"v={bool(flag & F_INVIS)} a={bool(flag & F_APPEAR)} "
            f"ai8={bool(flag & F_AI08)}"
        ),
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
        "child": int(ram[ADDR_CHILD + i]),
        "parent": int(ram[ADDR_PARENT + i]),
    }

def snap_active(ram) -> list[dict[str, Any]]:
    out = []
    for i in range(32):
        if int(ram[ADDR_OBJ_FLAG + i]) & F_EXIST:
            out.append(snap_obj(ram, i))
    return out

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
        "ys0": int(ram[ADDR_OBJ_YSPD + 0]),
        "ysf0": int(ram[ADDR_OBJ_YSPD_FRAC + 0]),
        "xs0": int(ram[ADDR_OBJ_XSPD + 0]),
        "fallen": is_fallen(ram),
    }

def poke_wram(env, writes: dict[int, int]) -> None:
    """Diagnostic only: patch WRAM via emulator state blob."""
    st = bytearray(env.em.get_state())
    for addr, val in writes.items():
        st[WRAM_STATE_BASE + addr] = val & 0xFF
    env.em.set_state(bytes(st))

def geom(m: dict, b: dict | None) -> dict[str, Any]:
    if not b:
        return {"dx": None, "dy": None, "feet_dy": None, "top_dy": None, "same_scr": None}
    dx = b["x"] - m["sx"]
    dy = b["y"] - m["sy"]
    feet_dy = b["y"] - m["feet_y"]  # 0 = feet at body y (center)
    top_y = b["y"] - CLOUD_TOP_OFF
    top_dy = top_y - m["feet_y"]  # 0 = feet at cloud top
    same = m["cam"] == b["scr"] or (m["cam"] >= 4 and b["scr"] >= 4)
    return {
        "dx": dx,
        "dy": dy,
        "feet_dy": feet_dy,
        "top_y": top_y,
        "top_dy": top_dy,
        "same_scr": same,
        "b_scr": b["scr"],
    }

def run_trial(params: dict[str, Any], out: Path) -> dict[str, Any]:
    env = make_env(GAME, "AirFanPlatform", GAME_DIR, render_mode="rgb_array")
    obs, _ = env.reset()

    camp = params["camp"]
    dx_min = params["dx_min"]
    dx_max = params["dx_max"]
    jh = params["jh"]
    hang = params["hang"]
    sp = params["sp"]
    post = params["post"]
    require_cam4 = params.get("require_cam4", True)
    wait_max = params.get("wait_max", 160)
    max_frames = params.get("max_frames", 380)
    label = params["label"]
    do_poke = params.get("poke")  # None | "fall_top" | "fall_center" | "flag08"

    phase = "walk"
    phase_t = 0
    jump_f = None
    kill_f = None
    first_ll = None
    rider_dead = False
    prev_rh = 99
    hits: list[dict] = []
    stands: list[dict] = []
    contact_log: list[dict] = []
    post_dumps: list[dict] = []
    y_hist: list[int] = []
    max_prog = 0
    max_cam = 0
    best_top = None
    best_same_scr_top = None
    left_ground = False
    prev_sy = None
    poke_done = False
    cam4_at = None

    for f in range(1, max_frames + 1):
        ram = env.get_ram()
        m = meta(ram)
        objs = snap_active(ram)
        lls = [o for o in objs if o["t"] in (LL_MOVE, LL_BODY, LL_BOLT, 6, 0x60, 118)]
        b = body_of(lls)
        r = rider_of(lls)
        g = geom(m, b)
        max_prog = max(max_prog, m["prog"])
        max_cam = max(max_cam, m["cam"])
        if m["cam"] >= 4 and cam4_at is None:
            cam4_at = f
        if m["ft"] == 0:
            left_ground = True

        rh = r["hp"] if r else None
        if r is not None and rh is not None and rh < prev_rh and prev_rh <= 20:
            hits.append(
                {
                    "f": f,
                    "prev": prev_rh,
                    "rh": rh,
                    **{k: m[k] for k in ("sx", "sy", "cam", "prog", "st")},
                    "bx": b["x"] if b else None,
                    "by": b["y"] if b else None,
                    "b_scr": b["scr"] if b else None,
                    "b_tsa": b["tsa"] if b else None,
                    "b_fl": b["fl"] if b else None,
                    "b_child": b["child"] if b else None,
                }
            )
        if r is not None:
            prev_rh = rh if rh is not None else prev_rh
        elif not rider_dead and first_ll is not None and prev_rh < 20:
            rider_dead = True
            kill_f = f

        if b and first_ll is None:
            first_ll = f

        # track best cloud-top approach (top_dy near 0, dx small) after kill
        if b and (rider_dead or hits):
            adx = abs(g["dx"] or 99)
            atd = abs(g["top_dy"] if g["top_dy"] is not None else 99)
            if adx <= 24 and atd <= 30:
                cand = {
                    "f": f,
                    **m,
                    **g,
                    "b_tsa": b["tsa"],
                    "b_fl": b["fl"],
                    "b_ys": b["ys"],
                    "b_xs": b["xs"],
                    "b_child": b["child"],
                    "rider_dead": rider_dead,
                }
                if best_top is None or atd < abs(best_top.get("top_dy", 99)):
                    best_top = cand
                if g["same_scr"] and (
                    best_same_scr_top is None
                    or atd < abs(best_same_scr_top.get("top_dy", 99))
                ):
                    best_same_scr_top = cand

        # stand detect: Y lock post-island
        y_hist.append(m["sy"])
        if len(y_hist) > 12:
            y_hist.pop(0)
        yvar = max(y_hist) - min(y_hist) if len(y_hist) >= 8 else 99
        on_top = (
            b is not None
            and abs(g["dx"] or 99) <= 18
            and abs(g["top_dy"] if g["top_dy"] is not None else 99) <= 6
        )
        near_center = (
            b is not None
            and abs(g["dx"] or 99) <= 18
            and abs(g["feet_dy"] if g["feet_dy"] is not None else 99) <= 4
        )
        if (
            left_ground
            and m["prog"] > 984
            and not m["fallen"]
            and m["hp"] > 0
            and m["sy"] < 100
            and yvar <= 2
            and len(y_hist) >= 8
            and (m["ft"] == 1 or on_top or (near_center and yvar <= 1 and m["ys0"] == 0))
        ):
            stands.append({"f": f, **m, **g, "yvar": yvar, "on_top": on_top, "b": b})

        # dense log when close after kill, or when same-screen
        close = (
            b is not None
            and g["dx"] is not None
            and abs(g["dx"]) <= 28
            and g["top_dy"] is not None
            and abs(g["top_dy"]) <= 28
        )
        if (rider_dead or hits) and close:
            contact_log.append(
                {
                    "f": f,
                    "ph": phase,
                    **m,
                    **g,
                    "body": b,
                    "rider": r,
                    "lls": lls,
                }
            )
        if kill_f and 0 <= f - kill_f <= 16:
            post_dumps.append(
                {
                    "f": f,
                    "meta": m,
                    "geom": g,
                    "body": b,
                    "rider": r,
                    "lls": lls,
                    "types": [(o["i"], o["t"], o["fl"], o["tsa"]) for o in objs],
                }
            )

        # --- diagnostic poke once after kill + contact class ---
        if (
            do_poke
            and rider_dead
            and not poke_done
            and b is not None
            and g["dx"] is not None
            and abs(g["dx"]) <= 20
        ):
            bi = b["i"]
            if do_poke == "fall_top":
                # place well above cloud top, zero yspeed → engine will apply gravity
                target_sy = max(8, b["y"] - CLOUD_TOP_OFF - MM_H - 12)
                poke_wram(
                    env,
                    {
                        ADDR_OBJ_X + 0: b["x"],
                        ADDR_OBJ_Y + 0: target_sy,
                        ADDR_OBJ_YSPD + 0: 0,
                        ADDR_OBJ_YSPD_FRAC + 0: 0,
                    },
                )
                poke_done = True
                contact_log.append(
                    {
                        "f": f,
                        "event": "poke",
                        "mode": do_poke,
                        "target_sy": target_sy,
                        "bx": b["x"],
                        "by": b["y"],
                    }
                )
            elif do_poke == "fall_center":
                target_sy = max(8, b["y"] - MM_H - 8)
                poke_wram(
                    env,
                    {
                        ADDR_OBJ_X + 0: b["x"],
                        ADDR_OBJ_Y + 0: target_sy,
                        ADDR_OBJ_YSPD + 0: 0,
                        ADDR_OBJ_YSPD_FRAC + 0: 0,
                    },
                )
                poke_done = True
                contact_log.append(
                    {
                        "f": f,
                        "event": "poke",
                        "mode": do_poke,
                        "target_sy": target_sy,
                        "bx": b["x"],
                        "by": b["y"],
                    }
                )
            elif do_poke == "flag08":
                # set AI phase bit $08 on body (disasm arms this in one state)
                poke_wram(env, {ADDR_OBJ_FLAG + bi: (b["fl"] | F_AI08) & 0xFF})
                poke_done = True
                contact_log.append(
                    {
                        "f": f,
                        "event": "poke",
                        "mode": do_poke,
                        "fl_before": b["fl"],
                        "fl_after": b["fl"] | F_AI08,
                    }
                )
            elif do_poke == "appear":
                poke_wram(env, {ADDR_OBJ_FLAG + bi: (b["fl"] | F_APPEAR) & 0xFF})
                poke_done = True
                contact_log.append(
                    {
                        "f": f,
                        "event": "poke",
                        "mode": do_poke,
                        "fl_before": b["fl"],
                        "fl_after": b["fl"] | F_APPEAR,
                    }
                )

        # --- control ---
        buttons: list[str] = []
        shoot = False
        dx = g["dx"]
        top_dy = g["top_dy"]
        feet_dy = g["feet_dy"]

        if phase == "walk":
            buttons = ["RIGHT"]
            if m["prog"] >= camp and m["ft"] == 1:
                phase = "camp"
                phase_t = 0
        elif phase == "camp":
            phase_t += 1
            if m["prog"] < camp - 2 and m["ft"] == 1:
                buttons = ["RIGHT"]
            shoot = (f % sp) == 0
            # screen-align gate: prefer jump when body on scr4 and cam approaching 4
            scr_ok = True
            if require_cam4 and b is not None:
                # allow jump from scr3 if body already scr4 and dx in window
                scr_ok = b["scr"] >= 4
            ready = (
                b is not None
                and dx is not None
                and dx_min <= dx <= dx_max
                and m["ft"] == 1
                and scr_ok
            )
            # if require_cam4 and still cam3: wait longer for scroll (walk right edge)
            if require_cam4 and m["cam"] < 4 and m["ft"] == 1 and b is not None:
                if m["prog"] < 1000:
                    buttons = ["RIGHT"]  # push scroll toward cam4
                if ready and m["cam"] >= 3:
                    phase = "jump"
                    phase_t = 0
                    jump_f = f
            elif ready or phase_t >= wait_max:
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
            if not rider_dead:
                buttons = ["RIGHT"]
                if phase_t <= hang:
                    buttons.append("A")
                elif phase_t % 3 == 0:
                    buttons.append("A")
                shoot = (f % sp) == 0
                # if same-screen and above top, start drop early
                if (
                    b is not None
                    and dx is not None
                    and abs(dx) <= 18
                    and top_dy is not None
                    and top_dy > 4
                    and phase_t > hang // 2
                ):
                    buttons = ["RIGHT"] if dx > 2 else (["LEFT"] if dx < -2 else [])
                    shoot = (f % sp) == 0
            else:
                buttons, shoot = postkill(post, phase_t, dx, top_dy, feet_dy, sp, g)
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

        if shoot:
            buttons.append("B")
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
        prev_sy = m["sy"]

    final = meta(env.get_ram())
    final_objs = snap_active(env.get_ram())
    final_ll = [o for o in final_objs if o["t"] in (LL_MOVE, LL_BODY, LL_BOLT, 6)]
    saved = None
    if stands and final["hp"] > 0 and not final["fallen"] and final["sy"] < 90:
        try:
            path = save_state(env, GAME_DIR, GAME, f"AirCloudAlign_{label[:32]}")
            saved = path.name
        except Exception as e:
            saved = f"err:{e}"
    env.close()

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
        "cam4_at": cam4_at,
        "best_top": best_top,
        "best_same_scr_top": best_same_scr_top,
        "max_prog": max_prog,
        "max_cam": max_cam,
        "final": {**final, "ll": final_ll},
        "saved": saved,
        "poke_done": poke_done,
        "post_dumps": post_dumps[:10],
        "contact_near": [
            c
            for c in contact_log
            if c.get("top_dy") is not None and abs(c["top_dy"]) <= 12 and abs(c.get("dx") or 99) <= 16
        ][:16],
        "contact_log_n": len(contact_log),
    }

def postkill(
    post: str,
    phase_t: int,
    dx: int | None,
    top_dy: int | None,
    feet_dy: int | None,
    sp: int,
    g: dict,
) -> tuple[list[str], bool]:
    def steer() -> list[str]:
        if dx is None:
            return ["RIGHT"]
        if dx > 3:
            return ["RIGHT"]
        if dx < -3:
            return ["LEFT"]
        return []

    buttons = steer()
    shoot = (phase_t % sp) == 0

    if post == "drop":
        pass  # pure drop
    elif post == "hover_top":
        # hold A while feet still above cloud top; release in top band
        if top_dy is None or top_dy > 4:
            buttons = steer() + ["A"]
    elif post == "top_band":
        if top_dy is not None:
            if top_dy < -3:
                buttons = steer() + ["A"]  # feet below top → hop
            elif top_dy > 10:
                if phase_t % 2 == 0:
                    buttons = steer() + ["A"]
        else:
            buttons = steer() + ["A"]
    elif post == "wait_cam4_drop":
        # if not same screen, hold A and drift; else drop
        if not g.get("same_scr"):
            buttons = steer() + ["A"]
        # else pure drop
    elif post == "pulse_a":
        if phase_t % 4 < 2:
            buttons = steer() + ["A"]
    else:
        if phase_t % 2 == 0:
            buttons = steer() + ["A"]
    return buttons, shoot

def main() -> None:
    configure_headless()
    out = RECORDINGS_DIR / "air_post4_screen_align"
    out.mkdir(parents=True, exist_ok=True)

    recipes: list[dict[str, Any]] = []

    # Clean screen-align recipes (known kill class + cam/body gate)
    for camp in (968, 972, 976, 980):
        for dx_min, dx_max in [(25, 55), (35, 70), (45, 85), (55, 100)]:
            for hang in (20, 28, 36, 44):
                for post in ("drop", "hover_top", "top_band", "wait_cam4_drop"):
                    for require_cam4 in (True, False):
                        if hang not in (28, 36) and post != "drop":
                            continue
                        if camp not in (972, 976) and dx_min not in (35, 45):
                            continue
                        label = (
                            f"c{camp}_dx{dx_min}-{dx_max}_h{hang}_{post}"
                            f"_cam{'4' if require_cam4 else 'any'}"
                        )
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
                                "require_cam4": require_cam4,
                                "wait_max": 150,
                                "max_frames": 360,
                            }
                        )

    # Deep pin: few best-class with full dumps
    for post in ("drop", "hover_top", "top_band"):
        recipes.append(
            {
                "label": f"deep_c972_dx35-70_h28_{post}",
                "camp": 972,
                "dx_min": 35,
                "dx_max": 70,
                "jh": 12,
                "hang": 28,
                "sp": 3,
                "post": post,
                "require_cam4": True,
                "wait_max": 160,
                "max_frames": 400,
            }
        )

    # Diagnostic pokes (not Clean evidence)
    for poke in ("fall_top", "fall_center", "flag08", "appear"):
        recipes.append(
            {
                "label": f"diag_{poke}_c972",
                "camp": 972,
                "dx_min": 30,
                "dx_max": 80,
                "jh": 12,
                "hang": 28,
                "sp": 3,
                "post": "drop",
                "require_cam4": False,
                "poke": poke,
                "wait_max": 140,
                "max_frames": 340,
            }
        )

    print(f"recipes {len(recipes)}", flush=True)
    results = []
    any5 = False
    any_stand = False
    best_top = None
    best_same = None
    diag_results = []

    for i, p in enumerate(recipes):
        r = run_trial(p, out)
        results.append(r)
        if r["success_cam5"]:
            any5 = True
            print("CAM5!", r["label"], r["final"], flush=True)
        if r["stand_sustained"] >= 4:
            any_stand = True
            print("STAND!", r["label"], r["stand_sustained"], r["stand_sample"][:2], flush=True)
        if r.get("best_top") and (
            best_top is None
            or abs(r["best_top"]["top_dy"]) < abs(best_top["best_top"]["top_dy"])
        ):
            best_top = r
        if r.get("best_same_scr_top") and (
            best_same is None
            or abs(r["best_same_scr_top"]["top_dy"])
            < abs(best_same["best_same_scr_top"]["top_dy"])
        ):
            best_same = r
        if p.get("poke"):
            diag_results.append(r)

        if (i + 1) % 15 == 0 or r["stand_sustained"] > 0 or r["success_cam5"]:
            bt = r.get("best_top")
            bs = r.get("best_same_scr_top")
            print(
                f"[{i+1}/{len(recipes)}] {r['label']} kill={r['rider_dead']} "
                f"p={r['max_prog']} c={r['max_cam']} st={r['stand_sustained']} "
                f"top={bt and (bt['top_dy'], bt['dx'], bt['cam'], bt.get('b_scr'))} "
                f"same={bs and (bs['top_dy'], bs['dx'], bs['cam'])}",
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda r: (
            0 if r["success_cam5"] else 1,
            0 if r["stand_sustained"] >= 4 else 1,
            -r["stand_sustained"],
            abs(r["best_same_scr_top"]["top_dy"]) if r.get("best_same_scr_top") else 99,
            abs(r["best_top"]["top_dy"]) if r.get("best_top") else 99,
            -r["max_prog"],
        ),
    )

    def slim(r: dict) -> dict:
        return {
            k: r[k]
            for k in [
                "label",
                "success_cam5",
                "stand_sustained",
                "n_stand",
                "rider_dead",
                "kill_f",
                "hits",
                "cam4_at",
                "best_top",
                "best_same_scr_top",
                "max_prog",
                "max_cam",
                "final",
                "params",
                "stand_sample",
                "poke_done",
                "contact_near",
            ]
            if k in r
        }

    summary = {
        "n": len(results),
        "n_kills": sum(1 for r in results if r["rider_dead"]),
        "any_cam5": any5,
        "any_stand": any_stand,
        "best_top": slim(best_top) if best_top else None,
        "best_same_scr_top": slim(best_same) if best_same else None,
        "diag": [slim(r) for r in diag_results],
        "top": [slim(r) for r in ranked[:20]],
        "disasm_notes": {
            "body_ai": "14_19 bank14: spawn 0x3D child; rider locks y=by-0x14; no solid rewrite on kill",
            "flag_ai8": "body ORA #$08 / AND #$F7 phase bit — not objects_appearing_block",
            "appear_solid": "appearing_block solid needs flag $10 — never on 0x3E in live dumps",
            "prg_cmp": "full PRG: 4x CMP #$3E (AI only), 0x CMP #$3D — no type solid whitelist",
            "cloud_top": "oamcoord_3e y=-16..+16 → surface ≈ by-16; feet_dy=0 is body center not top",
        },
    }
    write_json_report(out / "summary.json", summary)

    # Keep one deep post_dumps sample
    deep = next((r for r in results if r.get("post_dumps") and r["rider_dead"]), None)
    if deep:
        write_json_report(
            out / "deep_kill.json",
            {
                "label": deep["label"],
                "kill_f": deep["kill_f"],
                "hits": deep["hits"],
                "post_dumps": deep["post_dumps"],
                "best_top": deep.get("best_top"),
                "best_same_scr_top": deep.get("best_same_scr_top"),
                "contact_near": deep.get("contact_near"),
            },
        )

    print(
        json.dumps(
            {k: summary[k] for k in summary if k not in ("top", "diag")},
            indent=2,
            default=str,
        ),
        flush=True,
    )
    print("DIAG:", flush=True)
    for r in diag_results:
        print(
            r["label"],
            "kill",
            r["rider_dead"],
            "st",
            r["stand_sustained"],
            "top",
            r.get("best_top")
            and (r["best_top"]["top_dy"], r["best_top"]["dx"], r["best_top"]["st"], r["best_top"]["ft"]),
            "near",
            len(r.get("contact_near") or []),
            flush=True,
        )
    print("TOP5:", flush=True)
    for t in ranked[:5]:
        print(
            t["label"],
            "kill",
            t["rider_dead"],
            "st",
            t["stand_sustained"],
            "same_top",
            t.get("best_same_scr_top")
            and (
                t["best_same_scr_top"]["top_dy"],
                t["best_same_scr_top"]["dx"],
                t["best_same_scr_top"]["cam"],
            ),
            flush=True,
        )

if __name__ == "__main__":
    main()
