"""rr-54ui v2: wait-for-LL at safe solid, jump when dx+altitude good.

Prior grid closed X to ~7px but at sy~96 (too low). This probe:
- camps on solid (prog 960–975), no edge wobble
- jumps when LL dx in band AND body y still high
- tracks real stand: feet==1 past 984 OR multi-frame Y lock on cloud band
- optional contact/invuln land
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
    ADDR_TILE_FEET,
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
ADDR_ENEMY_HP = 0x06C0  # base; slot i often at 0x06C0+i

LL_TYPES = {0x3D, 0x3E, 0x3F}
LL_BODY = 0x3E

def snap_ll(ram) -> list[dict[str, Any]]:
    out = []
    for i in range(32):
        flag = int(ram[ADDR_OBJ_FLAG + i])
        if not (flag & 0x80):
            continue
        t = int(ram[ADDR_OBJ_PTR + i])
        if t not in LL_TYPES:
            continue
        out.append(
            {
                "i": i,
                "t": t,
                "x": int(ram[ADDR_OBJ_X + i]),
                "y": int(ram[ADDR_OBJ_Y + i]),
                "scr": int(ram[ADDR_OBJ_SCREEN + i]),
                "fl": flag,
                "hp": int(ram[ADDR_ENEMY_HP + i]),
            }
        )
    return out

def body(lls):
    bs = [o for o in lls if o["t"] == LL_BODY]
    return bs[0] if bs else (lls[0] if lls else None)

def meta(ram):
    return {
        "cam": int(ram[ADDR_CAMERA_X_SCREEN]),
        "cx": int(ram[ADDR_CAMERA_X]),
        "prog": camera_progress_x(ram),
        "sx": player_screen_x(ram),
        "sy": player_screen_y(ram),
        "hp": int(ram[ADDR_HEALTH]),
        "ft": int(ram[ADDR_TILE_FEET]),
        "inv": int(ram[ADDR_INVULN_TIMER]),
        "fallen": is_fallen(ram),
    }

def run_one(params: dict[str, Any], out: Path, shots: bool = False) -> dict[str, Any]:
    env = make_env(GAME, "AirFanPlatform", GAME_DIR, render_mode="rgb_array")
    obs, _ = env.reset()

    camp_prog = params["camp_prog"]  # walk until progress >= this
    dx_min = params["dx_min"]
    dx_max = params["dx_max"]
    ll_y_max = params["ll_y_max"]  # only jump if cloud still high
    jh = params["jh"]
    hang = params["hang"]  # frames hold A after leave ground
    shoot_ground = params["shoot_ground"]
    shoot_air = params["shoot_air"]
    wait_max = params["wait_max"]
    face = params.get("face", "RIGHT")  # shoot direction
    post_jump = params.get("post_jump", "RIGHT")  # air horizontal

    phase = "walk"
    phase_t = 0
    jump_f = None
    first_ll = None
    min_dist = 999.0
    best = None
    min_sy4 = 255
    max_prog = 0
    max_cam = 0
    stands: list[dict] = []
    contacts: list[dict] = []
    log: list[dict] = []
    prev_hp = int(env.get_ram()[ADDR_HEALTH])
    y_hist: list[int] = []
    left_ground = False
    ll_hp_min = 99
    ll_seen_ids: set[int] = set()

    for f in range(1, params.get("max_frames", 280) + 1):
        ram = env.get_ram()
        m = meta(ram)
        lls = snap_ll(ram)
        b = body(lls)
        max_prog = max(max_prog, m["prog"])
        max_cam = max(max_cam, m["cam"])
        if m["cam"] >= 4:
            min_sy4 = min(min_sy4, m["sy"])

        dx = dy = None
        if b:
            if first_ll is None:
                first_ll = f
            ll_seen_ids.add(b["i"])
            ll_hp_min = min(ll_hp_min, b["hp"])
            dx = b["x"] - m["sx"]
            dy = b["y"] - m["sy"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best = {"f": f, **m, "ll": b, "dx": dx, "dy": dy, "dist": round(dist, 1)}

        # contact damage
        if m["hp"] < prev_hp:
            contacts.append({"f": f, **m, "ll": b, "dx": dx, "dy": dy, "dmg": prev_hp - m["hp"]})
        prev_hp = m["hp"]

        if m["ft"] == 0:
            left_ground = True
        y_hist.append(m["sy"])
        if len(y_hist) > 12:
            y_hist.pop(0)

        # Real stand: past island, Y nearly flat for 6f, sy in cloud band, not fallen
        if (
            left_ground
            and m["prog"] > 984
            and not m["fallen"]
            and m["hp"] > 0
            and 20 <= m["sy"] <= 70
            and len(y_hist) >= 6
            and max(y_hist[-6:]) - min(y_hist[-6:]) <= 2
        ):
            # require near LL body or feet solid
            near = b is not None and abs(m["sx"] - b["x"]) <= 22 and abs(m["sy"] - b["y"]) <= 20
            if m["ft"] == 1 or near or m["inv"] > 0:
                stands.append(
                    {
                        "f": f,
                        **m,
                        "ll": b,
                        "dx": dx,
                        "near": near,
                        "yvar": max(y_hist[-6:]) - min(y_hist[-6:]),
                    }
                )

        # phase logic
        shoot = False
        buttons: list[str] = []

        if phase == "walk":
            buttons = ["RIGHT"]
            if shoot_ground:
                shoot = f % 4 < 2
            if m["prog"] >= camp_prog and m["ft"] == 1:
                phase = "camp"
                phase_t = 0
        elif phase == "camp":
            phase_t += 1
            # hold position; optional tiny right if still short of camp
            if m["prog"] < camp_prog - 2 and m["ft"] == 1:
                buttons = ["RIGHT"]
            else:
                buttons = []  # stand
            if face == "RIGHT" and not buttons:
                pass
            if shoot_ground:
                shoot = True  # hold B pulses via game fire rate
                if f % 3 == 0:
                    buttons = buttons + []  # B added below
            # jump condition
            ready = (
                b is not None
                and dx is not None
                and dx_min <= dx <= dx_max
                and b["y"] <= ll_y_max
                and m["ft"] == 1
            )
            # also: if LL has sunk past us but still dx ok and we never jumped
            timeout = phase_t >= wait_max
            if ready or (timeout and m["ft"] == 1):
                phase = "jump"
                phase_t = 0
                jump_f = f
            if m["ft"] != 1:
                # slipped — try jump anyway
                phase = "jump"
                phase_t = 0
                jump_f = f
        elif phase == "jump":
            phase_t += 1
            buttons = [post_jump, "A"]
            if shoot_air:
                shoot = f % 3 < 2
            if phase_t >= jh or m["ft"] == 0:
                phase = "air"
                phase_t = 0
        elif phase == "air":
            phase_t += 1
            buttons = [post_jump]
            if phase_t <= hang:
                buttons.append("A")
            elif phase_t % 2 == 0:
                buttons.append("A")  # partial hang
            if shoot_air:
                shoot = True
            # if landed on something solid high, walk right
            if m["ft"] == 1 and m["prog"] > 984:
                phase = "ride"
                phase_t = 0
            if stands and m["sy"] < 70 and len(stands) >= 3:
                phase = "ride"
                phase_t = 0
        elif phase == "ride":
            phase_t += 1
            buttons = ["RIGHT"]
            if shoot_air and f % 5 < 2:
                shoot = True
            # hop if needed
            if m["ft"] == 1 and phase_t % 40 < 12:
                buttons.append("A")
            elif m["ft"] == 0:
                buttons.append("A")

        if shoot:
            buttons.append("B")
        if not buttons:
            action = nes_action("B") if shoot else nes_idle_action()
        else:
            # dedupe
            ub = []
            for x in buttons:
                if x not in ub:
                    ub.append(x)
            action = nes_action(*ub)

        obs, _, term, trunc, _ = env.step(action)

        if f % 4 == 0 or phase in ("jump", "air", "ride") or (dx is not None and abs(dx) < 40):
            log.append({"f": f, "ph": phase, **m, "dx": dx, "dy": dy, "ll": b})

        m2 = meta(env.get_ram())
        if m2["fallen"] or m2["hp"] <= 0 or term or trunc or m2["cam"] >= 5:
            if shots or stands or m2["cam"] >= 5:
                try:
                    save_rgb_png(
                        obs,
                        out
                        / f"{params['label']}_p{m2['prog']}_c{m2['cam']}_sy{m2['sy']}.png",
                    )
                except Exception:
                    pass
            break

    final = meta(env.get_ram())
    final_ll = body(snap_ll(env.get_ram()))
    # save state if standing high post-island
    saved = None
    if stands and final["hp"] > 0 and not final["fallen"] and final["sy"] < 80:
        try:
            # re-load not available; save current env
            path = save_state(env, GAME_DIR, GAME, f"AirCloudProbe_{params['label'][:40]}")
            saved = path.name
        except Exception as e:
            saved = f"err:{e}"
    env.close()

    # filter stands: need sustained
    real_stands = [s for s in stands if s["f"] > (jump_f or 0)]
    sustained = []
    if real_stands:
        # group consecutive
        run = [real_stands[0]]
        for s in real_stands[1:]:
            if s["f"] == run[-1]["f"] + 1:
                run.append(s)
            else:
                if len(run) >= 4:
                    sustained.extend(run)
                run = [s]
        if len(run) >= 4:
            sustained.extend(run)

    return {
        "label": params["label"],
        "params": {k: v for k, v in params.items() if k != "label"},
        "success_cam5": final["cam"] >= 5 and final["hp"] > 0 and not final["fallen"],
        "max_prog": max_prog,
        "max_cam": max_cam,
        "min_dist": round(min_dist, 1) if min_dist < 999 else None,
        "min_sy4": min_sy4 if min_sy4 < 255 else None,
        "best": best,
        "jump_f": jump_f,
        "first_ll": first_ll,
        "n_stand_raw": len(real_stands),
        "n_stand_sustained": len(sustained),
        "stand_sample": sustained[:5] or real_stands[:3],
        "contacts": contacts[:6],
        "ll_hp_min": ll_hp_min if ll_hp_min < 99 else None,
        "final": {**final, "ll": final_ll},
        "saved": saved,
        "log_tail": log[-8:],
        "log_near": [x for x in log if x.get("dx") is not None and abs(x["dx"]) <= 30][:10],
    }

def main():
    configure_headless()
    out = RECORDINGS_DIR / "air_post4_cloud_v2"
    out.mkdir(parents=True, exist_ok=True)

    recipes = []
    for camp in (960, 968, 972, 976, 980):
        for dx_min, dx_max in [(8, 28), (12, 36), (18, 44), (24, 52), (30, 60), (36, 72)]:
            for ll_y_max in (40, 48, 56, 70):
                for jh in (8, 12, 16):
                    for hang in (12, 20, 28):
                        for wait_max in (60, 100, 160):
                            for sg, sa in [(True, True), (False, True)]:
                                # prune combinatorial explosion
                                if camp not in (968, 976) and wait_max != 100:
                                    continue
                                if hang != 20 and jh != 12:
                                    continue
                                if ll_y_max == 70 and dx_min not in (18, 30):
                                    continue
                                label = (
                                    f"c{camp}_dx{dx_min}-{dx_max}_ly{ll_y_max}"
                                    f"_jh{jh}_h{hang}_w{wait_max}_sg{int(sg)}"
                                )
                                recipes.append(
                                    {
                                        "label": label,
                                        "camp_prog": camp,
                                        "dx_min": dx_min,
                                        "dx_max": dx_max,
                                        "ll_y_max": ll_y_max,
                                        "jh": jh,
                                        "hang": hang,
                                        "shoot_ground": sg,
                                        "shoot_air": sa,
                                        "wait_max": wait_max,
                                        "max_frames": 300,
                                    }
                                )

    # Extra: jump at fixed frame after LL spawn (spawn ~f16 when walking)
    for delay in (20, 30, 40, 50, 60, 70, 80, 90):
        for camp in (970, 978):
            recipes.append(
                {
                    "label": f"fixed_c{camp}_d{delay}",
                    "camp_prog": camp,
                    "dx_min": 0,
                    "dx_max": 200,  # any
                    "ll_y_max": 255,
                    "jh": 12,
                    "hang": 24,
                    "shoot_ground": True,
                    "shoot_air": True,
                    "wait_max": delay,  # jump after delay frames of camp
                    "max_frames": 260,
                }
            )

    print(f"recipes {len(recipes)}", flush=True)
    results = []
    best_d = None
    best_stand = None
    any5 = False

    for i, p in enumerate(recipes):
        r = run_one(p, out, shots=False)
        results.append(r)
        if best_d is None or (r["min_dist"] or 999) < (best_d["min_dist"] or 999):
            best_d = r
        if r["n_stand_sustained"] > 0 and (
            best_stand is None
            or r["n_stand_sustained"] > best_stand["n_stand_sustained"]
            or (
                r["n_stand_sustained"] == best_stand["n_stand_sustained"]
                and r["max_prog"] > best_stand["max_prog"]
            )
        ):
            best_stand = r
        if r["success_cam5"]:
            any5 = True
            print("CAM5!", r["label"], r["final"], flush=True)
        if (i + 1) % 25 == 0 or r["n_stand_sustained"] > 0 or (r["min_dist"] or 99) < 12:
            print(
                f"[{i+1}/{len(recipes)}] {r['label']} d={r['min_dist']} "
                f"p={r['max_prog']} c={r['max_cam']} st={r['n_stand_sustained']} "
                f"sy4={r['min_sy4']} hp_ll={r['ll_hp_min']} contacts={len(r['contacts'])}",
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda r: (
            0 if r["success_cam5"] else 1,
            0 if r["n_stand_sustained"] > 0 else 1,
            -(r["n_stand_sustained"]),
            r["min_dist"] if r["min_dist"] is not None else 999,
            -r["max_prog"],
        ),
    )
    top = []
    for r in ranked[:30]:
        top.append(
            {
                k: r[k]
                for k in [
                    "label",
                    "success_cam5",
                    "max_prog",
                    "max_cam",
                    "min_dist",
                    "min_sy4",
                    "n_stand_sustained",
                    "stand_sample",
                    "best",
                    "contacts",
                    "ll_hp_min",
                    "final",
                    "jump_f",
                    "params",
                ]
            }
        )

    summary = {
        "n": len(results),
        "any_cam5": any5,
        "any_sustained_stand": any(r["n_stand_sustained"] > 0 for r in results),
        "best_dist": {
            k: best_d[k]
            for k in [
                "label",
                "min_dist",
                "max_prog",
                "best",
                "final",
                "n_stand_sustained",
                "contacts",
                "params",
            ]
        }
        if best_d
        else None,
        "best_stand": {
            k: best_stand[k]
            for k in [
                "label",
                "min_dist",
                "max_prog",
                "stand_sample",
                "final",
                "params",
                "n_stand_sustained",
            ]
        }
        if best_stand
        else None,
        "top": top,
    }
    write_json_report(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k != "top"}, indent=2))
    print("TOP:", flush=True)
    for t in top[:8]:
        print(
            t["label"],
            "d",
            t["min_dist"],
            "st",
            t["n_stand_sustained"],
            "p",
            t["max_prog"],
            "best",
            t["best"],
            flush=True,
        )

if __name__ == "__main__":
    main()
