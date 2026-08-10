"""rr-54ui: Clean land on first Thunder Chariot (LL cloud).

From AirFanPlatform:
1. Spawn LL (mapset4 0x3D/0x3E ~prog 961)
2. Close ~28px X gap vs pure edge-jump apex
3. Kill rider (buster ~3 hits) then stand on object-solid cloud
4. Progress camera ≥5 if possible

Stand detection uses Y-stable + object overlap (not only tile_feet).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_NES = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO), str(_NES)]

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
ADDR_ENEMY_HP_BASE = 0x06C2

LL_BODY = 0x3E  # kaminari_goro (cloud body)
LL_MOVE = 0x3D  # rider / move
LL_TYPES = {0x3D, 0x3E, 0x3F}


def snap_ll(ram) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(32):
        flag = int(ram[ADDR_OBJ_FLAG + i])
        if not (flag & 0x80):
            continue
        t = int(ram[ADDR_OBJ_PTR + i])
        if t not in LL_TYPES:
            continue
        # enemy HP: try slot-aligned (object index maps loosely; store nearby)
        hp = int(ram[ADDR_ENEMY_HP_BASE + max(0, i - 1)]) if i > 0 else -1
        out.append(
            {
                "i": i,
                "t": t,
                "x": int(ram[ADDR_OBJ_X + i]),
                "y": int(ram[ADDR_OBJ_Y + i]),
                "scr": int(ram[ADDR_OBJ_SCREEN + i]),
                "fl": flag,
                "hp_guess": hp,
            }
        )
    return out


def meta(ram) -> dict[str, Any]:
    return {
        "cam_scr": int(ram[ADDR_CAMERA_X_SCREEN]),
        "cam_x": int(ram[ADDR_CAMERA_X]),
        "prog": camera_progress_x(ram),
        "sx": player_screen_x(ram),
        "sy": player_screen_y(ram),
        "hp": int(ram[ADDR_HEALTH]),
        "feet": int(ram[ADDR_TILE_FEET]),
        "inv": int(ram[ADDR_INVULN_TIMER]),
        "fallen": is_fallen(ram),
    }


def body_ll(lls: list[dict[str, Any]]) -> dict[str, Any] | None:
    bodies = [o for o in lls if o["t"] == LL_BODY]
    if bodies:
        return min(bodies, key=lambda o: abs(o["x"] - 128))  # nearest mid
    moves = [o for o in lls if o["t"] == LL_MOVE]
    return moves[0] if moves else None


def dist_xy(sx: int, sy: int, o: dict[str, Any]) -> float:
    return math.hypot(sx - o["x"], sy - o["y"])


def action_for(phase: str, f: int, *, jh: int, shoot_mode: str, near_ll: bool) -> Any:
    """Map phase name to buttons."""
    shoot = False
    if shoot_mode == "spam":
        shoot = (f % 4) < 2
    elif shoot_mode == "pulse":
        shoot = (f % 8) < 2
    elif shoot_mode == "near":
        shoot = near_ll and (f % 3) < 2
    elif shoot_mode == "always":
        shoot = True

    if phase == "walk":
        btns = ["RIGHT"]
        if shoot:
            btns.append("B")
        return nes_action(*btns)
    if phase == "edge_idle":
        # micro-walk to keep cam/LL advancing without falling
        if f % 30 < 2:
            btns = ["RIGHT"]
        elif f % 30 < 4:
            btns = ["LEFT"]
        else:
            btns = []
        if shoot:
            btns.append("B")
        return nes_action(*btns) if btns else nes_idle_action()
    if phase == "edge_shoot":
        # stand near edge, spam buster toward approaching LL
        if f % 20 < 1:
            btns = ["RIGHT"]
        else:
            btns = []
        btns.append("B")
        return nes_action(*btns)
    if phase == "jump":
        # rising-edge friendly: first jh frames hold A
        btns = ["RIGHT", "A"]
        if shoot:
            btns.append("B")
        return nes_action(*btns)
    if phase == "air":
        btns = ["RIGHT"]
        # hold A for hang (variable)
        if f % 2 == 0:
            btns.append("A")
        if shoot:
            btns.append("B")
        return nes_action(*btns)
    if phase == "air_hold_a":
        btns = ["RIGHT", "A"]
        if shoot:
            btns.append("B")
        return nes_action(*btns)
    if phase == "nudge_left":
        btns = ["LEFT"]
        if shoot:
            btns.append("B")
        return nes_action(*btns)
    return nes_idle_action()


def run_recipe(
    *,
    label: str,
    walk_frames: int,
    wait_mode: str,
    wait_max: int,
    approach_dx: int,
    jh: int,
    shoot_mode: str,
    max_frames: int,
    out: Path,
    save_shots: bool = False,
) -> dict[str, Any]:
    """One Clean recipe trial from AirFanPlatform."""
    env = make_env(GAME, "AirFanPlatform", GAME_DIR, render_mode="rgb_array")
    obs, _ = env.reset()
    ram = env.get_ram()

    timeline: list[dict[str, Any]] = []
    min_dist = 999.0
    best_near: dict[str, Any] | None = None
    min_sy_scr4 = 255
    max_prog = 0
    max_cam = 0
    ll_alive_frames = 0
    ll_gone_frames = 0
    stand_hits: list[dict[str, Any]] = []
    killed = False
    first_ll_f: int | None = None
    approach_f: int | None = None
    jump_started = False
    jump_start_f: int | None = None
    air_frames = 0
    prev_ll_present = False
    y_stable_count = 0
    prev_sy: int | None = None
    prev_feet = 0

    phase = "walk"
    phase_f0 = 0

    for f in range(1, max_frames + 1):
        ram = env.get_ram()
        m = meta(ram)
        lls = snap_ll(ram)
        body = body_ll(lls)
        max_prog = max(max_prog, m["prog"])
        max_cam = max(max_cam, m["cam_scr"])
        if m["cam_scr"] >= 4:
            min_sy_scr4 = min(min_sy_scr4, m["sy"])

        ll_present = body is not None
        if ll_present and first_ll_f is None:
            first_ll_f = f
        if ll_present:
            ll_alive_frames += 1
            d = dist_xy(m["sx"], m["sy"], body)
            if d < min_dist:
                min_dist = d
                best_near = {
                    "f": f,
                    **m,
                    "ll": body,
                    "dist": round(d, 1),
                    "dx": body["x"] - m["sx"],
                    "dy": body["y"] - m["sy"],
                }
        elif prev_ll_present and first_ll_f is not None:
            ll_gone_frames += 1
            if not killed and ll_gone_frames >= 3:
                killed = True  # object despawn after presence
        prev_ll_present = ll_present

        # stand heuristic: after leaving platform (prog>984 or feet was 0),
        # sy near cloud band and Y stable while not fallen
        near_cloud_y = body is not None and abs(m["sy"] - (body["y"] + 0)) <= 18
        overlap = (
            body is not None
            and abs(m["sx"] - body["x"]) <= 20
            and m["sy"] <= body["y"] + 12
            and m["sy"] >= body["y"] - 8
        )
        if prev_sy is not None and abs(m["sy"] - prev_sy) <= 1 and m["sy"] < 80:
            y_stable_count += 1
        else:
            y_stable_count = 0
        if (
            m["prog"] > 984
            and not m["fallen"]
            and m["hp"] > 0
            and m["sy"] < 90
            and (m["feet"] == 1 or (y_stable_count >= 4 and (overlap or near_cloud_y)))
        ):
            stand_hits.append(
                {
                    "f": f,
                    **m,
                    "ll": body,
                    "y_stable": y_stable_count,
                    "overlap": overlap,
                }
            )
        prev_sy = m["sy"]
        prev_feet = m["feet"]

        # --- phase machine ---
        rel = f - phase_f0
        near_ll = body is not None and abs(body["x"] - m["sx"]) <= 48
        dx_to_ll = (body["x"] - m["sx"]) if body else 999

        if phase == "walk":
            if m["feet"] != 1 or m["prog"] >= 978 or rel >= walk_frames:
                phase = wait_mode  # edge_idle | edge_shoot | jump_now
                phase_f0 = f
                rel = 0
        elif phase in ("edge_idle", "edge_shoot"):
            # wait until LL close enough in X, or timeout
            close = body is not None and 0 < dx_to_ll <= approach_dx
            # also accept LL slightly left (overshot) within 12
            close = close or (body is not None and -12 <= dx_to_ll <= approach_dx)
            if close and m["feet"] == 1:
                approach_f = f
                phase = "jump"
                phase_f0 = f
                jump_started = True
                jump_start_f = f
                rel = 0
            elif rel >= wait_max or m["fallen"] or m["feet"] != 1:
                # force jump before falling off or timeout
                phase = "jump"
                phase_f0 = f
                jump_started = True
                jump_start_f = f
                rel = 0
        elif phase == "jump_now":
            phase = "jump"
            phase_f0 = f
            jump_started = True
            jump_start_f = f
            rel = 0
        elif phase == "jump":
            if rel >= jh or m["feet"] == 0:
                phase = "air_hold_a"
                phase_f0 = f
                rel = 0
                air_frames = 0
        elif phase in ("air_hold_a", "air"):
            air_frames += 1
            # after hang, keep air with intermittent A
            if phase == "air_hold_a" and rel >= max(8, jh):
                phase = "air"
                phase_f0 = f
                rel = 0
            # if we landed somehow, try to walk right on cloud
            if m["feet"] == 1 and m["prog"] > 990:
                phase = "walk"
                phase_f0 = f
            elif stand_hits and m["sy"] < 70 and y_stable_count >= 6:
                phase = "walk"  # try advance on cloud
                phase_f0 = f

        act_phase = phase if phase != "jump_now" else "jump"
        if phase == "edge_idle":
            act_phase = "edge_idle"
        elif phase == "edge_shoot":
            act_phase = "edge_shoot"
        elif phase == "jump":
            act_phase = "jump"
        elif phase == "air_hold_a":
            act_phase = "air_hold_a"
        elif phase == "air":
            act_phase = "air"
        elif phase == "walk":
            act_phase = "walk"

        action = action_for(
            act_phase, f, jh=jh, shoot_mode=shoot_mode, near_ll=near_ll or dx_to_ll < 60
        )
        obs, _, term, trunc, _ = env.step(action)
        ram = env.get_ram()
        m2 = meta(ram)

        if f % 5 == 0 or phase != "walk" or (body and abs(body["x"] - m["sx"]) < 40):
            timeline.append(
                {
                    "f": f,
                    "phase": phase,
                    **m,
                    "ll": body,
                    "dx": (body["x"] - m["sx"]) if body else None,
                }
            )

        if m2["fallen"] or m2["hp"] <= 0 or term or trunc or m2["cam_scr"] >= 5:
            timeline.append({"f": f + 1, "phase": "end", **m2, "ll": snap_ll(ram) and body_ll(snap_ll(ram))})
            if save_shots or m2["cam_scr"] >= 5 or stand_hits:
                try:
                    save_rgb_png(obs, out / f"{label}_end_p{m2['prog']}_c{m2['cam_scr']}.png")
                except Exception:
                    pass
            break

    final = meta(env.get_ram())
    final_ll = body_ll(snap_ll(env.get_ram()))
    env.close()

    success = final["cam_scr"] >= 5 and final["hp"] > 0 and not final["fallen"]
    cloud_land = len(stand_hits) > 0 and not (
        stand_hits[0]["prog"] <= 984 and stand_hits[0]["feet"] == 1
    )
    # filter stand hits to post-island only
    post_stands = [s for s in stand_hits if s["prog"] > 984 and s["sy"] < 80]
    report = {
        "label": label,
        "success_cam5": success,
        "cloud_stand": len(post_stands) > 0,
        "killed_guess": killed,
        "max_prog": max_prog,
        "max_cam": max_cam,
        "min_sy_scr4": min_sy_scr4 if min_sy_scr4 < 255 else None,
        "min_dist": round(min_dist, 1) if min_dist < 999 else None,
        "best_near": best_near,
        "first_ll_f": first_ll_f,
        "approach_f": approach_f,
        "jump_start_f": jump_start_f,
        "ll_alive_frames": ll_alive_frames,
        "stand_hits": post_stands[:8],
        "n_stand": len(post_stands),
        "final": {**final, "ll": final_ll},
        "params": {
            "walk_frames": walk_frames,
            "wait_mode": wait_mode,
            "wait_max": wait_max,
            "approach_dx": approach_dx,
            "jh": jh,
            "shoot_mode": shoot_mode,
        },
        "timeline_tail": timeline[-12:],
        "timeline_approach": [t for t in timeline if t.get("dx") is not None and abs(t["dx"]) < 50][:8],
    }
    return report


def main() -> None:
    configure_headless()
    out = RECORDINGS_DIR / "air_post4_cloud"
    out.mkdir(parents=True, exist_ok=True)

    recipes: list[dict[str, Any]] = []
    # Grid focused on: wait-for-approach then jump+shoot
    for walk in (28, 34, 38, 42):
        for wait_mode in ("edge_idle", "edge_shoot"):
            for approach_dx in (24, 32, 40, 48, 56):
                for wait_max in (40, 80, 120):
                    for jh in (10, 12, 14, 18):
                        for shoot in ("spam", "always", "near"):
                            # prune: don't explode fully
                            if wait_mode == "edge_idle" and shoot == "always":
                                continue
                            if approach_dx not in (32, 40, 48) and wait_max != 80:
                                continue
                            if jh not in (12, 14) and shoot != "spam":
                                continue
                            recipes.append(
                                {
                                    "walk_frames": walk,
                                    "wait_mode": wait_mode,
                                    "wait_max": wait_max,
                                    "approach_dx": approach_dx,
                                    "jh": jh,
                                    "shoot_mode": shoot,
                                }
                            )

    # Immediate jump baselines (known RED envelope) for comparison
    for walk in (36, 40):
        for jh in (12, 16):
            recipes.append(
                {
                    "walk_frames": walk,
                    "wait_mode": "jump_now",
                    "wait_max": 0,
                    "approach_dx": 99,
                    "jh": jh,
                    "shoot_mode": "spam",
                }
            )

    # Dedup
    seen = set()
    uniq = []
    for r in recipes:
        key = tuple(sorted(r.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    recipes = uniq
    print(f"running {len(recipes)} recipes", flush=True)

    results: list[dict[str, Any]] = []
    best_by_dist: dict[str, Any] | None = None
    best_stand: dict[str, Any] | None = None
    any_cam5 = False

    for i, p in enumerate(recipes):
        label = (
            f"w{p['walk_frames']}_{p['wait_mode']}_dx{p['approach_dx']}"
            f"_wm{p['wait_max']}_jh{p['jh']}_{p['shoot_mode']}"
        )
        rep = run_recipe(label=label, max_frames=220, out=out, save_shots=False, **p)
        results.append(rep)
        if best_by_dist is None or (rep["min_dist"] or 999) < (best_by_dist["min_dist"] or 999):
            best_by_dist = rep
        if rep["cloud_stand"] and (
            best_stand is None or rep["max_prog"] > best_stand["max_prog"]
        ):
            best_stand = rep
        if rep["success_cam5"]:
            any_cam5 = True
            print("CAM5", label, rep["final"], flush=True)
        if (i + 1) % 20 == 0 or rep["cloud_stand"] or (rep["min_dist"] or 99) < 20:
            print(
                f"[{i+1}/{len(recipes)}] {label} "
                f"dist={rep['min_dist']} prog={rep['max_prog']} "
                f"cam={rep['max_cam']} stand={rep['n_stand']} "
                f"killed={rep['killed_guess']} min_sy={rep['min_sy_scr4']}",
                flush=True,
            )

    # Rank top by min_dist, then by max_prog, then stands
    ranked = sorted(
        results,
        key=lambda r: (
            0 if r["success_cam5"] else 1,
            0 if r["cloud_stand"] else 1,
            r["min_dist"] if r["min_dist"] is not None else 999,
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
                    "cloud_stand",
                    "killed_guess",
                    "max_prog",
                    "max_cam",
                    "min_dist",
                    "min_sy_scr4",
                    "n_stand",
                    "best_near",
                    "stand_hits",
                    "params",
                    "final",
                    "approach_f",
                    "jump_start_f",
                ]
            }
        )

    summary = {
        "n": len(results),
        "any_cam5": any_cam5,
        "any_stand": any(r["cloud_stand"] for r in results),
        "any_killed": any(r["killed_guess"] for r in results),
        "best_dist": {
            k: best_by_dist[k]
            for k in [
                "label",
                "min_dist",
                "max_prog",
                "max_cam",
                "best_near",
                "params",
                "final",
                "n_stand",
            ]
        }
        if best_by_dist
        else None,
        "best_stand": {
            k: best_stand[k]
            for k in [
                "label",
                "min_dist",
                "max_prog",
                "stand_hits",
                "params",
                "final",
            ]
        }
        if best_stand
        else None,
        "top": top,
    }
    write_json_report(out / "cloud_land_grid.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k != "top"}, indent=2), flush=True)
    print("TOP5:", flush=True)
    for t in top[:5]:
        print(
            t["label"],
            "dist",
            t["min_dist"],
            "prog",
            t["max_prog"],
            "stand",
            t["n_stand"],
            "bn",
            t.get("best_near"),
            flush=True,
        )
    print("DONE", out)


if __name__ == "__main__":
    main()
