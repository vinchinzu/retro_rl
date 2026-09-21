"""Scratch probe: is 0x68 EAST on row 6 open to shop_p7 0x6F?

One hypothesis per run. Screenshot every screen/mode change. Halt on
first stall / wrong screen / cave / death. Not a production CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.controls import pressed_nes_buttons
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.overworld.graph import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    ScreenHop,
    is_5c_maze_hop,
)
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController, sword_segment_success
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist
from zelda_i.screen_glance import leftover_from_snapshot

OUT_DIR = RECORDINGS_DIR / "scratch_pre_l1_6f"
STALL_FRAMES = 24
HOP_CAP = 20000
BANNED = frozenset({0x79, 0x67})
DPAD = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})

# Live prefix + row-6 east hypothesis (PRE_L1.md 4.5.1). Not live past 0x68.
TRIAL1_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x69, "RIGHT", align_y=141),
    ScreenHop(0x6A, "RIGHT", align_y=141),
    ScreenHop(0x6B, "RIGHT", align_y=141),
    ScreenHop(0x6C, "RIGHT", align_y=141),
    ScreenHop(0x6D, "RIGHT", align_y=141),
    ScreenHop(0x6E, "RIGHT", align_y=141),
    ScreenHop(0x6F, "RIGHT", align_y=141),
)

# Live L8/candle corridor through 0x5C maze to 0x5E, then 0x5F / south 0x6F.
# Last south of LEVEL8_BUSH_HOPS (0x5D→0x6D pocket) is omitted on purpose.
TRIAL2_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x5A, "RIGHT", y_band_lo=120, y_band_hi=145),
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x5E, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5F, "RIGHT", align_y=141),
    ScreenHop(0x6F, "DOWN", align_x=120),
)


def _glance(snap) -> dict:
    leftover = leftover_from_snapshot(snap)
    leftover["sword"] = int(snap.sword)
    leftover["rupees"] = int(snap.rupees)
    leftover["bombs"] = int(snap.bombs)
    leftover["health"] = int(snap.health)
    leftover["hearts"] = f"{snap.filled_hearts}/{snap.heart_containers}"
    leftover["hearts_lo"] = int(snap.health) & 0x0F
    leftover["hearts_hi"] = (int(snap.health) >> 4) & 0x0F
    leftover["facing"] = int(snap.facing)
    leftover["level"] = int(snap.level)
    leftover["tile"] = int(snap.colliding_tile)
    return leftover


def _held_dir(action) -> str | None:
    dirs = [b for b in pressed_nes_buttons(list(action)) if b in DPAD]
    return dirs[0] if dirs else None


def _png_name(idx: int, snap, tag: str) -> Path:
    return OUT_DIR / (
        f"{tag}_{idx:03d}_m{snap.mode:02d}_s{snap.screen:02X}_"
        f"x{snap.link_x}_y{snap.link_y}.png"
    )


def _event(frame: int, snap, reason: str, hop=None) -> dict:
    rec = {
        "frame": frame,
        "screen": int(snap.screen),
        "screen_hex": f"0x{int(snap.screen):02X}",
        "mode": int(snap.mode),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "reason": reason,
        "health": int(snap.health),
        "rupees": int(snap.rupees),
        "sword": int(snap.sword),
        "bombs": int(snap.bombs),
        "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
    }
    if hop is not None:
        rec["hop"] = hop
    return rec


def _run_loop(
    *,
    env,
    obs,
    controller,
    assist,
    max_frames: int,
    tag: str,
    pngs: list[str],
    trail: list[dict],
    stop_on_6f: bool,
    frame_base: int,
) -> tuple[object, int, str]:
    snap = read_snapshot(env.get_ram())
    last_screen = int(snap.screen)
    last_mode = int(snap.mode)
    stall = 0
    stall_xy: tuple[int, int] | None = None
    stall_btn: str | None = None
    png_i = len(pngs)
    path = save_rgb_png(obs, _png_name(png_i, snap, tag))
    pngs.append(str(path))
    trail.append(_event(frame_base, snap, f"{tag}_start"))
    stop = ""
    frames = 0
    while frames < max_frames:
        snap = read_snapshot(env.get_ram())
        screen, mode = int(snap.screen), int(snap.mode)
        changed = screen != last_screen or mode != last_mode
        if changed:
            png_i = len(pngs)
            path = save_rgb_png(obs, _png_name(png_i, snap, tag))
            pngs.append(str(path))
            hop = None
            if hasattr(controller, "hop_index"):
                hop = controller.hop_index
            why = "screen" if screen != last_screen else "mode"
            trail.append(_event(frame_base + frames, snap, why, hop=hop))
            last_screen, last_mode = screen, mode
        if mode == 17:
            stop = "death"
            break
        if screen in BANNED:
            stop = f"banned_0x{screen:02X}"
            break
        if mode == 11 and screen == 0x68:
            stop = "door_repair_cave_0x68"
            break
        if stop_on_6f and mode == PLAY_MODE and screen == 0x6F:
            stop = "reached_0x6F"
            break
        phase = getattr(controller, "phase", None)
        phase_name = getattr(phase, "name", str(phase) if phase is not None else "")
        if getattr(controller, "success", False) or phase_name == "DONE":
            stop = "controller_done"
            break
        if phase_name == "FAILED" or getattr(controller, "failed", False):
            stop = "controller_failed"
            break
        pose = (int(snap.link_x), int(snap.link_y))
        act = controller.step(snap)
        held = _held_dir(act.action)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frame_base + frames)
        after = read_snapshot(env.get_ram())
        if (
            held
            and not after.transitioning
            and after.mode == PLAY_MODE
            and (int(after.link_x), int(after.link_y)) == pose
        ):
            if stall_xy == pose and stall_btn == held:
                stall += 1
            else:
                stall = 1
                stall_xy = pose
                stall_btn = held
        else:
            stall = 0
            stall_xy = None
            stall_btn = None
        if stall >= STALL_FRAMES:
            stop = f"stall_{held}_{pose[0]},{pose[1]}"
            snap = after
            break
    else:
        stop = "timeout"
        snap = read_snapshot(env.get_ram())
    path = save_rgb_png(obs, OUT_DIR / f"{tag}_final.png")
    pngs.append(str(path))
    trail.append(_event(frame_base + frames, snap, f"stop_{stop}"))
    return obs, frames, stop


def run_trial(trial: int, tag: str | None = None) -> dict:
    hops = TRIAL1_HOPS if trial == 1 else TRIAL2_HOPS
    tag = tag or f"t{trial}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    pngs: list[str] = []
    trail: list[dict] = []
    try:
        obs, _ = reset_obs(env)
        obs, boot_frames = boot_to_ready(env, first_playthrough=True)
        snap0 = read_snapshot(env.get_ram())
        entry = _glance(snap0)
        entry["boot_frames"] = boot_frames

        sword = SwordCaveController()
        obs, sword_frames, sword_stop = _run_loop(
            env=env,
            obs=obs,
            controller=sword,
            assist=assist,
            max_frames=SWORD_MAX,
            tag=f"{tag}_sword",
            pngs=pngs,
            trail=trail,
            stop_on_6f=False,
            frame_base=boot_frames,
        )
        ram = env.get_ram()
        sword_ok = bool(
            sword_segment_success(ram)
            or (sword.success and read_snapshot(ram).sword >= 1)
        )
        sword_snap = read_snapshot(ram)
        sword_glance = _glance(sword_snap)
        if not sword_ok:
            report = {
                "ok": False,
                "trial": trial,
                "stop": f"sword_{sword_stop}",
                "entry": entry,
                "sword": sword.report(),
                "sword_glance": sword_glance,
                "leftover": sword_glance,
                "pngs": pngs,
                "trail": trail,
                "assist": assist.report() if assist is not None else None,
            }
            write_json_report(OUT_DIR / f"{tag}_report.json", report)
            return report

        nav = OverworldPathController(
            hops=hops,
            require_sword=True,
            farm_below_hearts=0,
            evade=True,
            occupied_lane=True,
            need_rupees=20,
            max_farm_attempts=0,  # scoop only; restock farm stalled 0x78 (48,149)
            max_frames=HOP_CAP,
        )
        if trial == 2:
            nav.maze_waypoints = LEVEL2_5C_MAZE_WAYPOINTS
            nav.maze_hop_pred = is_5c_maze_hop
        obs, hop_frames, hop_stop = _run_loop(
            env=env,
            obs=obs,
            controller=nav,
            assist=assist,
            max_frames=HOP_CAP,
            tag=f"{tag}_hop",
            pngs=pngs,
            trail=trail,
            stop_on_6f=True,
            frame_base=boot_frames + sword_frames,
        )
        leftover = _glance(read_snapshot(env.get_ram()))
        assist_rep = assist.report() if assist is not None else None
        ok = hop_stop == "reached_0x6F" or (
            leftover.get("screen") == 0x6F and leftover.get("mode") == PLAY_MODE
        )
        report = {
            "ok": ok,
            "trial": trial,
            "stop": hop_stop,
            "entry": entry,
            "sword": sword.report(),
            "sword_glance": sword_glance,
            "nav": nav.report(),
            "leftover": leftover,
            "frames_after_sword": hop_frames,
            "boot_frames": boot_frames,
            "pngs": pngs,
            "trail": trail,
            "hops": [
                {
                    "target": f"0x{h.target:02X}",
                    "dir": h.direction,
                    "align_x": h.align_x,
                    "align_y": h.align_y,
                    "y_band": list(h.y_band) if h.y_band else None,
                }
                for h in hops
            ],
            "assist": assist_rep,
        }
        write_json_report(OUT_DIR / f"{tag}_report.json", report)
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    rep = run_trial(args.trial, tag=args.tag or None)
    left = rep["leftover"]
    assist = rep.get("assist") or {}
    print(
        f"trial={rep['trial']} ok={rep['ok']} stop={rep['stop']} "
        f"screen=0x{int(left.get('screen', 0)):02X} mode={left.get('mode')} "
        f"xy=({left.get('x')},{left.get('y')}) sword={left.get('sword')} "
        f"rupees={left.get('rupees')} bombs={left.get('bombs')} "
        f"hp=0x{int(left.get('health', 0)):02X} hearts={left.get('hearts')} "
        f"prog={assist.get('progression_writes')} cap={assist.get('capacity_writes')}"
    )
    nav = rep.get("nav") or {}
    print(
        f"nav_frames={nav.get('frames')} hop_index={nav.get('hop_index')} "
        f"hits={nav.get('hits_taken')} notes={nav.get('notes')}"
    )
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
