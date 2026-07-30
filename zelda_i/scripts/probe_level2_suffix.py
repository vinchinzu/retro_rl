"""Probe post-0x4A overworld path toward Level 2 door (0x3C) and entry.

Uses Level1ExitOverworld → existing hop controller to 0x4A, then continues
with candidate hops / farm / door hunt. Writes a JSON report + screenshots.

Examples::

    uv run python zelda_i/scripts/probe_level2_suffix.py
    uv run python zelda_i/scripts/probe_level2_suffix.py --from-state At4A --max-frames 12000
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.chain import run_controller_stage
from zelda_i.level2_overworld import (
    SEGMENT_MAX_FRAMES,
    OverworldToLevel2Controller,
    level2_path_prefix_success,
    post_triforce_overworld_ready,
    PostTriforceSettleController,
    SETTLE_MAX_FRAMES,
)
from zelda_i.nav_common import align_and_push, swing_action, track_stuck, unstick_wiggle
from zelda_i.overworld import LEVEL2_PATH_HOPS, ScreenHop
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

# Walkthrough-planned suffix after verified 0x4A stop.
SUFFIX_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x4B, "RIGHT", align_y=141),
    ScreenHop(0x5B, "DOWN", align_x=48),
    ScreenHop(0x5C, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x5D, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x4D, "UP", align_x=112),
    ScreenHop(0x4C, "LEFT", align_y=141),
    ScreenHop(0x3C, "UP", align_x=112),
)

FULL_HOPS: tuple[ScreenHop, ...] = LEVEL2_PATH_HOPS + SUFFIX_HOPS


def _snapshot_dict(snap) -> dict:
    objs = [
        {
            "slot": o.slot,
            "type": o.type_id,
            "x": o.x,
            "y": o.y,
            "hp": o.hp,
        }
        for o in snap.objects
        if o.slot >= 1 and o.type_id not in (0, 0xFF) and o.y > 0
    ][:12]
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
        "sword": snap.sword,
        "bombs": snap.bombs,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "objects": objs,
    }


def run_probe(
    *,
    start_state: str,
    max_frames: int,
    stop_screen: int,
    enter_dungeon: bool,
    farm_hearts_min: int,
    save_checkpoint: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = _snapshot_dict(read_snapshot(env.get_ram()))

        if start_state == "Level1ExitOverworld" and not post_triforce_overworld_ready(
            env.get_ram()
        ):
            settle = PostTriforceSettleController()
            obs, _ = run_controller_stage(
                env, obs, name="settle", controller=settle, max_frames=SETTLE_MAX_FRAMES
            )

        # If not already on/past 0x4A, run prefix first.
        snap = read_snapshot(env.get_ram())
        prefix_report = None
        if snap.level == 0 and snap.screen not in {
            0x4A,
            0x4B,
            0x5B,
            0x5C,
            0x5D,
            0x4D,
            0x4C,
            0x3C,
        }:
            nav = OverworldToLevel2Controller()
            obs, stage = run_controller_stage(
                env, obs, name="prefix", controller=nav, max_frames=SEGMENT_MAX_FRAMES
            )
            prefix_report = stage.report()
            snap = read_snapshot(env.get_ram())
            if not (level2_path_prefix_success(env.get_ram()) or nav.success):
                png = RECORDINGS_DIR / f"{tag}_prefix_fail.png"
                save_rgb_png(obs, png)
                return {
                    "ok": False,
                    "stage": "prefix",
                    "entry": entry,
                    "prefix": prefix_report,
                    "final": _snapshot_dict(snap),
                    "screenshot": str(png),
                }

        # Heart farm on 0x4A if under threshold filled hearts.
        farm_log: list[dict] = []
        if (
            snap.screen == 0x4A
            and snap.filled_hearts < farm_hearts_min
            and farm_hearts_min > 0
        ):
            stuck = 0
            last_x = last_y = last_sc = -1
            farm_frames = 0
            waypoints = ((64, 141), (120, 141), (176, 141), (120, 181), (120, 101))
            wi = 0
            while farm_frames < 2400 and snap.filled_hearts < farm_hearts_min:
                farm_frames += 1
                stuck, last_x, last_y, last_sc = track_stuck(
                    snap,
                    last_x=last_x,
                    last_y=last_y,
                    last_screen=last_sc,
                    stuck=stuck,
                )
                if snap.mode == 17:
                    break
                if stuck > 40:
                    act, stuck = unstick_wiggle(stuck)
                    obs, *_ = env.step(act.action)
                else:
                    tx, ty = waypoints[wi % len(waypoints)]
                    if abs(snap.link_x - tx) <= 6 and abs(snap.link_y - ty) <= 6:
                        wi += 1
                        tx, ty = waypoints[wi % len(waypoints)]
                    # Prefer horizontal then vertical; swing A.
                    if abs(snap.link_x - tx) > 6:
                        d = "RIGHT" if snap.link_x < tx else "LEFT"
                    else:
                        d = "DOWN" if snap.link_y < ty else "UP"
                    act = swing_action(farm_frames, d, "farm", period=8, hold=3)
                    obs, *_ = env.step(act.action)
                snap = read_snapshot(env.get_ram())
                if farm_frames % 200 == 0:
                    farm_log.append(_snapshot_dict(snap))
            farm_log.append({"done": True, **_snapshot_dict(snap), "frames": farm_frames})

        # Continue with full hops from current screen index if possible.
        hops = FULL_HOPS
        # If we started mid-path, find first hop whose target is ahead.
        start_idx = 0
        for i, hop in enumerate(hops):
            if hop.target == snap.screen:
                start_idx = i + 1
                break
            # If currently on a hop source screen matching previous target chain.
        # Simpler: rebuild remaining hops by target sequence from current.
        targets = [h.target for h in hops]
        if snap.screen in targets:
            start_idx = targets.index(snap.screen) + 1
        remaining = hops[start_idx:]

        nav2 = OverworldToLevel2Controller(
            hops=remaining,
            require_level2_screen=stop_screen == 0x3C and not enter_dungeon,
            require_dungeon=enter_dungeon,
        )
        # If remaining empty and already on 0x3C, door hunt only.
        if not remaining and snap.screen == 0x3C:
            nav2 = OverworldToLevel2Controller(
                hops=(),
                require_level2_screen=not enter_dungeon,
                require_dungeon=enter_dungeon,
            )

        # Manual step loop with better logging than run_controller_stage.
        trail: list[dict] = []
        frames = 0
        last_screen = snap.screen
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen:
                trail.append({"f": frames, **_snapshot_dict(snap)})
                last_screen = snap.screen
                png = RECORDINGS_DIR / f"{tag}_sc{snap.screen:02x}.png"
                save_rgb_png(obs, png)

            if snap.mode == 17:
                break
            if enter_dungeon and snap.level == 2:
                nav2.success = True
                break
            if (
                not enter_dungeon
                and snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == stop_screen
            ):
                # Need to still drive door if stop is 0x3C via controller.
                if stop_screen != 0x3C or not nav2.require_level2_screen:
                    nav2.success = True
                    break
            act = nav2.step(snap)
            obs, *_ = env.step(act.action)
            frames += 1
            if nav2.success or nav2.phase.name == "FAILED":
                break

        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, png)
        ok = (
            (enter_dungeon and snap.level == 2)
            or (
                not enter_dungeon
                and snap.level == 0
                and snap.screen == stop_screen
                and snap.mode == PLAY_MODE
            )
            or nav2.success
        )
        checkpoint = None
        if ok and save_checkpoint:
            name = "Level2Entrance" if snap.level == 2 else f"OW_{snap.screen:02X}"
            checkpoint = str(save_state(env, GAME_DIR, GAME, name))

        return {
            "ok": ok,
            "entry": entry,
            "prefix": prefix_report,
            "farm": farm_log,
            "trail": trail,
            "nav": nav2.report(),
            "remaining_hops": [
                {"t": f"0x{h.target:02x}", "d": h.direction} for h in remaining
            ],
            "final": _snapshot_dict(snap),
            "screenshot": str(png),
            "checkpoint": checkpoint,
            "frames": frames,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level1ExitOverworld")
    p.add_argument("--max-frames", type=int, default=12000)
    p.add_argument("--stop-screen", type=lambda s: int(s, 0), default=0x3C)
    p.add_argument("--enter-dungeon", action="store_true")
    p.add_argument(
        "--farm-hearts",
        type=int,
        default=2,
        help="Minimum filled hearts before leaving 0x4A (0=skip farm)",
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l2_suffix")
    args = p.parse_args(argv)

    rep = run_probe(
        start_state=args.from_state,
        max_frames=args.max_frames,
        stop_screen=args.stop_screen,
        enter_dungeon=args.enter_dungeon,
        farm_hearts_min=args.farm_hearts,
        save_checkpoint=args.save_state,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}_probe.json"
    write_json_report(out, rep)
    fin = rep["final"]
    print(
        f"ok={rep['ok']} sc={fin['screen']:#04x} lvl={fin['level']} "
        f"hp={fin['health']:#04x} hearts={fin['hearts']} "
        f"xy=({fin['x']},{fin['y']}) frames={rep.get('frames')} "
        f"trail={len(rep.get('trail', []))}"
    )
    for t in rep.get("trail", [])[-12:]:
        print(
            f"  f={t['f']} sc={t['screen']:#04x} hp={t['health']:#04x} "
            f"xy=({t['x']},{t['y']})"
        )
    print(f"report={out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
