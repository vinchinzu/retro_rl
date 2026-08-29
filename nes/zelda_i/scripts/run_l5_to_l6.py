"""Assisted predecessor: L5 Triforce → OW 0x0B → Level 6 entry 0x79.

Starts from ``Level5Complete``, idles the real fanfare onto door 0x0B, then
walks ``POST_L5_TO_LEVEL6_HOPS``. Survival health refill only. No Whistle
grant, no Rod/door/key pokes.

    UV_CACHE_DIR=/tmp/retro_rl_uv_cache QT_QPA_PLATFORM=offscreen \
      uv run python nes/zelda_i/scripts/run_l5_to_l6.py \
        --infinite-life --trials 1 --tag l5_to_l6_v1
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.chain import run_controller_stage
from zelda_i.level6_hops import l6_prefix
from zelda_i.level6_overworld import level6_entrance_success
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8


def _pin(env) -> dict[str, Any]:
    snap = read_snapshot(env.get_ram())
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "triforce": snap.triforce,
        "raft": snap.raft,
        "ladder": snap.ladder,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def run_once(*, start_state: str, tag: str) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    entry = _pin(env)
    trail: list[dict[str, Any]] = [{"frame": 0, **entry}]
    last = (entry["level"], entry["screen"])
    frame_base = 0
    failed = None
    reports: list[dict[str, Any]] = []
    try:
        entry_hop = l6_prefix(env)[0]
        stages = entry_hop.stages() if callable(entry_hop.stages) else entry_hop.stages
        for name, controller, max_frames in stages:
            obs, stage = run_controller_stage(
                env,
                obs,
                name=name,
                controller=controller,
                max_frames=max_frames,
                assist=assist,
                frame_base=frame_base,
            )
            frame_base = stage.end_frame
            reports.append(stage.report())
            pin = _pin(env)
            loc = (pin["level"], pin["screen"])
            if loc != last:
                trail.append({"frame": frame_base, "stage": name, **pin})
                last = loc
            if not stage.success:
                failed = name
                break
        snap = read_snapshot(env.get_ram())
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        # Isolated Level5Complete may lack Raft/Ladder; hops still must land 0x79.
        ok = failed is None and bool(level6_entrance_success(env.get_ram()))
        spine_ok = ok and bool(entry_hop.success(snap))
        screenshot = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "spine_inventory_ok": spine_ok,
            "failed_stage": failed,
            "start_state": start_state,
            "entry": entry,
            "stages": reports,
            "trail": trail,
            "final": _pin(env),
            "assist": assist.report(),
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level5Complete")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default="l5_to_l6")
    args = parser.parse_args(argv)
    del args.infinite_life  # Survival health is required for this probe.
    reports = []
    for trial in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{trial}"
        report = run_once(start_state=args.from_state, tag=tag)
        reports.append(report)
        final = report.get("final") or {}
        print(
            f"trial={trial} ok={report.get('ok')} failed={report.get('failed_stage')} "
            f"room=0x{final.get('screen', -1):02x} xy=({final.get('x')},{final.get('y')}) "
            f"tf={final.get('triforce')} whistle={final.get('whistle')}",
            flush=True,
        )
    output = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(
        output,
        {
            "bead": "rr-g3c1",
            "segment": "l5_complete_to_l6_entrance",
            "natural_entry": False,
            "trials": reports,
            "successes": sum(bool(r.get("ok")) for r in reports),
        },
    )
    print(f"wrote {output}")
    return 0 if all(r.get("ok") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
