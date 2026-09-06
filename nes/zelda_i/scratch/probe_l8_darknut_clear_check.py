"""Fixture check: run Level8DarknutKeyController from a settled 0x3E / 0x5E.

Development evidence only. Starts from a disclosed recon fixture, Survival
health refill, normal input, no RAM writes.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_darknut_clear_check.py \
        --from-state Level8Interior3EReconFixture --tag t1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level8.magic_key import (
    make_blue_gohma_1e_controller,
    make_magic_key_stairs_live_controller,
)
from zelda_i.level8.north_column import (
    make_darknut_key_controller,
    make_north_manhandla_controller,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default="Level8Interior3EReconFixture")
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--max-frames", type=int, default=60000)
    ap.add_argument(
        "--stages", default="darknut", help="'darknut' or 'manhandla,darknut'"
    )
    ap.add_argument("--poke-keys", type=int, default=None)
    ap.add_argument("--poke-bombs", type=int, default=None)
    ap.add_argument("--poke-health", type=lambda v: int(v, 0), default=None)
    args = ap.parse_args()

    configure_headless()
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    factories = {
        "manhandla": make_north_manhandla_controller,
        "darknut": make_darknut_key_controller,
        "gohma": make_blue_gohma_1e_controller,
        "stairs": make_magic_key_stairs_live_controller,
    }
    try:
        obs, _ = reset_obs(env)
        if args.poke_keys is not None:
            env.get_ram()[0x066E] = args.poke_keys
        if args.poke_bombs is not None:
            env.get_ram()[0x0658] = args.poke_bombs
        if args.poke_health is not None:
            env.get_ram()[0x066F] = args.poke_health
        s0 = read_snapshot(env.get_ram())
        print(
            f"start L{s0.level} 0x{s0.screen:02x} m{s0.mode} "
            f"({s0.link_x},{s0.link_y}) keys={s0.keys} bombs={s0.bombs}"
        )
        room_hist: list[str] = []
        last_room = None
        frame = 0
        ok_all = True
        stage_reports = []
        for stage_name in args.stages.split(","):
            ctl = factories[stage_name]()
            bind = getattr(ctl, "bind_env", None)
            if callable(bind):
                bind(env)
            f0 = frame
            diag: list[str] = []
            while frame < args.max_frames:
                frame += 1
                snap = read_snapshot(env.get_ram())
                if snap.screen != last_room:
                    clr = getattr(ctl, "_clear", None)
                    cinfo = (
                        f" clear={clr.phase.name}/{clr.frames}"
                        if clr is not None
                        else ""
                    )
                    room_hist.append(
                        f"0x{snap.screen:02x}@{frame} "
                        f"({snap.link_x},{snap.link_y}) k={snap.keys}{cinfo}"
                    )
                    last_room = snap.screen
                if frame % 500 == 0:
                    clr = getattr(ctl, "_clear", None)
                    diag.append(
                        f"f{frame} 0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
                        f"clr={None if clr is None else clr.phase.name + '/' + str(clr.frames)}"
                    )
                action = ctl.step(snap).action
                obs, *_ = env.step(action)
                assist.apply_env(env, frame=frame)
                if getattr(ctl, "success", False) or getattr(ctl, "failed", False):
                    break
            r = ctl.report()
            stage_reports.append(
                {
                    "stage": stage_name,
                    "success": bool(ctl.success),
                    "failed": bool(ctl.failed),
                    "frames": frame - f0,
                    "notes": r.get("notes"),
                    "diag_tail": diag[-12:],
                }
            )
            if not ctl.success:
                ok_all = False
                break
        snap = read_snapshot(env.get_ram())
        out = {
            "from_state": args.from_state,
            "success": ok_all,
            "stage_reports": stage_reports,
            "total_frames": frame,
            "end": {
                "level": int(snap.level),
                "screen": f"0x{snap.screen:02x}",
                "mode": int(snap.mode),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "keys": int(snap.keys),
                "bombs": int(snap.bombs),
                "magic_key": int(env.get_ram()[0x0664]),
                "rupees": int(snap.rupees),
            },
            "room_hist": room_hist,
        }
        print(json.dumps(out, indent=1))
        return 0 if ok_all else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
