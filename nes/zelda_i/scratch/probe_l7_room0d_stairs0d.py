"""Drive `level7.stairs0d.Level7Stairs0DController` from the 0x0D pin.

RAM claim is `zelda_i.level7.stairs0d.RAM_CLAIM` (written before the first
live trial). Dest is RAM: mode 9 cellar 0x7B. No position pokes.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \\
        nes/zelda_i/scratch/probe_l7_room0d_stairs0d.py --tag 20260904_S1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import ascii_room, link_cell, stair_cells
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level7.cellar import make_nose_cellar_cross_controller
from zelda_i.level7.stairs0d import (
    RAM_CLAIM,
    make_stairs0d_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import make_assist

CELLAR_MODES = {9, 10, 11, 16}


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cell": list(link_cell(int(s.link_x), int(s.link_y))),
        "colliding_tile": int(s.colliding_tile),
        "level": int(s.level),
        "stair_cells": [list(c) for c in stair_cells(ram)],
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "candle": int(s.candle),
        "triforce": int(s.triforce),
        "blocks": [
            [int(o.slot), int(o.x), int(o.y)]
            for o in s.objects
            if int(o.type_id) == 0x68
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260904_S1")
    ap.add_argument("--from-state", default="Level7Interior0DClearedReconFixture")
    ap.add_argument("--save-fixture", default="")
    ap.add_argument("--then-cross", action="store_true",
                    help="continue with the 2/2 0x7B B->A cross to play 0x29")
    args = ap.parse_args()
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "ram_claim": RAM_CLAIM}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        out["map_before"] = ascii_room(env.get_ram())
        print("START", out["start"])
        print(out["map_before"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")

        ctl = make_stairs0d_controller()
        frame = 0
        for _ in range(ctl.max_frames):
            snap = read_snapshot(env.get_ram())
            act = ctl.step(snap)
            env.step(act.action)
            assist.apply_env(env, frame=frame)
            frame += 1
            if ctl.success or ctl.failed:
                break
        out["controller_frames"] = int(ctl.frames)
        out["total_frames"] = frame
        out["success"] = bool(ctl.success)
        out["failed"] = bool(ctl.failed)
        out["report"] = ctl.report()
        out["after_controller"] = _glance(env)
        out["map_after"] = ascii_room(env.get_ram())
        print("CTL", ctl.success, ctl.failed, ctl.frames, ctl.phase.name)
        print("NOTES", ctl.notes)
        print("AFTER", out["after_controller"])
        print(out["map_after"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_dest.png")

        for _ in range(240):
            s = read_snapshot(env.get_ram())
            if int(s.mode) == PLAY_MODE and int(s.screen) != 0x0D:
                break
            env.step(nes_idle_action())
            assist.apply_env(env, frame=frame)
            frame += 1
        out["settled"] = _glance(env)
        print("SETTLED", out["settled"])

        if args.then_cross and ctl.success:
            cross = make_nose_cellar_cross_controller()
            for _ in range(cross.max_frames):
                snap = read_snapshot(env.get_ram())
                act = cross.step(snap)
                env.step(act.action)
                assist.apply_env(env, frame=frame)
                frame += 1
                if cross.success or cross.failed:
                    break
            out["cross"] = {
                "success": bool(cross.success),
                "failed": bool(cross.failed),
                "frames": int(cross.frames),
                "notes": list(cross.notes),
                "glance": _glance(env),
            }
            print("CROSS", cross.success, cross.failed, cross.frames)
            print("CROSS AT", out["cross"]["glance"])
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_cross.png"
            )

        end = _glance(env)
        end["deaths"] = int(assist.telemetry.deaths)
        end["progression_writes"] = int(assist.telemetry.progression_writes)
        end["capacity_writes"] = int(assist.telemetry.capacity_writes)
        end["position_writes"] = 0
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        if args.save_fixture:
            path = save_state(env, GAME_DIR, GAME, args.save_fixture)
            src = state_path(GAME_DIR, GAME, args.from_state)
            write_state_provenance(
                path,
                source_state_path=src if src.exists() else None,
                request={
                    "bead": "rr-8t4.3",
                    "phase": "level7_tip_of_nose_stairs",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "development_only": True,
                    "fixture_writes": [],
                    "notes": [
                        f"Derived from {args.from_state}: live "
                        "Level7Stairs0DController walk-on of the 0x0D hidden "
                        "staircase; no position/door/TF writes "
                        "(position_writes=0).",
                        "UnlimitedHealthAssist traversal aid only.",
                    ],
                },
                selected_trial={
                    "ok": bool(ctl.success),
                    "state": compact_snapshot(read_snapshot(env.get_ram())),
                    "glance": end,
                },
                natural_entry=False,
            )
            out["saved_fixture"] = str(path)
            print("saved", path)
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
    finally:
        env.close()


if __name__ == "__main__":
    main()
