"""Clean fixture-live L2 heatmap hotspots 0x4f / 0x0e. No bomb/key poke.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level2_hotspots.py \
      --room 0x4f --from-state Level2_4F --no-infinite-life --no-video --trials 1
    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level2_hotspots.py \
      --room 0x0e --from-state Level2_0E --no-infinite-life --no-video --trials 1

``route_eligible=false``. Integrator promotes. Spine CLI has no --from-state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _NES_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.level2.spine import Level2Clear4fController
from zelda_i.level2.tf_spine import Level2DodongoController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage
from zelda_i.runner import (
    add_common_args,
    add_video_args,
    make_assist,
    resolve_video,
)
from zelda_i.screen_glance import leftover_from_snapshot


def _run_hotspot(env, *, room: str, assist=None) -> dict:
    if room == "0x0e":
        controller = Level2DodongoController()
        name = "fight_dodongo"
    else:
        controller = Level2Clear4fController()
        name = "clear4f_boom"
    obs, stage = run_controller_stage(
        env,
        None,
        name=name,
        controller=controller,
        max_frames=controller.max_frames,
        assist=assist,
    )
    snap = read_snapshot(env.get_ram())
    leftover = leftover_from_snapshot(snap)
    leftover["health"] = int(snap.health)
    deaths = 1 if int(snap.mode) == 17 else 0
    lo = int(snap.health) & 0x0F
    hi = (int(snap.health) >> 4) & 0x0F
    return {
        "ok": bool(controller.success and deaths == 0),
        "failed_stage": None if controller.success else name,
        "stage": stage.report(),
        "controller": controller.report(),
        "leftover": leftover,
        "deaths": deaths,
        "hearts_lo": lo,
        "hearts_hi": hi,
        "hearts_lo_eq_hi": lo == hi,
        "frames": stage.end_frame,
        "route_eligible": False,
        "natural_entry": False,
        "intervention_class": "clean",
        "obs": obs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", choices=("0x0e", "0x4f"), default="0x0e")
    add_common_args(
        parser, default_state="Level2_0E", default_tag="l2_hotspot"
    )
    add_video_args(parser, default_on=False)
    parser.set_defaults(infinite_life=False)
    args = parser.parse_args()
    if args.room == "0x4f" and args.from_state == "Level2_0E":
        args.from_state = "Level2_4F"
    resolve_video(args, default_path=RECORDINGS_DIR / f"{args.tag}.mp4")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    ok = True
    for trial in range(int(args.trials)):
        configure_headless()
        env = make_env(
            GAME, args.from_state, GAME_DIR, render_mode="rgb_array"
        )
        env.reset()
        assist = make_assist(bool(args.infinite_life))
        try:
            result = _run_hotspot(env, room=args.room, assist=assist)
            obs = result.pop("obs", None)
            if obs is None:
                obs, *_ = env.step(nes_idle_action())
            png = RECORDINGS_DIR / f"{args.tag}_t{trial}_final.png"
            save_rgb_png(obs, png)
            result["png"] = str(png)
            result["trial"] = trial
            result["from_state"] = args.from_state
            result["room"] = args.room
            result["infinite_life"] = bool(args.infinite_life)
            leftover = result.get("leftover") or {}
            print(
                f"trial={trial} ok={result.get('ok')} failed={result.get('failed_stage')} "
                f"deaths={result.get('deaths')} "
                f"room=0x{int(leftover.get('room') or leftover.get('screen') or 0):02x} "
                f"xy={leftover.get('xy')} mode={leftover.get('mode')} "
                f"tf=0x{int(leftover.get('triforce') or 0):02x} "
                f"keys={leftover.get('keys')} bombs={leftover.get('bombs')} "
                f"health={leftover.get('health')} lo==hi={result.get('hearts_lo_eq_hi')}"
            )
            reports.append(result)
            ok = ok and bool(result.get("ok"))
        finally:
            env.close()
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(
        out,
        {
            "runner": "run_level2_hotspots.py",
            "from_state": args.from_state,
            "room": args.room,
            "infinite_life": bool(args.infinite_life),
            "route_eligible": False,
            "reports": reports,
            "ok": ok,
        },
    )
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
