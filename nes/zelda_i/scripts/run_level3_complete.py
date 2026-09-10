"""Clean fixture-live: Level3Entrance → TF 0x04 dest hops. No bomb poke.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level3_complete.py \
      --from-state Level3Entrance --no-infinite-life --no-video --trials 1

``route_eligible=false``. Integrator promotes. Spine CLI has no --from-state.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.level3.spine import run_level3_entrance_tf
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.runner import (
    add_common_args,
    add_video_args,
    make_assist,
    resolve_video,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser, default_state="Level3Entrance", default_tag="l3_entrance_tf"
    )
    add_video_args(parser, default_on=False)
    parser.set_defaults(infinite_life=False)
    args = parser.parse_args()
    video_path, _cfg, _intro = resolve_video(
        args, default_path=RECORDINGS_DIR / f"{args.tag}.mp4"
    )
    del video_path
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
            result = run_level3_entrance_tf(env, assist=assist)
            obs = result.pop("obs", None)
            if obs is None:
                obs, *_ = env.step(nes_idle_action())
            png = RECORDINGS_DIR / f"{args.tag}_t{trial}_final.png"
            save_rgb_png(obs, png)
            result["png"] = str(png)
            result["trial"] = trial
            result["from_state"] = args.from_state
            result["infinite_life"] = bool(args.infinite_life)
            leftover = result.get("leftover") or {}
            print(
                f"trial={trial} ok={result.get('ok')} failed={result.get('failed_stage')} "
                f"tf04={result.get('tf04')} deaths={result.get('deaths')} "
                f"room=0x{int(leftover.get('room') or leftover.get('screen') or 0):02x} "
                f"xy={leftover.get('xy')} mode={leftover.get('mode')} "
                f"tf=0x{int(leftover.get('triforce') or 0):02x} "
                f"keys={leftover.get('keys')} bombs={leftover.get('bombs')} "
                f"health={leftover.get('health')}"
            )
            reports.append(result)
            ok = ok and bool(result.get("ok"))
        finally:
            env.close()
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(
        out,
        {
            "runner": "run_level3_complete.py",
            "from_state": args.from_state,
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
