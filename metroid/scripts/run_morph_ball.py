"""Run the Maru Mari (Morph Ball) segment (M3 / natural-entry).

Examples::

    uv run python metroid/scripts/run_morph_ball.py
    uv run python metroid/scripts/run_morph_ball.py --natural-entry
    uv run python metroid/scripts/run_morph_ball.py --trials 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metroid.menus import boot_to_level1_script
from metroid.morph_ball import (
    SEGMENT_MAX_FRAMES,
    MorphBallController,
    morph_segment_success,
)
from metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR
from metroid.ram import is_level1_ready, parse_game_state, read_equipment, read_snapshot
from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)


def _boot_to_ready(env) -> tuple[object, int]:
    frame = 0
    obs = None
    stable = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            stable += 1
            if stable >= 15:
                return obs, frame
        else:
            stable = 0
    return obs, frame


def run_once(
    *,
    natural_entry: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "morph_ball",
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = MorphBallController()
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        boot_frames = 0
        if natural_entry:
            obs, boot_frames = _boot_to_ready(env)
        else:
            obs, *_ = env.step(nes_idle_action())

        snap0 = read_snapshot(env.get_ram(), env=env)
        entry = {
            "natural_entry": natural_entry,
            "map": list(snap0.map_cell),
            "xy": [snap0.samus_x, snap0.samus_y],
            "equipment": snap0.equipment,
            "boot_frames": boot_frames,
        }

        for _ in range(max_frames):
            action = controller.step(env)
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase.name == "FAILED":
                break

        # Brief settle for item fanfare / equipment latch.
        for _ in range(60):
            obs, *_ = env.step(nes_idle_action())

        ram = env.get_ram()
        snap = read_snapshot(ram, env=env)
        state = parse_game_state(ram, frame=controller.frames, env=env)
        ok = morph_segment_success(env) or controller.success
        png = RECORDINGS_DIR / f"{tag}_{'natural' if natural_entry else 'isolated'}.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "entry": entry,
            "controller": controller.report(),
            "final": {
                "map": list(snap.map_cell),
                "xy": [snap.samus_x, snap.samus_y],
                "equipment": snap.equipment,
                "morph_ball": snap.morph_ball,
                "capabilities": sorted(state.extras.get("capabilities", [])),
                "game_mode": state.mode.name,
            },
            "screenshot": str(png),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Boot from power-on instead of loading Level1.state",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        tag = f"morph_ball_t{i}"
        rep = run_once(
            natural_entry=args.natural_entry,
            max_frames=args.max_frames,
            tag=tag,
        )
        reports.append(rep)
        print(
            f"trial={i} ok={rep['ok']} frames={rep['controller']['frames']} "
            f"equip=0x{rep['final']['equipment']:02X} "
            f"map={rep['final']['map']} phase={rep['controller']['phase']}"
        )

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDINGS_DIR / (
        "morph_ball_natural.json" if args.natural_entry else "morph_ball_isolated.json"
    )
    payload = {
        "segment": "morph_ball",
        "natural_entry": args.natural_entry,
        "trials": args.trials,
        "successes": sum(1 for r in reports if r["ok"]),
        "reports": reports,
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
