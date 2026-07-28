"""Run the first-missiles route (verified through the upper west shaft).

Examples::

    uv run python metroid/scripts/run_first_missiles.py
    uv run python metroid/scripts/run_first_missiles.py --natural-entry
    uv run python metroid/scripts/run_first_missiles.py --from-level1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metroid.first_missiles import (
    SEGMENT_MAX_FRAMES,
    FirstMissilesController,
    missiles_segment_success,
)
from metroid.menus import boot_to_level1_script
from metroid.morph_ball import MorphBallController, morph_segment_success
from metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR
from metroid.ram import (
    is_level1_ready,
    parse_game_state,
    read_snapshot,
)
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
    from_level1: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "first_missiles",
) -> dict:
    configure_headless()
    if natural_entry:
        start_state = "NONE"
    elif from_level1:
        start_state = "Level1"
    else:
        start_state = "AfterMorph"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    # Level1 without morph: skip morph exit, treat as corridor start.
    controller = FirstMissilesController(start_from_corridor=from_level1)
    morph_report: dict | None = None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        boot_frames = 0
        morph_frames = 0
        if natural_entry:
            obs, boot_frames = _boot_to_ready(env)
            morph = MorphBallController()
            for _ in range(5000):
                action = morph.step(env)
                obs, *_ = env.step(action.action)
                morph_frames += 1
                if morph.success or morph.phase.name == "FAILED":
                    break
            for _ in range(90):
                obs, *_ = env.step(nes_idle_action())
            morph_report = morph.report()
            if not morph_segment_success(env):
                snap = read_snapshot(env.get_ram(), env=env)
                return {
                    "ok": False,
                    "error": "morph_failed",
                    "morph": morph_report,
                    "final": {
                        "map": list(snap.map_cell),
                        "xy": [snap.samus_x, snap.samus_y],
                        "health": snap.health_units,
                        "equipment": snap.equipment,
                        "missile_capacity": snap.missile_capacity,
                    },
                }
        else:
            obs, *_ = env.step(nes_idle_action())
            # AfterMorph may still be in item fanfare (mode 9).
            for _ in range(180):
                obs, *_ = env.step(nes_idle_action())
                if read_snapshot(env.get_ram()).game_mode == 3:
                    break

        snap0 = read_snapshot(env.get_ram(), env=env)
        entry = {
            "natural_entry": natural_entry,
            "from_level1": from_level1,
            "map": list(snap0.map_cell),
            "xy": [snap0.samus_x, snap0.samus_y],
            "equipment": snap0.equipment,
            "missile_capacity": snap0.missile_capacity,
            "boot_frames": boot_frames,
            "morph_frames": morph_frames,
        }

        for _ in range(max_frames):
            action = controller.step(env)
            obs, *_ = env.step(action.action)
            if controller.terminal:
                break

        if controller.success:
            for _ in range(60):
                obs, *_ = env.step(nes_idle_action())

        ram = env.get_ram()
        snap = read_snapshot(ram, env=env)
        state = parse_game_state(ram, frame=controller.frames, env=env)
        ok = missiles_segment_success(env) or controller.success
        mode = (
            "natural"
            if natural_entry
            else ("level1" if from_level1 else "after_morph")
        )
        png = RECORDINGS_DIR / f"{tag}_{mode}.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "entry": entry,
            "morph": morph_report,
            "controller": controller.report(),
            "final": {
                "map": list(snap.map_cell),
                "xy": [snap.samus_x, snap.samus_y],
                "health": snap.health_units,
                "equipment": snap.equipment,
                "morph_ball": snap.morph_ball,
                "missile_capacity": snap.missile_capacity,
                "missiles": snap.missiles,
                "missiles_enabled": snap.missiles_enabled,
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
        help="Power-on → morph → first missiles attempt",
    )
    parser.add_argument(
        "--from-level1",
        action="store_true",
        help="Start at Level1 (no morph); east corridor probe only",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        tag = f"first_missiles_t{i}"
        rep = run_once(
            natural_entry=args.natural_entry,
            from_level1=args.from_level1,
            max_frames=args.max_frames,
            tag=tag,
        )
        reports.append(rep)
        fin = rep.get("final", {})
        ctrl = rep.get("controller") or {}
        print(
            f"trial={i} ok={rep.get('ok')} frames={ctrl.get('frames')} "
            f"cap={fin.get('missile_capacity')} "
            f"map={fin.get('map')} phase={ctrl.get('phase')} "
            f"equip=0x{int(fin.get('equipment') or 0):02X}"
        )

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.natural_entry:
        out_name = "first_missiles_natural.json"
    elif args.from_level1:
        out_name = "first_missiles_level1.json"
    else:
        out_name = "first_missiles_after_morph.json"
    out = RECORDINGS_DIR / out_name
    payload = {
        "segment": "first_missiles",
        "natural_entry": args.natural_entry,
        "from_level1": args.from_level1,
        "trials": args.trials,
        "successes": sum(1 for r in reports if r.get("ok")),
        "reports": reports,
        "note": (
            "Verified naturally through three blue doors and the third stable "
            "west-shaft platform. Expect ok=false at FRONTIER until the upper "
            "shaft, bridge, east shaft, and missile pickup clear."
        ),
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    # Non-zero only on hard failures; reaching the verified frontier is useful.
    if any(r.get("error") for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
