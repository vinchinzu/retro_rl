"""Run the first-missiles route (verified through the upper west shaft).

Examples::

    uv run python metroid/scripts/run_first_missiles.py
    uv run python metroid/scripts/run_first_missiles.py --natural-entry
    uv run python metroid/scripts/run_first_missiles.py --from-level1
    uv run python metroid/scripts/run_first_missiles.py --natural-entry --screen-timing
    uv run python metroid/scripts/run_first_missiles.py --screen-timing  # AfterMorph diag
"""

from __future__ import annotations

import argparse
import json

from metroid.first_missiles import (
    SEGMENT_MAX_FRAMES,
    FirstMissilesController,
    missiles_segment_success,
)
from metroid.menus import boot_to_level1_script
from metroid.morph_ball import MorphBallController, morph_segment_success
from metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR, SCREEN_TIMINGS_DIR
from metroid.ram import (
    is_level1_ready,
    parse_game_state,
    read_snapshot,
)
from metroid.screen_timing_session import (
    ScreenTimingSession,
    default_timing_artifact_path,
)
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

def _boot_to_ready(env, timing: ScreenTimingSession | None) -> tuple[object, int]:
    frame = 0
    obs = None
    stable = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if timing is not None:
            timing.observe_env(env, phase="BOOT")
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            stable += 1
            if stable >= 15:
                return obs, frame
        else:
            stable = 0
    return obs, frame

def _step_idle(
    env,
    timing: ScreenTimingSession | None,
    *,
    phase: str,
    n: int = 1,
) -> object:
    obs = None
    for _ in range(n):
        obs, *_ = env.step(nes_idle_action())
        if timing is not None:
            timing.observe_env(env, phase=phase)
    return obs

def run_once(
    *,
    natural_entry: bool = False,
    from_level1: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "first_missiles",
    screen_timing: bool = False,
) -> dict:
    configure_headless()
    if natural_entry:
        start_state = "NONE"
        entry_mode = "natural"
        diagnostic_state: str | None = None
    elif from_level1:
        start_state = "Level1"
        entry_mode = "level1"
        diagnostic_state = "Level1"
    else:
        start_state = "AfterMorph"
        entry_mode = "after_morph"
        diagnostic_state = "AfterMorph"

    timing: ScreenTimingSession | None = None
    if screen_timing:
        timing = ScreenTimingSession(
            enabled=True,
            source=f"run_first_missiles:{entry_mode}",
            entry_mode=entry_mode,
            diagnostic_state_load=diagnostic_state,
        )

    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    # Level1 without morph: skip morph exit, treat as corridor start.
    controller = FirstMissilesController(start_from_corridor=from_level1)
    morph_report: dict | None = None
    timing_path: str | None = None
    try:
        obs, _ = reset_obs(env)
        boot_frames = 0
        morph_frames = 0
        if natural_entry:
            obs, boot_frames = _boot_to_ready(env, timing)
            morph = MorphBallController()
            for _ in range(5000):
                action = morph.step(env)
                obs, *_ = env.step(action.action)
                morph_frames += 1
                if timing is not None:
                    timing.observe_env(env, phase=f"MORPH_{morph.phase.name}")
                if morph.success or morph.phase.name == "FAILED":
                    break
            obs = _step_idle(env, timing, phase="MORPH_SETTLE", n=90)
            morph_report = morph.report()
            if not morph_segment_success(env):
                snap = read_snapshot(env.get_ram(), env=env)
                report_out: dict = {
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
                if timing is not None:
                    timing_report = timing.report(
                        extra={"run_error": "morph_failed"}
                    )
                    path = default_timing_artifact_path(entry_mode)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(timing_report, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    report_out["screen_timing_path"] = str(path)
                    report_out["screen_timing_summary"] = {
                        "visit_count": timing_report.get("visit_count"),
                        "bottleneck": (timing_report.get("extra") or {}).get(
                            "bottleneck"
                        ),
                    }
                return report_out
        else:
            obs = _step_idle(env, timing, phase="STATE_SETTLE", n=1)
            # AfterMorph may still be in item fanfare (mode 9).
            for _ in range(180):
                obs = _step_idle(env, timing, phase="FANFARE_WAIT", n=1)
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
            "evaluation_class": (
                "clean_natural_entry"
                if natural_entry
                else f"diagnostic_state_load:{start_state}"
            ),
        }

        for _ in range(max_frames):
            action = controller.step(env)
            obs, *_ = env.step(action.action)
            if timing is not None:
                timing.observe_env(env, phase=controller.phase.name)
            if controller.terminal:
                break

        if controller.success:
            obs = _step_idle(env, timing, phase="SUCCESS_SETTLE", n=60)

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

        result_payload: dict = {
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
        if timing is not None:
            written = timing.report(
                extra={
                    "controller_phase": controller.phase.name,
                    "controller_frames": controller.frames,
                    "final_map": list(snap.map_cell),
                    "final_xy": [snap.samus_x, snap.samus_y],
                    "missile_capacity": snap.missile_capacity,
                    "health": snap.health_units,
                }
            )
            path = default_timing_artifact_path(entry_mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(written, indent=2) + "\n", encoding="utf-8"
            )
            timing_path = str(path)
            result_payload["screen_timing_path"] = timing_path
            result_payload["screen_timing_summary"] = {
                "visit_count": written.get("visit_count"),
                "total_screen_frames": written.get("total_screen_frames"),
                "total_dwell_frames": written.get("total_dwell_frames"),
                "total_transition_frames": written.get("total_transition_frames"),
                "bottleneck": (written.get("extra") or {}).get("bottleneck"),
                "evaluation_class": (written.get("extra") or {}).get(
                    "evaluation_class"
                ),
                "absolute_frames": (written.get("extra") or {}).get(
                    "absolute_frames"
                ),
            }
        return result_payload
    finally:
        env.close()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Power-on → morph → first missiles attempt (Clean evaluation)",
    )
    parser.add_argument(
        "--from-level1",
        action="store_true",
        help="Start at Level1 (no morph); east corridor probe only "
        "(diagnostic state load)",
    )
    parser.add_argument(
        "--screen-timing",
        action="store_true",
        help=(
            "Opt-in map-cell hop timing via metroid.screen_timer; "
            f"writes JSON under {SCREEN_TIMINGS_DIR}/"
        ),
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
            screen_timing=args.screen_timing,
        )
        reports.append(rep)
        fin = rep.get("final", {})
        ctrl = rep.get("controller") or {}
        timing_note = ""
        if rep.get("screen_timing_path"):
            bn = (rep.get("screen_timing_summary") or {}).get("bottleneck") or {}
            long = bn.get("longest_by_screen_frames") or {}
            timing_note = (
                f" visits={bn.get('visit_count')} "
                f"longest={long.get('map_cell')}->{long.get('dest_map_cell')} "
                f"({long.get('screen_frames')}f) "
                f"path={rep.get('screen_timing_path')}"
            )
        print(
            f"trial={i} ok={rep.get('ok')} frames={ctrl.get('frames')} "
            f"cap={fin.get('missile_capacity')} "
            f"map={fin.get('map')} phase={ctrl.get('phase')} "
            f"equip=0x{int(fin.get('equipment') or 0):02X}"
            f"{timing_note}"
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
        "screen_timing": args.screen_timing,
        "trials": args.trials,
        "successes": sum(1 for r in reports if r.get("ok")),
        "reports": reports,
        "note": (
            "Verified naturally through three blue doors, the (11,13) west-"
            "shaft platform, and the leftover climb onto (11,12). Expect "
            "ok=false at FRONTIER until the upper shaft, bridge, east shaft, "
            "and missile pickup clear. AfterMorph/Level1 starts are "
            "diagnostic state loads, not Clean natural-entry evidence. "
            "Opt-in --screen-timing writes hop timing "
            f"under {SCREEN_TIMINGS_DIR}/."
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
