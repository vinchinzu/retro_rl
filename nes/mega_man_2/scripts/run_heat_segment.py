"""Clear Heat Man mid/late screens with hard timeout.

Success: ``camera_x_screen`` ≥ ``--target-screen`` with health > 0 and not
fallen, within ``--max-frames``. Recipe auto-selects from start state via
``HeatManPolicy.start_for_state``.

Verified (Clean Bronze, 2026-08-10):

- target 1 from ``Heat1``: early 50/12 (~243f, HP 24)
- target 2 from ``HeatScreen1``: early 50/12 (~194f, HP 24)
- target 3 from ``HeatScreen2``: mid 60/14 → 25/12 (~351f grounded)
- target 4 from ``HeatScreen3``: pillars 25/10 ph10 (~181f grounded)
- target 5 from ``HeatScreen4``: late 20/12 ph4 (~131f cam)
- target 7 from ``HeatScreen5Ground``: screen5 j1/LEFT/j2 + hop 9/gw3 (~305f)
- target 8 from ``HeatScreen7Mid``: screen7 high-path ladder/scroll_down (~587f)
- from ``HeatScreen8``: screen8 Yoku room → cam ≥ 9 (~680f; use
  ``--target-screen 9``). ``--yoku-land`` still hits first-block stand.
  Residual: E columns / F lava / G Sniper → boss door (rr-k1ea PARTIAL)

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen1 --target-screen 2 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen2 --target-screen 3 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen3 --target-screen 4 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen4 --target-screen 5 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen5Ground --target-screen 7 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen7Mid --target-screen 8 --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen8 --yoku-land --trials 3
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_heat_segment.py \\
  --state HeatScreen8 --target-screen 9 --trials 3
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.policy import HeatManPolicy
from mega_man_2.ram import (
    ADDR_CAMERA_STATE,
    ADDR_CAMERA_X_SCREEN,
    ADDR_CAMERA_Y,
    ADDR_HEALTH,
    ADDR_ITEMS,
    ADDR_LIVES,
    ADDR_TILE_FEET,
    ADDR_WEAPONS,
    camera_progress_x,
    is_fallen,
    parse_game_state,
    player_screen_x,
    player_screen_y,
    read_u8,
)
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

DEFAULT_STATE = "Heat1"
DEFAULT_TARGET_SCREEN = 1

def run_heat_segment(
    *,
    state_name: str = DEFAULT_STATE,
    target_screen: int = DEFAULT_TARGET_SCREEN,
    max_frames: int = 4000,
    out_dir: Path | None = None,
    save_clear: bool = True,
    trials: int = 1,
    yoku_land: bool = False,
) -> dict[str, Any]:
    """Load Heat checkpoint, run policy until screen target or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "heat_segment")
    out.mkdir(parents=True, exist_ok=True)

    trial_reports: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        trial_reports.append(
            _run_one(
                state_name=state_name,
                target_screen=target_screen,
                max_frames=max_frames,
                out=out,
                save_clear=save_clear and trial == 1,
                trial=trial,
                yoku_land=yoku_land,
            )
        )

    successes = sum(1 for r in trial_reports if r["success"])
    report: dict[str, Any] = {
        "success": successes == trials,
        "trials": trials,
        "successes": successes,
        "target_screen": target_screen,
        "yoku_land": yoku_land,
        "start_state": state_name,
        "trial_reports": trial_reports,
        "notes": (
            "Heat Man camera X screen ≥ target (or --yoku-land sy≤105 stand). "
            "Recipes: early 50/12; screen2 mid 60→25; screen3 pillars 25/10 ph10; "
            "screen4 late 20/12 ph4; screen5 j1/LEFT/j2 + hop 9/gw3 → cam≥7; "
            "screen7 high-path ladder → cam≥8 Sniper shaft; "
            "screen8 Yoku room catch → left ladder → cam≥9. "
            "E/F/G Sniper + boss door residual (rr-k1ea / rr-809 PARTIAL)."
        ),
    }
    write_json_report(out / "heat_segment.json", report)
    last = trial_reports[-1]
    print(
        f"outcome={'success' if report['success'] else 'fail'} "
        f"{successes}/{trials} last_frames={last['frames']} "
        f"hp={last['final_health']} screen={last['final_camera_screen']} "
        f"prog={last['final_progress_x']} "
        f"sx={last.get('final_sx', 0)} sy={last.get('final_sy', 0)} "
        f"wep={last.get('weapons', 0):02x} items={last.get('items', 0):02x}"
    )
    return report

def _run_one(
    *,
    state_name: str,
    target_screen: int,
    max_frames: int,
    out: Path,
    save_clear: bool,
    trial: int,
    yoku_land: bool = False,
) -> dict[str, Any]:
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    start_mode = HeatManPolicy.start_for_state(state_name)
    # Yoku-land milestone: hold target high so clear_hold does not freeze early
    policy_target = 99 if yoku_land else target_screen
    policy = HeatManPolicy(
        target_camera_screen=policy_target,
        start=start_mode,
    )
    reasons: dict[str, int] = {}
    screenshots: list[str] = []
    saved: list[str] = []
    prefix = f"t{trial:02d}"
    png = save_rgb_png(obs, out / f"{prefix}_0000_start.png")
    screenshots.append(png.name)

    start = parse_game_state(env.get_ram(), frame=0)
    start_lives = int(start.lives)
    outcome = "timeout"
    final_health = start.health
    final_screen = int(start.extras.get("camera_x_screen", 0))
    final_progress = int(start.extras.get("progress_x", 0))
    final_weapons = int(start.extras.get("weapons", 0))
    final_items = int(start.extras.get("items", 0))
    final_sx = int(start.extras.get("player_sx", 0))
    final_sy = int(start.extras.get("player_sy", 0))
    end_frame = 0
    yoku_stand_run = 0

    for frame in range(1, max_frames + 1):
        ram = env.get_ram()
        health = int(ram[ADDR_HEALTH])
        lives = int(ram[ADDR_LIVES])
        cam_scr = int(ram[ADDR_CAMERA_X_SCREEN])
        tile_feet = int(ram[ADDR_TILE_FEET])
        cam_y = int(ram[ADDR_CAMERA_Y])
        cam_st = int(ram[ADDR_CAMERA_STATE])
        sx = player_screen_x(ram)
        sy = player_screen_y(ram)
        # sy≥200 is a pit heuristic; false-fires on ladder scroll_down (sy~228)
        # while tile_feet==2 or vertical camera is mid-transition.
        on_ladder = tile_feet == 2
        vert_scroll = cam_y > 0 or (cam_st & 0x80) != 0
        fallen = is_fallen(ram) and not on_ladder and not vert_scroll
        final_health = health
        final_screen = cam_scr
        final_progress = camera_progress_x(ram)
        final_weapons = read_u8(ram, ADDR_WEAPONS)
        final_items = read_u8(ram, ADDR_ITEMS)
        final_sx = sx
        final_sy = sy
        end_frame = frame

        # feet==3 is instadeath pose; lives drop = pit/respawn without HP=0
        if health == 0 or lives <= 0 or lives < start_lives or fallen or tile_feet == 3:
            outcome = "death"
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}_death.png")
            screenshots.append(png.name)
            break

        # First Yoku land milestone (HeatScreen8): grounded sy≤105 for 3f
        if yoku_land and tile_feet == 1 and sy <= 105:
            yoku_stand_run += 1
            if yoku_stand_run >= 3:
                outcome = "success"
                png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}_yoku.png")
                screenshots.append(png.name)
                if save_clear:
                    path = save_state(env, GAME_DIR, GAME, "HeatScreen8Yoku")
                    saved.append(path.name)
                    path2 = save_state(
                        env,
                        GAME_DIR,
                        GAME,
                        f"HeatScreen8Yoku_sx{sx}_sy{sy}_hp{health}",
                    )
                    saved.append(path2.name)
                break
        else:
            yoku_stand_run = 0

        if not yoku_land and cam_scr >= target_screen:
            outcome = "success"
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}_clear.png")
            screenshots.append(png.name)
            # Only persist grounded clears so mid-air cam-hits do not clobber pins
            if save_clear and tile_feet == 1:
                tag = f"HeatScreen{target_screen}"
                path = save_state(env, GAME_DIR, GAME, tag)
                saved.append(path.name)
                path2 = save_state(
                    env, GAME_DIR, GAME, f"{tag}_scr{cam_scr}_hp{health}"
                )
                saved.append(path2.name)
            break

        tick = policy.tick(
            frame=frame,
            health=health,
            camera_x_screen=cam_scr,
            fallen=fallen,
            tile_feet=tile_feet,
        )
        reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
        obs, *_ = env.step(tick.action)

        if frame % 100 == 0:
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}.png")
            screenshots.append(png.name)
            print(
                f"t{trial} f={frame} scr={cam_scr} prog={final_progress} "
                f"hp={health} sx={sx} sy={sy} mode={start_mode}"
            )

    env.close()
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "trial": trial,
        "final_health": final_health,
        "final_camera_screen": final_screen,
        "final_progress_x": final_progress,
        "final_sx": final_sx,
        "final_sy": final_sy,
        "weapons": final_weapons,
        "items": final_items,
        "frames": end_frame,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "policy_start": start_mode,
        "yoku_land": yoku_land,
        "start": {
            "health": start.health,
            "lives": start.lives,
            "camera_x_screen": int(start.extras.get("camera_x_screen", 0)),
            "progress_x": int(start.extras.get("progress_x", 0)),
            "weapons": int(start.extras.get("weapons", 0)),
            "items": int(start.extras.get("items", 0)),
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--target-screen", type=int, default=DEFAULT_TARGET_SCREEN)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--yoku-land",
        action="store_true",
        help="Success = grounded Yoku stand sy≤105 (HeatScreen8 first land)",
    )
    args = parser.parse_args()
    report = run_heat_segment(
        state_name=args.state,
        target_screen=args.target_screen,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
        trials=args.trials,
        yoku_land=args.yoku_land,
    )
    raise SystemExit(0 if report["success"] else 1)

if __name__ == "__main__":
    main()
