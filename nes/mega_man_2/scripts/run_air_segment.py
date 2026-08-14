"""Clear Air Man screens with hard timeout.

Success: ``camera_x_screen`` ≥ ``--target-screen`` with health > 0 and not
fallen, within ``--max-frames``.

Verified (Clean Bronze):

- target 1 from ``Level1``: legacy ``AirScreen1Policy`` (~248f)
- target 2 from ``Level1``: ``AirManPolicy`` (~521f, HP 22, 3/3; 2026-08-08)
- target 2 from ``AirLanded``: ``AirManPolicy(start=landed)`` (~225f, 3/3)
- target 3 from ``AirScreen2``: ``AirManPolicy(start=screen2)`` (~241f, HP 20, 3/3; 2026-08-09)
- target 4 from ``AirScreen2``: same (~502f, HP 16, 3/3)

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/run_air_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirLanded --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 4 --trials 3
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.policy import AirManPolicy, AirScreen1Policy
from mega_man_2.ram import (
    ADDR_CAMERA_X_SCREEN,
    ADDR_HEALTH,
    ADDR_LIVES,
    camera_progress_x,
    is_fallen,
    parse_game_state,
    player_screen_x,
    player_screen_y,
)
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

DEFAULT_STATE = "Level1"
DEFAULT_TARGET_SCREEN = 2

def _make_policy(*, state_name: str, target_screen: int):
    """Pick policy for start state / target."""
    if target_screen <= 1:
        return AirScreen1Policy(target_camera_screen=target_screen)
    # Late-stage frame recipes are indexed from AirScreen2; later checkpoints
    # (AirScreen3/4 mid-air, AirFanPlatform grounded) reuse screen2 phases only
    # as a starting point — post-s4 still needs a new recipe (see STATUS).
    if state_name.startswith(
        ("AirScreen2", "AirScreen3", "AirScreen4", "AirFanPlatform")
    ):
        start = "screen2"
    elif state_name.startswith("AirLanded"):
        start = "landed"
    else:
        start = "level1"
    return AirManPolicy(target_camera_screen=target_screen, start=start)

def run_air_segment(
    *,
    state_name: str = DEFAULT_STATE,
    target_screen: int = DEFAULT_TARGET_SCREEN,
    max_frames: int = 4000,
    out_dir: Path | None = None,
    save_clear: bool = True,
    trials: int = 1,
) -> dict[str, Any]:
    """Load checkpoint, run policy until screen target or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "air_segment")
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
            )
        )

    successes = sum(1 for r in trial_reports if r["success"])
    report: dict[str, Any] = {
        "success": successes == trials,
        "trials": trials,
        "successes": successes,
        "target_screen": target_screen,
        "start_state": state_name,
        "trial_reports": trial_reports,
        "notes": (
            "Air Man camera X screen ≥ target. "
            "Level1→2: AirManPolicy (early 50/12, land jump, mid 50/12, gap@142). "
            "AirScreen2→3/4: start=screen2 (approach 45/16, fan hold 145–180, late 40/16)."
        ),
    }
    write_json_report(out / "air_segment.json", report)
    last = trial_reports[-1]
    print(
        f"outcome={'success' if report['success'] else 'fail'} "
        f"{successes}/{trials} last_frames={last['frames']} "
        f"hp={last['final_health']} screen={last['final_camera_screen']} "
        f"prog={last['final_progress_x']}"
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
) -> dict[str, Any]:
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    policy = _make_policy(state_name=state_name, target_screen=target_screen)
    reasons: dict[str, int] = {}
    screenshots: list[str] = []
    saved: list[str] = []
    prefix = f"t{trial:02d}"
    png = save_rgb_png(obs, out / f"{prefix}_0000_start.png")
    screenshots.append(png.name)

    start = parse_game_state(env.get_ram(), frame=0)
    outcome = "timeout"
    final_health = start.health
    final_screen = int(start.extras.get("camera_x_screen", 0))
    final_progress = int(start.extras.get("progress_x", 0))
    end_frame = 0

    for frame in range(1, max_frames + 1):
        ram = env.get_ram()
        health = int(ram[ADDR_HEALTH])
        lives = int(ram[ADDR_LIVES])
        cam_scr = int(ram[ADDR_CAMERA_X_SCREEN])
        fallen = is_fallen(ram)
        final_health = health
        final_screen = cam_scr
        final_progress = camera_progress_x(ram)
        end_frame = frame

        if health == 0 or lives <= 0 or fallen:
            outcome = "death"
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}_death.png")
            screenshots.append(png.name)
            break

        if cam_scr >= target_screen:
            outcome = "success"
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}_clear.png")
            screenshots.append(png.name)
            if save_clear:
                tag = f"AirScreen{target_screen}"
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
        )
        reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
        obs, *_ = env.step(tick.action)

        if frame % 100 == 0:
            png = save_rgb_png(obs, out / f"{prefix}_{frame:04d}.png")
            screenshots.append(png.name)
            print(
                f"t{trial} f={frame} scr={cam_scr} prog={final_progress} "
                f"hp={health} sx={player_screen_x(ram)} sy={player_screen_y(ram)}"
            )

    env.close()
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "trial": trial,
        "final_health": final_health,
        "final_camera_screen": final_screen,
        "final_progress_x": final_progress,
        "frames": end_frame,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "start": {
            "health": start.health,
            "lives": start.lives,
            "camera_x_screen": int(start.extras.get("camera_x_screen", 0)),
            "progress_x": int(start.extras.get("progress_x", 0)),
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
    args = parser.parse_args()
    report = run_air_segment(
        state_name=args.state,
        target_screen=args.target_screen,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
        trials=args.trials,
    )
    raise SystemExit(0 if report["success"] else 1)

if __name__ == "__main__":
    main()
