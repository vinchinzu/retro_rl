"""Leave North Palace from Level1 with a hard timeout.

Success: engine mode ``$0736 == 5`` (overworld play), HP > 0, within
``--max-frames``.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/zelda_ii/scripts/run_leave_palace.py --trials 3
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_ii.north_palace import (
    SEGMENT_MAX_FRAMES,
    SETTLE_FRAMES,
    LeavePalacePolicy,
)
from zelda_ii.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_ii.ram import is_dead, palace_exit_success, parse_game_state, read_snapshot

DEFAULT_STATE = "Level1"
CLEAR_STATE = "NorthPalaceExit"


def run_leave_palace(
    *,
    state_name: str = DEFAULT_STATE,
    max_frames: int = SEGMENT_MAX_FRAMES,
    out_dir: Path | None = None,
    save_clear: bool = True,
    trials: int = 1,
) -> dict[str, Any]:
    """Load checkpoint, walk west until overworld or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "leave_palace")
    out.mkdir(parents=True, exist_ok=True)

    trial_reports = [
        _run_one(
            state_name=state_name,
            max_frames=max_frames,
            out=out,
            save_clear=save_clear and trial == 1,
            trial=trial,
        )
        for trial in range(1, trials + 1)
    ]
    successes = sum(1 for r in trial_reports if r["success"])
    last = trial_reports[-1]
    report: dict[str, Any] = {
        "success": successes == trials,
        "trials": trials,
        "successes": successes,
        "start_state": state_name,
        "palace_left": bool(last.get("palace_left")),
        "leftover": last.get("leftover"),
        "trial_reports": trial_reports,
        "notes": (
            "North Palace west exit. Stop on $0736 overworld play (5). "
            "Level1 LEFT walk; idle on transition so leftover is the palace tile."
        ),
    }
    write_json_report(out / "leave_palace.json", report)
    leftover = last.get("leftover") or {}
    print(
        f"outcome={'success' if report['success'] else 'fail'} "
        f"{successes}/{trials} last_frames={last['frames']} "
        f"palace_left={report['palace_left']} "
        f"leftover_mode={leftover.get('engine_mode')} "
        f"ow=({leftover.get('ow_x')},{leftover.get('ow_y')}) "
        f"hp={leftover.get('health')}"
    )
    return report


def _run_one(
    *,
    state_name: str,
    max_frames: int,
    out: Path,
    save_clear: bool,
    trial: int,
) -> dict[str, Any]:
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    policy = LeavePalacePolicy()
    reasons: dict[str, int] = {}
    prefix = f"t{trial:02d}"
    screenshots = [save_rgb_png(obs, out / f"{prefix}_0000_start.png").name]
    start = parse_game_state(env.get_ram(), frame=0)
    outcome = "timeout"
    end_frame = 0
    snap = read_snapshot(env.get_ram())
    saved: list[str] = []

    try:
        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            snap = read_snapshot(ram)
            end_frame = frame
            if is_dead(ram):
                outcome = "death"
                screenshots.append(
                    save_rgb_png(obs, out / f"{prefix}_{frame:04d}_death.png").name
                )
                break
            if palace_exit_success(ram):
                outcome = "success"
                for _ in range(SETTLE_FRAMES):
                    obs, *_ = env.step(nes_idle_action())
                snap = read_snapshot(env.get_ram())
                screenshots.append(
                    save_rgb_png(obs, out / f"{prefix}_{frame:04d}_clear.png").name
                )
                if save_clear:
                    saved.append(save_state(env, GAME_DIR, GAME, CLEAR_STATE).name)
                break
            tick = policy.tick(ram)
            reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
            obs, *_ = env.step(tick.action)
    finally:
        env.close()

    leftover = snap.as_dict()
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "trial": trial,
        "frames": end_frame,
        "palace_left": leftover.get("overworld", False),
        "leftover": leftover,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "start": {
            "engine_mode": int(start.extras.get("engine_mode", 0)),
            "player_x": start.player_x,
            "player_y": start.player_y,
            "health": start.health,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    report = run_leave_palace(
        state_name=args.state,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
        trials=args.trials,
    )
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
