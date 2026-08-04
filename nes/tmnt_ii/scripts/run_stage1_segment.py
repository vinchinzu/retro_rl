"""Clear first Stage 1 wave from Level1 (score target) with hard timeout.

Success (M3 isolated segment): score ≥ ``--target`` (default 5) and
health > 0 within ``--max-frames``. Verified deterministic 3/3 from
``Level1.state`` at target=5 (~814 frames, HP 59).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python tmnt_ii/scripts/run_stage1_segment.py
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tmnt_ii.paths import GAME, GAME_DIR, RECORDINGS_DIR
from tmnt_ii.policy import Stage1Policy
from tmnt_ii.ram import ADDR_HEALTH, ADDR_LIVES, ADDR_SCORE, parse_game_state
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

DEFAULT_STATE = "Level1"
DEFAULT_TARGET = 5


def run_stage1_segment(
    *,
    state_name: str = DEFAULT_STATE,
    target_score: int = DEFAULT_TARGET,
    max_frames: int = 5000,
    out_dir: Path | None = None,
    save_clear: bool = True,
) -> dict[str, Any]:
    """Load checkpoint, run Stage1Policy until score target or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "stage1_segment")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    policy = Stage1Policy(target_score=target_score)
    reasons: dict[str, int] = {}
    screenshots: list[str] = []
    saved: list[str] = []
    png = save_rgb_png(obs, out / "s1_0000_start.png")
    screenshots.append(png.name)

    start = parse_game_state(env.get_ram(), frame=0)
    outcome = "timeout"
    final_score = 0
    final_health = start.health
    end_frame = 0

    for frame in range(1, max_frames + 1):
        ram = env.get_ram()
        score = int(ram[ADDR_SCORE])
        health = int(ram[ADDR_HEALTH])
        lives = int(ram[ADDR_LIVES])
        final_score = score
        final_health = health
        end_frame = frame

        if health == 0 or lives <= 0:
            outcome = "death"
            png = save_rgb_png(obs, out / f"s1_{frame:04d}_death.png")
            screenshots.append(png.name)
            break

        if score >= target_score:
            outcome = "success"
            png = save_rgb_png(obs, out / f"s1_{frame:04d}_clear.png")
            screenshots.append(png.name)
            if save_clear:
                path = save_state(env, GAME_DIR, GAME, "Stage1_Clear")
                saved.append(path.name)
                tagged = (
                    f"Stage1_Clear_sc{score}_hp{health}"
                )
                path2 = save_state(env, GAME_DIR, GAME, tagged)
                saved.append(path2.name)
            break

        tick = policy.tick(frame=frame, score=score, health=health)
        reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
        obs, *_ = env.step(tick.action)

        if frame % 250 == 0:
            png = save_rgb_png(obs, out / f"s1_{frame:04d}.png")
            screenshots.append(png.name)
            print(
                f"f={frame} score={score} hp={health} lives={lives} "
                f"sx={parse_game_state(ram, frame).player_x}"
            )

    env.close()
    report: dict[str, Any] = {
        "success": outcome == "success",
        "outcome": outcome,
        "start_state": state_name,
        "target_score": target_score,
        "final_score": final_score,
        "final_health": final_health,
        "frames": end_frame,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "start": {
            "health": start.health,
            "lives": start.lives,
            "score": int(start.extras.get("score", 0)),
        },
        "notes": (
            "M3 segment: first Stage 1 wave = score≥target kills from "
            "Level1. Policy: open RIGHT/A/B → face-LEFT lock clear → push."
        ),
    }
    write_json_report(out / "stage1_segment.json", report)
    print(
        f"outcome={outcome} score={final_score} hp={final_health} "
        f"frames={end_frame} saved={saved}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    report = run_stage1_segment(
        state_name=args.state,
        target_score=args.target,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
    )
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
