"""Clear first Stage 1 wave to a score target with a hard timeout.

Success: score ≥ ``--target`` (default 5) and health > 0 within
``--max-frames`` of play.

- Isolated (M3): load ``Level1.state``.
- Natural-entry (M4): ``--from-boot`` power-on + menus + leftover walk
  (no Level1 load). Skip ``--target 8`` until M4 is green.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python tmnt_ii/scripts/run_stage1_segment.py --from-boot --trials 3
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tmnt_ii.menus import BOOT_LEFTOVER_WALK_FRAMES, boot_to_leftover
from tmnt_ii.paths import GAME, GAME_DIR, RECORDINGS_DIR
from tmnt_ii.policy import Stage1Policy
from tmnt_ii.ram import ADDR_HEALTH, ADDR_LIVES, ADDR_SCORE, parse_game_state
from retro_harness.env import get_available_states, make_env, reset_obs, save_state
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
    from_boot: bool = False,
    walk_frames: int = BOOT_LEFTOVER_WALK_FRAMES,
    trial: int = 0,
) -> dict[str, Any]:
    """Run Stage1Policy from Level1 or power-on leftover until score target."""
    configure_headless()
    if from_boot:
        start_label = "NONE"
        out = out_dir or (RECORDINGS_DIR / "stage1_natural")
    else:
        start_label = state_name
        available = get_available_states(GAME, GAME_DIR)
        if state_name not in available:
            raise SystemExit(f"missing state {state_name}; have {available[:12]}")
        out = out_dir or (RECORDINGS_DIR / "stage1_segment")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, start_label, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    boot_frames = 0
    if from_boot:
        obs, boot_frames, ready = boot_to_leftover(env, walk_frames=walk_frames)
        if not ready:
            env.close()
            report: dict[str, Any] = {
                "success": False,
                "outcome": "boot_fail",
                "start_state": "NONE",
                "natural_entry": True,
                "target_score": target_score,
                "final_score": 0,
                "final_health": 0,
                "frames": 0,
                "boot_frames": boot_frames,
                "trial": trial,
            }
            write_json_report(out / f"stage1_segment_t{trial}.json", report)
            print(f"outcome=boot_fail boot_frames={boot_frames}")
            return report

    policy = Stage1Policy(target_score=target_score)
    reasons: dict[str, int] = {}
    screenshots: list[str] = []
    saved: list[str] = []
    tag = f"t{trial}_"
    png = save_rgb_png(obs, out / f"s1_{tag}0000_start.png")
    screenshots.append(png.name)

    start = parse_game_state(env.get_ram(), frame=boot_frames)
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
            png = save_rgb_png(obs, out / f"s1_{tag}{frame:04d}_death.png")
            screenshots.append(png.name)
            break

        if score >= target_score:
            outcome = "success"
            png = save_rgb_png(obs, out / f"s1_{tag}{frame:04d}_clear.png")
            screenshots.append(png.name)
            if save_clear:
                clear_name = "Stage1_Natural" if from_boot else "Stage1_Clear"
                path = save_state(env, GAME_DIR, GAME, clear_name)
                saved.append(path.name)
                tagged = f"{clear_name}_sc{score}_hp{health}"
                path2 = save_state(env, GAME_DIR, GAME, tagged)
                saved.append(path2.name)
            break

        tick = policy.tick(frame=frame, score=score, health=health)
        reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
        obs, *_ = env.step(tick.action)

        if frame % 250 == 0:
            png = save_rgb_png(obs, out / f"s1_{tag}{frame:04d}.png")
            screenshots.append(png.name)
            print(
                f"f={frame} score={score} hp={health} lives={lives} "
                f"sx={parse_game_state(ram, frame).player_x}"
            )

    env.close()
    report = {
        "success": outcome == "success",
        "outcome": outcome,
        "start_state": "NONE" if from_boot else state_name,
        "natural_entry": from_boot,
        "target_score": target_score,
        "final_score": final_score,
        "final_health": final_health,
        "frames": end_frame,
        "boot_frames": boot_frames,
        "trial": trial,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "start": {
            "health": start.health,
            "lives": start.lives,
            "score": int(start.extras.get("score", 0)),
        },
        "notes": (
            "first Stage 1 wave = score≥target kills. "
            + (
                "M4 natural-entry: power-on leftover, no Level1 load."
                if from_boot
                else "M3 isolated: Level1 load."
            )
            + " Policy: open RIGHT/A/B → face-LEFT lock clear → push."
        ),
    }
    report_name = (
        "stage1_segment.json" if trial == 0 else f"stage1_segment_t{trial}.json"
    )
    write_json_report(out / report_name, report)
    print(
        f"outcome={outcome} score={final_score} hp={final_health} "
        f"frames={end_frame} boot={boot_frames} saved={saved}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--from-boot",
        action="store_true",
        help="Power-on leftover (no Level1 load)",
    )
    parser.add_argument(
        "--walk-frames",
        type=int,
        default=BOOT_LEFTOVER_WALK_FRAMES,
        help="RIGHT frames after menus when --from-boot",
    )
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    reports = [
        run_stage1_segment(
            state_name=args.state,
            target_score=args.target,
            max_frames=args.max_frames,
            out_dir=args.out_dir,
            save_clear=not args.no_save,
            from_boot=args.from_boot,
            walk_frames=args.walk_frames,
            trial=i,
        )
        for i in range(args.trials)
    ]
    if args.trials > 1:
        out = args.out_dir or (
            RECORDINGS_DIR / ("stage1_natural" if args.from_boot else "stage1_segment")
        )
        summary = {
            "success": all(r["success"] for r in reports),
            "natural_entry": args.from_boot,
            "start_state": "NONE" if args.from_boot else args.state,
            "target_score": args.target,
            "trials": args.trials,
            "successes": sum(1 for r in reports if r["success"]),
            "reports": reports,
        }
        write_json_report(out / "stage1_segment.json", summary)
        print(f"{summary['successes']}/{args.trials} start={summary['start_state']}")
    raise SystemExit(0 if all(r["success"] for r in reports) else 1)


if __name__ == "__main__":
    main()
