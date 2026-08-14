"""Run the Star Fox Bronze completion policy with save-state recovery.

The current run begins at the verified Route 1 Corneria gameplay state.  It
uses only ordinary controller actions for progress, but may rewind to a recent
healthy checkpoint after a death.  This matches the repository's segmented
development workflow; true-reset chaining comes after the stage policies are
reliable.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

GAME_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root
_SNES_IMPORT_ROOT = GAME_DIR.parent

from retro_harness.env import make_env, save_state
from retro_harness.segment_runner import configure_headless, write_json_report

GAME = "StarFox-Snes"
DEFAULT_STATE = "CorneriaStart"

def signed_byte(value: int) -> int:
    """Interpret an unsigned byte as a signed two's-complement value."""
    return value if value < 128 else value - 256

def boss_meter_visible(obs: Any) -> bool:
    """Detect the pink ENEMY meter used during boss fights."""
    region = obs[10:36, 148:246]
    pink = (
        (region[:, :, 0] >= 224)
        & (region[:, :, 1] >= 48)
        & (region[:, :, 1] <= 128)
        & (region[:, :, 2] >= 72)
        & (region[:, :, 2] <= 144)
    )
    return int(pink.sum()) >= 40

def _press_toward(
    action: list[int],
    buttons: dict[str, int],
    *,
    value: int,
    target: int,
    low_button: str,
    high_button: str,
    deadzone: int = 5,
) -> None:
    """Press one direction until a scalar is inside a target deadzone."""
    if value < target - deadzone:
        action[buttons[high_button]] = 1
    elif value > target + deadzone:
        action[buttons[low_button]] = 1

def build_action(
    *,
    frame: int,
    variant: int,
    info: dict[str, Any],
    env_buttons: list[str],
    last_bomb_frame: int,
    boss_active: bool = False,
    boss_frame: int = 0,
) -> tuple[list[int], int, str]:
    """Build one controller action for the current recovery variant."""
    action = [0] * len(env_buttons)
    buttons = {button: index for index, button in enumerate(env_buttons)}

    # A held blaster charges; short taps give the useful continuous fire.
    if frame % 8 < 4:
        action[buttons["B"]] = 1

    x = int(info.get("player_x", 112))
    y = signed_byte(int(info.get("player_y_phase", 232)))
    variant %= 6

    if boss_active:
        # Attack Carrier begins by passing overhead.  Stay low only for that
        # pass; afterward the three launch bays must be engaged near the
        # middle of the screen before the main hull becomes vulnerable.
        if boss_frame < 210:
            target_x, target_y = 112, 8
            phase = "opening_pass"
            action[buttons["X"]] = 1
        else:
            # The carrier keeps each bay exposed for much longer than a
            # normal enemy pass.  Recovery variants therefore hold useful
            # lanes instead of racing between screen edges.
            if variant == 0:
                target_x, target_y, phase = 138, -22, "right"
            elif variant == 1:
                target_x, target_y, phase = 138, -46, "right_high"
            elif variant == 2:
                target_x, target_y, phase = 60, 4, "left_low"
            elif variant == 3:
                on_right = ((boss_frame - 210) // 900) % 2 == 0
                target_x, target_y = (138 if on_right else 72), -22
                phase = "right" if on_right else "left"
            elif variant == 4:
                target_x = 138
                target_y = -45 if (boss_frame // 240) % 2 == 0 else -5
                phase = "vertical_dodge"
            else:
                target_x, target_y, phase = 112, -22, "center"

        _press_toward(
            action,
            buttons,
            value=x,
            target=target_x,
            low_button="LEFT",
            high_button="RIGHT",
            deadzone=4,
        )
        _press_toward(
            action,
            buttons,
            value=y,
            target=target_y,
            low_button="DOWN",
            high_button="UP",
            deadzone=4,
        )
        if boss_frame >= 210 and boss_frame % 150 < 10:
            roll = "L" if (boss_frame // 150 + variant) % 2 == 0 else "R"
            action[buttons[roll]] = 1
        reason = f"boss_{phase}_{variant}"
    elif variant == 0:
        # Best opening baseline: a broad horizontal weave at neutral pitch.
        phase = frame % 600
        direction = "LEFT" if phase < 150 or phase >= 450 else "RIGHT"
        action[buttons[direction]] = 1
        reason = "horizontal_weave"
    elif variant == 1:
        # Four bounded targets prevent the open-loop policy from pinning an edge.
        phase = (frame // 360) % 4
        target_x = (82, 142, 142, 82)[phase]
        target_y = (-45, -45, 0, 0)[phase]
        _press_toward(
            action,
            buttons,
            value=x,
            target=target_x,
            low_button="LEFT",
            high_button="RIGHT",
        )
        _press_toward(
            action,
            buttons,
            value=y,
            target=target_y,
            low_button="DOWN",
            high_button="UP",
        )
        reason = "bounded_box"
    elif variant == 2:
        _press_toward(
            action,
            buttons,
            value=x,
            target=60,
            low_button="LEFT",
            high_button="RIGHT",
        )
        _press_toward(
            action,
            buttons,
            value=y,
            target=-55,
            low_button="DOWN",
            high_button="UP",
        )
        reason = "high_left"
    elif variant == 3:
        _press_toward(
            action,
            buttons,
            value=x,
            target=60,
            low_button="LEFT",
            high_button="RIGHT",
        )
        _press_toward(
            action,
            buttons,
            value=y,
            target=8,
            low_button="DOWN",
            high_button="UP",
        )
        reason = "low_left"
    elif variant == 4:
        # Brake through dense geometry while holding a narrow central corridor.
        action[buttons["X"]] = 1
        _press_toward(
            action,
            buttons,
            value=x,
            target=112,
            low_button="LEFT",
            high_button="RIGHT",
        )
        _press_toward(
            action,
            buttons,
            value=y,
            target=-24,
            low_button="DOWN",
            high_button="UP",
        )
        reason = "center_brake"
    else:
        phase = frame % 600
        direction = "LEFT" if phase < 150 or phase >= 450 else "RIGHT"
        action[buttons[direction]] = 1
        if frame % 180 < 12:
            roll = "L" if (frame // 180) % 2 == 0 else "R"
            action[buttons[roll]] = 1
        reason = "rolling_weave"

    health = int(info.get("health", 40))
    bombs = int(info.get("bombs", 0))
    boss_bomb_window = boss_active and any(
        abs(boss_frame - target) <= 2 for target in (510, 1710)
    )
    emergency_bomb = health <= 18 and frame - last_bomb_frame > 600
    if bombs > 0 and (boss_bomb_window or emergency_bomb):
        action[buttons["A"]] = 1
        last_bomb_frame = frame
        reason += "+bomb"

    return action, last_bomb_frame, reason

def build_parser() -> argparse.ArgumentParser:
    """Build the completion-run CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--max-retries", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=300)
    parser.add_argument("--checkpoint-health", type=int, default=24)
    parser.add_argument("--screenshot-every", type=int, default=6_000)
    parser.add_argument("--output", type=Path)
    return parser

def main(argv: list[str] | None = None) -> int:
    """Run until the step/retry budget is exhausted."""
    args = build_parser().parse_args(argv)
    configure_headless()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or GAME_DIR / "recordings" / f"completion_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR={output}", flush=True)

    env = make_env(
        game=GAME,
        state=args.state,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
    )
    obs, info = env.reset()
    checkpoint = env.em.get_state()
    checkpoint_frame = 0
    checkpoint_bomb_frame = -1_000
    checkpoint_boss_frame = 0
    logical_frame = 0
    boss_frame = 0
    retries = 0
    variant = 0
    last_bomb_frame = -1_000
    screenshot_paths: list[str] = []
    reason_counts: dict[str, int] = {}
    outcome = "budget_exhausted"

    try:
        for total_steps in range(1, args.max_steps + 1):
            logical_frame += 1
            boss_active = boss_meter_visible(obs)
            boss_frame = boss_frame + 1 if boss_active else 0
            action, last_bomb_frame, reason = build_action(
                frame=logical_frame,
                variant=variant,
                info=info,
                env_buttons=env.buttons,
                last_bomb_frame=last_bomb_frame,
                boss_active=boss_active,
                boss_frame=boss_frame,
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            obs, _, terminated, truncated, info = env.step(action)

            health = int(info.get("health", 40))
            lives = int(info.get("lives", 3))
            if total_steps % args.screenshot_every == 0:
                path = output / f"step_{total_steps:07d}.png"
                Image.fromarray(obs).save(path)
                screenshot_paths.append(str(path))
                print(
                    "STATUS "
                    f"steps={total_steps} route_frame={logical_frame} "
                    f"checkpoint={checkpoint_frame} retries={retries} "
                    f"variant={variant} health={health} lives={lives} "
                    f"bombs={info.get('bombs')} kills={info.get('kills')}",
                    flush=True,
                )

            healthy_checkpoint = (
                logical_frame > checkpoint_frame
                and logical_frame % args.checkpoint_every == 0
                and health >= args.checkpoint_health
                and lives > 0
            )
            if healthy_checkpoint:
                checkpoint = env.em.get_state()
                checkpoint_frame = logical_frame
                checkpoint_bomb_frame = last_bomb_frame
                checkpoint_boss_frame = boss_frame
                if checkpoint_frame % (args.checkpoint_every * 10) == 0:
                    save_state(
                        env,
                        GAME_DIR,
                        GAME,
                        "CompletionAutoCheckpoint",
                    )
                print(
                    f"CHECKPOINT frame={checkpoint_frame} health={health} "
                    f"lives={lives} kills={info.get('kills')}",
                    flush=True,
                )

            if health <= 0 or lives <= 0:
                retries += 1
                if retries > args.max_retries:
                    outcome = "retry_budget_exhausted"
                    break
                env.em.set_state(checkpoint)
                logical_frame = checkpoint_frame
                last_bomb_frame = checkpoint_bomb_frame
                boss_frame = checkpoint_boss_frame
                variant = retries % 6
                info = {
                    "health": args.checkpoint_health,
                    "lives": max(lives, 1),
                    "bombs": 3,
                    "kills": info.get("kills", 0),
                    "player_x": 112,
                    "player_y_phase": 232,
                }
                print(
                    f"RETRY count={retries} frame={checkpoint_frame} "
                    f"next_variant={variant}",
                    flush=True,
                )
                continue

            if terminated or truncated:
                outcome = "environment_stopped"
                break
        else:
            total_steps = args.max_steps
    finally:
        final_path = output / "final.png"
        Image.fromarray(obs).save(final_path)
        screenshot_paths.append(str(final_path))
        report = {
            "outcome": outcome,
            "state": args.state,
            "total_steps": total_steps,
            "route_frame": logical_frame,
            "boss_frame": boss_frame,
            "checkpoint_frame": checkpoint_frame,
            "retries": retries,
            "variant": variant,
            "health": int(info.get("health", 0)),
            "lives": int(info.get("lives", 0)),
            "bombs": int(info.get("bombs", 0)),
            "kills": int(info.get("kills", 0)),
            "reason_counts": dict(sorted(reason_counts.items())),
            "screenshots": screenshot_paths,
        }
        write_json_report(output / "report.json", report)
        env.close()
        print(f"REPORT={output / 'report.json'}", flush=True)

    return 0 if outcome == "environment_stopped" else 1

if __name__ == "__main__":
    raise SystemExit(main())
