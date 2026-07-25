"""Run the shared policy from a named save until an internal area advances."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from snes_oneshot.game_state import GameMode
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)
from super_double_dragon.paths import GAME, GAME_DIR, RECORDINGS_DIR
from super_double_dragon.policy import Stage1Policy
from super_double_dragon.ram import parse_game_state


def _parse_int(value: str) -> int:
    return int(value, 0)


def _heal_player(env: Any, player_base: int, health: int = 88) -> None:
    page = player_base >> 8
    env.set_value(f"actor{page:02x}_hp", health)
    env.set_value("player_lives", 2)


def run_area(
    *,
    state_name: str,
    target_area: int,
    save_name: str | None = None,
    max_frames: int = 30000,
    dev_heal: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one resumable area and report whether ``target_area`` loaded."""
    configure_headless()
    out = out_dir or RECORDINGS_DIR / f"{state_name}_to_{target_area:02x}"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    reasons: Counter[str] = Counter()
    heals = 0
    screenshots: list[str] = []
    saved_states: list[str] = []
    success = False
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        state = parse_game_state(env.get_ram())
        start = snapshot_state(state)
        screenshots.append(save_rgb_png(obs, out / "start.png").name)
        frame = 0
        for frame in range(1, max_frames + 1):
            if (
                dev_heal
                and state.mode is GameMode.PLAYING
                and (state.health < 32 or state.lives < 1)
            ):
                _heal_player(env, int(state.extras["player_base"]))
                state = parse_game_state(env.get_ram(), frame=frame)
                heals += 1
            tick = policy.tick(state)
            assert tick.action is not None
            reasons[tick.action.reason] += 1
            obs, _reward, _term, _trunc, _info = env.step(
                tick.action.action
            )
            state = parse_game_state(env.get_ram(), frame=frame)
            if state.stage == target_area:
                success = True
                break
            if state.stage == 0:
                break
        screenshots.append(save_rgb_png(obs, out / "end.png").name)
        if success and save_name:
            path = save_state(env, GAME_DIR, GAME, save_name)
            saved_states.append(path.name)
        report: dict[str, Any] = {
            "outcome": "success" if success else "timeout",
            "success": success,
            "frames": frame,
            "start_state": state_name,
            "target_area": target_area,
            "dev_heal": dev_heal,
            "dev_heals": heals,
            "start": start,
            "end": snapshot_state(state),
            "reason_counts": dict(reasons),
            "screenshots": screenshots,
            "saved_states": saved_states,
        }
        report_path = write_json_report(out / "area_report.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--target-area", required=True, type=_parse_int)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--max-frames", type=int, default=30000)
    parser.add_argument("--no-dev-heal", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_area(
        state_name=args.state,
        target_area=args.target_area,
        save_name=args.save_state,
        max_frames=args.max_frames,
        dev_heal=not args.no_dev_heal,
        out_dir=args.out_dir,
    )
    end = report["end"]
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"area={end['stage']:#04x} hp={end['health']} "
        f"lives={end['lives']} dev_heals={report['dev_heals']}"
    )
    print(f"report={report['report_path']}")
    if report["saved_states"]:
        print("states: " + ", ".join(report["saved_states"]))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
