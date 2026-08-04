"""Glass Joe bout segment from Match1 / Level1.

Milestones (M3 ladder toward bout win):

- ``knockdown`` (default): score ≥1 opponent knockdown with Mac still standing
  or successfully got up. Deterministic from Match1 with taunt counter.
- ``bout``: attempt full win (3 opp KDs / KO / decision). Still experimental.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python punch_out/scripts/run_glass_joe.py
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

from punch_out.paths import GAME, GAME_DIR, RECORDINGS_DIR
from punch_out.policy import GlassJoePolicy
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_HEALTH,
    ADDR_ROUND,
    FIGHT_BETWEEN,
    FIGHT_IN_RING,
    hearts,
    is_match_live,
    parse_game_state,
)
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

DEFAULT_STATE = "Match1"
DEFAULT_GOAL = "knockdown"


def _ensure_match_live(env, obs, max_wait: int = 2000):
    """If loaded pre-clock, idle until the bout clock is live.

    Returns ``(obs, frames_waited)``.
    """
    for waited in range(max_wait):
        ram = env.get_ram()
        if is_match_live(ram):
            return obs, waited
        step = env.step(nes_idle_action())
        obs = step[0] if isinstance(step, tuple) else step
    return obs, max_wait


def run_glass_joe(
    *,
    state_name: str = DEFAULT_STATE,
    goal: str = DEFAULT_GOAL,
    max_frames: int = 20000,
    out_dir: Path | None = None,
    save_clear: bool = True,
) -> dict[str, Any]:
    """Load checkpoint, run GlassJoePolicy until goal or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    # Prefer Match1; fall back to Level1
    if state_name not in available:
        if "Level1" in available:
            state_name = "Level1"
        else:
            raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "glass_joe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        obs, waited = _ensure_match_live(env, obs)

        policy = GlassJoePolicy()
        screenshots: list[str] = []
        saved: list[str] = []
        png = save_rgb_png(obs, out / "gj_0000_start.png")
        screenshots.append(png.name)

        outcome = "timeout"
        end_frame = 0
        final_mac = 96
        final_opp = 96

        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            fa = policy.tick(ram)
            step = env.step(fa.action)
            obs = step[0] if isinstance(step, tuple) else step
            end_frame = frame
            ram = env.get_ram()
            final_mac = int(ram[ADDR_HEALTH])
            final_opp = int(ram[ADDR_OPP_HEALTH])
            fight = int(ram[ADDR_FIGHT_FLAG])

            # Goal: first knockdown
            if goal == "knockdown" and policy.opp_kd >= 1:
                outcome = "knockdown"
                png = save_rgb_png(obs, out / f"gj_{frame:04d}_kd1.png")
                screenshots.append(png.name)
                if save_clear:
                    path = save_state(env, GAME_DIR, GAME, "GlassJoe_KD1")
                    saved.append(path.name)
                break

            if policy.mac_kd >= 3:
                outcome = "loss_tko"
                png = save_rgb_png(obs, out / f"gj_{frame:04d}_loss.png")
                screenshots.append(png.name)
                break

            if goal == "bout":
                if policy.opp_kd >= 3 and fight == FIGHT_BETWEEN:
                    outcome = "tko_win"
                    png = save_rgb_png(obs, out / f"gj_{frame:04d}_tko.png")
                    screenshots.append(png.name)
                    if save_clear:
                        path = save_state(env, GAME_DIR, GAME, "GlassJoe_Clear")
                        saved.append(path.name)
                    break
                if (
                    policy.opp_kd >= 1
                    and final_opp == 0
                    and policy.mode.name == "WATCH_KD"
                    and policy.mode_t > 650
                ):
                    outcome = "ko_win"
                    png = save_rgb_png(obs, out / f"gj_{frame:04d}_ko.png")
                    screenshots.append(png.name)
                    if save_clear:
                        path = save_state(env, GAME_DIR, GAME, "GlassJoe_Clear")
                        saved.append(path.name)
                    break
        else:
            png = save_rgb_png(obs, out / f"gj_{end_frame:04d}_timeout.png")
            screenshots.append(png.name)

        state = parse_game_state(env.get_ram(), frame=end_frame)
        report = {
            "game": GAME,
            "state": state_name,
            "goal": goal,
            "outcome": outcome,
            "success": outcome in ("knockdown", "tko_win", "ko_win"),
            "frames": end_frame,
            "waited_for_clock": waited,
            "opp_kd": policy.opp_kd,
            "mac_kd": policy.mac_kd,
            "hits": policy.hits,
            "mac_health": final_mac,
            "opp_health": final_opp,
            "hearts": hearts(env.get_ram()),
            "round": int(env.get_ram()[ADDR_ROUND]),
            "mode": state.mode.name,
            "policy_mode": policy.mode.name,
            "reasons": dict(
                sorted(policy.reasons.items(), key=lambda kv: -kv[1])[:20]
            ),
            "screenshots": screenshots,
            "saved_states": saved,
        }
        write_json_report(out / "report.json", report)
    finally:
        env.close()

    print(
        f"GLASS_JOE goal={goal} outcome={outcome} success={report['success']} "
        f"frames={end_frame} opp_kd={policy.opp_kd} mac_kd={policy.mac_kd} "
        f"mac={final_mac} opp={final_opp} state={state_name}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--goal",
        choices=("knockdown", "bout"),
        default=DEFAULT_GOAL,
        help="knockdown = M3 first KD; bout = full win attempt",
    )
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    report = run_glass_joe(
        state_name=args.state,
        goal=args.goal,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
    )
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
