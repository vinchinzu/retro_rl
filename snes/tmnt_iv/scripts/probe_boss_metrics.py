"""Quick headless boss-fight metrics from a save state.

Supports heal modes (default: emergency) to match the production low-assist
run and the slash_pattern_lab trial runner:

  - ``none``: no HP writes (pure survival stress)
  - ``emergency``: restore to 80 when HP <= 16 (production-like)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env  # noqa: E402
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402

_EMERGENCY_HP_THRESHOLD = 16
_EMERGENCY_HP_RESTORE = 80


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def run_probe(
    *,
    state_name: str,
    max_frames: int = 12000,
    stop_stage_gt: int | None = None,
    heal_mode: str = "emergency",
) -> dict[str, Any]:
    """Fight from ``state_name`` until timeout, KO, or stage advance.

    heal_mode:
      - ``none``: no HP writes (pure survival stress)
      - ``emergency``: restore to 80 when HP <= 16 (production-like)
    """
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    _reset(env)
    start = parse_game_state(env.get_ram(), frame=0)
    prev_hp = start.health if 0 < start.health <= 0x60 else None
    prev_lives = start.lives
    damage = 0
    max_hit = 0
    min_hp = prev_hp
    heals = 0
    reasons: dict[str, int] = {}
    boss_hp_start = int(start.extras.get("boss_hp", 0))
    final = start
    outcome = "timeout"
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state
            if 0 < state.health <= 0x60:
                if prev_hp is not None and state.health < prev_hp:
                    hit = prev_hp - state.health
                    damage += hit
                    max_hit = max(max_hit, hit)
                prev_hp = state.health
                if min_hp is None or state.health < min_hp:
                    min_hp = state.health

            # Emergency heal assist (production-like).
            if heal_mode == "emergency":
                if state.health == 0 or (
                    0 < state.health <= _EMERGENCY_HP_THRESHOLD
                ):
                    env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    prev_hp = state.health

            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives
            if stop_stage_gt is not None and state.stage > stop_stage_gt:
                outcome = "stage_advance"
                break
            if (
                start.boss_active
                and not state.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19}
            ):
                outcome = "boss_down"
                # keep a few frames for fade
            if outcome == "boss_down" and frame > 0 and frame % 60 == 0:
                if state.stage > start.stage or int(
                    state.extras.get("event", 0)
                ) in {0x19, 0x04}:
                    outcome = "cleared"
                    break
            tick = policy.tick(state)
            action = (
                tick.action.action
                if tick.action is not None
                else idle_action()
            )
            reason = (
                tick.action.reason
                if tick.action is not None
                else tick.reason or "idle"
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            if action[8]:
                outcome = "forbidden_a"
                break
            env.step(action)
        else:
            outcome = "timeout"
    finally:
        env.close()

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:12]
    return {
        "state": state_name,
        "outcome": outcome,
        "frames": final.frame,
        "start_stage": start.stage,
        "end_stage": final.stage,
        "start_hp": start.health,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "max_hit": max_hit,
        "heals": heals,
        "lives": f"{start.lives}->{final.lives}",
        "boss_hp": f"{boss_hp_start}->{int(final.extras.get('boss_hp', 0))}",
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": top,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="save state name")
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument(
        "--stop-stage-gt",
        type=int,
        default=None,
        help="stop when stage byte exceeds this value",
    )
    parser.add_argument(
        "--heal",
        choices=["none", "emergency"],
        default="emergency",
        help="HP assist mode (default: emergency, production-like)",
    )
    args = parser.parse_args(argv)
    report = run_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        stop_stage_gt=args.stop_stage_gt,
        heal_mode=args.heal,
    )
    print(report)
    return 0 if report["outcome"] not in {"life_loss", "forbidden_a"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
