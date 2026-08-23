#!/usr/bin/env python3
"""Trace one v3 episode to audit state identity, rewards, and termination."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _text = str(_p)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from mortal_kombat.eval_match import make_eval_env  # noqa: E402
from mortal_kombat.paths import MODEL_DIR  # noqa: E402
from mortal_kombat.ram import char_name, parse_ram  # noqa: E402
from mortal_kombat.roster import KIND_RAM_V3, v3_filename  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=15_000)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from stable_baselines3 import PPO

    model_name = args.model or v3_filename(args.stage)
    model = PPO.load(str(MODEL_DIR / model_name), device="cpu")
    env = make_eval_env(KIND_RAM_V3, f"{args.stage}_LiuKang")
    try:
        obs, _info = env.reset()
        first = parse_ram(env.unwrapped.get_ram())
        print(
            f"initial screen={first.screen.name} match={first.match_counter} "
            f"p1={char_name(first.p1_character)} p2={char_name(first.p2_character)} "
            f"hp={first.p1_health}/{first.p2_health} timer={first.timer} "
            f"rounds={first.p1_rounds}-{first.p2_rounds}"
        )
        actions: Counter[int] = Counter()
        total_reward = 0.0
        min_health = [first.p1_health, first.p2_health]
        last_info: dict = {}
        terminated = truncated = False
        steps = 0
        for steps in range(1, args.max_steps + 1):
            action, _state = model.predict(obs, deterministic=args.deterministic)
            action_id = int(action.item()) if hasattr(action, "item") else int(action)
            actions[action_id] += 1
            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            snap = parse_ram(env.unwrapped.get_ram())
            min_health[0] = min(min_health[0], snap.p1_health)
            min_health[1] = min(min_health[1], snap.p2_health)
            if terminated or truncated:
                break
        final = parse_ram(env.unwrapped.get_ram())
        print(
            f"final steps={steps} terminated={terminated} truncated={truncated} "
            f"screen={final.screen.name} hp={final.p1_health}/{final.p2_health} "
            f"min_hp={min_health[0]}/{min_health[1]} timer={final.timer} "
            f"rounds={last_info.get('rounds_won', 0)}-{last_info.get('rounds_lost', 0)} "
            f"damage={last_info.get('episode_damage_dealt', 0)}/"
            f"{last_info.get('episode_damage_taken', 0)} "
            f"timeouts={last_info.get('timeout_rounds', 0)} reward={total_reward:.3f}"
        )
        print("actions=" + ",".join(f"{key}:{value}" for key, value in actions.most_common()))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
