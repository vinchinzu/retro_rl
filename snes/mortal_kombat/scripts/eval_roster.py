#!/usr/bin/env python3
"""Benchmark current LiuKang models per fight and write models/roster.json."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from retro_harness.fighters.fighting_env import FightingGameConfig  # noqa: E402
from retro_harness.fighters.game_configs import get_game_config  # noqa: E402
from retro_harness.fighters.ram_observation import build_eval_env  # noqa: E402
from mortal_kombat.paths import GAME_DIR, MODEL_DIR  # noqa: E402
from mortal_kombat.ram_obs import make_mk_ram_env  # noqa: E402
from mortal_kombat.roster import (  # noqa: E402
    KIND_PIXEL,
    KIND_RAM_V3,
    STAGES,
    PIXEL_FALLBACK,
    record_stage,
    resolve_model,
    v3_filename,
)


def _play(model, env) -> bool:
    obs, info = env.reset()
    for _ in range(15000):
        action, _ = model.predict(obs, deterministic=False)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            return rw >= 2 and rw > rl
    return False


def _make_env(kind: str, state: str):
    config = get_game_config("mk1")
    fight = FightingGameConfig(
        max_health=config.max_health,
        health_key=config.health_key,
        enemy_health_key=config.enemy_health_key,
        ram_overrides=config.ram_overrides,
        actions=config.actions,
    )
    if kind == KIND_RAM_V3:
        return make_mk_ram_env(
            game=config.game_id,
            state=state,
            game_dir=GAME_DIR,
            config=fight,
        )
    return build_eval_env(
        game=config.game_id,
        state=state,
        game_dir=GAME_DIR,
        config=fight,
        ram=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--stages", default="", help="Comma prefixes (default: all 12)")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import torch
    from stable_baselines3 import PPO

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()] or [
        prefix for prefix, _, _ in STAGES
    ]
    print(f"{'Stage':<22} {'Model':<40} {'Kind':<8} {'Win%':>6} {'W':>3} {'L':>3}")
    print("-" * 88)
    for prefix, display, _mid in STAGES:
        if prefix not in wanted:
            continue
        try:
            path, kind = resolve_model(prefix)
        except FileNotFoundError:
            print(f"{display:<22} {'MISSING':<40} {'':8} {'SKIP':>6}")
            continue
        device = torch.device("cpu" if kind == KIND_RAM_V3 else "cuda" if torch.cuda.is_available() else "cpu")
        model = PPO.load(str(path), device=device)
        wins = 0
        losses = 0
        state = f"{prefix}_LiuKang"
        for _ in range(args.attempts):
            env = _make_env(kind, state)
            try:
                if _play(model, env):
                    wins += 1
                else:
                    losses += 1
            finally:
                env.close()
        rate = wins / max(1, wins + losses)
        record_stage(prefix, model=path.name, kind=kind, win_rate=rate, attempts=wins + losses)
        print(f"{display:<22} {path.name:<40} {kind:<8} {rate:>5.0%} {wins:>3} {losses:>3}")
    print(f"\nWrote {MODEL_DIR / 'roster.json'}")
    print("Pixel fallbacks (until v3 exists):")
    for prefix in wanted:
        print(f"  {prefix}: v3={v3_filename(prefix)} pixel={PIXEL_FALLBACK.get(prefix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
