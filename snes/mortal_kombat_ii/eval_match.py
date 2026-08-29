"""RAM-gated Fight_LiuKang eval: health KO edges, not CNN pixels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from mortal_kombat_ii.paths import FIGHT_LIUKANG, GAME_DIR, GAME_ID, MODEL_DIR
from mortal_kombat_ii.ram import (
    ADDR_P1_HEALTH,
    ADDR_P2_HEALTH,
    DECOY_NOT_HEALTH,
    MAX_HEALTH,
    is_match_lost,
    is_match_won,
    parse_ram,
)

EVAL_MAX_STEPS = 15_000
RAW_EVAL_MAX_STEPS = 60_000
DEFAULT_CNN_ZIP = MODEL_DIR / "mk2_ppo_final.zip"


def make_raw_eval_env(state: str = FIGHT_LIUKANG):
    """stable-retro env for a named save state: no DiscreteAction, no FrameSkip."""
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(str(GAME_DIR / "custom_integrations"))
    return retro.make(
        game=GAME_ID,
        state=state,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )


def make_cnn_eval_env(state: str = FIGHT_LIUKANG):
    """Pixel CNN eval env with DirectRAMReader health (same stack as train_ppo)."""
    from retro_harness.fighters.fighting_env import FightingGameConfig, make_fighting_env
    from retro_harness.fighters.game_configs import get_game_config

    config = get_game_config("mk2")
    fight = FightingGameConfig(
        max_health=config.max_health,
        health_key=config.health_key,
        enemy_health_key=config.enemy_health_key,
        ram_overrides=config.ram_overrides,
        actions=config.actions,
    )
    return make_fighting_env(
        game=config.game_id,
        state=state,
        game_dir=GAME_DIR,
        config=fight,
        monitor_dir=None,
    )


def probe_health(env, settle_frames: int = 30) -> dict[str, int]:
    """Read live health vs the 0x020A/0x020E decoys after a short settle."""
    import numpy as np

    from retro_harness.env import reset_obs

    reset_obs(env)
    noop = np.zeros(12, dtype=np.int8)
    for _ in range(settle_frames):
        env.step(noop)
    ram = env.unwrapped.get_ram()
    snap = parse_ram(ram)
    return {
        "p1_health": snap.p1_health,
        "p2_health": snap.p2_health,
        "decoy_020a": int(ram[DECOY_NOT_HEALTH[0]]) & 0xFF,
        "decoy_020e": int(ram[DECOY_NOT_HEALTH[1]]) & 0xFF,
        "addr_p1": ADDR_P1_HEALTH,
        "addr_p2": ADDR_P2_HEALTH,
        "ram_len": snap.ram_len,
        "max_health": MAX_HEALTH,
    }


def play_buttons_match(
    policy,
    env,
    *,
    max_steps: int = RAW_EVAL_MAX_STEPS,
) -> bool:
    """Score a 12-button RAM policy. True iff P1 takes the match on health KOs."""
    from retro_harness.env import reset_obs

    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    reset_obs(env)
    p1_kos = 0
    p2_kos = 0
    prev_p1_health = None
    prev_p2_health = None
    for _ in range(max_steps):
        ram = env.unwrapped.get_ram()
        snap = parse_ram(ram)
        if prev_p2_health is not None and prev_p2_health > 0 and snap.p2_health == 0:
            p1_kos += 1
        if prev_p1_health is not None and prev_p1_health > 0 and snap.p1_health == 0:
            p2_kos += 1
        prev_p1_health = snap.p1_health
        prev_p2_health = snap.p2_health
        if is_match_won(p1_kos, p2_kos):
            return True
        if is_match_lost(p1_kos, p2_kos):
            return False
        buttons = policy.act(ram, None, deterministic=True)
        env.step(buttons)
    return False


def play_match(
    model,
    env,
    *,
    deterministic: bool = False,
    max_steps: int = EVAL_MAX_STEPS,
) -> bool:
    """Score a DiscreteAction CNN/MLP policy via FightingEnv round counters."""
    obs, _info = env.reset()
    for _ in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            won = int(info.get("rounds_won", 0))
            lost = int(info.get("rounds_lost", 0))
            return is_match_won(won, lost)
    return False


def _eval_cnn(load: Path, state: str, attempts: int) -> tuple[int, int]:
    import torch
    from stable_baselines3 import PPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(load), device=device)
    wins = 0
    losses = 0
    for _ in range(attempts):
        env = make_cnn_eval_env(state)
        try:
            if play_match(model, env, deterministic=True):
                wins += 1
            else:
                losses += 1
        finally:
            env.close()
    return wins, losses


def _eval_scripted(state: str, attempts: int) -> tuple[int, int]:
    from mortal_kombat_ii.scripted import ScriptedPolicy

    wins = 0
    losses = 0
    for _ in range(attempts):
        env = make_raw_eval_env(state)
        try:
            if play_buttons_match(ScriptedPolicy(), env):
                wins += 1
            else:
                losses += 1
        finally:
            env.close()
    return wins, losses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=FIGHT_LIUKANG)
    parser.add_argument("--load", type=Path, default=None, help="PPO zip (CNN eval)")
    parser.add_argument("--scripted", action="store_true", help="RAM-gated fireball policy")
    parser.add_argument("--probe-health", action="store_true")
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args(argv)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    if args.probe_health:
        env = make_raw_eval_env(args.state)
        try:
            probe = probe_health(env)
        finally:
            env.close()
        print(
            f"HEALTH {args.state} p1={probe['p1_health']} p2={probe['p2_health']} "
            f"get_ram=0x{probe['addr_p1']:04X}/0x{probe['addr_p2']:04X} "
            f"decoy_020a={probe['decoy_020a']} decoy_020e={probe['decoy_020e']}"
        )
        ok = (
            probe["p1_health"] == MAX_HEALTH
            and probe["p2_health"] == MAX_HEALTH
            and probe["decoy_020a"] != MAX_HEALTH
        )
        return 0 if ok else 1

    if args.scripted:
        wins, losses = _eval_scripted(args.state, args.attempts)
        kind = "scripted"
    else:
        load = args.load or DEFAULT_CNN_ZIP
        wins, losses = _eval_cnn(load, args.state, args.attempts)
        kind = str(load)
    result = "WIN" if wins > losses else "LOSS"
    print(f"EVAL {kind} state={args.state} W={wins} L={losses} {result}")
    return 0 if wins > losses else 1


if __name__ == "__main__":
    sys.exit(main())
