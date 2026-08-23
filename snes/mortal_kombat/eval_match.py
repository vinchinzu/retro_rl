"""Shared match eval helpers for roster refresh and v3 checkpoint ranking."""

from __future__ import annotations

from pathlib import Path

EVAL_MAX_STEPS = 15_000
RAW_EVAL_MAX_STEPS = 60_000  # raw frames, no skip-4
PROMOTE_MIN_ATTEMPTS = 20


def may_promote(attempts: int) -> bool:
    return attempts >= PROMOTE_MIN_ATTEMPTS


def checkpoint_steps(path: Path) -> int:
    marker = "_ppo_"
    tail = path.stem.rsplit(marker, 1)[-1]
    if tail.endswith("_steps"):
        value = tail.removesuffix("_steps")
        return int(value) if value.isdigit() else -1
    return 10**18 if tail == "final" else -1


def list_v3_checkpoints(
    model_dir: Path, stage: str, names: list[str] | None = None
) -> list[Path]:
    if names:
        candidates = [model_dir / name for name in names]
    else:
        candidates = list(model_dir.glob(f"mk1_v3_{stage}_ppo_*"))
    return sorted(candidates, key=lambda path: (checkpoint_steps(path), path.name))


def make_eval_env(kind: str, state: str):
    from retro_harness.fighters.fighting_env import FightingGameConfig
    from retro_harness.fighters.game_configs import get_game_config
    from retro_harness.fighters.ram_observation import build_eval_env
    from mortal_kombat.paths import GAME_DIR
    from mortal_kombat.ram_obs import make_mk_ram_env
    from mortal_kombat.roster import KIND_RAM_V3

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


def make_raw_eval_env(state: str):
    """stable-retro env for a named save state: no DiscreteAction, no FrameSkip."""
    from retro_harness.env import make_env
    from mortal_kombat.paths import GAME_DIR, GAME_ID

    return make_env(GAME_ID, state, GAME_DIR, render_mode="rgb_array")


def play_buttons_match(
    policy,
    env,
    *,
    max_steps: int = RAW_EVAL_MAX_STEPS,
) -> bool:
    """Score a 12-button RAM policy on a raw retro env. True iff P1 takes the match."""
    from retro_harness.env import reset_obs
    from mortal_kombat.ram import (
        Screen,
        is_match_lost,
        is_match_won,
        parse_ram,
        rounds_settled,
    )

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
        if snap.screen is Screen.CONTINUE:
            return False
        # Health zero-crossings settle immediately and avoid the delayed/noisy
        # HUD round bytes. The bytes remain a fallback for states loaded at KO.
        if prev_p2_health is not None and prev_p2_health > 0 and snap.p2_health == 0:
            p1_kos += 1
        if prev_p1_health is not None and prev_p1_health > 0 and snap.p1_health == 0:
            p2_kos += 1
        prev_p1_health = snap.p1_health
        prev_p2_health = snap.p2_health
        if p1_kos >= 2 and p1_kos > p2_kos:
            return True
        if p2_kos >= 2 and p2_kos > p1_kos:
            return False
        # p2_rounds HUD flickers 0→2 during FIGHT; only score settled KO/timeout.
        if rounds_settled(snap):
            if is_match_won(snap):
                return True
            if is_match_lost(snap):
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
    obs, _info = env.reset()
    for _ in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            won = int(info.get("rounds_won", 0))
            lost = int(info.get("rounds_lost", 0))
            return won >= 2 and won > lost
    return False
