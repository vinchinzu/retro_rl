"""Train one Liu Kang RAM+hitbox (v3) specialist.

Fresh MLP — pixel CNNs and v1/v2 RAM checkpoints are the wrong observation
shape and are not loaded. Old zip files can still sit in models/ as pixel
fallbacks for the tournament runner.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from retro_harness.fighters.fighting_env import FightingGameConfig
from retro_harness.fighters.game_configs import get_game_config
from retro_harness.fighters.train_ppo import EntropySchedule, FightMetricsCallback
from mortal_kombat.paths import GAME_DIR, MODEL_DIR
from mortal_kombat.ram_obs import make_mk_ram_env
from mortal_kombat.roster import v3_filename


class V3TrainConfig:
    LEARNING_RATE = 3e-4
    ENT_COEF_START = 0.01
    ENT_COEF_END = 0.0002
    BATCH_SIZE = 256
    N_STEPS = 2048
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    FRAME_SKIP = 4
    CHECKPOINT_FREQ = 250_000
    NET_ARCH = dict(pi=[256, 128], vf=[256, 128])


def _env_fn(game_id: str, state: str, config: FightingGameConfig, monitor_dir: str):
    def _init():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        return make_mk_ram_env(
            game=game_id,
            state=state,
            game_dir=GAME_DIR,
            config=config,
            frame_skip=V3TrainConfig.FRAME_SKIP,
            monitor_dir=monitor_dir,
        )

    return _init


def train_stage(
    *,
    state: str,
    stage_prefix: str,
    steps: int,
    n_envs: int,
    load: str | None = None,
) -> Path:
    """Train one specialist. Returns the final zip path."""
    prefix = f"mk1_v3_{stage_prefix}_ppo"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    device = torch.device("cpu")
    game = get_game_config("mk1")
    env_config = FightingGameConfig(
        max_health=game.max_health,
        health_key=game.health_key,
        enemy_health_key=game.enemy_health_key,
        timer_key=game.timer_key,
        round_length_frames=game.round_length_frames,
        ram_overrides=game.ram_overrides,
        actions=game.actions,
    )
    monitor_dir = str(GAME_DIR / "monitor")
    MODEL_DIR.mkdir(exist_ok=True)
    fns = [_env_fn(game.game_id, state, env_config, monitor_dir) for _ in range(n_envs)]
    env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)

    if load and Path(load).exists():
        model = PPO.load(
            load,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": V3TrainConfig.LEARNING_RATE,
                "clip_range": V3TrainConfig.CLIP_RANGE,
            },
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=dict(net_arch=V3TrainConfig.NET_ARCH),
            learning_rate=V3TrainConfig.LEARNING_RATE,
            ent_coef=V3TrainConfig.ENT_COEF_START,
            n_steps=V3TrainConfig.N_STEPS,
            batch_size=V3TrainConfig.BATCH_SIZE,
            n_epochs=V3TrainConfig.N_EPOCHS,
            clip_range=V3TrainConfig.CLIP_RANGE,
            gae_lambda=V3TrainConfig.GAE_LAMBDA,
            gamma=V3TrainConfig.GAMMA,
            tensorboard_log=None,
        )

    callbacks = [
        EntropySchedule(
            V3TrainConfig.ENT_COEF_START,
            V3TrainConfig.ENT_COEF_END,
            steps,
            verbose=1,
        ),
        FightMetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(V3TrainConfig.CHECKPOINT_FREQ // max(n_envs, 1), 1),
            save_path=str(MODEL_DIR),
            name_prefix=prefix,
        ),
    ]
    print(f"v3 train state={state} prefix={prefix} steps={steps} n_envs={n_envs}")
    model.learn(total_timesteps=steps, callback=callbacks)
    final_path = MODEL_DIR / v3_filename(stage_prefix)
    model.save(str(final_path))
    env.close()
    print(f"saved {final_path}")
    return final_path
