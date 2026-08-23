"""Train one Liu Kang RAM+hitbox (v3) specialist.

Fresh MLP — pixel CNNs and v1/v2 RAM checkpoints are the wrong observation
shape and are not loaded. Old zip files can still sit in models/ as pixel
fallbacks for the tournament runner.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from retro_harness.fighters.fighting_env import FightingGameConfig
from retro_harness.fighters.game_configs import get_game_config
from retro_harness.fighters.train_ppo import EntropySchedule, FightMetricsCallback
from mortal_kombat.paths import GAME_DIR, MODEL_DIR
from mortal_kombat.ram_obs import make_mk_ram_env
from mortal_kombat.v3_run import TrainResult, V3Run, v3_artifact_name


class V3TrainConfig:
    BATCH_SIZE = 256
    N_STEPS = 2048
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    FRAME_SKIP = 4
    CHECKPOINT_FREQ = 250_000
    NET_ARCH = dict(pi=[256, 128], vf=[256, 128])


class WallClockStop(BaseCallback):
    """Stop learn() on wall-clock; does not write ``*_ppo_final.zip``."""

    def __init__(self, max_seconds: float, verbose: int = 1):
        super().__init__(verbose)
        self.max_seconds = float(max_seconds)
        self.stopped = False
        self._t0 = 0.0

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        if time.monotonic() - self._t0 < self.max_seconds:
            return True
        self.stopped = True
        print(
            f"wall cutoff {self.max_seconds:.0f}s at timesteps={self.num_timesteps}",
            flush=True,
        )
        return False


def _env_fn(
    game_id: str,
    state: str,
    config: FightingGameConfig,
    monitor_dir: str,
    randomize_state: bool,
):
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
            randomize_state=randomize_state,
        )

    return _init


def train_stage(run: V3Run) -> TrainResult:
    """Train one specialist. Returns the saved zip and wall-stop flag."""
    prefix = f"mk1_v3_{run.output_stage}_ppo"
    lr = run.learning_rate
    ent_start = run.ent_coef_start
    ent_end = run.ent_coef_end
    if run.load and not Path(run.load).is_file():
        raise FileNotFoundError(f"v3 checkpoint does not exist: {run.load}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
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
    fns = [
        _env_fn(game.game_id, run.state, env_config, monitor_dir, run.randomize_state)
        for _ in range(run.n_envs)
    ]
    env = SubprocVecEnv(fns) if run.n_envs > 1 else DummyVecEnv(fns)

    if run.load:
        model = PPO.load(
            run.load,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": lr,
                "clip_range": V3TrainConfig.CLIP_RANGE,
            },
        )
        model.ent_coef = ent_start
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=dict(net_arch=V3TrainConfig.NET_ARCH),
            learning_rate=lr,
            ent_coef=ent_start,
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
            ent_start,
            ent_end,
            run.steps,
            verbose=1,
        ),
        FightMetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(V3TrainConfig.CHECKPOINT_FREQ // max(run.n_envs, 1), 1),
            save_path=str(MODEL_DIR),
            name_prefix=prefix,
        ),
    ]
    if run.max_seconds > 0:
        callbacks.append(WallClockStop(run.max_seconds, verbose=1))
    print(
        f"v3 train state={run.state} prefix={prefix} steps={run.steps} "
        f"n_envs={run.n_envs} lr={lr} entropy={ent_start}->{ent_end} "
        f"randomize_state={run.randomize_state} max_seconds={run.max_seconds or 'none'} "
        f"load={run.load or 'fresh'}"
    )
    model.learn(total_timesteps=run.steps, callback=callbacks)
    wall_stopped = any(getattr(cb, "stopped", False) for cb in callbacks)
    timesteps = int(model.num_timesteps)
    name = v3_artifact_name(
        run.output_stage, wall_stopped=wall_stopped, timesteps=timesteps
    )
    path = MODEL_DIR / name
    model.save(str(path))
    env.close()
    print(f"saved {path}")
    return TrainResult(path=path, wall_stopped=wall_stopped, timesteps=timesteps)
