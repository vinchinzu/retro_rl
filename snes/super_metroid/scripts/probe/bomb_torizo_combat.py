#!/usr/bin/env python3
"""Probe full-knowledge Bomb Torizo strategy + structured RL (not continuous).

Uses RAM positions + catalog hitboxes — no vision network.

```bash
# Active mid-fight save (strategy-validated)
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py strategy
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py strategy --state BossTorizo

# Natural activation from continuous bombs prefix (slow; ~power-on to Torizo)
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural

# Structured Gym env: strategy projected onto discrete actions
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py eval --episodes 1
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py eval --state natural --episodes 1

# Short PPO smoke train on feature_vector (ml extras)
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py train --timesteps 4096

# Incomplete entry states freeze on statue spritemap (expected fail):
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py strategy \\
  --state "Bomb Torizo Room [from Flyway]"
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.bomb_torizo import (
    BombTorizoStrategy,
    play_bomb_torizo_fight,
)
from super_metroid.combat.features import bomb_torizo_catalog, features_from_state
from super_metroid.combat.natural_entry import (
    DEFAULT_NATURAL_ACTIVE_STATE,
    capture_natural_bomb_torizo_activation,
)
from super_metroid.combat.probe import ProbeSession, open_state_env, write_json_report
from super_metroid.paths import GAME_DIR, MODELS_DIR, SCRATCH_STATE_DIR


def _resolve_state(name: str) -> str | Path:
    if name in ("natural", "natural-active", "natural_active"):
        return DEFAULT_NATURAL_ACTIVE_STATE
    path = Path(name)
    if path.suffix == ".state" or "/" in name or path.exists():
        return path
    return name


def cmd_strategy(args: argparse.Namespace) -> int:
    catalog = bomb_torizo_catalog()
    state_spec = _resolve_state(args.state)
    env, loaded = open_state_env(
        state_spec,
        settle=0,
        missing_hint="capture with capture-natural first",
    )
    assist = UnlimitedResourcesAssist()
    try:
        session = ProbeSession(env, assist)
        entry = features_from_state(session.state, catalog)
        evidence = play_bomb_torizo_fight(
            session,
            strategy=BombTorizoStrategy(max_fight_frames=args.max_frames),
            require_active=not args.allow_inactive,
        )
        tel = assist.telemetry
        report = {
            "command": "strategy",
            "state": loaded,
            "success": evidence.outcome == "bomb_torizo_defeated",
            "entry": entry.to_dict(),
            "fight": evidence.to_dict(),
            "assist": {
                "energy_restored": tel.energy.restored,
                "energy_writes": tel.energy.writes,
                "maximum_single_frame_damage": tel.maximum_single_frame_damage,
                "deaths": tel.deaths,
            },
            "final": {
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "enemy0_hp": session.state.enemy0_hp,
                "health": session.state.health,
                "bombs": session.state.bombs,
            },
            "method": "full_knowledge_strategy",
            "notes": (
                "Vision BC deferred until gold. Strategy uses enemy0 x/y/hp/"
                "spritemap + sm-json-data hitbox dims. Keep hash-pinned replay "
                "on continuous route until natural-entry hybrid is proven."
            ),
        }
        write_json_report(report, args.report)
        return 0 if report["success"] else 1
    finally:
        env.close()


def cmd_capture_natural(args: argparse.Namespace) -> int:
    result = capture_natural_bomb_torizo_activation(
        output=Path(args.output),
        provenance_path=Path(args.provenance) if args.provenance else None,
        max_prefix_frames=args.max_prefix_frames,
        mode=args.mode,
    )
    text = json.dumps(result.to_dict(), indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if result.success else 1


def cmd_prove_natural(args: argparse.Namespace) -> int:
    """Capture if missing, then run strategy from the natural-active state."""
    state_path = Path(args.output)
    if not state_path.exists() or args.force_recapture:
        capture = capture_natural_bomb_torizo_activation(
            output=state_path,
            max_prefix_frames=args.max_prefix_frames,
            mode=args.mode,
        )
        if not capture.success:
            print(json.dumps(capture.to_dict(), indent=2))
            return 1
        print(
            json.dumps(
                {"capture": capture.to_dict(), "next": "strategy_from_natural"},
                indent=2,
            )
        )

    args.state = str(state_path)
    args.allow_inactive = args.mode == "statue"
    return cmd_strategy(args)


def cmd_eval(args: argparse.Namespace) -> int:
    from super_metroid.combat.env import BombTorizoFeatureEnv, resolve_state_spec

    state = resolve_state_spec(args.state)
    env = BombTorizoFeatureEnv(
        state=state,
        max_episode_frames=args.max_frames,
        require_active=not args.allow_inactive,
    )
    episodes = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total_r = 0.0
            frames = 0
            won = False
            while True:
                if args.policy == "strategy":
                    action = env.strategy_action()
                elif args.policy == "random":
                    action = int(env.action_space.sample())
                else:
                    raise ValueError(args.policy)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_r += reward
                frames += 1
                if terminated or truncated:
                    won = bool(
                        step_info.get("features", {}).get("enemy_defeated")
                        or step_info.get("features", {}).get("enemy_hp") == 0
                    )
                    episodes.append(
                        {
                            "episode": ep,
                            "frames": frames,
                            "return": total_r,
                            "won": won,
                            "final_enemy_hp": step_info.get("features", {}).get(
                                "enemy_hp"
                            ),
                            "episode_damage_taken": step_info.get(
                                "episode_damage_taken"
                            ),
                            "assist": step_info.get("assist"),
                        }
                    )
                    break
    finally:
        env.close()

    wins = sum(1 for e in episodes if e["won"])
    report = {
        "command": "eval",
        "state": str(state),
        "policy": args.policy,
        "episodes": episodes,
        "wins": wins,
        "n": len(episodes),
        "success": wins == len(episodes) and len(episodes) > 0,
        "method": "feature_vector_gym",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["success"] else 1


def _eval_policy(model, make_env, episodes: int) -> list[dict]:
    env = make_env()
    episode_stats = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            total_r = 0.0
            frames = 0
            done = False
            info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_r += float(reward)
                frames += 1
                done = terminated or truncated
            feat = info.get("features", {})
            won = bool(feat.get("enemy_defeated")) or feat.get("enemy_hp") == 0
            episode_stats.append(
                {
                    "episode": ep,
                    "frames": frames,
                    "return": total_r,
                    "won": won,
                    "final_enemy_hp": feat.get("enemy_hp"),
                    "episode_damage_taken": info.get("episode_damage_taken"),
                }
            )
    finally:
        env.close()
    return episode_stats


def cmd_train(args: argparse.Namespace) -> int:
    import time
    from datetime import datetime, timezone

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from super_metroid.combat.env import BombTorizoFeatureEnv, resolve_state_spec

    state = resolve_state_spec(args.state)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tb"
    model_path = out_dir / "bomb_torizo_feature_ppo"

    def _make():
        return BombTorizoFeatureEnv(
            state=state,
            max_episode_frames=args.max_frames,
            require_active=not args.allow_inactive,
        )

    # Long runs: larger rollout buffer is more efficient than tiny smoke defaults.
    n_steps = args.n_steps
    if n_steps is None:
        long_run = args.hours > 0 or args.timesteps >= 50_000 or args.timesteps <= 0
        n_steps = 2048 if long_run else min(512, max(64, max(args.timesteps, 256) // 4))

    vec = DummyVecEnv([_make])
    # stable-retro: only one emulator per process. Never create a second eval env
    # alongside training — EvalCallback would crash on reset. Periodic eval is a
    # post-checkpoint subprocess (or final eval after vec.close()).

    load_path = Path(args.load) if args.load else None
    if load_path is not None and load_path.exists():
        model = PPO.load(str(load_path), env=vec, device="cpu")
        print(f"resumed from {load_path}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            verbose=1 if args.verbose else 0,
            n_steps=n_steps,
            batch_size=min(256, n_steps),
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cpu",
            tensorboard_log=str(tb_dir) if args.tensorboard else None,
        )

    t0 = time.time()
    deadline = t0 + args.hours * 3600.0 if args.hours and args.hours > 0 else None
    step_budget = args.timesteps
    if deadline is not None and args.timesteps <= 0:
        # Soft step ceiling; wall-clock callback is the real stop.
        step_budget = 20_000_000
    if step_budget <= 0:
        step_budget = 4_096

    class TimeLimitCallback(BaseCallback):
        def __init__(self, deadline_ts: float, started: float) -> None:
            super().__init__(verbose=0)
            self.deadline_ts = deadline_ts
            self.started = started

        def _on_step(self) -> bool:
            if time.time() >= self.deadline_ts:
                print(
                    f"wall-clock budget reached after {self.num_timesteps} timesteps "
                    f"({(time.time() - self.started) / 3600:.2f} h)",
                    flush=True,
                )
                return False
            return True

    class PeriodicCheckpointEvalCallback(BaseCallback):
        """Save a sidecar note when checkpoints land (no second emulator)."""

        def __init__(self, every: int, note_path: Path) -> None:
            super().__init__(verbose=0)
            self.every = max(every, 1)
            self.note_path = note_path
            self._last = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last >= self.every:
                self._last = self.num_timesteps
                self.note_path.write_text(
                    json.dumps(
                        {
                            "num_timesteps": self.num_timesteps,
                            "note": (
                                "checkpoint cadence marker; full eval runs only "
                                "after training closes the single retro env"
                            ),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return True

    callbacks: list = []
    if args.checkpoint_freq > 0 and (args.hours > 0 or args.timesteps >= 50_000):
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(vec.num_envs, 1), 1),
                save_path=str(ckpt_dir),
                name_prefix="bt_ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    if args.eval_freq > 0 and (args.hours > 0 or args.timesteps >= 50_000):
        # Do not instantiate a second RetroEnv — only write progress markers.
        callbacks.append(
            PeriodicCheckpointEvalCallback(
                every=max(args.eval_freq // max(vec.num_envs, 1), 1),
                note_path=out_dir / "progress.json",
            )
        )
        print(
            "note: in-process EvalCallback disabled (stable-retro single-emulator "
            "limit); final eval runs after training",
            flush=True,
        )
    if deadline is not None:
        callbacks.append(TimeLimitCallback(deadline, t0))

    print(
        json.dumps(
            {
                "starting_train": True,
                "state": str(state),
                "step_budget": step_budget,
                "hours": args.hours,
                "n_steps": n_steps,
                "checkpoint_freq": args.checkpoint_freq,
                "eval_freq": args.eval_freq,
                "output_dir": str(out_dir),
            }
        ),
        flush=True,
    )

    model.learn(
        total_timesteps=step_budget,
        callback=callbacks or None,
        progress_bar=False,
        reset_num_timesteps=load_path is None,
    )
    model.save(str(model_path))
    elapsed = time.time() - t0
    trained_steps = int(getattr(model, "num_timesteps", step_budget))

    # stable-retro allows only one emulator instance per process — close train
    # vec env before a final non-vec evaluation pass.
    vec.close()

    episode_stats = _eval_policy(model, _make, args.eval_episodes)
    wins = sum(1 for e in episode_stats if e["won"])

    report = {
        "command": "train",
        "state": str(state),
        "requested_timesteps": args.timesteps,
        "hours_budget": args.hours,
        "trained_timesteps": trained_steps,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600.0, 3),
        "steps_per_second": round(trained_steps / max(elapsed, 1e-6), 1),
        "model_path": str(model_path) + ".zip",
        "checkpoint_dir": str(ckpt_dir),
        "best_dir": str(out_dir / "best"),
        "eval": episode_stats,
        "wins": wins,
        "eval_episodes": args.eval_episodes,
        "success": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "method": "ppo_feature_vector",
        "notes": (
            "Structured-state PPO. Prefer natural-entry state for distribution "
            "match; distill winning policies into a deterministic controller "
            "before continuous hybrid promotion."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    report_path = args.report or (out_dir / "train_report.json")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {report_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    # Default command: strategy (back-compat with --state BossTorizo alone)
    p_strategy = sub.add_parser("strategy", help="Run full-knowledge strategy")
    p_strategy.add_argument("--state", default="BossTorizo")
    p_strategy.add_argument("--max-frames", type=int, default=8_000)
    p_strategy.add_argument("--allow-inactive", action="store_true")
    p_strategy.add_argument("--report", type=Path, default=None)
    p_strategy.set_defaults(func=cmd_strategy)

    p_cap = sub.add_parser(
        "capture-natural",
        help="Power-on continuous prefix; save state at Torizo activation",
    )
    p_cap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NATURAL_ACTIVE_STATE,
    )
    p_cap.add_argument(
        "--provenance",
        type=Path,
        default=SCRATCH_STATE_DIR / "natural_bomb_torizo_active.provenance.json",
    )
    p_cap.add_argument("--max-prefix-frames", type=int, default=60_000)
    p_cap.add_argument(
        "--mode",
        choices=("active", "statue"),
        default="active",
        help="active=combat AI at full HP; statue=idle chozo before touch",
    )
    p_cap.add_argument("--report", type=Path, default=None)
    p_cap.set_defaults(func=cmd_capture_natural)

    p_prove = sub.add_parser(
        "prove-natural",
        help="Capture natural activation if needed, then run strategy",
    )
    p_prove.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NATURAL_ACTIVE_STATE,
    )
    p_prove.add_argument("--max-prefix-frames", type=int, default=60_000)
    p_prove.add_argument("--max-frames", type=int, default=8_000)
    p_prove.add_argument(
        "--mode",
        choices=("active", "statue"),
        default="active",
    )
    p_prove.add_argument("--force-recapture", action="store_true")
    p_prove.add_argument("--report", type=Path, default=None)
    p_prove.set_defaults(func=cmd_prove_natural)

    p_eval = sub.add_parser("eval", help="Eval Gym env (strategy or random)")
    p_eval.add_argument("--state", default="BossTorizo")
    p_eval.add_argument(
        "--policy",
        choices=("strategy", "random"),
        default="strategy",
    )
    p_eval.add_argument("--episodes", type=int, default=1)
    p_eval.add_argument("--max-frames", type=int, default=4_000)
    p_eval.add_argument("--allow-inactive", action="store_true")
    p_eval.add_argument("--report", type=Path, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_train = sub.add_parser("train", help="PPO train on feature_vector")
    p_train.add_argument(
        "--state",
        default="natural",
        help="BossTorizo | natural | path (default: natural activation capture)",
    )
    p_train.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Step budget (0 + --hours uses a large soft budget stopped by wall clock)",
    )
    p_train.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="Wall-clock training budget in hours (0 = disabled)",
    )
    p_train.add_argument("--max-frames", type=int, default=4_000)
    p_train.add_argument("--eval-episodes", type=int, default=3)
    p_train.add_argument("--n-steps", type=int, default=None)
    p_train.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N env steps (0 disables)",
    )
    p_train.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="EvalCallback frequency in env steps (0 disables)",
    )
    p_train.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Resume from a saved .zip",
    )
    p_train.add_argument("--tensorboard", action="store_true")
    p_train.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "bomb_torizo_feature",
    )
    p_train.add_argument("--allow-inactive", action="store_true")
    p_train.add_argument("--verbose", action="store_true")
    p_train.add_argument("--report", type=Path, default=None)
    p_train.set_defaults(func=cmd_train)

    # Back-compat: bare flags without subcommand → strategy
    # e.g. bomb_torizo_combat.py --state BossTorizo
    argv = list(sys.argv[1:])
    known = {
        "strategy",
        "capture-natural",
        "prove-natural",
        "eval",
        "train",
        "-h",
        "--help",
    }
    if not argv or argv[0] not in known:
        argv = ["strategy", *argv]

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
