#!/usr/bin/env python3
"""
Model Registry - track all trained models with metadata and benchmark scores.

Usage:
    python model_registry.py list                          # List all models
    python model_registry.py show <model_name>             # Show model details
    python model_registry.py register <model_path> [opts]  # Register a new model
    python model_registry.py benchmark <model_path>        # Run + record benchmarks
    python model_registry.py best                          # Show best model per level

Programmatic:
    from model_registry import Registry
    reg = Registry()
    reg.register("mk1_match2_ppo_final.zip", parent="mk1_multichar_ppo_2000000_steps.zip",
                 script="train_match2.py", steps=500000, notes="60/40 mix")
    reg.record_benchmark("mk1_match2_ppo_final.zip", level=1, results={...})
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
REGISTRY_FILE = SCRIPT_DIR / "models" / "registry.json"


class Registry:
    def __init__(self, path=REGISTRY_FILE):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"models": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def register(self, model_name, parent=None, script=None, steps=None,
                 lr=None, notes=None, training_mix=None):
        """Register a model with metadata."""
        model_path = SCRIPT_DIR / "models" / model_name
        entry = {
            "registered": datetime.now().isoformat(timespec="seconds"),
            "parent": parent,
            "script": script,
            "steps": steps,
            "lr": lr,
            "training_mix": training_mix,
            "notes": notes,
            "benchmarks": {},
        }
        if model_path.exists():
            entry["size_mb"] = round(model_path.stat().st_size / 1048576, 1)
            entry["created"] = datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat(timespec="seconds")

        self.data["models"][model_name] = entry
        self._save()
        return entry

    def record_benchmark(self, model_name, level, results):
        """Record benchmark results for a model at a match level.

        results: dict of {character: {"wins": N, "losses": N, "matches": N}}
        """
        if model_name not in self.data["models"]:
            self.register(model_name, notes="auto-registered from benchmark")

        entry = self.data["models"][model_name]
        total_w = sum(r["wins"] for r in results.values())
        total_l = sum(r["losses"] for r in results.values())
        total_m = total_w + total_l
        win_rate = round(total_w / total_m * 100, 1) if total_m > 0 else 0

        entry["benchmarks"][f"match{level}"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "overall_win_rate": win_rate,
            "total_wins": total_w,
            "total_matches": total_m,
            "per_character": results,
        }
        self._save()
        return win_rate

    def get(self, model_name):
        return self.data["models"].get(model_name)

    def list_models(self):
        return self.data["models"]

    def best_for_level(self, level):
        """Find model with highest win rate at a given match level."""
        key = f"match{level}"
        best_name, best_rate = None, -1
        for name, entry in self.data["models"].items():
            bm = entry.get("benchmarks", {}).get(key)
            if bm and bm["overall_win_rate"] > best_rate:
                best_rate = bm["overall_win_rate"]
                best_name = name
        return best_name, best_rate


def run_benchmark(model_path, level, matches=5):
    """Run benchmark and return structured results."""
    sys.path.insert(0, str(ROOT_DIR))
    import torch
    from stable_baselines3 import PPO
    from fighters_common.game_configs import get_game_config
    from fighters_common.fighting_env import (
        DirectRAMReader, DiscreteAction, FightingGameConfig,
        FightingEnv, FrameSkip, FrameStack, GrayscaleResize,
    )
    import stable_retro as retro

    CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]
    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    state_dir = game_dir / "custom_integrations" / config.game_id
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))

    state_prefix = "Fight" if level == 1 else f"Match{level}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(model_path), device=device)

    results = {}
    for char in CHARACTERS:
        state_name = f"{state_prefix}_{char}"
        state_path = state_dir / f"{state_name}.state"
        if not state_path.exists():
            continue

        base_env = retro.make(
            game=config.game_id, state=state_name, render_mode=None,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
        )
        fight_config = FightingGameConfig(
            max_health=config.max_health, health_key=config.health_key,
            enemy_health_key=config.enemy_health_key,
            ram_overrides=config.ram_overrides, actions=config.actions,
        )
        env = base_env
        if config.ram_overrides:
            env = DirectRAMReader(env, config.ram_overrides)
        env = FrameSkip(env, n_skip=4)
        env = GrayscaleResize(env, width=84, height=84)
        env = FightingEnv(env, fight_config)
        env = DiscreteAction(env, config.actions)
        env = FrameStack(env, n_frames=4)

        wins, losses = 0, 0
        for _ in range(matches):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            if rw >= 2 and rw > rl:
                wins += 1
            else:
                losses += 1
        env.close()

        results[char] = {"wins": wins, "losses": losses, "matches": matches}
        wr = wins / matches * 100
        print(f"  {char:<12} {wins}W/{losses}L ({wr:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="MK1 Model Registry")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all registered models")

    p_show = sub.add_parser("show", help="Show model details")
    p_show.add_argument("model", help="Model filename")

    p_reg = sub.add_parser("register", help="Register a model")
    p_reg.add_argument("model", help="Model filename")
    p_reg.add_argument("--parent", help="Parent model")
    p_reg.add_argument("--script", help="Training script used")
    p_reg.add_argument("--steps", type=int, help="Training steps")
    p_reg.add_argument("--lr", type=float, help="Learning rate")
    p_reg.add_argument("--notes", help="Notes")
    p_reg.add_argument("--mix", help="Training state mix description")

    p_bm = sub.add_parser("benchmark", help="Run + record benchmark")
    p_bm.add_argument("model", help="Model filename or path")
    p_bm.add_argument("--level", type=int, default=1, help="Match level")
    p_bm.add_argument("--matches", type=int, default=5, help="Matches per char")

    sub.add_parser("best", help="Show best model per level")

    args = parser.parse_args()
    reg = Registry()

    if args.command == "list":
        models = reg.list_models()
        if not models:
            print("No models registered.")
            return
        print(f"\n{'Model':<45} {'Steps':>8} {'M1':>5} {'M2':>5} {'Parent'}")
        print("-" * 100)
        for name, entry in models.items():
            steps = entry.get("steps") or ""
            m1 = entry.get("benchmarks", {}).get("match1", {}).get("overall_win_rate", "")
            m2 = entry.get("benchmarks", {}).get("match2", {}).get("overall_win_rate", "")
            parent = entry.get("parent") or ""
            m1_str = f"{m1}%" if m1 != "" else "-"
            m2_str = f"{m2}%" if m2 != "" else "-"
            print(f"{name:<45} {str(steps):>8} {m1_str:>5} {m2_str:>5} {parent}")

    elif args.command == "show":
        entry = reg.get(args.model)
        if not entry:
            print(f"Model not found: {args.model}")
            return
        print(json.dumps({args.model: entry}, indent=2))

    elif args.command == "register":
        entry = reg.register(
            args.model, parent=args.parent, script=args.script,
            steps=args.steps, lr=args.lr, notes=args.notes,
            training_mix=args.mix,
        )
        print(f"Registered: {args.model}")

    elif args.command == "benchmark":
        model_path = Path(args.model)
        if not model_path.exists():
            model_path = SCRIPT_DIR / "models" / args.model
        if not model_path.exists():
            print(f"Model not found: {args.model}")
            return

        print(f"\nBenchmarking {model_path.name} at Match {args.level}")
        print("=" * 50)
        results = run_benchmark(model_path, args.level, args.matches)
        wr = reg.record_benchmark(model_path.name, args.level, results)
        total_w = sum(r["wins"] for r in results.values())
        total_m = sum(r["wins"] + r["losses"] for r in results.values())
        print("=" * 50)
        print(f"Match {args.level}: {total_w}/{total_m} ({wr}%)")
        print(f"Saved to registry.")

    elif args.command == "best":
        print("\nBest models per level:")
        print("=" * 50)
        for level in range(1, 8):
            name, rate = reg.best_for_level(level)
            if name:
                print(f"  Match {level}: {name} ({rate}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
