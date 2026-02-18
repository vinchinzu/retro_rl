#!/usr/bin/env python3
"""Integration evaluation for ZebesStart -> flyway_to_torizo segment chain.

Two evaluation modes:
  1. Chained: Run all 12 segment models end-to-end from ZebesStart (default)
  2. Isolated: Test each segment individually from its own start state

Usage:
    # Chained eval (12 episodes, stochastic policy)
    python scripts/eval_torizo_integration.py --headless --episodes 12

    # Isolated per-segment eval (5 trials each)
    python scripts/eval_torizo_integration.py --headless --mode isolated --trials 5

    # Both modes
    python scripts/eval_torizo_integration.py --headless --mode both --episodes 8 --trials 5
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import stable_retro as retro
import torch
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
INTEGRATION_PATH = os.path.join(PROJECT_ROOT, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
INTEGRATION_GAME_DIR = os.path.join(INTEGRATION_PATH, "SuperMetroid-Snes")
INTEGRATION_ROM_PATH = os.path.join(INTEGRATION_GAME_DIR, "rom.sfc")
LOCAL_ROM_FALLBACK = os.path.join(PROJECT_ROOT, "roms", "rom.sfc")

from train_curriculum import (  # noqa: E402
    DISCRETE_ACTIONS,
    DiscreteAction,
    FrameStack,
    ROUTE_SEGMENTS,
    SanitizeAction,
    TRAINING_ORDER,
)

ROUTE_ORDER: List[str] = list(TRAINING_ORDER)


class SeededActionHoldRepeat(gym.Wrapper):
    """Repeat actions using a per-episode seeded RNG for reproducibility."""

    def __init__(self, env: gym.Env, min_hold: int = 2, max_hold: int = 4):
        super().__init__(env)
        self.min_hold = min_hold
        self.max_hold = max_hold
        self.rng = np.random.default_rng(0)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        repeat = int(self.rng.integers(self.min_hold, self.max_hold + 1))
        total_reward = 0.0
        obs = None
        info = {}
        terminated = False
        truncated = False
        for _ in range(repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_room_hex(info: Dict) -> str:
    return f"0x{int(info.get('room_id', 0)):04X}"


def _ensure_romfile() -> None:
    if os.path.exists(INTEGRATION_ROM_PATH):
        return
    if os.path.exists(LOCAL_ROM_FALLBACK):
        os.makedirs(INTEGRATION_GAME_DIR, exist_ok=True)
        rel_target = os.path.relpath(LOCAL_ROM_FALLBACK, INTEGRATION_GAME_DIR)
        os.symlink(rel_target, INTEGRATION_ROM_PATH)
        return
    raise FileNotFoundError(
        "ROM not found. Expected one of:\n"
        f"- {INTEGRATION_ROM_PATH}\n"
        f"- {LOCAL_ROM_FALLBACK}"
    )


def _make_env(start_state: str, min_hold: int, max_hold: int) -> gym.Env:
    env = retro.make(
        game="SuperMetroid-Snes",
        state=start_state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.ALL,
    )
    env = SanitizeAction(env)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = SeededActionHoldRepeat(env, min_hold=min_hold, max_hold=max_hold)
    env = FrameStack(env, n_frames=4)
    return env


def _load_models(device: torch.device) -> Dict[str, PPO]:
    models: Dict[str, PPO] = {}
    missing = []
    for segment_name in ROUTE_ORDER:
        model_path = os.path.join(PROJECT_ROOT, "models", f"segment_{segment_name}.zip")
        if not os.path.exists(model_path):
            missing.append(model_path)
            continue
        models[segment_name] = PPO.load(model_path, device=device)
    if missing:
        raise FileNotFoundError(
            "Missing required segment models:\n" + "\n".join(missing)
        )
    return models


def _segment_completed(segment_name: str, info: Dict, had_morph: bool) -> bool:
    segment = ROUTE_SEGMENTS[segment_name]
    items = int(info.get("collected_items", 0) or info.get("items", 0))
    if segment.direction == "collect":
        return bool(items & 0x1) and not had_morph
    return int(info.get("room_id", 0)) == int(segment.target_room_id)


# =========================================================================
# CHAINED EVALUATION (full route)
# =========================================================================
def _run_chained_episode(
    models: Dict[str, PPO],
    episode_index: int,
    seed: int,
    max_steps: int,
    min_hold: int,
    max_hold: int,
    deterministic: bool,
) -> Dict:
    _set_global_seed(seed)
    env = _make_env(start_state="ZebesStart", min_hold=min_hold, max_hold=max_hold)
    started_at = time.time()
    obs, info = env.reset(seed=seed)

    step_count = 0
    segment_ptr = 0
    terminated = False
    truncated = False
    transition_steps: Dict[str, int] = {}
    segment_active_steps: Counter = Counter()
    stuck_counter = 0
    prev_pos = None

    while step_count < max_steps and segment_ptr < len(ROUTE_ORDER):
        segment_name = ROUTE_ORDER[segment_ptr]
        model = models[segment_name]
        segment_active_steps[segment_name] += 1

        items_before = int(info.get("collected_items", 0) or info.get("items", 0))
        had_morph = bool(items_before & 0x1)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        step_count += 1

        # Stuck detection
        cur_pos = (info.get("samus_x", 0), info.get("samus_y", 0))
        if prev_pos == cur_pos:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_pos = cur_pos

        # Per-segment step budget: abort segment if stuck too long
        seg_budget = ROUTE_SEGMENTS[segment_name].max_steps
        if segment_active_steps[segment_name] > seg_budget:
            # Segment exceeded its own step budget - mark as failed
            break

        if _segment_completed(segment_name, info, had_morph):
            transition_steps[segment_name] = step_count
            segment_ptr += 1

        if terminated or truncated:
            break

    elapsed = time.time() - started_at
    env.close()

    success = segment_ptr == len(ROUTE_ORDER)
    final_items = int(info.get("collected_items", 0) or info.get("items", 0))
    failed_segment = None
    if not success and segment_ptr < len(ROUTE_ORDER):
        failed_segment = ROUTE_ORDER[segment_ptr]

    return {
        "episode": episode_index,
        "seed": seed,
        "success": success,
        "steps": step_count,
        "elapsed_sec": round(elapsed, 3),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "completed_segments": ROUTE_ORDER[:segment_ptr],
        "segments_completed_count": segment_ptr,
        "transition_steps": transition_steps,
        "segment_active_steps": dict(segment_active_steps),
        "failed_segment": failed_segment,
        "final_room_id": int(info.get("room_id", 0)),
        "final_room_hex": _to_room_hex(info),
        "final_position": {"x": int(info.get("samus_x", 0)), "y": int(info.get("samus_y", 0))},
        "morph_ball_collected": bool(final_items & 0x1),
        "final_health": int(info.get("health", 0)),
        "deterministic": deterministic,
    }


# =========================================================================
# ISOLATED SEGMENT EVALUATION
# =========================================================================
def _run_isolated_segment(
    model: PPO,
    segment_name: str,
    seed: int,
    min_hold: int,
    max_hold: int,
    deterministic: bool,
) -> Dict:
    segment = ROUTE_SEGMENTS[segment_name]
    _set_global_seed(seed)
    env = _make_env(start_state=segment.start_state, min_hold=min_hold, max_hold=max_hold)
    started_at = time.time()
    obs, info = env.reset(seed=seed)

    completed = False
    step_count = 0
    max_steps = segment.max_steps

    for step in range(max_steps):
        items_before = int(info.get("collected_items", 0) or info.get("items", 0))
        had_morph = bool(items_before & 0x1)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        step_count += 1

        if _segment_completed(segment_name, info, had_morph):
            completed = True
            break
        if terminated or truncated:
            break

    elapsed = time.time() - started_at
    env.close()

    return {
        "segment": segment_name,
        "seed": seed,
        "completed": completed,
        "steps": step_count,
        "elapsed_sec": round(elapsed, 3),
        "final_room_hex": _to_room_hex(info),
        "final_position": {"x": int(info.get("samus_x", 0)), "y": int(info.get("samus_y", 0))},
        "final_health": int(info.get("health", 0)),
    }


def _run_isolated_eval(
    models: Dict[str, PPO],
    trials: int,
    seed_base: int,
    min_hold: int,
    max_hold: int,
    deterministic: bool,
) -> Dict:
    results: Dict[str, List[Dict]] = {}
    for segment_name in ROUTE_ORDER:
        model = models[segment_name]
        segment_results = []
        for t in range(trials):
            seed = seed_base + t * 100
            result = _run_isolated_segment(
                model=model,
                segment_name=segment_name,
                seed=seed,
                min_hold=min_hold,
                max_hold=max_hold,
                deterministic=deterministic,
            )
            segment_results.append(result)
        successes = sum(1 for r in segment_results if r["completed"])
        rate = successes / trials if trials else 0.0
        status = "OK" if rate > 0.5 else ("WEAK" if rate > 0 else "FAIL")
        print(
            f"  {segment_name}: {successes}/{trials} ({rate:.0%}) [{status}]",
            flush=True,
        )
        results[segment_name] = segment_results

    return results


# =========================================================================
# RETRAIN SUGGESTIONS
# =========================================================================
def _build_retrain_suggestions(
    failure_data: Dict,
    device_arg: str,
    isolated_results: Optional[Dict] = None,
) -> List[Dict]:
    """Build retrain suggestions from both chained failures and isolated results."""
    suggestions = []
    seen = set()

    # From chained failures
    for segment_name, failures in failure_data.items():
        count = failures if isinstance(failures, int) else int(failures)
        if count <= 0:
            continue
        if count >= 6:
            steps = 500000
        elif count >= 3:
            steps = 300000
        else:
            steps = 150000
        suggestions.append({
            "segment": segment_name,
            "failure_count": count,
            "source": "chained",
            "suggested_steps": steps,
            "command": (
                f"python train_curriculum.py train --segment {segment_name} "
                f"--steps {steps} --device {device_arg}"
            ),
        })
        seen.add(segment_name)

    # From isolated failures (segments not already in suggestions)
    if isolated_results:
        for segment_name, trials in isolated_results.items():
            if segment_name in seen:
                continue
            successes = sum(1 for r in trials if r["completed"])
            total = len(trials)
            if total == 0:
                continue
            rate = successes / total
            if rate >= 0.6:
                continue
            if rate == 0:
                steps = 500000
            elif rate < 0.3:
                steps = 300000
            else:
                steps = 150000
            suggestions.append({
                "segment": segment_name,
                "failure_count": total - successes,
                "source": "isolated",
                "suggested_steps": steps,
                "command": (
                    f"python train_curriculum.py train --segment {segment_name} "
                    f"--steps {steps} --device {device_arg}"
                ),
            })

    # Sort by suggested_steps descending (worst segments first)
    suggestions.sort(key=lambda s: s["suggested_steps"], reverse=True)
    return suggestions


# =========================================================================
# SUMMARY WRITER
# =========================================================================
def _write_summary_markdown(path: str, payload: Dict) -> None:
    lines = [
        "# Worker C Overnight Integration Summary",
        "",
        f"- Generated: {payload['generated_at_utc']}",
    ]

    # Isolated results
    iso = payload.get("isolated_eval")
    if iso:
        lines.extend([
            "",
            "## Per-Segment Isolated Eval",
            "",
            "Each segment tested from its own start state (stochastic policy).",
            "",
            "| Segment | Pass Rate | Status | Avg Steps |",
            "| --- | ---: | :---: | ---: |",
        ])
        for segment_name in ROUTE_ORDER:
            trials = iso.get(segment_name, [])
            total = len(trials)
            if total == 0:
                lines.append(f"| `{segment_name}` | - | SKIP | - |")
                continue
            successes = sum(1 for r in trials if r["completed"])
            rate = successes / total
            avg_steps = sum(r["steps"] for r in trials) / total
            if rate > 0.5:
                status = "OK"
            elif rate > 0:
                status = "WEAK"
            else:
                status = "FAIL"
            lines.append(
                f"| `{segment_name}` | {successes}/{total} ({rate:.0%}) "
                f"| {status} | {avg_steps:.0f} |"
            )

    # Chained results
    chained = payload.get("chained_eval")
    if chained:
        summary = chained["summary"]
        lines.extend([
            "",
            "## Chained Route Eval (ZebesStart -> Torizo)",
            "",
            f"- Episodes: {summary['episodes']}",
            f"- Successes: {summary['successes']} ({summary['success_rate']:.1%})",
            f"- Avg steps: {summary['avg_steps']:.1f}",
            f"- Best segments completed: {summary['best_segments_completed']}/12",
            "",
            "### Transition Completion (chained)",
            "",
            "| Segment | Completed | Rate |",
            "| --- | ---: | ---: |",
        ])
        tc = summary["transition_completion"]
        for segment in ROUTE_ORDER:
            comp = tc.get(segment, 0)
            pct = (comp / summary["episodes"]) if summary["episodes"] else 0.0
            lines.append(f"| `{segment}` | {comp}/{summary['episodes']} | {pct:.0%} |")

        failures = summary.get("transition_failure_clusters", {})
        if failures:
            lines.extend(["", "### Failure Clusters (chained)", ""])
            lines.append("| Segment | Fail Count |")
            lines.append("| --- | ---: |")
            for seg, cnt in sorted(failures.items(), key=lambda x: -x[1]):
                lines.append(f"| `{seg}` | {cnt} |")

    # Retrain suggestions
    retrain = payload.get("retrain_suggestions", [])
    lines.extend(["", "## Targeted Retrain Commands", ""])
    if retrain:
        lines.append("Priority-ordered (worst first):")
        lines.append("")
        for item in retrain:
            lines.append(
                f"- **`{item['segment']}`** ({item['failure_count']} failures, "
                f"source: {item['source']}):  "
            )
            lines.append(f"  `{item['command']}`")
        # One-liner training script
        lines.extend(["", "### Batch retrain (copy-paste):", "", "```bash"])
        lines.append("cd " + PROJECT_ROOT)
        for item in retrain:
            lines.append(f".venv/bin/{item['command']}")
        lines.append("```")
    else:
        lines.append("No retrain commands suggested; route was stable.")

    lines.extend([
        "",
        "## Morning Check",
        "",
        "```bash",
        "bash scripts/morning_worker_c_check.sh",
        "```",
        "",
    ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================================
# MAIN
# =========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate segment chain from ZebesStart to flyway_to_torizo."
    )
    parser.add_argument(
        "--mode", choices=["chained", "isolated", "both"], default="both",
        help="Eval mode: chained (full route), isolated (per-segment), or both",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Chained episodes")
    parser.add_argument("--trials", type=int, default=5, help="Isolated trials per segment")
    parser.add_argument("--max-steps", type=int, default=18000, help="Max steps per chained episode")
    parser.add_argument("--seed-base", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic policy (default: stochastic)")
    parser.add_argument("--min-hold", type=int, default=2)
    parser.add_argument("--max-hold", type=int, default=4)
    parser.add_argument(
        "--output-json",
        default=os.path.join("logs", "overnight_worker_c_eval.json"),
    )
    parser.add_argument(
        "--output-summary",
        default=os.path.join("logs", "overnight_worker_c_summary.md"),
    )
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    _ensure_romfile()

    torch_device = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        torch_device = "cpu"
    device = torch.device(torch_device)

    det_label = "deterministic" if args.deterministic else "stochastic"
    print(
        f"Loading models for {len(ROUTE_ORDER)} segments on device={device} "
        f"(mode={args.mode}, policy={det_label})",
        flush=True,
    )
    models = _load_models(device=device)

    payload: Dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "mode": args.mode,
            "episodes": args.episodes,
            "trials_per_segment": args.trials,
            "max_steps": args.max_steps,
            "seed_base": args.seed_base,
            "device_requested": args.device,
            "device_used": str(device),
            "headless": bool(args.headless),
            "deterministic_policy": args.deterministic,
            "action_hold_range": [args.min_hold, args.max_hold],
        },
        "route_segments": ROUTE_ORDER,
    }

    # --- Isolated eval ---
    isolated_results = None
    if args.mode in ("isolated", "both"):
        print(f"\n=== Isolated Per-Segment Eval ({args.trials} trials each) ===", flush=True)
        isolated_results = _run_isolated_eval(
            models=models,
            trials=args.trials,
            seed_base=args.seed_base,
            min_hold=args.min_hold,
            max_hold=args.max_hold,
            deterministic=args.deterministic,
        )
        payload["isolated_eval"] = isolated_results

    # --- Chained eval ---
    if args.mode in ("chained", "both"):
        print(f"\n=== Chained Route Eval ({args.episodes} episodes) ===", flush=True)
        episode_results: List[Dict] = []
        for idx in range(1, args.episodes + 1):
            seed = args.seed_base + idx - 1
            result = _run_chained_episode(
                models=models,
                episode_index=idx,
                seed=seed,
                max_steps=args.max_steps,
                min_hold=args.min_hold,
                max_hold=args.max_hold,
                deterministic=args.deterministic,
            )
            episode_results.append(result)
            n_segs = result["segments_completed_count"]
            status = "SUCCESS" if result["success"] else "FAIL"
            print(
                f"  [{idx:02d}/{args.episodes:02d}] {status} "
                f"segs={n_segs}/12 steps={result['steps']} "
                f"failed={result['failed_segment']} "
                f"room={result['final_room_hex']}",
                flush=True,
            )

        successes = sum(1 for r in episode_results if r["success"])
        avg_steps = (
            float(sum(r["steps"] for r in episode_results)) / args.episodes
            if args.episodes else 0.0
        )
        best_segs = max(r["segments_completed_count"] for r in episode_results) if episode_results else 0

        failure_segments: Counter = Counter()
        transition_completion: Counter = Counter()
        for result in episode_results:
            for seg in result["completed_segments"]:
                transition_completion[seg] += 1
            if result["failed_segment"]:
                failure_segments[result["failed_segment"]] += 1

        payload["chained_eval"] = {
            "summary": {
                "episodes": args.episodes,
                "successes": successes,
                "success_rate": (successes / args.episodes) if args.episodes else 0.0,
                "avg_steps": avg_steps,
                "best_segments_completed": best_segs,
                "route_end_segment": ROUTE_ORDER[-1],
                "transition_completion": {s: int(transition_completion[s]) for s in ROUTE_ORDER},
                "transition_failure_clusters": {
                    s: int(c) for s, c in failure_segments.most_common() if c > 0
                },
            },
            "episodes_data": episode_results,
        }

    # --- Retrain suggestions ---
    chained_failures = {}
    if "chained_eval" in payload:
        chained_failures = payload["chained_eval"]["summary"]["transition_failure_clusters"]

    retrain_suggestions = _build_retrain_suggestions(
        failure_data=chained_failures,
        device_arg=args.device,
        isolated_results=isolated_results,
    )
    payload["retrain_suggestions"] = retrain_suggestions

    # --- Write outputs ---
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    _write_summary_markdown(path=args.output_summary, payload=payload)

    print(f"\nWrote JSON: {args.output_json}", flush=True)
    print(f"Wrote summary: {args.output_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
