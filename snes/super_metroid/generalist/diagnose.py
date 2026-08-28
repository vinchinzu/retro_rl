"""No-training RAM/RGB trace probe for one contractor repertoire hop.

This command never calls ``learn`` or writes a checkpoint.  It evaluates a
heuristic or existing PPO deterministically from one practice pin and records
Join separately from room-route progress.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from super_metroid.generalist.corpus import load_rows
from super_metroid.generalist.env import GeneralistEnv
from super_metroid.generalist.evaluate import act
from super_metroid.paths import MODELS_DIR

DEFAULT_OUT = MODELS_DIR / "generalist" / "diagnostics"


def _action(policy: Any, obs: Any) -> int:
    if policy == "heuristic":
        return act(None, None, obs)
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def _trace_row(
    info: dict[str, Any],
    *,
    action: int,
    reward: float,
    previous_candidate: float,
) -> tuple[dict[str, Any], float]:
    candidate = float(info.get("steer_distance") or 0.0)
    row = {
        "frame": int(info["frame"]),
        "room": info["room"],
        "xy": list(info["xy"]),
        "pose": int(info["pose"]),
        "gs": int(info["gs"]),
        "action": int(action),
        "reward": float(reward),
        "candidate_potential": candidate,
        "candidate_delta": previous_candidate - candidate,
        "target_kind": info.get("steer_kind"),
        "target_xy": info.get("steer_xy"),
        "route_stage": info.get("steer_remaining_doors"),
        "route": info.get("steer_route"),
        "next_room": info.get("steer_next_room"),
    }
    return row, candidate


def _save_rgb(env: GeneralistEnv, path: Path) -> str | None:
    rgb = env.render()
    if rgb is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return str(path)


def diagnose(
    *,
    session_id: str,
    episodes: int = 8,
    policy_name: str = "heuristic",
    checkpoint: Path | None = None,
    sample_every: int = 60,
    capture_rgb: bool = False,
    out_dir: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Evaluate one practice pin and persist compact RAM/action traces."""

    rows = load_rows(
        area="crateria",
        exclude_ceres=True,
        dedupe=True,
        session_ids=[session_id],
    )
    if not rows:
        raise KeyError(f"captured repertoire session unavailable: {session_id}")
    if policy_name == "ppo":
        if checkpoint is None:
            raise ValueError("policy='ppo' requires checkpoint")
        from stable_baselines3 import PPO

        policy: Any = PPO.load(str(checkpoint), device="cpu")
    else:
        policy = "heuristic"

    env = GeneralistEnv(rows=rows, area=None)
    slug = session_id.rsplit("/", 1)[-1]
    episode_reports: list[dict[str, Any]] = []
    try:
        for episode in range(max(1, int(episodes))):
            obs, info = env.reset(options={"session_id": session_id})
            previous_candidate = float(info.get("steer_distance") or 0.0)
            previous_room = str(info["room"])
            actions: Counter[int] = Counter()
            trace: list[dict[str, Any]] = []
            stills: list[str] = []
            done = False
            while not done:
                action = _action(policy, obs)
                actions[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                transitioned = str(info["room"]) != previous_room
                sampled = int(info["frame"]) % max(1, int(sample_every)) < env.frame_skip
                row, previous_candidate = _trace_row(
                    info,
                    action=action,
                    reward=reward,
                    previous_candidate=previous_candidate,
                )
                if transitioned or sampled or done:
                    row["transition"] = transitioned
                    trace.append(row)
                    if capture_rgb and (transitioned or done):
                        label = "transition" if transitioned else "final"
                        path = out_dir / "rgb" / (
                            f"{slug}_e{episode:02d}_f{int(info['frame']):04d}_{label}.png"
                        )
                        saved = _save_rgb(env, path)
                        if saved is not None:
                            stills.append(saved)
                previous_room = str(info["room"])
            episode_reports.append(
                {
                    "episode": episode,
                    "reason": info["reason"],
                    "join": bool(info["join"]),
                    "frames": int(info["frame"]),
                    "final": {
                        "room": info["room"],
                        "xy": info["xy"],
                        "pose": info["pose"],
                        "gs": info["gs"],
                    },
                    "action_counts": {
                        str(action): count for action, count in sorted(actions.items())
                    },
                    "trace": trace,
                    "rgb": stills,
                }
            )
    finally:
        env.close()

    n = len(episode_reports)
    report = {
        "session_id": session_id,
        "goal_session_id": rows[0].goal_session_id,
        "policy": policy_name,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "episodes": n,
        "joins": sum(1 for row in episode_reports if row["join"]),
        "join_rate": sum(1 for row in episode_reports if row["join"]) / max(1, n),
        "reasons": dict(Counter(str(row["reason"]) for row in episode_reports)),
        "practice_only": True,
        "training": False,
        "episode_reports": episode_reports,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"probe_{slug}_{policy_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "session_id": session_id,
                "episodes": n,
                "join_rate": report["join_rate"],
                "reasons": report["reasons"],
                "training": False,
            },
            indent=2,
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--policy", choices=("heuristic", "ppo"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-every", type=int, default=60)
    parser.add_argument("--capture-rgb", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    diagnose(
        session_id=args.session,
        episodes=args.episodes,
        policy_name=args.policy,
        checkpoint=args.checkpoint,
        sample_every=args.sample_every,
        capture_rgb=args.capture_rgb,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUT", "diagnose", "main"]
