"""BC then PPO on the practice-repertoire contractor env.

```bash
uv run python -m super_metroid.generalist.corpus --status
uv run python -m super_metroid.generalist.train --bc --same-room --n-envs 8 --timesteps 400000
uv run python -m super_metroid.generalist.train --eval-only --checkpoint models/generalist/overnight/best.zip
```

Weights go under ``models/generalist/`` (gitignored). Reports are JSON next
to the zip. Not a product tip.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from super_metroid.generalist.corpus import load_rows
from super_metroid.generalist.env import FRAME_SKIP, GeneralistEnv
from super_metroid.generalist.evaluate import (
    eval_join_rate,
    eval_per_session,
    heuristic_action,
)
from super_metroid.generalist.obs import N_ACTIONS, N_GRID, OBS_DIM, schema_digests
from super_metroid.generalist.solid import require_row_solids
from super_metroid.paths import MODELS_DIR

DEFAULT_OUT = MODELS_DIR / "generalist"


def collect_bc_samples(
    env: GeneralistEnv, *, n_steps: int = 4_000
) -> tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for _ in range(n_steps):
        action = heuristic_action(obs)
        xs.append(obs.copy())
        ys.append(int(action))
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def behavior_clone(
    model: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    steps: int = 200,
    batch_size: int = 64,
) -> float:
    """Supervised warm-start of an SB3 Discrete policy."""

    import torch

    device = model.device
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    n = len(observations)
    last = 0.0
    for _ in range(steps):
        index = np.random.randint(0, n, size=min(batch_size, n))
        batch_act = torch.as_tensor(actions[index], device=device)
        obs_t, _ = policy.obs_to_tensor(observations[index])
        dist = policy.get_distribution(obs_t)
        loss = -dist.log_prob(batch_act).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss.detach().cpu())
    return last


def resolve_rows(
    *,
    subset: str,
    session_ids: list[str] | None,
    same_room: bool,
) -> list[Any]:
    area = None if subset in {"all", "kpdr25"} else subset
    rows = load_rows(
        area=area,
        exclude_ceres=area == "crateria",
        dedupe=True,
        session_ids=session_ids or None,
        same_room=True if same_room else None,
    )
    if not rows:
        raise RuntimeError(f"no captured rows for subset={subset!r} same_room={same_room}")
    return rows


def _vec_env(rows: list[Any], n_envs: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make_thunk(_rank: int):
        def factory() -> GeneralistEnv:
            return GeneralistEnv(rows=rows, area=None, frame_skip=FRAME_SKIP)

        return factory

    thunks = [make_thunk(rank) for rank in range(max(1, n_envs))]
    if n_envs <= 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="fork")


def checkpoint_schema_path(zip_path: Path) -> Path:
    return Path(zip_path).with_suffix(".schema.json")


def _atomic_save(
    model: Any,
    zip_path: Path,
    *,
    schema: dict[str, str] | None = None,
) -> None:
    """Write a sibling tmp zip then replace, with an optional schema sidecar."""

    zip_path = Path(zip_path)
    tmp_path = zip_path.with_name(zip_path.stem + ".saving.zip")
    if tmp_path.exists():
        tmp_path.unlink()
    model.save(str(tmp_path))
    tmp_path.replace(zip_path)
    if schema is not None:
        schema_path = checkpoint_schema_path(zip_path)
        tmp_schema = schema_path.with_name(schema_path.stem + ".saving.json")
        tmp_schema.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        tmp_schema.replace(schema_path)


def require_compatible_checkpoint(
    checkpoint: Path,
    expected_schema: dict[str, str],
) -> None:
    """Fail closed before training across a changed reward/obs contract."""

    schema_path = checkpoint_schema_path(checkpoint)
    if not schema_path.is_file():
        raise RuntimeError(
            f"training resume refused: checkpoint schema missing: {schema_path}. "
            "Old checkpoints may be evaluated, but start a fresh model for the "
            "current reward contract"
        )
    try:
        actual = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"training resume refused: invalid checkpoint schema: {schema_path}"
        ) from exc
    mismatches = {
        key: {"checkpoint": actual.get(key), "current": value}
        for key, value in expected_schema.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "training resume refused: checkpoint contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _close_vec(vec: Any, timeout: float = 8.0) -> None:
    """SB3 SubprocVecEnv.close joins with no timeout; a stuck emu burns the night."""

    processes = list(getattr(vec, "processes", ()) or ())
    remotes = list(getattr(vec, "remotes", ()) or ())
    finished = threading.Event()

    def _close() -> None:
        try:
            vec.close()
        except Exception:
            pass
        finished.set()

    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(max(0.05, float(timeout)))
    if finished.is_set():
        return
    for remote in remotes:
        try:
            remote.close()
        except Exception:
            pass
    for proc in processes:
        pid = getattr(proc, "pid", None)
        if pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        kill = getattr(proc, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass
    for proc in processes:
        join = getattr(proc, "join", None)
        if callable(join):
            try:
                join(1.0)
            except Exception:
                pass
    try:
        vec.closed = True
    except Exception:
        pass


def train(
    *,
    subset: str = "crateria",
    timesteps: int = 50_000,
    bc: bool = False,
    ppo: bool = True,
    out_dir: Path = DEFAULT_OUT,
    seed: int = 0,
    eval_episodes: int = 8,
    n_envs: int = 1,
    same_room: bool = False,
    session_ids: list[str] | None = None,
    eval_only: bool = False,
    checkpoint: Path | None = None,
    skip_baselines: bool = False,
    tag: str | None = None,
) -> dict[str, Any]:
    rows = resolve_rows(subset=subset, session_ids=session_ids, same_room=same_room)
    require_row_solids(rows)

    from stable_baselines3 import PPO

    import torch

    torch.set_num_threads(1)
    stem = tag or ("same_room" if same_room else subset)
    report: dict[str, Any] = {
        "subset": subset,
        "same_room": same_room,
        "session_ids": [row.session_id for row in rows],
        "rows": len(rows),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "frame_skip": FRAME_SKIP,
        "n_envs": int(n_envs),
        "schema": schema_digests(frame_skip=FRAME_SKIP),
        "practice_only": True,
        "seed": seed,
        "started_unix": time.time(),
    }

    probe = GeneralistEnv(rows=rows, area=None, frame_skip=FRAME_SKIP)
    probe_obs, _probe_info = probe.reset()
    occupancy_max = float(np.max(np.abs(probe_obs[:N_GRID])))
    print(
        json.dumps(
            {
                "event": "probe",
                "session_ids": [row.session_id for row in rows],
                "rows": len(rows),
                "occupancy_max": occupancy_max,
                "collision_root": (
                    None if probe._collision_root is None else str(probe._collision_root)
                ),
                "same_room": same_room,
                "seed": seed,
                "tag": stem,
            }
        ),
        flush=True,
    )
    if occupancy_max <= 0.0:
        probe.close()
        raise RuntimeError(
            "generalist occupancy is empty on reset; editor collision missing "
            f"for seed={seed} tag={stem}"
        )
    if not skip_baselines and not eval_only:
        report["random"] = eval_join_rate(probe, "random", episodes=min(4, eval_episodes))
        report["heuristic"] = eval_per_session(probe, "heuristic", episodes=eval_episodes)
        print(
            json.dumps(
                {
                    "event": "baselines_done",
                    "seed": seed,
                    "tag": stem,
                    "random_join": report["random"]["join_rate"],
                    "heuristic_join": report["heuristic"]["join_rate"],
                    "heuristic_occupancy_filled": report["heuristic"][
                        "occupancy_filled"
                    ],
                }
            ),
            flush=True,
        )
    samples: tuple[np.ndarray, np.ndarray] | None = None
    if bc and not eval_only:
        samples = collect_bc_samples(probe, n_steps=4_000)
        report["bc_samples"] = int(len(samples[0]))

    if eval_only:
        if checkpoint is None:
            probe.close()
            raise RuntimeError("--eval-only needs --checkpoint")
        model = PPO.load(str(checkpoint), device="cpu")
        report["checkpoint"] = str(checkpoint)
        report["ppo"] = eval_per_session(probe, model, episodes=eval_episodes)
        probe.close()
        out_dir.mkdir(parents=True, exist_ok=True)
        report["finished_unix"] = time.time()
        path = out_dir / f"eval_{stem}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in report if k != "schema"}, indent=2))
        return report
    probe.close()

    if checkpoint is not None and Path(checkpoint).is_file():
        # Refuse incompatible reward resumes before allocating emulator workers.
        require_compatible_checkpoint(Path(checkpoint), report["schema"])
    vec = _vec_env(rows, n_envs)
    rollout = max(64, 2048 // max(1, n_envs))
    batch = min(256, rollout * max(1, n_envs))
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=rollout,
        batch_size=max(64, batch),
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device="cpu",
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )
    if checkpoint is not None and Path(checkpoint).is_file():
        model = PPO.load(str(checkpoint), env=vec, device="cpu")
        report["resumed"] = str(checkpoint)
    if bc and samples is not None:
        report["bc_loss"] = behavior_clone(model, samples[0], samples[1])
    if ppo:
        print(
            json.dumps(
                {
                    "event": "ppo_start",
                    "seed": seed,
                    "tag": stem,
                    "timesteps": int(timesteps),
                    "n_envs": int(n_envs),
                }
            ),
            flush=True,
        )
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        report["timesteps"] = int(timesteps)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"ppo_{stem}_s{seed}.zip"
    _atomic_save(model, zip_path, schema=report["schema"])
    report["checkpoint"] = str(zip_path)
    eval_env = GeneralistEnv(rows=rows, area=None, frame_skip=FRAME_SKIP)
    eval_error: BaseException | None = None
    try:
        report["ppo"] = eval_per_session(eval_env, model, episodes=eval_episodes)
    except BaseException as exc:
        eval_error = exc
        report["ppo_eval_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            eval_env.close()
        except Exception:
            pass
    report["finished_unix"] = time.time()
    report_path = out_dir / f"train_{stem}_s{seed}.json"
    if "ppo" in report:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = {k: report[k] for k in report if k not in {"schema", "heuristic"}}
        if "heuristic" in report:
            summary["heuristic"] = {
                k: report["heuristic"][k]
                for k in report["heuristic"]
                if k != "by_session"
            }
        print(json.dumps(summary, indent=2), flush=True)
    _close_vec(vec)
    if eval_error is not None:
        raise eval_error
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="crateria")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--no-ppo", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--same-room", action="store_true")
    parser.add_argument("--session", action="append", dest="sessions", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    train(
        subset=args.subset,
        timesteps=args.timesteps,
        bc=args.bc,
        ppo=not args.no_ppo,
        out_dir=args.out,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        same_room=args.same_room,
        session_ids=args.sessions,
        eval_only=args.eval_only,
        checkpoint=args.checkpoint,
        skip_baselines=args.skip_baselines,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
