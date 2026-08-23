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

from mortal_kombat.eval_match import (  # noqa: E402
    PROMOTE_MIN_ATTEMPTS,
    checkpoint_steps,
    list_v3_checkpoints,
    make_eval_env,
    make_raw_eval_env,
    may_promote,
    play_buttons_match,
    play_match,
)
from mortal_kombat.paths import MODEL_DIR  # noqa: E402
from mortal_kombat.roster import (  # noqa: E402
    KIND_PIXEL,
    KIND_RAM_V3,
    STAGES,
    PIXEL_FALLBACK,
    record_stage,
    resolve_model,
    v3_filename,
)

_KIND_CHOICES = ("", KIND_RAM_V3, KIND_PIXEL, "script")


def _print_table_header() -> None:
    print(f"{'Stage':<22} {'Model':<40} {'Kind':<8} {'Win%':>6} {'W':>3} {'L':>3}")
    print("-" * 88)


def _print_result_row(display: str, model: str, kind: str, wins: int, losses: int) -> None:
    rate = wins / max(1, wins + losses)
    print(f"{display:<22} {model:<40} {kind:<8} {rate:>5.0%} {wins:>3} {losses:>3}")


def _eval_ppo(
    path: Path, kind: str, state: str, attempts: int, deterministic: bool
) -> tuple[int, int]:
    import torch
    from stable_baselines3 import PPO

    from mortal_kombat.compat import install_fighters_common_alias

    install_fighters_common_alias()

    device = torch.device(
        "cpu" if kind == KIND_RAM_V3 else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = PPO.load(str(path), device=device)
    wins = 0
    losses = 0
    for _ in range(attempts):
        env = make_eval_env(kind, state)
        try:
            if play_match(model, env, deterministic=deterministic):
                wins += 1
            else:
                losses += 1
        finally:
            env.close()
    return wins, losses


def _eval_scripted(state: str, attempts: int) -> tuple[int, int]:
    from mortal_kombat.scripted import ScriptedPolicy

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


def _run_scripted_table(wanted: list[str], attempts: int) -> int:
    _print_table_header()
    for prefix, display, _mid in STAGES:
        if prefix not in wanted:
            continue
        wins, losses = _eval_scripted(f"{prefix}_LiuKang", attempts)
        _print_result_row(display, "scripted", "script", wins, losses)
    return 0


def _run_kind_table(wanted: list[str], kind: str, attempts: int, deterministic: bool) -> int:
    _print_table_header()
    for prefix, display, _mid in STAGES:
        if prefix not in wanted:
            continue
        if kind == KIND_RAM_V3:
            path = MODEL_DIR / v3_filename(prefix)
        else:
            fallback = PIXEL_FALLBACK.get(prefix)
            path = MODEL_DIR / fallback if fallback else None
        if path is None or not path.is_file():
            print(f"{display:<22} {'MISSING':<40} {'':8} {'SKIP':>6}")
            continue
        wins, losses = _eval_ppo(
            path, kind, f"{prefix}_LiuKang", attempts, deterministic
        )
        _print_result_row(display, path.name, kind, wins, losses)
    return 0


def _run_compare(wanted: list[str], attempts: int, deterministic: bool) -> int:
    _print_table_header()
    for prefix, display, _mid in STAGES:
        if prefix not in wanted:
            continue
        state = f"{prefix}_LiuKang"
        v3_path = MODEL_DIR / v3_filename(prefix)
        if v3_path.is_file():
            wins, losses = _eval_ppo(
                v3_path, KIND_RAM_V3, state, attempts, deterministic
            )
            _print_result_row(display, v3_path.name, KIND_RAM_V3, wins, losses)
        pixel_name = PIXEL_FALLBACK.get(prefix)
        pixel_path = MODEL_DIR / pixel_name if pixel_name else None
        if pixel_path is not None and pixel_path.is_file():
            wins, losses = _eval_ppo(
                pixel_path, KIND_PIXEL, state, attempts, deterministic
            )
            _print_result_row(display, pixel_path.name, KIND_PIXEL, wins, losses)
        wins, losses = _eval_scripted(state, attempts)
        _print_result_row(display, "scripted", "script", wins, losses)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--stages", default="", help="Comma prefixes (default: all 12)")
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Rank every v3 checkpoint for each selected stage",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated checkpoint filenames (default: every checkpoint)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="With --checkpoints, record the winning checkpoint in roster.json (N>=20)",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--kind",
        default="",
        choices=_KIND_CHOICES,
        help="Policy kind (default: resolve_model). script uses 12-button RAM eval.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Rank v3 vs pixel vs scripted per stage; do not write roster.json",
    )
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.kind == "script" and (args.promote or args.checkpoints):
        raise SystemExit("--promote is v3 checkpoints only")
    if args.compare and args.checkpoints:
        raise SystemExit("--compare and --checkpoints are mutually exclusive")
    if args.promote and not args.checkpoints:
        raise SystemExit("--promote requires --checkpoints")
    if args.checkpoints and args.kind not in ("", KIND_RAM_V3):
        raise SystemExit("--checkpoints ranks v3 zips only")
    if args.checkpoints:
        import torch
        from stable_baselines3 import PPO

        if not wanted:
            raise SystemExit("--checkpoints requires explicit --stages")
        names = [name.strip() for name in args.models.split(",") if name.strip()] or None
        if names and len(wanted) > 1:
            raise SystemExit("--models requires a single --stages prefix")
        for prefix in wanted:
            pattern = f"mk1_v3_{prefix}_ppo_*"
            candidates = list_v3_checkpoints(MODEL_DIR, prefix, names)
            missing = [path for path in candidates if not path.is_file()]
            if missing:
                raise SystemExit(
                    "missing checkpoints: " + ", ".join(str(path) for path in missing)
                )
            if not candidates:
                raise SystemExit(f"no checkpoints match {MODEL_DIR / pattern}")
            results: list[tuple[int, int, Path]] = []
            state = f"{prefix}_LiuKang"
            for path in candidates:
                model = PPO.load(str(path), device=torch.device("cpu"))
                wins = 0
                for _ in range(args.attempts):
                    env = make_eval_env(KIND_RAM_V3, state)
                    try:
                        wins += int(
                            play_match(model, env, deterministic=args.deterministic)
                        )
                    finally:
                        env.close()
                results.append((wins, checkpoint_steps(path), path))
                print(
                    f"{path.name:<52} {wins}/{args.attempts} "
                    f"({wins / args.attempts:.0%})",
                    flush=True,
                )
            wins, _steps, best = max(results, key=lambda row: (row[0], row[1]))
            print(f"best={best.name} wins={wins}/{args.attempts}")
            if args.promote:
                if not may_promote(args.attempts):
                    print(
                        f"--promote refused: need N>={PROMOTE_MIN_ATTEMPTS} attempts "
                        f"(PROMOTE_MIN_ATTEMPTS), got {args.attempts}"
                    )
                else:
                    record_stage(
                        prefix,
                        model=best.name,
                        kind=KIND_RAM_V3,
                        win_rate=wins / args.attempts,
                        attempts=args.attempts,
                    )
                    print(f"promoted={best.name}")
            elif best.name != v3_filename(prefix):
                print("not promoted; rerun with a larger N before --promote")
        return 0

    wanted = wanted or [prefix for prefix, _, _ in STAGES]
    if args.compare:
        return _run_compare(wanted, args.attempts, args.deterministic)
    if args.kind == "script":
        return _run_scripted_table(wanted, args.attempts)
    if args.kind in (KIND_RAM_V3, KIND_PIXEL):
        return _run_kind_table(wanted, args.kind, args.attempts, args.deterministic)

    import torch
    from stable_baselines3 import PPO

    from mortal_kombat.compat import install_fighters_common_alias

    install_fighters_common_alias()

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
        device = torch.device(
            "cpu" if kind == KIND_RAM_V3 else "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = PPO.load(str(path), device=device)
        wins = 0
        losses = 0
        state = f"{prefix}_LiuKang"
        for _ in range(args.attempts):
            env = make_eval_env(kind, state)
            try:
                if play_match(model, env, deterministic=args.deterministic):
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
