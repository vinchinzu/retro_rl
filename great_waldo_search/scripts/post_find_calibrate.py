"""Post-first-find Waldo calibration for Scene1.

Flow:
1. Load Scene1 with players=2
2. P2-A assist → P1-A confirm (+1000 scroll)
3. Settle score; optionally save Scene1_AfterFind1000.state
4. Second P2-A seek; grid/candidates around landing
5. Gate success on settled score (>=2500 or +1500) and/or found-flag /
   scene-frame change — never mid-animation RAM alone
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.targets import (
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    FOUND_FLAG_ADDR,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.env import make_env, save_state
from snes_oneshot.actions import buttons_multi, idle_action_multi
from snes_oneshot.cursor import CursorPose, CursorTarget, step_toward_target


SETTLE_FRAMES = 100
STABLE_SAMPLES = 5
STABLE_GAP = 8
WALDO_POINTS = 1500
TARGET_TOTAL = 2500


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _idle(env: object, frames: int) -> np.ndarray:
    obs = None
    for _ in range(frames):
        obs, *_rest = env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
    assert obs is not None
    return np.asarray(obs)


def _hold_p2a(env: object, frames: int) -> CursorPose:
    for _ in range(frames):
        env.step(buttons_multi(p2=("A",)))  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _drive(
    env: object,
    target: CursorTarget,
    *,
    frames: int = 500,
) -> CursorPose:
    for _ in range(frames):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
        action = step_toward_target(pose, target, fast_button="Y")
        if action.reason == "confirm_at_target":
            return pose
        multi = idle_action_multi(players=2)
        multi[:12] = list(action.action)
        env.step(multi)  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _read_score_flag(env: object) -> tuple[int, int, CursorPose]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    score = score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR])
    found = int(ram[FOUND_FLAG_ADDR])
    pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
    return score, found, pose


def _settle_score(env: object, *, settle: int = SETTLE_FRAMES) -> dict:
    """Idle then require score/flag stable across spaced samples."""
    obs = _idle(env, settle)
    samples: list[tuple[int, int]] = []
    for _ in range(STABLE_SAMPLES):
        score, found, _pose = _read_score_flag(env)
        samples.append((score, found))
        obs = _idle(env, STABLE_GAP)
    scores = [s for s, _f in samples]
    flags = [f for _s, f in samples]
    stable = len(set(scores)) == 1 and len(set(flags)) == 1
    return {
        "score": scores[-1],
        "found": flags[-1],
        "samples": samples,
        "stable": stable,
        "obs": obs,
    }


def _frame_signature(obs: np.ndarray) -> dict:
    """Compact signature to detect Carpet Flyers → next scene."""
    top = obs[:180]
    return {
        "mean_rgb": [float(x) for x in top.reshape(-1, 3).mean(axis=0)],
        "std": float(top.std()),
        "hash16": int(
            np.abs(top[::12, ::12].astype(np.int16)).sum() % 1_000_003
        ),
    }


def _confirm_click(env: object) -> None:
    for _ in range(6):
        env.step(buttons_multi(p1=("A",)))  # type: ignore[attr-defined]


def do_first_find(
    env: object,
    *,
    p2a_frames: int,
    out_dir: Path,
) -> dict:
    """P2-A seek + P1-A for the confirmed +1000 find."""
    _idle(env, 12)
    pose = _hold_p2a(env, p2a_frames)
    _idle(env, 4)
    before_score, before_found, _ = _read_score_flag(env)
    _confirm_click(env)
    settled = _settle_score(env)
    Image.fromarray(settled.pop("obs")).save(out_dir / "01_after_first_find.png")
    return {
        "assist_pose": {"x": pose.x, "y": pose.y},
        "score_before": before_score,
        "found_before": before_found,
        "score_after": settled["score"],
        "found_after": settled["found"],
        "stable": settled["stable"],
        "samples": settled["samples"],
        "sig": _frame_signature(
            np.asarray(Image.open(out_dir / "01_after_first_find.png"))
        ),
        "ok": (
            settled["stable"]
            and settled["score"] >= 1000
            and settled["found"] == 2
        ),
    }


def probe_click(
    env: object,
    target: CursorTarget,
    *,
    baseline_sig: dict,
    label: str,
    out_dir: Path,
) -> dict:
    """Drive to target, click, settle, detect scene change."""
    final = _drive(env, target)
    score_b, found_b, _ = _read_score_flag(env)
    _confirm_click(env)
    settled = _settle_score(env)
    obs = settled.pop("obs")
    png = out_dir / f"click_{label}_{target.x}_{target.y}.png"
    Image.fromarray(obs).save(png)
    sig = _frame_signature(obs)
    mean_delta = float(
        np.linalg.norm(
            np.asarray(sig["mean_rgb"]) - np.asarray(baseline_sig["mean_rgb"])
        )
    )
    delta = settled["score"] - score_b
    success = bool(
        settled["stable"]
        and (
            settled["score"] >= TARGET_TOTAL
            or delta >= WALDO_POINTS
            or mean_delta > 40.0
        )
    )
    return {
        "label": label,
        "target": {"x": target.x, "y": target.y},
        "final_pose": {"x": final.x, "y": final.y},
        "score_before": score_b,
        "found_before": found_b,
        "score_after": settled["score"],
        "found_after": settled["found"],
        "delta": delta,
        "stable": settled["stable"],
        "samples": settled["samples"],
        "mean_rgb_delta": mean_delta,
        "scene_changed": mean_delta > 40.0,
        "success": success,
        "png": str(png),
        "sig": sig,
    }


def candidate_grid(cx: int, cy: int, *, radius: int, step: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            x = cx + dx
            y = cy + dy
            if 8 <= x <= 248 and 16 <= y <= 180:
                pts.append((x, y))
    return pts


def run_calibrate(
    *,
    p2a_frames: int = 300,
    save_after_find: bool = True,
    grid_radius: int = 16,
    grid_step: int = 4,
    max_grid: int = 40,
    extra: list[tuple[int, int]] | None = None,
) -> dict:
    """Run first find, save state, then probe second-objective candidates."""
    _configure_headless()
    out_dir = RECORDINGS_DIR / "post_find_calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        game=GAME,
        state="Scene1",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    report: dict = {"rows": [], "first_find": None, "second_landing": None}
    try:
        env.reset()
        first = do_first_find(env, p2a_frames=p2a_frames, out_dir=out_dir)
        report["first_find"] = first
        print(
            f"[cal] first_find score={first['score_after']} "
            f"found={first['found_after']} ok={first['ok']}"
        )
        if not first["ok"]:
            raise RuntimeError(f"first find failed: {first}")

        state_path = None
        if save_after_find:
            state_path = str(
                save_state(env, GAME_DIR, GAME, "Scene1_AfterFind1000")
            )
            print(f"[cal] saved {state_path}")
        report["after_find_state"] = state_path
        baseline_sig = first["sig"]

        land = _hold_p2a(env, p2a_frames)
        _idle(env, 4)
        report["second_landing"] = {"x": land.x, "y": land.y}
        Image.fromarray(_idle(env, 2)).save(out_dir / "02_second_assist.png")
        print(f"[cal] second P2-A landing=({land.x},{land.y})")

        # Click exact assist landing first (fresh reload from after-find).
        points: list[tuple[str, int, int]] = [
            ("assist_landing", land.x, land.y),
            ("assist_206_100", 206, 100),
            ("visual_carpet", 98, 75),
            ("visual_roof", 205, 125),
            ("visual_dome", 88, 108),
            ("visual_sky_waldo", 220, 28),
            ("prior_r2100", 198, 94),
            ("prior_1350a", 200, 82),
            ("prior_1350b", 202, 82),
            ("prior_1350c", 204, 82),
            ("left_waldo_guess", 20, 115),
        ]
        if extra:
            for i, (x, y) in enumerate(extra):
                points.append((f"extra_{i}", x, y))

        grid = candidate_grid(land.x, land.y, radius=grid_radius, step=grid_step)
        # Prefer near landing; cap count.
        for x, y in grid[:max_grid]:
            points.append((f"grid_{x}_{y}", x, y))

        # Dedup by coord, keep first label.
        seen: set[tuple[int, int]] = set()
        unique: list[tuple[str, int, int]] = []
        for label, x, y in points:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            unique.append((label, x, y))

        for label, x, y in unique:
            env.close()
            env = make_env(
                game=GAME,
                state="Scene1_AfterFind1000",
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 8)
            # Re-seek so camera/objective assist matches post-find pattern.
            _hold_p2a(env, p2a_frames)
            _idle(env, 4)
            row = probe_click(
                env,
                CursorTarget(x=x, y=y, deadzone=2, label=label),
                baseline_sig=baseline_sig,
                label=label,
                out_dir=out_dir,
            )
            report["rows"].append(row)
            print(
                f"[cal] {label} ({x},{y}) "
                f"{row['score_before']}->{row['score_after']} "
                f"d={row['delta']} flag={row['found_after']} "
                f"stable={row['stable']} scene={row['scene_changed']} "
                f"ok={row['success']}"
            )
            if row["success"]:
                cleared = save_state(env, GAME_DIR, GAME, "Scene1_Cleared")
                report["cleared_state"] = str(cleared)
                report["waldo_hit"] = row
                Image.fromarray(
                    np.asarray(Image.open(row["png"]))
                ).save(out_dir / "03_success.png")
                print(f"[cal] SUCCESS saved {cleared}")
                break
    finally:
        env.close()

    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cal] wrote {path}")
    return report


def _parse_xy_list(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        xs, ys = part.split(":")
        out.append((int(xs), int(ys)))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2a-frames", type=int, default=300)
    parser.add_argument("--grid-radius", type=int, default=16)
    parser.add_argument("--grid-step", type=int, default=4)
    parser.add_argument("--max-grid", type=int, default=40)
    parser.add_argument(
        "--extra",
        default="",
        help="Extra points as x:y,x:y",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry for post-find Waldo calibration."""
    args = _build_parser().parse_args(argv)
    run_calibrate(
        p2a_frames=args.p2a_frames,
        save_after_find=not args.no_save,
        grid_radius=args.grid_radius,
        grid_step=args.grid_step,
        max_grid=args.max_grid,
        extra=_parse_xy_list(args.extra),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
