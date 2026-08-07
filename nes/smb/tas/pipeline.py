"""1-1 TAS optimize pipeline: complete → analyze → multi-window hillclimb → trim."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import smb.platformer_levels  # noqa: F401 — register LevelConfigs

from retro_harness.platformer.evaluator import EvalResult, Evaluator
from retro_harness.platformer.frame_tools import (
    analyze_seed_static,
    clone_frames,
    load_raw_frames,
    search_hold_compressions,
    trim_after_completion,
    trim_leading_idle,
)
from retro_harness.platformer.level_config import get_level_config
from retro_harness.platformer.segment_hillclimb import segment_hillclimb_raw
from smb.paths import GAME_DIR, MODELS_DIR
from smb.policy import compress_nes9_rle
from smb.tas.search import polish_systematic
from smb.tas.trace import SeedTrace, pad_to_completion, trace_seed
from smb.tas.windows import TasWindow, discover_windows, windows_from_labels

DEFAULT_SEEDS = (
    MODELS_DIR / "smb_1_1_stairs_clear_fragment.json",
    MODELS_DIR / "smb_1_1_clear.json",
)


@dataclass
class OptimizeReport:
    """Outcome of a full 1-1 optimize pass."""

    seed_in: str
    seed_out: str | None
    baseline_clear: int | None
    best_clear: int | None
    baseline_flag: int | None
    best_flag: int | None
    frames_saved: int = 0
    completed: bool = False
    windows_run: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    static: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_nes9(frames: Sequence[Sequence[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in frames:
        b = [int(x) for x in f[:9]]
        if len(b) < 9:
            b.extend([0] * (9 - len(b)))
        out.append(b)
    return out


def save_nes9_seed(
    path: Path,
    frames: Sequence[Sequence[int]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a ``nes9_rle`` seed under models/ (or *path*)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nes9 = _to_nes9(frames)
    payload: dict[str, Any] = {
        "format": "nes9_rle",
        "level_id": "smb_1_1",
        "start_state": "Level1_1",
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(nes9),
        "segments": compress_nes9_rle(nes9),
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_completing_seed(
    frames: Sequence[Sequence[int]],
    *,
    state_name: str = "Level1_1",
) -> tuple[list[list[int]], SeedTrace]:
    """Pad idle until leave-1-1, then trim a small post-clear pad via re-trace."""
    padded, tr = pad_to_completion(frames, state_name=state_name)
    if not tr.completed or tr.leave_frame is None:
        return padded, tr
    # Keep leave + 25 idle so evaluator completion stays stable
    keep = min(len(padded), tr.leave_frame + 25)
    trimmed = clone_frames(padded[:keep])
    # zero trailing junk past leave
    for i in range(tr.leave_frame, keep):
        trimmed[i] = [0] * 9
    final = trace_seed(trimmed, state_name=state_name)
    return trimmed, final


def _eval_clear(evaluator: Evaluator, frames: list[list[int]]) -> EvalResult:
    return evaluator.evaluate(frames, early_terminate=False)


def _close_evaluator(evaluator: Evaluator) -> None:
    """Release the emulator so other tools (trace_seed) can open one."""
    env = getattr(evaluator, "_env", None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
        evaluator._env = None
        evaluator._cached_state = None
        evaluator._initial_values = None


def pick_best_seed(paths: Sequence[Path] | None = None) -> Path:
    """Choose the completing seed with the earliest leave frame among *paths*."""
    candidates = list(paths) if paths else [p for p in DEFAULT_SEEDS if p.exists()]
    if not candidates:
        raise FileNotFoundError("no 1-1 seed candidates found under models/")

    best_path = candidates[0]
    best_leave: int | None = None
    for p in candidates:
        frames = load_raw_frames(p)
        _, tr = ensure_completing_seed(frames)
        if not tr.completed or tr.leave_frame is None:
            continue
        if best_leave is None or tr.leave_frame < best_leave:
            best_leave = tr.leave_frame
            best_path = p
    return best_path


def optimize_1_1(
    seed_path: Path | str | None = None,
    *,
    out_path: Path | str | None = None,
    window_labels: Sequence[str] | None = None,
    iters_per_window: int = 300,
    hold_compress: bool = True,
    trim_leading: bool = True,
    systematic: bool = True,
    delete_stride: int = 2,
    max_windows: int = 5,
    verbose: bool = True,
    output_dir: Path | str | None = None,
) -> tuple[list[list[int]], OptimizeReport]:
    """Full 1-1 TAS polish loop.

    1. Load seed (or pick best of known completing seeds)
    2. Pad to completion
    3. Discover / resolve windows
    4. Segment hill-climb each window (prefer frame deletes)
    5. Optional systematic delete + edge-shift sweep
    6. Hold compression (pre-flag only) + leading-idle trim
    7. Save best nes9_rle
    """
    t0 = time.time()
    if seed_path is None:
        seed_path = pick_best_seed()
    seed_path = Path(seed_path)
    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    raw = load_raw_frames(seed_path)
    frames, base_trace = ensure_completing_seed(raw)
    if not base_trace.completed:
        raise RuntimeError(
            f"seed {seed_path} does not complete 1-1 even after idle pad "
            f"(flag={base_trace.flag_frame}, max_x={base_trace.max_player_x})"
        )

    static = analyze_seed_static(frames)
    if verbose:
        print(
            f"[TAS-1-1] seed={seed_path.name} frames={len(frames)} "
            f"flag={base_trace.flag_frame} leave={base_trace.leave_frame} "
            f"wall_slams={len(base_trace.wall_slams)}"
        )

    config = get_level_config("smb_1_1")
    evaluator = Evaluator(config)
    out_dir = Path(output_dir) if output_dir else (GAME_DIR / "optimizer" / "runs" / "tas_1_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clear_of(result: EvalResult, fallback: int) -> int:
        if result.completed:
            return int(result.total_frames)
        return fallback

    base_eval = _eval_clear(evaluator, frames)
    baseline_clear = _clear_of(base_eval, base_trace.leave_frame or len(frames))
    baseline_flag = base_trace.flag_frame
    best = clone_frames(frames)
    best_eval = base_eval
    best_clear = baseline_clear
    steps: list[dict[str, Any]] = []
    windows_run: list[dict[str, Any]] = []

    steps.append(
        {
            "step": "baseline",
            "clear": baseline_clear,
            "flag": baseline_flag,
            "completed": base_eval.completed,
            "fitness": base_eval.fitness,
        }
    )

    # Resolve windows from the pad-to-completion trace (no second emulator).
    if window_labels:
        wins = windows_from_labels(
            window_labels,
            seed_len=len(best),
            flag_frame=baseline_flag,
        )
    else:
        wins = discover_windows(
            base_trace, seed_len=len(best), max_windows=max_windows
        )

    if verbose:
        print(f"[TAS-1-1] windows ({len(wins)}):")
        for w in wins:
            print(f"  {w.label:16s} [{w.start:4d}:{w.end:4d}] ({w.length}f)  {w.reason}")

    # Window coords are in baseline frame space. As deletes shrink the seed we
    # clamp to current length; avoid re-opening a second emulator mid-loop
    # (stable-retro allows one instance per process).
    flag_cap = max(1, (baseline_flag or len(best)) - 2)
    for w in wins:
        if iters_per_window <= 0:
            break
        hard = min(len(best), flag_cap)
        w = w.clamp(hard)
        if w.length < 20:
            continue
        if verbose:
            print(
                f"[TAS-1-1] hillclimb window {w.label} [{w.start}:{w.end}] "
                f"iters={iters_per_window} clear={best_clear}"
            )
        win_dir = out_dir / f"win_{w.label.replace('/', '_')}"
        cand, result = segment_hillclimb_raw(
            best,
            evaluator,
            window=(w.start, w.end),
            max_iterations=iters_per_window,
            prefer_trim=True,
            require_completion=True,
            output_dir=win_dir,
            verbose=verbose,
        )
        clear = _clear_of(result, best_clear)
        entry = {
            "window": w.to_dict(),
            "completed": result.completed,
            "clear": clear if result.completed else None,
            "fitness": result.fitness,
            "improved": bool(result.completed and clear < best_clear),
        }
        windows_run.append(entry)
        if result.completed and clear < best_clear:
            best = clone_frames(cand)
            best_eval = result
            best_clear = clear
            # Deleted frames before flag shift the absolute flag index earlier.
            deleted = max(0, (baseline_clear or 0) - best_clear)
            flag_cap = max(1, min(len(best) - 1, (baseline_flag or len(best)) - 2 - deleted // 4))
            if verbose:
                print(f"[TAS-1-1]  improved → clear={best_clear} (−{baseline_clear - best_clear}f cum)")
        steps.append({"step": f"hill:{w.label}", **{k: entry[k] for k in entry if k != "window"}})

    # Systematic delete + button-edge polish (stronger than random on local optima)
    if systematic:
        if verbose:
            print(
                f"[TAS-1-1] systematic sweep (stride={delete_stride}) "
                f"pre-flag~{baseline_flag}…"
            )
        best, best_eval, sys_rep = polish_systematic(
            best,
            evaluator,
            flag_frame=baseline_flag,
            delete_stride=delete_stride,
            verbose=verbose,
        )
        if sys_rep.best_clear is not None:
            best_clear = sys_rep.best_clear
        steps.append(
            {
                "step": "systematic",
                "clear": best_clear,
                "improvements": sys_rep.improvements,
                "moves": sys_rep.moves[:20],
                "elapsed_s": sys_rep.elapsed_s,
            }
        )
        if verbose:
            print(
                f"[TAS-1-1]  systematic imps={sys_rep.improvements} "
                f"clear={best_clear}"
            )

    # Hold compression — only mutate pre-flag body (castle idle is forced)
    if hold_compress:
        if verbose:
            print("[TAS-1-1] hold compression (pre-flag)…")

        def _ev(fr: list[list[int]]) -> EvalResult:
            return evaluator.evaluate(fr, early_terminate=False)

        # Temporarily zero post-flag for hold discovery by truncating search
        # to pre-flag+small pad via a wrapper length: compress on a prefix view.
        flag_cut = None
        # Prefer current best_flag estimate from baseline adjusted by length delta
        est_flag = baseline_flag
        if est_flag and est_flag < len(best):
            flag_cut = est_flag + 30
            prefix = clone_frames(best[:flag_cut])
            suffix = clone_frames(best[flag_cut:])
        else:
            prefix = best
            suffix = []

        hold_res = search_hold_compressions(
            prefix,
            # evaluate full sequence (prefix candidate + frozen suffix)
            lambda fr: _ev(fr + suffix),
            min_hold=24,
            min_keep=1,
            max_trials_per_hold=6,
            require_completion=True,
            verbose=verbose,
        )
        if hold_res.completed and hold_res.clear_frames is not None:
            cand_full = clone_frames(hold_res.frames) + suffix
            if hold_res.clear_frames < best_clear or len(cand_full) < len(best):
                best = cand_full
                best_clear = hold_res.clear_frames
                best_eval = _eval_clear(evaluator, best)
                steps.append(
                    {
                        "step": "hold_compress",
                        "clear": best_clear,
                        "trim": hold_res.trim,
                        "notes": hold_res.notes,
                    }
                )
                if verbose:
                    print(f"[TAS-1-1]  after holds clear={best_clear}")

    # Leading idle trim (Level1_1: even trims are usually phase-safe)
    if trim_leading:
        if verbose:
            print("[TAS-1-1] leading idle trim…")

        def _ev2(fr: list[list[int]]) -> EvalResult:
            return evaluator.evaluate(fr, early_terminate=False)

        trim_res = trim_leading_idle(
            best,
            _ev2,
            parity="even",
            require_completion=True,
            verbose=verbose,
        )
        if trim_res.completed and trim_res.clear_frames is not None:
            if trim_res.clear_frames <= best_clear:
                best = clone_frames(trim_res.frames)
                best_clear = trim_res.clear_frames
                best_eval = _eval_clear(evaluator, best)
                steps.append(
                    {
                        "step": "lead_trim",
                        "trim": trim_res.trim,
                        "clear": best_clear,
                        "notes": trim_res.notes,
                    }
                )

    # Post-clear pad trim
    def _ev3(fr: list[list[int]]) -> EvalResult:
        return evaluator.evaluate(fr, early_terminate=False)

    post = trim_after_completion(best, _ev3, pad=20, verbose=verbose)
    if post.completed and post.clear_frames is not None:
        best = clone_frames(post.frames)
        best_clear = post.clear_frames
        best_eval = _eval_clear(evaluator, best)
        steps.append({"step": "post_trim", "clear": best_clear, "notes": post.notes})

    # Final trace for flag metric (must close evaluator first — one emu/process)
    _close_evaluator(evaluator)
    final_tr = trace_seed(best)
    best_flag = final_tr.flag_frame
    if final_tr.leave_frame is not None:
        best_clear = final_tr.leave_frame

    meta = {
        "verified_completed": bool(final_tr.completed and best_eval.completed),
        "verified_clear_frames": best_clear,
        "verified_flag_frames": best_flag,
        "source": str(seed_path),
        "optimization": {
            "tool": "smb.tas.pipeline.optimize_1_1",
            "baseline_clear": baseline_clear,
            "baseline_flag": baseline_flag,
            "frames_saved": max(0, (baseline_clear or 0) - (best_clear or 0)),
            "windows": windows_run,
        },
        "notes": (
            f"TAS 1-1 polish: clear {baseline_clear}→{best_clear}, "
            f"flag {baseline_flag}→{best_flag}"
        ),
    }

    if out_path is None:
        out_path = MODELS_DIR / "smb_1_1_tas_best.json"
    out_path = Path(out_path)
    save_nes9_seed(out_path, best, metadata=meta)

    # Also dump report JSON next to runs
    report = OptimizeReport(
        seed_in=str(seed_path),
        seed_out=str(out_path),
        baseline_clear=baseline_clear,
        best_clear=best_clear,
        baseline_flag=baseline_flag,
        best_flag=best_flag,
        frames_saved=max(0, (baseline_clear or 0) - (best_clear or 0)),
        completed=bool(final_tr.completed),
        windows_run=windows_run,
        steps=steps,
        static=static,
        elapsed_s=time.time() - t0,
    )
    report_path = out_dir / "optimize_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(
            f"[TAS-1-1] DONE clear {baseline_clear}→{best_clear} "
            f"(−{report.frames_saved}f) flag {baseline_flag}→{best_flag} "
            f"in {report.elapsed_s:.1f}s → {out_path}"
        )
    return best, report
