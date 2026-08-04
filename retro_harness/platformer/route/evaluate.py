"""Independent per-segment route evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.level_config import get_level_config, LevelConfig
from retro_harness.platformer.route.models import (
    RouteConfig,
    RouteSegment,
    find_best_recording,
    load_recording_data,
)


# -- Route evaluation --------------------------------------------------------

@dataclass
class SegmentResult:
    """Result of evaluating one route segment."""

    segment: RouteSegment
    config: LevelConfig
    recording_path: Path | None
    eval_result: EvalResult | None
    error: str = ""


@dataclass
class RouteResult:
    """Result of evaluating a full route."""

    route: RouteConfig
    segments: list[SegmentResult] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(
            s.eval_result.total_frames
            for s in self.segments
            if s.eval_result and s.eval_result.completed
        )

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.segments if s.eval_result and s.eval_result.completed)

    @property
    def total_count(self) -> int:
        return len(self.segments)

    @property
    def all_completed(self) -> bool:
        return self.completed_count == self.total_count


def evaluate_route(
    route: RouteConfig,
    *,
    verbose: bool = True,
    pad_frames: int = 100,
) -> RouteResult:
    """Evaluate a full route by running each segment independently.

    Each segment gets its own Evaluator with its own saved state,
    so there's no emulator state leakage between segments.

    Args:
        route: The route to evaluate.
        verbose: Print per-segment results.
        pad_frames: Extra no-input frames appended to catch delayed transitions.
    """
    result = RouteResult(route=route)

    if verbose:
        print(f"Route: {route.display_name} ({len(route.segments)} segments)\n")
        print(f"{'#':>2s}  {'Label':<20s}  {'Config':<20s}  {'Recording':<35s}  "
              f"{'Frames':>6s}  {'Progress':>8s}  {'Fitness':>10s}  {'Status'}")
        print("-" * 130)

    cumulative = 0

    for i, seg in enumerate(route.segments):
        try:
            config = get_level_config(seg.config_id)
        except KeyError as e:
            sr = SegmentResult(seg, None, None, None, error=str(e))  # type: ignore[arg-type]
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {'':35s}  CONFIG ERROR: {e}")
            continue

        # Find recording
        if seg.recording:
            rec_path = Path(seg.recording)
            if not rec_path.is_absolute():
                rec_path = config.runs_dir / seg.recording
        else:
            rec_path = find_best_recording(config)

        if rec_path is None or not rec_path.exists():
            sr = SegmentResult(seg, config, rec_path, None, error="no recording")
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {'MISSING':35s}")
            continue

        try:
            actions, is_raw = load_recording_data(rec_path)

            # Pad with no-input frames for delayed transitions
            if is_raw:
                btn_len = len(actions[0]) if actions else 12
                actions = actions + [[0] * btn_len] * pad_frames
            else:
                actions = actions + [0] * pad_frames

            ev = Evaluator(config)
            er = ev.evaluate(actions, early_terminate=False)
            ev.close()

            sr = SegmentResult(seg, config, rec_path, er)
            result.segments.append(sr)

            if er.completed:
                cumulative += er.total_frames

            if verbose:
                status = "COMPLETED" if er.completed else ("DIED" if er.died else "incomplete")
                icon = {"COMPLETED": "+", "DIED": "X", "incomplete": "-"}[status]
                rec_name = rec_path.name
                if rec_path.parent.name != config.level_id:
                    rec_name = f"{rec_path.parent.name}/{rec_name}"
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {rec_name:<35s}  "
                      f"{er.total_frames:>6d}f  {er.max_progress:>8.1f}  "
                      f"{er.fitness:>10.0f}  {icon} {status}")

        except Exception as e:
            sr = SegmentResult(seg, config, rec_path, None, error=str(e))
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {rec_path.name:<35s}  ERROR: {e}")

    if verbose:
        print(f"\n{'='*80}")
        print(f"Results: {result.completed_count}/{result.total_count} segments completed")
        print(f"Total completion frames: {result.total_frames} ({result.total_frames / 60:.1f}s)")

        failed = [i for i, s in enumerate(result.segments)
                  if s.eval_result and not s.eval_result.completed]
        missing = [i for i, s in enumerate(result.segments)
                   if s.eval_result is None]
        if failed:
            labels = [result.segments[i].segment.label for i in failed]
            print(f"Failed/incomplete: {labels}")
        if missing:
            labels = [result.segments[i].segment.label for i in missing]
            print(f"Missing: {labels}")

    return result
