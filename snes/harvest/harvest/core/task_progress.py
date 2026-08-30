"""Structured progress snapshots for autoplay watchdog and diagnostics.

``ProgressSnapshot.signature()`` is a *semantic* stall key: task identity,
phase, child signature, and non-tick details. Elapsed ``step_count`` stays on
the snapshot for UI/diagnostics but is not progress, including any
``("step_count", ...)`` pair smuggled in ``details``.

Two independent stall windows sit beside the snapshot for later D2 / PlaySession
use (leftover_exec still has its own comparator this slice):

- Motion liveness (``motion_liveness_key``): target / approach / player
  position. Short window: ``MOTION_STALL_FRAMES`` (6s at 60fps).
- Goal progress (``goal_progress_key``): debris counts, crop planted/wet,
  carry, stamina. Long window: ``GOAL_STALL_FRAMES`` (leftover_exec default).

``stalled(last_frame, now, window)`` is the shared comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


MOTION_STALL_FRAMES = 360
GOAL_STALL_FRAMES = 24_000
_TICK_DETAIL_KEYS = frozenset({"step_count"})


def _semantic_details(details: Sequence[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(pair for pair in details if pair[0] not in _TICK_DETAIL_KEYS)


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable progress view for a task tree node."""

    task_name: str
    phase_text: str = ""
    phase_index: Optional[int] = None
    step_count: Optional[int] = None
    details: Tuple[Tuple[str, Any], ...] = ()
    child: Optional["ProgressSnapshot"] = None

    def signature(self) -> Tuple[Any, ...]:
        """Hashable semantic signature for stall detection.

        Elapsed ``step_count`` is diagnostic-only and is excluded.
        """
        return (
            self.task_name,
            self.phase_text,
            self.phase_index,
            _semantic_details(self.details),
            self.child.signature() if self.child is not None else None,
        )


def motion_liveness_key(*, target: Any, approach: Any, pos: Any) -> Tuple[Any, ...]:
    """Short-window key: navigation target / approach / player position."""
    return (target, approach, pos)


def goal_progress_key(
    *,
    debris: Any,
    planted: Any,
    wet: Any,
    carry: Any,
    stamina: Any,
) -> Tuple[Any, ...]:
    """Long-window key: debris / crop planted-wet / carry / stamina."""
    return (debris, planted, wet, carry, stamina)


def stalled(last_frame: int, now: int, window: int) -> bool:
    """True when ``now - last_frame`` has reached a positive ``window``."""
    return window > 0 and now - last_frame >= window


def _read_attr(task: object, name: str) -> Any:
    try:
        return getattr(task, name)
    except AttributeError:
        return None


def task_progress_snapshot(task: object, *, depth: int = 0) -> Optional[ProgressSnapshot]:
    """Build a progress snapshot from a task, using public APIs when present."""
    if task is None or depth > 4:
        return None

    progress = getattr(task, "progress_snapshot", None)
    if callable(progress):
        snap = progress()
        if isinstance(snap, ProgressSnapshot):
            return snap

    phase_text = _read_attr(task, "phase_text")
    if phase_text is None:
        phase = _read_attr(task, "_phase")
        if phase is not None:
            phase_text = str(phase).upper()

    details: list[tuple[str, Any]] = []
    for attr in (
        "phase_index",
        "step_count",
        "_wp_index",
        "_plot_phase",
        "_water_index",
        "_target_tile",
        "_approach_tile",
        "_target_cow_slot",
    ):
        value = _read_attr(task, attr)
        if value is not None:
            details.append((attr, value))

    child_task = _read_attr(task, "current_task")
    if child_task is None:
        child_task = _read_attr(task, "_current_task")
    if child_task is None:
        child_task = _read_attr(task, "_task")
    if child_task is None:
        child_task = _read_attr(task, "_nav")
    if child_task is None:
        child_task = _read_attr(task, "_inner")
    child = None
    if child_task is not None and child_task is not task:
        child = task_progress_snapshot(child_task, depth=depth + 1)

    return ProgressSnapshot(
        task_name=task.__class__.__name__,
        phase_text=str(phase_text or ""),
        phase_index=_read_attr(task, "phase_index"),
        step_count=_read_attr(task, "step_count"),
        details=tuple(details),
        child=child,
    )


def task_progress_chain(task: object) -> Tuple[Any, ...]:
    """Return a hashable chain for watchdog comparisons."""
    snap = task_progress_snapshot(task)
    return () if snap is None else (snap.signature(),)
