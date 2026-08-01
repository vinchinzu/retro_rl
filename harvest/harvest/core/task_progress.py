"""Structured progress snapshots for autoplay watchdog and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


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
        """Hashable signature for stall detection."""
        return (
            self.task_name,
            self.phase_text,
            self.phase_index,
            self.step_count,
            self.details,
            self.child.signature() if self.child is not None else None,
        )


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
