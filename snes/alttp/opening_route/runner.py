"""Shared phase / segment runners for pre-room-engine hops.

Collapses the repeated settle → phase list → early-return →
``run_from_state(build_boot_env)`` pattern used by castle_to_sword,
secret_entrance_clear, pocket_to_main_hall, castle_dungeon, etc.

Callers pass an ordered list of phase callables (no per-module
``try_escape`` / ``try_south`` mode flags).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from alttp.ram import AlttpSnapshot
from alttp.route_report import RoutePhaseResult, SegmentResult

PhaseFn = Callable[[object], RoutePhaseResult]
AcceptFn = Callable[[AlttpSnapshot], dict[str, bool]]
PredFn = Callable[[AlttpSnapshot], bool]
PlayFn = Callable[..., SegmentResult]

R = TypeVar("R", bound=SegmentResult)


def run_phases(
    env: object,
    phases: Sequence[PhaseFn],
    *,
    evaluate_acceptance: AcceptFn,
    success_when: PredFn,
    source: str = "state_load_dev",
    notes: Sequence[str] = (),
    success_phase: str = "complete",
    success_notes: Sequence[str] = (),
    partial_phase: str = "partial",
    partial_blocker: str = "phases finished without acceptance",
    result_factory: Callable[..., R] = SegmentResult,  # type: ignore[assignment]
) -> R:
    """Run ordered phase callables with shared early-success / early-fail.

    After each phase:
    1. If ``success_when(snapshot)`` → ok with ``success_phase``.
    2. If phase ``ok`` is False → fail with phase detail as blocker.
    3. Otherwise continue.

    When the list is empty or finishes without success, returns a partial
    failure (or success if ``success_when`` already holds on the final snap).
    """
    from alttp.startup import snapshot_env

    phase_rows: list[RoutePhaseResult] = []
    total = 0
    note_list = list(notes)
    snap: AlttpSnapshot | None = None

    for phase_fn in phases:
        row = phase_fn(env)
        phase_rows.append(row)
        total += row.frames
        snap = row.snapshot
        if success_when(snap):
            acc = evaluate_acceptance(snap)
            return result_factory(
                ok=True,
                phase=success_phase,
                frames=total,
                snapshot=snap,
                phases=phase_rows,
                source=source,
                acceptance=acc,
                blocker="",
                notes=note_list + list(success_notes),
            )
        if not row.ok:
            acc = evaluate_acceptance(snap)
            return result_factory(
                ok=False,
                phase=row.phase,
                frames=total,
                snapshot=snap,
                phases=phase_rows,
                source=source,
                acceptance=acc,
                blocker=row.detail,
                notes=note_list,
            )

    if snap is None:
        snap = snapshot_env(env)
    acc = evaluate_acceptance(snap)
    if success_when(snap):
        return result_factory(
            ok=True,
            phase=success_phase,
            frames=total,
            snapshot=snap,
            phases=phase_rows,
            source=source,
            acceptance=acc,
            blocker="",
            notes=note_list + list(success_notes),
        )
    return result_factory(
        ok=False,
        phase=phase_rows[-1].phase if phase_rows else partial_phase,
        frames=total,
        snapshot=snap,
        phases=phase_rows,
        source=source,
        acceptance=acc,
        blocker=partial_blocker,
        notes=note_list,
    )


def run_from_state(
    state_name: str,
    play: PlayFn,
    *,
    close: bool = True,
    settle: bool = False,
    play_kwargs: Mapping[str, Any] | None = None,
) -> SegmentResult:
    """Load ``state_name``, optionally settle, run ``play(env, source=…)``.

    Shared replacement for the per-module ``run_from_state`` clones that all
    did ``build_boot_env`` → ``reset`` → optional settle → play → ``close``.
    """
    from alttp import primitives
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    kwargs = dict(play_kwargs or {})
    kwargs.setdefault("source", "state_load_dev")
    try:
        env.reset()  # type: ignore[attr-defined]
        if settle:
            primitives.settle_control(env)
        return play(env, **kwargs)
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]


__all__ = [
    "AcceptFn",
    "PhaseFn",
    "PlayFn",
    "PredFn",
    "run_from_state",
    "run_phases",
]
