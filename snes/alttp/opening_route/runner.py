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
from alttp.startup import BootEnv

PhaseFn = Callable[[BootEnv], RoutePhaseResult]
AcceptFn = Callable[[AlttpSnapshot], dict[str, bool]]
PredFn = Callable[[AlttpSnapshot], bool]
PlayFn = Callable[..., SegmentResult]
DiagFn = Callable[[AlttpSnapshot], dict[str, bool]]

R = TypeVar("R", bound=SegmentResult)


def run_phases(
    env: BootEnv,
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
    evaluate_diagnostics: DiagFn | None = None,
) -> R:
    """Run ordered phase callables with shared early-success / early-fail.

    After each phase:
    1. If ``success_when(snapshot)`` → ok with ``success_phase``.
    2. If phase ``ok`` is False → fail with phase detail as blocker.
    3. Otherwise continue.

    When the list is empty or finishes without success, returns a partial
    failure (or success if ``success_when`` already holds on the final snap).

    ``evaluate_acceptance`` must return only this segment's contract keys.
    Optional ``evaluate_diagnostics`` fills log-only flags (later-route state)
    on the result without mixing them into acceptance.
    """
    from alttp.startup import snapshot_env

    phase_rows: list[RoutePhaseResult] = []
    total = 0
    note_list = list(notes)
    snap: AlttpSnapshot | None = None

    def _pack(
        *,
        ok: bool,
        phase: str,
        blocker: str,
        notes_extra: Sequence[str] = (),
    ) -> R:
        assert snap is not None
        acc = evaluate_acceptance(snap)
        diag = evaluate_diagnostics(snap) if evaluate_diagnostics else {}
        return result_factory(
            ok=ok,
            phase=phase,
            frames=total,
            snapshot=snap,
            phases=phase_rows,
            source=source,
            acceptance=acc,
            diagnostics=diag,
            blocker=blocker,
            notes=note_list + list(notes_extra),
        )

    for phase_fn in phases:
        row = phase_fn(env)
        phase_rows.append(row)
        total += row.frames
        snap = row.snapshot
        if success_when(snap):
            return _pack(
                ok=True,
                phase=success_phase,
                blocker="",
                notes_extra=success_notes,
            )
        if not row.ok:
            return _pack(ok=False, phase=row.phase, blocker=row.detail)

    if snap is None:
        snap = snapshot_env(env)
    if success_when(snap):
        return _pack(
            ok=True,
            phase=success_phase,
            blocker="",
            notes_extra=success_notes,
        )
    return _pack(
        ok=False,
        phase=phase_rows[-1].phase if phase_rows else partial_phase,
        blocker=partial_blocker,
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
        env.reset()
        if settle:
            primitives.settle_control(env)
        return play(env, **kwargs)
    finally:
        if close:
            env.close()


__all__ = [
    "AcceptFn",
    "DiagFn",
    "PhaseFn",
    "PlayFn",
    "PredFn",
    "run_from_state",
    "run_phases",
]
