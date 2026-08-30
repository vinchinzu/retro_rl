"""Replay a candidate from the immutable start and grade successor Join.

``grade(prepared, candidate_ref)`` replays from the PreparedTask entry digest,
records every intervention, and returns evidence. Never edits the product
manifest or promotes itself. RED always writes a leftover package; GREEN
always includes the next room's hop_glance Join grade. ``replay_green`` is
two GREEN replays from the same start. ``sync_green`` is not claimed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from super_metroid.hop_glance import LeaveMiss, final_from_state, grade_final
from super_metroid.paths import REPO_DIR
from super_metroid.splice.errors import GradeError, SchemaError
from super_metroid.splice.prepare import PreparedTask
from super_metroid.splice.schema import (
    CandidateArtifact,
    JoinPredicate,
    MemoryWrite,
    ReplayRow,
    candidate_kind,
)

GREEN = "GREEN"
RED = "RED"

Runner = Callable[[PreparedTask, CandidateArtifact], Any]


@dataclass(frozen=True)
class LeftoverPackage:
    """RED still plus miss list. Next boot is ``path``, not the pin."""

    path: str
    misses: tuple[str, ...]
    leftover: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "misses": list(self.misses),
            "leftover": dict(self.leftover),
        }


@dataclass(frozen=True)
class JoinGrade:
    """hop_glance against the task JoinPredicate (next-room LeaveSpec)."""

    passed: bool
    misses: tuple[str, ...]
    join: JoinPredicate
    leftover: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        nxt = self.join.next_entry
        return {
            "passed": self.passed,
            "misses": list(self.misses),
            "leave": self.join.leave.to_dict(),
            "next_entry": None if nxt is None else nxt.to_dict(),
            "leftover": dict(self.leftover),
        }


@dataclass(frozen=True)
class GradeReport:
    """Evidence only. Never a bank/manifest write or a promotion."""

    task_id: str
    candidate_id: str
    verdict: str
    start_digest: str
    replay_green: bool
    sync_green: bool
    interventions: tuple[MemoryWrite, ...]
    replay_rows: tuple[ReplayRow, ...]
    join: JoinGrade | None
    leftover_package: LeftoverPackage | None
    frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "start_digest": self.start_digest,
            "replay_green": self.replay_green,
            "sync_green": self.sync_green,
            "interventions": [w.to_dict() for w in self.interventions],
            "replay_rows": [r.to_dict() for r in self.replay_rows],
            "join": None if self.join is None else self.join.to_dict(),
            "leftover_package": (
                None if self.leftover_package is None else self.leftover_package.to_dict()
            ),
            "frames": self.frames,
        }


def _fail(message: str, code: str, **details: Any) -> NoReturn:
    raise GradeError(message, code=code, details=details)


def _artifact_digest(prepared: PreparedTask, kind: str) -> str | None:
    for art in prepared.artifacts:
        if art.kind == kind and art.digest:
            return art.digest
    return None


def _start_digest(prepared: PreparedTask) -> str:
    expected = prepared.card.entry_state_digest
    live = _artifact_digest(prepared, "state")
    if not expected:
        _fail("prepared entry digest missing", "grade.digest")
    if live and live != expected:
        _fail(
            "prepared pin digest does not match card entry digest",
            "grade.digest",
            expected=expected,
            actual=live,
        )
    return expected


def _as_candidate(
    candidate_ref: CandidateArtifact | Mapping[str, Any] | str,
    prepared: PreparedTask,
) -> CandidateArtifact:
    if isinstance(candidate_ref, CandidateArtifact):
        cand = candidate_ref
    elif isinstance(candidate_ref, Mapping):
        try:
            cand = CandidateArtifact.from_dict(candidate_ref)
        except SchemaError as exc:
            raise GradeError(
                str(exc),
                code=exc.code or "grade.candidate",
                details=exc.details,
            ) from exc
    else:
        cid = str(candidate_ref).strip()
        if not cid:
            _fail("candidate_ref required", "grade.candidate")
        try:
            kind = candidate_kind(cid)
        except SchemaError as exc:
            raise GradeError(
                str(exc),
                code=exc.code or "grade.candidate",
                details=exc.details,
            ) from exc
        impl = cid.split(":", 1)[1] if ":" in cid else cid
        cand = CandidateArtifact(
            candidate_id=cid,
            kind=kind,
            implementation_id=impl,
            task_id=prepared.task_id,
            entry_fingerprint=prepared.entry_fingerprint,
            start_state_digest=prepared.card.entry_state_digest,
            rom_digest=_artifact_digest(prepared, "rom"),
            core_digest=_artifact_digest(prepared, "core"),
            tape_digest=prepared.card.tape_digest,
        )
    if cand.task_id != prepared.task_id:
        _fail(
            f"candidate task {cand.task_id!r} does not match prepared {prepared.task_id!r}",
            "grade.candidate",
            candidate_task=cand.task_id,
            prepared_task=prepared.task_id,
        )
    return cand


def _check_start_digest(candidate: CandidateArtifact, expected: str) -> None:
    given = candidate.start_state_digest
    if not given or given != expected:
        _fail(
            "candidate start digest does not match prepared entry digest",
            "grade.digest",
            expected=expected,
            actual=given,
        )


def _as_writes(value: Any) -> tuple[MemoryWrite, ...]:
    if not value:
        return ()
    if isinstance(value, MemoryWrite):
        return (value,)
    if not isinstance(value, (list, tuple)):
        _fail("interventions must be a sequence", "grade.intervention")
    out: list[MemoryWrite] = []
    for item in value:
        if isinstance(item, MemoryWrite):
            out.append(item)
            continue
        if not isinstance(item, Mapping):
            _fail("each intervention must be an object", "grade.intervention")
        try:
            out.append(MemoryWrite.from_dict(item))
        except SchemaError as exc:
            raise GradeError(
                str(exc),
                code="grade.intervention",
                details=exc.details,
            ) from exc
    return tuple(out)


def _as_leftover_map(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("runner leftover must be an object", "grade.runner")
    return dict(value)


def _as_outcome(
    raw: Any,
) -> tuple[dict[str, Any], tuple[MemoryWrite, ...], int | None, str | None]:
    if isinstance(raw, LeaveMiss):
        return dict(raw.leftover), (), None, None
    if raw is None:
        _fail("runner returned no leftover", "grade.runner")
    session = getattr(raw, "state", None)
    if session is not None and not isinstance(raw, Mapping):
        return final_from_state(session), (), None, None
    if hasattr(raw, "samus_x") or hasattr(raw, "room_id") or hasattr(raw, "room"):
        if not isinstance(raw, Mapping):
            return final_from_state(raw), (), None, None
    if not isinstance(raw, Mapping):
        _fail("runner must return a leftover dict or session", "grade.runner")
    if "leftover" in raw or "interventions" in raw or "memory_writes" in raw:
        leftover = raw.get("leftover")
        if leftover is None:
            sess = raw.get("session")
            if sess is not None:
                leftover = final_from_state(getattr(sess, "state", sess))
        writes = _as_writes(raw.get("interventions") or raw.get("memory_writes") or ())
        frames = raw.get("frames")
        start = raw.get("start_digest")
        return (
            _as_leftover_map(leftover),
            writes,
            None if frames is None else int(frames),
            None if start is None else str(start),
        )
    return dict(raw), (), None, None


def _invoke_runner(
    runner: Runner | Any,
    prepared: PreparedTask,
    candidate: CandidateArtifact,
) -> tuple[dict[str, Any], tuple[MemoryWrite, ...], int | None, str | None]:
    try:
        if hasattr(runner, "replay") and callable(getattr(runner, "replay")):
            raw = runner.replay(prepared, candidate)
        elif callable(runner):
            raw = runner(prepared, candidate)
        else:
            _fail("runner hook is not callable", "grade.runner")
    except LeaveMiss as exc:
        return dict(exc.leftover), (), None, None
    except GradeError:
        raise
    except Exception as exc:
        raise GradeError(
            f"runner failed: {exc}",
            code="grade.runner",
            details={"error": type(exc).__name__},
        ) from exc
    return _as_outcome(raw)


def _join_grade(leftover: Mapping[str, Any], join: JoinPredicate) -> JoinGrade:
    spec = join.leave.to_leave_spec()
    try:
        misses = list(grade_final(leftover, spec))
    except (KeyError, TypeError, ValueError) as exc:
        misses = [f"{type(exc).__name__}: {exc}"]
    return JoinGrade(
        passed=not misses,
        misses=tuple(misses),
        join=join,
        leftover=dict(leftover),
    )


def _still_rel(still_path: Path, *, dest: Path, planned_still: str) -> str:
    # Dest-relative first so host-absolute tmp dirs stay leftover.state, not a stripped repo path.
    try:
        rel = still_path.resolve().relative_to(dest.resolve()).as_posix()
        if rel and not Path(rel).is_absolute() and not rel.startswith("/"):
            return rel.replace("\\", "/")
    except (OSError, ValueError):
        pass
    planned = str(planned_still or "").replace("\\", "/")
    if planned and not Path(planned).is_absolute() and not planned.startswith("/"):
        return planned
    return still_path.name


def _save_leftover(
    *,
    dest: Path,
    planned_still: str,
    leftover: Mapping[str, Any],
    misses: Sequence[str],
    task_id: str,
    candidate_id: str,
) -> LeftoverPackage:
    dest.mkdir(parents=True, exist_ok=True)
    still_path = dest / "leftover.state"
    package_path = dest / "leftover.json"
    still_rel = _still_rel(still_path, dest=dest, planned_still=planned_still)
    package = {
        "path": still_rel,
        "misses": list(misses),
        "leftover": dict(leftover),
        "task_id": task_id,
        "candidate_id": candidate_id,
    }
    still_path.write_text(json.dumps(dict(leftover), indent=2) + "\n", encoding="utf-8")
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    return LeftoverPackage(
        path=still_rel,
        misses=tuple(misses),
        leftover=dict(leftover),
    )


def _artifact_dest(prepared: PreparedTask, artifact_dir: Path | str | None) -> Path:
    if artifact_dir is not None:
        return Path(artifact_dir)
    planned = prepared.card.candidate_artifact_dir
    return Path(planned) if Path(planned).is_absolute() else REPO_DIR / planned


def grade(
    prepared: PreparedTask,
    candidate_ref: CandidateArtifact | Mapping[str, Any] | str,
    *,
    runner: Runner | Any | None = None,
    session: Any | None = None,
    trials: int | None = None,
    artifact_dir: Path | str | None = None,
) -> GradeReport:
    """Replay from the immutable start and return Join evidence. Never promotes."""
    if not isinstance(prepared, PreparedTask):
        _fail("prepared task required", "grade.prepared")
    if runner is None and session is None:
        _fail(
            "grade refuses to boot without a runner hook",
            "grade.runner",
            task_id=prepared.task_id,
        )
    if runner is None:
        captured = session

        def _session_runner(_prepared: PreparedTask, _candidate: CandidateArtifact) -> Any:
            return captured

        runner = _session_runner
        if trials is None:
            trials = 1
    if trials is None:
        trials = 2
    if int(trials) < 1:
        _fail("trials must be >= 1", "grade.trials")

    expected = _start_digest(prepared)
    candidate = _as_candidate(candidate_ref, prepared)
    _check_start_digest(candidate, expected)

    dest = _artifact_dest(prepared, artifact_dir)
    planned_still = prepared.card.completion.leftover_state_path
    rows: list[ReplayRow] = []
    writes: list[MemoryWrite] = []
    leftover_package: LeftoverPackage | None = None
    join_last: JoinGrade | None = None
    join_failed: JoinGrade | None = None
    total_frames = 0
    saw_frames = False

    for trial in range(1, int(trials) + 1):
        leftover, trial_writes, frames, start = _invoke_runner(runner, prepared, candidate)
        if start and start != expected:
            _fail(
                "runner start digest does not match prepared entry digest",
                "grade.digest",
                expected=expected,
                actual=start,
                trial=trial,
            )
        writes.extend(trial_writes)
        join = _join_grade(leftover, prepared.card.join)
        join_last = join
        if frames is not None:
            saw_frames = True
            total_frames += int(frames)
        miss = None if join.passed else "; ".join(join.misses)
        rows.append(ReplayRow(trial=trial, passed=join.passed, frames=frames, miss=miss))
        if not join.passed and leftover_package is None:
            join_failed = join
            leftover_package = _save_leftover(
                dest=dest,
                planned_still=planned_still,
                leftover=join.leftover,
                misses=join.misses,
                task_id=prepared.task_id,
                candidate_id=candidate.candidate_id,
            )

    verdict = GREEN if rows and all(row.passed for row in rows) else RED
    report_join = join_failed if verdict == RED else join_last
    if verdict == RED and leftover_package is None and report_join is not None:
        leftover_package = _save_leftover(
            dest=dest,
            planned_still=planned_still,
            leftover=report_join.leftover,
            misses=report_join.misses or ("leave failed",),
            task_id=prepared.task_id,
            candidate_id=candidate.candidate_id,
        )
    return GradeReport(
        task_id=prepared.task_id,
        candidate_id=candidate.candidate_id,
        verdict=verdict,
        start_digest=expected,
        replay_green=verdict == GREEN and len(rows) >= 2,
        sync_green=False,
        interventions=tuple(writes),
        replay_rows=tuple(rows),
        join=report_join,
        leftover_package=leftover_package if verdict == RED else None,
        frames=total_frames if saw_frames else None,
    )
