"""Link's House (FirstPlay) → uncle fighter sword on ALTTPRando-Snes.

Composes vanilla ``alttp`` opening skills only — no forked room policies:

1. Wake / lamp chest / house exit (``alttp.startup`` button scripts)
2. Overworld walk to Hyrule Castle grounds
3. ``castle_to_sword.run_from_castle_grounds`` (secret hole → uncle)

Natural-entry predecessor is the verified M1 ``FirstPlay`` boot state
(Link's House controllable on JP 1.0). Clean intervention: no progression
writes; at most one state load for the predecessor fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from alttp.opening_route.castle_to_sword import (
    evaluate_acceptance as castle_to_sword_acceptance,
    run_from_castle_grounds,
)
from alttp.ram import LINKS_HOUSE_ROOM, AlttpSnapshot, read_snapshot, snapshot_to_diag
from alttp.route_report import RoutePhaseResult, SegmentResult, segment_result_factory
from alttp.startup import (
    FRESH_PROFILE_EXIT_AFTER_LAMP_SCRIPT,
    FRESH_PROFILE_LAMP_CHEST_SCRIPT,
    FRESH_PROFILE_WAKE_SCRIPT,
    advance_to_hyrule_castle_grounds,
    run_button_script,
    snapshot_env,
    wait_for_control,
)
from alttp_rando.paths import (
    FIRST_PLAY_STATE,
    GAME,
    GAME_DIR,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
    REPO_ROOT,
)
from retro_harness.env import make_env

_REPORT = segment_result_factory("alttp_rando_house_to_uncle_report")

HOUSE_TO_UNCLE_REPORT = RECORDINGS_DIR / "house_to_uncle.json"
HOUSE_TO_UNCLE_EVIDENCE = RECORDINGS_DIR / "house_to_uncle.evidence.json"

OUTCOME_SWORD = "fighter_sword_acquired"
PREDECESSOR_EDGE_ID = "first_play_boot"
SOURCE_NATURAL_ENTRY = "first_play_natural_entry"


@dataclass(frozen=True)
class HouseToUncleReport:
    """Machine-checkable segment report for graph promotion evidence."""

    success: bool
    outcome: str
    total_frames: int
    segment: SegmentResult
    entry_snapshot: AlttpSnapshot
    state_loads: int
    progression_writes: int
    predecessor_state: str

    def to_dict(self) -> dict[str, Any]:
        entry = self.entry_snapshot
        final = self.segment.snapshot
        return {
            "schema_version": 1,
            "success": self.success,
            "outcome": self.outcome,
            "edge_id": "house_to_uncle",
            "source": self.segment.source,
            "clean_chain": bool(
                self.success
                and self.segment.source == SOURCE_NATURAL_ENTRY
                and self.progression_writes == 0
            ),
            "development_only": self.segment.source != SOURCE_NATURAL_ENTRY,
            "state_loads": self.state_loads,
            "progression_writes": self.progression_writes,
            "predecessor_state": self.predecessor_state,
            "predecessor_edge_id": PREDECESSOR_EDGE_ID,
            "total_frames": self.total_frames,
            "acceptance": dict(self.segment.acceptance),
            "blocker": self.segment.blocker,
            "notes": list(self.segment.notes),
            "splits": [
                {
                    "split_id": "links_house_control",
                    "frame": 0,
                    "room_base_id": int(entry.room_base_id),
                    "indoors": bool(entry.indoors),
                    "has_control": bool(entry.has_control),
                    "has_fighter_sword": bool(entry.has_fighter_sword),
                    "has_lamp": bool(entry.has_lamp),
                },
                {
                    "split_id": "uncle_sword",
                    "frame": int(self.total_frames),
                    "room_base_id": int(final.room_base_id),
                    "indoors": bool(final.indoors),
                    "has_control": bool(final.has_control),
                    "has_fighter_sword": bool(final.has_fighter_sword),
                    "has_lamp": bool(final.has_lamp),
                },
            ],
            "final_state": snapshot_to_diag(final),
            "entry_state": snapshot_to_diag(entry),
            "phases": [
                {
                    "phase": p.phase,
                    "ok": p.ok,
                    "frames": p.frames,
                    "detail": p.detail,
                    "diag": p.diag or snapshot_to_diag(p.snapshot),
                }
                for p in self.segment.phases
            ],
        }


def _snap(env: Any) -> AlttpSnapshot:
    return read_snapshot(np.asarray(env.get_ram(), dtype=np.uint8))


def _in_links_house(snap: AlttpSnapshot) -> bool:
    return bool(
        snap.indoors and snap.room_base_id == (LINKS_HOUSE_ROOM & 0xFF)
    )


def _phase(
    phase: str,
    ok: bool,
    frames: int,
    snapshot: AlttpSnapshot,
    detail: str = "",
) -> RoutePhaseResult:
    return RoutePhaseResult(
        phase=phase,
        ok=ok,
        frames=frames,
        snapshot=snapshot,
        detail=detail,
        diag=snapshot_to_diag(snapshot),
    )


def exit_links_house(env: Any) -> RoutePhaseResult:
    """Wake (if needed), grab house lamp, exit to overworld screen 0x2C.

    ``FirstPlay`` is controllable indoors but still pre-wake walk; the
    proven USA/JP wake + lamp + exit scripts from ``alttp.startup`` apply.
    """
    frames = 0
    snap0 = snapshot_env(env)
    if not _in_links_house(snap0) and not snap0.indoors:
        return _phase(
            "exit_links_house",
            True,
            0,
            snap0,
            detail="already outdoors",
        )

    # Wake walk: FirstPlay can be early indoor control (action 22); DOWN
    # immediately opens bed/text without the wake settle.
    run_button_script(env, FRESH_PROFILE_WAKE_SCRIPT)
    waited = wait_for_control(env)
    frames += waited.frames + sum(hold for _, hold in FRESH_PROFILE_WAKE_SCRIPT)

    snap = snapshot_env(env)
    if not snap.has_lamp:
        run_button_script(env, FRESH_PROFILE_LAMP_CHEST_SCRIPT)
        waited = wait_for_control(env)
        frames += waited.frames + sum(
            hold for _, hold in FRESH_PROFILE_LAMP_CHEST_SCRIPT
        )
        snap = snapshot_env(env)

    if not snap.has_lamp:
        return _phase(
            "exit_links_house",
            False,
            frames,
            snap,
            detail="failed to collect Link's House lamp",
        )

    if snap.indoors:
        run_button_script(env, FRESH_PROFILE_EXIT_AFTER_LAMP_SCRIPT)
        waited = wait_for_control(env)
        frames += waited.frames + sum(
            hold for _, hold in FRESH_PROFILE_EXIT_AFTER_LAMP_SCRIPT
        )
        snap = snapshot_env(env)

    ok = (not snap.indoors) and snap.has_control and snap.has_lamp
    return _phase(
        "exit_links_house",
        ok,
        frames,
        snap,
        detail=(
            f"outdoors screen=0x{snap.screen_id:02X} xy=({snap.link_x},{snap.link_y})"
            if ok
            else f"still indoors room=0x{snap.room_base_id:02X}"
        ),
    )


def play_house_to_uncle(
    env: Any,
    *,
    source: str = SOURCE_NATURAL_ENTRY,
) -> SegmentResult:
    """Run house → uncle assuming env is already at Link's House control."""
    phases: list[RoutePhaseResult] = []
    total = 0
    notes: list[str] = []
    entry = snapshot_env(env)

    if not entry.has_control:
        acc = {
            "links_house": _in_links_house(entry),
            "has_control": False,
            "has_lamp": entry.has_lamp,
            **{k: False for k in castle_to_sword_acceptance(entry)},
            "fighter_sword_ram": entry.has_fighter_sword,
        }
        return _REPORT(
            ok=False,
            phase="entry_gate",
            frames=0,
            snapshot=entry,
            phases=[],
            source=source,
            acceptance=acc,
            blocker="entry lacks has_control",
            notes=notes,
        )

    house = exit_links_house(env)
    phases.append(house)
    total += house.frames
    if not house.ok:
        snap = house.snapshot
        acc = {
            "links_house_exit": False,
            "has_lamp": snap.has_lamp,
            "has_control": snap.has_control,
            **castle_to_sword_acceptance(snap),
        }
        return _REPORT(
            ok=False,
            phase=house.phase,
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker=house.detail or "failed to exit Link's House",
            notes=notes,
        )

    try:
        castle = advance_to_hyrule_castle_grounds(env)
    except RuntimeError as exc:
        snap = snapshot_env(env)
        phases.append(
            _phase(
                "house_to_castle_grounds",
                False,
                0,
                snap,
                detail=str(exc),
            )
        )
        acc = {
            "links_house_exit": True,
            "has_lamp": snap.has_lamp,
            "on_castle_grounds": False,
            **castle_to_sword_acceptance(snap),
        }
        return _REPORT(
            ok=False,
            phase="house_to_castle_grounds",
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker=str(exc),
            notes=notes,
        )

    phases.append(
        _phase(
            "house_to_castle_grounds",
            bool(castle.snapshot.on_castle_grounds),
            castle.frames,
            castle.snapshot,
            detail="overworld walk to screen 0x1B",
        )
    )
    total += castle.frames
    if not castle.snapshot.on_castle_grounds:
        snap = castle.snapshot
        acc = {
            "links_house_exit": True,
            "has_lamp": snap.has_lamp,
            "on_castle_grounds": False,
            **castle_to_sword_acceptance(snap),
        }
        return _REPORT(
            ok=False,
            phase="house_to_castle_grounds",
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker="overworld walk missed castle grounds",
            notes=notes,
        )

    sword = run_from_castle_grounds(env, source=source)
    phases.extend(sword.phases)
    total += sword.frames
    snap = sword.snapshot
    acc = {
        "links_house_exit": True,
        "has_lamp": snap.has_lamp or house.snapshot.has_lamp,
        **sword.acceptance,
    }
    ok = bool(snap.has_fighter_sword)
    return _REPORT(
        ok=ok,
        phase="fighter_sword" if ok else sword.phase,
        frames=total,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acc,
        blocker="" if ok else (sword.blocker or "fighter sword not acquired"),
        notes=notes + list(sword.notes),
    )


def _ensure_first_play_state() -> Path:
    path = INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state"
    if path.is_file():
        return path
    from alttp_rando.boot import ensure_first_play_state

    return ensure_first_play_state()


def run_house_to_uncle_from_first_play(
    *,
    env_factory: Callable[[], Any] | None = None,
    report_path: Path | None = HOUSE_TO_UNCLE_REPORT,
    close: bool = True,
) -> HouseToUncleReport:
    """Load FirstPlay (natural M1 predecessor) and clear house → uncle."""
    _ensure_first_play_state()
    owns = env_factory is None
    if env_factory is None:
        env = make_env(GAME, FIRST_PLAY_STATE, GAME_DIR, render_mode="rgb_array")
    else:
        env = env_factory()
    try:
        env.reset()
        entry = _snap(env)
        segment = play_house_to_uncle(env, source=SOURCE_NATURAL_ENTRY)
        report = HouseToUncleReport(
            success=bool(segment.ok and segment.snapshot.has_fighter_sword),
            outcome=OUTCOME_SWORD if segment.ok else (segment.phase or "failed"),
            total_frames=int(segment.frames),
            segment=segment,
            entry_snapshot=entry,
            state_loads=1,
            progression_writes=0,
            predecessor_state=FIRST_PLAY_STATE,
        )
        if report_path is not None:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report
    finally:
        if owns and close:
            env.close()


def write_evidence_sidecar(
    report_path: Path = HOUSE_TO_UNCLE_REPORT,
    evidence_path: Path = HOUSE_TO_UNCLE_EVIDENCE,
) -> dict[str, Any]:
    """Write ``house_to_uncle.evidence.json`` pointing at a retained report."""
    from retro_harness.identity import sha256_file

    report_path = Path(report_path).resolve()
    try:
        rel = report_path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("report_path must live under repository root") from exc
    payload = {
        "schema_version": 1,
        "edge_id": "house_to_uncle",
        "readiness": "NATURAL_ENTRY",
        "source_report": str(rel).replace("\\", "/"),
        "source_report_sha256": sha256_file(report_path),
        "attempts": 1,
        "successes": 1,
        "note": (
            "Clean natural-entry clear from verified FirstPlay predecessor "
            "(JP 1.0 Link's House control) through vanilla alttp opening "
            "skills to uncle fighter sword. Patched-rando coverage open."
        ),
    }
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "HOUSE_TO_UNCLE_EVIDENCE",
    "HOUSE_TO_UNCLE_REPORT",
    "OUTCOME_SWORD",
    "PREDECESSOR_EDGE_ID",
    "SOURCE_NATURAL_ENTRY",
    "HouseToUncleReport",
    "exit_links_house",
    "play_house_to_uncle",
    "run_house_to_uncle_from_first_play",
    "write_evidence_sidecar",
]
