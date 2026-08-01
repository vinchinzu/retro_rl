"""Clean power-on runner for ALTTP's currently verified continuous tip.

This is intentionally a *tip* runner, not a misleading Sanctuary full run.
It keeps one environment alive from title through room ``0x50`` and executes
only segments whose graph contract is currently ``verification=continuous``:

``boot → castle_to_sword → secret_entrance_clear → pocket_to_main_hall
→ castle_dungeon_prefix``.

Zelda/Sanctuary legs remain outside this runner until they have real
natural-entry evidence.  The result preserves per-segment controller phases,
fails fast, and identifies the final tip from the same multi-truth anchor
resolver used by :class:`alttp.session.AlttpSession`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from alttp import primitives
from alttp.opening_route.anchors import resolve_continuous_tip_node
from alttp.opening_route.escape_graph import N_ROOM_50
from alttp.opening_route.segment import SegmentEvidence, ScriptSegment, get_segment
from alttp.ram import AlttpSnapshot, snapshot_to_diag
from alttp.route_report import RoutePhaseResult
from alttp.startup import StartupResult, boot_past_title_to_castle, build_boot_env


VERIFIED_TIP_SEGMENT_IDS: tuple[str, ...] = (
    "castle_to_sword",
    "sword_to_secret_entrance_clear",
    "pocket_to_main_hall",
    "castle_dungeon_prefix",
)

_BootFn = Callable[..., StartupResult]
_SegmentLookup = Callable[[str], ScriptSegment]
_SettleFn = Callable[..., primitives.PrimitiveResult]


@dataclass
class FullTipResult:
    """Evidence for one clean power-on attempt through the verified tip."""

    ok: bool
    phase: str
    frames: int
    snapshot: AlttpSnapshot
    tip_node: str
    boot: RoutePhaseResult
    segments: list[SegmentEvidence] = field(default_factory=list)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)
    source: str = "natural_boot"

    def to_report(self) -> dict[str, Any]:
        return {
            "kind": "alttp_verified_tip_run",
            "ok": self.ok,
            "phase": self.phase,
            "frames": self.frames,
            "source": self.source,
            "clean_chain": self.source == "natural_boot" and self.ok,
            "development_only": self.source != "natural_boot",
            "intervention": "clean",
            "verifiedTip": N_ROOM_50,
            "tipNode": self.tip_node,
            "blocker": self.blocker,
            "notes": list(self.notes),
            "final": snapshot_to_diag(self.snapshot),
            "boot": {
                "phase": self.boot.phase,
                "ok": self.boot.ok,
                "frames": self.boot.frames,
                "detail": self.boot.detail,
                "diag": self.boot.diag or snapshot_to_diag(self.boot.snapshot),
            },
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _boot_phase(result: StartupResult) -> RoutePhaseResult:
    """Normalize startup evidence into the shared route-phase report shape."""
    ok = bool(result.snapshot.on_castle_grounds)
    return RoutePhaseResult(
        phase="boot_to_castle",
        ok=ok,
        frames=result.frames,
        snapshot=result.snapshot,
        detail=(
            "verified controllable Hyrule Castle grounds predecessor"
            if ok
            else "boot did not reach controllable Hyrule Castle grounds"
        ),
        diag=snapshot_to_diag(result.snapshot),
    )


def _exit_acceptance_ok(segment: ScriptSegment, evidence: SegmentEvidence) -> bool:
    """Apply the registered exit contract to evidence returned by a segment."""
    keys = segment.exit.acceptance_keys
    if not keys:
        return evidence.ok
    values = [bool(evidence.acceptance.get(key, False)) for key in keys]
    accepted = all(values) if segment.exit.require_all else any(values)
    return evidence.ok and accepted


def run_to_verified_tip(
    env: object | None = None,
    *,
    close: bool = True,
    boot_fn: _BootFn = boot_past_title_to_castle,
    get_segment_fn: _SegmentLookup = get_segment,
    settle_fn: _SettleFn = primitives.settle_control,
    segment_ids: Sequence[str] = VERIFIED_TIP_SEGMENT_IDS,
) -> FullTipResult:
    """Power on once and run the continuous opening chain through room ``0x50``.

    Injectable ``boot_fn`` and ``get_segment_fn`` make the composition testable
    without an emulator.  Production callers should use the defaults.  This
    function never loads a development state, writes progression RAM, or
    dispatches planned graph legs.
    """
    owns_env = env is None
    active_env = build_boot_env() if env is None else env
    notes = [
        "Clean power-on composition; one environment is kept for all segments.",
        "Stops at verified continuous tip room_50; planned Zelda/Sanctuary legs are not run.",
    ]
    try:
        boot_result = boot_fn(active_env, close=False)
        boot = _boot_phase(boot_result)
        total_frames = boot.frames
        evidence_rows: list[SegmentEvidence] = []
        if not boot.ok:
            return FullTipResult(
                ok=False,
                phase="boot_to_castle",
                frames=total_frames,
                snapshot=boot.snapshot,
                tip_node=resolve_continuous_tip_node(boot.snapshot),
                boot=boot,
                blocker="natural boot did not reach Hyrule Castle grounds",
                notes=notes,
            )

        for segment_id in segment_ids:
            segment = get_segment_fn(segment_id)
            evidence = segment.play_checked(active_env, source="natural_boot")
            evidence_rows.append(evidence)
            total_frames += evidence.frames
            if not _exit_acceptance_ok(segment, evidence):
                missing = [
                    key
                    for key in segment.exit.acceptance_keys
                    if not evidence.acceptance.get(key, False)
                ]
                detail = evidence.blocker or "segment did not satisfy registered exit"
                if missing:
                    detail = f"{detail}; missing acceptance: {', '.join(missing)}"
                return FullTipResult(
                    ok=False,
                    phase=segment_id,
                    frames=total_frames,
                    snapshot=evidence.snapshot,
                    tip_node=resolve_continuous_tip_node(evidence.snapshot),
                    boot=boot,
                    segments=evidence_rows,
                    blocker=detail,
                    notes=notes,
                )

        settle = settle_fn(active_env)
        total_frames += settle.frames
        final = settle.snapshot
        if evidence_rows:
            evidence_rows[-1].phases.append(
                RoutePhaseResult(
                    phase="tip_settle_control",
                    ok=settle.ok,
                    frames=settle.frames,
                    snapshot=final,
                    detail=settle.reason,
                    diag=snapshot_to_diag(final),
                )
            )
            evidence_rows[-1].snapshot = final
        if not settle.ok:
            return FullTipResult(
                ok=False,
                phase="tip_settle_control",
                frames=total_frames,
                snapshot=final,
                tip_node=resolve_continuous_tip_node(final),
                boot=boot,
                segments=evidence_rows,
                blocker=f"tip transition did not settle: {settle.reason}",
                notes=notes,
            )
        tip_node = resolve_continuous_tip_node(final)
        if tip_node != N_ROOM_50:
            return FullTipResult(
                ok=False,
                phase="tip_mismatch",
                frames=total_frames,
                snapshot=final,
                tip_node=tip_node,
                boot=boot,
                segments=evidence_rows,
                blocker=f"expected verified tip {N_ROOM_50!r}, resolved {tip_node!r}",
                notes=notes,
            )
        return FullTipResult(
            ok=True,
            phase="verified_tip_reached",
            frames=total_frames,
            snapshot=final,
            tip_node=tip_node,
            boot=boot,
            segments=evidence_rows,
            notes=notes,
        )
    finally:
        if owns_env and close:
            active_env.close()  # type: ignore[attr-defined]
