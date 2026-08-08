"""Zelda escort → Sanctuary (room 0x12) — planned scaffold.

Entry: Zelda follower + lamp + control.
Exit (planned): ``in_sanctuary``; graph tip ``sanctuary``.

No fake progress: if already in Sanctuary, report ok; otherwise blocker
``escort → Sanctuary not implemented``.
"""

from __future__ import annotations

from alttp import primitives
from alttp.ram import AlttpSnapshot, snapshot_to_diag, zelda_rescued_accepted
from alttp.route_report import RoutePhaseResult, SegmentResult


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    return {
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "has_lamp": snapshot.has_lamp,
        "in_sanctuary": snapshot.in_sanctuary,
        "has_control": snapshot.has_control,
    }


def run_from_escort(
    env: object,
    *,
    source: str = "state_load_dev",
) -> SegmentResult:
    """Scaffold: settle, evaluate acceptance, no route implementation yet."""
    settle = primitives.settle_control(env)
    snap = settle.snapshot
    frames = settle.frames
    acc = evaluate_acceptance(snap)
    notes = [
        "Planned segment: Zelda escort through sewers → Sanctuary.",
        "verification=planned until continuous natural-entry proof.",
    ]
    phases = [
        RoutePhaseResult(
            phase="settle_control",
            ok=snap.has_control,
            frames=frames,
            snapshot=snap,
            detail="settled before escort_to_sanctuary scaffold",
            diag=snapshot_to_diag(snap),
        )
    ]

    if acc["in_sanctuary"]:
        return SegmentResult(
            ok=True,
            phase="sanctuary_reached",
            frames=frames,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker="",
            notes=notes + ["Already in Sanctuary (acceptance satisfied)."],
        )

    missing: list[str] = []
    if not acc["zelda_follower"]:
        missing.append("zelda_follower")
    if not acc["has_lamp"]:
        missing.append("lamp")
    if missing:
        return SegmentResult(
            ok=False,
            phase="entry_incomplete",
            frames=frames,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker=f"escort entry incomplete: missing {', '.join(missing)}",
            notes=notes,
        )

    return SegmentResult(
        ok=False,
        phase="not_implemented",
        frames=frames,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acc,
        blocker="escort → Sanctuary not implemented",
        notes=notes,
    )


def run_from_state(
    state_name: str = "CastleZeldaFollower",
    *,
    close: bool = True,
) -> SegmentResult:
    """Development diagnostic from an escort-prep checkpoint state."""
    from alttp.opening_route.runner import run_from_state as _run_from_state

    return _run_from_state(state_name, run_from_escort, close=close)
