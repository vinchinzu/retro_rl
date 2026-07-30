"""Post-sword castle escape toward Zelda (room 0x55 → cell → Sanctuary).

Composes after ``castle_to_sword`` / ``FighterSword`` predecessor. Clean
intervention only — no progression writes or door warps.

Measured 2026-07-29 (headless, FighterSword / natural sword predecessor):

- After fighter sword, Link is in ``kPlayerState_HoldUpItem`` ($5D==21);
  dismiss with ~95 frames of LEFT before combat works.
- Room ``0x55`` is multi-screen. From uncle corridor (~2803,2680):
  LEFT×100 + DOWN×250 reaches the south chamber (~2680,2925).
  DOWN×280+ sinks into the stair pocket (~y=3000) and soft-traps.
- Floor stairs in the south chamber exit to castle grounds near Yaze
  entrance ``0x32`` (secret cellar door) — not deeper dungeon.
- Sprite type ``0x4B`` (75) = soldiers (killable with B); type ``0x73``
  (115) at uncle pose = non-combat corpse (ignore).
- Soldier kills can drop collectibles (family ``0xD8+``); small-key sprite
  is ``0xE4`` per zelda3 sources. ``$F36F==0xFF`` is the blank key HUD
  sentinel until dungeon key state initializes.
- Acceptance for rescue: ``follower_indicator`` ($F3CC) == 1 (Zelda).

Current blocker: leave room 0x55 into the wider castle (key door / shutter
path still under measurement). Do not claim Zelda rescue until
``has_zelda_follower`` is true on real RAM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from alttp.castle_to_sword import (
    RoutePhaseResult,
    _mash_text,
    _run_macro,
    dismiss_hold_up_item,
    settle_control,
)
from alttp.ram import (
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.startup import (
    action_for,
    no_action,
    snapshot_env,
    step_frames,
)

# Measured: uncle corridor → south combat chamber (stay above stair pocket).
SWORD_TO_SOUTH_CHAMBER_SCRIPT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("LEFT",), 100),
    (("DOWN",), 250),
)

# Soft y cap: deeper south enters stair exit / dead-end pocket.
SOUTH_CHAMBER_Y_MAX = 2965


@dataclass
class SwordToZeldaResult:
    """Segment result from a fighter-sword predecessor toward Zelda."""

    ok: bool
    phase: str
    frames: int
    snapshot: AlttpSnapshot
    phases: list[RoutePhaseResult] = field(default_factory=list)
    source: str = "unknown"
    acceptance: dict[str, bool] = field(default_factory=dict)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "kind": "alttp_sword_to_zelda_report",
            "ok": self.ok,
            "phase": self.phase,
            "frames": self.frames,
            "source": self.source,
            "clean_chain": self.source == "natural_boot" and self.ok,
            "development_only": self.source != "natural_boot",
            "acceptance": dict(self.acceptance),
            "blocker": self.blocker,
            "notes": list(self.notes),
            "final": snapshot_to_diag(self.snapshot),
            "phases": [
                {
                    "phase": p.phase,
                    "ok": p.ok,
                    "frames": p.frames,
                    "detail": p.detail,
                    "diag": p.diag or snapshot_to_diag(p.snapshot),
                }
                for p in self.phases
            ],
        }


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "in_secret_passage": snapshot.in_secret_passage,
        "hold_up_cleared": not snapshot.is_hold_up_item,
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "in_zelda_cell": snapshot.in_zelda_cell,
        "in_sanctuary": snapshot.in_sanctuary,
    }


def ensure_sword_control(env: object) -> RoutePhaseResult:
    """Dismiss hold-up-item and require fighter sword + control."""
    frames = settle_control(env)
    frames += dismiss_hold_up_item(env)
    frames += settle_control(env)
    snap = snapshot_env(env)
    ok = (
        snap.has_fighter_sword
        and snap.has_control
        and (not snap.is_hold_up_item)
        and snap.in_secret_passage
    )
    return RoutePhaseResult(
        phase="ensure_sword_control",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            "sword equipped, hold-up cleared, controllable in 0x55"
            if ok
            else (
                f"sword={snap.has_fighter_sword} hold_up={snap.is_hold_up_item} "
                f"control={snap.has_control} room=0x{snap.room_base_id:02X}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def approach_south_chamber(env: object) -> RoutePhaseResult:
    """Walk from uncle corridor into the south multi-screen chamber."""
    frames = settle_control(env)
    start = snapshot_env(env)
    if not start.in_secret_passage:
        return RoutePhaseResult(
            phase="approach_south_chamber",
            ok=False,
            frames=frames,
            snapshot=start,
            detail="not in secret-passage room 0x55",
            diag=snapshot_to_diag(start),
        )

    frames += _run_macro(
        env, SWORD_TO_SOUTH_CHAMBER_SCRIPT, stop_when_indoors=False
    )
    frames += settle_control(env)
    snap = snapshot_env(env)
    # Success: still indoors 0x55 and clearly south of uncle corridor y.
    ok = (
        snap.in_secret_passage
        and snap.link_y >= 2850
        and snap.link_y <= SOUTH_CHAMBER_Y_MAX + 20
    )
    return RoutePhaseResult(
        phase="approach_south_chamber",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"south chamber approach xy=({snap.link_x},{snap.link_y})"
            if ok
            else f"missed south chamber xy=({snap.link_x},{snap.link_y})"
        ),
        diag=snapshot_to_diag(snap),
    )


def run_from_sword(
    env: object,
    *,
    source: str = "state_load_dev",
    try_south: bool = True,
) -> SwordToZeldaResult:
    """Run post-sword segment assuming fighter sword already obtained."""
    phases: list[RoutePhaseResult] = []
    total = 0
    notes: list[str] = [
        "Zelda rescue path still under measurement after south chamber.",
        "Do not claim clean natural Zelda rescue until follower_indicator==1.",
    ]

    ready = ensure_sword_control(env)
    phases.append(ready)
    total += ready.frames
    if not ready.ok:
        acc = evaluate_acceptance(ready.snapshot)
        return SwordToZeldaResult(
            ok=False,
            phase=ready.phase,
            frames=total,
            snapshot=ready.snapshot,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker=ready.detail,
            notes=notes,
        )

    if try_south:
        south = approach_south_chamber(env)
        phases.append(south)
        total += south.frames
        snap = south.snapshot
        if not south.ok:
            acc = evaluate_acceptance(snap)
            return SwordToZeldaResult(
                ok=False,
                phase=south.phase,
                frames=total,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker=south.detail,
                notes=notes,
            )
    else:
        snap = snapshot_env(env)

    snap = snapshot_env(env)
    acc = evaluate_acceptance(snap)
    if acc["zelda_follower"]:
        return SwordToZeldaResult(
            ok=True,
            phase="zelda_rescued",
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker="",
            notes=notes,
        )

    return SwordToZeldaResult(
        ok=False,
        phase="south_chamber",
        frames=total,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acc,
        blocker=(
            "reached south chamber of room 0x55; need key/shutter path out "
            f"toward Zelda cell (room 0x{SECRET_PASSAGE_ROOM:02X} still active, "
            f"xy=({snap.link_x},{snap.link_y}), keys={snap.num_keys})"
        ),
        notes=notes,
    )


def run_from_state(
    state_name: str = "FighterSword",
    *,
    close: bool = True,
) -> SwordToZeldaResult:
    """Development diagnostic from a saved fighter-sword state."""
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        return run_from_sword(env, source="state_load_dev")
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]
