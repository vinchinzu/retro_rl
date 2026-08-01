"""Secret-entrance clear: post-sword room 0x55 → outdoors courtyard pocket.

Segment success is **left_secret_entrance only** (stairs exit outdoors).
Does not claim Zelda rescue — that is a later planned hop
(``main_hall_to_zelda``). Diagnostic acceptance keys may still report
follower/cell/sanctuary for logs.

Composes after ``castle_to_sword`` / ``FighterSword`` predecessor. Clean
intervention only — no progression writes or door warps.

Measured (headless, FighterSword / natural sword predecessor):

- After fighter sword, Link is in ``kPlayerState_HoldUpItem`` ($5D==21);
  dismiss with ~95 frames of LEFT before combat works.
- Secret entrance (RAM base ``0x55``) is multi-screen. From uncle corridor
  (~2803,2680): LEFT×100 + DOWN×250 reaches the south combat chamber
  (~2680,2925) — the second chamber with guards after uncle.
- From south chamber, align stairs at ~x=2672,y=2916 then walk DOWN:
  transitions outdoors to castle grounds screen ``0x1B`` (~2248,1755).
  That is the measured **secret-entrance clear** (room finished).
- Misaligned further-south walks (~y≥2960 off-center) stay indoors in a
  stair pocket without transitioning — soft-lock risk.
- Outdoor landing is a tight hedge pocket (stairs re-entry is UP). Escape
  to main door is ``pocket_to_main_hall`` (bush-cut S/W → south corridor →
  door ~(2040,1790) → room 0x61).
- Sprite type ``0x4B`` = soldiers; type ``0x73`` at uncle = non-combat corpse.
- Green-platform chest is the secret-passage item location (not required
  for the stairs exit).

Compat: ``alttp.opening_route.sword_to_zelda`` re-exports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alttp import primitives
from alttp.opening_route.anchors import (
    ROOM_55_SOUTH_Y_MIN,
    STAIRS_ALIGN_TOLERANCE,
    STAIRS_ALIGN_X,
    STAIRS_ALIGN_Y,
)
from alttp.opening_route.castle_to_sword import dismiss_hold_up_item
from alttp.ram import (
    SECRET_PASSAGE_ROOM,
    AlttpSnapshot,
    room_label,
    snapshot_to_diag,
    zelda_rescued_accepted,
)
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import action_for, no_action, snapshot_env, step_frames

# Re-export measured stairs constants (single source: anchors.py).
__all__ = [
    "STAIRS_ALIGN_TOLERANCE",
    "STAIRS_ALIGN_X",
    "STAIRS_ALIGN_Y",
    "SOUTH_CHAMBER_Y_MAX",
    "STAIRS_EXIT_MAX_FRAMES",
    "SWORD_TO_SOUTH_CHAMBER_SCRIPT",
    "SwordToZeldaResult",
    "SecretEntranceClearResult",
    "approach_south_chamber",
    "ensure_sword_control",
    "evaluate_acceptance",
    "exit_secret_entrance_stairs",
    "left_secret_entrance",
    "run_from_state",
    "run_from_sword",
]

# Measured: uncle corridor → south combat chamber (stay above stair pocket).
SWORD_TO_SOUTH_CHAMBER_SCRIPT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("LEFT",), 100),
    (("DOWN",), 250),
)

# Soft y cap: deeper south without stair alignment fails to transition.
SOUTH_CHAMBER_Y_MAX = 2965

STAIRS_EXIT_MAX_FRAMES = 320


@dataclass
class SecretEntranceClearResult(SegmentResult):
    """Segment result from fighter-sword predecessor through stairs exit."""

    def to_report(self, kind: str = "alttp_sword_to_zelda_report") -> dict[str, Any]:
        return super().to_report(kind)


# Compat alias — older call sites / tests expect SwordToZeldaResult.
SwordToZeldaResult = SecretEntranceClearResult


def left_secret_entrance(snapshot: AlttpSnapshot) -> bool:
    """True when Link is no longer indoors in the secret-entrance room."""
    if not snapshot.indoors:
        return True
    return snapshot.room_base_id != SECRET_PASSAGE_ROOM


def evaluate_acceptance(snapshot: AlttpSnapshot) -> dict[str, bool]:
    """Acceptance keys for secret-entrance clear (+ diagnostic Zelda keys).

    Segment ``ok`` uses ``left_secret_entrance`` only. ``zelda_follower`` /
    cell / sanctuary remain diagnostic and must not alone mark segment success.
    """
    return {
        "fighter_sword_ram": snapshot.has_fighter_sword,
        "in_secret_passage": snapshot.in_secret_passage,
        "hold_up_cleared": not snapshot.is_hold_up_item,
        "left_secret_entrance": left_secret_entrance(snapshot),
        "zelda_follower": zelda_rescued_accepted(snapshot),
        "in_zelda_cell": snapshot.in_zelda_cell,
        "in_sanctuary": snapshot.in_sanctuary,
    }


def ensure_sword_control(env: object) -> RoutePhaseResult:
    """Dismiss hold-up-item and require fighter sword + control."""
    # LEFT dismiss first: primitives.settle_control waits for hold-up clear
    # but only advances no_action/text, so active LEFT is required.
    frames = dismiss_hold_up_item(env)
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
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
            "sword equipped, hold-up cleared, controllable in secret entrance"
            if ok
            else (
                f"sword={snap.has_fighter_sword} hold_up={snap.is_hold_up_item} "
                f"control={snap.has_control} room={room_label(snap.room_base_id)}"
            )
        ),
        diag=snapshot_to_diag(snap),
    )


def approach_south_chamber(env: object) -> RoutePhaseResult:
    """Walk from uncle corridor into the south multi-screen combat chamber."""
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if not start.in_secret_passage:
        return RoutePhaseResult(
            phase="approach_south_chamber",
            ok=False,
            frames=frames,
            snapshot=start,
            detail="not in secret entrance",
            diag=snapshot_to_diag(start),
        )

    script = primitives.run_script(env, SWORD_TO_SOUTH_CHAMBER_SCRIPT)
    frames += script.frames
    settle = primitives.settle_control(env)
    frames += settle.frames
    snap = settle.snapshot
    # Success: still indoors secret entrance and clearly south of uncle y.
    ok = (
        snap.in_secret_passage
        and snap.link_y >= ROOM_55_SOUTH_Y_MIN
        and snap.link_y <= SOUTH_CHAMBER_Y_MAX + 20
    )
    return RoutePhaseResult(
        phase="approach_south_chamber",
        ok=ok,
        frames=frames,
        snapshot=snap,
        detail=(
            f"south chamber (guards) xy=({snap.link_x},{snap.link_y})"
            if ok
            else f"missed south chamber xy=({snap.link_x},{snap.link_y})"
        ),
        diag=snapshot_to_diag(snap),
    )


def exit_secret_entrance_stairs(env: object) -> RoutePhaseResult:
    """Align south-chamber stairs and walk DOWN until outdoors.

    Measured: x≈2672 at y≈2916, then DOWN. Off-center deep south soft-locks
    indoors without transitioning.
    """
    frames = 0
    settle = primitives.settle_control(env)
    frames += settle.frames
    start = settle.snapshot
    if left_secret_entrance(start):
        return RoutePhaseResult(
            phase="exit_secret_entrance_stairs",
            ok=True,
            frames=frames,
            snapshot=start,
            detail="already left secret entrance",
            diag=snapshot_to_diag(start),
        )
    if not start.in_secret_passage:
        return RoutePhaseResult(
            phase="exit_secret_entrance_stairs",
            ok=False,
            frames=frames,
            snapshot=start,
            detail=f"not in secret entrance ({room_label(start.room_base_id)})",
            diag=snapshot_to_diag(start),
        )

    # Nudge north if already too deep (stair pocket without alignment).
    if start.link_y > SOUTH_CHAMBER_Y_MAX:
        up = primitives.run_script(env, ((("UP",), 40),))
        frames += up.frames

    align = primitives.move_to(
        env,
        primitives.Waypoint(
            STAIRS_ALIGN_X,
            STAIRS_ALIGN_Y,
            tolerance=STAIRS_ALIGN_TOLERANCE,
            room=SECRET_PASSAGE_ROOM,
            label="stairs_align",
        ),
        max_frames=500,
    )
    frames += align.frames
    if left_secret_entrance(align.snapshot):
        return RoutePhaseResult(
            phase="exit_secret_entrance_stairs",
            ok=True,
            frames=frames,
            snapshot=align.snapshot,
            detail=(
                f"left during align xy=({align.snapshot.link_x},"
                f"{align.snapshot.link_y})"
            ),
            diag=snapshot_to_diag(align.snapshot),
        )

    # Walk down; wait through door transition modules.
    walked = 0
    while walked < STAIRS_EXIT_MAX_FRAMES:
        step_frames(env, action_for("DOWN"), 2)
        walked += 2
        frames += 2
        snap = snapshot_env(env)
        if left_secret_entrance(snap):
            # Settle outdoor control (mode 0x09 overworld; 0x10 is mid-transition).
            for _ in range(120):
                s2 = snapshot_env(env)
                if s2.has_control and not s2.is_text_mode and not s2.indoors:
                    break
                step_frames(env, no_action(), 2)
                frames += 2
            snap = snapshot_env(env)
            return RoutePhaseResult(
                phase="exit_secret_entrance_stairs",
                ok=True,
                frames=frames,
                snapshot=snap,
                detail=(
                    f"exited secret entrance → outdoors "
                    f"screen=0x{snap.screen_id:02X} "
                    f"xy=({snap.link_x},{snap.link_y})"
                ),
                diag=snapshot_to_diag(snap),
            )
        # Transition in progress: idle through submodule animation.
        if snap.submodule != 0 or snap.game_mode not in (0x07, 0x09):
            for _ in range(40):
                step_frames(env, no_action(), 2)
                frames += 2
                snap = snapshot_env(env)
                if left_secret_entrance(snap):
                    for _ in range(120):
                        s2 = snapshot_env(env)
                        if s2.has_control and not s2.indoors:
                            break
                        step_frames(env, no_action(), 2)
                        frames += 2
                    snap = snapshot_env(env)
                    return RoutePhaseResult(
                        phase="exit_secret_entrance_stairs",
                        ok=True,
                        frames=frames,
                        snapshot=snap,
                        detail=(
                            f"exited secret entrance (transition) "
                            f"screen=0x{snap.screen_id:02X} "
                            f"xy=({snap.link_x},{snap.link_y})"
                        ),
                        diag=snapshot_to_diag(snap),
                    )
                if snap.has_control and snap.in_secret_passage:
                    break

    snap = snapshot_env(env)
    return RoutePhaseResult(
        phase="exit_secret_entrance_stairs",
        ok=False,
        frames=frames,
        snapshot=snap,
        detail=(
            f"stairs exit timeout xy=({snap.link_x},{snap.link_y}) "
            f"room={room_label(snap.room_base_id)} indoors={snap.indoors}"
        ),
        diag=snapshot_to_diag(snap),
    )


def run_from_sword(
    env: object,
    *,
    source: str = "state_load_dev",
    try_south: bool = True,
    try_exit: bool = True,
) -> SecretEntranceClearResult:
    """Run post-sword secret-entrance clear assuming fighter sword obtained.

    Default path: hold-up clear → south combat chamber → stairs exit outdoors.
    Segment ``ok`` means ``left_secret_entrance`` only (not Zelda).
    """
    phases: list[RoutePhaseResult] = []
    total = 0
    notes: list[str] = [
        "Secret-entrance clear = stairs exit outdoors (screen 0x1B pocket).",
        "Next hop: pocket_to_main_hall (bush-cut → door → room 0x61).",
        "Do not claim Zelda rescue until follower_indicator==1.",
    ]

    ready = ensure_sword_control(env)
    phases.append(ready)
    total += ready.frames
    if not ready.ok:
        acc = evaluate_acceptance(ready.snapshot)
        return SecretEntranceClearResult(
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
        if not south.ok:
            acc = evaluate_acceptance(south.snapshot)
            return SecretEntranceClearResult(
                ok=False,
                phase=south.phase,
                frames=total,
                snapshot=south.snapshot,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker=south.detail,
                notes=notes,
            )

    if try_exit:
        exit_phase = exit_secret_entrance_stairs(env)
        phases.append(exit_phase)
        total += exit_phase.frames
        snap = exit_phase.snapshot
        acc = evaluate_acceptance(snap)
        if not exit_phase.ok:
            return SecretEntranceClearResult(
                ok=False,
                phase=exit_phase.phase,
                frames=total,
                snapshot=snap,
                phases=phases,
                source=source,
                acceptance=acc,
                blocker=exit_phase.detail,
                notes=notes,
            )
    else:
        snap = snapshot_env(env)
        acc = evaluate_acceptance(snap)

    # Success only for secret-entrance clear (outdoors). Zelda keys are diagnostic.
    if acc["left_secret_entrance"]:
        return SecretEntranceClearResult(
            ok=True,
            phase="secret_entrance_exited",
            frames=total,
            snapshot=snap,
            phases=phases,
            source=source,
            acceptance=acc,
            blocker="",
            notes=notes
            + [
                "Secret entrance finished (outdoors). Next: "
                "pocket_to_main_hall → B1 → Zelda."
            ],
        )

    return SecretEntranceClearResult(
        ok=False,
        phase="south_chamber",
        frames=total,
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acc,
        blocker=(
            "still in secret entrance after south chamber; stairs exit failed "
            f"xy=({snap.link_x},{snap.link_y}) room={room_label(snap.room_base_id)}"
        ),
        notes=notes,
    )


def run_from_state(
    state_name: str = "FighterSword",
    *,
    close: bool = True,
) -> SecretEntranceClearResult:
    """Development diagnostic from a saved fighter-sword state."""
    from alttp.startup import build_boot_env

    env = build_boot_env(state_name)
    try:
        env.reset()  # type: ignore[attr-defined]
        primitives.settle_control(env)
        return run_from_sword(env, source="state_load_dev")
    finally:
        if close:
            env.close()  # type: ignore[attr-defined]
