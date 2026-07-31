"""Parlor → Crateria map portal (door ``$8976``) on the SMZ3 combo ROM.

Verified findings (test seed 1337, 2026-07-30):

* Door pointer ``$8976`` is the **Parlor bottom-right red door** (block
  ``[31, 55]``). tewtal labels it "Crateria map station → Fortune teller";
  on the combo ROM entering that door **is** the SM→Z3 teleport (it does
  **not** load Pre-Map Flyway first).
* Natural path (no post-portal RAM pokes): Landing → Parlor top-right →
  left-shaft descent → red door with missiles → walk in →
  ``transition_to_zelda`` → cave ``$0122`` Fortune Teller.
* Preferred dev checkpoint: stop **at the red door still in SM**
  (``PortalRedDoor``) so you can see the room and walk the portal yourself.
* After walking in, transition stores module ``$0F`` then needs ~300 idle
  frames under stable-retro before Link is controllable (module ``$09`` OW at
  screen ``$35``). Stopping on first ``$0F`` looks like a hang. Requires ALttP
  **JP 1.0** base ROM. See ``docs/EARLY_ROOMS.md``.

Missile capacity is a **dev assist** (:mod:`smz3.assist`) until the natural
morph → first-missile leg is wired on the combo ROM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from retro_harness.snes import idle_action
from smz3.assist import MISSILE_RED_DOOR_ASSIST, grant_missiles  # re-export grant_missiles
from smz3.boot import boot_to_controllable, make_boot_env
from smz3.control import hold
from smz3.early_route import (
    EARLY_ROOM_BASELINES,
    PARLOR_ROOM_ID,
    EarlySegmentResult,
    leave_landing_site_to_parlor,
)
from smz3.portals import FORTUNE_TELLER_CAVE_ID, early_portal
from smz3.ram import ComboSnapshot, snapshot_env
from smz3.room_timeout import RoomTimeoutWatchdog, TimeoutEvent
from smz3.segment import RoomVisit, close_last_visit, track_room
from smz3.world import ActiveWorld, detect_world

# Morph-route parlor button log (vanilla Super Metroid recording).
_PARLOR_POLICY = (
    Path(__file__).resolve().parents[1]
    / "super_metroid"
    / "policies"
    / "start_to_morph"
    / "seg01_parlor.json"
)

# Red door target (pixel space; block [31, 55] × 16).
RED_DOOR_X = 480
RED_DOOR_Y_MIN = 860
RED_DOOR_Y_MAX = 930

# Dev save states under custom_integrations/SMZ3-Snes/ (gitignored *.state).
PORTAL_RED_DOOR_STATE = "PortalRedDoor"
PORTAL_RESIDUE_STATE = "PortalResidue"
PORTAL_SETTLED_STATE = "PortalSettled"

STOP_AT_RED_DOOR = "red_door"
STOP_AFTER_PORTAL = "after_portal"
STOP_CHOICES = (STOP_AT_RED_DOOR, STOP_AFTER_PORTAL)

PORTAL_SETTLE_MAX_FRAMES = 900


@dataclass
class PortalSegmentResult(EarlySegmentResult):
    """Extends early segment result with portal-phase fields."""

    portal_started: bool = False
    z3_module: int | None = None
    z3_room_id: int | None = None
    z3_settled: bool = False
    assist_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "portal_started": self.portal_started,
                "z3_module": self.z3_module,
                "z3_room_id": (
                    f"0x{self.z3_room_id:04X}" if self.z3_room_id is not None else None
                ),
                "z3_settled": self.z3_settled,
                "portal": early_portal().to_dict(),
                "assist_used": list(self.assist_used),
                "assist": (
                    [MISSILE_RED_DOOR_ASSIST.to_dict()] if self.assist_used else []
                ),
            }
        )
        return d


def load_parlor_policy_buttons() -> list[list[int]]:
    data = json.loads(_PARLOR_POLICY.read_text())
    return list(data["raw_buttons"])


def play_parlor_policy(env: Any, *, start_frame: int = 0) -> int:
    """Replay vanilla morph-route parlor segment (top-right → left shaft)."""
    frame = start_frame
    for raw in load_parlor_policy_buttons():
        env.step(np.array(raw, dtype=np.int8))
        frame += 1
    return frame


def descend_left_shaft_to_red_door(
    env: Any,
    *,
    start_frame: int = 0,
    max_iters: int = 280,
) -> tuple[int, ComboSnapshot]:
    """From upper left shaft, spin-drop into red-door Y band and align X."""
    frame = start_frame
    frame = hold(env, ("LEFT",), 50, frame=frame)

    for i in range(max_iters):
        snap = snapshot_env(env, frame=frame)
        if snap.sm_room_id != PARLOR_ROOM_ID:
            return frame, snap
        y = snap.sm_samus_y
        if RED_DOOR_Y_MIN <= y <= RED_DOOR_Y_MAX and abs(snap.sm_samus_x - RED_DOOR_X) < 40:
            break
        if y > RED_DOOR_Y_MAX + 40:
            frame = hold(env, ("A",), 10, frame=frame)
            continue
        if y >= RED_DOOR_Y_MIN and abs(snap.sm_samus_x - RED_DOOR_X) >= 40:
            if snap.sm_samus_x < RED_DOOR_X:
                frame = hold(env, ("RIGHT",), 4, frame=frame)
            else:
                frame = hold(env, ("LEFT",), 4, frame=frame)
            continue
        phase = i % 10
        if phase < 3:
            frame = hold(env, ("LEFT", "B"), 12, frame=frame)
            frame = hold(env, ("LEFT", "B", "A"), 8, frame=frame)
        elif phase < 6:
            frame = hold(env, ("RIGHT", "B"), 12, frame=frame)
            frame = hold(env, ("RIGHT", "B", "A"), 8, frame=frame)
        else:
            frame = hold(env, None, 30, frame=frame)

    for _ in range(60):
        snap = snapshot_env(env, frame=frame)
        if snap.sm_samus_y > RED_DOOR_Y_MAX + 30:
            break
        if abs(snap.sm_samus_x - RED_DOOR_X) < 16:
            break
        if snap.sm_samus_x < RED_DOOR_X:
            frame = hold(env, ("RIGHT",), 3, frame=frame)
        else:
            frame = hold(env, ("LEFT",), 3, frame=frame)

    return frame, snapshot_env(env, frame=frame)


def open_red_door_portal(
    env: Any,
    *,
    start_frame: int = 0,
    max_attempts: int = 50,
    settle_frames: int = PORTAL_SETTLE_MAX_FRAMES,
) -> tuple[int, ComboSnapshot]:
    """Shoot + walk right into parlor red door; idle until Z3 settle if possible.

    Does **not** write RAM mid-transition. Stops when Link is controllable
    (module ``$07``/``$09``), settle budget expires after portal residue, room
    leaves Parlor without teleport, or attempts are exhausted.
    """
    frame = start_frame
    for _ in range(max_attempts):
        snap = snapshot_env(env, frame=frame)
        if snap.sm_game_state != 8 or snap.sm_door_transition != 0:
            break
        if snap.z3_module == 0x0F or (snap.z3_room_id & 0xFFFF) == FORTUNE_TELLER_CAVE_ID:
            break
        if snap.sm_room_id != PARLOR_ROOM_ID:
            break
        frame = hold(env, ("RIGHT", "X"), 5, frame=frame)
        frame = hold(env, ("RIGHT",), 8, frame=frame)

    idle = idle_action(dtype=np.int8)
    saw_portal = False
    for _ in range(max(0, settle_frames)):
        env.step(idle)
        frame += 1
        snap = snapshot_env(env, frame=frame)
        if snap.z3_controllable:
            break
        if snap.z3_module == 0x0F or (snap.z3_room_id & 0xFFFF) == FORTUNE_TELLER_CAVE_ID:
            saw_portal = True
            continue
        if saw_portal:
            continue
        if snap.sm_room_id not in (0, PARLOR_ROOM_ID) and snap.sm_controllable:
            break

    return frame, snapshot_env(env, frame=frame)


def is_portal_residue(snap: ComboSnapshot) -> bool:
    """True when SM→Z3 teleport left Fortune Teller residue (not yet settled)."""
    if (snap.z3_room_id & 0xFFFF) == FORTUNE_TELLER_CAVE_ID and snap.z3_module == 0x0F:
        return True
    if snap.z3_module == 0x0F and not snap.sm_controllable:
        return True
    return False


def _at_red_door_band(snap: ComboSnapshot) -> bool:
    if snap.sm_room_id != PARLOR_ROOM_ID or not snap.sm_controllable:
        return False
    if not (RED_DOOR_Y_MIN <= snap.sm_samus_y <= RED_DOOR_Y_MAX):
        return False
    return abs(snap.sm_samus_x - RED_DOOR_X) < 80


def run_landing_to_portal(
    env: Any | None = None,
    *,
    close: bool = False,
    max_frames: int = 30_000,
    room_timeout_multiplier: float = 3.0,
    grant_missile_assist: bool = True,
    stop: str = STOP_AT_RED_DOOR,
) -> PortalSegmentResult:
    """Power-on → Parlor → red door (default) or through portal.

    Parameters
    ----------
    stop:
        ``red_door`` — still SM, aligned on red-door band (playable, not black).
        ``after_portal`` — walk into door; idle until Z3 settle (or residue timeout).
    grant_missile_assist:
        Apply :data:`smz3.assist.MISSILE_RED_DOOR_ASSIST` (SM ammo only).
    """
    if stop not in STOP_CHOICES:
        raise ValueError(f"stop must be one of {STOP_CHOICES}, got {stop!r}")

    owns = env is None
    if env is None:
        env = make_boot_env(render_mode="rgb_array")
        env.reset()

    watchdog = RoomTimeoutWatchdog.from_mapping(
        EARLY_ROOM_BASELINES,
        multiplier=room_timeout_multiplier,
        source="portal_route_provisional",
    )
    visits: list[RoomVisit] = []
    timeout_event: TimeoutEvent | None = None
    assist_used: list[str] = []
    goal = (
        "landing_to_red_door" if stop == STOP_AT_RED_DOOR else "landing_to_portal"
    )

    def on_frame(frame: int, snap: ComboSnapshot) -> None:
        nonlocal timeout_event
        world = detect_world(snap)
        track_room(visits, snap, world)
        if snap.sm_controllable:
            key = f"0x{snap.sm_room_id:04X}"
            ev = watchdog.observe(frame=frame, room_key=key, settled=True)
            if ev is not None:
                timeout_event = ev

    def apply_missile_assist() -> None:
        if grant_missile_assist:
            grant_missiles(env, count=20)
            if MISSILE_RED_DOOR_ASSIST.assist_id not in assist_used:
                assist_used.append(MISSILE_RED_DOOR_ASSIST.assist_id)

    try:
        boot = boot_to_controllable(env, close=False, max_frames=max_frames)
        if not boot.ok:
            return PortalSegmentResult(
                ok=False,
                goal=goal,
                frames=boot.frames,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=boot.snapshot,
                world=boot.world,
                detail=f"boot failed: {boot.detail}",
                assist_used=assist_used,
            )

        on_frame(boot.frames, boot.snapshot)
        frame, snap = leave_landing_site_to_parlor(
            env, start_frame=boot.frames, on_frame=on_frame
        )
        if snap.sm_room_id != PARLOR_ROOM_ID or not snap.sm_controllable:
            world = detect_world(snap)
            return PortalSegmentResult(
                ok=False,
                goal=goal,
                frames=frame,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=snap,
                world=world,
                detail=(
                    f"missed parlor: room=0x{snap.sm_room_id:04X} "
                    f"ctrl={snap.sm_controllable}"
                ),
                assist_used=assist_used,
            )

        apply_missile_assist()

        frame = play_parlor_policy(env, start_frame=frame)
        snap = snapshot_env(env, frame=frame)
        on_frame(frame, snap)

        frame, snap = descend_left_shaft_to_red_door(env, start_frame=frame)
        on_frame(frame, snap)

        apply_missile_assist()

        if stop == STOP_AT_RED_DOOR:
            world = detect_world(snap)
            close_last_visit(visits, frame)
            at_door = _at_red_door_band(snap)
            if timeout_event is not None:
                return PortalSegmentResult(
                    ok=False,
                    goal=goal,
                    frames=frame,
                    boot_frames=boot.frames,
                    visits=visits,
                    final_snapshot=snap,
                    world=world,
                    timeout=timeout_event,
                    detail="room timeout game over",
                    assist_used=assist_used,
                )
            detail = (
                f"at red door room=0x{snap.sm_room_id:04X} "
                f"xy=({snap.sm_samus_x},{snap.sm_samus_y}) "
                f"missiles_assist={grant_missile_assist}"
                if at_door
                else (
                    f"not at door band: room=0x{snap.sm_room_id:04X} "
                    f"xy=({snap.sm_samus_x},{snap.sm_samus_y})"
                )
            )
            return PortalSegmentResult(
                ok=at_door,
                goal=goal,
                frames=frame,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=snap,
                world=world,
                detail=detail,
                portal_started=False,
                z3_module=snap.z3_module,
                z3_room_id=snap.z3_room_id,
                z3_settled=False,
                assist_used=assist_used,
            )

        frame, snap = open_red_door_portal(env, start_frame=frame)
        on_frame(frame, snap)
        world = detect_world(snap)
        close_last_visit(visits, frame)

        portal_started = is_portal_residue(snap) or (
            (snap.z3_room_id & 0xFFFF) == FORTUNE_TELLER_CAVE_ID
        )
        z3_settled = bool(snap.z3_controllable and world is ActiveWorld.ALTTP)

        if timeout_event is not None:
            return PortalSegmentResult(
                ok=False,
                goal=goal,
                frames=frame,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=snap,
                world=world,
                timeout=timeout_event,
                detail="room timeout game over",
                portal_started=portal_started,
                z3_module=snap.z3_module,
                z3_room_id=snap.z3_room_id,
                z3_settled=z3_settled,
                assist_used=assist_used,
            )

        ok = portal_started or z3_settled
        if z3_settled:
            detail = (
                f"portal settled module=${snap.z3_module:02X} "
                f"screen=${snap.z3_screen_id:02X} "
                f"xy=({snap.z3_link_x},{snap.z3_link_y})"
            )
        elif portal_started:
            detail = (
                f"portal residue module=${snap.z3_module:02X} cave=$"
                f"{snap.z3_room_id:04X} settled=False "
                f"(waited settle budget)"
            )
        else:
            detail = (
                f"no portal: room=0x{snap.sm_room_id:04X} "
                f"xy=({snap.sm_samus_x},{snap.sm_samus_y}) "
                f"z3_mod=${snap.z3_module:02X}"
            )
        return PortalSegmentResult(
            ok=ok,
            goal=goal,
            frames=frame,
            boot_frames=boot.frames,
            visits=visits,
            final_snapshot=snap,
            world=world,
            detail=detail,
            portal_started=portal_started,
            z3_module=snap.z3_module,
            z3_room_id=snap.z3_room_id,
            z3_settled=z3_settled,
            assist_used=assist_used,
        )
    finally:
        if owns and close:
            env.close()
