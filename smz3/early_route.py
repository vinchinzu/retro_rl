"""Early SM segment on combo ROM: power-on → Landing Site → Parlor.

Natural entry only (no save states, no door warps). Room timeout watchdog
uses provisional baselines (3× → game over).

Verified on test seed 1337 (2026-07-29): leave Landing Site bottom-left blue
door into Parlor (``0x92FD``) with controllable settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from retro_harness.snes import idle_action, snes_action
from smz3.boot import LANDING_SITE_ROOM_ID, boot_to_controllable, make_boot_env
from smz3.portals import EARLY_SM_ROOMS, room_name
from smz3.ram import ComboSnapshot, snapshot_env
from smz3.room_timeout import RoomTimeoutWatchdog, TimeoutEvent
from smz3.world import ActiveWorld, detect_world

PARLOR_ROOM_ID = 0x92FD

# Provisional standard dwells (frames @ 60fps) for early rooms.
EARLY_ROOM_BASELINES: dict[str, int] = {
    "0x91F8": 90 * 60,  # Landing Site — large, ship settle
    "0x92FD": 60 * 60,  # Parlor
    "0x98E2": 30 * 60,  # Pre-Map Flyway
    "0x9994": 20 * 60,  # Crateria Map Room
}


@dataclass
class RoomVisit:
    room_id: int
    enter_frame: int
    leave_frame: int | None = None
    world: str = "super_metroid"

    @property
    def dwell_frames(self) -> int | None:
        if self.leave_frame is None:
            return None
        return self.leave_frame - self.enter_frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "room_name": room_name(self.room_id),
            "enter_frame": self.enter_frame,
            "leave_frame": self.leave_frame,
            "dwell_frames": self.dwell_frames,
            "world": self.world,
        }


@dataclass
class EarlySegmentResult:
    ok: bool
    goal: str
    frames: int
    boot_frames: int
    visits: list[RoomVisit] = field(default_factory=list)
    final_snapshot: ComboSnapshot | None = None
    world: ActiveWorld = ActiveWorld.UNKNOWN
    timeout: TimeoutEvent | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "goal": self.goal,
            "frames": self.frames,
            "boot_frames": self.boot_frames,
            "world": self.world.value,
            "detail": self.detail,
            "visits": [v.to_dict() for v in self.visits],
            "timeout": self.timeout.to_dict() if self.timeout else None,
            "final_snapshot": (
                self.final_snapshot.to_dict() if self.final_snapshot else None
            ),
            "reached_parlor": any(v.room_id == PARLOR_ROOM_ID for v in self.visits),
            "room_names": [room_name(v.room_id) for v in self.visits],
        }


def _hold(env: Any, *buttons: str, frames: int) -> None:
    action = snes_action(*buttons, dtype=np.int8) if buttons else idle_action(dtype=np.int8)
    for _ in range(max(0, frames)):
        env.step(action)


def _track_room(
    visits: list[RoomVisit],
    snap: ComboSnapshot,
    world: ActiveWorld,
) -> None:
    rid = snap.sm_room_id
    if rid == 0:
        return
    if not visits or visits[-1].room_id != rid:
        if visits and visits[-1].leave_frame is None:
            visits[-1].leave_frame = snap.frame
        visits.append(
            RoomVisit(
                room_id=rid,
                enter_frame=snap.frame,
                world=world.value,
            )
        )


def leave_landing_site_to_parlor(
    env: Any,
    *,
    start_frame: int = 0,
    on_frame: Callable[[int, ComboSnapshot], None] | None = None,
) -> tuple[int, ComboSnapshot]:
    """From controllable Landing Site, exit bottom-left blue door to Parlor.

    Returns ``(frame, snapshot)`` after Parlor controllable settle (or last snap).
    """
    frame = start_frame
    idle = idle_action(dtype=np.int8)

    def step(buttons: tuple[str, ...] | None, n: int) -> ComboSnapshot:
        nonlocal frame
        action = (
            snes_action(*buttons, dtype=np.int8)
            if buttons
            else idle
        )
        for _ in range(n):
            env.step(action)
            frame += 1
        snap = snapshot_env(env, frame=frame)
        if on_frame is not None:
            on_frame(frame, snap)
        return snap

    # Ship/spawn settle: early frames often ignore input (pose 0).
    snap = step(None, 60)
    # Drop/walk toward bottom-left door (block [0,71]).
    for _ in range(10):
        step(("LEFT",), 40)
        step(("LEFT", "A"), 10)
        step(None, 5)
        if snapshot_env(env, frame=frame).sm_samus_x < 400:
            break
    step(("LEFT",), 150)

    # Open blue door (shoot) and enter.
    for _ in range(10):
        snap = step(("LEFT", "X"), 25)
        snap = step(("LEFT",), 40)
        if snap.sm_room_id == PARLOR_ROOM_ID or snap.sm_game_state == 11:
            break
    for _ in range(250):
        snap = step(("LEFT",), 1)
        if snap.sm_room_id == PARLOR_ROOM_ID:
            break
    for _ in range(360):
        snap = step(None, 1)
        if snap.sm_controllable and snap.sm_room_id == PARLOR_ROOM_ID:
            break
    return frame, snapshot_env(env, frame=frame)


def run_landing_to_parlor(
    env: Any | None = None,
    *,
    close: bool = False,
    max_frames: int = 12_000,
    room_timeout_multiplier: float = 3.0,
) -> EarlySegmentResult:
    """Power-on → first controllable → Landing Site → Parlor with timeout."""
    owns = env is None
    if env is None:
        env = make_boot_env(render_mode="rgb_array")
        env.reset()

    watchdog = RoomTimeoutWatchdog.from_mapping(
        EARLY_ROOM_BASELINES,
        multiplier=room_timeout_multiplier,
        source="early_route_provisional",
    )
    visits: list[RoomVisit] = []
    timeout_event: TimeoutEvent | None = None

    def on_frame(frame: int, snap: ComboSnapshot) -> None:
        nonlocal timeout_event
        world = detect_world(snap)
        _track_room(visits, snap, world)
        if snap.sm_controllable:
            key = f"0x{snap.sm_room_id:04X}"
            ev = watchdog.observe(
                frame=frame,
                room_key=key,
                settled=True,
            )
            if ev is not None:
                timeout_event = ev

    try:
        boot = boot_to_controllable(env, close=False, max_frames=max_frames)
        if not boot.ok:
            return EarlySegmentResult(
                ok=False,
                goal="landing_to_parlor",
                frames=boot.frames,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=boot.snapshot,
                world=boot.world,
                detail=f"boot failed: {boot.detail}",
            )

        on_frame(boot.frames, boot.snapshot)
        frame, snap = leave_landing_site_to_parlor(
            env,
            start_frame=boot.frames,
            on_frame=on_frame,
        )
        world = detect_world(snap)
        if visits and visits[-1].leave_frame is None:
            visits[-1].leave_frame = frame

        if timeout_event is not None:
            return EarlySegmentResult(
                ok=False,
                goal="landing_to_parlor",
                frames=frame,
                boot_frames=boot.frames,
                visits=visits,
                final_snapshot=snap,
                world=world,
                timeout=timeout_event,
                detail="room timeout game over",
            )

        ok = (
            snap.sm_room_id == PARLOR_ROOM_ID
            and snap.sm_controllable
            and world is ActiveWorld.SUPER_METROID
        )
        return EarlySegmentResult(
            ok=ok,
            goal="landing_to_parlor",
            frames=frame,
            boot_frames=boot.frames,
            visits=visits,
            final_snapshot=snap,
            world=world,
            detail=(
                f"parlor controllable xy=({snap.sm_samus_x},{snap.sm_samus_y})"
                if ok
                else (
                    f"missed parlor: room=0x{snap.sm_room_id:04X} "
                    f"gs={snap.sm_game_state} ctrl={snap.sm_controllable}"
                )
            ),
        )
    finally:
        if owns and close:
            env.close()


def default_baselines() -> dict[str, int]:
    return dict(EARLY_ROOM_BASELINES)
