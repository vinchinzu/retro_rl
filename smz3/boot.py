"""Power-on → first RAM-verified controllable frame on the SMZ3 combo ROM.

The combo randomizer always resets into Super Metroid (see upstream
``common.asm``: ``SRAM_CURRENT_GAME = #$00FF``). Fresh file → Landing Site
(``room_id = 0x91F8``) is the verified natural entry for test seed 1337.

This module only reaches SM ordinary gameplay (``game_state == 8``). ALttP
controllable entry requires a world portal later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retro_harness.env import make_env
from retro_harness.snes import idle_action, snes_action
from smz3.paths import GAME_DIR, INTEGRATION
from smz3.ram import ComboSnapshot, snapshot_env
from smz3.world import ActiveWorld, WorldContext, context_for, detect_world


# Idle through Nintendo logo / early boot before START is useful.
DEFAULT_PRE_START_IDLE = 120
# Pulse START this many frames, then gap.
DEFAULT_START_HOLD = 6
DEFAULT_START_GAP = 34
# Hard stop so a hung boot fails cleanly.
DEFAULT_MAX_FRAMES = 3600

# Landing Site (Crateria) — SMZ3 post-file-select start for standard seeds.
LANDING_SITE_ROOM_ID = 0x91F8


@dataclass(frozen=True)
class BootResult:
    """Outcome of power-on → first controllable SM frame."""

    ok: bool
    frames: int
    snapshot: ComboSnapshot
    world: ActiveWorld
    context: WorldContext
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "frames": self.frames,
            "world": self.world.value,
            "context": self.context.to_dict(),
            "detail": self.detail,
            "snapshot": self.snapshot.to_dict(),
            "sm_controllable": self.snapshot.sm_controllable,
            "landing_site": self.snapshot.sm_room_id == LANDING_SITE_ROOM_ID,
        }


def make_boot_env(*, render_mode: str | None = "rgb_array") -> Any:
    """Open SMZ3-Snes from power-on (no save state)."""
    return make_env(INTEGRATION, None, GAME_DIR, render_mode=render_mode)


def _action_start() -> np.ndarray:
    return snes_action("START", dtype=np.int8)


def _action_idle() -> np.ndarray:
    return idle_action(dtype=np.int8)


def boot_to_controllable(
    env: Any | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    pre_start_idle: int = DEFAULT_PRE_START_IDLE,
    start_hold: int = DEFAULT_START_HOLD,
    start_gap: int = DEFAULT_START_GAP,
    close: bool = False,
    render_mode: str | None = "rgb_array",
) -> BootResult:
    """Run power-on through SM file select until ``sm_controllable``.

    Parameters
    ----------
    env:
        Existing RetroEnv, or ``None`` to create one (closed if ``close``).
    max_frames:
        Abort after this many steps.
    pre_start_idle:
        Idle frames before the first START pulse (logo / boot).
    start_hold / start_gap:
        START pulse pattern used on SM title and file select.
    """
    owns_env = env is None
    if env is None:
        env = make_boot_env(render_mode=render_mode)
        env.reset()
    elif not hasattr(env, "get_ram"):
        raise TypeError("env must provide get_ram/step (stable-retro RetroEnv)")

    idle = _action_idle()
    start = _action_start()
    frame = 0
    last = snapshot_env(env, frame=0)

    try:
        while frame < max_frames:
            last = snapshot_env(env, frame=frame)
            if last.sm_controllable:
                world = detect_world(last)
                return BootResult(
                    ok=True,
                    frames=frame,
                    snapshot=last,
                    world=world,
                    context=context_for(world),
                    detail=(
                        f"sm controllable room=0x{last.sm_room_id:04X} "
                        f"hp={last.sm_health} xy=({last.sm_samus_x},{last.sm_samus_y})"
                    ),
                )

            # Early boot: pure idle until logo settles, then START pulses.
            if frame < pre_start_idle:
                env.step(idle)
            else:
                phase = (frame - pre_start_idle) % (start_hold + start_gap)
                env.step(start if phase < start_hold else idle)
            frame += 1

        world = detect_world(last)
        return BootResult(
            ok=False,
            frames=frame,
            snapshot=last,
            world=world,
            context=context_for(world),
            detail=(
                f"timeout after {max_frames} frames; "
                f"sm_gs={last.sm_game_state} room=0x{last.sm_room_id:04X}"
            ),
        )
    finally:
        if owns_env and close:
            env.close()
