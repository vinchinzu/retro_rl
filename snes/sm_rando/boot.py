"""Power-on → first RAM-verified controllable frame on SM Rando (vanilla ROM).

Vanilla Super Metroid boots into **Ceres** after file select / intro
(``room_id = 0xDF45``, ``game_state == 8``), not Landing Site. SMZ3 combo seeds
start at Landing Site; this package uses the vanilla dump until a real rando
generator is wired.

Pattern mirrors ``smz3.boot`` (idle then START/A pulses until ordinary
gameplay). Button choice is START+A because vanilla menus respond to either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, save_state
from retro_harness.snes import idle_action, snes_action
from sm_rando.paths import (
    FIRST_PLAY_STATE,
    GAME,
    GAME_DIR,
    INTEGRATION,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
)

# Prefer super_metroid.ram when importable; fall back to known WRAM offsets.
try:
    from super_metroid.ram import (  # type: ignore[import-not-found]
        ADDR_DOOR_TRANSITION,
        ADDR_GAME_STATE,
        ADDR_HEALTH,
        ADDR_ROOM_ID,
        ADDR_SAMUS_X,
        ADDR_SAMUS_Y,
    )
except Exception:  # pragma: no cover - offline / partial import path
    ADDR_ROOM_ID = 0x079B
    ADDR_DOOR_TRANSITION = 0x0797
    ADDR_GAME_STATE = 0x0998
    ADDR_HEALTH = 0x09C2
    ADDR_SAMUS_X = 0x0AF6
    ADDR_SAMUS_Y = 0x0AFA

# Idle through Nintendo logo / early boot before START/A is useful.
DEFAULT_PRE_START_IDLE = 180
# Pulse confirm this many frames, then gap (title / file / name / intro).
DEFAULT_START_HOLD = 8
DEFAULT_START_GAP = 28
# Vanilla intro is long; hard stop so a hung boot fails cleanly.
DEFAULT_MAX_FRAMES = 15000

# Ordinary controllable gameplay (same as super_metroid / smz3).
ORDINARY_GAME_STATE = 8
# Ceres Elevator — vanilla first controllable room after file select.
CERES_ELEVATOR_ROOM_ID = 0xDF45


def _u16(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


@dataclass(frozen=True)
class BootSnapshot:
    """Minimal SM WRAM fields from one boot frame."""

    frame: int
    game_state: int
    room_id: int
    door_transition: int
    health: int
    samus_x: int
    samus_y: int

    @property
    def controllable(self) -> bool:
        """True when ordinary gameplay is settled (game_state 8, no door)."""
        return (
            self.game_state == ORDINARY_GAME_STATE
            and self.door_transition == 0
            and self.room_id != 0
            and self.health > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "game_state": self.game_state,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "door_transition": self.door_transition,
            "health": self.health,
            "samus_x": self.samus_x,
            "samus_y": self.samus_y,
            "controllable": self.controllable,
            "ceres_elevator": self.room_id == CERES_ELEVATOR_ROOM_ID,
        }


@dataclass(frozen=True)
class BootResult:
    """Outcome of power-on → first controllable SM frame."""

    ok: bool
    frames: int
    room_id: int
    game_state: int
    detail: str = ""
    snapshot: BootSnapshot | None = None
    state_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "frames": self.frames,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "game_state": self.game_state,
            "detail": self.detail,
            "state_path": self.state_path,
            "ceres_elevator": self.room_id == CERES_ELEVATOR_ROOM_ID,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
        }


def make_boot_env(*, render_mode: str | None = "rgb_array") -> Any:
    """Open SMRando-Snes from power-on (no save state)."""
    return make_env(INTEGRATION, None, GAME_DIR, render_mode=render_mode)


def snapshot_env(env: Any, *, frame: int = 0) -> BootSnapshot:
    """Read a compact boot snapshot from ``env.get_ram()``."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    return BootSnapshot(
        frame=frame,
        game_state=_u16(ram, ADDR_GAME_STATE),
        room_id=_u16(ram, ADDR_ROOM_ID),
        door_transition=_u16(ram, ADDR_DOOR_TRANSITION),
        health=_u16(ram, ADDR_HEALTH),
        samus_x=_u16(ram, ADDR_SAMUS_X),
        samus_y=_u16(ram, ADDR_SAMUS_Y),
    )


def _action_confirm() -> np.ndarray:
    # START + A: title/file/name/intro all accept one of these on vanilla SM.
    return snes_action("START", "A", dtype=np.int8)


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
    """Run power-on through SM file select / intro until ordinary gameplay.

    Parameters
    ----------
    env:
        Existing RetroEnv, or ``None`` to create one (closed if ``close``).
    max_frames:
        Abort after this many steps.
    pre_start_idle:
        Idle frames before the first confirm pulse (logo / boot).
    start_hold / start_gap:
        Confirm pulse pattern used on title, file select, and intro.
    """
    owns_env = env is None
    if env is None:
        env = make_boot_env(render_mode=render_mode)
        env.reset()
    elif not hasattr(env, "get_ram"):
        raise TypeError("env must provide get_ram/step (stable-retro RetroEnv)")

    idle = _action_idle()
    confirm = _action_confirm()
    frame = 0
    last = snapshot_env(env, frame=0)

    try:
        while frame < max_frames:
            last = snapshot_env(env, frame=frame)
            if last.controllable:
                return BootResult(
                    ok=True,
                    frames=frame,
                    room_id=last.room_id,
                    game_state=last.game_state,
                    detail=(
                        f"controllable room=0x{last.room_id:04X} "
                        f"hp={last.health} xy=({last.samus_x},{last.samus_y})"
                    ),
                    snapshot=last,
                )

            if frame < pre_start_idle:
                env.step(idle)
            else:
                phase = (frame - pre_start_idle) % (start_hold + start_gap)
                env.step(confirm if phase < start_hold else idle)
            frame += 1

        return BootResult(
            ok=False,
            frames=frame,
            room_id=last.room_id,
            game_state=last.game_state,
            detail=(
                f"timeout after {max_frames} frames; "
                f"gs={last.game_state} room=0x{last.room_id:04X}"
            ),
            snapshot=last,
        )
    finally:
        if owns_env and close:
            env.close()


def create_first_play_state(
    *,
    state_name: str = FIRST_PLAY_STATE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    save_png: bool = True,
    render_mode: str | None = "rgb_array",
) -> BootResult:
    """Boot to controllable gameplay and write ``FirstPlay.state`` (+ optional PNG)."""
    if not (INTEGRATION_DIR / "rom.sfc").exists():
        raise FileNotFoundError(
            f"Missing {INTEGRATION_DIR / 'rom.sfc'}; "
            "run: uv run python -m sm_rando.scripts.setup_rom"
        )

    env = make_boot_env(render_mode=render_mode)
    try:
        env.reset()
        result = boot_to_controllable(env, max_frames=max_frames, close=False)
        if not result.ok:
            return result

        path = save_state(env, GAME_DIR, GAME, state_name)
        png_path: Path | None = None
        if save_png:
            try:
                from PIL import Image

                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                img = env.render()
                if img is not None:
                    png_path = RECORDINGS_DIR / "boot_first_play.png"
                    Image.fromarray(np.asarray(img)).save(png_path)
            except Exception:  # pragma: no cover - optional snapshot
                png_path = None

        detail = result.detail
        if png_path is not None:
            detail = f"{detail}; png={png_path}"
        return BootResult(
            ok=True,
            frames=result.frames,
            room_id=result.room_id,
            game_state=result.game_state,
            detail=f"{detail}; state={path}",
            snapshot=result.snapshot,
            state_path=str(path),
        )
    finally:
        env.close()


def ensure_first_play_state(
    *,
    rebuild: bool = False,
    **kwargs: Any,
) -> Path:
    """Return path to ``FirstPlay.state``, creating it via boot if missing."""
    path = INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state"
    if path.is_file() and not rebuild:
        return path
    result = create_first_play_state(**kwargs)
    if not result.ok or result.state_path is None:
        raise RuntimeError(f"boot failed: {result.detail}")
    return Path(result.state_path)
