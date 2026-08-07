"""Power-on → first controllable frame on Japanese 1.0 ALttP.

Stops at the earliest real play (Link's House wake / indoor or OW module
with control) — not full castle routing. Automated boot may perform name
entry once; the resulting ``FirstPlay.state`` skips menus for ``./play``.

Strategy:
1. Try ``alttp.startup`` helpers (SRAM inject + US name-entry scripts).
2. If that fails on JP (name entry / menus differ), mash START/A/B until
   ``alttp.ram.AlttpSnapshot.has_control`` (module overworld/dungeon).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from alttp.ram import AlttpSnapshot, LINKS_HOUSE_ROOM, read_snapshot
from alttp_rando.paths import (
    FIRST_PLAY_STATE,
    GAME_DIR,
    INTEGRATION,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
)
from retro_harness.env import make_env, write_state_bytes
from retro_harness.snes import idle_action, snes_action

DEFAULT_MAX_FRAMES = 12_000
DEFAULT_PRE_START_IDLE = 90
DEFAULT_PULSE_HOLD = 6
DEFAULT_PULSE_GAP = 28


BootMethod = Literal["alttp_startup", "mash", "unknown"]


@dataclass(frozen=True)
class BootResult:
    """Outcome of power-on → first controllable frame."""

    ok: bool
    frames: int
    snapshot: AlttpSnapshot
    method: BootMethod = "unknown"
    detail: str = ""
    state_path: str | None = None
    png_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        snap = self.snapshot
        return {
            "ok": self.ok,
            "frames": self.frames,
            "method": self.method,
            "detail": self.detail,
            "state_path": self.state_path,
            "png_path": self.png_path,
            "game_mode": snap.game_mode,
            "submodule": snap.submodule,
            "room_id": snap.room_id,
            "room_base_id": snap.room_base_id,
            "indoors": snap.indoors,
            "screen_id": snap.screen_id,
            "link_x": snap.link_x,
            "link_y": snap.link_y,
            "has_control": snap.has_control,
            "in_links_house": (
                snap.indoors and snap.room_base_id == (LINKS_HOUSE_ROOM & 0xFF)
            ),
        }


def make_boot_env(*, render_mode: str | None = "rgb_array") -> Any:
    """Open ALTTPRando-Snes from power-on (no save state)."""
    return make_env(INTEGRATION, None, GAME_DIR, render_mode=render_mode)


def _idle() -> np.ndarray:
    return idle_action(dtype=np.int8)


def _btn(*buttons: str) -> np.ndarray:
    return snes_action(*buttons, dtype=np.int8)


def _snap(env: Any) -> AlttpSnapshot:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    return read_snapshot(ram)


def _step(env: Any, action: np.ndarray, frames: int = 1) -> None:
    for _ in range(max(0, frames)):
        env.step(action)


def _is_play_module(snap: AlttpSnapshot) -> bool:
    """True when module is dungeon (0x07) or overworld (0x09)."""
    return snap.game_mode in (0x07, 0x09)


def _try_alttp_startup(env: Any, *, max_frames: int) -> BootResult | None:
    """Reuse alttp.startup through first control; return None on hard failure."""
    try:
        from alttp.startup import (
            build_blank_sram,
            create_fresh_profile,
            inject_sram_into_env,
            load_fresh_profile_slot_one,
            open_file_select,
            wait_for_control,
            wait_for_title_screen,
        )
    except Exception as exc:  # pragma: no cover - import guard
        return BootResult(
            ok=False,
            frames=0,
            snapshot=_snap(env),
            method="alttp_startup",
            detail=f"import failed: {exc}",
        )

    frames = 0
    try:
        title = wait_for_title_screen(env, max_frames=min(3600, max_frames))
        frames += title.frames
        if title.phase != "title":
            return BootResult(
                ok=False,
                frames=frames,
                snapshot=title.snapshot,
                method="alttp_startup",
                detail=(
                    f"never reached title (mode={title.snapshot.game_mode:#04x})"
                ),
            )

        inject_sram_into_env(env, build_blank_sram())
        _step(env, _idle(), 2)
        frames += 2

        open_file_select(env)
        frames += 125
        create_fresh_profile(env)
        frames += 200  # approximate name-entry script length

        # load_fresh_profile_slot_one already waits for control.
        load_fresh_profile_slot_one(env)
        snap = _snap(env)
        # Account frames spent waiting inside wait_for_control by re-polling budget.
        if snap.has_control and _is_play_module(snap):
            return BootResult(
                ok=True,
                frames=frames,
                snapshot=snap,
                method="alttp_startup",
                detail=(
                    f"control mode=0x{snap.game_mode:02X} "
                    f"room=0x{snap.room_base_id:02X} indoors={snap.indoors} "
                    f"xy=({snap.link_x},{snap.link_y})"
                ),
            )

        # Extra wait if name entry partially worked.
        waited = wait_for_control(env, max_cycles=200)
        frames += waited.frames
        snap = waited.snapshot
        if snap.has_control and _is_play_module(snap):
            return BootResult(
                ok=True,
                frames=frames,
                snapshot=snap,
                method="alttp_startup",
                detail=(
                    f"control after wait mode=0x{snap.game_mode:02X} "
                    f"room=0x{snap.room_base_id:02X}"
                ),
            )
        return BootResult(
            ok=False,
            frames=frames,
            snapshot=snap,
            method="alttp_startup",
            detail=(
                f"no control after profile load "
                f"mode=0x{snap.game_mode:02X} sub=0x{snap.submodule:02X}"
            ),
        )
    except Exception as exc:
        return BootResult(
            ok=False,
            frames=frames,
            snapshot=_snap(env),
            method="alttp_startup",
            detail=f"exception: {type(exc).__name__}: {exc}",
        )


def _mash_to_control(
    env: Any,
    *,
    max_frames: int,
    pre_start_idle: int,
    pulse_hold: int,
    pulse_gap: int,
    start_frame: int = 0,
) -> BootResult:
    """Mash START/A (and B on text) until has_control on a play module."""
    idle = _idle()
    start = _btn("START")
    a_btn = _btn("A")
    b_btn = _btn("B")
    frame = start_frame
    last = _snap(env)
    period = pulse_hold + pulse_gap

    while frame < max_frames:
        last = _snap(env)
        if last.has_control and _is_play_module(last):
            return BootResult(
                ok=True,
                frames=frame,
                snapshot=last,
                method="mash",
                detail=(
                    f"control mode=0x{last.game_mode:02X} "
                    f"room=0x{last.room_base_id:02X} indoors={last.indoors} "
                    f"screen=0x{last.screen_id:02X} "
                    f"xy=({last.link_x},{last.link_y})"
                ),
            )

        if frame < pre_start_idle:
            env.step(idle)
        else:
            phase = (frame - pre_start_idle) % period
            # Cycle: START pulse → gap → A pulse → gap → B on text modes.
            cycle = ((frame - pre_start_idle) // period) % 4
            if phase < pulse_hold:
                if last.is_text_mode:
                    env.step(b_btn if cycle % 2 else a_btn)
                elif cycle == 0:
                    env.step(start)
                elif cycle == 1:
                    env.step(a_btn)
                elif cycle == 2:
                    env.step(start)
                else:
                    env.step(a_btn)
            else:
                env.step(idle)
        frame += 1

    return BootResult(
        ok=False,
        frames=frame,
        snapshot=last,
        method="mash",
        detail=(
            f"timeout after {max_frames} frames; "
            f"mode=0x{last.game_mode:02X} sub=0x{last.submodule:02X} "
            f"room=0x{last.room_base_id:02X}"
        ),
    )


def boot_to_controllable(
    env: Any | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    pre_start_idle: int = DEFAULT_PRE_START_IDLE,
    pulse_hold: int = DEFAULT_PULSE_HOLD,
    pulse_gap: int = DEFAULT_PULSE_GAP,
    close: bool = False,
    render_mode: str | None = "rgb_array",
    prefer_alttp_startup: bool = True,
) -> BootResult:
    """Run power-on through menus until first ``has_control`` play frame."""
    owns_env = env is None
    if env is None:
        env = make_boot_env(render_mode=render_mode)
        env.reset()
    elif not hasattr(env, "get_ram"):
        raise TypeError("env must provide get_ram/step (stable-retro RetroEnv)")

    try:
        # Already controllable (e.g. re-entry after state load)?
        snap0 = _snap(env)
        if snap0.has_control and _is_play_module(snap0):
            return BootResult(
                ok=True,
                frames=0,
                snapshot=snap0,
                method="unknown",
                detail="already controllable",
            )

        if prefer_alttp_startup:
            # Clone-less try: if alttp path fails, re-power and mash.
            # Working on the same env after a failed name entry is unreliable,
            # so on failure we reset the emulator and fall back to mash.
            attempt = _try_alttp_startup(env, max_frames=max_frames)
            if attempt is not None and attempt.ok:
                return attempt
            # Reset to power-on for clean mash path.
            env.reset()
            detail_prefix = ""
            if attempt is not None:
                detail_prefix = f"alttp_startup failed ({attempt.detail}); "
            mash = _mash_to_control(
                env,
                max_frames=max_frames,
                pre_start_idle=pre_start_idle,
                pulse_hold=pulse_hold,
                pulse_gap=pulse_gap,
            )
            if detail_prefix and not mash.ok:
                return BootResult(
                    ok=False,
                    frames=mash.frames,
                    snapshot=mash.snapshot,
                    method="mash",
                    detail=detail_prefix + mash.detail,
                )
            if detail_prefix and mash.ok:
                return BootResult(
                    ok=True,
                    frames=mash.frames,
                    snapshot=mash.snapshot,
                    method="mash",
                    detail=detail_prefix + mash.detail,
                )
            return mash

        return _mash_to_control(
            env,
            max_frames=max_frames,
            pre_start_idle=pre_start_idle,
            pulse_hold=pulse_hold,
            pulse_gap=pulse_gap,
        )
    finally:
        if owns_env and close:
            env.close()


def create_first_play_state(
    *,
    state_name: str = FIRST_PLAY_STATE,
    render_mode: str | None = "rgb_array",
    save_png: bool = True,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> BootResult:
    """Boot to first control and write ``FirstPlay.state`` (+ optional PNG)."""
    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    env = make_boot_env(render_mode=render_mode)
    try:
        env.reset()
        result = boot_to_controllable(
            env, max_frames=max_frames, close=False, render_mode=render_mode
        )
        if not result.ok:
            return result

        state_bytes = env.em.get_state()
        state_path = INTEGRATION_DIR / f"{state_name}.state"
        write_state_bytes(state_path, state_bytes)

        png_path: Path | None = None
        if save_png:
            obs = env.render()
            if obs is None:
                # Step once idle to force a frame if needed.
                _step(env, _idle(), 1)
                obs = env.render()
            if obs is not None:
                png_path = RECORDINGS_DIR / f"{state_name}.png"
                Image.fromarray(np.asarray(obs)).save(png_path)
                # Also drop a copy next to the state for quick inspection.
                side = INTEGRATION_DIR / f"{state_name}.png"
                Image.fromarray(np.asarray(obs)).save(side)

        return BootResult(
            ok=True,
            frames=result.frames,
            snapshot=result.snapshot,
            method=result.method,
            detail=result.detail,
            state_path=str(state_path),
            png_path=str(png_path) if png_path else None,
        )
    finally:
        env.close()


def ensure_first_play_state(*, rebuild: bool = False) -> Path:
    """Return path to FirstPlay.state, creating it if missing (or rebuild)."""
    path = INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state"
    if path.is_file() and not rebuild:
        return path
    # Ensure ROM is wired before boot.
    rom = INTEGRATION_DIR / "rom.sfc"
    if not rom.exists() and not rom.is_symlink():
        from alttp_rando.scripts.setup_rom import main as setup_main

        rc = setup_main()
        if rc != 0:
            raise RuntimeError("setup_rom failed; cannot create FirstPlay.state")
    result = create_first_play_state()
    if not result.ok:
        raise RuntimeError(f"boot failed: {result.detail}")
    return path


__all__ = [
    "BootResult",
    "boot_to_controllable",
    "create_first_play_state",
    "ensure_first_play_state",
    "make_boot_env",
    "FIRST_PLAY_STATE",
]
