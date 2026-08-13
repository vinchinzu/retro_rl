"""Run a short joypad tape on the pure stepper and the emulator; report R(τ)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smb.approx import idle_action, press, rollout
from smb.observation import Observation, level1_start_obs, observation_from_ram
from smb.residual import ResidualProfile, compute_residual_profile, format_profile

_IDX_A = 8

__all__ = [
    "ROM_AVAILABLE",
    "SegmentResult",
    "SEGMENTS",
    "measure_segment",
    "replay_on_emulator",
    "segment_actions",
]


def _rom_available() -> bool:
    try:
        from smb.paths import GAME_DIR, INTEGRATION_V0_DIR

        if not (INTEGRATION_V0_DIR / "Level1_1.state").exists():
            return False
        if not (INTEGRATION_V0_DIR / "rom.nes").exists() and not (
            GAME_DIR / "roms" / "Super Mario Bros..nes"
        ).exists():
            return False
        import stable_retro as retro

        return hasattr(getattr(retro.data, "Integrations", None), "CUSTOM")
    except Exception:
        return False


ROM_AVAILABLE = _rom_available()

SEGMENTS: dict[str, tuple[tuple[int, ...], ...]] = {
    "idle": tuple(idle_action() for _ in range(24)),
    "walk": tuple(press("RIGHT") for _ in range(24)),
    "jump": tuple(press("A") for _ in range(4)) + tuple(idle_action() for _ in range(20)),
    "run_jump": tuple(press("RIGHT", "B", "A") for _ in range(30)),
    "jump_to_land": tuple(press("A") for _ in range(4)) + tuple(idle_action() for _ in range(28)),
    "run_jump_to_land": tuple(press("RIGHT", "B", "A") for _ in range(60)),
    "run_then_jump": (
        tuple(press("RIGHT", "B") for _ in range(16))
        + tuple(press("RIGHT", "B", "A") for _ in range(4))
        + tuple(press("RIGHT", "B") for _ in range(16))
    ),
    # |vx| bands 2 and 4 at takeoff (smbdis InitJS $10/$1C).
    "run24_then_jump": (
        tuple(press("RIGHT", "B") for _ in range(24))
        + tuple(press("RIGHT", "B", "A") for _ in range(4))
        + tuple(press("RIGHT", "B") for _ in range(16))
    ),
    "run32_then_jump": (
        tuple(press("RIGHT", "B") for _ in range(32))
        + tuple(press("RIGHT", "B", "A") for _ in range(4))
        + tuple(press("RIGHT", "B") for _ in range(16))
    ),
}


def segment_actions(name: str) -> tuple[tuple[int, ...], ...]:
    try:
        return SEGMENTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SEGMENTS))
        raise KeyError(f"unknown segment {name!r}; expected one of: {known}") from exc


@dataclass(frozen=True)
class SegmentResult:
    name: str
    profile: ResidualProfile
    approx_obs: list[Observation]
    emu_obs: list[Observation] | None
    horizon: int

    def summary(self) -> str:
        return f"{self.name:10s}  {format_profile(self.profile)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "profile": self.profile.to_dict(),
            "summary": self.summary(),
        }


def replay_on_emulator(
    actions: Sequence[Sequence[int]],
    *,
    state_name: str = "Level1_1",
) -> list[Observation]:
    """Step ``actions`` on fceumm and return start + one observation per frame."""
    import os

    import numpy as np

    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = make_env(GAME_V0, state_name, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        frames = [observation_from_ram(env.get_ram(), 0)]
        a_held = False
        ground_y = frames[0].ground_y
        for i, action in enumerate(actions):
            env.step(np.asarray(action, dtype=np.int8))
            a_held = bool(list(action)[_IDX_A] if len(action) > _IDX_A else 0)
            frames.append(
                observation_from_ram(
                    env.get_ram(),
                    i + 1,
                    a_held=a_held,
                    ground_y=ground_y,
                )
            )
        return frames
    finally:
        env.close()


def measure_segment(
    name: str,
    actions: Sequence[Sequence[int]] | None = None,
    *,
    start: Observation | None = None,
    emu_obs: list[Observation] | None = None,
    run_emulator: bool = True,
    state_name: str = "Level1_1",
) -> SegmentResult:
    """Roll the pure stepper (and optionally the emulator) and compute R(τ)."""
    tape = tuple(tuple(int(v) for v in frame) for frame in (actions or segment_actions(name)))
    live_emu = emu_obs
    if live_emu is None and run_emulator:
        if not ROM_AVAILABLE:
            live_emu = None
        else:
            live_emu = replay_on_emulator(tape, state_name=state_name)

    if start is None:
        start = live_emu[0] if live_emu else level1_start_obs()

    approx_obs = rollout(start, tape)
    profile = compute_residual_profile(approx_obs, live_emu)
    horizon = min(len(approx_obs), len(live_emu or approx_obs))
    return SegmentResult(
        name=name,
        profile=profile,
        approx_obs=approx_obs,
        emu_obs=live_emu,
        horizon=horizon,
    )


def write_report(results: Sequence[SegmentResult], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [item.to_dict() for item in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
