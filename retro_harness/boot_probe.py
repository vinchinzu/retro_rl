"""Shared reset → first-playable-state boot probe for NES and SNES CLIs."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import configure_headless, save_rgb_png


@dataclass(frozen=True)
class BootProbeConfig:
    game: str
    game_dir: Path
    recordings_dir: Path
    script: Callable[[], Iterable[Any]]  # yields FrameAction
    parse_state: Callable[..., Any]
    is_ready: Callable[[Any, float], bool] | None = None
    # None → run entire script; success if parse_state(...).mode is GameMode.PLAYING
    state_name: str = "Level1"
    screenshot_name: str = "boot_level1.png"
    label: str = "LEVEL1"
    min_frame: int = 0
    stable_frames: int = 1
    hold_idle_frames: int = 0
    require_playing: bool = False
    motion_check: bool = False
    motion_frames: int = 45
    motion_min_changed: int = 3
    walk_frames: int = 30
    walk_button: str = "RIGHT"
    post_script_button: str | None = None
    post_script_frames: int = 0
    action_fn: Callable[[str], Any] | None = None
    idle_fn: Callable[[], Any] | None = None
    extras_fmt: Callable[[Any], str] | None = None
    start_state: str = "NONE"


def _is_nes(game: str) -> bool:
    return str(game).endswith("-Nes")


def _action_fn(cfg: BootProbeConfig) -> Callable[[str], Any]:
    if cfg.action_fn is not None:
        return cfg.action_fn
    return nes_action if _is_nes(cfg.game) else buttons


def _idle_fn(cfg: BootProbeConfig) -> Callable[[], Any]:
    if cfg.idle_fn is not None:
        return cfg.idle_fn
    return nes_idle_action if _is_nes(cfg.game) else idle_action


def _obs_mean(obs: Any) -> float:
    return float(obs.mean())


def _script_action(scripted: Any) -> Any:
    return getattr(scripted, "action", scripted)


def _parse_state(cfg: BootProbeConfig, ram: Any, *, frame: int, mean: float) -> Any:
    try:
        params = inspect.signature(cfg.parse_state).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"frame": frame}
    if "obs_mean" in params:
        kwargs["obs_mean"] = mean
    return cfg.parse_state(ram, **kwargs)


def _ready_now(cfg: BootProbeConfig, ram: Any, mean: float) -> bool:
    assert cfg.is_ready is not None
    try:
        params = inspect.signature(cfg.is_ready).parameters
    except (TypeError, ValueError):
        return bool(cfg.is_ready(ram, mean))
    if "obs_mean" in params:
        return bool(cfg.is_ready(ram, obs_mean=mean))
    return bool(cfg.is_ready(ram, mean))


def _uses_stable_gate(cfg: BootProbeConfig) -> bool:
    return cfg.min_frame > 0 or cfg.stable_frames > 1


def _screenshot(cfg: BootProbeConfig, obs: Any) -> Path:
    return save_rgb_png(obs, cfg.recordings_dir / cfg.screenshot_name)


def _print_line(
    cfg: BootProbeConfig,
    *,
    frame: int,
    png: Path,
    mean: float,
    ready: bool | None = None,
    state: Any | None = None,
    motion_changed: int | None = None,
) -> None:
    parts = [f"{cfg.label} frame={frame}"]
    if state is not None:
        parts.append(f"mode={state.mode.name}")
    if cfg.is_ready is not None:
        if motion_changed is not None:
            parts.append("ready=False")
            parts.append("motion_fail")
            parts.append(f"changed={motion_changed}")
        elif ready is not None:
            parts.append(f"ready={ready}")
        parts.append(f"mean={mean:.1f}")
    if cfg.extras_fmt is not None and state is not None:
        extra = cfg.extras_fmt(state)
        if extra:
            parts.append(extra)
    parts.append(f"screenshot={png}")
    print(" ".join(parts))


def _hold_idle(
    env: Any,
    cfg: BootProbeConfig,
    idle_fn: Callable[[], Any],
    *,
    frame: int,
    obs: Any,
) -> tuple[Any, int, bool]:
    for _ in range(cfg.hold_idle_frames):
        obs, *_ = env.step(idle_fn())
        frame += 1
        if not _ready_now(cfg, env.get_ram(), _obs_mean(obs)):
            return obs, frame, False
    return obs, frame, True


def _step_until_ready(
    env: Any,
    cfg: BootProbeConfig,
    idle_fn: Callable[[], Any],
    *,
    frame: int,
    obs: Any,
) -> tuple[Any, int, bool]:
    stable = 0
    for scripted in cfg.script():
        obs, *_ = env.step(_script_action(scripted))
        frame += 1
        mean = _obs_mean(obs)
        if frame >= cfg.min_frame and _ready_now(cfg, env.get_ram(), mean):
            stable += 1
        else:
            stable = 0
        if stable < cfg.stable_frames:
            continue
        if cfg.hold_idle_frames:
            obs, frame, hold_ok = _hold_idle(
                env, cfg, idle_fn, frame=frame, obs=obs
            )
            if not hold_ok:
                stable = 0
                continue
        return obs, frame, True
    return obs, frame, False


def _run_full_script(env: Any, cfg: BootProbeConfig, *, frame: int, obs: Any) -> tuple[Any, int]:
    for scripted in cfg.script():
        obs, *_ = env.step(_script_action(scripted))
        frame += 1
    return obs, frame


def _hold_button(
    env: Any,
    action: Any,
    frames: int,
    *,
    frame: int,
    obs: Any,
) -> tuple[Any, int]:
    for _ in range(frames):
        obs, *_ = env.step(action)
        frame += 1
    return obs, frame


def _motion_changed(env: Any, action: Any, frames: int, *, frame: int, obs: Any) -> tuple[Any, int, int]:
    before = env.get_ram().copy()
    obs, frame = _hold_button(env, action, frames, frame=frame, obs=obs)
    after = env.get_ram()
    return obs, frame, int((before != after).sum())


def _success(cfg: BootProbeConfig, *, ready: bool, state: Any) -> bool:
    if cfg.is_ready is None:
        return state.mode is GameMode.PLAYING
    if cfg.require_playing and state.mode is not GameMode.PLAYING:
        return False
    return ready


def run_boot_probe(
    cfg: BootProbeConfig,
    *,
    save: bool = True,
    walk_frames: int | None = None,
    post_script_frames: int | None = None,
    env: Any | None = None,
) -> int:
    """Reach the first playable state, screenshot it, and optionally save it."""
    walk_n = cfg.walk_frames if walk_frames is None else walk_frames
    post_n = cfg.post_script_frames if post_script_frames is None else post_script_frames
    action_fn = _action_fn(cfg)
    idle_fn = _idle_fn(cfg)
    walk_action = action_fn(cfg.walk_button)
    owned = env is None
    if owned:
        configure_headless()
        env = make_env(cfg.game, cfg.start_state, cfg.game_dir, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        frame = 0

        if cfg.is_ready is not None:
            obs, frame, reached = _step_until_ready(
                env, cfg, idle_fn, frame=frame, obs=obs
            )
            if not reached and _uses_stable_gate(cfg):
                png = _screenshot(cfg, obs)
                _print_line(cfg, frame=frame, png=png, mean=_obs_mean(obs), ready=False)
                return 1
            if cfg.motion_check:
                obs, frame, changed = _motion_changed(
                    env, walk_action, cfg.motion_frames, frame=frame, obs=obs
                )
                if changed < cfg.motion_min_changed:
                    png = _screenshot(cfg, obs)
                    _print_line(
                        cfg,
                        frame=frame,
                        png=png,
                        mean=_obs_mean(obs),
                        ready=False,
                        motion_changed=changed,
                    )
                    return 1
            else:
                obs, frame = _hold_button(env, walk_action, walk_n, frame=frame, obs=obs)
        else:
            obs, frame = _run_full_script(env, cfg, frame=frame, obs=obs)
            if post_n > 0:
                button = cfg.post_script_button or cfg.walk_button
                obs, frame = _hold_button(
                    env, action_fn(button), post_n, frame=frame, obs=obs
                )

        mean = _obs_mean(obs)
        ram = env.get_ram()
        state = _parse_state(cfg, ram, frame=frame, mean=mean)
        ready = _ready_now(cfg, ram, mean) if cfg.is_ready is not None else state.mode is GameMode.PLAYING
        png = _screenshot(cfg, obs)
        _print_line(cfg, frame=frame, png=png, mean=mean, ready=ready, state=state)
        ok = _success(cfg, ready=ready, state=state)
        if save and ok:
            path = save_state(env, cfg.game_dir, cfg.game, cfg.state_name)
            print(f"saved {path}")
        return 0 if ok else 1
    finally:
        env.close()


def main_boot_probe(
    cfg: BootProbeConfig,
    *,
    argv: list[str] | None = None,
    walk_default: int | None = 30,
    approach_default: int | None = None,
) -> int:
    """Parse a daily CLI and run ``run_boot_probe``."""
    parser = argparse.ArgumentParser(
        description=f"Boot {cfg.game} and save {cfg.state_name}."
    )
    parser.add_argument("--no-save", action="store_true")
    if walk_default is not None:
        parser.add_argument("--walk-frames", type=int, default=walk_default)
    if approach_default is not None:
        parser.add_argument("--approach-frames", type=int, default=approach_default)
    args = parser.parse_args(argv)
    return run_boot_probe(
        cfg,
        save=not args.no_save,
        walk_frames=getattr(args, "walk_frames", None),
        post_script_frames=getattr(args, "approach_frames", None),
    )
