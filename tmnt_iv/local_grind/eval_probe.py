"""Run a headless probe under knob overrides and capture screenshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from retro_harness.env import make_env
from snes_oneshot.actions import idle_action
from snes_oneshot.segment_runner import configure_headless
from tmnt_iv.grind_knobs import override_knobs
from tmnt_iv.paths import GAME, GAME_DIR
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state

_EMERGENCY_HP_THRESHOLD = 16
_EMERGENCY_HP_RESTORE = 80


def run_knob_probe(
    *,
    state_name: str,
    knobs: Mapping[str, Any],
    max_frames: int,
    stop_stage_gt: int | None,
    screenshot_dir: Path | None = None,
    screenshot_every: int = 900,
    max_screenshots: int = 4,
    heal_mode: str = "emergency",
) -> tuple[dict[str, Any], list[Path]]:
    """Eval ``Stage1Policy`` with temporary knobs; return metrics + images."""
    configure_headless()
    image_paths: list[Path] = []
    with override_knobs(knobs):
        env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
        policy = Stage1Policy()
        _reset(env)
        start = parse_game_state(env.get_ram(), frame=0)
        prev_hp = start.health if 0 < start.health <= 0x60 else None
        prev_lives = start.lives
        damage = 0
        max_hit = 0
        min_hp = prev_hp
        heals = 0
        reasons: dict[str, int] = {}
        boss_hp_start = int(start.extras.get("boss_hp", 0))
        final = start
        outcome = "timeout"
        try:
            for frame in range(1, max_frames + 1):
                state = parse_game_state(env.get_ram(), frame=frame)
                final = state
                if 0 < state.health <= 0x60:
                    if prev_hp is not None and state.health < prev_hp:
                        hit = prev_hp - state.health
                        damage += hit
                        max_hit = max(max_hit, hit)
                    prev_hp = state.health
                    if min_hp is None or state.health < min_hp:
                        min_hp = state.health

                if heal_mode == "emergency":
                    if state.health == 0 or (
                        0 < state.health <= _EMERGENCY_HP_THRESHOLD
                    ):
                        env.set_value("player_hp", _EMERGENCY_HP_RESTORE)
                        heals += 1
                        state = parse_game_state(env.get_ram(), frame=frame)
                        final = state
                        prev_hp = state.health

                if state.lives < prev_lives:
                    outcome = "life_loss"
                    _maybe_snap(
                        env,
                        screenshot_dir,
                        image_paths,
                        frame,
                        tag="life_loss",
                        max_screenshots=max_screenshots,
                    )
                    break
                prev_lives = state.lives
                if stop_stage_gt is not None and state.stage > stop_stage_gt:
                    outcome = "stage_advance"
                    break
                if (
                    start.boss_active
                    and not state.boss_active
                    and int(state.extras.get("event", 0)) in {0x0B, 0x19}
                ):
                    outcome = "boss_down"
                if outcome == "boss_down" and frame > 0 and frame % 60 == 0:
                    if state.stage > start.stage or int(
                        state.extras.get("event", 0)
                    ) in {0x19, 0x04}:
                        outcome = "cleared"
                        break

                tick = policy.tick(state)
                action = (
                    tick.action.action
                    if tick.action is not None
                    else idle_action()
                )
                reason = (
                    tick.action.reason
                    if tick.action is not None
                    else tick.reason or "idle"
                )
                reasons[reason] = reasons.get(reason, 0) + 1
                if action[8]:
                    outcome = "forbidden_a"
                    break
                step = env.step(action)
                obs = step[0] if isinstance(step, tuple) else step
                if (
                    screenshot_dir is not None
                    and frame % screenshot_every == 0
                    and len(image_paths) < max_screenshots
                ):
                    path = _save_rgb(
                        obs,
                        screenshot_dir / f"frame_{frame:06d}.png",
                    )
                    if path is not None:
                        image_paths.append(path)
            else:
                outcome = "timeout"
                _maybe_snap(
                    env,
                    screenshot_dir,
                    image_paths,
                    final.frame,
                    tag="timeout",
                    max_screenshots=max_screenshots,
                )
        finally:
            env.close()

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:12]
    metrics = {
        "state": state_name,
        "outcome": outcome,
        "frames": final.frame,
        "start_stage": start.stage,
        "end_stage": final.stage,
        "start_hp": start.health,
        "end_hp": final.health,
        "min_hp": min_hp,
        "damage_taken": damage,
        "max_hit": max_hit,
        "heals": heals,
        "lives": f"{start.lives}->{final.lives}",
        "boss_hp": f"{boss_hp_start}->{int(final.extras.get('boss_hp', 0))}",
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": top,
        "knobs": dict(knobs),
    }
    return metrics, image_paths


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def _maybe_snap(
    env: Any,
    screenshot_dir: Path | None,
    image_paths: list[Path],
    frame: int,
    *,
    tag: str,
    max_screenshots: int,
) -> None:
    if screenshot_dir is None or len(image_paths) >= max_screenshots:
        return
    try:
        obs = env.render()
    except Exception:
        return
    path = _save_rgb(obs, screenshot_dir / f"{tag}_{frame:06d}.png")
    if path is not None:
        image_paths.append(path)


def _save_rgb(obs: Any, path: Path) -> Path | None:
    if obs is None:
        return None
    array = np.asarray(obs)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array[..., :3].astype("uint8")).save(path)
    return path


__all__ = ["run_knob_probe"]
