"""Headless auto-state creator for platformer levels.

Navigates from an existing save state (e.g. overworld) to a target level
using scripted button presses, then saves the resulting state.

Navigation format: list of NavStep(buttons, hold_frames, wait_frames)
CLI string format: "RIGHT:15:60 B:10:300 RIGHT+Y:15:60"
Pseudo-buttons `WAIT`, `NOOP`, and `NONE` are also accepted for pure delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retro_harness.actions import indexed_action
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX
from retro_harness.env import GameSpec
from retro_harness.input_script import (
    InputStep,
    parse_input_script,
)

from platformer_common.level_config import PlatformerRAM

# Button name -> SNES index
BUTTON_MAP: dict[str, int] = {
    name: index
    for name, index in SNES_BUTTON_NAME_TO_INDEX.items()
    if index is not None
}

NUM_BUTTONS = 12
NavStep = InputStep


@dataclass
class AutoStateResult:
    """Result of an auto-state creation attempt."""

    success: bool
    path: Path | None = None
    level_id: int = 0
    camera_x: int = 0
    lives: int = 0
    screenshot_path: Path | None = None
    message: str = ""


def parse_nav_string(nav_string: str) -> list[NavStep]:
    """Parse CLI navigation string into NavStep list.

    Format: "BUTTON[+BUTTON]:hold_frames:wait_frames ..."
    Examples:
        "RIGHT:15:60"           - press RIGHT for 15 frames, wait 60
        "RIGHT+Y:15:60"         - press RIGHT+Y for 15 frames, wait 60
        "B:10:300"              - press B for 10 frames, wait 300
        "WAIT:0:300"            - wait 300 frames with no buttons pressed
    """
    return parse_input_script(nav_string)


def _make_action(*buttons: int) -> np.ndarray:
    """Create a 12-element action array with given buttons pressed."""
    return indexed_action(buttons, action_size=NUM_BUTTONS, dtype=np.int8)


NOOP = _make_action()


def navigate_and_save(
    game_name: str,
    game_dir: str | Path,
    from_state: str,
    save_name: str,
    steps: list[NavStep],
    ram: PlatformerRAM,
    expected_level_id: int | None = None,
    settle_frames: int = 30,
    save_screenshot: bool = False,
) -> AutoStateResult:
    """Navigate from a starting state and save the result.

    Args:
        game_name: Retro game ID (e.g. "DonkeyKongCountry-Snes")
        game_dir: Path to game directory
        from_state: Starting state name (e.g. "WinkysWalkwayBk2Start")
        save_name: Name for the saved state (e.g. "BouncyBonanza")
        steps: Navigation steps to execute
        ram: RAM layout for reading game state
        expected_level_id: If set, verify level_id matches after navigation
        settle_frames: Extra NOOP frames after all nav steps
        save_screenshot: Whether to save a screenshot

    Returns:
        AutoStateResult with success status and state details
    """
    game_dir = Path(game_dir)
    schema = ram.to_schema()

    print(f"Creating state '{save_name}' from '{from_state}'")
    print(f"  Game: {game_name}")
    print(f"  Steps: {len(steps)}")

    # Create headless env
    game = GameSpec(game_name, game_dir)
    env = game.make_env(
        from_state,
        render_mode="rgb_array" if save_screenshot else None,
    )
    env.reset()
    base = env.unwrapped

    def read_state() -> dict:
        r = env.get_ram()
        return schema.read(r)

    def step_n(action: np.ndarray, n: int, label: str = "") -> None:
        for _ in range(n):
            base.step(action)
        if label:
            vals = read_state()
            lid = vals.get("level_id", -1)
            cam = vals.get("camera_x", 0)
            lives = vals.get("lives", 0)
            print(f"  [{label}] level_id=0x{lid:02X} camera_x={cam} lives={lives}")

    # Let initial state settle
    step_n(NOOP, 30, "initial")

    # Execute navigation steps
    for i, step in enumerate(steps):
        action = _make_action(*step.buttons)
        button_names = "+".join(
            name for name, idx in BUTTON_MAP.items() if idx in step.buttons
        )
        step_n(action, step.hold_frames)
        step_n(
            NOOP,
            step.wait_frames,
            f"step {i}: {button_names}:{step.hold_frames}:{step.wait_frames}",
        )

    # Final settle
    step_n(NOOP, settle_frames, "settle")

    # Read final state
    vals = read_state()
    level_id = vals.get("level_id", -1)
    camera_x = vals.get("camera_x", 0)
    lives = vals.get("lives", 0)

    # Verify expected level_id
    if expected_level_id is not None and level_id != expected_level_id:
        msg = (
            f"Level ID mismatch: got 0x{level_id:02X}, "
            f"expected 0x{expected_level_id:02X}"
        )
        print(f"\nWARNING: {msg}")
        print("State will still be saved - verify visually.")

    # Save screenshot
    screenshot_path = None
    if save_screenshot:
        obs = env.render()
        if obs is not None:
            from PIL import Image

            screenshot_dir = game_dir / "state_screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"{save_name}.png"
            Image.fromarray(obs).save(screenshot_path)
            print(f"  Screenshot: {screenshot_path}")

    path = game.save_state(env, save_name)
    env.close()

    success = True
    msg = f"Saved {save_name}: level_id=0x{level_id:02X} camera_x={camera_x} lives={lives}"
    if expected_level_id is not None and level_id != expected_level_id:
        msg += f" (expected level_id=0x{expected_level_id:02X})"
        success = False

    print(f"\n{'=' * 60}")
    print(f"{'SUCCESS' if success else 'WARNING'}: {msg}")
    print(f"  State file: {path}")
    print(f"{'=' * 60}")

    return AutoStateResult(
        success=success,
        path=path,
        level_id=level_id,
        camera_x=camera_x,
        lives=lives,
        screenshot_path=screenshot_path,
        message=msg,
    )
