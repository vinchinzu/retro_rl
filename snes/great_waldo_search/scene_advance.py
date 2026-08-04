"""Advance from a cleared search into the next scene HUD.

Documented rebuild recipes (STATUS.md):

- Scene1_Cleared → Scene2: ~8× (A hold 6 + idle 60)
- Scene2–4 Cleared → next: idle ~5, then ~7× (A hold 6 + idle 60)

Soft Scene3–5 layouts send P2-A to ~(206,100) instead of the scroll
assist. Callers should probe assist landing after advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from great_waldo_search.targets import CURSOR_X_ADDR, CURSOR_Y_ADDR
from retro_harness.actions import buttons_multi
from retro_harness.video import RecordingSession


@dataclass(frozen=True)
class AdvanceRecipe:
    """Button timing that rebuilds the next search HUD."""

    pre_idle: int
    pulses: int
    hold_a: int = 6
    gap_idle: int = 60


# Proven rebuild timings for the continuous boot→ending path.
# Cleared `.state` rebuilds differ (stable-retro load mutates RNG) — do not
# copy those pre_idle values here. Verified via em.set_state sweeps:
#   Scene3: 1,2,9,…  | Scene4: 2,6,7,9,…  | Scene5: 0,3,4,…
ADVANCE_TO_SCENE2 = AdvanceRecipe(pre_idle=0, pulses=8)
ADVANCE_TO_SCENE3 = AdvanceRecipe(pre_idle=1, pulses=7)
ADVANCE_TO_SCENE4 = AdvanceRecipe(pre_idle=2, pulses=7)
ADVANCE_TO_SCENE5 = AdvanceRecipe(pre_idle=0, pulses=7)

ADVANCE_AFTER_SCENE: dict[int, AdvanceRecipe] = {
    1: ADVANCE_TO_SCENE2,
    2: ADVANCE_TO_SCENE3,
    3: ADVANCE_TO_SCENE4,
    4: ADVANCE_TO_SCENE5,
}

# Expected P2-A scroll landings for favorable layouts (STATUS / targets).
FAVORABLE_SCROLL_X: dict[int, int] = {
    3: 160,
    4: 34,
    5: 32,
}
SOFT_LAYOUT_X = 206


def advance_scene(
    session: RecordingSession,
    *,
    cleared_scene: int,
) -> AdvanceRecipe:
    """Dismiss congrats / dialogue into the next search scene."""
    recipe = ADVANCE_AFTER_SCENE[cleared_scene]
    if recipe.pre_idle:
        session.idle(recipe.pre_idle)
    for _ in range(recipe.pulses):
        session.hold(buttons_multi(p1=("A",)), recipe.hold_a)
        session.idle(recipe.gap_idle)
    return recipe


def cursor_xy(env: object) -> tuple[int, int]:
    """Read cursor coordinates from RAM."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR])


def probe_assist_landing(
    session: RecordingSession,
    env: object,
    *,
    frames: int = 120,
) -> tuple[int, int]:
    """Hold P2-A briefly and return the assist landing coordinates."""
    session.hold(buttons_multi(p2=("A",)), frames)
    session.idle(4)
    return cursor_xy(env)


def is_favorable_scroll_layout(
    scene: int,
    landing_x: int,
    *,
    soft_x: int = SOFT_LAYOUT_X,
    soft_tolerance: int = 24,
    expected_tolerance: int = 24,
) -> bool:
    """Return True when P2-A settled near the documented scroll assist.

    Soft Scene3–5 layouts land near ``soft_x`` (~206). Other wrong lands
    (e.g. Scene3 at x≈32) also fail the scroll click, so callers require
    proximity to the favorable scroll X when one is known.
    """
    expected = FAVORABLE_SCROLL_X.get(scene)
    if expected is None:
        return True
    if abs(landing_x - soft_x) <= soft_tolerance:
        return False
    return abs(landing_x - expected) <= expected_tolerance
