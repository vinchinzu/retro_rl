"""Great Waldo Search segmented showcase hooks for generic recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.scene_recipe import SCENE_RECIPES, run_scene_recipe
from great_waldo_search.targets import (
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.video import CaptureSession, FooterLabels
from retro_harness.video import short_clock
from retro_harness.showcase import ShowcaseClip, ShowcaseGame


@dataclass
class GreatWaldoSearchShowcase:
    """Showcase metadata and replay hooks for The Great Waldo Search."""

    slug: str = "great_waldo_search"
    game: str = GAME
    game_dir: Path = GAME_DIR
    recordings_dir: Path = RECORDINGS_DIR
    players: int = 2
    manifest_format: str = "great-waldo-search-segmented-completion-showcase"
    ending_scope: str = (
        "five-scrolls ending reached from Scene5 clear recipe"
    )

    def intro_lines(self) -> tuple[str, ...]:
        return (
            "GREAT WALDO SEARCH",
            "Segmented five-scene completion showcase",
            "Development checkpoints; not a continuous run",
            "Live score/cursor footer + P1/P2 button tracking",
        )

    def clips(self) -> tuple[ShowcaseClip, ...]:
        return tuple(
            ShowcaseClip(
                label=f"Scene {recipe.banner.number} - {recipe.banner.name}",
                state=recipe.state,
                note="Documented clear recipe from save state",
                hold_frames=30,
            )
            for recipe in SCENE_RECIPES
        )

    def footer_labels(
        self,
        env: object,
        action: list[int],
        frame: int,
        fps: float,
        clip: ShowcaseClip,
    ) -> FooterLabels:
        del action
        recipe = _recipe_for_state(clip.state)
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        score = score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR])
        x = int(ram[CURSOR_X_ADDR])
        y = int(ram[CURSOR_Y_ADDR])
        return FooterLabels(
            upper_left=(
                f"SCENE {recipe.banner.number:02d}/05 "
                f"{recipe.banner.name.upper()}"
            ),
            upper_right=short_clock(frame, fps),
            lower_left=f"SCORE {score:05d}  CUR {x:03d},{y:03d}",
        )

    def run_clip(
        self,
        clip: ShowcaseClip,
        session: CaptureSession,
        env: object,
    ) -> dict[str, Any]:
        recipe = _recipe_for_state(clip.state)
        return run_scene_recipe(session, env, recipe)


def _recipe_for_state(state: str):
    for recipe in SCENE_RECIPES:
        if recipe.state == state:
            return recipe
    raise KeyError(f"unknown showcase state: {state}")


def build_showcase() -> ShowcaseGame:
    """Return the ladder game's showcase hooks."""
    return GreatWaldoSearchShowcase()
