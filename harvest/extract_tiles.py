#!/usr/bin/env python3
"""Extract pixel-perfect tile atlas from Harvest Moon SNES.

Uses the same camera model as the editor: the viewport is anchored by the
player's pixel position and clamped to map bounds. A small set of explicit
capture recipes handles the stubborn farm tiles that sit on map edges and were
missed by the previous heuristic camera finder.

Usage:
    uv run python extract_tiles.py
    uv run python extract_tiles.py --states Y1_After_Buy_Potato pretill
    uv run python extract_tiles.py --no-existing-atlas
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_PATH = SCRIPT_DIR / "custom_integrations"
MAPS_DIR = SCRIPT_DIR / "maps"
GAME = "HarvestMoon-Snes"

TILE = 16
MAP_W = 64
ADDR_MAP = 0x09B6
SW, SH = 256, 224
MAP_PX_W = MAP_W * TILE
MAP_PX_H = MAP_W * TILE
FARM_TILEMAP = 0x00

BUTTON_TO_INDEX = {
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
}

REQUIRED_FARM_TILE_IDS = frozenset({0x00, 0x08, 0x56, 0xC0, 0xD3, 0xF7, 0xF9, 0xFB})


@dataclass(frozen=True)
class CaptureStep:
    label: str
    frames: int = 0
    button: str | None = None


@dataclass(frozen=True)
class CaptureRecipe:
    state_name: str
    steps: tuple[CaptureStep, ...]
    settle_frames: int = 10


GENERIC_WALK_STEPS = (
    CaptureStep("start"),
    CaptureStep("up_1", frames=45, button="up"),
    CaptureStep("up_2", frames=45, button="up"),
    CaptureStep("up_3", frames=45, button="up"),
    CaptureStep("left_1", frames=45, button="left"),
    CaptureStep("left_2", frames=45, button="left"),
    CaptureStep("left_3", frames=45, button="left"),
    CaptureStep("down_1", frames=45, button="down"),
    CaptureStep("down_2", frames=45, button="down"),
    CaptureStep("down_3", frames=45, button="down"),
    CaptureStep("down_4", frames=45, button="down"),
    CaptureStep("down_5", frames=45, button="down"),
    CaptureStep("down_6", frames=45, button="down"),
    CaptureStep("right_1", frames=45, button="right"),
    CaptureStep("right_2", frames=45, button="right"),
    CaptureStep("right_3", frames=45, button="right"),
    CaptureStep("right_4", frames=45, button="right"),
    CaptureStep("right_5", frames=45, button="right"),
    CaptureStep("right_6", frames=45, button="right"),
    CaptureStep("up_return_1", frames=45, button="up"),
    CaptureStep("up_return_2", frames=45, button="up"),
    CaptureStep("up_return_3", frames=45, button="up"),
    CaptureStep("up_return_4", frames=45, button="up"),
    CaptureStep("up_return_5", frames=45, button="up"),
    CaptureStep("up_return_6", frames=45, button="up"),
)

FARM_COVERAGE_RECIPES = (
    CaptureRecipe("Start", steps=(CaptureStep("blank_farm"),), settle_frames=0),
    CaptureRecipe("Y1_After_Buy_Potato", steps=(CaptureStep("left_building_edge"),)),
    CaptureRecipe("Y1_Watered_Test", steps=(CaptureStep("watered_plot"),)),
    CaptureRecipe("current", steps=(CaptureStep("crop_patch"),)),
    CaptureRecipe(
        "Y1_Tilled_Test",
        settle_frames=89,
        steps=(CaptureStep("top_pond_tiles"),),
    ),
    CaptureRecipe(
        "TMP_SMASH_ROCK_END",
        settle_frames=0,
        steps=(CaptureStep("lower_pond_edge", frames=3, button="right"),),
    ),
)

DEFAULT_CAPTURE_RECIPES = FARM_COVERAGE_RECIPES + (
    CaptureRecipe("Y1_Front_House", steps=GENERIC_WALK_STEPS),
    CaptureRecipe("pretill", steps=GENERIC_WALK_STEPS),
    CaptureRecipe("current_crop_end", steps=GENERIC_WALK_STEPS),
)


def get_pos(ram: np.ndarray) -> tuple[int, int]:
    return int(ram[0xD6]) | (int(ram[0xD7]) << 8), int(ram[0xD8]) | (int(ram[0xD9]) << 8)


def camera_offset(px: int, py: int) -> tuple[int, int]:
    """Calculate the top-left viewport pixel offset from player position."""
    cx = max(0, min(px - SW // 2, MAP_PX_W - SW))
    cy = max(0, min(py - SH // 2, MAP_PX_H - SH))
    return cx, cy


def iter_visible_tiles(
    ram: np.ndarray,
    cx: int,
    cy: int,
) -> tuple[int, int, int, int, int]:
    """Yield fully visible tiles for the current viewport."""
    tile_data = ram[ADDR_MAP : ADDR_MAP + MAP_W * MAP_W].reshape(MAP_W, MAP_W)
    ox = cx % TILE
    oy = cy % TILE
    btx = cx // TILE
    bty = cy // TILE
    for vy in range(15):
        for vx in range(17):
            sx = vx * TILE - ox
            sy = vy * TILE - oy
            if sx < 0 or sy < 0 or sx + TILE > SW or sy + TILE > SH:
                continue
            tx = btx + vx
            ty = bty + vy
            if tx >= MAP_W or ty >= MAP_W:
                continue
            yield int(tile_data[ty, tx]), tx, ty, sx, sy


def extract_tiles_from_frame(
    obs: np.ndarray,
    ram: np.ndarray,
    cx: int,
    cy: int,
    atlas: dict[int, np.ndarray],
) -> int:
    """Extract visible tiles from frame given camera position."""
    count = 0
    for tid, _tx, _ty, sx, sy in iter_visible_tiles(ram, cx, cy):
        if tid not in atlas:
            atlas[tid] = obs[sy : sy + TILE, sx : sx + TILE].copy()
            count += 1
    return count


def load_existing_atlas() -> dict[int, np.ndarray]:
    atlas: dict[int, np.ndarray] = {}
    atlas_path = MAPS_DIR / "tile_atlas.npy"
    ids_path = MAPS_DIR / "tile_ids.npy"
    if not atlas_path.exists() or not ids_path.exists():
        return atlas

    existing = np.load(atlas_path)
    for tid in np.load(ids_path):
        tid_int = int(tid)
        if existing[tid_int].any():
            atlas[tid_int] = existing[tid_int]
    print(f"Loaded {len(atlas)} existing tiles from {atlas_path}")
    return atlas


def build_generic_recipes(state_names: list[str]) -> tuple[CaptureRecipe, ...]:
    return tuple(CaptureRecipe(state_name=state_name, steps=GENERIC_WALK_STEPS) for state_name in state_names)


def _step_frames(env, obs: np.ndarray, button: str | None, frames: int) -> np.ndarray:
    if frames <= 0:
        return obs

    action = np.zeros(12, dtype=np.int32)
    if button is not None:
        action[BUTTON_TO_INDEX[button]] = 1

    for _ in range(frames):
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return obs


def run_capture_recipe(recipe: CaptureRecipe, atlas: dict[int, np.ndarray]) -> set[int]:
    """Run one capture recipe and return the new tile IDs collected."""
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(str(INTEGRATION_PATH))

    env = retro.make(
        GAME,
        state=recipe.state_name,
        inttype=retro.data.Integrations.ALL,
        render_mode="rgb_array",
    )

    new_ids: set[int] = set()
    try:
        obs, _ = env.reset()
        obs = _step_frames(env, obs, None, recipe.settle_frames)
        for step in recipe.steps:
            obs = _step_frames(env, obs, step.button, step.frames)
            ram = env.get_ram()
            px, py = get_pos(ram)
            cam = camera_offset(px, py)
            before = set(atlas)
            count = extract_tiles_from_frame(obs, ram, cam[0], cam[1], atlas)
            if count > 0:
                gained = sorted(set(atlas) - before)
                new_ids.update(gained)
                print(
                    f"  {recipe.state_name}:{step.label} ({px},{py})"
                    f" cam={cam} +{count} (total={len(atlas)})"
                )
    finally:
        env.close()

    return new_ids


def run_extraction(
    state_names: list[str] | None = None,
    *,
    recipes: tuple[CaptureRecipe, ...] | None = None,
    load_existing: bool = True,
) -> dict[int, np.ndarray]:
    """Run tile extraction and return a tile atlas keyed by tile ID."""
    if recipes is None:
        if state_names is None:
            recipes = DEFAULT_CAPTURE_RECIPES
        else:
            recipes = build_generic_recipes(state_names)

    tile_atlas = load_existing_atlas() if load_existing else {}
    for recipe in recipes:
        try:
            run_capture_recipe(recipe, tile_atlas)
        except Exception as exc:
            print(f"  {recipe.state_name}: ERROR {exc}")
    return tile_atlas


def collect_farm_tile_ids(state_names: list[str] | None = None) -> set[int]:
    """Return the union of tile IDs observed across farm save states."""
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(str(INTEGRATION_PATH))

    if state_names is None:
        state_names = sorted(p.stem for p in (INTEGRATION_PATH / GAME).glob("*.state"))

    farm_tiles: set[int] = set()
    for state_name in state_names:
        env = None
        try:
            env = retro.make(
                GAME,
                state=state_name,
                inttype=retro.data.Integrations.ALL,
                render_mode="rgb_array",
            )
            env.reset()
            ram = env.get_ram()
            if int(ram[0x22]) != FARM_TILEMAP:
                continue
            farm_tiles.update(int(tile_id) for tile_id in ram[ADDR_MAP : ADDR_MAP + MAP_W * MAP_W])
        except Exception as exc:
            print(f"  {state_name}: coverage scan error {exc}")
        finally:
            if env is not None:
                env.close()
    return farm_tiles


def missing_farm_tile_ids(atlas_ids: set[int], state_names: list[str] | None = None) -> set[int]:
    return collect_farm_tile_ids(state_names) - atlas_ids


def save_atlas(atlas: dict[int, np.ndarray]) -> None:
    MAPS_DIR.mkdir(exist_ok=True)
    atlas_arr = np.zeros((256, TILE, TILE, 3), dtype=np.uint8)
    for tid, pixels in atlas.items():
        atlas_arr[tid] = pixels
    np.save(MAPS_DIR / "tile_atlas.npy", atlas_arr)
    np.save(MAPS_DIR / "tile_ids.npy", np.array(sorted(atlas.keys()), dtype=np.uint8))
    print(f"\nSaved {len(atlas)} tiles to {MAPS_DIR}/tile_atlas.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Harvest Moon tile atlas")
    parser.add_argument("--states", nargs="+", default=None, help="Run generic capture walk on specific states")
    parser.add_argument("--no-existing-atlas", action="store_true", help="Ignore maps/tile_atlas.npy and rebuild from scratch")
    args = parser.parse_args()

    atlas = run_extraction(args.states, load_existing=not args.no_existing_atlas)
    save_atlas(atlas)

    farm_missing = missing_farm_tile_ids(set(atlas))
    if farm_missing:
        print(
            "Farm atlas missing:"
            f" {' '.join(f'0x{tile_id:02X}' for tile_id in sorted(farm_missing))}"
        )
    else:
        print("Farm atlas coverage: complete across tracked farm states")

    print(f"Tile IDs: {sorted(f'0x{tile_id:02X}' for tile_id in atlas)}")


if __name__ == "__main__":
    main()
