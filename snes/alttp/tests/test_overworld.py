"""Unit tests for light-world castle routing."""

from __future__ import annotations

import numpy as np
import pytest

from alttp.overworld import (
    OwTiles,
    direction_to_screen,
    find_overworld_path,
    next_direction_to_hyrule_castle,
    next_screen_in_path,
    shortest_screen_path,
    standing_screen,
)
from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_SCREEN,
    AlttpSnapshot,
)


def _snap(**kwargs: object) -> AlttpSnapshot:
    base = dict(
        game_mode=0x09,
        submodule=0x00,
        room_id=0,
        indoors=False,
        screen_id=LINKS_HOUSE_SCREEN,
        link_x=2394,
        link_y=3000,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_shortest_path_links_house_to_castle() -> None:
    path = shortest_screen_path(LINKS_HOUSE_SCREEN, HYRULE_CASTLE_SCREEN)
    assert path[0] == LINKS_HOUSE_SCREEN
    assert path[-1] == HYRULE_CASTLE_SCREEN
    assert 0x24 in path  # north field


def test_next_screen_steps_north_from_links_house() -> None:
    assert next_screen_in_path(LINKS_HOUSE_SCREEN, HYRULE_CASTLE_SCREEN) == 0x24


def test_direction_helpers() -> None:
    assert direction_to_screen(0x2C, 0x24) == "UP"
    assert direction_to_screen(0x24, 0x1C) == "UP"
    assert direction_to_screen(0x1C, 0x1B) == "LEFT"


def _tiles(screen: int, big: bool) -> OwTiles:
    shape = (128, 128) if big else (64, 64)
    return OwTiles(
        origin_x=0,
        origin_y=0,
        cols=shape[1],
        rows=shape[0],
        screen=screen,
        big=big,
        attr=np.zeros(shape, dtype=np.uint8),
        map16=np.zeros(shape, dtype=np.uint16),
        liftable=np.zeros(shape, dtype=np.bool_),
    )


def test_standing_screen_splits_a_big_area() -> None:
    small = _tiles(0x13, False)
    assert standing_screen(small, 10, 10) == 0x13
    assert standing_screen(small, 512, 512) == 0x13
    big = _tiles(0x1B, True)
    assert standing_screen(big, 10, 10) == 0x1B
    assert standing_screen(big, 512, 10) == 0x1C
    assert standing_screen(big, 10, 512) == 0x23
    assert standing_screen(big, 512, 512) == 0x24


def test_internal_quadrant_path_uses_outdoor_stairs() -> None:
    tiles = _tiles(0x1E, True)
    tiles.attr[:] = 1
    # A corridor between the southern and northern halves of one loaded area.
    tiles.attr[56:81, 8:10] = 0
    tiles.attr[64:66, 8:10] = 0x22
    path = find_overworld_path(tiles, (64, 608), screen=0x1E)
    assert path is not None
    assert standing_screen(tiles, *path[0]) == 0x26
    assert standing_screen(tiles, *path[-1]) == 0x1E
    assert any(y == 512 for _, y in path)
    tiles.attr[64:66, 8:10] = 1
    assert find_overworld_path(tiles, (64, 608), screen=0x1E) is None


@pytest.mark.rom
def test_eastern_grounds_from_sanctuary_exit() -> None:
    """The real rr-eog9.2 leave must reach the northwest Eastern quadrant."""
    from alttp.paths import INTEGRATION_DIR
    from alttp.ram import read_inventory
    from alttp.rollout import restores
    from alttp.scripts.pin_probe import open_pin
    from alttp.spine.tape import LiveRun
    from alttp.startup import snapshot_env
    from alttp.overworld import walk_to_screen

    if not (INTEGRATION_DIR / "Play_eog9_2.state").is_file():
        pytest.skip("requires the replay-derived Sanctuary exit checkpoint")
    raw = open_pin("Play_eog9_2")
    try:
        before = read_inventory(raw.get_ram())
        live = LiveRun(raw)
        first_restore = restores()
        result = walk_to_screen(live.env, 0x1E, max_frames=12000, guard=True)
        live.note_rollout_restores(restores() - first_restore)
        audit = live.fail_mid_run_state_load()
        snap = snapshot_env(live.env)
        after = read_inventory(raw.get_ram())
        assert result["ok"], result["line"]
        assert result["screen"] == result["standing"] == 0x1E
        assert not snap.indoors and snap.has_control and snap.follower == 0
        assert snap.hp == before.hp and result["hits"] == 0
        assert after == before
        assert audit.writes == 0
    finally:
        raw.close()


def test_porch_escape_then_north() -> None:
    # Still too far left of the porch corridor → move right after clearing south.
    assert (
        next_direction_to_hyrule_castle(_snap(link_x=2300, link_y=3000)) == "RIGHT"
    )
    assert next_direction_to_hyrule_castle(_snap(link_x=2394, link_y=3000)) == "UP"
