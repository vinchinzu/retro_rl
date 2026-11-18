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


@pytest.mark.rom
def test_sahasrahla_hut_from_eastern_grounds_predecessor() -> None:
    """The real rr-eog9.3 checkpoint completes the mapped hut visit."""
    from alttp.opening_route.segment import visit_sahasrahla_hut
    from alttp.paths import INTEGRATION_DIR
    from alttp.ram import (
        FOLLOWER,
        SRAM_PROGRESS,
        SRAM_PROGRESS_FLAGS,
        read_inventory,
        read_u8_safe,
        wram_index,
    )
    from alttp.rollout import restores
    from alttp.scripts.pin_probe import open_pin
    from alttp.spine.tape import LiveRun
    from alttp.startup import snapshot_env

    if not (INTEGRATION_DIR / "Play_eog9_3.state").is_file():
        pytest.skip("requires the replay-derived Eastern grounds checkpoint")
    raw = open_pin("Play_eog9_3")
    try:
        before_inv = read_inventory(raw.get_ram())
        before_ram = raw.get_ram()
        before_marker = read_u8_safe(before_ram, wram_index(0xF3C7))
        before_progress = read_u8_safe(before_ram, wram_index(SRAM_PROGRESS))
        before_flags = read_u8_safe(before_ram, wram_index(SRAM_PROGRESS_FLAGS))
        before_follower = read_u8_safe(before_ram, wram_index(FOLLOWER))
        live = LiveRun(raw, provider="alttp.eog9_4_sahasrahla_hut")
        first_restore = restores()
        result = visit_sahasrahla_hut(live.env, source="state_load_dev")
        live.note_rollout_restores(restores() - first_restore)
        audit = live.fail_mid_run_state_load()
        snap = snapshot_env(live.env)
        after_ram = live.env.get_ram()
        after_inv = read_inventory(after_ram)
        after_marker = read_u8_safe(after_ram, wram_index(0xF3C7))
        after_progress = read_u8_safe(after_ram, wram_index(SRAM_PROGRESS))
        after_flags = read_u8_safe(after_ram, wram_index(SRAM_PROGRESS_FLAGS))
        after_follower = read_u8_safe(after_ram, wram_index(FOLLOWER))
    finally:
        raw.close()

    assert before_marker == 1
    assert before_progress == 2 and before_flags == 0x15
    assert before_follower == 0
    assert before_inv.hp == before_inv.max_hp
    assert result.ok, result.blocker
    assert result.acceptance["hut_completed"]
    assert not snap.indoors and snap.screen_id == 0x1E and snap.has_control
    assert snap.follower == after_follower == 0
    from alttp.room_map import load_room_map

    door = load_room_map("room_05").door("south_to_0x1e")
    assert door is not None and door.landing_xy is not None
    landing = door.landing_xy
    assert abs(snap.link_x - landing[0]) <= 4
    assert abs(snap.link_y - landing[1]) <= 4
    assert after_marker == 3
    assert after_progress == before_progress and after_flags == before_flags
    assert after_inv == before_inv
    assert after_follower == before_follower
    assert audit.writes == 0
    assert audit.set_state == audit.rollout_restores

    entry = next(p for p in result.phases if p.phase == "settle_destination")
    assert entry.ok and entry.snapshot.room_id == 0x0105
    assert entry.snapshot.indoors and entry.snapshot.has_control
    for pot in ("pot_left_front", "pot_right_front"):
        for action in ("lift", "throw"):
            phase = next(p for p in result.phases if p.phase == f"{pot}_{action}")
            assert phase.ok, f"{pot}_{action}: {phase.detail}"


@pytest.mark.rom
def test_eastern_approach_from_completed_hut_predecessor() -> None:
    """Play the actual hut leave through the hedge and stone stair collisions."""
    from dataclasses import asdict

    from alttp.opening_route.segment import get_segment
    from alttp.paths import INTEGRATION_DIR
    from alttp.ram import read_inventory, wram_index
    from alttp.rollout import restores
    from alttp.room_map import load_room_map
    from alttp.scripts.pin_probe import open_pin
    from alttp.spine.tape import LiveRun
    from alttp.startup import snapshot_env

    if not (INTEGRATION_DIR / "Play_eog9_4.state").is_file():
        pytest.skip("requires the replay-derived completed hut checkpoint")
    raw = open_pin("Play_eog9_4")
    try:
        before = asdict(read_inventory(raw.get_ram()))
        story = tuple(int(raw.get_ram()[wram_index(a)]) for a in (0xF3C5, 0xF3C6, 0xF3C7))
        live = LiveRun(raw, provider="alttp.eog9_5_eastern_approach")
        first_restore = restores()
        result = get_segment("eastern_approach").play_checked(live.env, source="state_load_dev")
        live.note_rollout_restores(restores() - first_restore)
        audit = live.fail_mid_run_state_load()
        snap = snapshot_env(live.env)
        after = asdict(read_inventory(raw.get_ram()))
        after_story = tuple(int(raw.get_ram()[wram_index(a)]) for a in (0xF3C5, 0xF3C6, 0xF3C7))
    finally:
        raw.close()

    assert result.ok, result.blocker
    assert result.acceptance["eastern_lobby"]
    assert snap.indoors and snap.room_base_id == 0xC9 and snap.has_control
    assert snap.follower == 0 and snap.max_hp == 32 and 0 < snap.hp <= before["hp"]
    door = load_room_map("screen_1e_eastern").door("eastern_door_to_0xc9")
    assert door is not None and door.landing_xy is not None
    assert abs(snap.link_x - door.landing_xy[0]) <= 4
    assert abs(snap.link_y - door.landing_xy[1]) <= 4
    assert after_story == story == (2, 0x15, 3)
    assert {k: v for k, v in after.items() if k != "hp"} == {
        k: v for k, v in before.items() if k != "hp"
    }
    assert audit.writes == 0 and audit.set_state == audit.rollout_restores
    assert 0 < result.frames <= 6000


def test_eastern_door_hop_from_hedge_replay() -> None:
    """rr-eog9.6 pushed the mapped door from Play_eog9_5, not EasternPalaceEntry."""
    import json

    from alttp.paths import RECORDINGS_DIR
    from alttp.room_map import load_room_map
    from alttp.scripts.audit_power_on import replay_goal_accepted

    path = RECORDINGS_DIR / "eog9_door_20261023" / "replay.json"
    if not path.is_file():
        pytest.skip("requires the Play_eog9_5 door replay")
    report = json.loads(path.read_text(encoding="utf-8"))
    door = load_room_map("screen_1e_eastern").door("eastern_door_to_0xc9")
    assert door is not None and door.landing_xy is not None and door.to_room == 0xC9
    final = report["final"]
    assert report["boot_state"] == "Play_eog9_5"
    assert "EasternPalaceEntry" not in report["saved_state"]
    assert report["desynced"] == 0
    assert report["audit"]["set_state"] == report["audit"]["writes"] == 0
    assert final["game_mode"] == 0x07 and final["submodule"] == 0
    assert final["indoors"] and final["has_control"]
    assert final["room_base_id"] == door.to_room
    assert final["follower"] == 0 and final["has_fighter_sword"] and final["has_lamp"]
    assert final["max_hp"] == 32 and final["hp"] == 28 and final["keys"] == 0
    assert max(
        abs(final["link_x"] - door.landing_xy[0]),
        abs(final["link_y"] - door.landing_xy[1]),
    ) <= 4
    assert replay_goal_accepted(
        {"through": "eastern_entry", "source_ok": True}, final, report["story"],
    )


def test_porch_escape_then_north() -> None:
    # Still too far left of the porch corridor → move right after clearing south.
    assert (
        next_direction_to_hyrule_castle(_snap(link_x=2300, link_y=3000)) == "RIGHT"
    )
    assert next_direction_to_hyrule_castle(_snap(link_x=2394, link_y=3000)) == "UP"
