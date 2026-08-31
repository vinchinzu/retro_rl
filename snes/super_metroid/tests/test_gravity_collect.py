"""Gravity collect controller: stop on the item bit (no ROM)."""

from __future__ import annotations

import pytest

from retro_harness.controls import pressed_snes_buttons
from super_metroid.combat.enemies import Enemy
from super_metroid.ram import GRAVITY_MASK
from super_metroid.routes.catalog import DEFAULT_CONTINUOUS_TIP, get_continuous_tip
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.routes.kpdr.room_ids import (
    ROOM_GRAVITY,
    ROOM_HOMING_GEEMER,
    ROOM_PANCAKES,
    ROOM_WEST_OCEAN,
)
from super_metroid.routes.kpdr.wrecked_ship.gravity_collect import (
    ATTIC_ATOMIC_ID,
    ATTIC_GUIDE_BODY,
    ATTIC_KIHUNTER_ID,
    ATTIC_KIHUNTER_WINGS_ID,
    CANDIDATE_ID,
    COLLECT_FRAMES,
    GRAVITY_HOP_BODY,
    PANCAKES_HOMING_GEEMER_SETTLE,
    PANCAKES_HOP_BODY,
    PARENT_TAPE_ID,
    TAPE_BODY_FRAMES,
    WEST_OCEAN_ENTRY_RUN_FRAMES,
    WEST_OCEAN_GUIDE_BODY,
    attic_required_enemies,
    load_gravity_body,
    load_s23_body,
    play_gravity_collect,
    play_pancakes_to_homing_geemer,
    play_west_ocean_to_pancakes,
)


class _State:
    def __init__(self) -> None:
        self.room_id = ROOM_GRAVITY
        self.collected_items = 0x3105
        self.samus_x = 216
        self.samus_y = 118
        self.pose = 82
        self.game_state = 8
        self.door_transition = 0


class _Session:
    def __init__(self) -> None:
        self.frame = 0
        self.state = _State()
        self.actions: list[tuple[int, ...]] = []

    def step(self, action, reason: str = ""):
        del reason
        self.actions.append(tuple(int(v) for v in action))
        self.frame += 1
        if self.frame >= 4:
            self.state.collected_items = 0x3105 | GRAVITY_MASK
        return self.state


def test_stops_on_gravity_bit_before_tape_tail() -> None:
    idle = [0] * 12
    body = [idle] * 12
    session = _Session()
    out = play_gravity_collect(session, frames=body)
    assert out.collected_items & GRAVITY_MASK
    assert session.frame == 4
    assert len(session.actions) == 4
    assert session.frame < TAPE_BODY_FRAMES
    assert COLLECT_FRAMES < TAPE_BODY_FRAMES
    assert CANDIDATE_ID == "controller:gravity_collect"
    assert PARENT_TAPE_ID == "tape:s23_gravity"


def test_wrong_room_fails_closed() -> None:
    session = _Session()
    session.state.room_id = 0xC98E
    with pytest.raises(RuntimeError, match="gravity_collect"):
        play_gravity_collect(session, frames=[[0] * 12])


def test_s23_gravity_body_is_320_frames() -> None:
    if not GRAVITY_HOP_BODY.is_file():
        pytest.skip("s23 Gravity hop body not on disk")
    body = load_gravity_body()
    assert len(body) == TAPE_BODY_FRAMES == 320
    assert all(len(row) == 12 for row in body)


def test_attic_required_enemies_excludes_coverns_and_off_map_links() -> None:
    enemies = (
        Enemy(0, ATTIC_KIHUNTER_ID, 1_083, 206, 360, 0),
        Enemy(1, ATTIC_KIHUNTER_WINGS_ID, 458, 129, 360, 0),
        Enemy(2, ATTIC_ATOMIC_ID, 850, 97, 250, 0),
        Enemy(3, 0xEA3F, 224, 88, 80, 0),
        Enemy(4, ATTIC_KIHUNTER_WINGS_ID, 0, 0, 360, 0),
    )

    assert attic_required_enemies(enemies) == enemies[:3]


def test_attic_guide_is_the_newer_power_bomb_human_take() -> None:
    if not ATTIC_GUIDE_BODY.is_file():
        pytest.skip("gravity_path_v2 human body not on disk")
    body = load_gravity_body(ATTIC_GUIDE_BODY)
    assert len(body) == 2_713
    assert any(row[2] for row in body)  # SELECT cycles to Power Bombs.
    assert any(row[9] for row in body)  # X places the bomb / fires beams.


def test_west_ocean_guide_is_the_newer_gravity_human_take() -> None:
    if not WEST_OCEAN_GUIDE_BODY.is_file():
        pytest.skip("gravity_path_v2 West Ocean body not on disk")
    body = load_gravity_body(WEST_OCEAN_GUIDE_BODY)
    assert len(body) == 1_353


def test_west_ocean_natural_entry_restores_run_momentum() -> None:
    class WestOceanSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.state.room_id = ROOM_WEST_OCEAN

        def step(self, action, reason: str = ""):
            del reason
            self.actions.append(tuple(int(v) for v in action))
            self.frame += 1
            if self.frame == WEST_OCEAN_ENTRY_RUN_FRAMES + 1:
                self.state.room_id = ROOM_PANCAKES
            return self.state

    session = WestOceanSession()
    out = play_west_ocean_to_pancakes(session)

    assert out.room_id == ROOM_PANCAKES
    assert session.frame == WEST_OCEAN_ENTRY_RUN_FRAMES + 1
    assert all(
        tuple(pressed_snes_buttons(action)) == ("B", "LEFT")
        for action in session.actions[:WEST_OCEAN_ENTRY_RUN_FRAMES]
    )


def test_west_ocean_wrong_room_fails_closed() -> None:
    session = _Session()
    with pytest.raises(RuntimeError, match="west_ocean_to_pancakes"):
        play_west_ocean_to_pancakes(session)


def test_pancakes_s23_body_is_416_frames() -> None:
    if not PANCAKES_HOP_BODY.is_file():
        pytest.skip("s23 Pancakes hop body not on disk")
    body = load_s23_body(PANCAKES_HOP_BODY)
    assert len(body) == 416
    assert PANCAKES_HOMING_GEEMER_SETTLE > 120


def test_pancakes_settle_covers_dest_door_transition() -> None:
    if not PANCAKES_HOP_BODY.is_file():
        pytest.skip("s23 Pancakes hop body not on disk")
    body_len = len(load_s23_body(PANCAKES_HOP_BODY))
    dest_settle_needed = 122

    class PancakesSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.state.room_id = ROOM_PANCAKES
            self.state.samus_x = 39
            self.state.samus_y = 139
            self.state.pose = 9

        def step(self, action, reason: str = ""):
            del reason
            self.actions.append(tuple(int(v) for v in action))
            self.frame += 1
            if self.frame >= body_len:
                self.state.room_id = ROOM_HOMING_GEEMER
                self.state.pose = 11
                self.state.game_state = 11
                self.state.door_transition = 1
            if self.frame - body_len >= dest_settle_needed:
                self.state.game_state = 8
                self.state.door_transition = 0
                self.state.pose = 9
            return self.state

    session = PancakesSession()
    out = play_pancakes_to_homing_geemer(session)

    assert out.room_id == ROOM_HOMING_GEEMER
    assert out.game_state == 8
    assert out.door_transition == 0
    assert session.frame == body_len + dest_settle_needed


def test_pancakes_wrong_room_fails_closed() -> None:
    session = _Session()
    with pytest.raises(RuntimeError, match="pancakes_to_homing_geemer"):
        play_pancakes_to_homing_geemer(session)


def test_gravity_collect_is_registered_scratch_tip() -> None:
    assert KPDR_SEGMENTS["gravity_collect"] is play_gravity_collect
    assert get_continuous_tip("gravity").tip_id == "gravity"
    assert DEFAULT_CONTINUOUS_TIP == "phantoon"
