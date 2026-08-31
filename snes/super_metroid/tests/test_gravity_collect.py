"""Gravity collect controller: stop on the item bit (no ROM)."""

from __future__ import annotations

import pytest

from super_metroid.combat.enemies import Enemy
from super_metroid.ram import GRAVITY_MASK
from super_metroid.routes.catalog import DEFAULT_CONTINUOUS_TIP, get_continuous_tip
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.routes.kpdr.room_ids import ROOM_GRAVITY
from super_metroid.routes.kpdr.wrecked_ship.gravity_collect import (
    ATTIC_ATOMIC_ID,
    ATTIC_GUIDE_BODY,
    ATTIC_KIHUNTER_ID,
    ATTIC_KIHUNTER_WINGS_ID,
    CANDIDATE_ID,
    COLLECT_FRAMES,
    GRAVITY_HOP_BODY,
    PARENT_TAPE_ID,
    TAPE_BODY_FRAMES,
    attic_required_enemies,
    load_gravity_body,
    play_gravity_collect,
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


def test_gravity_collect_is_registered_scratch_tip() -> None:
    assert KPDR_SEGMENTS["gravity_collect"] is play_gravity_collect
    assert get_continuous_tip("gravity").tip_id == "gravity"
    assert DEFAULT_CONTINUOUS_TIP == "phantoon"
