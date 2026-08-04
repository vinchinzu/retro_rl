"""Offline (no-ROM) tests for ALTTP controller primitives."""

from __future__ import annotations

import numpy as np

from alttp.primitives import (
    SPRITE_BLUE_SOLDIER,
    SPRITE_HEART,
    SPRITE_HP,
    SPRITE_SMALL_KEY,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
    Waypoint,
    active_sprites,
    collect_nearby,
    fight_nearby,
    move_path,
    move_to,
    run_script,
    settle_control,
    sprites_of_type,
)
from alttp.ram import (
    INDOORS,
    LINK_ACTION,
    LINK_ACTION_HOLD_UP_ITEM,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SUBMODULE,
)
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX

# Indices used when decoding snes_action vectors from step().
_UP = SNES_BUTTON_NAME_TO_INDEX["UP"]
_DOWN = SNES_BUTTON_NAME_TO_INDEX["DOWN"]
_LEFT = SNES_BUTTON_NAME_TO_INDEX["LEFT"]
_RIGHT = SNES_BUTTON_NAME_TO_INDEX["RIGHT"]


def _ram(writes: dict[int, int], *, size: int = 0x20000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    for addr, value in writes.items():
        if 0 <= addr < len(ram):
            ram[addr] = value & 0xFF
    return ram


def _set_u16(writes: dict[int, int], addr: int, value: int) -> None:
    writes[addr] = value & 0xFF
    writes[addr + 1] = (value >> 8) & 0xFF


def _set_sprite(
    writes: dict[int, int],
    slot: int,
    *,
    sprite_type: int,
    x: int,
    y: int,
    state: int = 9,
    hp: int = 4,
) -> None:
    writes[SPRITE_STATE + slot] = state
    writes[SPRITE_TYPE + slot] = sprite_type
    writes[SPRITE_HP + slot] = hp
    writes[SPRITE_X_LOW + slot] = x & 0xFF
    writes[SPRITE_X_HIGH + slot] = (x >> 8) & 0xFF
    writes[SPRITE_Y_LOW + slot] = y & 0xFF
    writes[SPRITE_Y_HIGH + slot] = (y >> 8) & 0xFF


class FakeEnv:
    """Minimal env: mutable RAM dict + step() that can move Link from actions."""

    def __init__(
        self,
        *,
        link_x: int = 100,
        link_y: int = 100,
        room: int = 0x61,
        module: int = 0x07,
        submodule: int = 0x00,
        link_action: int = 0,
        move_speed: int = 4,
        stuck: bool = False,
    ) -> None:
        self.writes: dict[int, int] = {
            MODULE: module,
            SUBMODULE: submodule,
            INDOORS: 1,
            LINK_ACTION: link_action,
        }
        _set_u16(self.writes, LINK_X, link_x)
        _set_u16(self.writes, LINK_Y, link_y)
        _set_u16(self.writes, ROOM_ID, room)
        self.move_speed = move_speed
        self.stuck = stuck
        self.steps = 0
        self.actions: list[np.ndarray] = []
        # Optional hooks for specialized tests.
        self._on_step: list = []

    def get_ram(self) -> np.ndarray:
        return _ram(self.writes)

    def set_link_xy(self, x: int, y: int) -> None:
        _set_u16(self.writes, LINK_X, x)
        _set_u16(self.writes, LINK_Y, y)

    def set_room(self, room: int) -> None:
        _set_u16(self.writes, ROOM_ID, room)

    def step(self, action: object) -> None:
        self.steps += 1
        arr = np.asarray(action)
        self.actions.append(arr.copy() if arr.ndim else arr)
        for hook in self._on_step:
            hook(self, arr)
        if self.stuck:
            return
        x = int(self.writes.get(LINK_X, 0)) | (
            int(self.writes.get(LINK_X + 1, 0)) << 8
        )
        y = int(self.writes.get(LINK_Y, 0)) | (
            int(self.writes.get(LINK_Y + 1, 0)) << 8
        )
        speed = self.move_speed
        if len(arr) > _RIGHT:
            if arr[_RIGHT]:
                x += speed
            if arr[_LEFT]:
                x -= speed
            if arr[_DOWN]:
                y += speed
            if arr[_UP]:
                y -= speed
        self.set_link_xy(max(0, x), max(0, y))


# --- Waypoint helpers -------------------------------------------------------


def test_waypoint_reached_and_direction() -> None:
    wp = Waypoint(x=100, y=200, tolerance=5, room=0x55, label="nook")
    assert wp.reached(100, 200) is True
    assert wp.reached(104, 197) is True
    assert wp.reached(110, 200) is False
    assert wp.direction_from(100, 100) == "DOWN"
    assert wp.direction_from(50, 200) == "RIGHT"
    shared = wp.as_shared()
    assert shared.x == 100 and shared.room == 0x55


# --- active_sprites / sprites_of_type ---------------------------------------


def test_active_sprites_uses_16bit_coords() -> None:
    env = FakeEnv(link_x=0x0100, link_y=0x0200)
    # Sprite with high bytes set so coords are not low-byte only.
    _set_sprite(
        env.writes,
        0,
        sprite_type=SPRITE_BLUE_SOLDIER,
        x=0x03A5,
        y=0x02B0,
        state=9,
        hp=3,
    )
    # Inactive slot ignored.
    _set_sprite(
        env.writes,
        1,
        sprite_type=SPRITE_HEART,
        x=50,
        y=50,
        state=0,
    )
    sprites = active_sprites(env)
    assert len(sprites) == 1
    s = sprites[0]
    assert s.slot == 0
    assert s.sprite_type == SPRITE_BLUE_SOLDIER
    assert s.x == 0x03A5
    assert s.y == 0x02B0
    assert s.hp == 3
    assert s.state == 9


def test_sprites_of_type_filters_and_max_distance() -> None:
    env = FakeEnv(link_x=100, link_y=100)
    _set_sprite(env.writes, 0, sprite_type=SPRITE_BLUE_SOLDIER, x=110, y=110)
    _set_sprite(env.writes, 1, sprite_type=SPRITE_HEART, x=105, y=105)
    _set_sprite(env.writes, 2, sprite_type=SPRITE_BLUE_SOLDIER, x=400, y=400)

    near = sprites_of_type(env, (SPRITE_BLUE_SOLDIER,), max_distance=50)
    assert len(near) == 1
    assert near[0].slot == 0

    all_soldiers = sprites_of_type(env, (SPRITE_BLUE_SOLDIER,))
    assert {s.slot for s in all_soldiers} == {0, 2}

    hearts = sprites_of_type(env, {SPRITE_HEART})
    assert len(hearts) == 1 and hearts[0].x == 105


# --- run_script -------------------------------------------------------------


def test_run_script_stop_when() -> None:
    env = FakeEnv(link_x=0, link_y=0)
    # After a few steps, flip room so stop_when can fire.
    def stop_after_room(snap) -> bool:
        return snap.room_base_id == 0x80

    def flip_room(e: FakeEnv, _action) -> None:
        if e.steps >= 5:
            e.set_room(0x80)

    env._on_step.append(flip_room)
    result = run_script(
        env,
        ((("RIGHT",), 20),),
        stop_when=stop_after_room,
    )
    assert result.ok is True
    assert result.reason == "acceptance reached"
    assert result.frames == 5
    assert result.snapshot.room_base_id == 0x80


def test_run_script_completes_without_stop_when() -> None:
    env = FakeEnv()
    result = run_script(env, ((("NONE",), 3), (("A",), 2)))
    assert result.ok is True
    assert result.reason == "script complete"
    assert result.frames == 5


def test_run_script_acceptance_not_reached() -> None:
    env = FakeEnv()
    result = run_script(
        env,
        ((("NONE",), 4),),
        stop_when=lambda snap: snap.room_base_id == 0xFF,
    )
    assert result.ok is False
    assert result.reason == "acceptance not reached"


# --- settle_control ---------------------------------------------------------


def test_settle_control_waits_for_control_and_not_hold_up() -> None:
    env = FakeEnv(module=0x07, submodule=0x00, link_action=LINK_ACTION_HOLD_UP_ITEM)

    def clear_hold_up(e: FakeEnv, _action) -> None:
        # After ~20 idle frames, leave hold-up pose.
        if e.steps >= 20:
            e.writes[LINK_ACTION] = 0

    env._on_step.append(clear_hold_up)
    result = settle_control(env, max_frames=120)
    assert result.ok is True
    assert result.reason == "control ready"
    assert result.snapshot.has_control is True
    assert result.snapshot.is_hold_up_item is False
    assert result.frames >= 20


def test_settle_control_timeout() -> None:
    env = FakeEnv(module=0x07, submodule=0x01)  # no has_control
    result = settle_control(env, max_frames=16)
    assert result.ok is False
    assert result.reason == "control timeout"


def test_settle_control_advances_text_mode() -> None:
    env = FakeEnv(module=0x0E, submodule=0x00)  # text mode

    def leave_text(e: FakeEnv, _action) -> None:
        if e.steps >= 8:
            e.writes[MODULE] = 0x07
            e.writes[SUBMODULE] = 0x00
            e.writes[LINK_ACTION] = 0

    env._on_step.append(leave_text)
    result = settle_control(env, max_frames=80)
    assert result.ok is True
    assert result.snapshot.has_control is True
    # Text path presses A/B (index 8 / 0).
    pressed_face = any(
        (len(a) > 8 and (a[8] or a[0])) for a in env.actions
    )
    assert pressed_face


# --- move_to / move_path ----------------------------------------------------


def test_move_to_reaches_waypoint() -> None:
    env = FakeEnv(link_x=100, link_y=100, move_speed=5)
    wp = Waypoint(x=130, y=100, tolerance=5, label="east_nook")
    result = move_to(env, wp, max_frames=200, step_size=1, stuck_cycles=40)
    assert result.ok is True
    assert "east_nook" in result.reason or result.reason == "east_nook"
    assert abs(result.snapshot.link_x - 130) <= 5
    assert abs(result.snapshot.link_y - 100) <= 5


def test_move_to_diagonal_reaches() -> None:
    env = FakeEnv(link_x=50, link_y=50, move_speed=4)
    wp = Waypoint(x=90, y=90, tolerance=6, label="corner")
    result = move_to(env, wp, max_frames=300, step_size=1, stuck_cycles=50)
    assert result.ok is True
    assert wp.reached(result.snapshot.link_x, result.snapshot.link_y)


def test_move_to_fails_stuck() -> None:
    env = FakeEnv(link_x=100, link_y=100, stuck=True)
    wp = Waypoint(x=200, y=100, tolerance=5, label="blocked")
    result = move_to(env, wp, max_frames=600, step_size=3, stuck_cycles=5)
    assert result.ok is False
    assert "stuck" in result.reason
    assert result.snapshot.link_x == 100


def test_move_to_left_room() -> None:
    env = FakeEnv(link_x=100, link_y=100, room=0x61)

    def leave_room(e: FakeEnv, _action) -> None:
        if e.steps >= 2:
            e.set_room(0x62)

    env._on_step.append(leave_room)
    wp = Waypoint(x=200, y=100, room=0x61, label="stay")
    result = move_to(env, wp, max_frames=100, step_size=1, stuck_cycles=50)
    assert result.ok is False
    assert "left room" in result.reason


def test_move_path_stops_on_first_failure() -> None:
    env = FakeEnv(link_x=100, link_y=100, move_speed=5)
    # First waypoint is reachable; second is far and we freeze after first.
    first = Waypoint(x=120, y=100, tolerance=5, label="a")
    second = Waypoint(x=300, y=100, tolerance=5, label="b")

    steps_at_freeze = {"n": None}

    def freeze_after_first(e: FakeEnv, _action) -> None:
        # Once near first waypoint, stop moving so second fails stuck.
        x = int(e.writes.get(LINK_X, 0)) | (int(e.writes.get(LINK_X + 1, 0)) << 8)
        if abs(x - 120) <= 5:
            if steps_at_freeze["n"] is None:
                steps_at_freeze["n"] = e.steps
            e.stuck = True

    env._on_step.append(freeze_after_first)
    result = move_path(
        env,
        (first, second),
        max_frames_per_waypoint=200,
    )
    assert result.ok is False
    assert "stuck" in result.reason or "timeout" in result.reason
    assert "b" in result.reason  # failed on second segment


# --- fight_nearby / collect_nearby ------------------------------------------


def test_fight_nearby_ok_when_no_targets() -> None:
    env = FakeEnv(link_x=100, link_y=100, room=0x61)
    result = fight_nearby(env, max_cycles=10)
    assert result.ok is True
    assert result.reason == "no nearby targets"
    assert result.defeated_slots == ()


def test_fight_nearby_respects_room_leave() -> None:
    env = FakeEnv(link_x=100, link_y=100, room=0x61)
    _set_sprite(
        env.writes,
        0,
        sprite_type=SPRITE_BLUE_SOLDIER,
        x=130,
        y=100,
    )

    def leave(e: FakeEnv, _action) -> None:
        if e.steps >= 3:
            e.set_room(0x99)

    env._on_step.append(leave)
    result = fight_nearby(env, room=0x61, max_cycles=50, max_distance=200)
    assert result.ok is False
    assert "left combat room" in result.reason


def test_collect_nearby_walks_toward_item() -> None:
    env = FakeEnv(link_x=100, link_y=100, move_speed=5)
    # Place key to the east; collect when Link is within 3 on both axes
    # (primitive stops when sprite is gone — we despawn on contact).
    _set_sprite(
        env.writes,
        3,
        sprite_type=SPRITE_SMALL_KEY,
        x=140,
        y=100,
        state=9,
    )

    def despawn_on_contact(e: FakeEnv, _action) -> None:
        x = int(e.writes.get(LINK_X, 0)) | (int(e.writes.get(LINK_X + 1, 0)) << 8)
        y = int(e.writes.get(LINK_Y, 0)) | (int(e.writes.get(LINK_Y + 1, 0)) << 8)
        sx = 140
        sy = 100
        if abs(x - sx) <= 6 and abs(y - sy) <= 6:
            e.writes[SPRITE_STATE + 3] = 0

    env._on_step.append(despawn_on_contact)
    result = collect_nearby(env, (SPRITE_SMALL_KEY,), max_distance=200, max_frames=300)
    assert result.ok is True
    assert result.reason == "pickup collected"
    assert result.snapshot.link_x > 100  # walked toward item
    assert abs(result.snapshot.link_x - 140) <= 10


def test_collect_nearby_already_gone() -> None:
    env = FakeEnv(link_x=50, link_y=50)
    result = collect_nearby(env, (SPRITE_HEART,), max_frames=30)
    assert result.ok is True
    assert result.reason == "pickup collected"
    assert result.frames == 0
