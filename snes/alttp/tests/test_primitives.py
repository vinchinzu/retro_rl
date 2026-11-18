"""Offline (no-ROM) tests for ALTTP controller primitives."""

from __future__ import annotations

import numpy as np

from alttp.primitives import (
    ANCILLA_FLOOR,
    ANCILLA_TIMER,
    ANCILLA_TYPE,
    ANCILLA_X_HIGH,
    ANCILLA_X_LOW,
    ANCILLA_Y_HIGH,
    ANCILLA_Y_LOW,
    SPRITE_AI,
    SPRITE_BLUE_SOLDIER,
    SPRITE_DIRECTION,
    SPRITE_FLOOR,
    SPRITE_HEART,
    SPRITE_HIT_TIMER,
    SPRITE_HP,
    SPRITE_SMALL_KEY,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
    Waypoint,
    active_ancillae,
    active_sprites,
    collect_nearby,
    fire_rod,
    fight_nearby,
    ride_platform,
    somaria_block,
    move_path,
    move_to,
    run_script,
    settle_control,
    sprites_of_type,
)
from alttp.startup import snapshot_env
from alttp.ram import (
    INDOORS,
    LINK_ACTION,
    LINK_ACTION_HOLD_UP_ITEM,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SUBMODULE,
    wram_index,
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
    ai_state: int = 0,
    direction: int = 0,
    floor: int = 0,
    hit_timer: int = 0,
) -> None:
    writes[SPRITE_STATE + slot] = state
    writes[SPRITE_TYPE + slot] = sprite_type
    writes[SPRITE_HP + slot] = hp
    writes[SPRITE_X_LOW + slot] = x & 0xFF
    writes[SPRITE_X_HIGH + slot] = (x >> 8) & 0xFF
    writes[SPRITE_Y_LOW + slot] = y & 0xFF
    writes[SPRITE_Y_HIGH + slot] = (y >> 8) & 0xFF
    writes[SPRITE_AI + slot] = ai_state
    writes[SPRITE_DIRECTION + slot] = direction
    writes[SPRITE_FLOOR + slot] = floor
    writes[SPRITE_HIT_TIMER + slot] = hit_timer


def _set_ancilla(
    writes: dict[int, int],
    slot: int,
    *,
    ancilla_type: int,
    x: int,
    y: int,
    floor: int = 0,
    timer: int = 0,
) -> None:
    writes[ANCILLA_TYPE + slot] = ancilla_type
    writes[ANCILLA_X_LOW + slot] = x & 0xFF
    writes[ANCILLA_X_HIGH + slot] = (x >> 8) & 0xFF
    writes[ANCILLA_Y_LOW + slot] = y & 0xFF
    writes[ANCILLA_Y_HIGH + slot] = (y >> 8) & 0xFF
    writes[ANCILLA_FLOOR + slot] = floor
    writes[ANCILLA_TIMER + slot] = timer


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
    assert s.ai_state == 0
    assert s.direction == 0
    assert s.floor == 0
    assert s.hit_timer == 0


def test_sprite_ai_direction_floor_and_hit_timer() -> None:
    """Synthetic RAM for the fields measured on CastleB1Guard type 0x41."""
    env = FakeEnv()
    _set_sprite(
        env.writes,
        3,
        sprite_type=SPRITE_BLUE_SOLDIER,
        x=0x0140,
        y=0x0288,
        ai_state=0x04,
        direction=0x02,
        floor=0x01,
        hit_timer=0x9D,
    )
    sprite = active_sprites(env)[0]
    assert sprite.slot == 3
    assert sprite.ai_state == 0x04
    assert sprite.direction == 0x02
    assert sprite.floor == 0x01
    assert sprite.hit_timer == 0x9D


def test_active_ancillae_keeps_link_shots_and_bombs() -> None:
    env = FakeEnv(link_x=100, link_y=80)
    _set_ancilla(env.writes, 0, ancilla_type=0x00, x=1, y=1)
    _set_ancilla(env.writes, 1, ancilla_type=0x07, x=0x0210, y=0x0134, timer=0x30)
    _set_ancilla(env.writes, 2, ancilla_type=0x20, x=50, y=50)  # bedspread
    _set_ancilla(env.writes, 3, ancilla_type=0x02, x=140, y=70)  # fire rod
    _set_ancilla(
        env.writes, 4, ancilla_type=0x09, x=120, y=90, floor=1, timer=8
    )
    _set_ancilla(env.writes, 5, ancilla_type=0x1F, x=110, y=60)  # hookshot
    # Soldier arrow is a sprite, not an ancilla slot.
    _set_sprite(env.writes, 0, sprite_type=0x1B, x=130, y=90, hp=0)

    found = active_ancillae(env)
    assert [(a.slot, a.ancilla_type) for a in found] == [
        (1, 0x07),
        (3, 0x02),
        (4, 0x09),
        (5, 0x1F),
    ]
    bomb, rod, arrow, hook = found
    assert rod.x == 140 and rod.y == 70
    assert hook.ancilla_type == 0x1F and hook.x == 110
    assert bomb.x == 0x0210 and bomb.y == 0x0134 and bomb.timer == 0x30
    assert arrow.floor == 1 and arrow.timer == 8
    assert arrow.distance_to(snapshot_env(env)) == 30
    assert active_sprites(env)[0].sprite_type == 0x1B


def _own_fire_rod(env: FakeEnv, magic: int) -> None:
    env.writes[wram_index(0xF345)] = 1
    env.writes[wram_index(0xF36E)] = magic & 0xFF
    env.writes[0x0202] = 0x06


def test_fire_rod_refuses_without_item_magic_or_doorway() -> None:
    bare = FakeEnv()
    missed = fire_rod(bare)
    assert missed.ok is False
    assert missed.reason == "fire rod is not owned"
    assert missed.frames == 0

    short = FakeEnv()
    _own_fire_rod(short, 8)
    poor = fire_rod(short)
    assert poor.ok is False
    assert poor.reason == "magic 8 < 16"
    assert poor.frames == 0

    blocked = FakeEnv()
    _own_fire_rod(blocked, 0x80)
    blocked.writes[0x006C] = 1
    door = fire_rod(blocked)
    assert door.ok is False
    assert door.reason == "doorway"


def test_fire_rod_spends_sixteen_on_a_new_shot() -> None:
    """Menu and the 16-point drop are scripted. AP-072 measured 16 on Y."""
    env = FakeEnv()
    _own_fire_rod(env, 0x80)
    start = SNES_BUTTON_NAME_TO_INDEX["START"]
    use_y = SNES_BUTTON_NAME_TO_INDEX["Y"]

    def menu(fake: FakeEnv, action: np.ndarray) -> None:
        if int(action[start]):
            if fake.writes.get(MODULE) == 0x0E:
                fake.writes[MODULE] = 0x07
            else:
                fake.writes[MODULE] = 0x0E
                fake.writes[0x0200] = 4
        if int(action[use_y]) and fake.writes.get(MODULE) == 0x07:
            fake.writes[0x0C4A] = 0x02
            fake.writes[wram_index(0xF36E)] = 0x70

    env._on_step.append(menu)
    shot = fire_rod(env)
    assert shot.ok is True
    assert shot.reason == "fire rod (0,0) magic 128->112"


def _own_somaria(env: FakeEnv, magic: int) -> None:
    env.writes[wram_index(0xF350)] = 1
    env.writes[wram_index(0xF36E)] = magic & 0xFF
    env.writes[0x0202] = 0x11
    env.writes[0x002F] = 0
    env.writes[0x00EC] = 0xF8
    env.writes[0x00ED] = 0x01


def _menu_y(env: FakeEnv, on_y) -> None:
    start = SNES_BUTTON_NAME_TO_INDEX["START"]
    use_y = SNES_BUTTON_NAME_TO_INDEX["Y"]

    def menu(fake: FakeEnv, action: np.ndarray) -> None:
        if int(action[start]):
            if fake.writes.get(MODULE) == 0x0E:
                fake.writes[MODULE] = 0x07
            else:
                fake.writes[MODULE] = 0x0E
                fake.writes[0x0200] = 4
        if int(action[use_y]) and fake.writes.get(MODULE) == 0x07:
            on_y(fake)

    env._on_step.append(menu)


def test_somaria_block_refuses_without_item_magic_or_doorway() -> None:
    bare = FakeEnv()
    missed = somaria_block(bare)
    assert missed.ok is False
    assert missed.reason == "somaria is not owned"
    assert missed.frames == 0

    short = FakeEnv()
    _own_somaria(short, 4)
    poor = somaria_block(short)
    assert poor.ok is False
    assert poor.reason == "magic 4 < 8"

    blocked = FakeEnv()
    _own_somaria(blocked, 0x80)
    blocked.writes[0x006C] = 1
    door = somaria_block(blocked)
    assert door.ok is False
    assert door.reason == "doorway"


def test_somaria_block_pushes_onto_a_switch() -> None:
    """Menu, ancilla 0x2C, and the 8-point drop are scripted. AP-072 measured 8."""
    env = FakeEnv(link_x=100, link_y=100)
    _own_somaria(env, 0x80)
    # (100, 100) under mask $01F8 is tile (12, 12). The push writes the switch.
    tile = wram_index(0x12000) + 12 * 64 + 12
    up = SNES_BUTTON_NAME_TO_INDEX["UP"]

    def on_y(fake: FakeEnv) -> None:
        fake.writes[0x0C4A] = 0x2C
        fake.writes[0x0C04] = 100
        fake.writes[0x0BFA] = 100
        fake.writes[wram_index(0xF36E)] = 0x78

    def on_push(fake: FakeEnv, action: np.ndarray) -> None:
        if int(action[up]) and fake.writes.get(0x0C4A) == 0x2C:
            fake.writes[tile] = 0x23
            fake.writes[0x0646] = 1

    _menu_y(env, on_y)
    env._on_step.append(on_push)
    made = somaria_block(env)
    assert made.ok is True
    assert made.reason == "block tile (12,12) attr 0x23 $0646=1 magic 128->120"


def test_ride_platform_stands_and_presses_the_junction() -> None:
    env = FakeEnv(link_x=100, link_y=100)
    _own_somaria(env, 0x80)
    env.writes[wram_index(0x12000) + 12 * 64 + 12] = 0xB6
    up = SNES_BUTTON_NAME_TO_INDEX["UP"]

    def on_y(fake: FakeEnv) -> None:
        _set_sprite(fake.writes, 0, sprite_type=0xED, x=100, y=100, state=9, hp=0, direction=1)
        fake.writes[0x0E90] = 0xB6
        fake.writes[0x02F5] = 1
        fake.writes[wram_index(0xF36E)] = 0x78

    def on_press(fake: FakeEnv, action: np.ndarray) -> None:
        if fake.writes.get(SPRITE_TYPE) == 0xED:
            fake.writes[0x02F5] = 1
        if int(action[up]) and fake.writes.get(SPRITE_TYPE) == 0xED:
            y = int(fake.writes.get(SPRITE_Y_LOW, 0))
            fake.writes[SPRITE_Y_LOW] = (y - 4) & 0xFF
            link_y = int(fake.writes.get(LINK_Y, 0))
            fake.writes[LINK_Y] = (link_y - 4) & 0xFF

    _menu_y(env, on_y)
    env._on_step.append(on_press)
    rode = ride_platform(env)
    assert rode.ok is True
    assert rode.reason == (
        "platform 0xED tile (12,12) attr 0xB6 press UP $02F5 magic 128->120"
    )


def test_ride_platform_walks_to_the_edge_of_a_pit() -> None:
    """A pit tile ahead does not cancel floor still left in this tile."""
    env = FakeEnv(link_x=100, link_y=100, move_speed=1)
    _own_somaria(env, 0x80)
    base = wram_index(0x12000)
    env.writes[base + 11 * 64 + 12] = 0x20
    env.writes[base + 10 * 64 + 12] = 0xB6
    up = SNES_BUTTON_NAME_TO_INDEX["UP"]

    def on_y(fake: FakeEnv) -> None:
        _set_sprite(fake.writes, 0, sprite_type=0xED, x=100, y=80, state=9, hp=0, direction=1)
        fake.writes[0x0E90] = 0xB6
        fake.writes[wram_index(0xF36E)] = 0x78

    def on_step(fake: FakeEnv, action: np.ndarray) -> None:
        if fake.writes.get(SPRITE_TYPE) != 0xED:
            return
        y = int(fake.writes.get(LINK_Y, 0)) | (int(fake.writes.get(LINK_Y + 1, 0)) << 8)
        if (y & 7) <= 1:
            fake.writes[0x02F5] = 1
        if int(action[up]) and fake.writes.get(0x02F5):
            sprite_y = int(fake.writes.get(SPRITE_Y_LOW, 0))
            fake.writes[SPRITE_Y_LOW] = (sprite_y - 4) & 0xFF
            fake.writes[LINK_Y] = (y - 4) & 0xFF

    _menu_y(env, on_y)
    env._on_step.append(on_step)
    rode = ride_platform(env)
    assert rode.ok is True
    assert "tile (12,10) attr 0xB6 press UP" in rode.reason
    assert "magic 128->120" in rode.reason


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
