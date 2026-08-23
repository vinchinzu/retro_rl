"""MK1 SNES WRAM layout, fight snapshot, and approximate hitboxes.

Addresses are stable-retro ``get_ram()`` offsets (WRAM ``$7E:xxxx``).
Confirmed via ``data.json`` and GameHacking.org USA (CRC32 DEF42945).

v3 obs still uses object-stride ``0x00DA``/``0x0174`` (animation noise the
overnight zips were trained on). Live screen pose is ``0x1966``/``0x1968``
and ``0x030F``/``0x032F`` — scripted policy only. Sprite tables at
``0x7688`` / ``0x7788`` were empty on Fight_LiuKang.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

MAX_HEALTH = 161
MAX_TIMER = 154
LIU_KANG_ID = 3
FIGHTER_STRIDE = 0x9A
# Old object-stride guess. Still used for anim/state pokes in tests.
P1_OBJ = 0x00DA
P2_OBJ = 0x0174
OFF_X = 0x00
OFF_Y = 0x01
OFF_STATE = 0x38
# Screen pose (not P1_OBJ). 16-bit LE low bytes; high byte stays 0x01 on-screen.
ADDR_P1_X = 0x1966
ADDR_P1_Y = 0x1968
ADDR_P2_X = 0x030F
ADDR_P2_Y = 0x032F

ADDR_GAME_MODE = 0x0022
ADDR_MATCH_COUNTER = 0x000A
ADDR_P2_CHARACTER = 0x0024
ADDR_TIMER = 0x0122
ADDR_CONTINUE_TIMER = 0x03E7
ADDR_P2_ROUNDS = 0x04B7
ADDR_P1_HEALTH = 0x04B9
ADDR_P2_HEALTH = 0x04BB
ADDR_P1_ROUNDS = 0x196E
ADDR_P1_CHARACTER = 0x1972
ADDR_P1_CHAR_ALT = 0x1921

SPRITE_P1_BASE = 0x7688
SPRITE_P2_BASE = 0x7788

CHAR_NAMES: dict[int, str] = {
    0: "JohnnyCage",
    1: "Kano",
    2: "Raiden",
    3: "LiuKang",
    4: "Scorpion",
    5: "SubZero",
    6: "Sonya",
    7: "Goro",
    8: "ShangTsung",
}

# Standing hurtbox (pixels). Attack box extends in facing direction.
HURT_W = 28
HURT_H_STAND = 80
HURT_H_CROUCH = 48
ATTACK_W = 40
ATTACK_H = 24
ATTACK_STATE_MIN = 1


class Screen(IntEnum):
    """Coarse screen class from HUD / health / continue RAM."""

    UNKNOWN = 0
    BOOT = 1
    MENU = 2
    CHAR_SELECT = 3
    FIGHT = 4
    BETWEEN_ROUNDS = 5
    CONTINUE = 6
    CREDITS = 7


@dataclass(frozen=True)
class Box:
    """Axis-aligned hit/hurt box in screen pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    def overlaps(self, other: Box) -> bool:
        return self.x0 < other.x1 and self.x1 > other.x0 and self.y0 < other.y1 and self.y1 > other.y0

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)


@dataclass(frozen=True)
class Fighter:
    """One fighter object + derived boxes."""

    x: int
    y: int
    state: int
    health: int
    rounds: int
    char_id: int
    facing: int  # +1 right, -1 left
    hurt: Box
    attack: Box | None


@dataclass(frozen=True)
class FightSnapshot:
    """Read-only MK1 snapshot used by boot, tournament, and RAM obs."""

    p1_health: int
    p2_health: int
    timer: int
    continue_timer: int
    p1_character: int
    p2_character: int
    p1_rounds: int
    p2_rounds: int
    match_counter: int
    game_mode: int
    p1: Fighter
    p2: Fighter
    ram_len: int
    screen: Screen

    @property
    def distance_x(self) -> int:
        return abs(self.p2.x - self.p1.x)

    @property
    def p1_hit_connects(self) -> bool:
        return self.p1.attack is not None and self.p1.attack.overlaps(self.p2.hurt)

    @property
    def p2_hit_connects(self) -> bool:
        return self.p2.attack is not None and self.p2.attack.overlaps(self.p1.hurt)

    @property
    def bodies_overlap(self) -> bool:
        return self.p1.hurt.overlaps(self.p2.hurt)


def _u8(ram: np.ndarray, addr: int) -> int:
    if addr < 0 or addr >= len(ram):
        return 0
    return int(ram[addr]) & 0xFF


def _round_count(raw: int) -> int:
    """Best-of-three HUD. Bytes >2 are reused by other screens / cheats."""
    return raw if 0 <= raw <= 2 else 0


def _facing(p1_x: int, p2_x: int, is_p1: bool) -> int:
    if p1_x == p2_x:
        return 1 if is_p1 else -1
    if is_p1:
        return 1 if p1_x < p2_x else -1
    return 1 if p2_x < p1_x else -1


def _hurt_box(x: int, y: int, crouching: bool) -> Box:
    h = HURT_H_CROUCH if crouching else HURT_H_STAND
    x0 = int(x) - HURT_W // 2
    y1 = int(y)
    return Box(x0, y1 - h, x0 + HURT_W, y1)


def _attack_box(x: int, y: int, facing: int, state: int, crouching: bool) -> Box | None:
    if state < ATTACK_STATE_MIN:
        return None
    hurt = _hurt_box(x, y, crouching)
    torso = (hurt.y0 + hurt.y1) // 2
    if facing >= 0:
        x0 = hurt.x1
        x1 = x0 + ATTACK_W
    else:
        x1 = hurt.x0
        x0 = x1 - ATTACK_W
    return Box(x0, torso - ATTACK_H // 2, x1, torso + ATTACK_H // 2)


def _read_fighter(
    x: int,
    y: int,
    state: int,
    health: int,
    rounds: int,
    char_id: int,
    facing: int,
) -> Fighter:
    crouching = y > 160
    hurt = _hurt_box(x, y, crouching)
    attack = _attack_box(x, y, facing, state, crouching)
    return Fighter(
        x=x,
        y=y,
        state=state,
        health=health,
        rounds=rounds,
        char_id=char_id,
        facing=facing,
        hurt=hurt,
        attack=attack,
    )


def classify_screen(
    *,
    p1_health: int,
    p2_health: int,
    timer: int,
    continue_timer: int,
    p1_character: int,
    match_counter: int,
    p1_rounds: int,
    p2_rounds: int,
) -> Screen:
    """Classify the current screen from HUD RAM (no pixels)."""
    del p1_character
    if continue_timer > 0:
        return Screen.CONTINUE
    fighting = (
        timer > 0
        and (p1_health > 0 or p2_health > 0)
        and (p1_health <= MAX_HEALTH)
        and (p2_health <= MAX_HEALTH)
    )
    if fighting:
        return Screen.FIGHT
    if match_counter >= 12 and p1_rounds >= 2 and timer == 0:
        return Screen.CREDITS
    # Char select (including leftover 0xA1 bars) only before any round is played.
    # Match 1 between-rounds also has match_counter=0 and timer=0. A timeout-KO
    # with p2_health=1 must not look like choose-your-fighter.
    if (
        match_counter == 0
        and timer == 0
        and continue_timer == 0
        and p1_rounds == 0
        and p2_rounds == 0
        and p1_health in (0, MAX_HEALTH)
        and p2_health in (0, MAX_HEALTH)
    ):
        return Screen.CHAR_SELECT
    if p1_health == 0 and p2_health == 0 and timer == 0:
        return Screen.MENU
    if p1_health == 0 or p2_health == 0:
        return Screen.BETWEEN_ROUNDS
    if timer == 0:
        return Screen.BETWEEN_ROUNDS
    return Screen.MENU


def is_char_select(snap: FightSnapshot) -> bool:
    """Choose-your-fighter: both health bars sit at 161 with the timer down."""
    return (
        snap.timer == 0
        and snap.continue_timer == 0
        and snap.p1_health == MAX_HEALTH
        and snap.p2_health == MAX_HEALTH
        and snap.match_counter == 0
        and snap.p1_rounds == 0
        and snap.p2_rounds == 0
    )


def is_match_won(snap: FightSnapshot) -> bool:
    """True when P1 has taken the match (best of three, strict majority)."""
    return snap.p1_rounds >= 2 and snap.p1_rounds > snap.p2_rounds


def is_match_lost(snap: FightSnapshot) -> bool:
    """True when P2 has taken the match (best of three, strict majority)."""
    return snap.p2_rounds >= 2 and snap.p2_rounds > snap.p1_rounds


def rounds_settled(snap: FightSnapshot) -> bool:
    """True when round bytes are HUD-stable (KO / timeout, not VS flicker)."""
    return snap.screen is Screen.BETWEEN_ROUNDS and snap.timer == 0


def is_fight_ready(snap: FightSnapshot, *, character: int = LIU_KANG_ID) -> bool:
    """True when round 1 is live at full health for the chosen character."""
    return (
        snap.screen is Screen.FIGHT
        and snap.p1_character == character
        and snap.p1_health == MAX_HEALTH
        and snap.p2_health == MAX_HEALTH
        and snap.timer > 50
    )


def parse_ram(ram: np.ndarray) -> FightSnapshot:
    """Parse a ``get_ram()`` buffer into a fight snapshot."""
    # v3 MLP was trained on these object-stride bytes (noisy). Scripted pose
    # reads ADDR_P1_X / ADDR_P2_X instead.
    p1_x = _u8(ram, P1_OBJ + OFF_X)
    p1_y = _u8(ram, P1_OBJ + OFF_Y)
    p2_x = _u8(ram, P2_OBJ + OFF_X)
    p2_y = _u8(ram, P2_OBJ + OFF_Y)
    p1_health = _u8(ram, ADDR_P1_HEALTH)
    p2_health = _u8(ram, ADDR_P2_HEALTH)
    p1_char = _u8(ram, ADDR_P1_CHARACTER)
    p2_char = _u8(ram, ADDR_P2_CHARACTER)
    timer = _u8(ram, ADDR_TIMER)
    continue_timer = _u8(ram, ADDR_CONTINUE_TIMER)
    match_counter = _u8(ram, ADDR_MATCH_COUNTER)
    p1_rounds = _round_count(_u8(ram, ADDR_P1_ROUNDS))
    p2_rounds = _round_count(_u8(ram, ADDR_P2_ROUNDS))
    # Full-health fight-ready cannot already be a 0-2 loss; that byte is leftover.
    if (
        p1_health == MAX_HEALTH
        and p2_health == MAX_HEALTH
        and timer > 50
        and p1_rounds == 0
        and p2_rounds >= 2
    ):
        p2_rounds = 0
    p1_facing = _facing(p1_x, p2_x, True)
    p2_facing = _facing(p1_x, p2_x, False)
    p1_state = _u8(ram, P1_OBJ + OFF_STATE)
    p2_state = _u8(ram, P2_OBJ + OFF_STATE)
    p1 = _read_fighter(p1_x, p1_y, p1_state, p1_health, p1_rounds, p1_char, p1_facing)
    p2 = _read_fighter(p2_x, p2_y, p2_state, p2_health, p2_rounds, p2_char, p2_facing)
    screen = classify_screen(
        p1_health=p1_health,
        p2_health=p2_health,
        timer=timer,
        continue_timer=continue_timer,
        p1_character=p1_char,
        match_counter=match_counter,
        p1_rounds=p1_rounds,
        p2_rounds=p2_rounds,
    )
    return FightSnapshot(
        p1_health=p1_health,
        p2_health=p2_health,
        timer=timer,
        continue_timer=continue_timer,
        p1_character=p1_char,
        p2_character=p2_char,
        p1_rounds=p1_rounds,
        p2_rounds=p2_rounds,
        match_counter=match_counter,
        game_mode=_u8(ram, ADDR_GAME_MODE),
        p1=p1,
        p2=p2,
        ram_len=int(len(ram)),
        screen=screen,
    )


def char_name(char_id: int) -> str:
    return CHAR_NAMES.get(char_id, f"char_{char_id}")


V3_DIM = 20
PUNCH_RANGE = 48


def snapshot_features(
    snap: FightSnapshot,
    prev_health: tuple[int, int] = (MAX_HEALTH, MAX_HEALTH),
) -> tuple[np.ndarray, tuple[int, int]]:
    """20-dim RAM+hitbox vector for MLP PPO (fresh v3; not compatible with v1/v2)."""
    p1_delta = (snap.p1_health - prev_health[0]) / MAX_HEALTH
    p2_delta = (snap.p2_health - prev_health[1]) / MAX_HEALTH
    in_range = 1.0 if snap.distance_x <= PUNCH_RANGE else 0.0
    values = [
        snap.p1_health / MAX_HEALTH,
        snap.p2_health / MAX_HEALTH,
        float(np.clip(p1_delta, -1.0, 1.0)),
        float(np.clip(p2_delta, -1.0, 1.0)),
        snap.timer / MAX_TIMER,
        snap.p2_character / 8.0,
        snap.p1_rounds / 2.0,
        snap.p2_rounds / 2.0,
        snap.match_counter / 11.0,
        snap.p1.x / 255.0,
        snap.p2.x / 255.0,
        snap.p1.y / 255.0,
        snap.p2.y / 255.0,
        snap.distance_x / 255.0,
        snap.p1.state / 255.0,
        snap.p2.state / 255.0,
        1.0 if snap.p1.facing > 0 else 0.0,
        in_range,
        1.0 if snap.bodies_overlap else 0.0,
        1.0 if snap.p1_hit_connects else 0.0,
    ]
    vector = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    return vector, (snap.p1_health, snap.p2_health)


def make_test_ram(**fields: int) -> np.ndarray:
    """Build a synthetic WRAM buffer for unit tests."""
    ram = np.zeros(0x2000, dtype=np.uint8)
    defaults = {
        "p1_health": MAX_HEALTH,
        "p2_health": MAX_HEALTH,
        "timer": 99,
        "continue_timer": 0,
        "p1_character": LIU_KANG_ID,
        "p2_character": 0,
        "p1_rounds": 0,
        "p2_rounds": 0,
        "match_counter": 0,
        "game_mode": 0,
        "p1_x": 80,
        "p1_y": 180,
        "p2_x": 180,
        "p2_y": 180,
        "p1_state": 0,
        "p2_state": 0,
    }
    defaults.update(fields)
    ram[ADDR_P1_HEALTH] = defaults["p1_health"]
    ram[ADDR_P2_HEALTH] = defaults["p2_health"]
    ram[ADDR_TIMER] = defaults["timer"]
    ram[ADDR_CONTINUE_TIMER] = defaults["continue_timer"]
    ram[ADDR_P1_CHARACTER] = defaults["p1_character"]
    ram[ADDR_P2_CHARACTER] = defaults["p2_character"]
    ram[ADDR_P1_ROUNDS] = defaults["p1_rounds"]
    ram[ADDR_P2_ROUNDS] = defaults["p2_rounds"]
    ram[ADDR_MATCH_COUNTER] = defaults["match_counter"]
    ram[ADDR_GAME_MODE] = defaults["game_mode"]
    ram[P1_OBJ + OFF_X] = defaults["p1_x"]
    ram[P1_OBJ + OFF_Y] = defaults["p1_y"]
    ram[P1_OBJ + OFF_STATE] = defaults["p1_state"]
    ram[P2_OBJ + OFF_X] = defaults["p2_x"]
    ram[P2_OBJ + OFF_Y] = defaults["p2_y"]
    ram[P2_OBJ + OFF_STATE] = defaults["p2_state"]
    ram[ADDR_P1_X] = defaults["p1_x"]
    ram[ADDR_P1_Y] = defaults["p1_y"]
    ram[ADDR_P2_X] = defaults["p2_x"]
    ram[ADDR_P2_Y] = defaults["p2_y"]
    return ram
