"""Typed Super Metroid WRAM state used by navigation and assist guards.

Addresses are WRAM offsets.  The game-state names and resource addresses are
cross-checked against the local reverse-engineered source used by the previous
project.  Route confidence and live-probe evidence are documented in
``docs/ram_map.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

ADDR_ROOM_ID = 0x079B
ADDR_AREA_INDEX = 0x079F
ADDR_DOOR_TRANSITION = 0x0797
ADDR_TRANSITION_DIRECTION = 0x0791
ADDR_GAME_STATE = 0x0998
ADDR_EQUIPPED_ITEMS = 0x09A2
ADDR_COLLECTED_ITEMS = 0x09A4
ADDR_EQUIPPED_BEAMS = 0x09A6
ADDR_COLLECTED_BEAMS = 0x09A8
ADDR_HEALTH = 0x09C2
ADDR_MAX_HEALTH = 0x09C4
ADDR_MISSILES = 0x09C6
ADDR_MAX_MISSILES = 0x09C8
ADDR_SUPER_MISSILES = 0x09CA
ADDR_MAX_SUPER_MISSILES = 0x09CC
ADDR_POWER_BOMBS = 0x09CE
ADDR_MAX_POWER_BOMBS = 0x09D0
ADDR_SELECTED_ITEM = 0x09D2
ADDR_MAX_RESERVE_HEALTH = 0x09D4
ADDR_RESERVE_HEALTH = 0x09D6
ADDR_SAMUS_POSE = 0x0A1C
ADDR_SAMUS_X = 0x0AF6
ADDR_SAMUS_Y = 0x0AFA
ADDR_VELOCITY_Y = 0x0B2E
ADDR_VELOCITY_X = 0x0B42
ADDR_TIMER_TYPE = 0x0943
ADDR_ESCAPE_TIMER_FRAMES = 0x0945
ADDR_ESCAPE_TIMER_SECONDS = 0x0946
ADDR_ESCAPE_TIMER_MINUTES = 0x0947
ADDR_EVENT_FLAGS = 0xD820
ADDR_BOSS_BITS = 0xD828
ADDR_NUM_ENEMIES = 0x0E4E
ADDR_ENEMIES_KILLED = 0x0E50
ADDR_ENEMY0_X = 0x0F7A
ADDR_ENEMY0_Y = 0x0F7E
ADDR_ENEMY0_HP = 0x0F8C
ADDR_ENEMY0_SPRITEMAP = 0x0F8E

MORPH_BALL_MASK = 0x0004
BOMBS_MASK = 0x1000
AREA_NAMES = (
    "Crateria",
    "Brinstar",
    "Norfair",
    "Wrecked Ship",
    "Maridia",
    "Tourian",
    "Ceres",
)


class GameplayPhase(str, Enum):
    """Coarse phase used to guard legal assist writes."""

    BOOT_OR_MENU = "boot_or_menu"
    ORDINARY_GAMEPLAY = "ordinary_gameplay"
    ROOM_TRANSITION = "room_transition"
    PAUSE_OR_INVENTORY = "pause_or_inventory"
    SCRIPTED_SEQUENCE = "scripted_sequence"
    DEATH_OR_GAME_OVER = "death_or_game_over"
    ENDING_OR_CREDITS = "ending_or_credits"
    UNKNOWN = "unknown"


def _u8(ram: np.ndarray, address: int) -> int:
    return int(ram[address])


def _u16(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def _i16(ram: np.ndarray, address: int) -> int:
    value = _u16(ram, address)
    return value - 0x10000 if value & 0x8000 else value


def phase_for_game_state(game_state: int, door_transition: int = 0) -> GameplayPhase:
    """Map the source-defined game-state enum to an assist phase."""
    if game_state in {0, 1, 2, 3, 4, 5, 6, 30, 31, 40, 41, 42, 43, 44}:
        return GameplayPhase.BOOT_OR_MENU
    if game_state == 8:
        if door_transition:
            return GameplayPhase.ROOM_TRANSITION
        return GameplayPhase.ORDINARY_GAMEPLAY
    if game_state in {7, 9, 10, 11}:
        return GameplayPhase.ROOM_TRANSITION
    if game_state in {12, 13, 14, 15, 16, 17, 18}:
        return GameplayPhase.PAUSE_OR_INVENTORY
    if game_state in {19, 20, 21, 22, 23, 24, 25, 26, 29, 35, 36, 37}:
        return GameplayPhase.DEATH_OR_GAME_OVER
    if game_state in {27, 32, 33, 34}:
        return GameplayPhase.SCRIPTED_SEQUENCE
    if game_state in {38, 39}:
        return GameplayPhase.ENDING_OR_CREDITS
    return GameplayPhase.UNKNOWN


@dataclass(frozen=True)
class SuperMetroidState:
    """Compact state vector for room progress, navigation, and integrity."""

    frame: int
    game_state: int
    phase: GameplayPhase
    room_id: int
    area_index: int
    door_transition: int
    transition_direction: int
    samus_x: int
    samus_y: int
    velocity_x: int
    velocity_y: int
    pose: int
    health: int
    max_health: int
    reserve_health: int
    max_reserve_health: int
    missiles: int
    max_missiles: int
    super_missiles: int
    max_super_missiles: int
    power_bombs: int
    max_power_bombs: int
    selected_item: int
    equipped_items: int
    collected_items: int
    equipped_beams: int
    collected_beams: int
    timer_type: int
    escape_timer_frames: int
    escape_timer_seconds: int
    escape_timer_minutes: int
    num_enemies: int
    enemies_killed: int
    enemy0_x: int
    enemy0_y: int
    enemy0_hp: int
    enemy0_spritemap: int
    event_flags: tuple[int, ...]
    boss_bits: tuple[int, ...]

    @property
    def morph_ball(self) -> bool:
        return bool(self.collected_items & MORPH_BALL_MASK)

    @property
    def bombs(self) -> bool:
        return bool(self.collected_items & BOMBS_MASK)

    @property
    def area_name(self) -> str:
        if 0 <= self.area_index < len(AREA_NAMES):
            return AREA_NAMES[self.area_index]
        return f"Unknown {self.area_index}"

    @property
    def dead(self) -> bool:
        return self.phase is GameplayPhase.DEATH_OR_GAME_OVER

    @property
    def controllable(self) -> bool:
        return self.phase is GameplayPhase.ORDINARY_GAMEPLAY

    def progress_vector(self) -> tuple[int, ...]:
        """Progress identity intentionally richer than coordinates alone."""
        return (
            self.room_id,
            self.area_index,
            self.game_state,
            self.door_transition,
            self.collected_items,
            self.collected_beams,
            self.max_missiles,
            self.max_super_missiles,
            self.max_power_bombs,
            self.enemy0_hp,
            self.num_enemies,
            self.enemies_killed,
            self.timer_type,
            int.from_bytes(self.event_flags, "little"),
            int.from_bytes(self.boss_bits, "little"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["phase"] = self.phase.value
        data["room_id_hex"] = f"0x{self.room_id:04X}"
        data["area_name"] = self.area_name
        data["morph_ball"] = self.morph_ball
        data["bombs"] = self.bombs
        return data


def parse_state(ram: np.ndarray, *, frame: int = 0) -> SuperMetroidState:
    """Parse one full WRAM snapshot."""
    game_state = _u16(ram, ADDR_GAME_STATE)
    door_transition = _u16(ram, ADDR_DOOR_TRANSITION)
    return SuperMetroidState(
        frame=frame,
        game_state=game_state,
        phase=phase_for_game_state(game_state, door_transition),
        room_id=_u16(ram, ADDR_ROOM_ID),
        area_index=_u16(ram, ADDR_AREA_INDEX),
        door_transition=door_transition,
        transition_direction=_u16(ram, ADDR_TRANSITION_DIRECTION),
        samus_x=_u16(ram, ADDR_SAMUS_X),
        samus_y=_u16(ram, ADDR_SAMUS_Y),
        velocity_x=_i16(ram, ADDR_VELOCITY_X),
        velocity_y=_i16(ram, ADDR_VELOCITY_Y),
        pose=_u16(ram, ADDR_SAMUS_POSE),
        health=_u16(ram, ADDR_HEALTH),
        max_health=_u16(ram, ADDR_MAX_HEALTH),
        reserve_health=_u16(ram, ADDR_RESERVE_HEALTH),
        max_reserve_health=_u16(ram, ADDR_MAX_RESERVE_HEALTH),
        missiles=_u16(ram, ADDR_MISSILES),
        max_missiles=_u16(ram, ADDR_MAX_MISSILES),
        super_missiles=_u16(ram, ADDR_SUPER_MISSILES),
        max_super_missiles=_u16(ram, ADDR_MAX_SUPER_MISSILES),
        power_bombs=_u16(ram, ADDR_POWER_BOMBS),
        max_power_bombs=_u16(ram, ADDR_MAX_POWER_BOMBS),
        selected_item=_u16(ram, ADDR_SELECTED_ITEM),
        equipped_items=_u16(ram, ADDR_EQUIPPED_ITEMS),
        collected_items=_u16(ram, ADDR_COLLECTED_ITEMS),
        equipped_beams=_u16(ram, ADDR_EQUIPPED_BEAMS),
        collected_beams=_u16(ram, ADDR_COLLECTED_BEAMS),
        timer_type=_u8(ram, ADDR_TIMER_TYPE),
        escape_timer_frames=_u8(ram, ADDR_ESCAPE_TIMER_FRAMES),
        escape_timer_seconds=_u8(ram, ADDR_ESCAPE_TIMER_SECONDS),
        escape_timer_minutes=_u8(ram, ADDR_ESCAPE_TIMER_MINUTES),
        num_enemies=_u16(ram, ADDR_NUM_ENEMIES),
        enemies_killed=_u16(ram, ADDR_ENEMIES_KILLED),
        enemy0_x=_u16(ram, ADDR_ENEMY0_X),
        enemy0_y=_u16(ram, ADDR_ENEMY0_Y),
        enemy0_hp=_u16(ram, ADDR_ENEMY0_HP),
        enemy0_spritemap=_u16(ram, ADDR_ENEMY0_SPRITEMAP),
        event_flags=tuple(int(value) for value in ram[ADDR_EVENT_FLAGS:ADDR_BOSS_BITS]),
        boss_bits=tuple(int(value) for value in ram[ADDR_BOSS_BITS : ADDR_BOSS_BITS + 8]),
    )
