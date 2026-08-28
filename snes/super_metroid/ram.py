"""Typed Super Metroid WRAM state used by navigation and assist guards.

Addresses are WRAM offsets.  The game-state names and resource addresses are
cross-checked against the local reverse-engineered source used by the previous
project.  Route confidence and live-probe evidence are documented in
``docs/ram_map.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any  # env protocol objects for WRAM helpers

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
# Special Setting Mode moonwalk (PJBoy RAM map). 0 = off, 1 = on.
# Default-off on every new file; gameplay reads this WRAM copy.
ADDR_MOONWALK = 0x09E4
ADDR_MAX_RESERVE_HEALTH = 0x09D4
ADDR_RESERVE_HEALTH = 0x09D6
ADDR_SAMUS_POSE = 0x0A1C
# Facing (u8): 4 = left, 8 = right. Movement type (u8) is the next byte.
ADDR_SAMUS_FACING = 0x0A1E
ADDR_MOVEMENT_TYPE = 0x0A1F
# Shine-spark charge / crystal-flash shared timer (source: 0A68).
ADDR_SHINESPARK_TIMER = 0x0A68
ADDR_SAMUS_X = 0x0AF6
ADDR_SAMUS_X_SUB = 0x0AF8
ADDR_SAMUS_Y = 0x0AFA
ADDR_SAMUS_Y_SUB = 0x0AFC
# Vertical: subpixel then pixel (unsigned); direction at 0B36 (0 ground, 1 up, 2 down).
ADDR_VELOCITY_Y_SUB = 0x0B2C
ADDR_VELOCITY_Y = 0x0B2E
ADDR_VERTICAL_DIRECTION = 0x0B36
# Speed-booster counter word: hi = echoes charge (0–4+), lo = anim tick.
# 0B3C non-zero gates permanent blue-suit conversion of temp blue.
ADDR_SPEED_FLAG = 0x0B3C
ADDR_SPEED_COUNTER = 0x0B3E
# Horizontal speed (pixels/sub) then separate momentum (mockball / dash carry).
ADDR_VELOCITY_X = 0x0B42
ADDR_VELOCITY_X_SUB = 0x0B44
ADDR_MOMENTUM_X = 0x0B46
ADDR_MOMENTUM_X_SUB = 0x0B48
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
ADDR_DOOR_DEF_PTR = 0x078D
ADDR_INVINCIBILITY_TIMER = 0x18A8
ADDR_KNOCKBACK_TIMER = 0x18AA

# Facing nibble values (0A1E).
FACING_LEFT = 0x04
FACING_RIGHT = 0x08

# Game-state enum ($0998). Controllers use these — do not re-encode in rooms.
GS_ORDINARY = 8
GS_DEAD = frozenset({26, 36})  # Samus dying / game over (fail-fast)
GS_CERES_LEAVE = frozenset({32, 33, 34})  # Ceres success / Zebes load

# stable-retro maps bank $7E WRAM as a 128 KiB block at this base.
SNES_WRAM_BANK = 0x7E0000

MORPH_BALL_MASK = 0x0004
BOMBS_MASK = 0x1000
VARIA_MASK = 0x0001
HI_JUMP_MASK = 0x0100
# Event 0x0E is set when Mother Brain dies and the escape door sequence starts.
EVENT_MOTHER_BRAIN_DEFEATED = 0x0E
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
    if game_state == GS_ORDINARY:
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
    # Door-entry kinematics (defaults keep manual SuperMetroidState(...) ergonomic).
    door_def_ptr: int = 0
    samus_x_sub: int = 0
    samus_y_sub: int = 0
    velocity_x_sub: int = 0
    velocity_y_sub: int = 0
    momentum_x: int = 0
    momentum_x_sub: int = 0
    # Hi byte of $0B3E — speed-booster charge / echo level (0–4+).
    speed_counter: int = 0
    speed_flag: int = 0
    vertical_direction: int = 0
    facing: int = 0
    movement_type: int = 0
    shinespark_timer: int = 0
    # $09E4 Special Setting Mode; 1 = moonwalk on (required for moonfall).
    moonwalk: int = 0

    @property
    def morph_ball(self) -> bool:
        return bool(self.collected_items & MORPH_BALL_MASK)

    @property
    def bombs(self) -> bool:
        return bool(self.collected_items & BOMBS_MASK)

    @property
    def varia(self) -> bool:
        return bool(self.collected_items & VARIA_MASK)

    @property
    def hi_jump(self) -> bool:
        return bool(self.collected_items & HI_JUMP_MASK)

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

    @property
    def facing_left(self) -> bool:
        return self.facing == FACING_LEFT

    @property
    def facing_right(self) -> bool:
        return self.facing == FACING_RIGHT

    @property
    def speed_boosting(self) -> bool:
        """True when speed-booster charge has reached echo threshold (≥4)."""
        return self.speed_counter >= 4

    @property
    def shinesparking(self) -> bool:
        return self.shinespark_timer > 0

    @property
    def moonwalk_enabled(self) -> bool:
        return self.moonwalk != 0

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

    def kinematics_dict(self) -> dict[str, Any]:
        """Compact leave/entry kinematics for door-transition reports."""
        return {
            "frame": self.frame,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "samus_x": self.samus_x,
            "samus_x_sub": self.samus_x_sub,
            "samus_y": self.samus_y,
            "samus_y_sub": self.samus_y_sub,
            "velocity_x": self.velocity_x,
            "velocity_x_sub": self.velocity_x_sub,
            "velocity_y": self.velocity_y,
            "velocity_y_sub": self.velocity_y_sub,
            "momentum_x": self.momentum_x,
            "momentum_x_sub": self.momentum_x_sub,
            "speed_counter": self.speed_counter,
            "speed_flag": self.speed_flag,
            "speed_boosting": self.speed_boosting,
            "vertical_direction": self.vertical_direction,
            "facing": self.facing,
            "movement_type": self.movement_type,
            "shinespark_timer": self.shinespark_timer,
            "moonwalk": self.moonwalk,
            "moonwalk_enabled": self.moonwalk_enabled,
            "pose": self.pose,
            "door_transition": self.door_transition,
            "transition_direction": self.transition_direction,
            "door_def_ptr": self.door_def_ptr,
            "door_def_ptr_hex": f"0x{self.door_def_ptr:04X}",
            "game_state": self.game_state,
            "phase": self.phase.value,
        }

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["phase"] = self.phase.value
        data["room_id_hex"] = f"0x{self.room_id:04X}"
        data["area_name"] = self.area_name
        data["morph_ball"] = self.morph_ball
        data["bombs"] = self.bombs
        data["varia"] = self.varia
        data["hi_jump"] = self.hi_jump
        data["speed_boosting"] = self.speed_boosting
        data["shinesparking"] = self.shinesparking
        data["moonwalk_enabled"] = self.moonwalk_enabled
        data["facing_left"] = self.facing_left
        data["facing_right"] = self.facing_right
        data["door_def_ptr_hex"] = f"0x{self.door_def_ptr:04X}"
        return data


def _wram_mapped_address(address: int) -> int:
    """Map a WRAM offset to the stable-retro memory key."""
    return SNES_WRAM_BANK + address if address >= 0x2000 else address


def read_wram_u8(env: Any, address: int) -> int:
    """Read one WRAM byte without copying the full 128 KiB bank."""
    mapped = _wram_mapped_address(address)
    # stable-retro exposes bank $7E as a contiguous block; low addresses also
    # appear via get_ram(). Prefer direct block peeks for high WRAM.
    if address >= 0x2000:
        block = env.data.memory.blocks[SNES_WRAM_BANK]
        return int(np.frombuffer(block, dtype=np.uint8, count=1, offset=address)[0])
    ram = env.get_ram()
    return int(ram[address])


def read_wram_u16(env: Any, address: int) -> int:
    """Read one little-endian WRAM word without a full-bank copy."""
    lo = read_wram_u8(env, address)
    hi = read_wram_u8(env, address + 1)
    return lo | (hi << 8)


def peek_wram(env: Any, addresses: dict[str, int]) -> dict[str, int]:
    """Selective WRAM peeks: ``{name: address}`` → ``{name: u8_or_u16}``.

    Values use u16 for known multi-byte navigation addresses, else u8.
    """
    u16_addrs = {
        ADDR_ROOM_ID,
        ADDR_AREA_INDEX,
        ADDR_DOOR_TRANSITION,
        ADDR_TRANSITION_DIRECTION,
        ADDR_GAME_STATE,
        ADDR_EQUIPPED_ITEMS,
        ADDR_COLLECTED_ITEMS,
        ADDR_EQUIPPED_BEAMS,
        ADDR_COLLECTED_BEAMS,
        ADDR_HEALTH,
        ADDR_MAX_HEALTH,
        ADDR_MISSILES,
        ADDR_MAX_MISSILES,
        ADDR_SUPER_MISSILES,
        ADDR_MAX_SUPER_MISSILES,
        ADDR_POWER_BOMBS,
        ADDR_MAX_POWER_BOMBS,
        ADDR_SELECTED_ITEM,
        ADDR_MAX_RESERVE_HEALTH,
        ADDR_RESERVE_HEALTH,
        ADDR_SAMUS_POSE,
        ADDR_SHINESPARK_TIMER,
        ADDR_SAMUS_X,
        ADDR_SAMUS_X_SUB,
        ADDR_SAMUS_Y,
        ADDR_SAMUS_Y_SUB,
        ADDR_VELOCITY_Y_SUB,
        ADDR_VELOCITY_Y,
        ADDR_VERTICAL_DIRECTION,
        ADDR_SPEED_FLAG,
        ADDR_SPEED_COUNTER,
        ADDR_VELOCITY_X,
        ADDR_VELOCITY_X_SUB,
        ADDR_MOMENTUM_X,
        ADDR_MOMENTUM_X_SUB,
        ADDR_NUM_ENEMIES,
        ADDR_ENEMIES_KILLED,
        ADDR_ENEMY0_X,
        ADDR_ENEMY0_Y,
        ADDR_ENEMY0_HP,
        ADDR_ENEMY0_SPRITEMAP,
        ADDR_INVINCIBILITY_TIMER,
        ADDR_KNOCKBACK_TIMER,
        ADDR_DOOR_DEF_PTR,
        ADDR_MOONWALK,
    }
    out: dict[str, int] = {}
    for name, address in addresses.items():
        if address in u16_addrs:
            out[name] = read_wram_u16(env, address)
        else:
            out[name] = read_wram_u8(env, address)
    return out


def read_bank7e_wram(env: Any) -> np.ndarray:
    """Return a copy of SNES bank $7E WRAM (128 KiB).

    ``env.get_ram()`` is reliable for low WRAM (``$7E:0000``–``$7E:1FFF``) but
    returns open-bus garbage for high addresses such as event/boss flags at
    ``$7E:D820``. Prefer this helper whenever those fields matter.

    For hot controller loops prefer :func:`read_wram_u16` / :func:`peek_wram`
    or :func:`parse_env_state` with ``mode="nav"``.
    """
    blocks = env.data.memory.blocks
    raw = blocks[SNES_WRAM_BANK]
    return np.frombuffer(raw, dtype=np.uint8).copy()


def write_wram_u8(env: Any, address: int, value: int) -> None:
    """Write one WRAM byte. Uses bank $7E for addresses at or above ``0x2000``."""
    mapped = SNES_WRAM_BANK + address if address >= 0x2000 else address
    env.data.memory.assign(mapped, "|u1", value & 0xFF)


def write_wram_u16(env: Any, address: int, value: int) -> None:
    """Write one little-endian WRAM word."""
    mapped = SNES_WRAM_BANK + address if address >= 0x2000 else address
    env.data.memory.assign(mapped, "<u2", value & 0xFFFF)


def set_moonwalk(env: Any, enabled: bool = True) -> bool:
    """Set Special Setting Mode moonwalk (``$09E4``).

    Returns True when a write happened. This is a file-option poke, not
    progression — see ``docs/ASSIST_CONTRACT.md``. Gameplay reads this
    WRAM copy; SRAM is only involved if the file is saved afterward.
    """
    want = 1 if enabled else 0
    if read_wram_u16(env, ADDR_MOONWALK) == want:
        return False
    write_wram_u16(env, ADDR_MOONWALK, want)
    return True


def set_event_flag(env: Any, event_id: int) -> None:
    """Set one bit in ``events_that_happened`` (``$7E:D820`` bitfield)."""
    byte_index = event_id >> 3
    bit = 1 << (event_id & 7)
    address = ADDR_EVENT_FLAGS + byte_index
    current = int(read_bank7e_wram(env)[address])
    write_wram_u8(env, address, current | bit)


def parse_state(ram: np.ndarray, *, frame: int = 0) -> SuperMetroidState:
    """Parse one full WRAM snapshot.

    Pass bank-$7E WRAM from :func:`read_bank7e_wram` when event/boss flags are
    needed. Low-only ``env.get_ram()`` slices still work for combat/nav fields
    below ``0x2000``.
    """
    game_state = _u16(ram, ADDR_GAME_STATE)
    door_transition = _u16(ram, ADDR_DOOR_TRANSITION)
    event_end = min(len(ram), ADDR_BOSS_BITS)
    boss_end = min(len(ram), ADDR_BOSS_BITS + 8)
    event_flags = tuple(int(value) for value in ram[ADDR_EVENT_FLAGS:event_end])
    boss_bits = tuple(int(value) for value in ram[ADDR_BOSS_BITS:boss_end])
    if len(event_flags) < 8:
        event_flags = event_flags + (0,) * (8 - len(event_flags))
    if len(boss_bits) < 8:
        boss_bits = boss_bits + (0,) * (8 - len(boss_bits))
    return SuperMetroidState(
        frame=frame,
        game_state=game_state,
        phase=phase_for_game_state(game_state, door_transition),
        room_id=_u16(ram, ADDR_ROOM_ID),
        area_index=_u16(ram, ADDR_AREA_INDEX),
        door_transition=door_transition,
        transition_direction=_u16(ram, ADDR_TRANSITION_DIRECTION),
        door_def_ptr=_u16(ram, ADDR_DOOR_DEF_PTR),
        samus_x=_u16(ram, ADDR_SAMUS_X),
        samus_x_sub=_u16(ram, ADDR_SAMUS_X_SUB),
        samus_y=_u16(ram, ADDR_SAMUS_Y),
        samus_y_sub=_u16(ram, ADDR_SAMUS_Y_SUB),
        velocity_x=_i16(ram, ADDR_VELOCITY_X),
        velocity_x_sub=_u16(ram, ADDR_VELOCITY_X_SUB),
        velocity_y=_i16(ram, ADDR_VELOCITY_Y),
        velocity_y_sub=_u16(ram, ADDR_VELOCITY_Y_SUB),
        momentum_x=_i16(ram, ADDR_MOMENTUM_X),
        momentum_x_sub=_u16(ram, ADDR_MOMENTUM_X_SUB),
        # Hi byte is the speed-booster charge level used by TAS/speedrun tech.
        speed_counter=(_u16(ram, ADDR_SPEED_COUNTER) >> 8) & 0xFF,
        speed_flag=_u16(ram, ADDR_SPEED_FLAG),
        vertical_direction=_u16(ram, ADDR_VERTICAL_DIRECTION),
        facing=_u8(ram, ADDR_SAMUS_FACING),
        movement_type=_u8(ram, ADDR_MOVEMENT_TYPE),
        shinespark_timer=_u16(ram, ADDR_SHINESPARK_TIMER),
        moonwalk=_u16(ram, ADDR_MOONWALK),
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
        event_flags=event_flags[:8],
        boss_bits=boss_bits[:8],
    )


# Optional process-wide parse counters (profiling long continuous / pure runs).
_PARSE_COUNTS: dict[str, int] = {"nav": 0, "full": 0}


def reset_parse_counts() -> None:
    """Zero :func:`parse_env_state` mode counters (tests / long-run profiles)."""
    _PARSE_COUNTS["nav"] = 0
    _PARSE_COUNTS["full"] = 0


def parse_counts() -> dict[str, int]:
    """Return a copy of nav/full parse counts since last reset."""
    return dict(_PARSE_COUNTS)


def parse_env_state(
    env: Any,
    *,
    frame: int = 0,
    mode: str = "full",
) -> SuperMetroidState:
    """Parse state from the emulator.

    Parameters
    ----------
    mode:
        ``"full"`` — copy bank $7E (correct event/boss flags at ``$D820+``).
        ``"nav"`` — low WRAM via ``env.get_ram()`` (fast; high fields zero-padded).

    Use ``nav`` in tight wait loops and pure geometry probes; switch to
    ``full`` (or :func:`read_wram_u8` peeks) when event/boss integrity matters.
    Hot controller paths should go through :class:`StateCache` or a session
    that already holds per-step state (see :class:`routes.runtime.RouteSession`).
    """
    if mode == "nav":
        _PARSE_COUNTS["nav"] = _PARSE_COUNTS.get("nav", 0) + 1
        return parse_state(env.get_ram(), frame=frame)
    if mode != "full":
        raise ValueError(f"unknown parse_env_state mode {mode!r} (use 'full' or 'nav')")
    _PARSE_COUNTS["full"] = _PARSE_COUNTS.get("full", 0) + 1
    return parse_state(read_bank7e_wram(env), frame=frame)


def probe_pin(state: SuperMetroidState) -> dict[str, object]:
    """Compact residual pin for pure/geometry cards (PROCESS residual schema)."""
    return {
        "room": f"0x{state.room_id:04X}",
        "roomId": state.room_id,
        "pose": state.pose,
        "x": state.samus_x,
        "y": state.samus_y,
        "x_sub": state.samus_x_sub,
        "y_sub": state.samus_y_sub,
        "door_transition": state.door_transition,
        "phase": state.phase.name if hasattr(state.phase, "name") else str(state.phase),
        "frame": state.frame,
        "velocity_x": state.velocity_x,
        "velocity_y": state.velocity_y,
        "momentum_x": state.momentum_x,
        "speed_counter": state.speed_counter,
        "speed_boosting": state.speed_boosting,
        "facing": state.facing,
        "movement_type": state.movement_type,
        "shinespark_timer": state.shinespark_timer,
        "moonwalk": state.moonwalk,
        "door_def_ptr": f"0x{state.door_def_ptr:04X}",
        "collected_items": f"0x{state.collected_items:04X}",
    }


class StateCache:
    """Optional per-frame SuperMetroidState cache for continuous / probe loops.

    Default mode is ``nav`` (low WRAM only). Call :meth:`invalidate` after every
    emulator step, or pass the current frame so a stale cache is rebuilt.

    Prefer this (or session-owned state) over bare :func:`parse_env_state` in
    tight wait loops so accidental full-bank copies stay rare.

    :meth:`stats` reports **cache-local** parse counts (nav/full misses) in
    addition to hits/misses so long pure probes can avoid relying solely on
    process-global :func:`parse_counts`.
    """

    def __init__(self, env: Any, *, mode: str = "nav") -> None:
        if mode not in ("nav", "full"):
            raise ValueError(f"StateCache mode must be 'nav' or 'full', got {mode!r}")
        self.env = env
        self.mode = mode
        self._frame: int | None = None
        self._state: SuperMetroidState | None = None
        self.hits = 0
        self.misses = 0
        self.nav_parses = 0
        self.full_parses = 0

    def invalidate(self) -> None:
        self._frame = None
        self._state = None

    def reset_stats(self) -> None:
        """Zero hit/miss and local parse counters (session-scoped profiles)."""
        self.hits = 0
        self.misses = 0
        self.nav_parses = 0
        self.full_parses = 0

    def get(self, *, frame: int = 0, mode: str | None = None) -> SuperMetroidState:
        use_mode = mode if mode is not None else self.mode
        if use_mode not in ("nav", "full"):
            raise ValueError(f"unknown StateCache mode {use_mode!r}")
        if (
            self._state is not None
            and self._frame == frame
            and use_mode == self.mode
        ):
            self.hits += 1
            return self._state
        self.misses += 1
        if use_mode == "nav":
            self.nav_parses += 1
        else:
            self.full_parses += 1
        self.mode = use_mode
        self._frame = frame
        self._state = parse_env_state(self.env, frame=frame, mode=use_mode)
        return self._state

    def stats(self) -> dict[str, int | str]:
        """Hit/miss + cache-local parse counters for long pure/continuous loops."""
        return {
            "mode": self.mode,
            "hits": self.hits,
            "misses": self.misses,
            "nav_parses": self.nav_parses,
            "full_parses": self.full_parses,
        }
