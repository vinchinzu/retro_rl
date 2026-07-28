"""RAM fields and snapshots for Metroid (NES).

System RAM ($0000–$07FF) comes from ``env.get_ram()``. Cartridge WRAM
(equipment, missiles, tanks at $6877+) is exposed by fceumm as memory blocks
starting at $6000 — read via ``env.data.memory.extract``.

Addresses from Data Crystal / Dirty McDingus disassembly; verified live
against fceumm (2026-07-27).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

# --- Engine / mode (system RAM) ---
ADDR_ENGINE_MODE = 0x001D  # 0 = game, 1 = title/password
ADDR_GAME_MODE = 0x001E  # 3 = playing, 5 = paused, 8 = intro settle
ADDR_TITLE_MODE = 0x001F
ADDR_PAUSED = 0x0031
ADDR_FRAME = 0x002D

# --- Samus / room ---
ADDR_SAMUS_DIR = 0x004D  # 0 = right, 1 = left
ADDR_MAP_Y = 0x004F
ADDR_MAP_X = 0x0050
ADDR_SAMUS_SCREEN_X = 0x0051
ADDR_SAMUS_SCREEN_Y = 0x0052
ADDR_IN_DOOR = 0x0056
ADDR_ROOM_LAYOUT = 0x005A
ADDR_AREA = 0x0074  # often 0 early; $10 Brinstar once set

# Samus object (room coordinates)
ADDR_SAMUS_STATUS = 0x0300
ADDR_SAMUS_Y = 0x030D  # room Y
ADDR_SAMUS_X = 0x030E  # room X

# Health (system RAM)
ADDR_HEALTH_LO = 0x0106
ADDR_HEALTH_HI = 0x0107
ADDR_ITEM_PAUSE = 0x0109
ADDR_MISSILES_ENABLED = 0x010E

# Cartridge WRAM (via memory.extract, not get_ram)
ADDR_ENERGY_TANKS = 0x6877
ADDR_EQUIPMENT = 0x6878
ADDR_MISSILES = 0x6879
ADDR_MISSILE_CAPACITY = 0x687A

# Equipment bits ($6878)
EQUIP_BOMBS = 0x01
EQUIP_HIGH_JUMP = 0x02
EQUIP_LONG_BEAM = 0x04
EQUIP_SCREW_ATTACK = 0x08
EQUIP_MORPH = 0x10  # Maru Mari
EQUIP_VARIA = 0x20
EQUIP_WAVE_BEAM = 0x40
EQUIP_ICE_BEAM = 0x80

ENGINE_GAME = 0
ENGINE_TITLE = 1
GAME_MODE_PLAYING = 3
GAME_MODE_PAUSED = 5
GAME_MODE_INTRO = 8

# Verified start + morph cells (map_x, map_y)
START_MAP_X = 3
START_MAP_Y = 14
MORPH_MAP_X = 1
MORPH_MAP_Y = 14

# Probe-reachable east cells (same row as start; door beyond still WIP)
EAST_CORRIDOR_MAP_X = 5
EAST_CORRIDOR_MAP_Y = 14


class _Memory(Protocol):
    def extract(self, address: int, type: str) -> int: ...


class _Data(Protocol):
    memory: _Memory


class _Env(Protocol):
    def get_ram(self) -> np.ndarray: ...

    data: _Data


def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])


def read_wram_u8(env: Any, addr: int) -> int:
    """Read one byte from fceumm-mapped memory (system RAM or cart WRAM)."""
    return int(env.data.memory.extract(addr, "|u1"))


def read_equipment(env: Any) -> int:
    return read_wram_u8(env, ADDR_EQUIPMENT)


def read_missile_capacity(env: Any) -> int:
    return read_wram_u8(env, ADDR_MISSILE_CAPACITY)


def read_missiles(env: Any) -> int:
    return read_wram_u8(env, ADDR_MISSILES)


def read_energy_tanks(env: Any) -> int:
    return read_wram_u8(env, ADDR_ENERGY_TANKS)


@dataclass(frozen=True)
class MetroidSnapshot:
    """Frame snapshot for routing and segment stop predicates."""

    engine_mode: int
    game_mode: int
    paused: int
    map_x: int
    map_y: int
    samus_x: int
    samus_y: int
    samus_dir: int
    in_door: int
    room_layout: int
    area: int
    health_lo: int
    health_hi: int
    item_pause: int
    missiles_enabled: int
    samus_status: int
    frame_counter: int
    equipment: int = 0
    missiles: int = 0
    missile_capacity: int = 0
    energy_tanks: int = 0

    @property
    def playing(self) -> bool:
        return (
            self.engine_mode == ENGINE_GAME
            and self.game_mode == GAME_MODE_PLAYING
            and self.paused == 0
        )

    @property
    def controllable(self) -> bool:
        """True once intro settle is done and play mode is live."""
        return (
            self.engine_mode == ENGINE_GAME
            and self.game_mode == GAME_MODE_PLAYING
            and self.paused == 0
            and self.map_x < 0xF0
        )

    @property
    def on_title(self) -> bool:
        return self.engine_mode == ENGINE_TITLE

    @property
    def morph_ball(self) -> bool:
        return bool(self.equipment & EQUIP_MORPH)

    @property
    def bombs(self) -> bool:
        return bool(self.equipment & EQUIP_BOMBS)

    @property
    def has_missiles(self) -> bool:
        return self.missile_capacity > 0 or self.missiles_enabled != 0

    @property
    def health_units(self) -> int:
        ones = (self.health_lo >> 4) & 0x0F
        tens = self.health_hi & 0x0F
        tanks = (self.health_hi >> 4) & 0x0F
        return tanks * 100 + tens * 10 + ones

    @property
    def map_cell(self) -> tuple[int, int]:
        return (self.map_x, self.map_y)

    @property
    def door_transition(self) -> bool:
        return self.in_door != 0

    def at_map(self, x: int, y: int) -> bool:
        return self.map_x == x and self.map_y == y


def read_snapshot(ram: np.ndarray, env: Any | None = None) -> MetroidSnapshot:
    """Read a routing snapshot.

    Pass ``env`` to include WRAM equipment/missiles/tanks. Without ``env``,
    those fields stay 0 (system-RAM-only view).
    """
    equipment = missiles = capacity = tanks = 0
    if env is not None:
        equipment = read_equipment(env)
        missiles = read_missiles(env)
        capacity = read_missile_capacity(env)
        tanks = read_energy_tanks(env)
    return MetroidSnapshot(
        engine_mode=read_u8(ram, ADDR_ENGINE_MODE),
        game_mode=read_u8(ram, ADDR_GAME_MODE),
        paused=read_u8(ram, ADDR_PAUSED),
        map_x=read_u8(ram, ADDR_MAP_X),
        map_y=read_u8(ram, ADDR_MAP_Y),
        samus_x=read_u8(ram, ADDR_SAMUS_X),
        samus_y=read_u8(ram, ADDR_SAMUS_Y),
        samus_dir=read_u8(ram, ADDR_SAMUS_DIR),
        in_door=read_u8(ram, ADDR_IN_DOOR),
        room_layout=read_u8(ram, ADDR_ROOM_LAYOUT),
        area=read_u8(ram, ADDR_AREA),
        health_lo=read_u8(ram, ADDR_HEALTH_LO),
        health_hi=read_u8(ram, ADDR_HEALTH_HI),
        item_pause=read_u8(ram, ADDR_ITEM_PAUSE),
        missiles_enabled=read_u8(ram, ADDR_MISSILES_ENABLED),
        samus_status=read_u8(ram, ADDR_SAMUS_STATUS),
        frame_counter=read_u8(ram, ADDR_FRAME),
        equipment=equipment,
        missiles=missiles,
        missile_capacity=capacity,
        energy_tanks=tanks,
    )


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Brinstar gameplay is controllable (not title/intro)."""
    snap = read_snapshot(ram)
    if snap.engine_mode != ENGINE_GAME:
        return False
    if snap.game_mode != GAME_MODE_PLAYING:
        return False
    if snap.paused:
        return False
    if snap.map_x > 0x20 or snap.map_y > 0x20:
        return False
    # Health bytes live after game init (start energy is 30 → hi nibble/lo).
    if snap.health_lo == 0 and snap.health_hi == 0:
        return False
    if obs_mean is not None and obs_mean <= 8.0:
        return False
    return True


def is_morph_obtained(env: Any) -> bool:
    return bool(read_equipment(env) & EQUIP_MORPH)


def is_missiles_obtained(env: Any) -> bool:
    """True once first missile expansion has raised capacity ($687A > 0)."""
    return read_missile_capacity(env) > 0


def is_morph_room(snap: MetroidSnapshot) -> bool:
    return snap.at_map(MORPH_MAP_X, MORPH_MAP_Y)


def capabilities_from_snapshot(snap: MetroidSnapshot) -> frozenset[str]:
    caps: set[str] = set()
    if snap.morph_ball:
        caps.add("morph_ball")
    if snap.bombs:
        caps.add("bombs")
    if snap.equipment & EQUIP_LONG_BEAM:
        caps.add("long_beam")
    if snap.equipment & EQUIP_ICE_BEAM:
        caps.add("ice_beam")
    if snap.equipment & EQUIP_WAVE_BEAM:
        caps.add("wave_beam")
    if snap.equipment & EQUIP_VARIA:
        caps.add("varia_suit")
    if snap.equipment & EQUIP_HIGH_JUMP:
        caps.add("hi_jump")
    if snap.equipment & EQUIP_SCREW_ATTACK:
        caps.add("screw_attack")
    if snap.has_missiles:
        caps.add("missiles")
    return frozenset(caps)


def parse_game_state(
    ram: np.ndarray,
    frame: int = 0,
    obs_mean: float | None = None,
    env: Any | None = None,
) -> GameState:
    """Project confirmed fields into shared ``GameState``."""
    snap = read_snapshot(ram, env=env)
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    caps = capabilities_from_snapshot(snap)
    extras = {
        "level1_ready": ready,
        "engine_mode": snap.engine_mode,
        "game_mode": snap.game_mode,
        "map_x": snap.map_x,
        "map_y": snap.map_y,
        "samus_x": snap.samus_x,
        "samus_y": snap.samus_y,
        "area": snap.area,
        "equipment": snap.equipment,
        "morph_ball": snap.morph_ball,
        "missiles": snap.missiles,
        "missile_capacity": snap.missile_capacity,
        "energy_tanks": snap.energy_tanks,
        "samus_status": snap.samus_status,
        "health_units": snap.health_units,
        "capabilities": sorted(caps),
    }
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if ready else GameMode.MENU,
        stage=int(snap.area),
        room=(snap.map_y << 8) | snap.map_x,
        player_x=snap.samus_x,
        player_y=snap.samus_y,
        health=snap.health_units,
        lives=0,
        enemies=(),
        extras=extras,
    )
