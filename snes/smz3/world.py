"""World detection bridge: SMZ3 sessions reuse vanilla game packages.

SMZ3 is one ROM that portals between Super Metroid and ALttP. While the
player is in a given world, bot policy should call into the existing
``super_metroid`` / ``alttp`` RAM parsers and (eventually) controllers rather
than reimplementing them.

Combo flag (upstream ``sram.asm``): ``!SRAM_CURRENT_GAME = $a173fe``
(NMI 8-bit: 0 = ALTTP, negative = SM, positive nonzero = credits). That bus
address is not in stable-retro ``get_ram()`` / known memory blocks yet, so
detection uses dual WRAM heuristics verified on the test seed combo ROM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Union

import numpy as np

from smz3.ram import (
    SM_ENGINE_GAME_STATES,
    SM_MENU_GAME_STATES,
    SM_ORDINARY_GAME_STATE,
    Z3_ACTIVE_MODULES,
    Z3_MENU_MODULES,
    ComboSnapshot,
    read_snapshot,
)

# Cave id after Crateria map → fortune teller portal (combo teleport table).
Z3_FORTUNE_TELLER_CAVE = 0x0122

RamLike = Union[bytes, bytearray, np.ndarray, ComboSnapshot, None]


class ActiveWorld(str, Enum):
    """Which game logic currently owns the controller."""

    SUPER_METROID = "super_metroid"
    ALTTP = "alttp"
    UNKNOWN = "unknown"
    MENU = "menu"


class WorldAdapter(Protocol):
    """Minimal surface a world backend must provide to the race/session loop."""

    name: ActiveWorld

    def parse_state(self, ram: Any, *, frame: int = 0) -> Any: ...

    def room_key(self, state: Any) -> str: ...

    def is_settled(self, state: Any) -> bool: ...


@dataclass(frozen=True)
class WorldContext:
    """Resolved world + package import path for logging/routing."""

    world: ActiveWorld
    package: str | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "world": self.world.value,
            "package": self.package,
            "notes": self.notes,
        }


# Documented reuse targets (import when implementing live sessions).
VANILLA_PACKAGES: dict[ActiveWorld, str] = {
    ActiveWorld.SUPER_METROID: "super_metroid",
    ActiveWorld.ALTTP: "alttp",
}


def context_for(world: ActiveWorld) -> WorldContext:
    if world is ActiveWorld.SUPER_METROID:
        return WorldContext(
            world=world,
            package="super_metroid",
            notes="Reuse super_metroid.ram / room_timer / route controllers",
        )
    if world is ActiveWorld.ALTTP:
        return WorldContext(
            world=world,
            package="alttp",
            notes="Reuse alttp.ram / overworld / opening route primitives",
        )
    if world is ActiveWorld.MENU:
        return WorldContext(world=world, package=None, notes="File select / title")
    return WorldContext(
        world=ActiveWorld.UNKNOWN,
        package=None,
        notes="WRAM heuristic inconclusive; need portal session or SRAM flag",
    )


def detect_world_from_snapshot(snap: ComboSnapshot) -> ActiveWorld:
    """Core classifier operating on a parsed snapshot."""
    gs = snap.sm_game_state
    room_ok = snap.sm_room_plausible or (
        gs == SM_ORDINARY_GAME_STATE and snap.sm_room_id != 0
    )

    # Strong SM: ordinary controllable gameplay.
    if snap.sm_controllable:
        return ActiveWorld.SUPER_METROID

    # SM engine owns the machine (boot, doors, pause, death, etc.).
    if gs in SM_ENGINE_GAME_STATES and room_ok:
        if gs in SM_MENU_GAME_STATES:
            return ActiveWorld.MENU
        return ActiveWorld.SUPER_METROID

    if gs in SM_ENGINE_GAME_STATES and gs in SM_MENU_GAME_STATES:
        # Title/file select before a room id is loaded.
        if snap.z3_module not in Z3_ACTIVE_MODULES:
            return ActiveWorld.MENU

    # Z3 owns the machine when SM state is not a known SM value.
    if gs not in SM_ENGINE_GAME_STATES:
        # Portal residue: fortune-teller cave id after Crateria map teleport.
        if (snap.z3_room_id & 0xFFFF) == Z3_FORTUNE_TELLER_CAVE:
            return ActiveWorld.ALTTP
        if snap.z3_module in Z3_ACTIVE_MODULES or snap.z3_controllable:
            return ActiveWorld.ALTTP
        if snap.z3_module in Z3_MENU_MODULES:
            return ActiveWorld.MENU

    # Ambiguous: SM-looking state without room, or garbage both sides.
    if gs in SM_ENGINE_GAME_STATES:
        if gs in SM_MENU_GAME_STATES:
            return ActiveWorld.MENU
        # e.g. mid-load with room not yet set
        return ActiveWorld.SUPER_METROID

    return ActiveWorld.UNKNOWN


def detect_world(
    ram_or_snap: RamLike = None,
    *,
    frame: int = 0,
) -> ActiveWorld:
    """Classify active world from combo WRAM (or a :class:`ComboSnapshot`).

    Priority (verified on SMZ3 test seed power-on → Landing Site):

    1. SM ordinary gameplay / engine states with a plausible SM room pointer
       → ``SUPER_METROID`` (or ``MENU`` for title/file-select states).
    2. Z3 active modules when SM game_state is not a known SM engine value
       → ``ALTTP``.
    3. Otherwise ``UNKNOWN``.

    Pass ``None`` only for scaffolding tests; returns ``UNKNOWN``.
    """
    if ram_or_snap is None:
        return ActiveWorld.UNKNOWN
    if isinstance(ram_or_snap, ComboSnapshot):
        snap = ram_or_snap
    else:
        snap = read_snapshot(ram_or_snap, frame=frame)
    return detect_world_from_snapshot(snap)


def detect_world_stub(ram: bytes | bytearray | None = None) -> ActiveWorld:
    """Back-compat alias for :func:`detect_world`."""
    return detect_world(ram)


@dataclass
class DualWorldSessionHooks:
    """Scaffold for a future dual-bot / dual-world race session.

    One seed ROM, two bot instances (same seed), each with its own emulator
    process and room-timeout watchdog. Video capture is part of the quest
    artifact for every serious run.
    """

    seed_name: str
    record_video: bool = True
    room_timeout_multiplier: float = 3.0
    bots: int = 2

    def plan(self) -> dict[str, Any]:
        return {
            "seed_name": self.seed_name,
            "bots": self.bots,
            "record_video": self.record_video,
            "room_timeout_multiplier": self.room_timeout_multiplier,
            "world_packages": {
                ActiveWorld.SUPER_METROID.value: VANILLA_PACKAGES[
                    ActiveWorld.SUPER_METROID
                ],
                ActiveWorld.ALTTP.value: VANILLA_PACKAGES[ActiveWorld.ALTTP],
            },
            "status": "scaffold",
            "next": (
                "Boot combo ROM → detect_world → dispatch to vanilla "
                "package controllers; 3× room baseline ends bot run."
            ),
        }
