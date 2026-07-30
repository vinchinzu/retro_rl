"""Post-Spore Super / Pink PB room ids and evidence dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

ROOM_SUPER = 0x9B5B
ROOM_FARMING = 0xA0A4
ROOM_BIG_PINK = 0x9D19
ROOM_PINK_PB = 0x9E11


@dataclass(frozen=True)
class SuperCollectEvidence:
    entry_frame: int
    collect_frame: int
    exit_frame: int | None
    max_super_missiles: int
    final_room_id: int
    samus_x: int
    samus_y: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PowerBombEvidence:
    entry_frame: int
    collect_frame: int | None
    max_super_missiles: int
    max_power_bombs: int
    final_room_id: int
    samus_x: int
    samus_y: int
    reached_big_pink: bool
    reached_pb_room: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
