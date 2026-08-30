"""Level 8 chapter specs and exact stop predicates.

No L8 dungeon room has been observed live.  Room IDs from walkthrough grids
must not become ``DungeonRoomSpec`` registrations or executable claims.  These
chapter contracts stay route-ineligible until RAM evidence supplies the
topology and the integrator recomposes them from the natural predecessor.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import TF_BIT_L8
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

LEVEL8 = 8
TF_BEFORE_LEVEL8 = 0x7F
TF_AFTER_LEVEL8 = 0xFF


@dataclass(frozen=True)
class Level8Topology:
    """Observed room anchors only; ``None`` explicitly means unknown."""

    entry_room: int | None = None
    magic_key_room: int | None = None
    boss_room: int | None = None
    triforce_room: int | None = None
    evidence: str = "hypothesis"
    route_eligible: bool = False


UNOBSERVED_LEVEL8_TOPOLOGY = Level8Topology()


@dataclass(frozen=True)
class Level8ChapterSpec:
    chapter_id: str
    objective: str
    required_inventory: tuple[str, ...]
    omitted_optional_items: tuple[str, ...] = ()
    evidence: str = "hypothesis"
    route_eligible: bool = False
    max_frames: int = 30_000


ENTRY_TO_MAGIC_KEY_SPEC = Level8ChapterSpec(
    chapter_id="level8_entry_to_magic_key",
    objective="live entry to natural Magical Key acquisition",
    required_inventory=("red_candle", "bow", "wooden_arrows"),
    omitted_optional_items=("book", "map", "compass"),
)

MAGIC_KEY_TO_SHARD_SPEC = Level8ChapterSpec(
    chapter_id="level8_magic_key_to_shard",
    objective="Magical Key through confirmed four-head Gleeok, heart, and shard",
    required_inventory=("magic_key", "bow", "wooden_arrows"),
    omitted_optional_items=("book", "map", "compass"),
)


@dataclass(frozen=True)
class Level8ClearEndpoint:
    """Measured settled post-fanfare endpoint; unknown in Wave A."""

    level: int | None = None
    screen: int | None = None
    mode: int | None = None
    incoming_heart_containers: int | None = None
    outgoing_heart_containers: int | None = None
    evidence: str = "hypothesis"
    route_eligible: bool = False

    def complete(self) -> bool:
        return (
            self.route_eligible
            and self.level is not None
            and self.screen is not None
            and self.mode is not None
            and self.incoming_heart_containers is not None
            and self.outgoing_heart_containers is not None
        )


UNOBSERVED_LEVEL8_CLEAR = Level8ClearEndpoint()


def level8_entry_stop(
    snap: ZeldaSnapshot,
    *,
    candle: int,
    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY,
) -> bool:
    """Exact live entry; refuses the claim while the entry room is unknown."""
    return (
        topology.route_eligible
        and topology.entry_room is not None
        and snap.level == LEVEL8
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == topology.entry_room
        and snap.triforce == TF_BEFORE_LEVEL8
        and int(candle) == 2
    )


def level8_magic_key_stop(
    snap: ZeldaSnapshot,
    *,
    magic_key: int,
    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY,
) -> bool:
    """Natural Magical Key boundary at a RAM-observed room."""
    return (
        topology.route_eligible
        and topology.magic_key_room is not None
        and snap.level == LEVEL8
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == topology.magic_key_room
        and snap.triforce == TF_BEFORE_LEVEL8
        and int(magic_key) >= 1
    )


def level8_clear_stop(
    snap: ZeldaSnapshot,
    *,
    magic_key: int,
    endpoint: Level8ClearEndpoint = UNOBSERVED_LEVEL8_CLEAR,
) -> bool:
    """Shard plus one natural heart at the measured settled L8 leave."""
    if not endpoint.complete():
        return False
    return (
        snap.level == endpoint.level
        and snap.screen == endpoint.screen
        and snap.mode == endpoint.mode
        and not snap.transitioning
        and snap.triforce == TF_AFTER_LEVEL8
        and bool(snap.triforce & TF_BIT_L8)
        and int(magic_key) >= 1
        and snap.health_is_full
        and snap.heart_containers == endpoint.outgoing_heart_containers
        and endpoint.outgoing_heart_containers
        == int(endpoint.incoming_heart_containers) + 1
    )


__all__ = [
    "ENTRY_TO_MAGIC_KEY_SPEC",
    "LEVEL8",
    "Level8ChapterSpec",
    "Level8ClearEndpoint",
    "Level8Topology",
    "MAGIC_KEY_TO_SHARD_SPEC",
    "TF_AFTER_LEVEL8",
    "TF_BEFORE_LEVEL8",
    "UNOBSERVED_LEVEL8_CLEAR",
    "UNOBSERVED_LEVEL8_TOPOLOGY",
    "level8_clear_stop",
    "level8_entry_stop",
    "level8_magic_key_stop",
]
