"""Level 8 chapter specs, hypothesis door graph, and exact stop predicates.

One L8 room (the entry, screen 0x7E) has been observed live via isolated
fixture recon (rr-6o7.1: OW 0x6D bush burn from a fixture-only stand, not a
natural walk) -- see ``LIVE_RECON_LEVEL8_TOPOLOGY`` and the "entry" row of
``LEVEL8_HYPOTHESIS_ROOMS``, both ``evidence="live_recon_fixture"`` and
``route_eligible=False``.  Every other room remains unobserved.  Walkthrough
grid labels stay hypothesis only: they must not become ``DungeonRoomSpec``
registrations, executable room IDs, or route-eligible claims.  Magical Key is
the selected investment for the L9 key bottleneck; Book/Map/Compass stay
omitted until live topology proves a cheaper detour.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import TF_BIT_L8
from zelda_i.dungeon.engine import DungeonRoomSpec
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

LEVEL8 = 8
TF_BEFORE_LEVEL8 = 0x7F
TF_AFTER_LEVEL8 = 0xFF
# Four-head Gleeok body type is unknown.  L4 is 0x43 and L6 is 0x44; do not
# treat the absent 0x45 as the L8 body.
GLEEOK_FOUR_HEAD_OBJECT_TYPE: int | None = None
BLUE_GOHMA_ARROWS_REQUIRED = 3


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

# Fixture-only live recon (rr-6o7.1): entry room observed as screen 0x7E,
# Link facing UP at (120, 205), via nes/zelda_i/scratch/level8_bush_burn_sweep.py
# + capture_level8_entrance_fixture.py from Level8BushWithCandleFixture (not
# the natural post-L7 walk). See Level8EntranceReconFixture.provenance.json.
# ``route_eligible`` stays False: this is fixture-derived evidence, not a
# natural-entry / route promotion.
LIVE_RECON_LEVEL8_TOPOLOGY = Level8Topology(
    entry_room=0x7E,
    evidence="live_recon_fixture",
    route_eligible=False,
)
LEVEL8_ROOM_SPECS: tuple[DungeonRoomSpec, ...] = ()


@dataclass(frozen=True)
class Level8InteriorRoomRecon:
    """A single interior room observed live from a disclosed fixture replay.

    Development recon only: ``route_eligible`` is always False and these rows
    are never attached to ``L8_THROUGH`` or promoted to ``DungeonRoomSpec``.
    They only record what RAM showed, with a 2/2 byte-identical recording.
    """

    room_id: int
    entered_from: int
    entry_direction: str
    entry_gate: str
    entry_pose: tuple[int, int]
    keys_in: int
    keys_out: int
    bombs_in: int
    bombs_out: int
    room_item_id: int
    census: tuple[tuple[int, int, int], ...]  # (type_id, hp, count)
    evidence: str = "live_recon_fixture"
    route_eligible: bool = False
    fixture: str = ""
    recording_tag: str = ""


# rr-6o7.2 groundwork: ONE guarded replay past the confirmed 0x4E arrival.
# From Level8InteriorReconFixture, reuse the confirmed 0x7E -> clear 0x6E ->
# bomb-N 0x5E -> clear/center-key -> shutter-N 0x4E policy unchanged, then take
# the north KEY door from 0x4E.  2/2 byte-identical (probe l8_4e_north B1/B2,
# frames 2918): one natural key spent (10 -> 9), bombs unchanged, settled play
# room 0x3E at (120,205), deaths 0, progression_writes 0, capacity_writes 0.
# The mixed 0x4E census was NOT cleared.  Not route eligible; not on L8_THROUGH.
LEVEL8_INTERIOR_0X3E_RECON = Level8InteriorRoomRecon(
    room_id=0x3E,
    entered_from=0x4E,
    entry_direction="UP",
    entry_gate="north_key_door",
    entry_pose=(120, 205),
    keys_in=10,
    keys_out=9,
    bombs_in=7,
    bombs_out=7,
    room_item_id=0x03,
    census=((0x0C, 128, 6),),  # 6 blue Darknuts, room_all_dead=0
    fixture="Level8Interior3EReconFixture",
    recording_tag="l8_4e_north_fixture_B1",
)

# rr-6o7.2: ONE guarded boundary past 0x3E.  Same probe start
# (Level8InteriorReconFixture) and the same confirmed 0x7E -> clear 0x6E ->
# bomb-N 0x5E -> clear/center-key -> shutter-N 0x4E -> key-N 0x3E policy
# replayed unchanged, then the 0x3E census cleared with the sword only and ONE
# bomb placed at the north wall from (120,105) facing UP.
#
# 2/2 byte-identical (probe l8_3e_north A2/A3, 4788 frames each; 32 payload
# keys compared, only the screenshot paths and the A3-only saved_fixture entry
# differ).  Live RAM: first settled play room 0x2E at (120,189), keys 9 -> 9,
# bombs 7 -> 6 (exactly one natural bomb), deaths 0, progression_writes 0,
# capacity_writes 0, direct runtime writes 0.  Assist = Survival health refill
# only (26 health writes, max single-frame damage 2).
#
# Gate evidence, in 0x3E: arrival cur_opened_doors/open_doorway_mask = 0x04
# (the DOWN key door we came through).  Clearing the six 0x0C bodies raised
# only the RIGHT bit (0x04 -> 0x05) -- i.e. the 0x3E clear opens an EAST
# shutter, never the north.  The UP bit (doors 0x0D, mask 0x0C) appeared only
# after the bomb blast, so the north gate of 0x3E is a bomb wall, matching the
# hypothesis edge "blue_darknuts -> map_manhandla UP bomb".  0x3E is an open
# floor with two statue blocks on the mid row (x~96 and x~144 at y~141) that
# wedge a naive stand walk; the bomb stand must be approached along the clear
# north band (y=109).
#
# 0x2E census names Manhandla from dungeon/ids.py (0x3C).  Its four heads plus
# body occupy five slots at HP 64.  Four transient 0x56 projectile residuals
# (HP 240) were also live in the settled census; they are projectile state,
# not room population, so they are not recorded as census rows.  Development
# recon only: not route eligible, no DungeonRoomSpec, not on L8_THROUGH, and
# the hypothesis graph keeps room_id=None for every non-entry node.
LEVEL8_INTERIOR_0X2E_RECON = Level8InteriorRoomRecon(
    room_id=0x2E,
    entered_from=0x3E,
    entry_direction="UP",
    entry_gate="north_bomb_wall",
    entry_pose=(120, 189),
    keys_in=9,
    keys_out=9,
    bombs_in=7,
    bombs_out=6,
    room_item_id=0x17,  # ids.room_item_name -> dungeon_map_walkthrough_correlated
    census=((0x3C, 64, 5),),  # one Manhandla (body + 4 heads), room_all_dead=0
    fixture="Level8Interior2EReconFixture",
    recording_tag="l8_3e_north_fixture_20260904_A2",
)

LEVEL8_INTERIOR_ROOM_RECON: tuple[Level8InteriorRoomRecon, ...] = (
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_0X2E_RECON,
)


@dataclass(frozen=True)
class Level8HypothesisRoom:
    """Walkthrough node. ``room_id`` stays None until RAM observes it."""

    name: str
    grid_col: int
    grid_row: int
    feature: str
    on_magic_key_route: bool = False
    on_gleeok_route: bool = False
    omitted: bool = False
    room_id: int | None = None
    evidence: str = "hypothesis"


@dataclass(frozen=True)
class Level8HypothesisExit:
    src: str
    dst: str
    direction: str
    gate: str
    notes: str = ""
    evidence: str = "hypothesis"


# GameFAQs / Zelda Dungeon first-quest lion shape. Columns A=0 … E=4,
# rows 1=0 … 8=7.  Entrance is D8.  Hex room IDs remain unobserved.
LEVEL8_HYPOTHESIS_ROOMS: tuple[Level8HypothesisRoom, ...] = (
    # room_id + evidence: live fixture recon (rr-6o7.1), not a route claim.
    # See LIVE_RECON_LEVEL8_TOPOLOGY / Level8EntranceReconFixture.provenance.json.
    Level8HypothesisRoom(
        "entry", 3, 7, "south_mouth", True, True,
        room_id=0x7E, evidence="live_recon_fixture",
    ),
    Level8HypothesisRoom("east_key", 4, 7, "optional_key", omitted=True),
    Level8HypothesisRoom("west_manhandla", 2, 7, "manhandla", omitted=True),
    Level8HypothesisRoom(
        "book_stairs", 1, 7, "book_of_magic_stairs", omitted=True
    ),
    Level8HypothesisRoom("north_manhandla", 3, 6, "manhandla_bomb_north", True),
    Level8HypothesisRoom("darknut_key", 3, 5, "blue_darknuts_key", True),
    Level8HypothesisRoom("pols_key", 2, 5, "pols_gibdo_key", omitted=True),
    Level8HypothesisRoom("red_darknut_key", 1, 5, "red_darknuts_key", omitted=True),
    Level8HypothesisRoom("compass", 4, 5, "pols_compass", omitted=True),
    Level8HypothesisRoom("shutter_darknuts", 3, 4, "key_up_mixed", True, True),
    Level8HypothesisRoom(
        "optional_gohma_west", 2, 4, "blue_gohma_hint", omitted=True
    ),
    Level8HypothesisRoom("blue_darknuts", 3, 3, "blue_darknuts_bomb_north", True, True),
    Level8HypothesisRoom("passage_east", 4, 3, "kill_clear_stairs", False, True),
    Level8HypothesisRoom("map_manhandla", 3, 2, "manhandla_map_optional", True, True),
    Level8HypothesisRoom("blue_gohma", 3, 1, "blue_gohma_three_arrows", True, True),
    Level8HypothesisRoom("magic_key_stairs", 4, 1, "magical_key_cellar", True),
    Level8HypothesisRoom("optional_gohma_west2", 2, 1, "blue_gohma", omitted=True),
    Level8HypothesisRoom("pols_west", 2, 3, "pols_bomb_north", False, True),
    Level8HypothesisRoom("gleeok", 1, 3, "four_head_gleeok", False, True),
    Level8HypothesisRoom("triforce", 1, 2, "shard_8", False, True),
)

LEVEL8_HYPOTHESIS_EXITS: tuple[Level8HypothesisExit, ...] = (
    Level8HypothesisExit("entry", "east_key", "RIGHT", "open", "optional_key"),
    Level8HypothesisExit("entry", "west_manhandla", "LEFT", "open", "book_detour"),
    Level8HypothesisExit(
        "west_manhandla", "book_stairs", "LEFT", "kill_clear", "book_omitted"
    ),
    Level8HypothesisExit("entry", "north_manhandla", "UP", "open"),
    Level8HypothesisExit(
        "north_manhandla", "darknut_key", "UP", "bomb", "north_wall"
    ),
    Level8HypothesisExit("darknut_key", "pols_key", "LEFT", "open", "extra_keys"),
    Level8HypothesisExit("pols_key", "red_darknut_key", "LEFT", "open", "extra_keys"),
    Level8HypothesisExit("darknut_key", "compass", "RIGHT", "key", "compass_omitted"),
    Level8HypothesisExit("darknut_key", "shutter_darknuts", "UP", "key"),
    Level8HypothesisExit("shutter_darknuts", "blue_darknuts", "UP", "key"),
    Level8HypothesisExit(
        "blue_darknuts", "map_manhandla", "UP", "bomb", "map_kill_optional"
    ),
    Level8HypothesisExit("map_manhandla", "blue_gohma", "UP", "key"),
    Level8HypothesisExit("blue_gohma", "magic_key_stairs", "RIGHT", "kill_clear"),
    Level8HypothesisExit("blue_gohma", "map_manhandla", "DOWN", "open"),
    Level8HypothesisExit("map_manhandla", "blue_darknuts", "DOWN", "open"),
    Level8HypothesisExit(
        "blue_darknuts", "passage_east", "RIGHT", "kill_clear", "gleeok_return"
    ),
    Level8HypothesisExit("passage_east", "pols_west", "STAIRS", "stairs"),
    Level8HypothesisExit("pols_west", "gleeok", "UP", "bomb", "boss_shortcut"),
    Level8HypothesisExit("gleeok", "triforce", "UP", "kill_clear"),
)

MAGIC_KEY_ROUTE: tuple[str, ...] = (
    "entry",
    "north_manhandla",
    "darknut_key",
    "shutter_darknuts",
    "blue_darknuts",
    "map_manhandla",
    "blue_gohma",
    "magic_key_stairs",
)
GLEEOK_ROUTE: tuple[str, ...] = (
    "blue_gohma",
    "map_manhandla",
    "blue_darknuts",
    "passage_east",
    "pols_west",
    "gleeok",
    "triforce",
)
OMITTED_OPTIONAL_ROOMS: tuple[str, ...] = (
    "east_key",
    "west_manhandla",
    "book_stairs",
    "pols_key",
    "red_darknut_key",
    "compass",
    "optional_gohma_west",
    "optional_gohma_west2",
)


def hypothesis_room(name: str) -> Level8HypothesisRoom:
    for room in LEVEL8_HYPOTHESIS_ROOMS:
        if room.name == name:
            return room
    raise KeyError(name)


def hypothesis_room_ids_unobserved() -> bool:
    """True when no room carries a RAM id without disclosed live evidence.

    rr-6o7.1 disclosed exactly one live-recon room id (the "entry" row, via
    fixture-only OW 0x6D burn recon).  A room_id is only ever legitimate
    here when it is paired with ``evidence="live_recon_fixture"`` -- an
    undisclosed/hypothesis room_id is still what this guards against.
    """
    return all(
        room.room_id is None or room.evidence == "live_recon_fixture"
        for room in LEVEL8_HYPOTHESIS_ROOMS
    )


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
    magic_key_before: int | None = None,
) -> bool:
    """Natural Magical Key boundary at a RAM-observed room."""
    gained = int(magic_key) >= 1
    if magic_key_before is not None:
        gained = gained and int(magic_key) > int(magic_key_before)
    return (
        topology.route_eligible
        and topology.magic_key_room is not None
        and snap.level == LEVEL8
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == topology.magic_key_room
        and snap.triforce == TF_BEFORE_LEVEL8
        and gained
    )


def level8_magic_key_ledger(
    snap: ZeldaSnapshot,
    *,
    magic_key_before: int,
    magic_key_after: int,
    keys_before: int,
    bombs_before: int,
) -> dict[str, int]:
    """Record the public gate's incoming/outgoing key and bomb counts."""
    return {
        "keys_before": int(keys_before),
        "keys_after": int(snap.keys),
        "bombs_before": int(bombs_before),
        "bombs_after": int(snap.bombs),
        "magic_key_before": int(magic_key_before),
        "magic_key_after": int(magic_key_after),
        "triforce": int(snap.triforce),
    }


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
    "BLUE_GOHMA_ARROWS_REQUIRED",
    "ENTRY_TO_MAGIC_KEY_SPEC",
    "GLEEOK_FOUR_HEAD_OBJECT_TYPE",
    "GLEEOK_ROUTE",
    "LEVEL8",
    "LEVEL8_HYPOTHESIS_EXITS",
    "LEVEL8_HYPOTHESIS_ROOMS",
    "LEVEL8_INTERIOR_0X2E_RECON",
    "LEVEL8_INTERIOR_0X3E_RECON",
    "LEVEL8_INTERIOR_ROOM_RECON",
    "LEVEL8_ROOM_SPECS",
    "Level8InteriorRoomRecon",
    "Level8ChapterSpec",
    "Level8ClearEndpoint",
    "Level8HypothesisExit",
    "Level8HypothesisRoom",
    "Level8Topology",
    "MAGIC_KEY_ROUTE",
    "MAGIC_KEY_TO_SHARD_SPEC",
    "OMITTED_OPTIONAL_ROOMS",
    "TF_AFTER_LEVEL8",
    "TF_BEFORE_LEVEL8",
    "UNOBSERVED_LEVEL8_CLEAR",
    "UNOBSERVED_LEVEL8_TOPOLOGY",
    "hypothesis_room",
    "hypothesis_room_ids_unobserved",
    "level8_clear_stop",
    "level8_entry_stop",
    "level8_magic_key_ledger",
    "level8_magic_key_stop",
]
