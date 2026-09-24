"""Level 9 natural-spine endpoint specs and stop predicates.

Navigation stays out of this module.  First-quest Magical Key topology is a
labeled hypothesis until RAM observes each room; recon fixtures are not
natural-prefix evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from zelda_i.level9.ganon import credits_rolling, final_ending_screen
from zelda_i.level9.patra import (
    NORTH_DOOR,
    PATRA_EYE_COUNT,
    final_patra_live,
    patra_eyes,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

LEVEL9 = 9
FULL_TRIFORCE = 0xFF
ROOM_LEVEL9_ENTRY = 0x76
ROOM_OLD_MAN_TF = 0x66
ROOM_SILVER_ARROWS_HYP = 0x10
ROOM_SUFFIX_JOIN = 0x41
ROOM_FINAL_PATRA = 0x52
ROOM_KEESE_CORRIDOR = 0x62
ROOM_RED_RING_HYP = 0x07
SILVER_ARROWS = 2
MAGICAL_SWORD = 3
# The natural route's sword ceiling. The Magical Sword needs 12 heart
# containers and the power-on run reaches Level 9 with 10, so it is out of
# reach; the White Sword needs 5 and is taken on the way in (overworld/
# white_sword.py, live 2026-09-07). Ending stops therefore require
# WHITE_SWORD, not MAGICAL_SWORD -- the boss policies are hitbox-driven and
# swing until the boss dies, so a weaker sword costs frames, not outcomes.
WHITE_SWORD = 2

# Magical Key minimum (Red Ring excluded). Cellars 0x60/0x70/0x75/0x67/0x77.
L9_SELECTED_PREFIX_ROOMS: tuple[int, ...] = (
    0x76, 0x66, 0x65, 0x55, 0x60, 0x14, 0x15, 0x16, 0x06, 0x05,
    0x70, 0x63, 0x62, 0x61, 0x75, 0x20, 0x10,
)
L9_SELECTED_JOIN_ROOMS: tuple[int, ...] = (
    0x10, 0x20, 0x61, 0x51, 0x41, 0x31, 0x30, 0x67, 0x04, 0x03, 0x77, 0x52,
)

MISSING_POST_L8_LEFTOVER = "post_l8_ow_leftover_unmeasured"
MISSING_SPECTACLE_BOMB = (
    "spectacle_rock_0x05_bomb_entrance_unverified_from_post_l8"
)
MISSING_OLD_MAN_GATE = "old_man_room_0x66_full_tf_gate_unobserved"
MISSING_SILVER_ARROW_ROOM = "silver_arrow_room_0x10_unobserved"
MISSING_51_NORTH_WALK = "0x51_north_dest_walk_unverified_statue_diamond"
TRIFORCE_NOT_FULL = "triforce_not_0xff"
BOMBS_NOT_NATURAL = "bombs_not_natural"


@dataclass(frozen=True)
class Level9EndpointSpec:
    """Public chapter boundary without implying a live walk to that boundary."""

    through: str
    stop: str
    evidence: str
    description: str
    predicate: Callable[..., bool] | None


@dataclass(frozen=True)
class PostLevel8Handoff:
    """Measured L8 leave required before natural L9 overworld may move."""

    screen: int | None = None
    link_x: int | None = None
    link_y: int | None = None
    keys: int | None = None
    bombs: int | None = None
    rupees: int | None = None
    heart_containers: int | None = None
    selected_item: int | None = None
    magic_key: int | None = None
    bow: int | None = None
    arrows: int | None = None
    xy_tolerance: int = 4
    evidence: str = "hypothesis"
    verified: bool = False
    route_eligible: bool = False

    def complete(self) -> bool:
        measured = (
            self.screen, self.link_x, self.link_y, self.keys, self.bombs,
            self.rupees, self.heart_containers, self.selected_item,
            self.magic_key, self.bow, self.arrows,
        )
        return self.verified and all(value is not None for value in measured)

    def mismatch(self, snap: ZeldaSnapshot) -> str | None:
        if snap.triforce != FULL_TRIFORCE:
            return TRIFORCE_NOT_FULL
        if self.bombs is not None and snap.bombs < self.bombs:
            return BOMBS_NOT_NATURAL
        if self.bombs is None and snap.bombs < 1:
            return BOMBS_NOT_NATURAL
        if not self.complete():
            return MISSING_POST_L8_LEFTOVER
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return "post_l8_not_settled_overworld"
        if snap.screen != self.screen:
            return "post_l8_screen_mismatch"
        return None


UNMEASURED_POST_L8_HANDOFF = PostLevel8Handoff()

# rr-6o7.3 / rr-ps7.5: `--through level8` is power-on spine-green
# (`level8_ow_leave_settle` stage, natl8_3). The shard fanfare returns Link to
# OW `0x6D` `(96,93)` mode 5 with TF `0xFF`, Magical Key 1, heart containers 14,
# bombs 0, rupees 38. Link leaves L8 with 0 bombs and purchases bombs on the
# post-L8 walk to Spectacle Rock.
# Keys/bombs/rupees are recorded for `complete()` but `mismatch()` only gates
# on screen / level / mode / TF / bombs>=floor.
MEASURED_POST_L8_HANDOFF = PostLevel8Handoff(
    screen=0x6D,
    link_x=96,
    link_y=93,
    keys=0,
    bombs=0,
    rupees=38,
    heart_containers=14,
    selected_item=0,
    magic_key=1,
    bow=1,
    arrows=1,
    evidence="spine-green",
    verified=True,
    route_eligible=True,
)


def level9_entry_snapshot_stop(snap: ZeldaSnapshot) -> bool:
    """Snapshot-visible part of the natural L9 entry contract."""
    return (
        snap.level == LEVEL9
        and snap.screen == ROOM_LEVEL9_ENTRY
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
        and snap.bombs > 0
    )


def level9_entry_stop(snap: ZeldaSnapshot, *, magic_key: bool) -> bool:
    """Play 0x76, TF 0xFF, Magic Key, natural/declared bombs. No RAM writes."""
    return level9_entry_snapshot_stop(snap) and magic_key


def level9_silver_arrows_stop(
    snap: ZeldaSnapshot,
    *,
    room: int | None,
) -> bool:
    """ADDR_ARROWS==2, Bow owned, TF 0xFF, in the selected Silver Arrow room."""
    return (
        room is not None
        and snap.level == LEVEL9
        and snap.screen == room
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
        and snap.bow > 0
        and snap.arrows == SILVER_ARROWS
    )


def level9_live_patra_stop(snap: ZeldaSnapshot) -> bool:
    """Live uncleared Patra 0x52; natural prefix, not a fixture census rewrite."""
    return (
        snap.level == LEVEL9
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
        and snap.bow > 0
        and snap.arrows == SILVER_ARROWS
        and snap.screen == ROOM_FINAL_PATRA
        and snap.sword >= WHITE_SWORD
        and final_patra_live(snap)
        and len(patra_eyes(snap)) == PATRA_EYE_COUNT
        and not (snap.cur_opened_doors & NORTH_DOOR)
    )


def level9_credits_stop(snap: ZeldaSnapshot, *, deaths: int = 0) -> bool:
    """Credits/final page, deaths 0. Writes are a controller-report contract."""
    return deaths == 0 and (credits_rolling(snap) or final_ending_screen(snap))


L9_ENTRY_ENDPOINT = Level9EndpointSpec(
    through="level9-entry",
    stop="level9_entry_0x76",
    evidence="hypothesis",
    description="post-L8 leftover, Spectacle Rock 0x05 bomb, Old Man 0x66, play 0x76",
    predicate=level9_entry_snapshot_stop,
)
L9_SILVER_ARROWS_ENDPOINT = Level9EndpointSpec(
    through="level9-silver-arrows",
    stop="level9_silver_arrows",
    evidence="hypothesis",
    description="natural Silver Arrows in hypothesized room 0x10 (ADDR_ARROWS==2)",
    predicate=None,
)
L9_PATRA_ENDPOINT = Level9EndpointSpec(
    through="level9-patra",
    stop="level9_live_patra_0x52",
    evidence="hypothesis",
    description="natural join 0x41 suffix into live uncleared Patra 0x52",
    predicate=level9_live_patra_stop,
)
L9_CREDITS_ENDPOINT = Level9EndpointSpec(
    through="level9-credits",
    stop="level9_credits",
    evidence="fixture-live",
    description="write-free Patra, Ganon, Zelda, and ending input policies",
    predicate=level9_credits_stop,
)

L9_ENDPOINTS = (
    L9_ENTRY_ENDPOINT,
    L9_SILVER_ARROWS_ENDPOINT,
    L9_PATRA_ENDPOINT,
    L9_CREDITS_ENDPOINT,
)

L9_PUBLIC_THROUGH: tuple[str, ...] = tuple(spec.through for spec in L9_ENDPOINTS)

__all__ = [
    "BOMBS_NOT_NATURAL",
    "FULL_TRIFORCE",
    "L9_CREDITS_ENDPOINT",
    "L9_ENDPOINTS",
    "L9_ENTRY_ENDPOINT",
    "L9_PATRA_ENDPOINT",
    "L9_PUBLIC_THROUGH",
    "L9_SELECTED_JOIN_ROOMS",
    "L9_SELECTED_PREFIX_ROOMS",
    "L9_SILVER_ARROWS_ENDPOINT",
    "LEVEL9",
    "MAGICAL_SWORD",
    "WHITE_SWORD",
    "MISSING_51_NORTH_WALK",
    "MISSING_OLD_MAN_GATE",
    "MISSING_POST_L8_LEFTOVER",
    "MISSING_SILVER_ARROW_ROOM",
    "MISSING_SPECTACLE_BOMB",
    "PostLevel8Handoff",
    "ROOM_FINAL_PATRA",
    "ROOM_KEESE_CORRIDOR",
    "ROOM_LEVEL9_ENTRY",
    "ROOM_OLD_MAN_TF",
    "ROOM_RED_RING_HYP",
    "ROOM_SILVER_ARROWS_HYP",
    "ROOM_SUFFIX_JOIN",
    "SILVER_ARROWS",
    "TRIFORCE_NOT_FULL",
    "MEASURED_POST_L8_HANDOFF",
    "UNMEASURED_POST_L8_HANDOFF",
    "Level9EndpointSpec",
    "level9_credits_stop",
    "level9_entry_stop",
    "level9_entry_snapshot_stop",
    "level9_live_patra_stop",
    "level9_silver_arrows_stop",
]
