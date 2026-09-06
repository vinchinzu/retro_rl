"""Ordered L8 Gleeok-suffix composition, inert until a natural predecessor.

The fixture-live gates below (plus the mode-9 ``0x2F`` settle that sits between
two of them) are each 2/2 in isolation from their own pin.  This module is the
one place that records their *order* and their measured pin-to-pin contract, so
that greening ``--through level8`` after rr-6o7.2 (Magical Key, power-on 2/2)
lands is a single explicit flip instead of a re-derivation.

The chain now begins at the Magical-Key return pose (play ``0x1F`` ``(96,157)``)
and runs ``0x1F -> 0x1E -> 0x2E -> 0x3E -> 0x3F -> cellar 0x2F -> 0x4C ->
0x3C`` (Gleeok + heart) ``-> 0x2C`` (shard).  The first three gates
(``west_1f`` / ``south_1e`` / ``south_2e``) were the missing link between the
old suffix table (which started at ``0x3E``) and the live magic-key endpoint.

It is deliberately inert today:

* ``suffix_stages`` returns ``()`` for every lineage that is not both
  ``route_eligible`` and ``natural_predecessor``, and the only lineage constant
  exported here (``FIXTURE_LINEAGE_LEVEL8_SUFFIX``) is neither.  With the empty
  tuple ``zelda_i.level8.hops`` keeps its current fail-closed clear chapter, so
  ``make_gleeok_passage_controller`` is still the first row of ``L8_THROUGH``.
* Even a caller that hand-builds a composable lineage cannot green
  ``--through level8``: ``level8_clear_stop`` still needs a complete
  ``Level8ClearEndpoint``, and ``Level8PostShardOWReconFixture`` is
  fixture-lineage (hc 4, TF ``0xFF`` reached from a fixture pin), **not** a
  Survival-true post-L8 leave, so it may not fill one.

Composition constraint worth keeping: the ``0x3F`` stairs walk-on lands in
mode-9 ``0x2F`` at the K3/K4 leftover ``(208,141)``, but the cellar-cross
(P1/P2) is only measured from the *settled* east-ladder pose ``(192,93)`` that
``scratch/probe_l8_2f_settle.py`` reached after a ~400-600f no-input idle.
``Level8Cellar2FSettleController`` is that idle, so the two measured hops are
never chained across an unmeasured gap.

No RAM writes.  ``route_eligible`` stays False on every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level8.dungeon import (
    LEVEL8,
    LEVEL8_INTERIOR_0X1E_WEST_RECON,
    LEVEL8_INTERIOR_0X2C_TF_RECON,
    LEVEL8_INTERIOR_0X2E_SOUTH_RECON,
    LEVEL8_INTERIOR_0X2F_STAIRS_RECON,
    LEVEL8_INTERIOR_0X3C_KILL_RECON,
    LEVEL8_INTERIOR_0X3C_NORTH_RECON,
    LEVEL8_INTERIOR_0X3E_SOUTH_RECON,
    LEVEL8_INTERIOR_0X3F_EAST_RECON,
    LEVEL8_INTERIOR_0X4C_WEST_RECON,
    Level8InteriorRoomRecon,
)
from zelda_i.level8.gleeok import make_four_head_gleeok_controller
from zelda_i.level8.gleeok_entry import make_bomb_north_4c_controller
from zelda_i.level8.passage import (
    CELLAR_ROOM,
    SOURCE_ROOM,
    SPAWN_XY,
    make_passage_2f_controller,
)
from zelda_i.level8.path import (
    make_east_3e_controller,
    make_south_1e_controller,
    make_south_2e_controller,
    make_west_1f_controller,
)
from zelda_i.level8.stairs import make_stairs_3f_controller
from zelda_i.level8.triforce import (
    make_north_3c_controller,
    make_ow_leave_controller,
    make_shard_2c_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "CELLAR_2F_SETTLE_FRAMES",
    "CELLAR_2F_SETTLE_MAX_FRAMES",
    "FIXTURE_LINEAGE_LEVEL8_SUFFIX",
    "NATURAL_LINEAGE_LEVEL8_SUFFIX",
    "LEVEL8_SUFFIX_GATES",
    "SUFFIX_PREREQUISITE_BEADS",
    "Level8Cellar2FSettleController",
    "Level8SuffixGate",
    "Level8SuffixLineage",
    "make_bomb_north_4c_controller",
    "make_cellar_2f_settle_controller",
    "make_east_3e_controller",
    "make_four_head_gleeok_controller",
    "make_north_3c_controller",
    "make_passage_2f_controller",
    "make_shard_2c_controller",
    "make_stairs_3f_controller",
    "suffix_blockers",
    "suffix_stages",
]

Stage = tuple[str, object, int]

# The suffix may only compose behind L8-A; L7 leave is already measured.
SUFFIX_PREREQUISITE_BEADS: tuple[str, ...] = ("rr-6o7.1",)

# scratch/probe_l8_2f_settle.py S1/S2: the mode-9 room paints after a ~400f
# no-input idle (the probe idled 600f) and Link settles on the east/source
# ladder mouth.  The cross pin is that settled state, never the K3/K4 arrival.
CELLAR_2F_SETTLE_FRAMES = 600
CELLAR_2F_SETTLE_MAX_FRAMES = 1200
_SETTLE_TOL = 8


@dataclass(frozen=True)
class Level8SuffixLineage:
    """Provenance gate for composing the suffix rows.

    ``natural_predecessor`` is the rr-8t4.3 / rr-6o7.1 question: did this run
    reach ``0x3E`` by a power-on Survival walk, or by loading a fixture?  A
    fixture answer keeps the rows out of the chapter entirely.
    """

    evidence: str = "fixture_lineage"
    route_eligible: bool = False
    natural_predecessor: bool = False

    def composable(self) -> bool:
        return bool(self.route_eligible and self.natural_predecessor)


# Fixture lineage: every suffix hop was originally measured from its own pin.
FIXTURE_LINEAGE_LEVEL8_SUFFIX = Level8SuffixLineage()

# rr-6o7.3: the whole suffix is now spine-green from the power-on
# Magical-Key frontier -- `scripts/level8_clear_lab.py` drove all 11 gates
# 2/2 byte-identical from `Level8SuffixEntryLive` (a `--through
# level8-magic-key` power-on pin), settling OW `0x6D` `(192,157)` TF `0xFF`.
# The natural predecessor (`--through level8-magic-key`) is power-on 2/2, so
# this lineage is composable.
NATURAL_LINEAGE_LEVEL8_SUFFIX = Level8SuffixLineage(
    evidence="spine_green_from_magic_key",
    route_eligible=True,
    natural_predecessor=True,
)


@dataclass(frozen=True)
class Level8SuffixGate:
    """One ordered suffix row bound to the recon that measured it."""

    stage: str
    factory: Callable[[], Any]
    recon: Level8InteriorRoomRecon
    note: str = ""

    @property
    def dest_room(self) -> int:
        return int(self.recon.room_id)

    @property
    def origin_room(self) -> int:
        return int(self.recon.entered_from)

    @property
    def gate(self) -> str:
        return self.recon.entry_gate


def _settle_leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "magic_key": int(getattr(snap, "magic_key", 0)),
        "triforce": int(snap.triforce),
    }


@dataclass(kw_only=True)
class Level8Cellar2FSettleController(HopController):
    """Idle the freshly warped mode-9 ``0x2F`` onto the settled east ladder.

    Presses nothing: the room paints itself.  Never UP (that is the source
    ladder back to play ``0x3F``), never a direction at all.
    """

    spec_id: str = "level8_cellar_2f_settle"
    max_frames: int = CELLAR_2F_SETTLE_MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "settled_0x2f_east_ladder"
    settle_frames: int = CELLAR_2F_SETTLE_FRAMES
    pose: tuple[int, int] = SPAWN_XY
    tolerance: int = _SETTLE_TOL
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if self.frames < self.settle_frames:
            return False
        if snap.mode != PASSAGE_MODE or snap.screen != CELLAR_ROOM:
            return False
        px, py = self.pose
        return (
            abs(int(snap.link_x) - px) <= self.tolerance
            and abs(int(snap.link_y) - py) <= self.tolerance
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"m{snap.mode}_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_idle_action(), "settle_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % 12 == 0:
            self.leftover = _settle_leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PLAY_MODE and snap.screen == SOURCE_ROOM:
            return self.mark_fail("returned_source_0x3f")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            return self.mark_fail(f"left_cellar_to_play_0x{snap.screen:02x}")
        if snap.mode == PASSAGE_MODE and snap.screen != CELLAR_ROOM:
            return self.mark_fail(f"unexpected_cellar_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), f"cellar_settle_{snap.mode}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "settle_frames": int(self.settle_frames),
            "pose": list(self.pose),
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "leftover": dict(self.leftover),
        }


def make_cellar_2f_settle_controller() -> Level8Cellar2FSettleController:
    return Level8Cellar2FSettleController()


# Ordered Gleeok suffix.  Every row is 2/2 fixture-live from its own pin; the
# order (and the settle between rows 2 and 3) is what this table adds.  Stage
# names expand rr-6o7.3's ``level8_return_passage`` / ``level8_heart_shard_leave``
# while keeping ``level8_four_head_gleeok`` exactly as the bead names it.
LEVEL8_SUFFIX_GATES: tuple[Level8SuffixGate, ...] = (
    Level8SuffixGate(
        "level8_return_passage_west_1f",
        make_west_1f_controller,
        LEVEL8_INTERIOR_0X1E_WEST_RECON,
        "from the MK-return pose (96,157): cardinal LEFT past the 0x68, "
        "y-align, LEFT push into the already-cleared 0x1E",
    ),
    Level8SuffixGate(
        "level8_return_passage_south_1e",
        make_south_1e_controller,
        LEVEL8_INTERIOR_0X2E_SOUTH_RECON,
        "x-align LEFT to 120, DOWN push; SE-corner occupancy BFS is banned",
    ),
    Level8SuffixGate(
        "level8_return_passage_south_2e",
        make_south_2e_controller,
        LEVEL8_INTERIOR_0X3E_SOUTH_RECON,
        "DOWN the aligned x=120 centre aisle between the y~141 statues; "
        "picks up map 0x17 incidentally",
    ),
    Level8SuffixGate(
        "level8_return_passage_east_3e",
        make_east_3e_controller,
        LEVEL8_INTERIOR_0X3F_EAST_RECON,
        "idle for the RIGHT bit, north band past the statues, RIGHT push",
    ),
    Level8SuffixGate(
        "level8_return_passage_stairs_3f",
        make_stairs_3f_controller,
        LEVEL8_INTERIOR_0X2F_STAIRS_RECON,
        "CheckWarp is tile 0x71 at (193,141); the visual stairs 0x77 are not",
    ),
    Level8SuffixGate(
        "level8_return_passage_cellar_2f_settle",
        make_cellar_2f_settle_controller,
        LEVEL8_INTERIOR_0X2F_STAIRS_RECON,
        "no-input idle from the (208,141) arrival to the (192,93) cross pin",
    ),
    Level8SuffixGate(
        "level8_return_passage_cross_2f",
        make_passage_2f_controller,
        LEVEL8_INTERIOR_0X4C_WEST_RECON,
        "DOWN, floor LEFT, west ladder UP; never UP on the east source ladder",
    ),
    Level8SuffixGate(
        "level8_return_passage_bomb_north_4c",
        make_bomb_north_4c_controller,
        LEVEL8_INTERIOR_0X3C_NORTH_RECON,
        "perimeter to the (120,93) north wall, one bomb; never the centre stairs",
    ),
    Level8SuffixGate(
        "level8_four_head_gleeok",
        make_four_head_gleeok_controller,
        LEVEL8_INTERIOR_0X3C_KILL_RECON,
        "south-stand the live 0x45 body, then the SW heart container",
    ),
    Level8SuffixGate(
        "level8_heart_shard_north_3c",
        make_north_3c_controller,
        LEVEL8_INTERIOR_0X2C_TF_RECON,
        "UP inland then the RAM-open north shutter; never DOWN into 0x4C",
    ),
    Level8SuffixGate(
        "level8_heart_shard_leave_2c",
        make_shard_2c_controller,
        LEVEL8_INTERIOR_0X2C_TF_RECON,
        "walk onto room_item 0x1B; TF 0x80 is a rising edge, not a pin state",
    ),
    Level8SuffixGate(
        "level8_ow_leave_settle",
        make_ow_leave_controller,
        LEVEL8_INTERIOR_0X2C_TF_RECON,
        "chapter epilogue: idle the shard fanfare to the settled OW 0x6D "
        "(192,157) leave -- the level8_clear_stop / L9-predecessor pose",
    ),
)


def suffix_blockers(
    lineage: Level8SuffixLineage = FIXTURE_LINEAGE_LEVEL8_SUFFIX,
) -> tuple[str, ...]:
    """Why the suffix is not composed. Empty only for a composable lineage."""
    reasons: list[str] = []
    if not lineage.natural_predecessor:
        reasons.append("post_l7_handoff_unmeasured")
    if not lineage.route_eligible:
        reasons.append("suffix_lineage_not_route_eligible")
    return tuple(reasons)


def suffix_stages(
    *, lineage: Level8SuffixLineage = FIXTURE_LINEAGE_LEVEL8_SUFFIX
) -> tuple[Stage, ...]:
    """Ordered suffix rows, or ``()`` while the lineage is fixture-only."""
    if not lineage.composable():
        return ()
    rows: list[Stage] = []
    for gate in LEVEL8_SUFFIX_GATES:
        controller = gate.factory()
        rows.append(
            (gate.stage, controller, int(getattr(controller, "max_frames", 1)))
        )
    return tuple(rows)
