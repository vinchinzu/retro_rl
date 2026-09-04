"""Level 8 interior one-frame policies.

North-column factories (`make_north_manhandla_controller`,
`make_darknut_key_controller`) are fixture-live in `level8.north_column`.
West gate from play 0x1F leftover (96,157) is fixture-live cardinal LEFT
into the first settled dest (0x1E); fail cellar 0x0F and Gleeok 0x3C.
G1 occupancy 1px-grade boxed at (88,157) tile 118; do not re-grade 2px steps.
South gate from play 0x1E leftover (208,141) is cardinal x-align to 120
then DOWN into first settled dest 0x2E (120,77) north mouth. Occupancy
BFS closes empty-grid but first dir is DOWN along x=208 into the SE
statue, and 1px-grade still false-misses 2px dungeon steps — occupancy
not used live.
South gate from play 0x2E leftover (120,77) is DOWN along x=120 into first
settled dest 0x3E (120,93) north mouth. Occupancy still banned. Statues
in 0x2E sit ~x=96 and x=144 at y~141; center x=120 passes between them.
Map 0x17 is incidental (ADDR_MAP 0→0x80 on the aisle; not a detour).
Hypothesis rooms past 0x1E cannot press a direction on the cumulative spine.
The 0x1E body is live type 0x33 HP96 (`LEVEL8_INTERIOR_0X1E_RECON`); colour
is not asserted.  Blue Gohma still requires naturally owned Bow + wooden
arrows and never pokes L6's one-time arrow grant or L6 room 0x1C.  Four-head
Gleeok waits for a live object type; 0x45 is not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.level8.cellar import CELLAR_ROOM
from zelda_i.level8.dungeon import (
    BLUE_GOHMA_ARROWS_REQUIRED,
    ENTRY_TO_MAGIC_KEY_SPEC,
    GLEEOK_FOUR_HEAD_OBJECT_TYPE,
    MAGIC_KEY_TO_SHARD_SPEC,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ChapterSpec,
    Level8Topology,
)
from zelda_i.level8.north_column import (
    Level8DarknutKeyController,
    Level8NorthManhandlaController,
    make_darknut_key_controller as _make_darknut_key_controller,
    make_north_manhandla_controller as _make_north_manhandla_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

WEST_DOOR = DOOR_TARGETS["LEFT"]  # (32, 141)
WEST_ORIGIN = 0x1F
WEST_ORIGIN_POSE = (96, 157)
WEST_DEST = 0x1E
WEST_DEST_POSE = (208, 141)  # live G2/G3 arrival; east mouth
WEST_GRID_XMIN = 16
# West of 0x68 (96,144) before any UP. G1 occupancy 1px-grade boxed at
# (88,157) tile 118 (walkable floor) after 2px LEFT steps.
STAIRS_WEST_X = 80
SOUTH_DOOR = DOOR_TARGETS["DOWN"]  # (120, 205)
SOUTH_ORIGIN = WEST_DEST  # 0x1E
SOUTH_ORIGIN_POSE = WEST_DEST_POSE  # (208, 141) east mouth
SOUTH_DEST = 0x2E
SOUTH_DEST_POSE = (120, 77)  # live H2/H3 arrival; north mouth (ymin band)
SOUTH_2E_ORIGIN = SOUTH_DEST  # 0x2E
SOUTH_2E_ORIGIN_POSE = SOUTH_DEST_POSE  # (120, 77) north mouth
SOUTH_2E_DEST_HYP = 0x3E  # confirmed live I1/I2/I3; not 0x3C / 0x0F
SOUTH_2E_DEST = 0x3E  # live $EB from 0x2E south; north mouth
SOUTH_2E_DEST_POSE = (120, 93)  # live I1/I2/I3 arrival
GLEEOK_HYP = 0x3C
_DOOR_TOL = 4
_SAMPLE_PERIOD = 12
_WEST_MAX_FRAMES = 4000
_SOUTH_MAX_FRAMES = 4000


def west_1f_step(snap: ZeldaSnapshot) -> FrameAction:
    """Cardinal LEFT past 0x68, y-align, LEFT push. No occupancy grade."""
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = WEST_DOOR
    if x > STAIRS_WEST_X:
        return FrameAction(nes_action("LEFT"), "west_clear_stairs")
    if abs(y - gy) > _DOOR_TOL:
        btn = "UP" if y > gy else "DOWN"
        return FrameAction(nes_action(btn), "west_align")
    if x > gx + _DOOR_TOL:
        return FrameAction(nes_action("LEFT"), "west_approach")
    return FrameAction(nes_action("LEFT"), "west_push")


def south_1e_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align to 120, then DOWN push. No occupancy, no sword, no UP."""
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = SOUTH_DOOR
    if abs(x - gx) > _DOOR_TOL:
        btn = "LEFT" if x > gx else "RIGHT"
        return FrameAction(nes_action(btn), "south_align")
    if y < gy - _DOOR_TOL:
        return FrameAction(nes_action("DOWN"), "south_approach")
    return FrameAction(nes_action("DOWN"), "south_push")


def south_2e_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align to 120, then DOWN push. Origin is already on x=120."""
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = SOUTH_DOOR
    if abs(x - gx) > _DOOR_TOL:
        btn = "LEFT" if x > gx else "RIGHT"
        return FrameAction(nes_action(btn), "south_align")
    if y < gy - _DOOR_TOL:
        return FrameAction(nes_action("DOWN"), "south_approach")
    return FrameAction(nes_action("DOWN"), "south_push")


def _west_leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
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
class Level8West1FController(HopController):
    """0x1F leftover → west door LEFT. Dest is RAM; fail 0x0F / 0x3C."""

    spec_id: str = "level8_west_1f"
    max_frames: int = _WEST_MAX_FRAMES
    require_level: int = 8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x1f_west"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (CELLAR_ROOM, GLEEOK_HYP):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen != WEST_ORIGIN

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("LEFT"), "west_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _west_leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE or snap.screen == CELLAR_ROOM:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == GLEEOK_HYP:
            return self.mark_fail("gleeok_0x3c")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != WEST_ORIGIN
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != WEST_ORIGIN:
            return FrameAction(nes_action("LEFT"), "west_settle")
        return west_1f_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "LEFT",
            "leftover": dict(self.leftover),
        }


def make_west_1f_controller(
    *, dest: int | None = WEST_DEST
) -> Level8West1FController:
    return Level8West1FController(dest=dest)


@dataclass(kw_only=True)
class Level8South1EController(HopController):
    """0x1E leftover → south door DOWN. Dest is RAM; fail 0x0F / 0x3C."""

    spec_id: str = "level8_south_1e"
    max_frames: int = _SOUTH_MAX_FRAMES
    require_level: int = 8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x1e_south"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (CELLAR_ROOM, GLEEOK_HYP):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen != SOUTH_ORIGIN

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("DOWN"), "south_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _west_leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE or snap.screen == CELLAR_ROOM:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == GLEEOK_HYP:
            return self.mark_fail("gleeok_0x3c")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != SOUTH_ORIGIN
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != SOUTH_ORIGIN:
            return FrameAction(nes_action("DOWN"), "south_settle")
        return south_1e_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "DOWN",
            "leftover": dict(self.leftover),
        }


def make_south_1e_controller(
    *, dest: int | None = SOUTH_DEST
) -> Level8South1EController:
    return Level8South1EController(dest=dest)


@dataclass(kw_only=True)
class Level8South2EController(HopController):
    """0x2E leftover → south door DOWN. Dest is RAM; fail 0x0F / 0x3C."""

    spec_id: str = "level8_south_2e"
    max_frames: int = _SOUTH_MAX_FRAMES
    require_level: int = 8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x2e_south"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (CELLAR_ROOM, GLEEOK_HYP):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen != SOUTH_2E_ORIGIN

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("DOWN"), "south_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _west_leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE or snap.screen == CELLAR_ROOM:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == GLEEOK_HYP:
            return self.mark_fail("gleeok_0x3c")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != SOUTH_2E_ORIGIN
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != SOUTH_2E_ORIGIN:
            return FrameAction(nes_action("DOWN"), "south_settle")
        return south_2e_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "DOWN",
            "leftover": dict(self.leftover),
        }


def make_south_2e_controller(
    *, dest: int | None = SOUTH_2E_DEST
) -> Level8South2EController:
    return Level8South2EController(dest=dest)


# L6 red Gohma is 0x33; L8 source is blue 0x34.  Red is accepted only as a
# live-type observation, never as an L6 room check.
_GOHMA_TYPES = frozenset({GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE})


@dataclass
class UnverifiedLevel8PathController:
    """Stop immediately when a chapter has no live one-frame policy."""

    stage_id: str
    missing_evidence: str
    spec: Level8ChapterSpec | None = None
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "chapter_id": None if self.spec is None else self.spec.chapter_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "missing_evidence": self.missing_evidence,
            "writes": self.writes,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}:"
            f"xy={snap.link_x},{snap.link_y}"
        )
        if not self.notes:
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "blocked_unverified")


def unverified_path_controller(
    stage_id: str, missing_evidence: str, *, spec: Level8ChapterSpec | None = None
) -> UnverifiedLevel8PathController:
    return UnverifiedLevel8PathController(stage_id, missing_evidence, spec=spec)


@dataclass
class Level8BlueGohmaController:
    """Fail closed until a live kill measures type/arrows.

    0x1E live census (`LEVEL8_INTERIOR_0X1E_RECON`) is one body type 0x33
    HP96.  Colour is not asserted here.
    """

    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0
    poked_arrows: bool = False
    l6_room_check: bool = False

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level8_blue_gohma",
            "arrows_required": BLUE_GOHMA_ARROWS_REQUIRED,
            "gohma_types": tuple(sorted(_GOHMA_TYPES)),
            "poked_arrows": self.poked_arrows,
            "l6_room_check": self.l6_room_check,
            "writes": self.writes,
            "route_eligible": False,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        if snap.bow < 1 or snap.arrows < 1:
            reason = "l8_gohma_requires_natural_bow_arrows"
        elif not self.topology.route_eligible:
            reason = "l8_gohma_topology_unobserved"
        else:
            reason = "l8_gohma_room_unobserved"
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)


@dataclass
class Level8FourHeadGleeokController:
    """Fail closed until a live L8 Gleeok body type is observed."""

    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
    observed_body_type: int | None = GLEEOK_FOUR_HEAD_OBJECT_TYPE
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level8_four_head_gleeok",
            "observed_body_type": self.observed_body_type,
            "assumed_0x45": False,
            "writes": self.writes,
            "route_eligible": False,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        if self.observed_body_type == 0x45:
            reason = "l8_gleeok_refuses_assumed_0x45"
        elif self.observed_body_type is None:
            reason = "l8_gleeok_object_type_unobserved"
        elif not self.topology.route_eligible or self.topology.boss_room is None:
            reason = "l8_gleeok_topology_unobserved"
        else:
            reason = "l8_gleeok_room_unobserved"
        if not self.notes:
            self.notes.append(reason)
        _ = snap
        return FrameAction(nes_idle_action(), reason)


def make_north_manhandla_controller() -> Level8NorthManhandlaController:
    return _make_north_manhandla_controller()


def make_darknut_key_controller() -> Level8DarknutKeyController:
    return _make_darknut_key_controller()


def make_blue_gohma_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8BlueGohmaController:
    return Level8BlueGohmaController(topology=topology)


def make_magic_key_stairs_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_magic_key_stairs",
        "live Magical Key cellar and natural ADDR_MAGIC_KEY 0-to-1",
        spec=ENTRY_TO_MAGIC_KEY_SPEC,
    )


def make_gleeok_passage_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_return_passage",
        "live return from Magical Key through the west passage",
        spec=MAGIC_KEY_TO_SHARD_SPEC,
    )


def make_four_head_gleeok_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8FourHeadGleeokController:
    return Level8FourHeadGleeokController(topology=topology)


def make_shard_leave_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_heart_shard_leave",
        "live heart container, shard 0x80, and settled post-fanfare OW leave",
        spec=MAGIC_KEY_TO_SHARD_SPEC,
    )


__all__ = [
    "CELLAR_ROOM",
    "GLEEOK_HYP",
    "Level8BlueGohmaController",
    "Level8DarknutKeyController",
    "Level8FourHeadGleeokController",
    "Level8NorthManhandlaController",
    "Level8South1EController",
    "Level8South2EController",
    "Level8West1FController",
    "UnverifiedLevel8PathController",
    "WEST_DOOR",
    "STAIRS_WEST_X",
    "SOUTH_2E_DEST",
    "SOUTH_2E_DEST_HYP",
    "SOUTH_2E_DEST_POSE",
    "SOUTH_2E_ORIGIN",
    "SOUTH_2E_ORIGIN_POSE",
    "SOUTH_DEST",
    "SOUTH_DEST_POSE",
    "SOUTH_DOOR",
    "SOUTH_ORIGIN",
    "SOUTH_ORIGIN_POSE",
    "WEST_DEST",
    "WEST_DEST_POSE",
    "WEST_GRID_XMIN",
    "WEST_ORIGIN",
    "WEST_ORIGIN_POSE",
    "south_1e_step",
    "south_2e_step",
    "west_1f_step",
    "make_blue_gohma_controller",
    "make_darknut_key_controller",
    "make_four_head_gleeok_controller",
    "make_gleeok_passage_controller",
    "make_magic_key_stairs_controller",
    "make_north_manhandla_controller",
    "make_shard_leave_controller",
    "make_south_1e_controller",
    "make_south_2e_controller",
    "make_west_1f_controller",
    "unverified_path_controller",
]
