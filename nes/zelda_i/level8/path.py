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
East gate from play 0x3E leftover (120,93) idles until the RIGHT door bit
(arrival doors 0x0C, idle raises 0x0D), stays on the north band (y≈93-109)
past the x=144 statue, y-aligns to the east mouth, then RIGHT push. Dest
is live 0x3F (32,141) west mouth. Occupancy still banned. Do not chain
STAIRS into cellar 0x2F.
Hypothesis rooms past 0x1E cannot press a direction on the cumulative spine.
The 0x1E body is live type 0x33 HP96 (`LEVEL8_INTERIOR_0X1E_RECON`); colour
is not asserted.  Blue Gohma still requires naturally owned Bow + wooden
arrows and never pokes L6's one-time arrow grant or L6 room 0x1C.  Four-head
Gleeok live body is type 0x45 (idle census + fight pin); south-stand
in ``level8.gleeok``. 0x45 is RAM, not a ROM assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.door_hop import (
    HopFail,
    RoomHopController,
    RoomHopSpec,
    door_band_goal,
)
from zelda_i.dungeon.ids import (
    GOHMA_BLUE_OBJECT_TYPE,
    GOHMA_OBJECT_TYPE,
    MANHANDLA_OBJECT_TYPE,
)
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.dungeon.pause_select import B_SLOT_ARROWS, PauseSelectController
from zelda_i.level8.cellar import CELLAR_ROOM
from zelda_i.level8.dungeon import (
    BLUE_GOHMA_ARROWS_REQUIRED,
    LEVEL8,
    MAGIC_KEY_TO_SHARD_SPEC,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ChapterSpec,
    Level8Topology,
)
from zelda_i.level8.magic_key import (
    Level8BlueGohma1EController,
    Level8MagicKeyStairsController,
    STAND_Y,
    make_blue_gohma_1e_controller,
    make_magic_key_stairs_live_controller,
)
from zelda_i.level8.gleeok import (
    Level8FourHeadGleeokController,
    make_four_head_gleeok_controller as _make_four_head_gleeok_controller,
)
from zelda_i.level8.north_column import (
    Level8DarknutKeyController,
    Level8NorthManhandlaController,
    ROOM_MAP_MANHANDLA,
    make_north_manhandla_controller as _make_north_manhandla_controller,
)
from zelda_i.ram import ZeldaSnapshot

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
EAST_DOOR = DOOR_TARGETS["RIGHT"]  # (208, 141)
EAST_3E_ORIGIN = SOUTH_2E_DEST  # 0x3E
EAST_3E_ORIGIN_POSE = SOUTH_2E_DEST_POSE  # (120, 93) north mouth
EAST_3E_DEST_HYP = 0x3F  # confirmed live J1/J2; not 0x3C / 0x0F
EAST_3E_DEST = 0x3F  # live $EB from 0x3E east; west mouth
EAST_3E_DEST_POSE = (32, 141)  # live J1 arrival
EAST_RIGHT_BIT = 0x01  # DoorDir.RIGHT; idle raises doors 12→13
EAST_STATUE_CLEAR_X = 176  # past mid-row statue ~x=144; do not y-align earlier
EAST_NORTH_BAND_Y = 109  # stay north of statue row y~141
GLEEOK_HYP = 0x3C
_DOOR_TOL = 4
_SAMPLE_PERIOD = 12
_WEST_MAX_FRAMES = 4000
_SOUTH_MAX_FRAMES = 4000
_EAST_MAX_FRAMES = 4000


def west_1f_step(snap: ZeldaSnapshot) -> FrameAction:
    """Leftover-relative LEFT. Off-band y uses the door row, not leftover y."""
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = door_band_goal("LEFT", (x, y), WEST_DOOR)
    if x > STAIRS_WEST_X:
        return FrameAction(nes_action("LEFT"), "west_clear_stairs")
    if abs(y - gy) > _DOOR_TOL:
        btn = "UP" if y > gy else "DOWN"
        return FrameAction(nes_action(btn), "west_align")
    if x > gx + _DOOR_TOL:
        return FrameAction(nes_action("LEFT"), "west_approach")
    return FrameAction(nes_action("LEFT"), "west_push")


def _south_door_step(snap: ZeldaSnapshot) -> FrameAction:
    """Leftover-relative DOWN. Off-column leftover uses door x, not leftover x."""
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = door_band_goal("DOWN", (x, y), SOUTH_DOOR)
    if abs(x - gx) > _DOOR_TOL:
        btn = "LEFT" if x > gx else "RIGHT"
        return FrameAction(nes_action(btn), "south_align")
    if y < gy - _DOOR_TOL:
        return FrameAction(nes_action("DOWN"), "south_approach")
    return FrameAction(nes_action("DOWN"), "south_push")


def south_1e_step(snap: ZeldaSnapshot) -> FrameAction:
    """0x1E leftover → south door. Knockback at x=208 must LEFT, not DOWN."""
    return _south_door_step(snap)


def south_2e_step(snap: ZeldaSnapshot) -> FrameAction:
    """0x2E leftover → south door. On-column leftover keeps leftover x."""
    return _south_door_step(snap)


def east_3e_step(snap: ZeldaSnapshot) -> FrameAction:
    """Idle until RIGHT bit, north-band RIGHT past statues, leftover y-align.

    Do not walk the statue row at y=141 RIGHT into x=144. Occupancy banned.
    """
    x, y = int(snap.link_x), int(snap.link_y)
    gx, gy = door_band_goal("RIGHT", (x, y), EAST_DOOR)
    if not (int(snap.cur_opened_doors) & EAST_RIGHT_BIT):
        return FrameAction(nes_idle_action(), "east_wait_right_bit")
    if x < EAST_STATUE_CLEAR_X:
        if y > EAST_NORTH_BAND_Y:
            return FrameAction(nes_action("UP"), "east_north_band")
        return FrameAction(nes_action("RIGHT"), "east_approach")
    if abs(y - gy) > _DOOR_TOL:
        btn = "UP" if y > gy else "DOWN"
        return FrameAction(nes_action(btn), "east_align")
    if x < gx - _DOOR_TOL:
        return FrameAction(nes_action("RIGHT"), "east_approach")
    return FrameAction(nes_action("RIGHT"), "east_push")


_L8_GATE_FAILS = (
    HopFail((CELLAR_ROOM,), "cellar_0x{screen:02x}", on_passage=True),
    HopFail((GLEEOK_HYP,), "gleeok_0x3c"),
)


def _gate(
    spec_id: str,
    origin: int,
    door: str,
    tag: str,
    step,
    done_reason: str,
    *,
    max_frames: int,
) -> RoomHopSpec:
    """One L8 interior gate row. ``door`` doubles as the scroll/settle hold."""
    return RoomHopSpec(
        spec_id=spec_id,
        origin=origin,
        door=door,
        done_reason=done_reason,
        step=step,
        level=LEVEL8,
        max_frames=max_frames,
        sample_period=_SAMPLE_PERIOD,
        fails=_L8_GATE_FAILS,
        scroll_button=door,
        scroll_reason=f"{tag}_scroll",
        settle_button=door,
        settle_reason=f"{tag}_settle",
    )


WEST_1F_GATE = _gate(
    "level8_west_1f", WEST_ORIGIN, "LEFT", "west", west_1f_step,
    "left_0x1f_west", max_frames=_WEST_MAX_FRAMES,
)
SOUTH_1E_GATE = _gate(
    "level8_south_1e", SOUTH_ORIGIN, "DOWN", "south", south_1e_step,
    "left_0x1e_south", max_frames=_SOUTH_MAX_FRAMES,
)
SOUTH_2E_GATE = _gate(
    "level8_south_2e", SOUTH_2E_ORIGIN, "DOWN", "south", south_2e_step,
    "left_0x2e_south", max_frames=_SOUTH_MAX_FRAMES,
)
EAST_3E_GATE = _gate(
    "level8_east_3e", EAST_3E_ORIGIN, "RIGHT", "east", east_3e_step,
    "left_0x3e_east", max_frames=_EAST_MAX_FRAMES,
)
LEVEL8_PATH_GATES: tuple[RoomHopSpec, ...] = (
    WEST_1F_GATE, SOUTH_1E_GATE, SOUTH_2E_GATE, EAST_3E_GATE,
)


@dataclass(kw_only=True)
class Level8West1FController(RoomHopController):
    """0x1F leftover → west door LEFT. Dest is RAM; fail 0x0F / 0x3C."""

    spec: RoomHopSpec = WEST_1F_GATE


@dataclass(kw_only=True)
class Level8South1EController(RoomHopController):
    """0x1E leftover → south door DOWN. Dest is RAM; fail 0x0F / 0x3C."""

    spec: RoomHopSpec = SOUTH_1E_GATE


@dataclass(kw_only=True)
class Level8South2EController(RoomHopController):
    """0x2E leftover → south door DOWN. Dest is RAM; fail 0x0F / 0x3C."""

    spec: RoomHopSpec = SOUTH_2E_GATE


@dataclass(kw_only=True)
class Level8East3EController(RoomHopController):
    """0x3E leftover → east door RIGHT. Dest is RAM; fail 0x0F / 0x3C."""

    spec: RoomHopSpec = EAST_3E_GATE


def make_west_1f_controller(
    *, dest: int | None = WEST_DEST
) -> Level8West1FController:
    return Level8West1FController(dest=dest)


def make_south_1e_controller(
    *, dest: int | None = SOUTH_DEST
) -> Level8South1EController:
    return Level8South1EController(dest=dest)


def make_south_2e_controller(
    *, dest: int | None = SOUTH_2E_DEST
) -> Level8South2EController:
    return Level8South2EController(dest=dest)


def make_east_3e_controller(
    *, dest: int | None = EAST_3E_DEST
) -> Level8East3EController:
    return Level8East3EController(dest=dest)



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


def make_north_manhandla_controller() -> Level8NorthManhandlaController:
    return _make_north_manhandla_controller()


@dataclass(kw_only=True)
class Level8DarknutKeyArrowsController(Level8DarknutKeyController):
    """Select arrows in 0x2E before the north-door leave, inland of the lip.

    Hold ``combat_north_door`` / map-skip until B=arrows even if Manhandla is
    live off-corridor. Fight south (bombs stay on B). Do not START on
    y>STAND_Y (h6 leftover (106,189)). 0x3E still needs B=bombs.
    """

    _select: PauseSelectController | None = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        super().bind_env(env)
        if self._select is None:
            self._select = PauseSelectController(want=B_SLOT_ARROWS, name="arrows")
        self._select.bind_env(env)

    def _manhandla_live(self, snap: ZeldaSnapshot) -> bool:
        return any(
            1 <= obj.slot <= 12
            and int(obj.type_id) == MANHANDLA_OBJECT_TYPE
            and obj.hp > 0
            for obj in snap.objects
        )

    def _north_door_leave(self, act: FrameAction) -> bool:
        r = act.reason
        return (
            r == "combat_north_door"
            or r.startswith("map_skip")
            or r.startswith("north_key_0x2e")
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if int(snap.screen) != ROOM_MAP_MANHANDLA or self._select is None:
            return super().policy(snap)
        if self._select.success:
            return super().policy(snap)
        if int(snap.link_y) > STAND_Y:
            if self._manhandla_live(snap):
                return super().policy(snap)
            return FrameAction(nes_action("UP"), "climb")
        leave = None
        if self._manhandla_live(snap):
            leave = super().policy(snap)
            if not self._north_door_leave(leave):
                return leave
        driven = self._select.drive(snap)
        if self._select.failed:
            return self.mark_fail(
                self._select.fail_reason or "l8_darknut_arrow_select_failed"
            )
        if driven is not None:
            return driven
        return leave if leave is not None else super().policy(snap)


def make_darknut_key_controller() -> Level8DarknutKeyController:
    return Level8DarknutKeyArrowsController()


def make_blue_gohma_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8BlueGohma1EController:
    """Live 0x1E arrow kill + RIGHT shutter to 0x1F (``level8.magic_key``).

    ``topology`` is accepted for call-site compatibility and ignored: the
    kill is gated on naturally owned bow + wooden arrows, not on a topology
    flag.  Fails closed without them.
    """
    del topology
    return make_blue_gohma_1e_controller()


def make_magic_key_stairs_controller() -> Level8MagicKeyStairsController:
    """Live 0x1F clear -> 0x68 south slide -> centre stairs -> cellar 0x0F key
    loop -> two-ladder return to play 0x1F carrying ``ADDR_MAGIC_KEY`` 0->1.

    Promotes ``probe_l8_1f_magic_key`` (E2) + ``level8.cellar``.  Spine-green
    from the power-on 0x1F frontier (``scripts/magic_key_lab.py``: clear ~2760f,
    north-lane push slides the 0x68 to (96,160), cellar pickup + return in
    ~1600f, MK 0->1 back at play 0x1F (96,157), 0 writes, 0 deaths).
    ``route_eligible`` stays False (not a natural-entry promotion).  rr-6o7.2.
    """
    return make_magic_key_stairs_live_controller()


def make_gleeok_passage_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_return_passage",
        "live return from Magical Key through the west passage",
        spec=MAGIC_KEY_TO_SHARD_SPEC,
    )


def make_four_head_gleeok_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8FourHeadGleeokController:
    return _make_four_head_gleeok_controller(topology=topology)


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
    "Level8MagicKeyStairsController",
    "Level8NorthManhandlaController",
    "Level8East3EController",
    "Level8South1EController",
    "Level8South2EController",
    "Level8West1FController",
    "LEVEL8_PATH_GATES",
    "UnverifiedLevel8PathController",
    "WEST_DOOR",
    "STAIRS_WEST_X",
    "EAST_3E_DEST",
    "EAST_3E_DEST_HYP",
    "EAST_3E_DEST_POSE",
    "EAST_3E_ORIGIN",
    "EAST_3E_ORIGIN_POSE",
    "EAST_DOOR",
    "EAST_NORTH_BAND_Y",
    "EAST_RIGHT_BIT",
    "EAST_STATUE_CLEAR_X",
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
    "east_3e_step",
    "south_1e_step",
    "south_2e_step",
    "west_1f_step",
    "make_blue_gohma_controller",
    "make_darknut_key_controller",
    "make_east_3e_controller",
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
