"""Post-bomb gathering segments: one module, one dispatcher, strict stops."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import direction_to_facing
from zelda_i.dungeon.ops import B_ITEM_CANDLE
from zelda_i.dungeon.pause_select import PauseSelectController
from zelda_i.overworld.cave_shop import CaveShopBuyController
from zelda_i.overworld.gather_run import (
    PRE_L1_LEAVE,
    pin_pre_l1,
    run_chain,
    run_segment,
    write_entry_pin,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.heart_farm import PondFairyController
from zelda_i.overworld.hunt import link_busy
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.white_sword import (
    MIN_HEART_CONTAINERS,
    SCREEN_MAZE_GATE,
    SCREEN_WHITE_SWORD_CAVE,
    WHITE_SWORD,
    WhiteSwordDetourController,
    WhiteSwordPhase,
)
from zelda_i.ram import ADDR_CANDLE, CAVE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

BLAST_FRAMES = 80
# Frames after B by which a placed bomb has left ``$0658``.
BOMB_CONFIRM_FRAMES = 12
# Inside this many px of the bomb cell the approach pushes instead of swinging.
BOMB_CELL_PUSH_PX = 8
# Link's ObjState while a cave's text (or a pond fairy) holds him.
LINK_HALTED = 0x40

HEART_L8_SCREEN = 0x7B
HEART_L8_HOPS = (ScreenHop(HEART_L8_SCREEN, "LEFT"),)
# 2026-09-22 take: UP from this pose did not move. The cell ahead is a wall.
HEART_L8_STALL_XY = (112, 141)
HEART_L8_STALL_DIR = "UP"
# Take-any old man at (120, 128). Measured 2026-09-22: it shows up 31 frames
# after the bomb-mouth lock, once the cave has replaced the overworld wave.
HEART_CAVE_SPRITE = 0x6B
# Heart is the right-hand item (potion left). Same touch as the candle pedestal.
HEART_L8_ITEM_XY = (152, 149)

HEART_M3_SCREEN = 0x2C
# Measured 2026-09-22 by a bomb sweep on BFS_2C: the doorway is on the
# bottom face of the centre rock. Facing UP from y=165 opens it for
# x 136..152. The right face (x 176..184, facing LEFT) opened nothing.
HEART_M3_BOMB_XY = (144, 165)
# BFS_2C spawns at (0, 85). The rock spans x 80..176, so walk the open
# west column down to the bomb row before going east.
HEART_M3_APPROACH = ((16, 165),)
# The 0x2C cave is the same take-any layout as 0x7B: old man 0x6B at
# (120, 128), potion left, heart right. Both walk to HEART_L8_ITEM_XY.
TAKE_ANY_SCREENS = (HEART_L8_SCREEN, HEART_M3_SCREEN, 0x47)

# Burn secrets, measured 2026-09-22. Each is a hidden tree tile object
# (type 0x64) at a ROM-fixed spot: 0x48 (208, 96), 0x47 (176, 176),
# 0x46 (144, 176). The flame has to walk its 16 px before it stands, so
# Link cannot be flush with the tree: (192, 93) RIGHT stays shut, (188, 93)
# opens. The reveal sets room flag $80.
FLAME_WAIT = 100
BURN_48_STAND = (188, 93)
BURN_47_STAND = (176, 157)
SECRET_48_RUPEES = 30
# The 0x48 cave keeper is 0x7B, not 0x0F's 0x7C (measured in the chain).
SECRET_48_KEEPER = 0x7B

NE_HOPS = (
    ScreenHop(0x2D, "RIGHT", align_y=180),
    ScreenHop(0x1D, "UP", align_x=120),
    # East, not west: 0x1D -> 0x1E -> 0x1F is column 13 -> 15. The LEFT
    # table stalled on 0x1D.
    # y=141, not 146: tolerance 5 let y=149 through, and RIGHT from
    # (136, 149) is a wall (1713 idle frames). y=141 walked off the edge.
    ScreenHop(0x1E, "RIGHT", align_y=141),
    ScreenHop(0x1F, "RIGHT", align_y=141),
    ScreenHop(0x0F, "UP", align_x=128),
)
# Secret cave in the 0x0F centre arch, measured 2026-09-22: moblin 0x7C at
# (120, 128), rupee below it. Link frozen ~225 frames while the text types.
# Walking UP at x=112 stops at (112, 141) and misses; x=120 takes it, then
# the HUD counts 100 up from $067D over about 200 frames.
SECRET_0F_SCREEN = 0x0F
SECRET_MOBLIN = 0x7C
SECRET_REWARD = 100
# Any one-item cave with the keeper at (120, 128). The item sits below.
CENTRE_ITEM_XY = (120, 141)
LETTER_HOPS = (ScreenHop(0x0E, "UP", align_x=80),)
# 0x0E, measured 2026-09-22: old man 0x72 at (120, 128), letter below.
LETTER_KEEPER = 0x72

CANDLE_PRICE = 60
CANDLE_HOPS = (
    ScreenHop(0x1E, "DOWN", align_x=80),
    ScreenHop(0x1D, "LEFT", align_y=141),
    ScreenHop(0x0D, "UP", align_x=208),
    ScreenHop(0x0C, "LEFT", align_y=141),
)
# Pedestal contact was (152, 149). Purchase stop is the shop controller's
# (item and the 60 leaving), not these coordinates by themselves.
# BUY_Y is the lateral row: the buy climbs to it at x=112, walks RIGHT on
# it, then touches UP. On y=149 that walk crosses the middle pedestal, a
# 100-rupee key: the chain arrived with 101 and bought it (keys 0->1).
CAVE_X = 128
CAVE_Y = 96
BUY_X = 152
BUY_Y = 165

# 0x29 and 0x2A are walled on top (2026-09-22 screenshots), so 0x2A UP
# never reaches 0x1A. Row 2 runs west to 0x27, whose north gap crosses to
# 0x17 for x 112..160 (sweep from BFS_27 after dropping to y=133 under its
# east rock). Row 1 east to 0x1A is the L9 detour's measured y=141 band.
WHITE_GAP_27_X = 144
# 0x28's east half is two staggered bush columns (x~176 and x~208, the
# column alternating every 16 px of y), so no row walks straight across.
# Walked live with the wave cleared, 2026-09-22: enter at y=117, then these
# corners. From (104, 133) LEFT runs to the west edge.
WHITE_28_ENTRY_Y = 117
WAYPOINTS: dict[int, tuple[tuple[int, int], ...]] = {
    0x28: ((224, 117), (224, 133), (192, 133), (192, 149), (104, 149), (104, 133)),
    # 0x7B heart cave spits Link out at (144, 77), where align_x is dropped
    # (y <= 80) and UP is the wall. Drop to y=93, then the x=176 north gap.
    0x7B: ((144, 93), (176, 93)),
    # 0x6B: south gap x 176..223, open row y=93, north gap x 48..95.
    0x6B: ((176, 93), (48, 93)),
    # 0x5B from the south gap x=48: the open lane is y=93, the north gap
    # is x 192..223. Measured from a chain pin, 2026-09-22.
    0x5B: ((48, 93), (200, 93)),
    # 0x0C candle-shop mouth is (128, 77). The south exit is the x=80 stairs,
    # reached along y=141; y=125 at x<=96 is rock (12000-frame stall).
    0x0C: ((128, 141), (80, 141)),
}
WAYPOINT_TOL = 2
WHITE_HOPS = (
    ScreenHop(0x1C, "DOWN", align_x=80),
    ScreenHop(0x2C, "DOWN", align_x=48),
    ScreenHop(0x2B, "LEFT", align_y=141),
    ScreenHop(0x2A, "LEFT", align_y=141),
    ScreenHop(0x29, "LEFT", align_y=133),
    ScreenHop(0x28, "LEFT", align_y=WHITE_28_ENTRY_Y),
    ScreenHop(0x27, "LEFT", align_y=133),
    ScreenHop(0x17, "UP", align_x=WHITE_GAP_27_X),
    ScreenHop(0x18, "RIGHT", align_y=141),
    ScreenHop(0x19, "RIGHT", align_y=141),
    ScreenHop(SCREEN_MAZE_GATE, "RIGHT", align_y=141),
)

# Bomb shop 0x6F back to 0x7C, the coast walk reversed: 0x6F's south edge
# at x=82, then 0x7E's live band (y=133 on 0x7E does not scroll), then any
# row. heart_l8's own hop is the last LEFT into 0x7B.
RETURN_7C_HOPS = (
    ScreenHop(0x7F, "DOWN", align_x=82),
    ScreenHop(0x7E, "LEFT", y_band_lo=137, y_band_hi=145),
    ScreenHop(0x7D, "LEFT", y_band_lo=137, y_band_hi=145),
    ScreenHop(0x7C, "LEFT"),
)

# 0x7B -> 0x2C up column B. 0x7D's north edge is solid rock, so column D
# is not a way up from the coast. The BFS_7B/6B/5B/4B pins arrived going
# south on these same columns.
HEART_WALK_HOPS = (
    ScreenHop(0x6B, "UP", align_x=176),
    ScreenHop(0x5B, "UP", align_x=48),
    ScreenHop(0x4B, "UP", align_x=200),
    ScreenHop(0x3B, "UP", align_x=200),
    ScreenHop(0x2B, "UP"),
    ScreenHop(0x2C, "RIGHT", align_y=85),
)
# The 0x39 pond fairy (``heart_farm.PondFairyController``) is the one full
# refill on the way up column B. 0x4B is two rooms: the x=192..223 corridor
# the climb uses is sealed from the west half, and only the west half opens
# onto 0x4A (y 128..175) and down onto 0x5B (x 16..95). 0x39's basin opens
# only south, onto 0x49 at x 112..143. Lanes read from the ``$6530`` tile map
# (``$034A`` first-unwalkable), not from the walkthrough image.
POND_WALK_HOPS = (
    ScreenHop(0x6B, "UP", align_x=176),
    ScreenHop(0x5B, "UP", align_x=48),
    ScreenHop(0x4B, "UP", align_x=48),
    ScreenHop(0x4A, "LEFT", align_y=125),
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x39, "UP", align_x=120),
)
POND_RETURN_HOPS = (
    ScreenHop(0x49, "DOWN", align_x=120),
    ScreenHop(0x4A, "RIGHT", align_y=141),
    ScreenHop(0x4B, "RIGHT", align_y=141),
    ScreenHop(0x5B, "DOWN", align_x=16),
) + HEART_WALK_HOPS[2:]
# Back on 0x5B from the top at x=16: y=93 is the open row east.
POND_RETURN_WAYPOINTS = {0x5B: ((16, 93), (200, 93))}

# 0x1A back to the burn row: row 1 west, 0x27's gap down, 0x28 east on
# y=133, then the x=112..143 cut down through 0x38 into 0x48.
BURN_WALK_HOPS = (
    ScreenHop(0x19, "LEFT", align_y=141),
    ScreenHop(0x18, "LEFT", align_y=141),
    ScreenHop(0x17, "LEFT", align_y=141),
    ScreenHop(0x27, "DOWN", align_x=WHITE_GAP_27_X),
    ScreenHop(0x28, "RIGHT", align_y=133),
    ScreenHop(0x38, "DOWN", align_x=120),
    ScreenHop(0x48, "DOWN", align_x=120),
)

# 0x47 heart cave to the Level 1 mouth screen: back east, the x=120 cut up
# to 0x38, then west on the L1 approach band.
L1_MOUTH_HOPS = (
    ScreenHop(0x48, "RIGHT", align_y=141),
    ScreenHop(0x38, "UP", align_x=120),
    ScreenHop(0x37, "LEFT", align_y=141),
)

# 0x47 to the 0x39 pond, then the mouth. The White Sword's beam fires at
# full hearts only, and the chain reached 0x37 on 3 of 6 (rung 2,
# 2026-09-22). 0x48 and 0x38 are shut on the east (lattice), and 0x49 on
# the west, so the pond is the loop through 0x58/0x59.
L1_POND_HOPS = (
    ScreenHop(0x48, "RIGHT", align_y=141),
    ScreenHop(0x58, "DOWN", align_x=120),
    ScreenHop(0x59, "RIGHT", align_y=133),
    ScreenHop(0x49, "UP", align_x=120),
    ScreenHop(0x39, "UP", align_x=120),
)
L1_FROM_POND_HOPS = (
    ScreenHop(0x49, "DOWN", align_x=120),
    ScreenHop(0x59, "DOWN", align_x=120),
    ScreenHop(0x58, "LEFT", align_y=133),
    ScreenHop(0x48, "UP", align_x=120),
) + L1_MOUTH_HOPS[1:]

# NE ends in the 0x0F cave. Back down the way NE came up, then the letter.
LETTER_FROM_0F_HOPS = (
    ScreenHop(0x1F, "DOWN", align_x=128),
    ScreenHop(0x1E, "LEFT", align_y=141),
) + LETTER_HOPS

POTION_HOPS = (ScreenHop(0x64, "LEFT", align_y=141),)
RING_HOPS = (
    ScreenHop(0x36, "LEFT", align_y=133),
    ScreenHop(0x35, "LEFT", align_y=133),
    ScreenHop(0x34, "LEFT", align_y=133),
)


def step_toward(
    controller: Any,
    snap: Any,
    x: int,
    y: int,
    reason: str,
    tol: int = 6,
    y_first: bool = False,
) -> FrameAction | None:
    """Walk to ``(x, y)``. None means Link is already inside ``tol`` px.

    ``y_first`` fixes the row before the column: on a clear row (0x2C's
    y=165 under the rock) a knockback off it must be undone before walking
    on, or RIGHT jams on the rock corner.
    """
    dx = int(x) - int(snap.link_x)
    dy = int(y) - int(snap.link_y)
    moves = [
        (abs(dx) > tol, "RIGHT" if dx > 0 else "LEFT"),
        (abs(dy) > tol, "DOWN" if dy > 0 else "UP"),
    ]
    if y_first:
        moves.reverse()
    for off, direction in moves:
        if off:
            return controller._swing(direction, reason)
    return None


def waypoint_action(controller: Any, snap: ZeldaSnapshot) -> FrameAction | None:
    """Walk this screen's corners in order, then decline.

    A controller's own ``waypoints`` table wins over ``WAYPOINTS``. The
    return walk crosses 0x28 west to east, the reverse of those corners.
    """
    table = getattr(controller, "waypoints", None)
    corners = (WAYPOINTS if table is None else table).get(int(snap.screen))
    if not corners or snap.mode != PLAY_MODE:
        return None
    if controller._way_screen != int(snap.screen):
        controller._way_screen = int(snap.screen)
        controller._way_leg = 0
    while controller._way_leg < len(corners):
        x, y = corners[controller._way_leg]
        dx = int(x) - int(snap.link_x)
        dy = int(y) - int(snap.link_y)
        if abs(dx) > WAYPOINT_TOL:
            return controller._swing("RIGHT" if dx > 0 else "LEFT", "waypoint")
        if abs(dy) > WAYPOINT_TOL:
            return controller._swing("DOWN" if dy > 0 else "UP", "waypoint")
        controller._way_leg += 1
    return None


def _action_dir(act: FrameAction) -> str:
    """The d-pad direction a walk action holds (``_swing`` may add A)."""
    for name in ("UP", "DOWN", "LEFT", "RIGHT"):
        if act.action[NES_BUTTON_NAME_TO_INDEX[name]]:
            return name
    return "DOWN"


def press_b(reason: str) -> FrameAction:
    return FrameAction(nes_action("B"), reason)


def idle(reason: str) -> FrameAction:
    return FrameAction(nes_idle_action(), reason)


def heart_sprite_up(snap: ZeldaSnapshot) -> bool:
    """True when the cave old man is on screen, not the overworld wave."""
    return any(int(obj.type_id) == HEART_CAVE_SPRITE for obj in snap.objects)


def push(direction: str, reason: str) -> FrameAction:
    return FrameAction(nes_action(direction), reason)


def centre_item_walk(snap: ZeldaSnapshot, keeper: int, tag: str) -> FrameAction:
    """One-item cave: wait for ``keeper``, line up on x=120, walk UP.

    The keeper's text freezes Link, so pushes during it are harmless.
    """
    if not any(int(o.type_id) == keeper for o in snap.objects):
        return idle("cave_settle")
    x, y = CENTRE_ITEM_XY
    if abs(int(snap.link_x) - x) > 1:
        return push("RIGHT" if int(snap.link_x) < x else "LEFT", f"{tag}_align")
    if int(snap.link_y) > y:
        return push("UP", f"{tag}_item")
    return idle(f"{tag}_wait")


@dataclass
class BombWallController(OverworldPathController):
    hops: tuple[ScreenHop, ...] = ()
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 6000
    screen: int = HEART_L8_SCREEN
    bomb_x: int = 144
    bomb_y: int = 88
    bomb_face: str = "UP"
    approach: tuple[tuple[int, int], ...] = ()
    bomb_y_first: bool = False
    # None stays put: a candle flame walks away from Link on its own.
    retreat: str | None = "DOWN"
    # Frames from B to walking in. A bomb blasts at ~80. A flame walks 16
    # frames, stands $3F, and the tree reveals once its timer drops below 2.
    use_wait: int = BLAST_FRAMES
    # "container" (take-any heart) or "rupees" (one-item secret cave).
    reward: str = "container"
    reward_rupees: int = 0
    keeper: int = SECRET_MOBLIN
    interior_x: int | None = 116
    interior_y: int | None = 128
    interior_budget: int = 400
    bomb_fail: str = "north_wall_did_not_open"
    leave_fail: str = "left_0x7b"
    # A bomb shows in ``$0658``; a flame does not, so burns do not check it.
    consumes_bomb: bool = True
    _bombs_at_press: int = -1
    _bombed: int = 0
    _leg: int = 0
    _entry_containers: int = -1
    _cave_rupees: int = -1
    _interior: int = 0
    _cave_walker: OccupancyWalker | None = None
    _cave_walker_screen: int = -1

    def _wants_post_hop(self) -> bool:
        return True

    def _remember(self, snap: ZeldaSnapshot) -> None:
        if self._entry_containers < 0 and snap.mode == PLAY_MODE:
            self._entry_containers = int(snap.heart_containers)

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        self._remember(snap)
        in_cave = snap.level == 0 and snap.mode == CAVE_MODE and snap.screen == self.screen
        if self.reward == "rupees":
            return (
                in_cave
                and self._cave_rupees >= 0
                and int(snap.rupees) >= self._cave_rupees + self.reward_rupees
            )
        return (
            in_cave
            and self._entry_containers >= 0
            and int(snap.heart_containers) > self._entry_containers
        )

    def _after_hops(self, snap: ZeldaSnapshot):
        self._remember(snap)
        if snap.level != 0 or snap.screen != self.screen:
            return self._fail(self.leave_fail)
        if snap.mode == CAVE_MODE:
            if self.reward == "rupees":
                if self._cave_rupees < 0:
                    self._cave_rupees = int(snap.rupees)
                self._interior += 1
                if self._interior > self.interior_budget:
                    return self._fail("secret_rupees_not_taken")
                return centre_item_walk(snap, self.keeper, "secret")
            if self.interior_x is None or self.interior_y is None:
                return push("UP", "take_heart")
            return self._take_heart(snap)
        while self._bombed == 0 and self._leg < len(self.approach):
            wx, wy = self.approach[self._leg]
            walk = step_toward(self, snap, wx, wy, "approach", tol=2)
            if walk is not None:
                return walk
            self._leg += 1
        if self._bombed == 0:
            # 2 px, not 6: the 0x2C sweep proved y 165..173, and a chain
            # bomb from inside the 6 px box left the rock shut.
            walk = step_toward(
                self, snap, self.bomb_x, self.bomb_y, "bomb_cell",
                tol=2, y_first=self.bomb_y_first,
            )
            if walk is not None:
                # The last few pixels are a push, not a swing. 0x7B knocked
                # Link to (144, 85), 3 px past the cell, with a red leever in
                # the lane below: every DOWN step became a slash that pinned
                # him, and B was never pressed in 3827 frames.
                near = max(
                    abs(int(snap.link_x) - self.bomb_x), abs(int(snap.link_y) - self.bomb_y)
                ) <= BOMB_CELL_PUSH_PX
                if near and not link_busy(snap):
                    return push(_action_dir(walk), "bomb_cell_nudge")
                return walk
            # The bomb lands ahead of Link. A walk that ended on the other
            # axis faces the wrong way (0x2C's first try bombed open sand).
            if int(snap.facing) != direction_to_facing(self.bomb_face):
                return push(self.bomb_face, "face_wall")
            # A B press inside a swing is dropped: the default spine's 0x7B
            # pressed once at the end of a ``bomb_cell_slash``, kept all four
            # bombs and stood 401 frames against a shut wall.
            if link_busy(snap):
                return idle("bomb_wait_busy")
            self._bombed = 1
            self._bombs_at_press = int(snap.bombs)
            return press_b("place_bomb")
        self._bombed += 1
        if (
            self.consumes_bomb
            and self._bombed == BOMB_CONFIRM_FRAMES
            and int(snap.bombs) >= self._bombs_at_press
        ):
            # No bomb left the bag: go back and press again (the idle frames
            # since the press are the B release the edge needs).
            self._bombed = 0
            self.notes.append("bomb_press_retry")
            return idle("bomb_retry")
        if self._bombed < self.use_wait:
            if self.retreat is None:
                return idle("flame_wait")
            return push(self.retreat, "off_blast")
        if self._bombed > self.use_wait + 400:
            return self._fail(self.bomb_fail)
        return push(self.bomb_face, "into_wall")

    def _cave_walker_for(self, snap: ZeldaSnapshot) -> OccupancyWalker:
        """Occupancy for the open cave. The measured UP miss stays blocked."""
        if self._cave_walker is None or self._cave_walker_screen != int(snap.screen):
            grid = OccupancyGrid()
            if int(snap.screen) in TAKE_ANY_SCREENS:
                grid.mark_blocked_ahead(
                    *HEART_L8_STALL_XY, HEART_L8_STALL_DIR, inferred=False
                )
            self._cave_walker = OccupancyWalker(grid=grid, sticky=True, slide=True)
            self._cave_walker_screen = int(snap.screen)
        return self._cave_walker

    def _take_heart(self, snap: ZeldaSnapshot) -> FrameAction:
        """Walk to the heart. A miss blocks that cell; no path stands."""
        # The old man's text halts Link (slot 0 ObjState $40) for ~280
        # frames on some entries; charging those left him 25 px short of
        # the heart when the budget ran out.
        if not snap.objects or int(snap.objects[0].state) != LINK_HALTED:
            self._interior += 1
        if self._interior > self.interior_budget:
            self._note_blocks(snap)
            return self._fail("heart_not_taken")
        # Mode 11 at the bomb mouth still has the overworld wave. Grading
        # those frozen frames walled in (144, 93) in four misses.
        if not heart_sprite_up(snap):
            return idle("cave_settle")
        walker = self._cave_walker_for(snap)
        xy = (int(snap.link_x), int(snap.link_y))
        goal = (int(self.interior_x), int(self.interior_y))
        direction = walker.next_dir(xy, goal, slide=True, sticky=True)
        if direction is None:
            if max(abs(xy[0] - goal[0]), abs(xy[1] - goal[1])) <= 6:
                return push("UP", "take_heart")
            self._note_blocks(snap)
            return self._fail("heart_not_taken")
        return push(direction, "heart")

    def _note_blocks(self, snap: ZeldaSnapshot) -> None:
        walker = self._cave_walker
        if walker is None:
            return
        learned = sorted(walker.grid.inferred)
        shown = ",".join(f"{x}:{y}" for x, y in learned[:8])
        self.notes.append(
            f"heart_blocks miss={walker.misses} learned={len(learned)} "
            f"pose={int(snap.link_x)},{int(snap.link_y)} cells={shown}"
        )


@dataclass
class CaveMouthController(OverworldPathController):
    hops: tuple[ScreenHop, ...] = ()
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 4000
    screen: int = 0x0E
    push_budget: int = 240
    miss_fail: str = "letter_stairs_not_found"
    off_fail: str = "not_on_0x0e"
    # Set both to require the item, not just the cave. ``taken`` reads RAM.
    keeper: int | None = None
    taken: Callable[[ZeldaSnapshot], bool] | None = None
    interior_budget: int = 600
    _mouth: int = 0
    _interior: int = 0

    def _wants_post_hop(self) -> bool:
        return True

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        in_cave = snap.level == 0 and snap.screen == self.screen and snap.mode == CAVE_MODE
        if self.taken is None:
            return in_cave
        return in_cave and bool(self.taken(snap))

    def _after_hops(self, snap: ZeldaSnapshot):
        if snap.level != 0 or snap.screen != self.screen:
            return self._fail(self.off_fail)
        if snap.mode == CAVE_MODE and self.keeper is not None:
            self._interior += 1
            if self._interior > self.interior_budget:
                return self._fail("cave_item_not_taken")
            return centre_item_walk(snap, self.keeper, "cave")
        self._mouth += 1
        if self._mouth > self.push_budget:
            return self._fail(self.miss_fail)
        return push("UP", "cave_mouth")


@dataclass
class GatherWhiteController(OverworldPathController):
    """Hop to 0x1A, then the L9 detour's measured climb, cave and pedestal.

    The stop is the sword byte. The Old Man gates on five containers, so a
    three-container pin reaches ``TAKE_SWORD`` and times out there.
    """

    hops: tuple[ScreenHop, ...] = WHITE_HOPS
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 12000
    _detour: WhiteSwordDetourController | None = None
    _way_screen: int = -1
    _way_leg: int = 0

    def _extra_hop_action(self, snap: ZeldaSnapshot, hop: ScreenHop):
        return waypoint_action(self, snap)

    def _wants_post_hop(self) -> bool:
        return True

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.screen == SCREEN_WHITE_SWORD_CAVE
            and int(snap.sword) >= WHITE_SWORD
        )

    def _after_hops(self, snap: ZeldaSnapshot):
        # The cave is proven by here. The Old Man gates on containers, and
        # a 3-container pin pushed the pedestal for 6907 frames on 2026-09-22.
        if snap.mode == CAVE_MODE and int(snap.heart_containers) < MIN_HEART_CONTAINERS:
            return self._fail(f"white_cave_reached_containers_{int(snap.heart_containers)}_of_5")
        if self._detour is None:
            self._detour = WhiteSwordDetourController(
                max_frames=self.max_frames,
                phase=WhiteSwordPhase.MAZE_NORTH,
                start_checked=True,
            )
        action = self._detour.step(snap)
        if self._detour.failed:
            return self._fail(f"detour_{self._detour.notes[-1]}")
        return action

    def report(self) -> dict[str, Any]:
        out = super().report()
        if self._detour is not None:
            out["detour"] = self._detour.report()
        return out


@dataclass
class HopWalkController(OverworldPathController):
    """Hops plus per-screen ``WAYPOINTS``. Stops on the last target, in play."""

    hops: tuple[ScreenHop, ...] = HEART_WALK_HOPS
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 6000
    waypoints: dict[int, tuple[tuple[int, int], ...]] | None = None
    _way_screen: int = -1
    _way_leg: int = 0

    def _extra_hop_action(self, snap: ZeldaSnapshot, hop: ScreenHop):
        return waypoint_action(self, snap)

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.hops[-1].target
        )


# Every one-room cave here has its stairs bottom centre. Link entered the
# 0x0F cave at (112, 213).
CAVE_EXIT_X = 112
CAVE_EXIT_CLEAR = 16
EXIT_MODE = 10


@dataclass
class CaveExitController:
    """Walk out of an overworld cave: line up on the stairs, hold DOWN."""

    max_frames: int = 600
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    clear: int = CAVE_EXIT_CLEAR
    _out_y: int = -1

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        # Mode 10 is the walk out. On a burn cave's stairs, DOWN held
        # through it walks Link back down (600 frames on 0x47), and he comes
        # up beside the stairs, so those exits clear nothing (``clear`` 0).
        if snap.mode == EXIT_MODE:
            return idle("exit_rise")
        if snap.level == 0 and snap.mode == PLAY_MODE and self.clear <= 0:
            self.success = True
            return idle("cave_exited")
        if snap.level == 0 and snap.mode == PLAY_MODE:
            # Out on the mouth tile a sideways align slid back into the
            # 0x0C shop (11790 frames). The cell below a mouth is open:
            # Link walked in from it.
            if self._out_y < 0:
                self._out_y = int(snap.link_y)
            if int(snap.link_y) < self._out_y + self.clear:
                return push("DOWN", "cave_exit_clear")
            self.success = True
            return idle("cave_exited")
        if self.frames > self.max_frames:
            self.failed = True
            self.notes.append("cave_exit_timeout")
            return idle("cave_exit_timeout")
        if snap.mode == CAVE_MODE and abs(int(snap.link_x) - CAVE_EXIT_X) > 1:
            side = "RIGHT" if int(snap.link_x) < CAVE_EXIT_X else "LEFT"
            return push(side, "cave_exit_align")
        return push("DOWN", "cave_exit")

    def report(self) -> dict[str, Any]:
        return {"success": self.success, "frames": self.frames, "notes": list(self.notes)}


@dataclass
class WhiteReturnController:
    """The L9 detour's measured way back: cave, top band, corridor, 0x1A.

    Stops on 0x1A. The detour's own back legs go on to 0x05, not here.
    """

    max_frames: int = 3000
    success: bool = False
    failed: bool = False
    _detour: WhiteSwordDetourController | None = None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._detour is None:
            self._detour = WhiteSwordDetourController(
                max_frames=self.max_frames,
                phase=WhiteSwordPhase.EXIT_CAVE,
                start_checked=True,
            )
        if self._detour.phase is WhiteSwordPhase.BACK_LEGS:
            self.success = True
            return idle("on_0x1a")
        action = self._detour.step(snap)
        if self._detour.failed:
            self.failed = True
        return action

    def report(self) -> dict[str, Any]:
        return {} if self._detour is None else self._detour.report()


def _has_letter(snap: ZeldaSnapshot) -> bool:
    return int(snap.letter) >= 1


@dataclass
class NortheastController(OverworldPathController):
    hops: tuple[ScreenHop, ...] = NE_HOPS
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 12000
    interior_budget: int = 700
    _cave_rupees: int = -1
    _interior: int = 0

    def _wants_post_hop(self) -> bool:
        return True

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        """The whole 100 in the wallet. A floor rupee on the walk is not it."""
        return (
            snap.level == 0
            and snap.screen == SECRET_0F_SCREEN
            and snap.mode == CAVE_MODE
            and self._cave_rupees >= 0
            and int(snap.rupees) >= self._cave_rupees + SECRET_REWARD
        )

    def _after_hops(self, snap: ZeldaSnapshot):
        if snap.level != 0 or snap.screen != SECRET_0F_SCREEN:
            return self._fail("not_on_0x0f")
        if snap.mode != CAVE_MODE:
            return super()._after_hops(snap)
        if self._cave_rupees < 0:
            self._cave_rupees = int(snap.rupees)
        self._interior += 1
        if self._interior > self.interior_budget:
            return self._fail("secret_rupees_not_taken")
        return centre_item_walk(snap, SECRET_MOBLIN, "secret")


@dataclass
class ArrivalController(OverworldPathController):
    hops: tuple[ScreenHop, ...] = ()
    farm_below_hearts: int = 0
    evade: bool = False
    max_frames: int = 5000
    screen: int = 0
    aim_x: int = 128
    aim_y: int = 96
    aim_dir: str = "UP"
    _pushed: int = 0

    def _wants_post_hop(self) -> bool:
        return True

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return False

    def _after_hops(self, snap: ZeldaSnapshot):
        if snap.level != 0 or snap.screen != self.screen:
            return self._fail(f"not_on_{self.screen:#04x}")
        if snap.mode == CAVE_MODE:
            return self._finish("cave_open")
        if snap.mode != PLAY_MODE:
            return push(self.aim_dir, "wait_play")
        walk = step_toward(self, snap, self.aim_x, self.aim_y, "opening")
        if walk is not None:
            return walk
        self._pushed += 1
        if self._pushed < 30:
            return push(self.aim_dir, "try_opening")
        return self._fail("opening_closed")


def _candle(snap: ZeldaSnapshot) -> int:
    return int(snap.candle)


def make_candle_controller() -> CaveShopBuyController:
    return CaveShopBuyController(
        hops=CANDLE_HOPS,
        shop_screen=0x0C,
        cave_x=CAVE_X,
        cave_y=CAVE_Y,
        buy_x=BUY_X,
        buy_y=BUY_Y,
        price=CANDLE_PRICE,
        success_getter=_candle,
        success_threshold=1,
        success_addr=ADDR_CANDLE,
        success_note="blue_candle",
        farm_below_hearts=0,
        evade=False,
        max_frames=10000,
        require_sword=True,
    )


def make_heart_l8_controller() -> BombWallController:
    return BombWallController(
        hops=HEART_L8_HOPS,
        max_frames=6000,
        screen=HEART_L8_SCREEN,
        bomb_x=144,
        bomb_y=88,
        retreat="DOWN",
        interior_x=HEART_L8_ITEM_XY[0],
        interior_y=HEART_L8_ITEM_XY[1],
        bomb_fail="north_wall_did_not_open",
        leave_fail="left_0x7b",
    )


def make_heart_m3_controller() -> BombWallController:
    return BombWallController(
        hops=(),
        max_frames=5000,
        screen=HEART_M3_SCREEN,
        bomb_x=HEART_M3_BOMB_XY[0],
        bomb_y=HEART_M3_BOMB_XY[1],
        bomb_face="UP",
        approach=HEART_M3_APPROACH,
        bomb_y_first=True,
        retreat="DOWN",
        interior_x=HEART_L8_ITEM_XY[0],
        interior_y=HEART_L8_ITEM_XY[1],
        bomb_fail="center_rock_did_not_open",
        leave_fail="left_0x2c",
    )


def make_burn_48_controller() -> BombWallController:
    return BombWallController(
        hops=(),
        max_frames=3000,
        screen=0x48,
        bomb_x=BURN_48_STAND[0],
        bomb_y=BURN_48_STAND[1],
        bomb_face="RIGHT",
        # Arrives at (120, 61) from 0x38. Trees fill x>=144 above y=93.
        bomb_y_first=True,
        retreat=None,
        use_wait=FLAME_WAIT,
        reward="rupees",
        reward_rupees=SECRET_48_RUPEES,
        keeper=SECRET_48_KEEPER,
        interior_budget=700,
        bomb_fail="tree_0x48_did_not_open",
        leave_fail="left_0x48",
        consumes_bomb=False,
    )


def make_burn_47_controller() -> BombWallController:
    return BombWallController(
        hops=(ScreenHop(0x47, "LEFT", align_y=141),),
        max_frames=4000,
        screen=0x47,
        bomb_x=BURN_47_STAND[0],
        bomb_y=BURN_47_STAND[1],
        bomb_face="DOWN",
        retreat=None,
        use_wait=FLAME_WAIT,
        interior_x=HEART_L8_ITEM_XY[0],
        interior_y=HEART_L8_ITEM_XY[1],
        bomb_fail="tree_0x47_did_not_open",
        leave_fail="left_0x47",
        consumes_bomb=False,
    )


def make_letter_controller() -> CaveMouthController:
    return CaveMouthController(
        hops=LETTER_HOPS,
        max_frames=4000,
        screen=0x0E,
        push_budget=240,
        miss_fail="letter_stairs_not_found",
        off_fail="not_on_0x0e",
        keeper=LETTER_KEEPER,
        taken=_has_letter,
    )


def make_white_controller() -> GatherWhiteController:
    return GatherWhiteController()


def make_potion_controller() -> ArrivalController:
    return ArrivalController(
        hops=POTION_HOPS,
        max_frames=5000,
        screen=0x64,
        aim_x=128,
        aim_y=80,
        aim_dir="UP",
    )


def make_ring_controller() -> ArrivalController:
    return ArrivalController(
        hops=RING_HOPS,
        max_frames=8000,
        screen=0x34,
        aim_x=128,
        aim_y=88,
        aim_dir="UP",
    )


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    from_state: str
    enter: str
    leave: str
    factory: Callable[[], Any]
    bombs: int | None = None
    rupees: int | None = None
    candle: int | None = None
    select: str | None = None
    # A red pose is still written for the other stops. The heart leave is not.
    save_red: bool = True
    # Whole hearts at which the health assist refills. 1 is last-heart.
    engage_hearts: int = 1


SEGMENTS: dict[str, SegmentSpec] = {
    "gather_ne": SegmentSpec(
        name="gather_ne",
        from_state="BFS_2C",
        enter="GatherNEEnter",
        leave="GatherNE100Leave",
        factory=NortheastController,
    ),
    "gather_letter": SegmentSpec(
        name="gather_letter",
        from_state="BFS_1E",
        enter="GatherLetterEnter",
        leave="GatherLetterLeave",
        factory=make_letter_controller,
    ),
    "gather_candle": SegmentSpec(
        name="gather_candle",
        from_state="BFS_0E",
        enter="GatherCandleEnter",
        leave="GatherCandleLeave",
        factory=make_candle_controller,
        rupees=80,
    ),
    "heart_m3": SegmentSpec(
        name="heart_m3",
        from_state="BFS_2C",
        enter="GatherHeartM3Enter",
        leave="GatherHeartM3Leave",
        factory=make_heart_m3_controller,
        bombs=8,
        select="bomb",
        save_red=False,
    ),
    "gather_white": SegmentSpec(
        name="gather_white",
        from_state="BFS_0C",
        enter="GatherWhiteEnter",
        leave="GatherWhiteLeave",
        factory=make_white_controller,
        engage_hearts=2,
        save_red=False,
    ),
    "heart_l8": SegmentSpec(
        name="heart_l8",
        from_state="BFS_7C",
        enter="GatherHeartL8Enter",
        leave="GatherHeartL8Leave",
        factory=make_heart_l8_controller,
        bombs=8,
        select="bomb",
        save_red=False,
    ),
    "gather_potion": SegmentSpec(
        name="gather_potion",
        from_state="BFS_65",
        enter="GatherPotionEnter",
        leave="GatherPotionLeave",
        factory=make_potion_controller,
    ),
    "gather_ring": SegmentSpec(
        name="gather_ring",
        from_state="BFS_37",
        enter="GatherRingEnter",
        leave="GatherRingLeave",
        factory=make_ring_controller,
    ),
}


# Bomb shop to White Sword: 0x7B heart, 0x2C heart, NE 100, letter,
# candle, White Sword. One env, no pin between stops. The start is the
# power-on pre-l1 leave (``pin`` writes it).
CHAIN_FROM = PRE_L1_LEAVE
CHAIN_NAME = "GatherChain"


def chain_stages() -> list[tuple[str, Any]]:
    letter = make_letter_controller()
    letter.hops = LETTER_FROM_0F_HOPS
    return [
        ("exit_6f", CaveExitController()),
        ("walk_7c", HopWalkController(hops=RETURN_7C_HOPS)),
        ("heart_7b", make_heart_l8_controller()),
        ("exit_7b", CaveExitController()),
        ("walk_pond", HopWalkController(hops=POND_WALK_HOPS)),
        ("pond_39", PondFairyController()),
        ("walk_2c", HopWalkController(hops=POND_RETURN_HOPS, waypoints=POND_RETURN_WAYPOINTS)),
        ("heart_2c", make_heart_m3_controller()),
        ("exit_2c", CaveExitController()),
        ("ne_100", NortheastController()),
        ("exit_0f", CaveExitController()),
        ("letter", letter),
        ("exit_0e", CaveExitController()),
        ("candle", make_candle_controller()),
        ("exit_0c", CaveExitController()),
        ("white", make_white_controller()),
        ("back_1a", WhiteReturnController()),
        ("walk_48", HopWalkController(hops=BURN_WALK_HOPS, waypoints={})),
        ("select_candle", PauseSelectController(want=B_ITEM_CANDLE)),
        ("rupees_48", make_burn_48_controller()),
        ("exit_48", CaveExitController(clear=0)),
        ("heart_47", make_burn_47_controller()),
        ("exit_47", CaveExitController(clear=0)),
        ("walk_pond_l1", HopWalkController(hops=L1_POND_HOPS, waypoints={})),
        ("pond_39_l1", PondFairyController()),
        ("walk_37", HopWalkController(hops=L1_FROM_POND_HOPS, waypoints={})),
    ]


def _run_segment(spec: SegmentSpec) -> dict[str, Any]:
    poke: dict[str, Any] = {}
    if spec.bombs is not None:
        poke["bombs"] = spec.bombs
    if spec.rupees is not None:
        poke["rupees"] = spec.rupees
    if spec.candle is not None:
        poke["candle"] = spec.candle
    if spec.select is not None:
        poke["select"] = spec.select
    write_entry_pin(spec.from_state, spec.enter, segment=spec.name, **poke)
    return run_segment(
        spec.factory(),
        from_state=spec.enter,
        segment=spec.name,
        save_as=spec.leave,
        save_red=spec.save_red,
        engage_hearts=spec.engage_hearts,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        for name in SEGMENTS:
            print(name)
        return 2
    if len(args) != 1:
        return 2
    token = args[0]
    if token == "pin":
        return 0 if pin_pre_l1()["ok"] else 1
    if token == "chain":
        result = run_chain(
            chain_stages(), from_state=CHAIN_FROM, chain=CHAIN_NAME, engage_hearts=2
        )
        return 0 if result["ok"] else 1
    if token.startswith("chain:"):
        # Resume after a green stage's saved pose, e.g. chain:exit_0e.
        after = token.split(":", 1)[1]
        stages = chain_stages()
        names = [name for name, _ in stages]
        if after not in names:
            return 2
        result = run_chain(
            stages[names.index(after) + 1 :],
            from_state=f"{CHAIN_NAME}_{after}",
            chain=CHAIN_NAME,
            engage_hearts=2,
        )
        return 0 if result["ok"] else 1
    if token == "all":
        results = [_run_segment(spec) for spec in SEGMENTS.values()]
        return 0 if all(r["ok"] for r in results) else 1
    spec = SEGMENTS.get(token)
    if spec is None:
        return 2
    result = _run_segment(spec)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
