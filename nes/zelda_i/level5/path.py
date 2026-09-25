"""Level 5 multi-room path policy.

East key: 0x66 → 0x76 east key door → 0x77.
Post-east-key: 0x77 → 0x76 → 0x66 free UP → 0x56.
West hops live in ``level5.west_path`` (0x27 → 0x26 → 0x25 → 0x24).
Whistle inbound lives in ``level5.whistle_path``; cellar/east return in
``level5.cellar_path``.

Room specs and stop predicates remain in ``level5.dungeon``.

``walk_axis`` / ``_step`` are the shared env-stepping glue for every L5
library hop; ``Level5NavSpec`` rows are the shared settle-on-arrival nav
controller (0x66 return, 0x77 east key).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon.behaviors import fight_target
from zelda_i.dungeon.engine import GenericDungeonRoomController
from zelda_i.dungeon.hop_controller import HopController
from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_NORTH_56,
    ROOM_L5_POLS_77,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

EAST_DOOR_APPROACH_Y = 157
EAST_DOOR_CHANNEL_Y = 141
# Live statue blocks UP at x=200 (stuck 200,149). Start y-slide at wall x≈208.
EAST_DOOR_WALL_X = 208


def level5_east_key_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x66→0x76→key door→0x77 navigation policy.

    Room 0x66 supplies the key.  Return through its south door, leave the
    0x76 north/south mouth, approach the east wall on y≈157, then move to the
    door channel y≈141 without stepping back into the center statues.
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "east_key_wait_level5")
    if snap.screen == ROOM_L5_POLS_77 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "east_key_arrived")

    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning:
            return FrameAction(nes_action("DOWN"), "east_key_south_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_66")
        # The fixed key can be collected while Link is still standing on the
        # Stepladder across the horizontal river. Finish the crossing before
        # horizontal alignment; sideways input is locked on the ladder tile.
        # Spine leftover (32,101) is the north bank, not the ladder — reach
        # x≈56 first. DOWN at x=32 never crosses.
        if snap.link_y < 141:
            if snap.link_y < 117 and abs(snap.link_x - 56) > 4:
                direction = "LEFT" if snap.link_x > 56 else "RIGHT"
                return FrameAction(nes_action(direction), "east_key_to_ladder_x")
            return FrameAction(nes_action("DOWN"), "east_key_finish_ladder")
        if abs(snap.link_x - 120) > 4:
            direction = "LEFT" if snap.link_x > 120 else "RIGHT"
            return FrameAction(nes_action(direction), "east_key_align_south_x")
        return FrameAction(nes_action("DOWN"), "east_key_return_76")

    if snap.screen == ROOM_L5_ENTRY:
        if snap.transitioning:
            return FrameAction(nes_action("RIGHT"), "east_key_east_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_76")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "east_key_leave_south_mouth")
        if snap.link_x < EAST_DOOR_WALL_X - 4:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                direction = "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                return FrameAction(nes_action(direction), "east_key_align_approach_y")
            return FrameAction(nes_action("RIGHT"), "east_key_approach_wall")
        if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
            direction = "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
            return FrameAction(nes_action(direction), "east_key_align_channel_y")
        return FrameAction(nes_action("RIGHT"), "east_key_unlock_77")

    return FrameAction(
        nes_idle_action(), f"east_key_unexpected_room_0x{snap.screen:02x}"
    )


ROOM66_NORTH_BANK_Y = 101
ROOM66_WEST_AISLE_X = 48


def level5_room66_west_aisle_north_step(snap: ZeldaSnapshot) -> FrameAction:
    """West bomb-hole (32,141): UP to north bank. y=141 RIGHT is the river.

    TF suffix fight chased south Gibdos; leftover (79,165) then (152,189)
    with 1 Gibdo north. Historical fight leftover (46,101) is this aisle.
    """
    if snap.level != LEVEL_5 or snap.screen != ROOM_L5_GIBDO_66:
        return FrameAction(nes_idle_action(), "66_aisle_wait")
    if snap.link_y > ROOM66_NORTH_BANK_Y + 2:
        return FrameAction(nes_action("UP"), "66_west_aisle_up")
    if abs(snap.link_x - ROOM66_WEST_AISLE_X) > 4:
        btn = "LEFT" if snap.link_x > ROOM66_WEST_AISLE_X else "RIGHT"
        return FrameAction(nes_action(btn), "66_west_aisle_x")
    return FrameAction(nes_idle_action(), "66_north_bank")


NORTH_DOOR_X = 120
WEST_LEAVE_EAST_X = 140


def level5_west65_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x77→0x76→0x66 free UP→0x56 navigation policy.

    Source route after the east key: return west, north through 0x66, then UP
    into the next dark room (live 0x56). Reuse the 0x76 statue bypass (y≈157).
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "west65_wait_level5")
    if snap.screen == ROOM_L5_NORTH_56 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "west65_arrived_56")

    if snap.screen == ROOM_L5_POLS_77:
        if snap.transitioning:
            return FrameAction(nes_action("LEFT"), "west65_west_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_77")
        # Two 2x3 block clusters pinch y=141 around x≈96. Pass south (y≈173)
        # then drop to the west-door channel once past the left cluster.
        if snap.link_x > 48:
            if abs(snap.link_y - 173) > 3:
                direction = "UP" if snap.link_y > 173 else "DOWN"
                return FrameAction(nes_action(direction), "west65_align_77_south_y")
            return FrameAction(nes_action("LEFT"), "west65_pass_77_blocks")
        if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
            direction = "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
            return FrameAction(nes_action(direction), "west65_align_77_y")
        return FrameAction(nes_action("LEFT"), "west65_return_76")

    if snap.screen == ROOM_L5_ENTRY:
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_76")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "west65_leave_south_mouth")
        # East doorway (x≈224,y≈141) blocks DOWN. Step out to the wall
        # before the statue-bypass y-slide.
        if snap.link_x > EAST_DOOR_WALL_X:
            return FrameAction(nes_action("LEFT"), "west65_leave_east_mouth")
        if snap.link_x > WEST_LEAVE_EAST_X:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                direction = "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                return FrameAction(nes_action(direction), "west65_align_approach_y")
            return FrameAction(nes_action("LEFT"), "west65_leave_east_door")
        if abs(snap.link_x - NORTH_DOOR_X) > 4:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "west65_align_north_x")
        return FrameAction(nes_action("UP"), "west65_enter_66")

    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_66")
        if snap.link_y < 141 and snap.link_x < 80:
            return FrameAction(nes_action("DOWN"), "west65_finish_ladder")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "west65_leave_66_south")
        if abs(snap.link_x - NORTH_DOOR_X) > 4:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "west65_align_66_north_x")
        return FrameAction(nes_action("UP"), "west65_enter_56")

    return FrameAction(
        nes_idle_action(), f"west65_unexpected_room_0x{snap.screen:02x}"
    )


def level5_return_66_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x77→0x76→0x66 return. Stop in cleared 0x66.

    Same statue-bypass leave as ``level5_west65_step``, but do not take the
    free UP into Dodongos (0x56). West of 0x66 is a ROM bomb wall to 0x65.
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "return66_wait_level5")
    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning:
            return FrameAction(nes_action("UP"), "return66_north_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "return66_wait_66")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "return66_leave_south")
        return FrameAction(nes_idle_action(), "return66_arrived")
    return level5_west65_step(snap)


@dataclass(frozen=True)
class Level5NavSpec:
    """One settle-on-arrival L5 walk: policy step, destination, budgets."""

    spec_id: str
    dest_room: int
    step: Callable[[ZeldaSnapshot], FrameAction]
    max_frames: int
    settle_frames: int
    arrived_extra: Callable[[ZeldaSnapshot], bool] | None = None

    @property
    def tag(self) -> str:
        return f"{self.dest_room:02x}"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL_5
            and snap.screen == self.dest_room
            and snap.mode == PLAY_MODE
            and (self.arrived_extra is None or self.arrived_extra(snap))
        )


@dataclass
class Level5NavController:
    """Walk one ``Level5NavSpec`` to its room; settle, then stop.

    No combat. Does not poke keys or doors.
    """

    spec: Level5NavSpec
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    last_room: int = -1

    @property
    def max_frames(self) -> int:
        return self.spec.max_frames

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec.spec_id,
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        spec = self.spec
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= spec.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.screen != self.last_room:
            self.notes.append(
                f"room_0x{snap.screen:02x}_f{self.frames}"
                f"_xy={snap.link_x},{snap.link_y}_k={snap.keys}"
            )
            self.last_room = snap.screen
        if spec.arrived(snap):
            if self.settle_left <= 0 and f"settling_{spec.tag}" not in self.notes:
                self.settle_left = spec.settle_frames
                self.notes.append(f"settling_{spec.tag}")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), f"settle_{spec.tag}")
            self.success = True
            self.notes.append(f"arrived_{spec.tag}")
            return FrameAction(nes_idle_action(), f"arrived_{spec.tag}")
        return spec.step(snap)


RETURN_66_NAV = Level5NavSpec(
    spec_id="level5_return66_from_east_key",
    dest_room=ROOM_L5_GIBDO_66,
    step=level5_return_66_step,
    max_frames=8000,
    settle_frames=30,
    arrived_extra=lambda snap: snap.link_y <= 185,
)

EAST_KEY_77_NAV = Level5NavSpec(
    spec_id="level5_east_key_nav_0x77",
    dest_room=ROOM_L5_POLS_77,
    step=level5_east_key_step,
    max_frames=8000,
    settle_frames=40,
    arrived_extra=lambda snap: not snap.transitioning,
)


def make_return_66_controller() -> Level5NavController:
    return Level5NavController(RETURN_66_NAV)


def make_east_key_nav_controller() -> Level5NavController:
    return Level5NavController(EAST_KEY_77_NAV)


# Clean leftover (64,120): occupancy no-path stood 20000f. NW pocket peels
# DOWN to the y=149 patrol row, then RIGHT; contact_backstep stays 16.
_ROOM66_POCKET_X = 80
_ROOM66_POCKET_Y = 133
_ROOM66_SOUTH_ROW_Y = 149


@dataclass
class Level5Room66Controller(GenericDungeonRoomController):
    """0x66 Gibdos: leave the NW occupancy pocket south instead of standing."""

    _pocket_peel: bool = False

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        target = fight_target(x, y, live) if live else None
        dist = (
            abs(int(target.x) - x) + abs(int(target.y) - y)
            if target is not None
            else 10**9
        )
        close = (
            target is not None
            and self.spec.combat.contact_backstep > 0
            and dist < self.spec.combat.contact_backstep
        )
        if target is not None and not close and x <= _ROOM66_POCKET_X:
            outside = (
                int(target.x) > _ROOM66_POCKET_X or int(target.y) > _ROOM66_POCKET_Y
            )
            still_north = y <= _ROOM66_POCKET_Y or (
                self._pocket_peel and y < _ROOM66_SOUTH_ROW_Y
            )
            if outside and still_north:
                self._pocket_peel = True
                self.combat_frames += 1
                self.walker.last_dir = "DOWN"
                return FrameAction(nes_action("DOWN"), "66_pocket_south")
            if y >= _ROOM66_SOUTH_ROW_Y:
                self._pocket_peel = False
        act = super()._combat(snap, live)
        if (
            target is not None
            and not close
            and act.reason == "combat_wait"
            and x <= _ROOM66_POCKET_X
            and y >= _ROOM66_SOUTH_ROW_Y
        ):
            self.walker.last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "66_pocket_east")
        return act


def make_room66_controller(spec=None) -> Level5Room66Controller:
    return Level5Room66Controller(spec=spec if spec is not None else ROOM_66_SPEC)


def make_pols_south_controller() -> GenericDungeonRoomController:
    """L5 0x77: five Pols Voice and the key, on the generic room controller."""
    return GenericDungeonRoomController(spec=ROOM_77_SPEC)


def walk_axis(
    env,
    assist,
    total: list[int],
    axis: str,
    target: int,
    max_f: int = 500,
    *,
    stall_limit: int = 40,
    done: Callable[[ZeldaSnapshot], bool] | None = None,
) -> bool:
    """Step one axis toward ``target``; True on arrival (or ``done``).

    ``done`` short-circuits when the room already changed (cellar walks);
    ``stall_limit`` is how many frames of a frozen (x, y) end the walk —
    40 for play rooms, higher where fanfare or a mode-9 cellar freezes Link.
    """
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if done is not None and done(snap):
            return True
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        _step(env, assist, total, action)
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= stall_limit:
                return False
        else:
            stall = 0
        last = pos
    return False


def _step(env, assist, total: list[int], action) -> None:
    from zelda_i.dungeon.pause_select import drink_if_low

    drink_if_low(env, assist, total)
    env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])


@dataclass(kw_only=True)
class RamWaitHop(HopController):
    """Idle or hold until a RAM predicate. Dest is RAM; max_frames is the budget."""

    pred: Callable[[ZeldaSnapshot], bool] = field(repr=False)
    hold: str | None = None
    require_level: int | None = LEVEL_5

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return self.pred(snap)

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        if self.hold:
            return FrameAction(nes_action(self.hold), f"{self.hold.lower()}_scroll")
        return FrameAction(nes_idle_action(), "wait_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        if self.hold:
            return FrameAction(nes_action(self.hold), f"hold_{self.hold.lower()}")
        return FrameAction(nes_idle_action(), "wait_ram")


def drive_hop(env, assist, total: list[int], ctl: HopController) -> bool:
    """Step a dest-hop until success/fail. Dest is RAM; no idle(n)."""
    while not ctl.success and not ctl.failed:
        action = ctl.step(read_snapshot(env.get_ram()))
        _step(env, assist, total, action.action)
    return bool(ctl.success)


def wait_ram(
    env,
    assist,
    total: list[int],
    pred: Callable[[ZeldaSnapshot], bool],
    *,
    hold: str | None = None,
    max_frames: int = 240,
    spec_id: str = "wait_ram",
) -> bool:
    """Hold or idle until ``pred`` (room/mode/band/census). Budget, not a hop."""
    return drive_hop(
        env,
        assist,
        total,
        RamWaitHop(pred=pred, hold=hold, max_frames=max_frames, spec_id=spec_id),
    )


__all__ = [
    "EAST_DOOR_APPROACH_Y",
    "EAST_DOOR_CHANNEL_Y",
    "EAST_DOOR_WALL_X",
    "EAST_KEY_77_NAV",
    "Level5NavController",
    "Level5NavSpec",
    "NORTH_DOOR_X",
    "RETURN_66_NAV",
    "ROOM66_NORTH_BANK_Y",
    "ROOM66_WEST_AISLE_X",
    "WEST_LEAVE_EAST_X",
    "level5_east_key_step",
    "level5_return_66_step",
    "level5_room66_west_aisle_north_step",
    "level5_west65_step",
    "drive_hop",
    "Level5Room66Controller",
    "Level5PolsSouthController",
    "make_east_key_nav_controller",
    "make_return_66_controller",
    "make_room66_controller",
    "make_pols_south_controller",
    "RamWaitHop",
    "wait_ram",
    "walk_axis",
]
