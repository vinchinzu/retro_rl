"""Level 5 multi-room path policy (public facade).

East key: 0x66 → 0x76 east key door → 0x77.
Post-east-key: 0x77 → 0x76 → 0x66 free UP → 0x56.
West hops live in ``level5_west_path`` (0x27 → 0x26 → 0x25 → 0x24).
Whistle inbound lives in ``level5_whistle_path``; cellar/east return in
``level5_cellar_path``; Digdogger/TF north hop in ``level5_tf_path``.

Callers should keep importing from this module.

Room specs and stop predicates remain in ``level5_dungeon``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_NORTH_56,
    ROOM_L5_POLS_77,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

EAST_DOOR_APPROACH_Y = 157
EAST_DOOR_CHANNEL_Y = 141
# Live statue blocks UP at x=200 (stuck 200,149). Start y-slide at wall x≈208.
EAST_DOOR_WALL_X = 208

_ISOLATED_ROOM_77 = "L5_Room_77"


def should_force_keys_zero(
    start_state: str,
    *,
    keep_keys: bool = False,
    force_keys_zero: bool = False,
) -> bool:
    """True only for isolated ``L5_Room_77`` combat, unless explicitly overridden."""
    if keep_keys:
        return False
    if force_keys_zero:
        return True
    return start_state == _ISOLATED_ROOM_77


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


@dataclass
class Level5West65Controller:
    """Walk 0x77→0x76→0x66 free UP→0x56; stop on arrival.

    No combat. Success = room-ready 0x56. Does not poke keys or doors.
    """

    max_frames: int = 8000
    settle_frames: int = 45
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    last_room: int = -1

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level5_north56_from_east_key",
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if snap.screen != self.last_room:
            self.notes.append(
                f"room_0x{snap.screen:02x}_f{self.frames}_xy={snap.link_x},{snap.link_y}_k={snap.keys}"
            )
            self.last_room = snap.screen

        if (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_NORTH_56
            and snap.mode == PLAY_MODE
        ):
            if self.settle_left <= 0 and "settling_56" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_56")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_56")
            self.success = True
            self.notes.append("arrived_56")
            return FrameAction(nes_idle_action(), "arrived_56")

        return level5_west65_step(snap)


def make_west65_controller() -> Level5West65Controller:
    return Level5West65Controller()


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


@dataclass
class Level5Return66Controller:
    """Walk 0x77→0x76→0x66; stop room-ready on 0x66. No combat. No pokes."""

    max_frames: int = 8000
    settle_frames: int = 30
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    last_room: int = -1

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level5_return66_from_east_key",
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.screen != self.last_room:
            self.notes.append(
                f"room_0x{snap.screen:02x}_f{self.frames}_xy={snap.link_x},{snap.link_y}_k={snap.keys}"
            )
            self.last_room = snap.screen
        if (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_GIBDO_66
            and snap.mode == PLAY_MODE
            and snap.link_y <= 185
        ):
            if self.settle_left <= 0 and "settling_66" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_66")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_66")
            self.success = True
            self.notes.append("arrived_66")
            return FrameAction(nes_idle_action(), "arrived_66")
        return level5_return_66_step(snap)


def make_return_66_controller() -> Level5Return66Controller:
    return Level5Return66Controller()



def walk_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 500) -> bool:
    """Step one axis toward target. Used by the 0x27 west-key leave."""
    last = None
    stall = 0
    read_snapshot = __import__("zelda_i.ram", fromlist=["read_snapshot"]).read_snapshot
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        env.step(action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 40:
                return False
        else:
            stall = 0
        last = pos
    return False



def _step(env, assist, total: list[int], action) -> None:
    env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])



_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # west
    "SOUTH_PINCH_Y": ("zelda_i.level5.west_path", "SOUTH_PINCH_Y"),
    "WEST_DOOR_X": ("zelda_i.level5.west_path", "WEST_DOOR_X"),
    "BOMB_WEST_66_STAND": ("zelda_i.level5.whistle_path", "BOMB_WEST_66_STAND"),
    "bomb_west_from_66": ("zelda_i.level5.whistle_path", "bomb_west_from_66"),
    "Level5West26From27Controller": ("zelda_i.level5.west_path", "Level5West26From27Controller"),
    "level5_west26_from_27_step": ("zelda_i.level5.west_path", "level5_west26_from_27_step"),
    "make_west26_from_27_controller": ("zelda_i.level5.west_path", "make_west26_from_27_controller"),
    "walk_west_from_27": ("zelda_i.level5.west_path", "walk_west_from_27"),
    "walk_west_from_26": ("zelda_i.level5.west_path", "walk_west_from_26"),
    "walk_west_from_25": ("zelda_i.level5.west_path", "walk_west_from_25"),
    "WEST26_TO_25_PATHS": ("zelda_i.level5.west_path", "WEST26_TO_25_PATHS"),
    "WEST25_TO_24_PATHS": ("zelda_i.level5.west_path", "WEST25_TO_24_PATHS"),
    # whistle inbound
    "ROOM_L5_BLUE_64": ("zelda_i.level5.whistle_path", "ROOM_L5_BLUE_64"),
    "ROOM_L5_CELLAR_07": ("zelda_i.level5.whistle_path", "ROOM_L5_CELLAR_07"),
    "ROOM_L5_PASSAGE_06": ("zelda_i.level5.whistle_path", "ROOM_L5_PASSAGE_06"),
    "ROOM_L5_WHISTLE_05": ("zelda_i.level5.whistle_path", "ROOM_L5_WHISTLE_05"),
    "ROOM_L5_WHISTLE_ITEM": ("zelda_i.level5.whistle_path", "ROOM_L5_WHISTLE_ITEM"),
    "BLUE_DARKNUT_TYPE": ("zelda_i.level5.whistle_path", "BLUE_DARKNUT_TYPE"),
    "BOMB_WEST_STAND": ("zelda_i.level5.whistle_path", "BOMB_WEST_STAND"),
    "BOMB_EAST_STAND": ("zelda_i.level5.whistle_path", "BOMB_EAST_STAND"),
    "CENTER_STAIRS": ("zelda_i.level5.whistle_path", "CENTER_STAIRS"),
    "CELLAR_MODES": ("zelda_i.level5.whistle_path", "CELLAR_MODES"),
    "L5_CELLAR_FLOOR_Y": ("zelda_i.level5.whistle_path", "L5_CELLAR_FLOOR_Y"),
    "L5_CELLAR_LEFT_X": ("zelda_i.level5.whistle_path", "L5_CELLAR_LEFT_X"),
    "L5_CELLAR_RIGHT_X": ("zelda_i.level5.whistle_path", "L5_CELLAR_RIGHT_X"),
    "ROOM_06_BLOCK_X": ("zelda_i.level5.whistle_path", "ROOM_06_BLOCK_X"),
    "ROOM_06_BLOCK_REST_Y": ("zelda_i.level5.whistle_path", "ROOM_06_BLOCK_REST_Y"),
    "ROOM_06_BLOCK_PUSHED_Y": ("zelda_i.level5.whistle_path", "ROOM_06_BLOCK_PUSHED_Y"),
    "ROOM_06_STAIRS_X": ("zelda_i.level5.whistle_path", "ROOM_06_STAIRS_X"),
    "ROOM_06_STAIRS_Y": ("zelda_i.level5.whistle_path", "ROOM_06_STAIRS_Y"),
    "WHISTLE_04_LADDER_X": ("zelda_i.level5.whistle_path", "WHISTLE_04_LADDER_X"),
    "WHISTLE_04_PIT_Y": ("zelda_i.level5.whistle_path", "WHISTLE_04_PIT_Y"),
    "WHISTLE_04_MOUTH_X": ("zelda_i.level5.whistle_path", "WHISTLE_04_MOUTH_X"),
    "WHISTLE_04_MOUTH_Y": ("zelda_i.level5.whistle_path", "WHISTLE_04_MOUTH_Y"),
    "select_b_item_menu": ("zelda_i.level5.whistle_path", "select_b_item_menu"),
    "bomb_west_from_65": ("zelda_i.level5.whistle_path", "bomb_west_from_65"),
    "bomb_east_from_65": ("zelda_i.level5.whistle_path", "bomb_east_from_65"),
    "take_center_stairs_64": ("zelda_i.level5.whistle_path", "take_center_stairs_64"),
    "cellar_other_mouth": ("zelda_i.level5.whistle_path", "cellar_other_mouth"),
    "key_west_to": ("zelda_i.level5.whistle_path", "key_west_to"),
    "fight_blue_darknuts": ("zelda_i.level5.whistle_path", "fight_blue_darknuts"),
    "push_block_stairs": ("zelda_i.level5.whistle_path", "push_block_stairs"),
    "hunt_whistle": ("zelda_i.level5.whistle_path", "hunt_whistle"),
    "take_whistle_04": ("zelda_i.level5.whistle_path", "take_whistle_04"),
    "exit_whistle_04": ("zelda_i.level5.whistle_path", "exit_whistle_04"),
    "leave_whistle_cellar": ("zelda_i.level5.whistle_path", "leave_whistle_cellar"),
    "walk_out_of_04": ("zelda_i.level5.whistle_path", "walk_out_of_04"),
    # cellar / east return
    "_raw_axis": ("zelda_i.level5.cellar_path", "_raw_axis"),
    "take_center_stairs_06": ("zelda_i.level5.cellar_path", "take_center_stairs_06"),
    "cellar_to_64": ("zelda_i.level5.cellar_path", "cellar_to_64"),
    "walk_east_from_05": ("zelda_i.level5.cellar_path", "walk_east_from_05"),
    "take_block_stairs_06": ("zelda_i.level5.cellar_path", "take_block_stairs_06"),
    "cellar_07_to_64": ("zelda_i.level5.cellar_path", "cellar_07_to_64"),
    "walk_east_from_64": ("zelda_i.level5.cellar_path", "walk_east_from_64"),
    "EAST65_TO_66_PATHS": ("zelda_i.level5.cellar_path", "EAST65_TO_66_PATHS"),
    "walk_east_from_65": ("zelda_i.level5.cellar_path", "walk_east_from_65"),
    # TF / Digdogger north
    "ROOM_L5_EAST_ZOLS": ("zelda_i.level5.tf_path", "ROOM_L5_EAST_ZOLS"),
    "ROOM_L5_NORTH_GIBDOS": ("zelda_i.level5.tf_path", "ROOM_L5_NORTH_GIBDOS"),
    "STATUE_5F_TYPE": ("zelda_i.level5.tf_path", "STATUE_5F_TYPE"),
    "STATUE_5F_X": ("zelda_i.level5.tf_path", "STATUE_5F_X"),
    "STATUE_5F_Y": ("zelda_i.level5.tf_path", "STATUE_5F_Y"),
    "DIAMOND_NORTH_Y": ("zelda_i.level5.tf_path", "DIAMOND_NORTH_Y"),
    "NORTH_DOOR_Y": ("zelda_i.level5.tf_path", "NORTH_DOOR_Y"),
    "walk_north_from_57": ("zelda_i.level5.tf_path", "walk_north_from_57"),
}


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "EAST_DOOR_APPROACH_Y",
    "EAST_DOOR_CHANNEL_Y",
    "EAST_DOOR_WALL_X",
    "NORTH_DOOR_X",
    "SOUTH_PINCH_Y",
    "WEST_DOOR_X",
    "WEST_LEAVE_EAST_X",
    "Level5Return66Controller",
    "Level5West26From27Controller",
    "Level5West65Controller",
    "level5_east_key_step",
    "level5_return_66_step",
    "level5_west26_from_27_step",
    "level5_west65_step",
    "make_return_66_controller",
    "make_west26_from_27_controller",
    "walk_west_from_27",
    "walk_west_from_26",
    "walk_west_from_25",
    "walk_axis",
    "WEST26_TO_25_PATHS",
    "WEST25_TO_24_PATHS",
    "make_west65_controller",
    "should_force_keys_zero",
    "ROOM_L5_BLUE_64",
    "ROOM_L5_CELLAR_07",
    "ROOM_L5_PASSAGE_06",
    "ROOM_L5_WHISTLE_05",
    "ROOM_L5_WHISTLE_ITEM",
    "BLUE_DARKNUT_TYPE",
    "BOMB_WEST_STAND",
    "BOMB_EAST_STAND",
    "CENTER_STAIRS",
    "CELLAR_MODES",
    "BOMB_WEST_66_STAND",
    "bomb_west_from_65",
    "bomb_west_from_66",
    "bomb_east_from_65",
    "take_center_stairs_64",
    "cellar_other_mouth",
    "key_west_to",
    "fight_blue_darknuts",
    "push_block_stairs",
    "hunt_whistle",
    "take_whistle_04",
    "select_b_item_menu",
    "exit_whistle_04",
    "leave_whistle_cellar",
    "walk_out_of_04",
    "take_center_stairs_06",
    "_raw_axis",
    "cellar_to_64",
    "walk_east_from_05",
    "walk_east_from_64",
    "walk_north_from_57",
    "ROOM_L5_EAST_ZOLS",
    "ROOM_L5_NORTH_GIBDOS",
    "STATUE_5F_TYPE",
    "STATUE_5F_X",
    "STATUE_5F_Y",
    "DIAMOND_NORTH_Y",
    "NORTH_DOOR_Y",
    "walk_east_from_65",
    "EAST65_TO_66_PATHS",
    "take_block_stairs_06",
    "cellar_07_to_64",
    "WHISTLE_04_LADDER_X",
    "WHISTLE_04_PIT_Y",
    "WHISTLE_04_MOUTH_X",
    "WHISTLE_04_MOUTH_Y",
    "ROOM_06_BLOCK_X",
    "ROOM_06_BLOCK_REST_Y",
    "ROOM_06_BLOCK_PUSHED_Y",
    "ROOM_06_STAIRS_X",
    "ROOM_06_STAIRS_Y",
    "L5_CELLAR_FLOOR_Y",
    "L5_CELLAR_LEFT_X",
    "L5_CELLAR_RIGHT_X",
]
