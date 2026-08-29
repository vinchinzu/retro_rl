"""L6 Survival hop helpers + first-half rows (entry through stairs18)."""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import TF_BIT_L5
from zelda_i.level4_boss_combat import gleeok_heads_live
from zelda_i.level6_door_hop import DoorHopSpec, door_hop_stages, door_hop_success
from zelda_i.level6_dungeon import (
    LEVEL6_COMPASS_BIT,
    ROOM_28_SPEC,
    ROOM_38_SPEC,
    ROOM_58_SPEC,
    ROOM_68_SPEC,
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
)
from zelda_i.level6_gleeok18 import (
    east_door_open,
    gleeok_3head_live,
    make_gleeok_18_controller,
    make_postgleeok_18_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_SETTLE_MAX_FRAMES,
    Level6EntryRightController,
    Level6WestKeyDoorController,
    PostL5TriforceSettleController,
    make_post_l5_level6_controller,
)
from zelda_i.level6_path import (
    SETTLE_18_MAX_FRAMES,
    Level6North68Controller,
    make_north_18_controller,
    make_north_28_controller,
    make_north_38_controller,
    make_north_48_controller,
    make_north_58_controller,
    make_settle_18_controller,
)
from zelda_i.level6_room19 import SETTLE_19_MAX_FRAMES
from zelda_i.level6_stairs18 import make_stairs_18_controller
from zelda_i.level6_wizzrobe import (
    make_east_key_controller,
    make_west_wizzrobe_controller,
)
from zelda_i.ram import ADDR_WHISTLE, PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot, read_u8
from zelda_i.spine_hops import SpineHop, fight_stage, play_ready, ready

__all__ = [
    "Level6Return79Controller",
    "l6_prefix",
    "ok6",
    "one_hop",
    "door_row",
    "fight_hop",
    "settle_fight",
    "stairs_or_play",
]


@dataclass
class Level6Return79Controller:
    """Free LEFT 0x7a → 0x79. Never UP (Old Man wastes the key)."""

    max_frames: int = 4000
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level6_return_0x79",
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
        if (
            snap.level == LEVEL6
            and snap.screen == LEVEL6_ENTRY_ROOM
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("arrived_79")
            return FrameAction(nes_idle_action(), "arrived_79")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            return FrameAction(nes_action("LEFT"), "return_scroll")
        if snap.level != LEVEL6 or snap.screen != LEVEL6_EAST_KEY_ROOM:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        if abs(snap.link_y - 141) > 4:
            btn = "DOWN" if snap.link_y < 141 else "UP"
            return FrameAction(nes_action(btn), "return_ay")
        return FrameAction(nes_action("LEFT"), "return_left")


def ok6(**kw):
    return ready(level=LEVEL6, **kw)


def stairs_or_play(snap: ZeldaSnapshot, *, not_screen: int, **_) -> bool:
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    return play_ready(snap, level=LEVEL6, not_screen=not_screen, tf_eq=0x1F)


def _gleeok18_ok(snap: ZeldaSnapshot, **_) -> bool:
    return play_ready(
        snap, level=LEVEL6, screen=LEVEL6_GLEEOK_ROOM, tf_eq=0x1F
    ) and not gleeok_3head_live(snap)


def _postgleeok18_ok(snap: ZeldaSnapshot, **_) -> bool:
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if gleeok_3head_live(snap):
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    if not play_ready(snap, level=LEVEL6, screen=LEVEL6_GLEEOK_ROOM):
        return False
    return (not gleeok_heads_live(snap)) or east_door_open(snap)


def one_hop(through, stop, factory, success, *, dedicated=False, name=None):
    stage = name or stop

    def stages():
        ctl = factory()
        return ((stage, ctl, ctl.max_frames),)

    return SpineHop(through, stop, stages, success, dedicated=dedicated)


def door_row(through: str, spec: DoorHopSpec, *, dedicated: bool = False) -> SpineHop:
    return SpineHop(
        through,
        spec.spec_id,
        lambda s=spec: door_hop_stages(s),
        lambda snap, s=spec, **_: door_hop_success(s, snap),
        dedicated=dedicated,
    )


def fight_hop(through, stop, spec, **kw) -> SpineHop:
    return SpineHop(
        through,
        stop,
        lambda: (fight_stage(stop, spec),),
        ok6(screen=spec.room_id, spec=spec, **kw),
    )


def settle_fight(through, stop, settle_factory, settle_name, spec, **kw) -> SpineHop:
    def stages():
        return (
            (settle_name, settle_factory(), SETTLE_19_MAX_FRAMES),
            fight_stage(stop, spec),
        )

    return SpineHop(through, stop, stages, ok6(screen=spec.room_id, spec=spec, **kw))


def _entry_ok(env):
    def ok(snap, **_):
        return (
            play_ready(
                snap,
                level=LEVEL6,
                screen=LEVEL6_ENTRY_ROOM,
                tf_bit=TF_BIT_L5,
                item="raft",
            )
            and snap.ladder > 0
            and int(read_u8(env.get_ram(), ADDR_WHISTLE)) >= 1
        )

    return ok


def _entry_stages():
    return (
        (
            "settle_l5_tf",
            PostL5TriforceSettleController(),
            POST_L5_SETTLE_MAX_FRAMES,
        ),
        (
            "enter_level6",
            make_post_l5_level6_controller(),
            POST_L5_PATH_MAX_FRAMES,
        ),
    )


def _east_key_stages():
    right = Level6EntryRightController()
    fight = make_east_key_controller()
    return (
        ("level6_right_0x7a", right, right.max_frames),
        ("level6_east_key_0x7a", fight, ROOM_7A_SPEC.max_frames),
    )


def _west_stages():
    back = Level6Return79Controller()
    door = Level6WestKeyDoorController()
    fight = make_west_wizzrobe_controller()
    return (
        ("level6_return_0x79", back, back.max_frames),
        ("level6_west_key_0x78", door, door.max_frames),
        ("level6_west_clear_0x78", fight, ROOM_78_SPEC.max_frames),
    )


def l6_prefix(env) -> tuple[SpineHop, ...]:
    tf5 = dict(tf_bit=TF_BIT_L5)
    tf1f = dict(tf_eq=0x1F)
    return (
        SpineHop(
            "level6-entry", "level6_entry_0x79", _entry_stages, _entry_ok(env)
        ),
        SpineHop(
            "level6-east-key",
            "level6_east_key_0x7a",
            _east_key_stages,
            ok6(screen=LEVEL6_EAST_KEY_ROOM, spec=ROOM_7A_SPEC, keys_cmp="gt", **tf5),
            capture_keys=True,
        ),
        SpineHop(
            "level6-west",
            "level6_west_0x78",
            _west_stages,
            ok6(screen=ROOM_78_SPEC.room_id, spec=ROOM_78_SPEC, **tf5),
        ),
        one_hop(
            "level6-compass",
            "level6_compass_0x68",
            Level6North68Controller,
            ok6(screen=LEVEL6_COMPASS_ROOM, **tf5),
            name="level6_north_0x68",
        ),
        fight_hop(
            "level6-clear68",
            "level6_clear_0x68",
            ROOM_68_SPEC,
            compass_bit=LEVEL6_COMPASS_BIT,
            **tf5,
        ),
        one_hop(
            "level6-keese",
            "level6_keese_0x58",
            make_north_58_controller,
            ok6(screen=LEVEL6_KEESE_ROOM, **tf5),
            name="level6_north_0x58",
        ),
        fight_hop("level6-clear58", "level6_clear_0x58", ROOM_58_SPEC, **tf5),
        one_hop(
            "level6-room48",
            "level6_room_0x48",
            make_north_48_controller,
            ok6(screen=LEVEL6_TRAPS_ROOM, **tf5),
            name="level6_north_0x48",
        ),
        one_hop(
            "level6-room38",
            "level6_room_0x38",
            make_north_38_controller,
            ok6(screen=LEVEL6_WIZZROBE_38_ROOM, **tf5),
            name="level6_north_0x38",
        ),
        fight_hop("level6-clear38", "level6_clear_0x38", ROOM_38_SPEC, **tf5),
        one_hop(
            "level6-room28",
            "level6_room_0x28",
            make_north_28_controller,
            ok6(screen=LEVEL6_WIZZROBE_28_ROOM, **tf5),
            name="level6_north_0x28",
        ),
        fight_hop("level6-clear28", "level6_clear_0x28", ROOM_28_SPEC, **tf5),
        one_hop(
            "level6-room18",
            "level6_room_0x18",
            make_north_18_controller,
            ok6(screen=LEVEL6_GLEEOK_ROOM, **tf5),
            name="level6_north_0x18",
        ),
        SpineHop(
            "level6-settle18",
            "level6_settle_0x18",
            lambda: (
                (
                    "level6_settle_0x18",
                    make_settle_18_controller(),
                    SETTLE_18_MAX_FRAMES,
                ),
            ),
            ok6(screen=LEVEL6_GLEEOK_ROOM, **tf1f),
        ),
        one_hop(
            "level6-gleeok18",
            "level6_gleeok_0x18",
            make_gleeok_18_controller,
            _gleeok18_ok,
        ),
        one_hop(
            "level6-postgleeok18",
            "level6_postgleeok_0x18",
            make_postgleeok_18_controller,
            _postgleeok18_ok,
        ),
        one_hop(
            "level6-stairs18",
            "level6_stairs_0x18",
            make_stairs_18_controller,
            lambda snap, **_: stairs_or_play(snap, not_screen=LEVEL6_GLEEOK_ROOM),
            dedicated=True,
        ),
    )
