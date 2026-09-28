"""L6 Survival hop helpers + first-half rows (entry through the 0x09 key door)."""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_BRACELET_ARMOS, TF_BIT_L5
from zelda_i.dungeon.door_hop import DoorHopSpec, door_hop_stages, door_hop_success
from zelda_i.level6.dungeon import (
    ROOM_19_SPEC,
    ROOM_38_SPEC,
    ROOM_58_SPEC,
    ROOM_78_SPEC,
)
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, PauseSelectController
from zelda_i.overworld.gather_segments import (
    BombWallController,
    CaveExitController,
    HopWalkController,
    make_secret_rupee_controller,
)
from zelda_i.overworld.arrow_shop import (
    ARROW_SHOP_PRICE,
    SHOP_F3_SCREEN,
    arrow_restock_stages,
)
from zelda_i.overworld.cave_shop import (
    RED_POTION_PRICE,
    make_potion_restock_controller,
    restock_item,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.rollout import PolicyGuard
from zelda_i.overworld.magical_sword import magical_sword_stages
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_ENTRY_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_SETTLE_MAX_FRAMES,
    POST_L5_TO_LEVEL6_HOPS,
    Level6WestKeyDoorController,
    OverworldToLevel6Controller,
    PostL5TriforceSettleController,
)
from zelda_i.level6.path import (
    Level6North68Controller,
    make_bomb_east_28_controller,
    make_north_09_controller,
    make_north_19_controller,
    make_north_28_controller,
    make_north_38_controller,
    make_north_48_controller,
    make_north_58_controller,
)
from zelda_i.level6.room19 import SETTLE_19_MAX_FRAMES
from zelda_i.ram import (
    ADDR_WHISTLE,
    CAVE_MODE,
    PASSAGE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.spine.hops import LatchedPlan, SpineHop, fight_stage, gated_stages, play_ready, ready

__all__ = [
    "l6_prefix",
    "ok6",
    "one_hop",
    "door_row",
    "fight_hop",
    "settle_fight",
    "stairs_or_play",
    "rod_cellar_ok",
]


def ok6(**kw):
    return ready(level=LEVEL6, **kw)


def stairs_or_play(
    snap: ZeldaSnapshot, *, not_screen: int, require_prior_tf: bool = True, **_
) -> bool:
    if snap.level != LEVEL6:
        return False
    if require_prior_tf and snap.triforce != 0x1F:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    kw = dict(level=LEVEL6, not_screen=not_screen)
    if require_prior_tf:
        kw["tf_eq"] = 0x1F
    return play_ready(snap, **kw)


def rod_cellar_ok(snap: ZeldaSnapshot, *, require_prior_tf: bool = True, **_) -> bool:
    """Rod pickup leftover is cellar mode 9 room 0x75, not play mode 5."""
    if snap.level != LEVEL6 or not snap.rod:
        return False
    if require_prior_tf and snap.triforce != 0x1F:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    kw = dict(level=LEVEL6, rod=True)
    if require_prior_tf:
        kw["tf_eq"] = 0x1F
    return play_ready(snap, **kw)


def one_hop(through, stop, factory, success, *, dedicated=False, name=None):
    stage = name or stop

    def stages():
        ctl = factory()
        return ((stage, ctl, ctl.max_frames),)

    return SpineHop(through, stop, stages, success, dedicated=dedicated)


def door_row(
    through: str,
    spec: DoorHopSpec,
    *,
    dedicated: bool = False,
    success=None,
) -> SpineHop:
    pred = success if success is not None else (
        lambda snap, s=spec, **_: door_hop_success(s, snap)
    )
    return SpineHop(
        through,
        spec.spec_id,
        lambda s=spec: door_hop_stages(s),
        pred,
        dedicated=dedicated,
    )


def fight_hop(through, stop, spec, **kw) -> SpineHop:
    return SpineHop(
        through,
        stop,
        lambda: (fight_stage(stop, spec),),
        ok6(screen=spec.room_id, spec=spec, **kw),
    )


def settle_fight(
    through,
    stop,
    settle_factory,
    settle_name,
    spec,
    fight_factory=None,
    success=None,
    **kw,
) -> SpineHop:
    def stages():
        return (
            (settle_name, settle_factory(), SETTLE_19_MAX_FRAMES),
            fight_stage(stop, spec, factory=fight_factory),
        )

    pred = success if success is not None else ok6(
        screen=spec.room_id, spec=spec, **kw
    )
    return SpineHop(through, stop, stages, pred)


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


# The walk's two buys: 0x25's arrows for Gohma, then 0x33's red potion.
L6_WALK_BUYS = ARROW_SHOP_PRICE + RED_POTION_PRICE

# The 80R arrows for Gohma: 0x25's shop is one screen east of the walk's
# 0x24, after 0x13's 30R. A wallet still short skips the stop.
L6_ARROW_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(SCREEN_BRACELET_ARMOS, "DOWN", align_x=160),
    ScreenHop(SHOP_F3_SCREEN, "RIGHT", align_y=141),
)


@dataclass
class _BackFromArrowShop(HopWalkController):
    """0x25 -> 0x24 after the arrow buy; a skipped buy left Link on 0x14."""

    hops: tuple[ScreenHop, ...] = (
        ScreenHop(SCREEN_BRACELET_ARMOS, "LEFT", align_y=141),
    )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and int(snap.screen) != SHOP_F3_SCREEN:
            self.frames += 1
            return self._finish("no_arrow_detour")
        return super().step(snap)


@dataclass
class _OpenPotion33(BombWallController):
    """Bomb the measured 0x33 rock on the L5-to-L6 walk and enter its shop."""

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and restock_item(snap) is None:
            self.frames += 1
            return self._finish("potion_restock_nothing_to_buy")
        return super().step(snap)

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return snap.level == 0 and snap.screen == 0x33 and snap.mode == CAVE_MODE


@dataclass
class _DrinkBeforePotion33:
    """Spend the carried blue dose when a red refill can replace it."""

    max_frames: int = 1200
    drink_at_whole_hearts: int = 0
    frames: int = 0
    success: bool = False
    enabled: bool = False

    def bind_env(self, env) -> None:
        snap = read_snapshot(env.get_ram())
        self.enabled = (
            snap.level == 0
            and snap.screen == 0x33
            and snap.potion == 1
            and not snap.health_is_full
            and snap.rupees >= 68
        )
        self.drink_at_whole_hearts = 10 if self.enabled else 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.success = not self.enabled or (snap.potion == 0 and snap.health_is_full)
        reason = "potion_33_healed" if self.success else "potion_33_drink"
        return FrameAction(nes_idle_action(), reason)

    def report(self) -> dict:
        return {"success": self.success, "frames": self.frames, "enabled": self.enabled}


@dataclass
class _StepIntoPotion33:
    """Leave the north scroll strip before trying to use B on 0x33."""

    max_frames: int = 300
    frames: int = 0
    success: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.success = (
            snap.level == 0
            and snap.screen == 0x33
            and snap.mode == PLAY_MODE
            and snap.link_y >= 93
        )
        action = nes_idle_action() if self.success else nes_action("DOWN")
        reason = "potion_33_inside" if self.success else "potion_33_south"
        return FrameAction(action, reason)

    def report(self) -> dict:
        return {"success": self.success, "frames": self.frames}


def _potion_33_stages(rem_hops: tuple[ScreenHop, ...]):
    to_33 = rem_hops[:3]
    walk = HopWalkController(hops=to_33, resume_on_screen=True, max_frames=5000)
    inside = _StepIntoPotion33()
    drink = _DrinkBeforePotion33()
    opener = _OpenPotion33(
        hops=(),
        screen=0x33,
        bomb_x=160,
        bomb_y=85,
        bomb_face="UP",
        door_x=160,
        door_y=80,
        b_item=B_SLOT_BOMBS,
        interior_x=None,
        interior_y=None,
        max_frames=12000,
    )
    buy = make_potion_restock_controller(hops=())
    buy.shop_screen = 0x33
    buy.cave_x = 160
    buy.cave_y = 77
    return (
        ("walk_potion_33", walk, walk.max_frames),
        ("inside_potion_33", inside, inside.max_frames),
        ("drink_blue_before_l6", drink, drink.max_frames),
        ("open_potion_33", opener, opener.max_frames),
        ("potion_restock_l6", buy, buy.max_frames),
        ("exit_potion_l6", CaveExitController(clear=0), 600),
    )


def _rupees_13_plan() -> LatchedPlan:
    """0x13's 30R rock only when the wallet cannot cover the L6 walk's buys.

    It cost 5.5 hearts in 917 frames on c9_from_coast, whose wallet reached
    L6 with 36R after both buys: the coast hearts' drops had already paid.
    """
    return LatchedPlan(
        "rupees_13",
        lambda snap: f"wallet_{int(snap.rupees)}" if int(snap.rupees) >= L6_WALK_BUYS else None,
    )


def _entry_stages():
    hops_to_14 = POST_L5_TO_LEVEL6_HOPS[:8]
    assert hops_to_14[-1].target == 0x14
    rem_hops = POST_L5_TO_LEVEL6_HOPS[8:]
    return (
        (
            "settle_l5_tf",
            PostL5TriforceSettleController(),
            POST_L5_SETTLE_MAX_FRAMES,
        ),
        (
            "walk_to_0x14",
            OverworldToLevel6Controller(
                hops=hops_to_14, max_frames=POST_L5_PATH_MAX_FRAMES
            ),
            POST_L5_PATH_MAX_FRAMES,
        ),
        *gated_stages(
            _rupees_13_plan(),
            (
                (
                    "walk_to_cave_13",
                    HopWalkController(
                        hops=(ScreenHop(0x13, "LEFT", y_band_lo=165, y_band_hi=189),),
                        max_frames=5000,
                    ),
                ),
                (
                    "select_bombs_for_13",
                    PauseSelectController(want=B_SLOT_BOMBS, name="bombs", max_frames=600),
                ),
                ("rupees_13", make_secret_rupee_controller(0x13, max_frames=5000)),
                ("exit_cave_13", CaveExitController(clear=0)),
                (
                    "return_14_from_13",
                    HopWalkController(
                        hops=(ScreenHop(0x14, "RIGHT", align_y=165),),
                        max_frames=5000,
                    ),
                ),
            ),
        ),
        *arrow_restock_stages(
            L6_ARROW_HOPS, "l5", screen=SHOP_F3_SCREEN, skip_short=True
        ),
        ("return_24_from_25", _BackFromArrowShop(max_frames=5000), 5000),
        *_potion_33_stages(rem_hops),
        # 12 containers (coast hearts + L5): the 0x21 grave's Magical Sword.
        *magical_sword_stages(),
        (
            "enter_level6",
            OverworldToLevel6Controller(
                hops=rem_hops,
                resume_on_screen=True,
                require_dungeon=True,
                max_frames=POST_L5_PATH_MAX_FRAMES,
            ),
            POST_L5_PATH_MAX_FRAMES,
        ),
    )


def _west_stages():
    """0x79 spawn -> key door -> 0x78. Its N door is open in the ROM door table:
    the compass hop walks through, and the 6.5-heart clear is skipped. 0x7a's
    key fight (2-6h) is skipped too: L6 still ends with the old route's 2 keys
    (0x58's drop and 0x29's island key pay the 0x29/0x19/0x2c doors)."""
    door = Level6WestKeyDoorController()
    return (("level6_west_key_0x78", door, door.max_frames),)


def l6_prefix(env, *, require_prior_tf: bool = True) -> tuple[SpineHop, ...]:
    tf5 = dict(tf_bit=TF_BIT_L5) if require_prior_tf else {}
    tf1f = dict(tf_eq=0x1F) if require_prior_tf else {}
    return (
        SpineHop(
            "level6-entry", "level6_entry_0x79", _entry_stages, _entry_ok(env)
        ),
        SpineHop(
            "level6-west",
            "level6_west_0x78",
            _west_stages,
            ok6(screen=ROOM_78_SPEC.room_id, **tf5),
        ),
        one_hop(
            "level6-compass",
            "level6_compass_0x68",
            Level6North68Controller,
            ok6(screen=LEVEL6_COMPASS_ROOM, **tf5),
            name="level6_north_0x68",
        ),
        # 0x68's N door is open and its clear only pays the compass (538f on
        # clean_poweron_c12): walk through it on the ROM's next frames.
        one_hop(
            "level6-keese",
            "level6_keese_0x58",
            lambda: PolicyGuard(make_north_58_controller(), trigger_radius=64),
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
        # 0x28's N door is open and 0x18's E shutter waits on the Gleeok;
        # the east bomb wall reaches 0x19 through 0x29 with neither fight.
        one_hop(
            "level6-bomb28",
            "level6_bomb_east_0x28",
            make_bomb_east_28_controller,
            ok6(screen=LEVEL6_DARK_29_ROOM, **tf5),
        ),
        one_hop(
            "level6-room19",
            "level6_north_0x19",
            make_north_19_controller,
            ok6(screen=LEVEL6_MAP_ROOM, **tf1f),
        ),
        # Clear 0x19's west bank from the south mouth (0.9h mean, 6 offsets):
        # the post-Rod walk back to 0x29 met a live Like-Like and stood
        # engulfed for 4000 frames.
        fight_hop("level6-clear19", "level6_clear_0x19", ROOM_19_SPEC, **tf1f),
        one_hop(
            "level6-room09",
            "level6_north_0x09",
            make_north_09_controller,
            ok6(screen=LEVEL6_ROD_WIZZ_ROOM, **tf1f),
        ),
    )
