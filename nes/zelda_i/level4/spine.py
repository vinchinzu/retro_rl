"""Survival-spine L4 hops through Gleeok enter; TF suffix stays a library call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.input_script import FrameAction
from zelda_i.dungeon.engine import DungeonPhase
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, B_SLOT_CANDLE, PauseSelectController
from zelda_i.level4.dungeon import (
    LEVEL4,
    LEVEL4_MAP_BIT,
    LEVEL4_TRIFORCE_BIT,
    ROOM_12_SPEC,
    ROOM_31_SPEC,
    ROOM_32_SPEC,
    ROOM_40_SPEC,
    ROOM_51_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_GLEEOK_13,
    ROOM_L4_KEY_01,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_MAP_21,
    ROOM_L4_MID_11,
    ROOM_L4_NORTH_30,
    ROOM_L4_STEPLADDER,
    ROOM_L4_VIRES_12,
    ROOM_L4_VIRES_50,
    ROOM_L4_WATER_NORTH_20,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4.bomb11 import level4_bomb11_stages
from zelda_i.level4.clear12 import level4_clear12_stages
from zelda_i.level4.exit60 import level4_exit60_stages
from zelda_i.level4.gleeok13 import attach_level4_tf_suffix, level4_gleeok13_stages
from zelda_i.level4.key01 import level4_key01_stages
from zelda_i.level4.keyup20 import level4_keyup20_stages
from zelda_i.level4.map21 import level4_map21_stages
from zelda_i.level4.mappick import level4_mappick_stages
from zelda_i.level4.maze_path import (
    make_maze_31_east_controller,
    make_maze_31_inland_controller,
    make_maze_31_leave_controller,
    make_north_40_controller,
    make_room_40_key_controller,
)
from zelda_i.level4.overworld import (
    LEVEL4_BOMB_WALLS,
    LEVEL4_HOPS_FROM_POST_L3,
    LEVEL4_HOPS_VIA_SHOP_E5,
    LEVEL4_ENTRY_ROOM,
    POST_L3_PATH_MAX_FRAMES,
    OverworldToLevel4Controller,
    RUPEES_71_BACK_HOPS,
    RUPEES_71_HOPS,
    RUPEES_71_PAY,
    RUPEES_71_SCREEN,
    RUPEES_51_BACK_HOPS,
    RUPEES_51_HOPS,
    RUPEES_51_PAY,
    RUPEES_51_SCREEN,
)
from zelda_i.overworld.bomb_shop import BOMB_SHOP_PRICE, bomb_restock_stages
from zelda_i.overworld.cave_shop import potion_restock_stages
from zelda_i.overworld.gather_segments import (
    BombWallController,
    CaveExitController,
    HopWalkController,
    make_secret_rupee_controller,
)
from zelda_i.overworld.settle import PostL3TriforceSettleController
from zelda_i.overworld.settle import POST_L3_SETTLE_MAX_FRAMES
from zelda_i.level4.path import (
    make_bomb_61_north_controller,
    make_entry_up_controller,
    make_left_50_controller,
    make_room_31_clear_controller,
    make_room_32_clear_controller,
    make_room_51_key_controller,
)
from zelda_i.level4.stepladder import (
    make_key_right_31_controller,
    make_stepladder_controller,
)
from zelda_i.level4.north30 import make_north_30_controller
from zelda_i.level4.west31 import level4_west31_stages
from zelda_i.ram import PASSAGE_MODE, ZeldaSnapshot, ow_secret_taken
from zelda_i.rollout import PolicyGuard
from zelda_i.spine.hops import SpineHop, attach_hops, ready

L4_STOPS: dict[str, str] = {
    "level4": "level4_triforce_0x08",
    "level4-entry": "level4_entry_0x71",
    "level4-key": "level4_natural_key_0x51",
    "level4-clear50": "level4_left_0x50",
    "level4-room40-key": "level4_natural_key_0x40",
    "level4-room30": "level4_enter_0x30",
    "level4-room31": "level4_enter_0x31",
    "level4-clear31": "level4_clear_0x31",
    "level4-room32": "level4_enter_0x32",
    "level4-clear32": "level4_clear_0x32",
    "level4-stepladder": "level4_stepladder_0x60",
    "level4-exit60": "level4_exit_0x60",
    "level4-west31": "level4_west_0x31",
    "level4-keyup20": "level4_key_up_0x20",
    "level4-room21": "level4_enter_0x21",
    "level4-map": "level4_map_pickup_0x21",
    "level4-bomb11": "level4_enter_0x11",
    "level4-key01": "level4_natural_key_0x01",
    "level4-clear12": "level4_clear_0x12",
    "level4-gleeok13": "level4_enter_0x13",
}

__all__ = [
    "L4_STOPS",
    "attach_level4_tf_suffix",
    "continue_level4_spine",
    "run_level4_entrance_tf",
]


def _as_fight(factory):
    ctl = factory()
    ctl.phase = DungeonPhase.FIGHT
    return ctl


def _first_key_stages():
    return (
        ("level4_entry_up_0x61", make_entry_up_controller(), 4000),
        (
            "level4_bomb_north_0x61",
            make_bomb_61_north_controller(clear_vires=True),
            20000,
        ),
        ("level4_key_0x51", _as_fight(make_room_51_key_controller), ROOM_51_SPEC.max_frames),
    )


# Vires dive ~2 px/f and split into Keese: roll only with one this near.
TRANSIT_GUARD_RADIUS = 64


def _transit(ctl: Any) -> PolicyGuard:
    """A walk across a room left uncleared, on the ROM's own next frames."""
    return PolicyGuard(ctl, trigger_radius=TRANSIT_GUARD_RADIUS)


def _room50_stages():
    # 0x50's N door is open (``pin_probe.py --doors``) and it holds no item:
    # its five-Vire clear (1,301f on clean_poweron_c12) bought nothing.
    return (("level4_left_0x50", make_left_50_controller(), 2500),)


def _room40_key_stages():
    return (
        ("level4_north_0x40", _transit(make_north_40_controller()), 10000),
        ("level4_key_0x40", make_room_40_key_controller(), 25000),
    )


def _key_right_31_stages():
    # 0x30's E door is a key door: its Vires need not die (494f on c12).
    return (
        (
            "level4_key_right_0x31",
            _transit(make_key_right_31_controller(clear_vires=False)),
            4000,
        ),
    )


def _clear_31_stages():
    return (
        ("level4_inland_0x31", make_maze_31_inland_controller(), 4000),
        (
            "level4_clear_0x31",
            _as_fight(make_room_31_clear_controller),
            ROOM_31_SPEC.max_frames,
        ),
        ("level4_leave_0x31", make_maze_31_leave_controller(), 4000),
    )


def _clear_32_stages():
    return (
        (
            "level4_clear_0x32",
            _as_fight(make_room_32_clear_controller),
            ROOM_32_SPEC.max_frames,
        ),
    )


def _stepladder_stages():
    ctl = make_stepladder_controller(clear_first=False)
    return (("level4_stepladder", ctl, ctl.max_frames),)


def _stepladder_ok(snap: ZeldaSnapshot, **_) -> bool:
    return snap.level == LEVEL4 and snap.ladder > 0 and (
        snap.screen == ROOM_L4_STEPLADDER or snap.mode == PASSAGE_MODE
    )


def _ok(**kw):
    return ready(level=LEVEL4, **kw)


class _Rupees71Skip:
    """First-frame guard for the cave stages; each stage remains spine-owned."""

    def _skip_reason(
        self, snap: ZeldaSnapshot, *, require_screen: bool = False
    ) -> str | None:
        if require_screen and snap.screen != RUPEES_71_SCREEN:
            return "off_71"
        if int(snap.bombs) < 1:
            return "no_bomb"
        if int(snap.rupees) + RUPEES_71_PAY > 255:
            return "wallet_full"
        if self._env is not None and ow_secret_taken(self._env.get_ram(), RUPEES_71_SCREEN):
            return "taken"
        return None


@dataclass
class _WalkToRupees71(_Rupees71Skip, HopWalkController):
    hops: tuple = RUPEES_71_HOPS
    resume_on_screen: bool = True

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap)):
            self.frames += 1
            return self._finish(f"skip_71_{why}")
        return super().step(snap)


@dataclass
class _SelectBombs71(_Rupees71Skip, PauseSelectController):
    want: int = B_SLOT_BOMBS
    name: str = "bombs"

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap, require_screen=True)):
            self.frames += 1
            return self._finish(f"skip_71_{why}")
        return super().step(snap)


@dataclass
class _TakeRupees71(_Rupees71Skip, BombWallController):
    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap, require_screen=True)):
            self.frames += 1
            return self._finish(f"skip_71_{why}")
        return super().step(snap)


@dataclass
class _ReturnFromRupees71(HopWalkController):
    hops: tuple = RUPEES_71_BACK_HOPS

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and snap.screen != RUPEES_71_SCREEN:
            self.frames += 1
            return self._finish("no_rupees_71_detour")
        return super().step(snap)


def rupees_71_stages() -> tuple[tuple[str, Any, int], ...]:
    """0x74 -> 0x71's 30R cave -> 0x73, with a first-frame skip when short."""
    return (
        ("walk_71", _WalkToRupees71(max_frames=6000), 6000),
        ("select_bombs_71", _SelectBombs71(), 600),
        (
            "rupees_71",
            make_secret_rupee_controller(
                RUPEES_71_SCREEN, controller_type=_TakeRupees71
            ),
            5000,
        ),
        ("exit_cave_71", CaveExitController(clear=16), 600),
        ("return_73", _ReturnFromRupees71(max_frames=6000), 6000),
    )


class _Rupees51Skip:
    """Skip an already opened tree or a payout the wallet cannot hold."""

    def _skip_reason(
        self, snap: ZeldaSnapshot, *, require_screen: bool = False
    ) -> str | None:
        if require_screen and snap.screen != RUPEES_51_SCREEN:
            return "off_51"
        if int(snap.candle) < 1:
            return "no_candle"
        if int(snap.rupees) + RUPEES_51_PAY > 255:
            return "wallet_full"
        if self._env is not None and ow_secret_taken(self._env.get_ram(), RUPEES_51_SCREEN):
            return "taken"
        return None


@dataclass
class _WalkToRupees51(_Rupees51Skip, HopWalkController):
    hops: tuple = RUPEES_51_HOPS
    resume_on_screen: bool = True

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap)):
            self.frames += 1
            return self._finish(f"skip_51_{why}")
        return super().step(snap)


@dataclass
class _SelectCandle51(_Rupees51Skip, PauseSelectController):
    want: int = B_SLOT_CANDLE
    name: str = "candle"

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap, require_screen=True)):
            self.frames += 1
            return self._finish(f"skip_51_{why}")
        return super().step(snap)


@dataclass
class _TakeRupees51(_Rupees51Skip, BombWallController):
    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and (why := self._skip_reason(snap, require_screen=True)):
            self.frames += 1
            return self._finish(f"skip_51_{why}")
        return super().step(snap)


@dataclass
class _ReturnFromRupees51(HopWalkController):
    hops: tuple = RUPEES_51_BACK_HOPS

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and snap.screen != RUPEES_51_SCREEN:
            self.frames += 1
            return self._finish("no_rupees_51_detour")
        return super().step(snap)


def rupees_51_stages() -> tuple[tuple[str, Any, int], ...]:
    """0x73 → 0x51's 10R tree → 0x63, before the L4 shop stops."""
    return (
        ("walk_51", _WalkToRupees51(max_frames=8000), 8000),
        ("select_candle_51", _SelectCandle51(), 600),
        (
            "rupees_51",
            make_secret_rupee_controller(
                RUPEES_51_SCREEN, controller_type=_TakeRupees51
            ),
            5000,
        ),
        ("exit_cave_51", CaveExitController(clear=0), 600),
        ("return_63", _ReturnFromRupees51(max_frames=8000), 8000),
    )


def l4_hops(*, spine_fields) -> tuple[SpineHop, ...]:
    def set_entry(env, run, snap):
        if run.success:
            run.l4_entry = spine_fields(snap)

    return (
        SpineHop(
            "level4-entry",
            "level4_entry_0x71",
            (
                (
                    "settle_l3_tf",
                    PostL3TriforceSettleController(),
                    POST_L3_SETTLE_MAX_FRAMES,
                ),
                # The L4 walk crosses 0x64's potion shop. Clean L4 bleeds
                # ~7h into the Gleeok (clean_poweron73), so the potion comes
                # before the arrows: keep only the L4 bomb pack.
                *rupees_71_stages(),
                *rupees_51_stages(),
                *potion_restock_stages(
                    LEVEL4_HOPS_FROM_POST_L3,
                    "l3",
                    reserve=BOMB_SHOP_PRICE,
                    resume_on_screen=True,
                ),
                # Then 0x44's bombs when short of L4's four walls (rr-doua).
                *bomb_restock_stages(
                    LEVEL4_HOPS_VIA_SHOP_E5, "l3", want=LEVEL4_BOMB_WALLS
                ),
                (
                    "enter_level4",
                    OverworldToLevel4Controller(
                        hops=LEVEL4_HOPS_VIA_SHOP_E5,
                        require_dungeon=True,
                        resume_on_screen=True,
                    ),
                    POST_L3_PATH_MAX_FRAMES,
                ),
            ),
            _ok(screen=LEVEL4_ENTRY_ROOM),
            after=set_entry,
        ),
        SpineHop(
            "level4-key",
            "level4_natural_key_0x51",
            _first_key_stages,
            _ok(screen=ROOM_L4_KEESE_KEY_51, spec=ROOM_51_SPEC, keys_cmp="gt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-clear50",
            "level4_left_0x50",
            _room50_stages,
            _ok(screen=ROOM_L4_VIRES_50),
        ),
        SpineHop(
            "level4-room40-key",
            "level4_natural_key_0x40",
            _room40_key_stages,
            _ok(screen=ROOM_L4_ZOLS_40, spec=ROOM_40_SPEC, keys_cmp="gt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-room30",
            "level4_enter_0x30",
            (("level4_north_0x30", make_north_30_controller(), 4000),),
            _ok(screen=ROOM_L4_NORTH_30),
        ),
        SpineHop(
            "level4-room31",
            "level4_enter_0x31",
            _key_right_31_stages,
            _ok(screen=ROOM_L4_EAST_31, keys_cmp="lt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-clear31",
            "level4_clear_0x31",
            _clear_31_stages,
            _ok(screen=ROOM_L4_EAST_31, spec=ROOM_31_SPEC),
        ),
        SpineHop(
            "level4-room32",
            "level4_enter_0x32",
            (("level4_east_0x32", make_maze_31_east_controller(), 4000),),
            _ok(screen=ROOM_L4_EAST_32),
        ),
        SpineHop(
            "level4-clear32",
            "level4_clear_0x32",
            _clear_32_stages,
            _ok(screen=ROOM_L4_EAST_32, spec=ROOM_32_SPEC),
        ),
        SpineHop(
            "level4-stepladder",
            "level4_stepladder_0x60",
            _stepladder_stages,
            _stepladder_ok,
        ),
        SpineHop(
            "level4-exit60",
            "level4_exit_0x60",
            level4_exit60_stages,
            _ok(screen=ROOM_L4_EAST_32, item="ladder"),
        ),
        SpineHop(
            "level4-west31",
            "level4_west_0x31",
            level4_west31_stages,
            _ok(screen=ROOM_L4_EAST_31, item="ladder"),
        ),
        SpineHop(
            "level4-keyup20",
            "level4_key_up_0x20",
            level4_keyup20_stages,
            _ok(screen=ROOM_L4_WATER_NORTH_20, item="ladder"),
        ),
        SpineHop(
            "level4-room21",
            "level4_enter_0x21",
            level4_map21_stages,
            _ok(screen=ROOM_L4_MAP_21, item="ladder"),
        ),
        SpineHop(
            "level4-map",
            "level4_map_pickup_0x21",
            level4_mappick_stages,
            _ok(screen=ROOM_L4_MAP_21, map_bit=LEVEL4_MAP_BIT),
        ),
        SpineHop(
            "level4-bomb11",
            "level4_enter_0x11",
            level4_bomb11_stages,
            _ok(screen=ROOM_L4_MID_11),
        ),
        SpineHop(
            "level4-key01",
            "level4_natural_key_0x01",
            level4_key01_stages,
            _ok(screen=ROOM_L4_KEY_01, keys_cmp="gt"),
            capture_keys=True,
        ),
        SpineHop(
            "level4-clear12",
            "level4_clear_0x12",
            level4_clear12_stages,
            _ok(screen=ROOM_L4_VIRES_12, spec=ROOM_12_SPEC),
        ),
        SpineHop(
            "level4-gleeok13",
            "level4_enter_0x13",
            level4_gleeok13_stages,
            _ok(screen=ROOM_L4_GLEEOK_13),
        ),
    )


def continue_level4_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    spine_fields,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L4 suffix after L3 TF. Mutates ``run``; caller returns it."""
    attach_hops(
        env,
        run,
        l4_hops(spine_fields=spine_fields),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    if not run.success or (through in L4_STOPS and through != "level4"):
        return
    if getattr(run, "skipping", False):
        return
    attach_level4_tf_suffix(env, run, assist=assist)


def run_level4_entrance_tf(
    env,
    *,
    assist=None,
    through: str = "level4",
    on_frame=None,
    room_timer=None,
):
    """Fixture-live L4 from play 0x71. Skip OW entry. No pokes.

    ``route_eligible=false``. Integrator promotes. Spine CLI has no
    ``--from-state``.
    """
    from zelda_i.ram import read_snapshot
    from zelda_i.route.chain import run_controller_stage
    from zelda_i.screen_glance import leftover_from_snapshot
    from zelda_i.spine.hops import attach_hops

    class _Run:
        def __init__(self) -> None:
            self.through = through
            self.success = True
            self.stages: list = []
            self.end_frame = 0
            self.failed_stage = None
            self.obs = getattr(env, "last_observation", None)
            self.allow_pokes = False
            self.l4_entry = None

    def run_stages(env, run, stages, **kw):
        del kw
        for name, controller, max_frames in stages:
            next_obs, stage = run_controller_stage(
                env,
                run.obs,
                name=name,
                controller=controller,
                max_frames=max_frames,
                assist=assist,
                on_frame=on_frame,
                room_timer=room_timer,
                frame_base=run.end_frame,
            )
            run.obs = next_obs
            run.stages.append(stage)
            run.end_frame = stage.end_frame
            if not stage.success:
                run.success = False
                run.failed_stage = name
                return False
        return True

    run = _Run()
    interior = tuple(
        hop
        for hop in l4_hops(spine_fields=lambda snap: {})
        if hop.through != "level4-entry"
    )
    attach_hops(
        env,
        run,
        interior,
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    if run.success and through == "level4":
        attach_level4_tf_suffix(env, run, assist=assist)
    snap = read_snapshot(env.get_ram())
    leftover = leftover_from_snapshot(snap)
    leftover["health"] = int(snap.health)
    leftover["deaths"] = 1 if int(snap.mode) == 17 else 0
    leftover["hearts_lo"] = int(snap.filled_hearts)
    leftover["hearts_hi"] = int(snap.heart_containers) - 1
    tf08 = bool(int(snap.triforce) & LEVEL4_TRIFORCE_BIT)
    reports = []
    for stage in run.stages:
        report = stage.report() if callable(getattr(stage, "report", None)) else {}
        reports.append(report)
    return {
        "ok": bool(run.success and tf08 and leftover["deaths"] == 0),
        "tf08": tf08,
        "failed_stage": run.failed_stage,
        "stages": reports,
        "leftover": leftover,
        "deaths": leftover["deaths"],
        "hearts_lo": leftover["hearts_lo"],
        "hearts_hi": leftover["hearts_hi"],
        "frames": run.end_frame,
        "route_eligible": False,
        "natural_entry": False,
        "intervention_class": "clean",
        "obs": run.obs,
    }
