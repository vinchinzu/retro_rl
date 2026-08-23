"""Continuous Survival spine: one emulator session from power-on.

No mid-run state loads, no seam cards, no clip concat. The tape is whatever
this session actually walked. Stop at the first failed stage.

Clean M5 stays on ``run_level1_complete`` without ``--infinite-life``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from zelda_i.chain import (
    ControllerStageResult,
    run_controller_stage,
    run_natural_to_milestone,
)
from zelda_i.level1_finish import LEVEL1_TRIFORCE_BIT, level1_triforce_stages
from zelda_i.level2_overworld import (
    SEGMENT_MAX_FRAMES as L2_NAV_MAX_FRAMES,
    SETTLE_MAX_FRAMES,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
)
from zelda_i.dungeon_ops import apply_owned_inventory
from zelda_i.dungeon import DungeonPhase
from zelda_i.level2_bombs import spine_bomb_report
from zelda_i.level2_spine import level2_boom_success, level2_to_boom_stages
from zelda_i.level2_tf_spine import (
    SPINE_TF_BOMB_POKE,
    SPINE_TF_KEY_POKE,
    level2_tf_stages,
    level2_through_success,
)
from zelda_i.level3_spine import (
    level3_compass_stages,
    level3_compass_success,
    level3_west_darknuts_stages,
    level3_west_darknuts_success,
    level3_south_darknuts_stages,
    level3_south_darknuts_success,
    level3_raft_stages,
    level3_raft_success,
    level3_dest_6b_stages,
    level3_dest_6b_success,
    level3_entry_stages,
    level3_entry_success,
)
from zelda_i.level3_bomb_budget import L3_BOMB_WALL_SPEND
from zelda_i.level3_boss_path import BOSS_PATH_MAX_FRAMES, Level3BossPathController
from zelda_i.level3_dungeon import LEVEL3_TRIFORCE_BIT
from zelda_i.level4_overworld import (
    POST_L3_PATH_MAX_FRAMES,
    POST_L3_SETTLE_MAX_FRAMES,
    OverworldToLevel4Controller,
    PostL3TriforceSettleController,
    level4_entry_stop,
)
from zelda_i.level4_dungeon import (
    ROOM_51_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_31_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_NORTH_30,
    ROOM_L4_VIRES_50,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4_maze_path import (
    make_maze_31_east_controller,
    make_maze_31_inland_controller,
    make_north_40_controller,
    make_room_40_key_controller,
)
from zelda_i.level4_path import (
    make_bomb_61_north_controller,
    make_entry_up_controller,
    make_left_50_controller,
    make_room_31_clear_controller,
    make_room_50_clear_controller,
    make_room_51_key_controller,
)
from zelda_i.level4_stepladder import (
    make_key_right_31_controller,
    make_north_30_controller,
    make_room_30_clear_controller,
)
from zelda_i.menus import BOOT_FILE_SLOT, BOOT_QUEST
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

BOOT_POLICY = {
    "file_slot": BOOT_FILE_SLOT,
    "quest": BOOT_QUEST,
    "playthrough": "first",
    "file_menu_select": False,
}

Through = Literal[
    "level1", "level2", "level3", "level4-entry", "level4-key",
    "level4-clear50",
    "level4-room40-key",
    "level4-room30",
    "level4-room31",
    "level4-clear31",
    "level4-room32",
]

SPINE_THROUGH: tuple[Through, ...] = (
    "level1",
    "level2",
    "level3",
    "level4-entry",
    "level4-key",
    "level4-clear50",
    "level4-room40-key",
    "level4-room30",
    "level4-room31",
    "level4-clear31",
    "level4-room32",
)

# Bomb-consuming stages. Survival tops up owned bomb/key counts before these
# (ASSIST_CONTRACT shortcut until a farm pass). Includes the 0x6f north wall
# that power-on L2 entry (bombs=0) otherwise fails in 1f.
SPINE_BOMB_RETOPUP: frozenset[str] = frozenset(
    {
        "bomb_north_6f",
        "bomb_north_5f",
        "bomb_north_4f",
        "bomb_north_1e",
        "fight_dodongo",
    }
)


def level2_entry_stages():
    """After L1 TF: idle the fanfare, then walk the Moon door and enter L2."""
    return (
        ("settle_l1_tf", PostTriforceSettleController(), SETTLE_MAX_FRAMES),
        (
            "enter_level2",
            OverworldToLevel2Controller(door_path=True, require_dungeon=True),
            L2_NAV_MAX_FRAMES,
        ),
    )


def level4_entry_stages():
    """After L3 TF: settle on OW 0x74, cross the Raft dock, and enter L4."""
    return (
        (
            "settle_l3_tf",
            PostL3TriforceSettleController(),
            POST_L3_SETTLE_MAX_FRAMES,
        ),
        (
            "enter_level4",
            OverworldToLevel4Controller(require_dungeon=True),
            POST_L3_PATH_MAX_FRAMES,
        ),
    )


def level4_first_key_stages():
    """L4 entry 0x71 → clear 0x61 → bomb north → natural key on 0x51."""
    key = make_room_51_key_controller()
    key.phase = DungeonPhase.FIGHT
    return (
        ("level4_entry_up_0x61", make_entry_up_controller(), 4000),
        (
            "level4_bomb_north_0x61",
            make_bomb_61_north_controller(clear_vires=True),
            20000,
        ),
        ("level4_key_0x51", key, ROOM_51_SPEC.max_frames),
    )


def level4_first_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Exact natural-key stop; RoomAllDead may reset before reward pickup."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_KEESE_KEY_51
        and snap.keys > keys_before
        and not ROOM_51_SPEC.live_enemies(snap)
    )


def level4_room40_key_stages():
    """Natural 0x51 key → west 0x50 → scripted north 0x40 → natural key."""
    return (
        ("level4_north_0x40", make_north_40_controller(), 10000),
        ("level4_key_0x40", make_room_40_key_controller(), 25000),
    )


def level4_room50_stages():
    clear_50 = make_room_50_clear_controller()
    clear_50.phase = DungeonPhase.FIGHT
    return (
        ("level4_left_0x50", make_left_50_controller(), 2500),
        ("level4_clear_0x50", clear_50, ROOM_50_SPEC.max_frames),
    )


def level4_room50_success(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_VIRES_50
        and not ROOM_50_SPEC.live_enemies(snap)
    )


def level4_room40_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_ZOLS_40
        and snap.keys > keys_before
        and not ROOM_40_SPEC.live_enemies(snap)
    )


def level4_north_30_stages():
    """Cleared 0x40 with the natural key → free UP into 0x30 play-ready."""
    return (
        ("level4_north_0x30", make_north_30_controller(), 4000),
    )


def level4_north_30_success(snap: ZeldaSnapshot) -> bool:
    """Exact enter-0x30 stop; do not require the Vire clear or KEY-RIGHT."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_NORTH_30
        and not snap.transitioning
    )


def level4_key_right_31_stages():
    """Enter-0x30 leftover → north-band Vire clear (ignore 0x2b) → KEY-RIGHT."""
    return (
        ("level4_clear_0x30", make_room_30_clear_controller(), 20000),
        (
            "level4_key_right_0x31",
            make_key_right_31_controller(clear_vires=False),
            4000,
        ),
    )


def level4_key_right_31_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Exact enter-0x31 stop; 0x31 Vires stay live. KEY-RIGHT consumes one key."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_31
        and not snap.transitioning
        and snap.keys < keys_before
    )


def level4_clear_31_stages():
    """West-door leftover (16,141) → alcove clip → maze Vire clear."""
    clear_31 = make_room_31_clear_controller()
    clear_31.phase = DungeonPhase.FIGHT
    return (
        ("level4_inland_0x31", make_maze_31_inland_controller(), 4000),
        ("level4_clear_0x31", clear_31, ROOM_31_SPEC.max_frames),
    )


def level4_clear_31_success(snap: ZeldaSnapshot) -> bool:
    """Exact 0x31 maze-clear stop; do not require the free-RIGHT into 0x32."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_31
        and not snap.transitioning
        and not ROOM_31_SPEC.live_enemies(snap)
    )


def level4_east_32_stages():
    """Cleared-0x31 leftover (112,141) → maze thread → free RIGHT into 0x32."""
    return (
        ("level4_east_0x32", make_maze_31_east_controller(), 4000),
    )


def level4_east_32_success(snap: ZeldaSnapshot) -> bool:
    """Exact enter-0x32 stop; Zol/LikeLike stay live."""
    return (
        snap.level == 4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_EAST_32
        and not snap.transitioning
    )


def spine_final_fields(snap: ZeldaSnapshot) -> dict[str, Any]:
    """End-of-run snapshot. Includes bombs so the farm bead can measure inventory."""
    return {
        "mode": snap.mode,
        "level": snap.level,
        "room": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "triforce": snap.triforce,
    }


@dataclass
class SpineRun:
    """One continuous power-on session."""

    through: str
    success: bool
    boot_frames: int
    stages: list[ControllerStageResult] = field(default_factory=list)
    prefix: Any = None
    end_frame: int = 0
    failed_stage: str | None = None
    obs: Any = None
    l2_entry: dict[str, Any] | None = None
    l3_entry: dict[str, Any] | None = None
    l4_entry: dict[str, Any] | None = None
    bombs: dict[str, Any] | None = None
    inventory_assist: dict[str, Any] | None = None

    def report(self) -> dict[str, Any]:
        return {
            "ok": self.success,
            "through": self.through,
            "continuous_emulator_session": True,
            "tape_kind": "continuous_survival_spine",
            "mid_run_state_load": False,
            "seamed": False,
            "status_claim": False,
            "boot_policy": dict(BOOT_POLICY),
            "boot_frames": self.boot_frames,
            "end_frame": self.end_frame,
            "failed_stage": self.failed_stage,
            "prefix": self.prefix.report() if self.prefix is not None else None,
            "l2_entry": self.l2_entry,
            "l3_entry": self.l3_entry,
            "l4_entry": self.l4_entry,
            "bombs": self.bombs,
            "inventory_assist": self.inventory_assist,
            "poke_bombs": (
                (self.inventory_assist or {}).get("poke_bombs") or False
            ),
            "poke_keys": (self.inventory_assist or {}).get("poke_keys") or False,
            "stop": {
                "level1": "level1_triforce",
                "level2": "level2_triforce_0x02",
                "level3": "level3_triforce_0x04",
                "level4-entry": "level4_entry_0x71",
                "level4-key": "level4_natural_key_0x51",
                "level4-clear50": "level4_clear_0x50",
                "level4-room40-key": "level4_natural_key_0x40",
                "level4-room30": "level4_enter_0x30",
                "level4-room31": "level4_enter_0x31",
                "level4-clear31": "level4_clear_0x31",
                "level4-room32": "level4_enter_0x32",
            }.get(self.through),
            "stages": [stage.report() for stage in self.stages],
        }


def validate_l5_endpoint(report: dict[str, object]) -> None:
    """Accept only a continuous L5 TF stop (no stitch manifest)."""
    if not report.get("continuous_emulator_session"):
        raise ValueError("L5 endpoint must be a continuous emulator session")
    if report.get("seamed") or report.get("tape_kind") == "state_seamed_viewing_compose":
        raise ValueError("seamed L5 tapes are not a spine endpoint")
    final = report.get("final")
    if not isinstance(final, dict):
        raise ValueError("Level 5 report has no final snapshot")
    if not report.get("ok"):
        raise ValueError("Level 5 report is not successful")
    if int(final.get("level", -1)) != 5 or int(final.get("screen", -1)) != 0x14:
        raise ValueError("Level 5 report does not end in the Triforce room (0x14)")
    if int(final.get("triforce", 0)) & 0x10 == 0:
        raise ValueError("Level 5 report does not have Triforce bit 0x10")
    assist = report.get("assist")
    if not isinstance(assist, dict):
        raise ValueError("Level 5 report is missing Survival telemetry")
    if int(assist.get("progression_writes", -1)) != 0:
        raise ValueError("Level 5 report has progression writes")
    if int(assist.get("capacity_writes", -1)) != 0:
        raise ValueError("Level 5 report has capacity writes")


def merge_inventory_assist(
    prev: dict[str, Any] | None, extra: dict[str, Any]
) -> dict[str, Any]:
    """Append Survival inventory-count writes; keep the latest poke amounts."""
    if prev is None:
        return extra
    merged = dict(prev)
    merged["writes"] = list(prev.get("writes") or []) + list(
        extra.get("writes") or []
    )
    merged["notes"] = list(prev.get("notes") or []) + list(extra.get("notes") or [])
    if extra.get("poke_bombs") is not None:
        merged["poke_bombs"] = extra["poke_bombs"]
    if extra.get("poke_keys") is not None:
        merged["poke_keys"] = extra["poke_keys"]
    return merged


def topup_owned_inventory(env, run: SpineRun) -> None:
    """Documented Survival bomb/key count top-up + B-slot bombs. Not Clean."""
    extra = apply_owned_inventory(
        env,
        bombs=SPINE_TF_BOMB_POKE,
        keys=SPINE_TF_KEY_POKE,
        select_bomb=True,
    )
    run.inventory_assist = merge_inventory_assist(run.inventory_assist, extra)


def topup_owned_bombs(env, run: SpineRun) -> None:
    """Documented Survival count refill at the L3 boss suffix; preserves keys."""
    extra = apply_owned_inventory(
        env, bombs=SPINE_TF_BOMB_POKE, select_bomb=True
    )
    run.inventory_assist = merge_inventory_assist(run.inventory_assist, extra)


def _record_bombs_out(env, run: SpineRun) -> None:
    end = read_snapshot(env.get_ram())
    run.bombs = spine_bomb_report(
        run.l2_entry.get("bombs") if run.l2_entry else None,
        through="tf",
        bombs_out=end.bombs,
    )


def _run_stages(
    env,
    run: SpineRun,
    stages,
    *,
    assist: Any,
    on_frame=None,
    room_timer=None,
    retopup: frozenset[str] = frozenset(),
    update_bombs: bool = False,
) -> bool:
    """Run named controller stages onto ``run``. False if a stage failed."""
    for name, controller, max_frames in stages:
        if name in retopup:
            topup_owned_inventory(env, run)
        obs, stage = run_controller_stage(
            env,
            run.obs,
            name=name,
            controller=controller,
            max_frames=max_frames,
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
            frame_base=run.end_frame,
        )
        run.obs = obs
        run.stages.append(stage)
        run.end_frame = stage.end_frame
        if not stage.success:
            run.success = False
            run.failed_stage = name
            if update_bombs:
                _record_bombs_out(env, run)
            return False
    return True


def _run_level3_boss_suffix(env, run: SpineRun, *, assist: Any) -> bool:
    """Run Raft → Manhandla → TF in the same session, without inventory writes."""
    entry = read_snapshot(env.get_ram())
    controller = Level3BossPathController(
        poke_bombs=None, tag="survival_spine_l3", continuous_mode=True
    )
    stage = ControllerStageResult(
        name="level3_boss_tf",
        controller=controller,
        max_frames=BOSS_PATH_MAX_FRAMES,
        frame_base=run.end_frame,
        end_frame=run.end_frame,
    )
    run.stages.append(stage)
    if entry.bombs < L3_BOMB_WALL_SPEND:
        controller._fail(
            f"bomb_budget_gate:{entry.bombs}<{L3_BOMB_WALL_SPEND}_verified_walls"
        )
        run.success = False
        run.failed_stage = stage.name
        return False

    total = [run.end_frame]
    path = controller.path_to_5d(env, assist, total)
    ok = bool(path.get("ok"))
    if ok:
        gate = controller.open_5d_up(env, assist, total)
        ok = bool(gate.get("ok"))
    if ok:
        fight = controller.fight_manhandla(env, assist, total, max_frames=16000)
        ok = bool(fight.get("tf04"))

    stage.frames = total[0] - stage.frame_base
    stage.end_frame = total[0]
    stage.success = ok
    run.end_frame = total[0]
    # Hybrid boss helpers step the env directly; refresh the observation used by
    # the runner's final screenshot without mutating route state.
    run.obs = getattr(env, "last_observation", run.obs)
    if not ok:
        run.success = False
        run.failed_stage = stage.name
    return ok


def run_survival_spine(
    env,
    obs: Any,
    *,
    assist: Any,
    on_frame=None,
    room_timer=None,
    through: Through = "level1",
) -> SpineRun:
    """Power-on → requested dungeon stop. One env. No state reload."""
    if through not in SPINE_THROUGH:
        raise ValueError(f"unknown spine stop {through!r}; wired: {SPINE_THROUGH}")
    if assist is None:
        raise ValueError("Survival spine requires UnlimitedHealthAssist")

    prefix = run_natural_to_milestone(
        env,
        milestone="clear53",
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        first_playthrough=True,
    )
    run = SpineRun(
        through=through,
        success=bool(prefix.success),
        boot_frames=prefix.boot_frames,
        prefix=prefix,
        end_frame=prefix.end_frame,
        obs=prefix.obs,
        failed_stage=None if prefix.success else "prefix_clear53",
    )
    if not run.success:
        return run

    if not _run_stages(
        env,
        run,
        level1_triforce_stages(natural_entry=True, survival=True),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "triforce_bit"
        return run
    if through == "level1":
        return run

    if not _run_stages(
        env,
        run,
        level2_entry_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    if not (
        snap.level == 2
        and snap.mode == PLAY_MODE
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    ):
        run.success = False
        run.failed_stage = "level2_entry"
        return run

    run.l2_entry = spine_final_fields(snap)
    run.bombs = spine_bomb_report(snap.bombs, through="tf")
    # Survival shortcut until a farm pass: power-on L2 entry is bombs=0.
    # Documented in ASSIST_CONTRACT. Not Clean. No undiscovered items.
    topup_owned_inventory(env, run)

    if not _run_stages(
        env,
        run,
        level2_to_boom_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    if not level2_boom_success(snap):
        run.success = False
        run.failed_stage = "magic_boomerang"
        _record_bombs_out(env, run)
        return run

    if not _run_stages(
        env,
        run,
        level2_tf_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level2_through_success(snap)
    if not run.success:
        run.failed_stage = "triforce_bit_02"
        _record_bombs_out(env, run)
        return run
    _record_bombs_out(env, run)
    if through == "level2":
        return run

    if not _run_stages(
        env,
        run,
        level3_entry_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    if not level3_entry_success(snap):
        run.success = False
        run.failed_stage = "enter_level3"
        return run
    run.l3_entry = spine_final_fields(snap)

    if not _run_stages(
        env,
        run,
        level3_dest_6b_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    if not level3_dest_6b_success(snap):
        run.success = False
        run.failed_stage = "north_chain"
        return run

    if not _run_stages(
        env,
        run,
        level3_compass_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level3_compass_success(snap)
    if not run.success:
        run.failed_stage = "compass_0x5a"
        return run

    if not _run_stages(
        env,
        run,
        level3_west_darknuts_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level3_west_darknuts_success(snap)
    if not run.success:
        run.failed_stage = "west_darknuts_0x59"
        return run

    if not _run_stages(
        env,
        run,
        level3_south_darknuts_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level3_south_darknuts_success(snap)
    if not run.success:
        run.failed_stage = "south_darknuts_0x69"
        return run

    if not _run_stages(
        env,
        run,
        level3_raft_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level3_raft_success(snap)
    if not run.success:
        run.failed_stage = "raft_0x0f"
        return run

    # Temporary Survival shortcut until rr-doua supplies the natural farm.
    # The live 0x5c Darknut clear can consume the carried eight bombs.
    topup_owned_bombs(env, run)
    if not _run_level3_boss_suffix(env, run, assist=assist):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL3_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "level3_triforce_0x04"
        return run
    if through == "level3":
        return run

    if not _run_stages(
        env,
        run,
        level4_entry_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = level4_entry_stop(snap)
    if not run.success:
        run.failed_stage = "level4_entry_0x71"
        return run
    run.l4_entry = spine_final_fields(snap)
    if through == "level4-entry":
        return run

    entry_up, bomb_wall, natural_key = level4_first_key_stages()
    if not _run_stages(
        env, run, (entry_up,), room_timer=room_timer, assist=assist, on_frame=on_frame
    ):
        return run
    # Operator-authorized Survival exception: known 0x61 bomb wall, while the
    # continuous predecessor arrives at L4 with bombs=0. Preserve keys/items.
    topup_owned_bombs(env, run)
    if not _run_stages(
        env,
        run,
        (bomb_wall, natural_key),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_first_key_success(
        snap, keys_before=int(run.l4_entry["keys"])
    )
    if not run.success:
        run.failed_stage = "level4_natural_key_0x51"
        return run
    if through == "level4-key":
        return run

    keys_before_40 = snap.keys
    if not _run_stages(
        env,
        run,
        level4_room50_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_room50_success(snap)
    if not run.success:
        run.failed_stage = "level4_clear_0x50"
        return run
    if through == "level4-clear50":
        return run

    if not _run_stages(
        env,
        run,
        level4_room40_key_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_room40_key_success(snap, keys_before=keys_before_40)
    if not run.success:
        run.failed_stage = "level4_natural_key_0x40"
        return run
    if through == "level4-room40-key":
        return run

    if not _run_stages(
        env,
        run,
        level4_north_30_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_north_30_success(snap)
    if not run.success:
        run.failed_stage = "level4_enter_0x30"
        return run
    if through == "level4-room30":
        return run

    keys_before_31 = snap.keys
    if not _run_stages(
        env,
        run,
        level4_key_right_31_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_key_right_31_success(snap, keys_before=keys_before_31)
    if not run.success:
        run.failed_stage = "level4_enter_0x31"
        return run
    if through == "level4-room31":
        return run

    if not _run_stages(
        env,
        run,
        level4_clear_31_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_clear_31_success(snap)
    if not run.success:
        run.failed_stage = "level4_clear_0x31"
        return run
    if through == "level4-clear31":
        return run

    if not _run_stages(
        env,
        run,
        level4_east_32_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return run
    snap = read_snapshot(env.get_ram())
    run.success = level4_east_32_success(snap)
    if not run.success:
        run.failed_stage = "level4_enter_0x32"
    return run
