"""Continuous Survival spine: one emulator session from power-on.

No mid-run state loads, no seam cards, no clip concat. The tape is whatever
this session actually walked. Stop at the first failed stage.

Clean M5 stays on ``run_level1_complete`` without ``--infinite-life``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zelda_i.chain import (
    ControllerStageResult,
    run_controller_stage,
    run_natural_to_milestone,
)
from zelda_i.level1_bow import level1_bow_stages, level1_bow_success
from zelda_i.level1_bow_cellar import (
    level1_bow_cellar_stages,
    level1_bow_cellar_success,
)
from zelda_i.level1_bow_pickup import (
    level1_bow_pickup_stages,
    level1_bow_pickup_success,
    level1_survival_tf_stages,
)
from zelda_i.level1_finish import LEVEL1_TRIFORCE_BIT
from zelda_i.level2_overworld import (
    SEGMENT_MAX_FRAMES as L2_NAV_MAX_FRAMES,
    SETTLE_MAX_FRAMES,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
)
from zelda_i.dungeon_ops import apply_owned_inventory
from zelda_i.level2_bombs import spine_bomb_report
from zelda_i.level2_spine import level2_boom_success, level2_to_boom_stages
from zelda_i.level2_tf_spine import (
    SPINE_TF_BOMB_POKE,
    SPINE_TF_KEY_POKE,
    level2_tf_stages,
    level2_through_success,
)
from zelda_i.level3_spine import l3_hops
from zelda_i.level3_bomb_budget import L3_BOMB_WALL_SPEND
from zelda_i.level3_boss_path import BOSS_PATH_MAX_FRAMES, Level3BossPathController
from zelda_i.level3_dungeon import LEVEL3_TRIFORCE_BIT
from zelda_i.level4_spine import L4_STOPS, continue_level4_spine
from zelda_i.level5_spine import (
    L5_STOPS,
    L5_THROUGH,
    continue_level5_spine,
    validate_l5_endpoint,
)
from zelda_i.level6_spine import L6_STOPS, L6_THROUGH, continue_level6_spine
from zelda_i.menus import BOOT_FILE_SLOT, BOOT_QUEST
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.spine_hops import SpineHop, attach_hops

BOOT_POLICY = {
    "file_slot": BOOT_FILE_SLOT,
    "quest": BOOT_QUEST,
    "playthrough": "first",
    "file_menu_select": False,
}

_L4_THROUGH = tuple(k for k in L4_STOPS if k != "level4") + ("level4",)
SPINE_THROUGH: tuple[str, ...] = (
    "level1",
    "level1-bow",
    "level1-bow-cellar",
    "level1-bow-pickup",
    "level2-entry",
    "level2",
    "level3",
) + _L4_THROUGH + L5_THROUGH + L6_THROUGH

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

# Bow KEY-LEFT spends the 0x23 key. 0x43 E still needs one. Restore the
# spent count (ASSIST_CONTRACT). Natural extra is L1 0x72 west of entrance.
SPINE_L1_KEY_POKE = 1
SPINE_L1_KEY_RETOPUP: frozenset[str] = frozenset({"backtrack44"})

_BOW_HOPS = (
    SpineHop(
        "level1-bow",
        "level1_bow_0x22",
        level1_bow_stages,
        level1_bow_success,
        dedicated=True,
    ),
    SpineHop(
        "level1-bow-cellar",
        "level1_bow_cellar",
        level1_bow_cellar_stages,
        level1_bow_cellar_success,
        dedicated=True,
    ),
    SpineHop(
        "level1-bow-pickup",
        "level1_bow_pickup",
        level1_bow_pickup_stages,
        level1_bow_pickup_success,
        dedicated=True,
    ),
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
        "map": snap.map,
        "rod": int(getattr(snap, "rod", 0)),
        "bow": int(getattr(snap, "bow", 0)),
        "arrows": int(getattr(snap, "arrows", 0)),
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
    position_assist: dict[str, Any] | None = None

    def _position_assist_from_stages(self) -> dict[str, Any] | None:
        """Prefer an explicit field; else take it from a stage controller report."""
        if self.position_assist is not None:
            return self.position_assist
        for stage in self.stages:
            reporter = getattr(stage.controller, "report", None)
            if not callable(reporter):
                continue
            nested = reporter()
            extra = nested.get("position_assist") if isinstance(nested, dict) else None
            if extra:
                return extra
        return None

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
            "position_assist": self._position_assist_from_stages(),
            "poke_bombs": (
                (self.inventory_assist or {}).get("poke_bombs") or False
            ),
            "poke_keys": (self.inventory_assist or {}).get("poke_keys") or False,
            "stop": {
                "level1": "level1_triforce",
                "level1-bow": "level1_bow_0x22",
                "level1-bow-cellar": "level1_bow_cellar",
                "level1-bow-pickup": "level1_bow_pickup",
                "level2-entry": "level2_entry",
                "level2": "level2_triforce_0x02",
                "level3": "level3_triforce_0x04",
                **L4_STOPS,
                **L5_STOPS,
                **L6_STOPS,
            }.get(self.through),
            "stages": [stage.report() for stage in self.stages],
        }


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


def topup_owned_keys(env, run: SpineRun, *, keys: int = SPINE_L1_KEY_POKE) -> None:
    """Restore the key spent on 0x23 W. Survival only. No bomb write."""
    extra = apply_owned_inventory(env, keys=keys, select_bomb=False)
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
    key_retopup: frozenset[str] = frozenset(),
    update_bombs: bool = False,
) -> bool:
    """Run named controller stages onto ``run``. False if a stage failed."""
    for name, controller, max_frames in stages:
        if name in retopup:
            topup_owned_inventory(env, run)
        if name in key_retopup:
            topup_owned_keys(env, run)
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
    through: str = "level1",
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

    hop_kw = dict(room_timer=room_timer, assist=assist, on_frame=on_frame)
    if through in ("level1-bow", "level1-bow-cellar", "level1-bow-pickup"):
        attach_hops(env, run, _BOW_HOPS, through=through, run_stages=_run_stages, **hop_kw)
        return run

    if not _run_stages(
        env,
        run,
        level1_survival_tf_stages(),
        key_retopup=SPINE_L1_KEY_RETOPUP,
        **hop_kw,
    ):
        return run

    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "triforce_bit"
        return run
    if through == "level1":
        return run

    if not _run_stages(env, run, level2_entry_stages(), **hop_kw):
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
    if through == "level2-entry":
        return run
    run.bombs = spine_bomb_report(snap.bombs, through="tf")
    # Survival shortcut until a farm pass: power-on L2 entry is bombs=0.
    # Documented in ASSIST_CONTRACT. Not Clean. No undiscovered items.
    topup_owned_inventory(env, run)

    if not _run_stages(
        env,
        run,
        level2_to_boom_stages(),
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
        **hop_kw,
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
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
        **hop_kw,
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

    def _set_l3_entry(env, run, snap):
        if run.success:
            run.l3_entry = spine_final_fields(snap)

    attach_hops(
        env,
        run,
        l3_hops(after_entry=_set_l3_entry),
        through=through,
        run_stages=_run_stages,
        **hop_kw,
    )
    if not run.success:
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

    continue_level4_spine(
        env,
        run,
        through=through,
        run_stages=_run_stages,
        topup_bombs=topup_owned_bombs,
        spine_fields=spine_final_fields,
        **hop_kw,
    )
    if not run.success or through in L4_STOPS:
        return run
    continue_level5_spine(
        env,
        run,
        through=through,
        run_stages=_run_stages,
        **hop_kw,
    )
    if not run.success or through in L5_THROUGH:
        return run
    continue_level6_spine(
        env,
        run,
        through=through,
        run_stages=_run_stages,
        **hop_kw,
    )
    return run
