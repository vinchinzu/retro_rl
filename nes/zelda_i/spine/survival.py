"""Continuous Survival spine: one emulator session from power-on.

No mid-run state loads, no seam cards, no clip concat. The tape is whatever
this session actually walked. Stop at the first failed stage.

Clean M5 stays on ``run_level1_complete`` without ``--infinite-life``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zelda_i.route.chain import (
    ControllerStageResult,
    boot_to_ready,
    run_controller_stage,
    run_natural_to_milestone,
)
from zelda_i.level1.bow import level1_bow_stages, level1_bow_success
from zelda_i.level1.bow_cellar import (
    level1_bow_cellar_stages,
    level1_bow_cellar_success,
)
from zelda_i.level1.arrow_shop import (
    level1_arrows_stages,
    level1_arrows_success,
)
from zelda_i.level1.bomb_shop_stage import (
    level1_bombs_stages,
    level1_bombs_success,
)
from zelda_i.level1.bow_pickup import (
    level1_bow_pickup_stages,
    level1_bow_pickup_success,
    level1_survival_tf_stages,
)
from zelda_i.overworld.gathering import (
    pre_l1_bomb_shop_success,
    pre_l1_stages,
)
from zelda_i.level1.finish import LEVEL1_TRIFORCE_BIT
from zelda_i.level2.overworld import (
    SEGMENT_MAX_FRAMES as L2_NAV_MAX_FRAMES,
    SETTLE_MAX_FRAMES,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
)
from zelda_i.dungeon.ops import apply_owned_inventory
from zelda_i.level2.bombs import spine_bomb_report
from zelda_i.level2.spine import level2_boom_success, level2_to_boom_stages
from zelda_i.level2.tf_spine import (
    SPINE_TF_BOMB_POKE,
    SPINE_TF_KEY_POKE,
    level2_tf_stages,
    level2_through_success,
)
from zelda_i.level3.spine import l3_hops
from zelda_i.level3.bomb_budget import L3_BOMB_WALL_SPEND
from zelda_i.level3.boss_path import BOSS_PATH_MAX_FRAMES, Level3BossPathController
from zelda_i.anchors import TF_BIT_L3 as LEVEL3_TRIFORCE_BIT
from zelda_i.level4.spine import L4_STOPS, continue_level4_spine
from zelda_i.level5.spine import (
    L5_STOPS,
    L5_THROUGH,
    continue_level5_spine,
    validate_l5_endpoint,
)
from zelda_i.level6.spine import L6_STOPS, L6_THROUGH, continue_level6_spine
from zelda_i.level7.spine import L7_STOPS, L7_THROUGH, continue_level7_spine
from zelda_i.level8.spine import L8_STOPS, L8_THROUGH, continue_level8_spine
from zelda_i.level9.spine import L9_STOPS, L9_THROUGH, continue_level9_spine
from zelda_i.menus import BOOT_FILE_SLOT, BOOT_QUEST
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    is_level1_ready,
    read_snapshot,
    read_u8,
)
from zelda_i.spine.hops import SpineHop, attach_hops

BOOT_POLICY = {
    "file_slot": BOOT_FILE_SLOT,
    "quest": BOOT_QUEST,
    "playthrough": "first",
    "file_menu_select": False,
}

# L4's own catalog is keyed by stop, not ordered by route; the Triforce stop
# is the last hop, not the first. ``SPINE_THROUGH`` is assembled from the
# ``SPINE_LEVELS`` rows at the bottom of this module.
_L4_THROUGH = tuple(k for k in L4_STOPS if k != "level4") + ("level4",)

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
    SpineHop(
        "level1-arrows",
        "level1_arrows",
        level1_arrows_stages,
        level1_arrows_success,
        dedicated=True,
    ),
    SpineHop(
        "level1-bombs",
        "level1_bombs",
        level1_bombs_stages,
        level1_bombs_success,
        dedicated=True,
    ),
)

# Gathering is before L1, not after. Dedicated like bow/arrows/bombs.
_L1_DEDICATED_HOPS = (
    SpineHop(
        "pre-l1",
        "pre_l1_shop_p7",
        pre_l1_stages,
        pre_l1_bomb_shop_success,
        dedicated=True,
    ),
    *_BOW_HOPS,
)


# L1-L3 have no ``levelN/spine.py`` catalog of their own; their stop names
# live here beside the rows that run them. Route order == insertion order.
_L1_DEDICATED_THROUGH: tuple[str, ...] = tuple(
    hop.through for hop in _L1_DEDICATED_HOPS
)
_L1_STOPS: dict[str, str] = {"level1": "level1_triforce"} | {
    hop.through: hop.stop for hop in _L1_DEDICATED_HOPS
}
_L2_STOPS: dict[str, str] = {
    "level2-entry": "level2_entry",
    "level2": "level2_triforce_0x02",
}
_L3_STOPS: dict[str, str] = {"level3": "level3_triforce_0x04"}
_L1_THROUGH: tuple[str, ...] = tuple(_L1_STOPS)
_L2_THROUGH: tuple[str, ...] = tuple(_L2_STOPS)
_L3_THROUGH: tuple[str, ...] = tuple(_L3_STOPS)


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


def spine_final_fields(snap: ZeldaSnapshot, ram: Any = None) -> dict[str, Any]:
    """End-of-run snapshot. Includes bombs so the farm bead can measure inventory.

    Pass ``ram`` to also capture the B-slot / Whistle / Food / Candle bytes and
    the container count — the fields an ``OverworldHandoff`` leave packet needs
    that ``ZeldaSnapshot`` does not carry.
    """
    fields = {
        "mode": snap.mode,
        "level": snap.level,
        "room": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "rupees": int(getattr(snap, "rupees", 0)),
        "health": snap.health,
        "triforce": snap.triforce,
        "map": snap.map,
        "rod": int(getattr(snap, "rod", 0)),
        "bow": int(getattr(snap, "bow", 0)),
        "arrows": int(getattr(snap, "arrows", 0)),
    }
    if ram is not None:
        fields.update(
            {
                "selected_item": read_u8(ram, ADDR_SELECTED_ITEM),
                "whistle": read_u8(ram, ADDR_WHISTLE),
                "food": read_u8(ram, ADDR_FOOD),
                "candle": read_u8(ram, ADDR_CANDLE),
                "heart_containers": snap.heart_containers,
                "health_full": bool(snap.health_is_full),
            }
        )
    return fields


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
    set_state_count: int | None = None
    allow_pokes: bool = True

    def apply_state_audit(self, count: int) -> None:
        """Record measured post-reset ``env.em.set_state`` calls. Fail if any."""
        self.set_state_count = int(count)
        if self.set_state_count:
            self.success = False
            if self.failed_stage is None:
                self.failed_stage = "mid_run_state_load"

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
            "set_state_count": self.set_state_count,
            "mid_run_state_load": (
                None
                if self.set_state_count is None
                else bool(self.set_state_count)
            ),
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
            "stop": SPINE_STOPS.get(self.through),
            "stages": [stage.report() for stage in self.stages],
        }


def _pokes_allowed(run: SpineRun) -> bool:
    """Survival inventory pokes stay on unless ``SpineRun.allow_pokes`` is off."""
    return bool(getattr(run, "allow_pokes", True))


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
    if not _pokes_allowed(run):
        return
    extra = apply_owned_inventory(
        env,
        bombs=SPINE_TF_BOMB_POKE,
        keys=SPINE_TF_KEY_POKE,
        select_bomb=True,
    )
    run.inventory_assist = merge_inventory_assist(run.inventory_assist, extra)


def topup_owned_bombs(env, run: SpineRun) -> None:
    """Documented Survival count refill at the L3 boss suffix; preserves keys."""
    if not _pokes_allowed(run):
        return
    extra = apply_owned_inventory(
        env, bombs=SPINE_TF_BOMB_POKE, select_bomb=True
    )
    run.inventory_assist = merge_inventory_assist(run.inventory_assist, extra)


def topup_owned_keys(env, run: SpineRun, *, keys: int = SPINE_L1_KEY_POKE) -> None:
    """Restore the key spent on 0x23 W. Survival only. No bomb write."""
    if not _pokes_allowed(run):
        return
    extra = apply_owned_inventory(env, keys=keys, select_bomb=False)
    run.inventory_assist = merge_inventory_assist(run.inventory_assist, extra)


# The L7 Bait buy costs 60R; the measured post-L6 leave carries 42R. Top the
# owned rupee count up to 60 before the Bait stage (ASSIST_CONTRACT shortcut;
# a natural OW farm is bead rr-doua-style follow-up). Not Clean. No item grant.
SPINE_L7_BAIT_RUPEES = 60


def topup_owned_rupees(
    env, run: SpineRun, *, rupees: int = SPINE_L7_BAIT_RUPEES
) -> None:
    """Documented Survival rupee count top-up for the L7 Bait buy. Not Clean."""
    if not _pokes_allowed(run):
        return
    extra = apply_owned_inventory(env, rupees=rupees, select_bomb=False)
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
    rupee_retopup: frozenset[str] = frozenset(),
    update_bombs: bool = False,
) -> bool:
    """Run named controller stages onto ``run``. False if a stage failed."""
    pokes = bool(getattr(run, "allow_pokes", True))
    for name, controller, max_frames in stages:
        if pokes:
            if name in retopup:
                topup_owned_inventory(env, run)
            if name in key_retopup:
                topup_owned_keys(env, run)
            if name in rupee_retopup:
                topup_owned_rupees(env, run)
        elif getattr(controller, "poke_arrows", None) is True:
            controller.poke_arrows = False
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


def _continue_level1_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """L1: dedicated gathering / Bow hops, else the natural Triforce run."""
    hop_kw = dict(room_timer=room_timer, assist=assist, on_frame=on_frame)
    if through in _L1_DEDICATED_THROUGH:
        if through in ("level1-arrows", "level1-bombs"):
            hop_kw["key_retopup"] = SPINE_L1_KEY_RETOPUP
        attach_hops(
            env,
            run,
            _L1_DEDICATED_HOPS,
            through=through,
            run_stages=run_stages,
            **hop_kw,
        )
        return
    if not run_stages(
        env,
        run,
        level1_survival_tf_stages(),
        key_retopup=SPINE_L1_KEY_RETOPUP,
        **hop_kw,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "triforce_bit"


def _continue_level2_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """L2: settle the L1 fanfare, walk the Moon door, boomerang, Triforce 0x02."""
    hop_kw = dict(room_timer=room_timer, assist=assist, on_frame=on_frame)
    if not run_stages(env, run, level2_entry_stages(), **hop_kw):
        return

    snap = read_snapshot(env.get_ram())
    if not (
        snap.level == 2
        and snap.mode == PLAY_MODE
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    ):
        run.success = False
        run.failed_stage = "level2_entry"
        return

    run.l2_entry = spine_final_fields(snap)
    if through == "level2-entry":
        return
    run.bombs = spine_bomb_report(snap.bombs, through="tf")
    # Survival shortcut until a farm pass: power-on L2 entry is bombs=0.
    # Documented in ASSIST_CONTRACT. Not Clean. No undiscovered items.
    topup_owned_inventory(env, run)

    if not run_stages(
        env,
        run,
        level2_to_boom_stages(),
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
        **hop_kw,
    ):
        return

    snap = read_snapshot(env.get_ram())
    if not level2_boom_success(snap):
        run.success = False
        run.failed_stage = "magic_boomerang"
        _record_bombs_out(env, run)
        return

    if not run_stages(
        env,
        run,
        level2_tf_stages(),
        retopup=SPINE_BOMB_RETOPUP,
        update_bombs=True,
        **hop_kw,
    ):
        return

    snap = read_snapshot(env.get_ram())
    run.success = level2_through_success(snap)
    if not run.success:
        run.failed_stage = "triforce_bit_02"
    _record_bombs_out(env, run)


def _continue_level3_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """L3: hop rows to the raft passage, then Raft -> Manhandla -> Triforce."""

    def _set_l3_entry(env, run, snap):
        if run.success:
            run.l3_entry = spine_final_fields(snap)

    attach_hops(
        env,
        run,
        l3_hops(after_entry=_set_l3_entry),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    if not run.success:
        return

    # Temporary Survival shortcut until rr-doua supplies the natural farm.
    # The live 0x5c Darknut clear can consume the carried eight bombs.
    topup_owned_bombs(env, run)
    if not _run_level3_boss_suffix(env, run, assist=assist):
        return
    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL3_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "level3_triforce_0x04"


@dataclass(frozen=True)
class SpineLevel:
    """One level of the Survival spine. The Composer dispatches these rows.

    ``through`` is the level's own ordered stop ids, ``stops`` their reported
    stage names, ``run`` the ``continue_*_spine`` that attaches the level's
    hops onto a live ``SpineRun``, and ``handoff`` the id a downstream target
    collapses to. ``extra`` and ``overrides_kw`` carry the only two per-level
    call shapes left: L4 wants the Survival top-up/report helpers, L8 takes the
    opt-in recon packet forwarded from ``run_survival_spine``.
    """

    level: int
    through: tuple[str, ...]
    stops: dict[str, str]
    run: Any
    handoff: str
    extra: dict[str, Any] = field(default_factory=dict)
    overrides_kw: str | None = None

    def target(self, through: str) -> str:
        """Remap a downstream target onto this level's handoff stop.

        Each ``continue_level{N}_spine`` only knows its own stop ids; a target
        past level N must become level N's handoff, or the level either raises
        (L7, L8: ``if through not in L{N}_THROUGH: raise``) or -- worse --
        silently runs every one of its own rows, including a completionist tail
        past the real handoff (L6: an unrecognized downstream ``through`` made
        ``attach_hops`` run the east3a/north39/inland29/west19/south18 tail
        *after* the measured OW return, stranding Link inside the dungeon
        instead of on the overworld -- rr-mzxn). A target that is one of this
        level's own ids passes through unchanged so the level's ``attach_hops``
        loop still stops there.
        """
        return through if through in self.through else self.handoff


_L4_EXTRA = {"topup_bombs": topup_owned_bombs, "spine_fields": spine_final_fields}

# The one dispatch table: level, own stop ids, stop names, continue-fn, handoff.
# A new level, stop or handoff is a row here; nothing in ``run_survival_spine``
# below knows a level number. L6's handoff is the measured post-fanfare OW
# return, NOT ``level6`` -- L7 continues from screen 0x22 and the L6
# completionist tail would strand Link inside the dungeon (rr-mzxn). L8
# continues from MEASURED_POST_L7_HANDOFF, L9 from MEASURED_POST_L8_HANDOFF
# (OW 0x6D); every natural L9 chapter is still a fail-closed marker.
SPINE_LEVELS: tuple[SpineLevel, ...] = (
    SpineLevel(1, _L1_THROUGH, _L1_STOPS, _continue_level1_spine, "level1"),
    SpineLevel(2, _L2_THROUGH, _L2_STOPS, _continue_level2_spine, "level2"),
    SpineLevel(3, _L3_THROUGH, _L3_STOPS, _continue_level3_spine, "level3"),
    SpineLevel(4, _L4_THROUGH, L4_STOPS, continue_level4_spine, "level4", _L4_EXTRA),
    SpineLevel(5, L5_THROUGH, L5_STOPS, continue_level5_spine, "level5"),
    SpineLevel(6, L6_THROUGH, L6_STOPS, continue_level6_spine, "level6-exit"),
    SpineLevel(7, L7_THROUGH, L7_STOPS, continue_level7_spine, "level7"),
    SpineLevel(
        8, L8_THROUGH, L8_STOPS, continue_level8_spine, "level8",
        overrides_kw="level8_overrides",
    ),
    SpineLevel(9, L9_THROUGH, L9_STOPS, continue_level9_spine, L9_THROUGH[-1]),
)

SPINE_THROUGH: tuple[str, ...] = tuple(
    stop for row in SPINE_LEVELS for stop in row.through
)
SPINE_STOPS: dict[str, str] = {
    stop: name for row in SPINE_LEVELS for stop, name in row.stops.items()
}


@dataclass
class _BootPrefix:
    """Menu boot only. Gathering owns sword; not the L1 clear53 prefix."""

    obs: Any
    boot_frames: int
    success: bool
    end_frame: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "milestone": "boot",
            "success": self.success,
            "boot_frames": self.boot_frames,
            "end_frame": self.end_frame,
        }


def _boot_only_prefix(env, *, room_timer=None, assist=None, on_frame=None):
    """Power-on to the first playable overworld frame. No sword, no L1."""
    obs, boot_frames = boot_to_ready(
        env,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        first_playthrough=True,
    )
    mean = None if obs is None else float(obs.mean())
    ready = obs is not None and is_level1_ready(env.get_ram(), obs_mean=mean)
    return _BootPrefix(
        obs=obs,
        boot_frames=boot_frames,
        success=bool(ready),
        end_frame=boot_frames,
    )


def run_survival_spine(
    env,
    obs: Any,
    *,
    assist: Any,
    on_frame=None,
    room_timer=None,
    through: str = "level1",
    level8_overrides: dict[str, Any] | None = None,
    allow_pokes: bool = True,
) -> SpineRun:
    """Power-on → requested dungeon stop. One env. No state reload.

    ``level8_overrides`` is the explicit opt-in path for the disclosed L8 recon
    packets (``zelda_i.level8.spine.LIVE_RECON_L8_OVERRIDES``); it is forwarded
    verbatim to ``continue_level8_spine``. The default run supplies none of it,
    so L8 keeps the unmeasured handoff and the unobserved topology.

    ``allow_pokes`` is the Survival inventory shortcut (default on). ``False``
    skips owned-count top-ups, the L7 Food fixture, and Gohma wooden arrows.

    ``through="pre-l1"`` strips ``assist`` even if the caller passed one.
    Survival refill hides the ``$0670`` chip that zeros ``$50``/``$627``, so
    the bomb walk is a Clean farm.
    """
    if through not in SPINE_THROUGH:
        raise ValueError(f"unknown spine stop {through!r}; wired: {SPINE_THROUGH}")

    if through == "pre-l1":
        assist = None
        prefix = _boot_only_prefix(
            env,
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
        )
        fail_name = "prefix_boot"
    else:
        prefix = run_natural_to_milestone(
            env,
            milestone="clear53",
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
            first_playthrough=True,
        )
        fail_name = "prefix_clear53"
    run = SpineRun(
        through=through,
        success=bool(prefix.success),
        boot_frames=prefix.boot_frames,
        prefix=prefix,
        end_frame=prefix.end_frame,
        obs=prefix.obs,
        failed_stage=None if prefix.success else fail_name,
        allow_pokes=allow_pokes,
    )
    if not run.success:
        return run

    hop_kw = dict(room_timer=room_timer, assist=assist, on_frame=on_frame)
    runtime = {"level8_overrides": level8_overrides or {}}
    for row in SPINE_LEVELS:
        extra = dict(row.extra)
        if row.overrides_kw is not None:
            extra.update(runtime.get(row.overrides_kw) or {})
        row.run(
            env,
            run,
            through=row.target(through),
            run_stages=_run_stages,
            **hop_kw,
            **extra,
        )
        if not run.success or through in row.through:
            return run
    return run
