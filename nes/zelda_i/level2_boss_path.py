"""Level 2 Dodongo boss path — boom → 0x1e bomb-N → 0x0e → LEFT 0x0d TF.

Assisted-track path knowledge for rr-5dk / rr-n5i. Not Clean STATUS.

Live path (2026-08-07)::

    0x4f bomb-N → 0x3f → LEFT Moldorm 0x3e → UP ropes 0x2e clear → UP Goriya
    0x1e clear → **bomb-N @(120,101)** → boss **0x0e** (type 0x32 Dodongo)
    → 2× bomb-in-mouth → Heart → **LEFT** TF room **0x0d**
    → south-band waypoints → ADDR_TRIFORCE & 0x02

Reuses ``level2_puzzles`` bomb stands / ``PostBossTfPolicy`` and
``level2_dungeon`` room specs / post-boom bomb-north controller. Does not
grow ``level2_dungeon.py``. Combat/TF live in ``level2_boss_combat`` /
``level2_boss_tf``; this module is the public façade + path orchestration.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable

from retro_harness.nes import nes_idle_action
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    MOLDORM_OBJECT_TYPE,
    ROPE_OBJECT_TYPE,
)
from zelda_i.level2_boss_combat import (
    ADDR_SELECTED_ITEM,
    B_ITEM_BOMB,
    BOMB_1E_MAX_FRAMES,
    BOMB_STAND_1E,
    BOMB_WALL_1E,
    BombNorth1EPhase,
    DODONGO_FIGHT_MAX_FRAMES,
    DODONGO_TYPE,
    FACE_E,
    FACE_N,
    FACE_S,
    FACE_W,
    KEESE_TYPE,
    Level2BombNorth1EController,
    bomb_north_1e_wall,
    clear_types,
    ensure_bomb_selected,
    enter_left,
    enter_up,
    fight_dodongo,
    goto_action,
    idle,
    live_objects,
    make_bomb_north_1e_controller,
    mouth_target,
    poke_bombs,
    sample_snapshot,
    triforce_bit_02,
    wait_types,
)
from zelda_i.level2_boss_tf import (
    L2_TF_REACH_JSON,
    Level2PostBossTfController,
    PostBossTfPhase,
    TF_COLLECT_MAX_FRAMES,
    collect_and_tf,
    default_tf_waypoints,
    load_tf_policy,
    make_post_boss_tf_controller,
    policy_push,
    policy_waypoints,
)
from zelda_i.level2_dungeon import (
    BOMB_N_STAND,
    Level2PostBoomBombNorthController,
    PostBoomBombNorthPhase,
    ROOM_1E_SPEC,
    ROOM_2E_SPEC,
    ROOM_L2_GORIYA_BOMBS,
    ROOM_L2_NORTH_OF_4E,
    ROOM_L2_ROPES_UNLOCK,
    ROOM_L2_TRAPS_KEESE,
)
from zelda_i.level2_puzzles import (
    DOOR_DOWN,
    DOOR_LEFT,
    DOOR_RIGHT,
    DOOR_UP,
    LEVEL2_TRIFORCE_BIT,
    POST_BOSS_TF_POLICY,
    ROOM_L2_BOSS,
    ROOM_L2_TF,
    bomb_wall_for_room,
    bomb_wall_open_predicate,
    is_at_bomb_stand,
)
from zelda_i.ram import (
    PLAY_MODE,
    read_snapshot,
)

# ---------------------------------------------------------------------------
# Path / room constants
# ---------------------------------------------------------------------------

ROOM_4F_BOOM: int = 0x4F
ROOM_3F: int = ROOM_L2_TRAPS_KEESE  # 0x3F
ROOM_3E: int = ROOM_L2_NORTH_OF_4E  # 0x3E Moldorm
ROOM_2E: int = ROOM_L2_ROPES_UNLOCK  # 0x2E ropes
ROOM_1E: int = ROOM_L2_GORIYA_BOMBS  # 0x1E Goriya
ROOM_0E: int = ROOM_L2_BOSS  # 0x0E Dodongo
ROOM_TF: int = ROOM_L2_TF  # 0x0D WEST of boss
LEVEL2_TF_BIT: int = LEVEL2_TRIFORCE_BIT  # 0x02

# Ordered boom→boss room chain (exclusive of boom start / boss itself).
BOOM_TO_BOSS_ROOMS: tuple[int, ...] = (
    ROOM_3F,
    ROOM_3E,
    ROOM_2E,
    ROOM_1E,
    ROOM_0E,
)

# Full post-boom path including TF collect room.
BOSS_PATH_ROOMS: tuple[int, ...] = BOOM_TO_BOSS_ROOMS + (ROOM_TF,)


def is_boss_path_room(room: int) -> bool:
    """True if room is on boom→TF Dodongo path (incl. 0x0d / 0x0e)."""
    return room in BOSS_PATH_ROOMS or room == ROOM_4F_BOOM


def bomb_1e_open_predicate(*, from_room: int, to_room: int) -> bool:
    """True if transition matches catalog 0x1e bomb-N → Dodongo."""
    return bomb_wall_open_predicate(
        from_room=from_room,
        to_room=to_room,
        wall=BOMB_WALL_1E,
    )


class BossPathStart(Enum):
    """Where to begin the Dodongo path run."""

    BOOM = auto()  # Level2Boom / 0x4f
    BOSS = auto()  # Level2_0E / 0x0e
    TF_ROOM = auto()  # Level2_0D_PostBoss / 0x0d


def resolve_path_start(start_state: str, boot_screen: int) -> BossPathStart:
    """Map start state / boot screen to path entry point."""
    if start_state == "Level2_0D_PostBoss" or boot_screen == ROOM_TF:
        return BossPathStart.TF_ROOM
    if start_state == "Level2_0E" or boot_screen == ROOM_0E:
        return BossPathStart.BOSS
    return BossPathStart.BOOM


def run_boom_to_1e(
    env: Any,
    assist: Any | None = None,
    *,
    timeline: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """Level2Boom path through 0x3f→0x3e→0x2e→0x1e Goriya clear.

    Returns (ok, fail_reason). On success Link is on cleared 0x1e ready
    for bomb-N. Does not bomb into boss.
    """
    tl = timeline if timeline is not None else []

    ctrl = Level2PostBoomBombNorthController()
    for f in range(ctrl.max_frames):
        if assist is not None:
            assist.apply_env(env, frame=f)
        env.step(ctrl.step(read_snapshot(env.get_ram())).action)
        if ctrl.success or ctrl.phase is PostBoomBombNorthPhase.FAILED:
            break
    tl.append(sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="3f"))
    if read_snapshot(env.get_ram()).screen != ROOM_3F:
        return False, "no_3f"

    clear_types(env, (KEESE_TYPE,), max_frames=5000, min_n=1)
    if not enter_left(env, ROOM_3E):
        return False, "no_3e"
    tl.append(sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="3e"))

    cr = clear_types(env, (MOLDORM_OBJECT_TYPE,), max_frames=16000, min_n=1)
    tl.append({"event": "moldorm", **cr})
    for _ in range(250):
        s = read_snapshot(env.get_ram())
        act, _ = goto_action(s, 120, 141, 8)
        env.step(act)
    tl.append(
        sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="moldorm_key")
    )

    if not enter_up(env, ROOM_2E):
        return False, "no_2e"
    tl.append(sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="2e"))

    cr = clear_types(env, (ROPE_OBJECT_TYPE,), max_frames=18000, min_n=4)
    idle(env, 90)
    s = read_snapshot(env.get_ram())
    tl.append(
        {
            "event": "ropes",
            **cr,
            "doors": s.cur_opened_doors,
            "ropes_left": len(live_objects(s, frozenset({ROPE_OBJECT_TYPE}))),
            "xy": [s.link_x, s.link_y],
        }
    )
    if not (s.cur_opened_doors & DOOR_UP):
        cr2 = clear_types(env, (ROPE_OBJECT_TYPE,), max_frames=10000, min_n=1)
        idle(env, 60)
        s = read_snapshot(env.get_ram())
        tl.append(
            {
                "event": "ropes_mop2",
                **cr2,
                "doors": s.cur_opened_doors,
                "ropes_left": len(live_objects(s, frozenset({ROPE_OBJECT_TYPE}))),
            }
        )
    if not (read_snapshot(env.get_ram()).cur_opened_doors & DOOR_UP):
        return False, "no_2e_up_door"

    if not enter_up(env, ROOM_1E, budget=1600):
        return False, "no_1e"
    tl.append(sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="1e"))

    wait_types(env, (GORIYA_OBJECT_TYPE,), n=3, budget=200)
    c1 = GenericDungeonRoomController(ROOM_1E_SPEC)
    c1.phase = DungeonPhase.FIGHT
    f = 0
    for f in range(ROOM_1E_SPEC.max_frames):
        if assist is not None and f % 20 == 0:
            assist.apply_env(env, frame=6000 + f)
        env.step(c1.step(read_snapshot(env.get_ram())).action)
        if c1.success or c1.phase in (DungeonPhase.FAILED, DungeonPhase.DONE):
            break
    tl.append({"event": "goriya", **c1.report(), "frames": f + 1})
    idle(env, 50)
    tl.append(
        sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="1e_cleared")
    )
    return True, None


def run_boss_path(
    env: Any,
    *,
    start: BossPathStart = BossPathStart.BOOM,
    assist: Any | None = None,
    timeline: list[dict[str, Any]] | None = None,
    save_0e_checkpoint: Callable[[], None] | None = None,
    poke: bool = False,
) -> dict[str, Any]:
    """Run Dodongo path from ``start`` through TF bit 0x02.

    Returns dict with keys: ok, reason (on fail), fight, tf_report, timeline.
    ``poke`` defaults off for spine / library callers. Isolated recon CLIs
    still pass ``--poke-bombs`` → ``poke=True``.
    """
    tl: list[dict[str, Any]] = timeline if timeline is not None else []
    fight: dict[str, Any] = {"success": True, "skipped": True}

    if start is BossPathStart.TF_ROOM:
        tl.append(
            {
                "event": "skip_to_tf_room",
                "sc": f"0x{read_snapshot(env.get_ram()).screen:02x}",
            }
        )
        tf_report = collect_and_tf(env, assist, budget=TF_COLLECT_MAX_FRAMES)
        return {
            "ok": bool(tf_report.get("ok")),
            "reason": None if tf_report.get("ok") else "tf_fail",
            "fight": fight,
            "tf_report": tf_report,
            "timeline": tl,
        }

    if start is BossPathStart.BOSS:
        tl.append(
            {
                "event": "skip_to_boss",
                "sc": f"0x{read_snapshot(env.get_ram()).screen:02x}",
            }
        )
        for _ in range(90):
            if assist is not None:
                assist.apply_env(env, frame=8000)
            env.step(nes_idle_action())
        tl.append(
            sample_snapshot(
                read_snapshot(env.get_ram()), env.get_ram(), event="0e_settle"
            )
        )
        fight = fight_dodongo(
            env, assist, max_frames=DODONGO_FIGHT_MAX_FRAMES, poke=poke
        )
        tl.append(
            {
                "event": "dodongo_fight",
                **{k: v for k, v in fight.items() if k != "log"},
            }
        )
        tl.extend(fight.get("log") or [])
        if not fight.get("success"):
            return {
                "ok": False,
                "reason": "dodongo_alive",
                "fight": fight,
                "tf_report": {},
                "timeline": tl,
            }
        tf_report = collect_and_tf(env, assist, budget=TF_COLLECT_MAX_FRAMES)
        return {
            "ok": bool(tf_report.get("ok")),
            "reason": None if tf_report.get("ok") else "tf_fail",
            "fight": fight,
            "tf_report": tf_report,
            "timeline": tl,
        }

    # Full boom path
    ok, reason = run_boom_to_1e(env, assist, timeline=tl)
    if not ok:
        return {
            "ok": False,
            "reason": reason,
            "fight": fight,
            "tf_report": {},
            "timeline": tl,
        }

    bomb = bomb_north_1e_wall(env, dest=ROOM_0E)
    tl.append({"event": "bomb_1e_n", **bomb})
    if not bomb.get("ok"):
        return {
            "ok": False,
            "reason": "no_boss_bomb",
            "fight": fight,
            "tf_report": {},
            "timeline": tl,
        }
    tl.append(sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="0e"))

    if save_0e_checkpoint is not None:
        save_0e_checkpoint()

    for _ in range(90):
        if assist is not None:
            assist.apply_env(env, frame=8000)
        env.step(nes_idle_action())
    tl.append(
        sample_snapshot(read_snapshot(env.get_ram()), env.get_ram(), event="0e_settle")
    )

    fight = fight_dodongo(
        env, assist, max_frames=DODONGO_FIGHT_MAX_FRAMES, poke=poke
    )
    tl.append(
        {"event": "dodongo_fight", **{k: v for k, v in fight.items() if k != "log"}}
    )
    tl.extend(fight.get("log") or [])

    tf_report = collect_and_tf(env, assist, budget=TF_COLLECT_MAX_FRAMES)
    return {
        "ok": bool(tf_report.get("ok")),
        "reason": None if tf_report.get("ok") else "tf_fail",
        "fight": fight,
        "tf_report": tf_report,
        "timeline": tl,
    }


__all__ = [
    "ROOM_4F_BOOM",
    "ROOM_3F",
    "ROOM_3E",
    "ROOM_2E",
    "ROOM_1E",
    "ROOM_0E",
    "ROOM_TF",
    "LEVEL2_TF_BIT",
    "DODONGO_TYPE",
    "KEESE_TYPE",
    "BOMB_WALL_1E",
    "BOMB_STAND_1E",
    "BOOM_TO_BOSS_ROOMS",
    "BOSS_PATH_ROOMS",
    "L2_TF_REACH_JSON",
    "FACE_E",
    "FACE_W",
    "FACE_S",
    "FACE_N",
    "ADDR_SELECTED_ITEM",
    "B_ITEM_BOMB",
    "BOMB_1E_MAX_FRAMES",
    "DODONGO_FIGHT_MAX_FRAMES",
    "TF_COLLECT_MAX_FRAMES",
    "DOOR_LEFT",
    "DOOR_RIGHT",
    "DOOR_DOWN",
    "DOOR_UP",
    "is_boss_path_room",
    "bomb_1e_open_predicate",
    "triforce_bit_02",
    "default_tf_waypoints",
    "load_tf_policy",
    "policy_waypoints",
    "policy_push",
    "mouth_target",
    "goto_action",
    "idle",
    "ensure_bomb_selected",
    "poke_bombs",
    "live_objects",
    "sample_snapshot",
    "enter_up",
    "enter_left",
    "wait_types",
    "clear_types",
    "BombNorth1EPhase",
    "Level2BombNorth1EController",
    "make_bomb_north_1e_controller",
    "bomb_north_1e_wall",
    "PostBossTfPhase",
    "Level2PostBossTfController",
    "make_post_boss_tf_controller",
    "fight_dodongo",
    "collect_and_tf",
    "BossPathStart",
    "resolve_path_start",
    "run_boom_to_1e",
    "run_boss_path",
    # re-exports used by runners
    "Level2PostBoomBombNorthController",
    "PostBoomBombNorthPhase",
    "ROOM_1E_SPEC",
    "ROOM_2E_SPEC",
    "BOMB_N_STAND",
    "POST_BOSS_TF_POLICY",
    "bomb_wall_for_room",
    "is_at_bomb_stand",
]
