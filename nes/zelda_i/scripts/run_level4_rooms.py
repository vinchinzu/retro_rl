"""Level 4 interior pure/assisted room segments from LIVE IDs only.

Segments (rr-5lu / rr-2ysf children)::

    entry_up     (rr-zchy)  Level4Entrance 0x71 empty → UP → 0x61
    clear_61     (rr-yr77)  clear 3× Vire 0x12 (+ split 0x1c) on 0x61
    bomb_61      (rr-h278)  BOMB_UP stand~(120,105) 0x61 → 0x51
    key_51       (rr-wqdu)  clear 8× Keese + key 0x19 on 0x51
    chain_to_key            entry→clear→bomb→key (path proof)
    clear_50     (rr-2ysf)  LEFT 0x51 → clear 5× Vire pocket (dead-end)
    key_right_62 (rr-2ysf)  0x61 KEY-RIGHT @y141 → 0x62 compass maze
    chain_to_62  (rr-2ysf)  FirstKey → DOWN 0x61 clear → KEY-RIGHT 0x62
    clear_62     (rr-2ysf)  clear 5× Vire on 0x62 (compass pickup residual)
    compass_62   (rr-9so0)  dark maze → ADDR_COMPASS bit 0x08 → return 0x61
    north_40     (rr-xc3x)  0x50 cleared → scripted N → 0x40 Zols+key
    key_40       (rr-q8eq)  clear 5× Zol 0x13 (+ gel 0x14) + key @~(120,117)
    north_30     (rr-q8eq)  0x40 cleared → free UP → 0x30 Vires
    clear_30     (rr-n1wn)  clear 3× Vire 0x12 on 0x30 (ignore invuln 0x2b)
    key_right_31 (rr-n1wn)  0x30 KEY-RIGHT @y141 → 0x31 (5× Vire)

Live graph only — no walkthrough room hardcodes beyond recon.

Examples::

    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment entry_up --trials 2
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_50 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_right_62 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment chain_to_62 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment compass_62 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment north_40 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_40 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment north_30 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_30 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_right_31 --trials 2 --save-state
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _NES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level4_dungeon import (
    BOMB_61_NORTH_STAND,
    LEVEL4_COMPASS_BIT,
    ROOM_30_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_L4_COMPASS_62,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_ZOLS_40,
    level4_compass_route_success,
    level4_room_30_cleared,
    level4_room_40_key_success,
    level4_room_40_ready,
    level4_room_50_cleared,
    level4_room_51_key_success,
    level4_room_51_ready,
    level4_room_61_cleared,
    level4_room_61_ready,
    level4_room_62_ready,
    make_bomb_61_north_controller,
    make_compass_62_controller,
    make_entry_up_controller,
    make_key_right_62_controller,
    make_left_50_controller,
    make_north_30_controller,
    make_north_40_controller,
    make_key_right_31_controller,
    make_room_30_clear_controller,
    make_room_40_key_controller,
    make_room_50_clear_controller,
    level4_room_30_ready,
    level4_room_31_ready,
    ROOM_L4_EAST_31,
    ROOM_L4_NORTH_30,
    make_room_51_key_controller,
    make_room_61_clear_controller,
    make_room_62_clear_controller,
    level4_room_62_cleared,
    ROOM_62_SPEC,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

SEGMENTS = (
    "entry_up",
    "clear_61",
    "bomb_61",
    "key_51",
    "chain_to_key",
    "clear_50",
    "key_right_62",
    "chain_to_62",
    "clear_62",
    "compass_62",
    "north_40",
    "key_40",
    "north_30",
    "clear_30",
    "key_right_31",
)

_BEAD = {
    "entry_up": "rr-zchy",
    "clear_61": "rr-yr77",
    "bomb_61": "rr-h278",
    "key_51": "rr-wqdu",
    "chain_to_key": "rr-5lu",
    "clear_50": "rr-2ysf",
    "key_right_62": "rr-2ysf",
    "chain_to_62": "rr-2ysf",
    "clear_62": "rr-2ysf",
    "compass_62": "rr-9so0",
    "north_40": "rr-xc3x",
    "key_40": "rr-q8eq",
    "north_30": "rr-q8eq",
    "clear_30": "rr-n1wn",
    "key_right_31": "rr-n1wn",
}

_DEFAULT_STATE = {
    "entry_up": "Level4Entrance",
    "clear_61": "Level4Entrance",
    "bomb_61": "Level4Room61Cleared",
    "key_51": "Level4Room51Cleared",  # may still have Keese; fallback handled
    "chain_to_key": "Level4Entrance",
    "clear_50": "Level4FirstKey",
    "key_right_62": "Level4FirstKey",
    "chain_to_62": "Level4FirstKey",
    "clear_62": "Level4Room62",
    "compass_62": "Level4Room62Cleared",
    "north_40": "Level4Compass",
    "key_40": "Level4Room40",
    "north_30": "Level4Room40Cleared",
    "clear_30": "Level4Room30",
    "key_right_31": "Level4Room30Cleared",
}

_CHECKPOINT = {
    "entry_up": "Level4Room61",
    "clear_61": "Level4Room61Cleared",
    "bomb_61": "Level4Room51",
    "key_51": "Level4FirstKey",
    "chain_to_key": "Level4FirstKey",
    "clear_50": "Level4Room50Cleared",
    "key_right_62": "Level4Room62",
    "chain_to_62": "Level4Room62",
    "clear_62": "Level4Room62Cleared",
    "compass_62": "Level4Compass",
    "north_40": "Level4Room40",
    "key_40": "Level4Room40Cleared",
    "north_30": "Level4Room30",
    "clear_30": "Level4Room30Cleared",
    "key_right_31": "Level4Room31",
}


def _snap_fields(snap) -> dict[str, Any]:
    return {
        "mode": snap.mode,
        "level": snap.level,
        "room": snap.screen,
        "room_hex": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "compass": snap.compass,
        "compass_l4": bool(snap.compass & LEVEL4_COMPASS_BIT),
        "room_item_id": snap.room_item_id,
        "room_all_dead": snap.room_all_dead,
        "cur_opened_doors": snap.cur_opened_doors,
    }


def _run_until(
    env,
    controller,
    *,
    assist,
    max_frames: int,
    done: Callable[[Any], bool],
    frame0: int = 0,
) -> tuple[object, int]:
    obs = None
    frame = frame0
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        action = controller.step(snap)
        obs, *_ = env.step(action.action)
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        if done(controller):
            break
        phase = getattr(controller, "phase", None)
        if phase is not None and getattr(phase, "name", "") in ("FAILED", "DONE"):
            break
    return obs, frame


def _done_success(controller) -> bool:
    return bool(getattr(controller, "success", False))


def _bfs_50_to_north(env, *, assist, frame0: int, hold: int = 6, long_up: int = 220):
    """Live BFS on 0x50 to a north-band cell that admits long-UP into 0x40."""
    from collections import deque

    from zelda_i.level4_dungeon import MAZE_50_HOLD, MAZE_50_LONG_UP, ROOM_L4_ZOLS_40

    hold = MAZE_50_HOLD
    long_up = MAZE_50_LONG_UP
    em = env.unwrapped.em
    dirs = ("UP", "DOWN", "LEFT", "RIGHT")

    def cell(x: int, y: int, q: int = 8) -> tuple[int, int]:
        return (x // q * q, y // q * q)

    def snap_ok_40(s) -> bool:
        return (
            s.level == 4
            and s.screen == ROOM_L4_ZOLS_40
            and s.mode == PLAY_MODE
            and not s.transitioning
        )

    start = read_snapshot(env.get_ram())
    if start.screen != ROOM_L4_VIRES_50:
        return None, {"error": f"not_on_50_0x{start.screen:02x}", "success": False}

    start_c = cell(start.link_x, start.link_y)
    cell_state = {start_c: em.get_state()}
    parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start_c: None}
    q: deque[tuple[int, int]] = deque([start_c])
    seen = {start_c}
    best_path = None
    expansions = 0

    while q and expansions < 4000 and best_path is None:
        cur = q.popleft()
        for d in dirs:
            expansions += 1
            em.set_state(cell_state[cur])
            for _ in range(hold):
                env.step(nes_action(d))
                if assist is not None:
                    assist.apply_env(env, frame=frame0)
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.transitioning:
                for _ in range(40):
                    env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=frame0)
                s = read_snapshot(env.get_ram())
            if s.level != 4 or s.screen != ROOM_L4_VIRES_50 or s.mode != PLAY_MODE:
                continue
            nc = cell(s.link_x, s.link_y)
            if nc in seen:
                continue
            seen.add(nc)
            cell_state[nc] = em.get_state()
            parent[nc] = (cur, d)
            q.append(nc)
            if nc[1] <= 80 and 96 <= nc[0] <= 144:
                em.set_state(cell_state[nc])
                for _ in range(long_up):
                    env.step(nes_action("UP"))
                    if assist is not None:
                        assist.apply_env(env, frame=frame0)
                    s2 = read_snapshot(env.get_ram())
                    if s2.mode != PLAY_MODE or s2.transitioning:
                        for _ in range(40):
                            env.step(nes_idle_action())
                            if assist is not None:
                                assist.apply_env(env, frame=frame0)
                        s2 = read_snapshot(env.get_ram())
                    if snap_ok_40(s2):
                        path: list[str] = []
                        p: tuple[int, int] | None = nc
                        while p is not None and parent[p] is not None:
                            prev, pd = parent[p]
                            path.append(pd)
                            p = prev
                        path.reverse()
                        best_path = path
                        break
                if best_path is not None:
                    break

    # Restore start pose for follower.
    em.set_state(cell_state[start_c])
    for _ in range(5):
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=frame0)

    meta = {
        "success": best_path is not None,
        "expansions": expansions,
        "n_cells": len(seen),
        "start": [start.link_x, start.link_y],
        "path_len": len(best_path) if best_path else 0,
        "hold": hold,
        "long_up": long_up,
        "segment": "level4_north_0x40_bfs",
    }
    return best_path, meta


def _follow_50_north_path(env, path: list[str], *, assist, frame0: int):
    """Execute BFS path tokens then long UP into 0x40."""
    from zelda_i.level4_dungeon import MAZE_50_HOLD, MAZE_50_LONG_UP, ROOM_L4_ZOLS_40

    frame = frame0
    obs = None
    for d in path:
        for _ in range(MAZE_50_HOLD):
            obs, *_ = env.step(nes_action(d))
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.transitioning:
                for _ in range(40):
                    obs, *_ = env.step(nes_idle_action())
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
    for _ in range(MAZE_50_LONG_UP + 80):
        obs, *_ = env.step(nes_action("UP"))
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE or s.transitioning:
            for _ in range(50):
                obs, *_ = env.step(nes_idle_action())
                frame += 1
                if assist is not None:
                    assist.apply_env(env, frame=frame)
            s = read_snapshot(env.get_ram())
        if (
            s.level == 4
            and s.screen == ROOM_L4_ZOLS_40
            and s.mode == PLAY_MODE
            and not s.transitioning
        ):
            for _ in range(20):
                obs, *_ = env.step(nes_idle_action())
                frame += 1
                if assist is not None:
                    assist.apply_env(env, frame=frame)
            return obs, frame, True
    return obs, frame, False


def run_once(
    *,
    segment: str,
    start_state: str,
    infinite_life: bool,
    save_checkpoint: bool,
    tag: str,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    bead = _BEAD[segment]
    controllers: dict[str, Any] = {}
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        entry = read_snapshot(env.get_ram())
        entry_fields = _snap_fields(entry)
        frame = 1
        ok = False
        error: str | None = None

        if segment == "entry_up":
            ctl = make_entry_up_controller()
            controllers["entry_up"] = ctl
            obs, frame = _run_until(
                env, ctl, assist=assist, max_frames=ctl.max_frames, done=_done_success, frame0=frame
            )
            ok = level4_room_61_ready(env.get_ram()) and ctl.success

        elif segment == "clear_61":
            # Natural entry from 0x71 if needed, then clear.
            if entry.screen == ROOM_L4_ENTRY and entry.mode == PLAY_MODE:
                up = make_entry_up_controller()
                controllers["entry_up"] = up
                obs, frame = _run_until(
                    env, up, assist=assist, max_frames=up.max_frames, done=_done_success, frame0=frame
                )
                if not up.success:
                    error = "entry_up_failed"
            ctl = make_room_61_clear_controller()
            # Already on target (or just entered): skip ROUTE_ENTRY.
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_VIRES_61:
                ctl.phase = DungeonPhase.FIGHT
            controllers["clear_61"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_61_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_61_cleared(env.get_ram())

        elif segment == "bomb_61":
            # Prefer cleared 0x61; if on entrance/61 with live Vires, clear first.
            ctl = make_bomb_61_north_controller(clear_vires=True)
            controllers["bomb_61"] = ctl
            # If start is 0x71, enter first.
            if entry.screen == ROOM_L4_ENTRY:
                up = make_entry_up_controller()
                controllers["entry_up"] = up
                obs, frame = _run_until(
                    env, up, assist=assist, max_frames=up.max_frames, done=_done_success, frame0=frame
                )
                if not up.success:
                    error = "entry_up_failed"
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_51_ready(env.get_ram()) and ctl.success

        elif segment == "key_51":
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_KEESE_KEY_51:
                # Rebuild path: entry → clear+bomb
                if snap.screen == ROOM_L4_ENTRY or snap.screen == ROOM_L4_VIRES_61:
                    if snap.screen == ROOM_L4_ENTRY:
                        up = make_entry_up_controller()
                        controllers["entry_up"] = up
                        obs, frame = _run_until(
                            env,
                            up,
                            assist=assist,
                            max_frames=up.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not up.success:
                            error = "entry_up_failed"
                    if error is None:
                        bomb = make_bomb_61_north_controller(clear_vires=True)
                        controllers["bomb_61"] = bomb
                        obs, frame = _run_until(
                            env,
                            bomb,
                            assist=assist,
                            max_frames=bomb.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not bomb.success:
                            error = "bomb_61_failed"
                else:
                    error = f"unsupported_start_0x{snap.screen:02x}"
            ctl = make_room_51_key_controller()
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_KEESE_KEY_51:
                ctl.phase = DungeonPhase.FIGHT
            controllers["key_51"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_51_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_51_key_success(env.get_ram())

        elif segment == "chain_to_key":
            # Full assisted path: 0x71 → 0x61 clear → bomb → 0x51 key
            up = make_entry_up_controller()
            controllers["entry_up"] = up
            obs, frame = _run_until(
                env, up, assist=assist, max_frames=up.max_frames, done=_done_success, frame0=frame
            )
            if not up.success:
                error = "entry_up_failed"
            else:
                bomb = make_bomb_61_north_controller(clear_vires=True)
                controllers["bomb_61"] = bomb
                obs, frame = _run_until(
                    env,
                    bomb,
                    assist=assist,
                    max_frames=bomb.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not bomb.success:
                    error = "bomb_or_clear_failed"
                else:
                    key = make_room_51_key_controller()
                    key.phase = DungeonPhase.FIGHT
                    controllers["key_51"] = key
                    obs, frame = _run_until(
                        env,
                        key,
                        assist=assist,
                        max_frames=ROOM_51_SPEC.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not key.success:
                        error = "key_51_failed"
            ok = error is None and level4_room_51_key_success(env.get_ram())

        elif segment == "clear_50":
            # Level4FirstKey on 0x51 keys≥1 → LEFT 0x50 → clear 5× Vire.
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_VIRES_50:
                left = make_left_50_controller()
                controllers["left_50"] = left
                obs, frame = _run_until(
                    env,
                    left,
                    assist=assist,
                    max_frames=left.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not left.success:
                    error = "left_50_failed"
            ctl = make_room_50_clear_controller()
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_VIRES_50:
                ctl.phase = DungeonPhase.FIGHT
            controllers["clear_50"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_50_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_50_cleared(env.get_ram())

        elif segment == "key_right_62":
            # Prefer start on 0x51 (FirstKey) or 0x61 with key.
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_KEESE_KEY_51:
                # DOWN to 0x61 first (simple push controller via key_right's clear room check).
                # Use a short inline DOWN push.
                for _ in range(1500):
                    snap = read_snapshot(env.get_ram())
                    if (
                        snap.level == 4
                        and snap.screen == ROOM_L4_VIRES_61
                        and snap.mode == PLAY_MODE
                        and not snap.transitioning
                    ):
                        break
                    if snap.transitioning or snap.mode in (4, 6, 7):
                        btn = "DOWN"
                    elif abs(snap.link_x - 120) > 4:
                        btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                    else:
                        btn = "DOWN"
                    obs, *_ = env.step(nes_action(btn))
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                snap = read_snapshot(env.get_ram())
                if snap.screen != ROOM_L4_VIRES_61:
                    error = "down_61_failed"
            ctl = make_key_right_62_controller(clear_vires=True)
            controllers["key_right_62"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_62_ready(env.get_ram()) and ctl.success

        elif segment == "chain_to_62":
            # FirstKey 0x51 → DOWN 0x61 clear → KEY-RIGHT 0x62 (optionally via clear_50).
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_KEESE_KEY_51:
                for _ in range(1500):
                    snap = read_snapshot(env.get_ram())
                    if (
                        snap.level == 4
                        and snap.screen == ROOM_L4_VIRES_61
                        and snap.mode == PLAY_MODE
                        and not snap.transitioning
                    ):
                        break
                    if snap.transitioning or snap.mode in (4, 6, 7):
                        btn = "DOWN"
                    elif abs(snap.link_x - 120) > 4:
                        btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                    else:
                        btn = "DOWN"
                    obs, *_ = env.step(nes_action(btn))
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                if read_snapshot(env.get_ram()).screen != ROOM_L4_VIRES_61:
                    error = "down_61_failed"
            if error is None:
                ctl = make_key_right_62_controller(clear_vires=True)
                controllers["key_right_62"] = ctl
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not ctl.success:
                    error = "key_right_62_failed"
            ok = error is None and level4_room_62_ready(env.get_ram())

        elif segment == "clear_62":
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_COMPASS_62:
                # Rebuild: FirstKey → key_right path
                if snap.screen == ROOM_L4_KEESE_KEY_51:
                    for _ in range(1500):
                        snap = read_snapshot(env.get_ram())
                        if (
                            snap.level == 4
                            and snap.screen == ROOM_L4_VIRES_61
                            and snap.mode == PLAY_MODE
                            and not snap.transitioning
                        ):
                            break
                        if snap.transitioning or snap.mode in (4, 6, 7):
                            btn = "DOWN"
                        elif abs(snap.link_x - 120) > 4:
                            btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                        else:
                            btn = "DOWN"
                        obs, *_ = env.step(nes_action(btn))
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                if read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_61:
                    kr = make_key_right_62_controller(clear_vires=True)
                    controllers["key_right_62"] = kr
                    obs, frame = _run_until(
                        env,
                        kr,
                        assist=assist,
                        max_frames=kr.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not kr.success:
                        error = "key_right_62_failed"
                elif read_snapshot(env.get_ram()).screen != ROOM_L4_COMPASS_62:
                    error = f"unsupported_start_0x{read_snapshot(env.get_ram()).screen:02x}"
            ctl = make_room_62_clear_controller()
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_COMPASS_62:
                ctl.phase = DungeonPhase.FIGHT
            controllers["clear_62"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_62_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_62_cleared(env.get_ram())

        elif segment == "compass_62":
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_COMPASS_62:
                # Prefer cleared 0x62; rebuild key-right if on 0x61/0x51.
                if snap.screen == ROOM_L4_KEESE_KEY_51:
                    for _ in range(1500):
                        snap = read_snapshot(env.get_ram())
                        if (
                            snap.level == 4
                            and snap.screen == ROOM_L4_VIRES_61
                            and snap.mode == PLAY_MODE
                            and not snap.transitioning
                        ):
                            break
                        if snap.transitioning or snap.mode in (4, 6, 7):
                            btn = "DOWN"
                        elif abs(snap.link_x - 120) > 4:
                            btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                        else:
                            btn = "DOWN"
                        obs, *_ = env.step(nes_action(btn))
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                if read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_61:
                    kr = make_key_right_62_controller(clear_vires=True)
                    controllers["key_right_62"] = kr
                    obs, frame = _run_until(
                        env,
                        kr,
                        assist=assist,
                        max_frames=kr.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not kr.success:
                        error = "key_right_62_failed"
                    else:
                        # Clear Vires so maze path is unobstructed.
                        clr = make_room_62_clear_controller()
                        clr.phase = DungeonPhase.FIGHT
                        controllers["clear_62"] = clr
                        obs, frame = _run_until(
                            env,
                            clr,
                            assist=assist,
                            max_frames=ROOM_62_SPEC.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not clr.success:
                            error = "clear_62_failed"
                elif read_snapshot(env.get_ram()).screen != ROOM_L4_COMPASS_62:
                    error = (
                        f"unsupported_start_0x"
                        f"{read_snapshot(env.get_ram()).screen:02x}"
                    )
            ctl = make_compass_62_controller()
            controllers["compass_62"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_compass_route_success(env.get_ram()) and ctl.success

        elif segment == "north_40":
            # Level4Compass → clear 61 → bomb/free UP 51 → LEFT 50 clear → N → 0x40
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_VIRES_61 or snap.screen == ROOM_L4_ENTRY:
                if snap.screen == ROOM_L4_ENTRY:
                    up = make_entry_up_controller()
                    controllers["entry_up"] = up
                    obs, frame = _run_until(
                        env,
                        up,
                        assist=assist,
                        max_frames=up.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not up.success:
                        error = "entry_up_failed"
                if error is None:
                    clr = make_room_61_clear_controller()
                    if read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_61:
                        clr.phase = DungeonPhase.FIGHT
                    controllers["clear_61"] = clr
                    obs, frame = _run_until(
                        env,
                        clr,
                        assist=assist,
                        max_frames=ROOM_61_SPEC.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not level4_room_61_cleared(env.get_ram()):
                        error = "clear_61_failed"
                if error is None:
                    bomb = make_bomb_61_north_controller(clear_vires=False)
                    controllers["bomb_61"] = bomb
                    obs, frame = _run_until(
                        env,
                        bomb,
                        assist=assist,
                        max_frames=bomb.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not bomb.success:
                        error = "bomb_61_failed"
            snap = read_snapshot(env.get_ram())
            if error is None and snap.screen == ROOM_L4_KEESE_KEY_51:
                left = make_left_50_controller()
                controllers["left_50"] = left
                obs, frame = _run_until(
                    env,
                    left,
                    assist=assist,
                    max_frames=left.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not left.success:
                    error = "left_50_failed"
            snap = read_snapshot(env.get_ram())
            if error is None and snap.screen == ROOM_L4_VIRES_50:
                if not level4_room_50_cleared(env.get_ram()):
                    c50 = make_room_50_clear_controller()
                    c50.phase = DungeonPhase.FIGHT
                    controllers["clear_50"] = c50
                    obs, frame = _run_until(
                        env,
                        c50,
                        assist=assist,
                        max_frames=ROOM_50_SPEC.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not c50.success:
                        error = "clear_50_failed"
            if error is None and read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_50:
                # Online BFS from live pose (clear_50 end varies) then long UP.
                path, bfs_meta = _bfs_50_to_north(env, assist=assist, frame0=frame)
                controllers["north_40_bfs"] = bfs_meta
                if path is None:
                    error = "bfs_50_north_failed"
                    ok = False
                else:
                    obs, frame, n40_ok = _follow_50_north_path(
                        env, path, assist=assist, frame0=frame
                    )
                    controllers["north_40"] = {
                        "success": n40_ok,
                        "path": path,
                        "frames": frame,
                        "segment": "level4_north_0x40",
                    }
                    ok = n40_ok and level4_room_40_ready(env.get_ram())
            elif error is None and read_snapshot(env.get_ram()).screen == ROOM_L4_ZOLS_40:
                ok = True
            elif error is None:
                error = (
                    f"unsupported_start_0x"
                    f"{read_snapshot(env.get_ram()).screen:02x}"
                )
                ok = False
            else:
                ok = False

        elif segment == "key_40":
            # Level4Room40 (0x40, keys=0, RoomItemId 0x19) → clear Zols + key.
            # Rebuild from Level4Compass via north_40 path if not already on 0x40.
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_ZOLS_40:
                # Reuse north_40 approach: from 0x61/entry/51/50 → 0x40.
                if snap.screen in (ROOM_L4_VIRES_61, ROOM_L4_ENTRY):
                    if snap.screen == ROOM_L4_ENTRY:
                        up = make_entry_up_controller()
                        controllers["entry_up"] = up
                        obs, frame = _run_until(
                            env,
                            up,
                            assist=assist,
                            max_frames=up.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not up.success:
                            error = "entry_up_failed"
                    if error is None:
                        clr = make_room_61_clear_controller()
                        if read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_61:
                            clr.phase = DungeonPhase.FIGHT
                        controllers["clear_61"] = clr
                        obs, frame = _run_until(
                            env,
                            clr,
                            assist=assist,
                            max_frames=ROOM_61_SPEC.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not level4_room_61_cleared(env.get_ram()):
                            error = "clear_61_failed"
                    if error is None:
                        bomb = make_bomb_61_north_controller(clear_vires=False)
                        controllers["bomb_61"] = bomb
                        obs, frame = _run_until(
                            env,
                            bomb,
                            assist=assist,
                            max_frames=bomb.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not bomb.success:
                            error = "bomb_61_failed"
                snap = read_snapshot(env.get_ram())
                if error is None and snap.screen == ROOM_L4_KEESE_KEY_51:
                    left = make_left_50_controller()
                    controllers["left_50"] = left
                    obs, frame = _run_until(
                        env,
                        left,
                        assist=assist,
                        max_frames=left.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not left.success:
                        error = "left_50_failed"
                snap = read_snapshot(env.get_ram())
                if error is None and snap.screen == ROOM_L4_VIRES_50:
                    if not level4_room_50_cleared(env.get_ram()):
                        c50 = make_room_50_clear_controller()
                        c50.phase = DungeonPhase.FIGHT
                        controllers["clear_50"] = c50
                        obs, frame = _run_until(
                            env,
                            c50,
                            assist=assist,
                            max_frames=ROOM_50_SPEC.max_frames,
                            done=_done_success,
                            frame0=frame,
                        )
                        if not c50.success:
                            error = "clear_50_failed"
                if error is None and read_snapshot(env.get_ram()).screen == ROOM_L4_VIRES_50:
                    path, bfs_meta = _bfs_50_to_north(env, assist=assist, frame0=frame)
                    controllers["north_40_bfs"] = bfs_meta
                    if path is None:
                        error = "bfs_50_north_failed"
                    else:
                        obs, frame, n40_ok = _follow_50_north_path(
                            env, path, assist=assist, frame0=frame
                        )
                        controllers["north_40"] = {
                            "success": n40_ok,
                            "path": path,
                            "frames": frame,
                            "segment": "level4_north_0x40",
                        }
                        if not n40_ok:
                            error = "north_40_failed"
                elif error is None and read_snapshot(env.get_ram()).screen != ROOM_L4_ZOLS_40:
                    error = (
                        f"unsupported_start_0x"
                        f"{read_snapshot(env.get_ram()).screen:02x}"
                    )
            ctl = make_room_40_key_controller()
            controllers["key_40"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_40_key_success(env.get_ram()) and ctl.success

        elif segment == "north_30":
            # Prefer Level4Room40Cleared (keys≥1 on 0x40); else key_40 first.
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_ZOLS_40 and not level4_room_40_key_success(
                env.get_ram()
            ):
                key = make_room_40_key_controller()
                controllers["key_40"] = key
                obs, frame = _run_until(
                    env,
                    key,
                    assist=assist,
                    max_frames=key.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not level4_room_40_key_success(env.get_ram()):
                    error = "key_40_failed"
            elif snap.screen != ROOM_L4_ZOLS_40 and snap.screen != ROOM_L4_NORTH_30:
                error = f"unsupported_start_0x{snap.screen:02x}"
            ctl = make_north_30_controller()
            controllers["north_30"] = ctl
            if error is None and read_snapshot(env.get_ram()).screen != ROOM_L4_NORTH_30:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_30_ready(env.get_ram()) and (
                ctl.success or read_snapshot(env.get_ram()).screen == ROOM_L4_NORTH_30
            )

        elif segment == "clear_30":
            # Level4Room30: clear 3× Vire 0x12 (ignore invuln 0x2b).
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_ZOLS_40:
                # Enter 0x30 first if started on cleared 0x40.
                if not level4_room_40_key_success(env.get_ram()):
                    key = make_room_40_key_controller()
                    controllers["key_40"] = key
                    obs, frame = _run_until(
                        env,
                        key,
                        assist=assist,
                        max_frames=key.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not level4_room_40_key_success(env.get_ram()):
                        error = "key_40_failed"
                if error is None:
                    n30 = make_north_30_controller()
                    controllers["north_30"] = n30
                    obs, frame = _run_until(
                        env,
                        n30,
                        assist=assist,
                        max_frames=n30.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not level4_room_30_ready(env.get_ram()):
                        error = "north_30_failed"
            elif snap.screen != ROOM_L4_NORTH_30:
                error = f"unsupported_start_0x{snap.screen:02x}"
            ctl = make_room_30_clear_controller()
            controllers["clear_30"] = ctl
            if error is None:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_30_cleared(env.get_ram()) and ctl.success

        elif segment == "key_right_31":
            # Prefer Level4Room30Cleared (keys≥1); clear first if live Vires.
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_ZOLS_40:
                n30 = make_north_30_controller()
                controllers["north_30"] = n30
                obs, frame = _run_until(
                    env,
                    n30,
                    assist=assist,
                    max_frames=n30.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not level4_room_30_ready(env.get_ram()):
                    error = "north_30_failed"
            elif snap.screen != ROOM_L4_NORTH_30 and snap.screen != ROOM_L4_EAST_31:
                error = f"unsupported_start_0x{snap.screen:02x}"
            # clear_vires=True handles precleared or live Vires on 0x30.
            ctl = make_key_right_31_controller(clear_vires=True)
            controllers["key_right_31"] = ctl
            if error is None and read_snapshot(env.get_ram()).screen != ROOM_L4_EAST_31:
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ctl.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = error is None and level4_room_31_ready(env.get_ram()) and (
                ctl.success or read_snapshot(env.get_ram()).screen == ROOM_L4_EAST_31
            )
        else:
            error = f"unknown_segment_{segment}"

        snap = read_snapshot(env.get_ram())
        final = _snap_fields(snap)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            name = _CHECKPOINT[segment]
            path = save_state(env, GAME_DIR, GAME, name)
            checkpoint = str(path)
            provenance = str(
                write_state_provenance(
                    path,
                    source_state_path=(
                        GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
                    ),
                    request={
                        "bead": bead,
                        "segment": f"l4_{segment}",
                        "track": track,
                        "intervention_class": "survival" if infinite_life else "clean",
                    },
                    selected_trial={
                        "ok": ok,
                        "segment": segment,
                        "frames": frame,
                        "final": final,
                        "controllers": {
                            k: (
                                v.report()
                                if hasattr(v, "report")
                                else (v if isinstance(v, dict) else {})
                            )
                            for k, v in controllers.items()
                        },
                    },
                    natural_entry=False,
                )
            )

        screenshot = RECORDINGS_DIR / f"{tag}_{segment}.png"
        if obs is not None:
            save_rgb_png(obs, screenshot)

        def _ctrl_rep(v: Any) -> dict[str, Any]:
            if hasattr(v, "report"):
                return v.report()
            if isinstance(v, dict):
                return v
            return {}

        return {
            "ok": ok,
            "bead": bead,
            "segment": segment,
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "infinite_life": infinite_life,
            "error": error,
            "entry": entry_fields,
            "final": final,
            "frames": frame,
            "controllers": {k: _ctrl_rep(v) for k, v in controllers.items()},
            "assist": assist.report() if assist else None,
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
            "bomb_stand": list(BOMB_61_NORTH_STAND),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", choices=SEGMENTS, default="entry_up")
    parser.add_argument("--from-state", default=None)
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default="l4_rooms")
    args = parser.parse_args(argv)

    start = args.from_state or _DEFAULT_STATE[args.segment]
    reports: list[dict] = []
    for i in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{i}"
        r = run_once(
            segment=args.segment,
            start_state=start,
            infinite_life=args.infinite_life,
            save_checkpoint=args.save_state and i == 0,
            tag=tag,
        )
        reports.append(r)
        final = r.get("final") or {}
        print(
            f"trial={i} ok={r.get('ok')} err={r.get('error')} "
            f"room={final.get('room_hex')} keys={final.get('keys')} "
            f"bombs={final.get('bombs')} frames={r.get('frames')} "
            f"ctrl={list((r.get('controllers') or {}))}"
        )
        for name, rep in (r.get("controllers") or {}).items():
            print(
                f"  {name}: success={rep.get('success')} phase={rep.get('phase')} "
                f"frames={rep.get('frames')} notes={rep.get('notes', [])[:4]}"
            )

    successes = sum(1 for r in reports if r.get("ok"))
    out = RECORDINGS_DIR / f"{args.tag}_{args.segment}.json"
    write_json_report(
        out,
        {
            "bead": _BEAD[args.segment],
            "parent": "rr-5lu",
            "segment": args.segment,
            "start_state": start,
            "track": "assisted" if args.infinite_life else "clean",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "runtime_class": "bronze",
            "natural_entry": False,
            "trials": args.trials,
            "successes": successes,
            "live_graph": {
                "0x71": {"UP": "0x61"},
                "0x61": {
                    "BOMB_UP": "0x51",
                    "KEY_RIGHT": "0x62",
                    "enemies": "3x0x12→0x1c",
                },
                "0x51": {"LEFT": "0x50", "DOWN": "0x61", "enemies": "8x0x1b", "key": "0x19"},
                "0x50": {"enemies": "5x0x12", "note": "dead_end_pocket"},
                "0x62": {
                    "enemies": "5x0x12",
                    "item": "0x16_compass",
                    "compass_bit": "0x08",
                    "note": "dark_maze_return_west",
                },
            },
            "reports": reports,
        },
    )
    print(f"wrote {out} ok={successes}/{args.trials}")
    return 0 if successes == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
