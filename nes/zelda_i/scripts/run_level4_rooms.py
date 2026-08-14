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
    clear_31     (rr-resv)  clear 5× Vire on 0x31 maze (opens RIGHT door)
    east_32      (rr-resv)  0x31 cleared → free RIGHT → 0x32
    clear_32     (rr-tib8)  clear 2× Zol + 2× LikeLike on 0x32 (ignore 0x2b/0x68)
    stepladder   (rr-tib8)  push left block → stairs 0x60 → ADDR_LADDER
    exit_60      (rr-05fz)  Level4Stepladder mode-9 → clear Keese → BFS → 0x32 play
    west_31      (rr-05fz)  Level4PostLadder 0x32 → BFS LEFT → 0x31 (ladder=1)
    map_21       (rr-rvae)  Level4Room31PostLadder → KEY-UP 0x30→0x20→0x21 map bit

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
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_31 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment east_32 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_32 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment stepladder --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment exit_60 --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment west_31 --trials 2 --save-state
    # Map (assisted first-pass; recon key poke if keys=0)
    uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment map_21 --infinite-life --trials 2 --save-state
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level4_dungeon import (
    BOMB_61_NORTH_STAND,
    EXIT_60_HOLD,
    EXIT_60_SAMPLE_PATH,
    GEL_OBJECT_TYPE,
    KEY_30_NORTH_X,
    LEVEL4_COMPASS_BIT,
    MAP_21_HOLD,
    MAP_21_SAMPLE_PATH,
    MAZE_31_CELL_Q,
    MAZE_31_EAST_X_MIN,
    MAZE_31_EAST_Y,
    MAZE_31_EAST_Y_TOL,
    MAZE_31_HOLD,
    RIGHT_20_STAND,
    ROOM_30_SPEC,
    ROOM_31_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_L4_COMPASS_62,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_MAP_21,
    ROOM_L4_STEPLADDER,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_WATER_NORTH_20,
    ROOM_L4_ZOLS_40,
    VIRE_OBJECT_TYPE,
    VIRE_SPLIT_KEESE_TYPE,
    WEST_31_HOLD,
    WEST_31_SAMPLE_PATH,
    level4_compass_route_success,
    level4_room_30_cleared,
    level4_room_31_cleared,
    level4_room_32_cleared,
    level4_room_32_ready,
    level4_room_40_key_success,
    level4_room_40_ready,
    level4_room_50_cleared,
    level4_room_51_key_success,
    level4_room_51_ready,
    level4_room_61_cleared,
    level4_room_61_ready,
    level4_room_62_ready,
    level4_stepladder_success,
    level4_post_ladder_success,
    level4_west_31_success,
    level4_map_success,
    level4_map_room_success,
    make_bomb_61_north_controller,
    make_compass_62_controller,
    make_entry_up_controller,
    make_key_right_62_controller,
    make_left_50_controller,
    make_north_30_controller,
    make_north_40_controller,
    make_key_right_31_controller,
    make_room_30_clear_controller,
    make_room_31_clear_controller,
    make_room_32_clear_controller,
    make_room_40_key_controller,
    make_room_50_clear_controller,
    make_stepladder_controller,
    level4_room_30_ready,
    level4_room_31_ready,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_NORTH_30,
    ROOM_32_SPEC,
    make_room_51_key_controller,
    make_room_61_clear_controller,
    make_room_62_clear_controller,
    level4_room_62_cleared,
    ROOM_62_SPEC,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, ADDR_MAP, PLAY_MODE, read_snapshot, read_u8

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
    "clear_31",
    "east_32",
    "clear_32",
    "stepladder",
    "exit_60",
    "west_31",
    "map_21",
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
    "clear_31": "rr-resv",
    "east_32": "rr-resv",
    "clear_32": "rr-tib8",
    "stepladder": "rr-tib8",
    "exit_60": "rr-05fz",
    "west_31": "rr-05fz",
    "map_21": "rr-rvae",
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
    "clear_31": "Level4Room31",
    "east_32": "Level4Room31Cleared",
    "clear_32": "Level4Room32",
    "stepladder": "Level4Room32",
    "exit_60": "Level4Stepladder",
    "west_31": "Level4PostLadder",
    "map_21": "Level4Room31PostLadder",
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
    "clear_31": "Level4Room31Cleared",
    "east_32": "Level4Room32",
    "clear_32": "Level4Room32Cleared",
    "stepladder": "Level4Stepladder",
    "exit_60": "Level4PostLadder",
    "west_31": "Level4Room31PostLadder",
    "map_21": "Level4Map",
}

def _snap_fields(snap) -> dict[str, Any]:
    # snap has no ladder/map fields; callers may enrich from RAM separately.
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

def _bfs_31_to_east(env, *, assist, frame0: int, hold: int | None = None):
    """Live BFS on cleared 0x31 maze to east door band (free RIGHT → 0x32)."""
    from collections import deque

    hold = MAZE_31_HOLD if hold is None else hold
    qstep = MAZE_31_CELL_Q
    em = env.unwrapped.em
    dirs = ("UP", "DOWN", "LEFT", "RIGHT")

    def cell(x: int, y: int, q: int = qstep) -> tuple[int, int]:
        return (x // q * q, y // q * q)

    # Door R opens a few frames after clear (doors 2→3); settle first.
    for _ in range(20):
        s0 = read_snapshot(env.get_ram())
        if s0.cur_opened_doors & 0x01:  # RIGHT bit
            break
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=frame0)
        frame0 += 1

    start = read_snapshot(env.get_ram())
    if start.screen != ROOM_L4_EAST_31:
        return None, {
            "error": f"not_on_31_0x{start.screen:02x}",
            "success": False,
            "doors": start.cur_opened_doors,
        }

    start_c = cell(start.link_x, start.link_y)
    cell_state = {start_c: em.get_state()}
    parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        start_c: None
    }
    q: deque[tuple[int, int]] = deque([start_c])
    seen = {start_c}
    best_path: list[str] | None = None
    best_target: tuple[int, int] | None = None
    expansions = 0

    while q and expansions < 12000 and best_path is None:
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
            if s.screen != ROOM_L4_EAST_31 or s.mode != PLAY_MODE:
                continue
            c = cell(s.link_x, s.link_y)
            if c not in seen:
                seen.add(c)
                parent[c] = (cur, d)
                cell_state[c] = em.get_state()
                q.append(c)
                # East door band: x large, y near mid door.
                if (
                    c[0] >= MAZE_31_EAST_X_MIN
                    and abs(c[1] - MAZE_31_EAST_Y) <= MAZE_31_EAST_Y_TOL
                ):
                    path: list[str] = []
                    node = c
                    while parent[node] is not None:
                        prev, pd = parent[node]
                        path.append(pd)
                        node = prev
                    path.reverse()
                    best_path = path
                    best_target = c
                    break

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
        "target": list(best_target) if best_target else None,
        "hold": hold,
        "cell_q": qstep,
        "doors": start.cur_opened_doors,
        "max_x": max((c[0] for c in seen), default=0),
        "segment": "level4_east_0x32_bfs",
    }
    return best_path, meta

def _follow_31_east_path(env, path: list[str], *, assist, frame0: int):
    """Execute BFS path tokens then long RIGHT into 0x32."""
    frame = frame0
    obs = None
    for d in path:
        for _ in range(MAZE_31_HOLD):
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
    for _ in range(200):
        obs, *_ = env.step(nes_action("RIGHT"))
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.transitioning or s.mode in (4, 6, 7) or s.screen != ROOM_L4_EAST_31:
            # Hold RIGHT through scroll settle.
            for _ in range(80):
                obs, *_ = env.step(nes_action("RIGHT"))
                frame += 1
                if assist is not None:
                    assist.apply_env(env, frame=frame)
                s = read_snapshot(env.get_ram())
                if (
                    s.level == 4
                    and s.screen == ROOM_L4_EAST_32
                    and s.mode == PLAY_MODE
                    and not s.transitioning
                ):
                    for _ in range(20):
                        obs, *_ = env.step(nes_idle_action())
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                    return obs, frame, True
            s = read_snapshot(env.get_ram())
            if (
                s.level == 4
                and s.screen == ROOM_L4_EAST_32
                and s.mode == PLAY_MODE
                and not s.transitioning
            ):
                for _ in range(20):
                    obs, *_ = env.step(nes_idle_action())
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                return obs, frame, True
        if (
            s.level == 4
            and s.screen == ROOM_L4_EAST_32
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

def _scripted_60_ladder(env, *, assist, frame0: int):
    """Settle NW on 0x60 then follow MAZE_60_TO_LADDER hold4 to pedestal."""
    from zelda_i.level4_dungeon import (
        LADDER_60_PICKUP_XY,
        MAZE_60_HOLD,
        MAZE_60_SPAWN_XY,
        MAZE_60_TO_LADDER,
        ROOM_L4_STEPLADDER,
    )
    from zelda_i.ram import ADDR_LADDER, read_u8

    frame = frame0
    obs = None
    ladder0 = int(read_u8(env.get_ram(), ADDR_LADDER))

    # Settle mode-16 → mode-9 NW spawn band.
    for _ in range(200):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if (
            s.screen == ROOM_L4_STEPLADDER
            and s.mode == 9
            and s.link_x < 96
            and s.link_y < 120
            and not s.transitioning
        ):
            break

    s = read_snapshot(env.get_ram())
    if s.screen != ROOM_L4_STEPLADDER:
        return obs, frame, False, {
            "success": False,
            "error": f"not_on_60_0x{s.screen:02x}",
            "segment": "level4_scripted_ladder",
        }

    # Nudge toward spawn if needed.
    sx, sy = MAZE_60_SPAWN_XY
    for _ in range(80):
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - sx) <= 12 and abs(s.link_y - sy) <= 20:
            break
        dx, dy = sx - s.link_x, sy - s.link_y
        if abs(dx) > 4:
            obs, *_ = env.step(nes_action("RIGHT" if dx > 0 else "LEFT"))
        elif abs(dy) > 4:
            obs, *_ = env.step(nes_action("DOWN" if dy > 0 else "UP"))
        else:
            break
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)

    # Follow live BFS path tokens.
    for d in MAZE_60_TO_LADDER:
        for _ in range(MAZE_60_HOLD):
            obs, *_ = env.step(nes_action(d))
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.transitioning or s.mode in (4, 6, 7, 16):
                for _ in range(40):
                    obs, *_ = env.step(nes_idle_action())
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
            if int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0:
                return obs, frame, True, {
                    "success": True,
                    "frames": frame - frame0,
                    "method": "path",
                    "segment": "level4_scripted_ladder",
                }

    # Hunt pedestal.
    tx, ty = LADDER_60_PICKUP_XY
    for _ in range(200):
        s = read_snapshot(env.get_ram())
        if int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0:
            return obs, frame, True, {
                "success": True,
                "frames": frame - frame0,
                "method": "hunt",
                "segment": "level4_scripted_ladder",
            }
        dx, dy = tx - s.link_x, ty - s.link_y
        if abs(dy) > 6:
            obs, *_ = env.step(nes_action("DOWN" if dy > 0 else "UP"))
        elif abs(dx) > 4:
            obs, *_ = env.step(nes_action("RIGHT" if dx > 0 else "LEFT"))
        else:
            obs, *_ = env.step(nes_idle_action())
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)

    return obs, frame, int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0, {
        "success": int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0,
        "frames": frame - frame0,
        "final_xy": [
            read_snapshot(env.get_ram()).link_x,
            read_snapshot(env.get_ram()).link_y,
        ],
        "ladder": int(read_u8(env.get_ram(), ADDR_LADDER)),
        "segment": "level4_scripted_ladder",
    }

def _push_and_enter_60(env, *, assist, frame0: int):

    """Clear-room pose → push left block → stairs 0x60 (rr-tib8 live).

    Returns (obs, frame, ok, meta).
    """
    from zelda_i.level4_dungeon import (
        PUSH_32_DIR,
        PUSH_32_HOLD,
        PUSH_32_STAND,
        ROOM_L4_EAST_32,
        ROOM_L4_STEPLADDER,
        STAIRS_32_APPROACH,
        STAIRS_32_PUSH,
        STAIRS_32_PUSH_FRAMES,
    )

    frame = frame0
    obs = None

    def _nav(tx: int, ty: int, n: int = 500) -> None:
        nonlocal obs, frame
        for _ in range(n):
            s = read_snapshot(env.get_ram())
            if s.screen != ROOM_L4_EAST_32:
                return
            # Detour around center statues when approaching from the north.
            if s.link_y < ty - 6 and 72 <= s.link_x <= 168:
                side = 40 if s.link_x <= 120 else 200
                if abs(s.link_x - side) > 6:
                    obs, *_ = env.step(
                        nes_action("RIGHT" if s.link_x < side else "LEFT")
                    )
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                    continue
            dx, dy = tx - s.link_x, ty - s.link_y
            if abs(dx) <= 3 and abs(dy) <= 3:
                return
            if abs(dy) > 3 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
                obs, *_ = env.step(nes_action("DOWN" if dy > 0 else "UP"))
            else:
                obs, *_ = env.step(nes_action("RIGHT" if dx > 0 else "LEFT"))
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)

    # Settle after combat clear.
    for _ in range(40):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)

    # West aisle → push stand → hold LEFT.
    _nav(40, PUSH_32_STAND[1])
    _nav(PUSH_32_STAND[0], PUSH_32_STAND[1])
    for _ in range(PUSH_32_HOLD):
        obs, *_ = env.step(nes_action(PUSH_32_DIR))
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_L4_STEPLADDER or s.mode in (9, 16):
            break

    # NE approach → UP into stairs (live dual-green entry).
    if read_snapshot(env.get_ram()).screen == ROOM_L4_EAST_32:
        _nav(STAIRS_32_APPROACH[0], STAIRS_32_APPROACH[1], 500)
        for _ in range(STAIRS_32_PUSH_FRAMES):
            obs, *_ = env.step(nes_action(STAIRS_32_PUSH))
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
                s = read_snapshot(env.get_ram())
            if s.screen == ROOM_L4_STEPLADDER or s.mode in (9, 16):
                break

    s = read_snapshot(env.get_ram())
    ok = s.screen == ROOM_L4_STEPLADDER or s.mode in (9, 16)
    meta = {
        "success": ok,
        "frames": frame - frame0,
        "final_xy": [s.link_x, s.link_y],
        "final_room": f"0x{s.screen:02x}",
        "final_mode": s.mode,
        "segment": "level4_push_enter_0x60",
        "push_stand": list(PUSH_32_STAND),
        "stairs_approach": list(STAIRS_32_APPROACH),
    }
    return obs, frame, ok, meta

def _bfs_60_to_ladder(env, *, assist, frame0: int, hold: int | None = None):

    """Live BFS on mode-9 0x60 to ADDR_LADDER pickup (rr-tib8).

    Tries several hold/quantize grids — entry-frame jitter changes the
    walkable pocket under keese, so a single hold4/q4 pass can miss.
    """
    from collections import deque

    from zelda_i.level4_dungeon import MAZE_60_HOLD, ROOM_L4_STEPLADDER
    from zelda_i.ram import ADDR_LADDER, read_u8

    em = env.unwrapped.em
    dirs = ("UP", "DOWN", "LEFT", "RIGHT")

    # Mode-16 scroll → mode-9; entry pose may be transient NE (~208,93)
    # then resettles to NW spawn ~(48,80). Wait for stable mode-9 NW band.
    for _ in range(180):
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=frame0)
        frame0 += 1
        s_settle = read_snapshot(env.get_ram())
        if (
            s_settle.screen == ROOM_L4_STEPLADDER
            and s_settle.mode == 9
            and s_settle.link_x < 96
            and s_settle.link_y < 120
            and not s_settle.transitioning
        ):
            break

    start = read_snapshot(env.get_ram())
    if start.screen != ROOM_L4_STEPLADDER:
        return None, {
            "error": f"not_on_60_0x{start.screen:02x}_m{start.mode}",
            "success": False,
        }

    ladder0 = int(read_u8(env.get_ram(), ADDR_LADDER))
    base_state = em.get_state()
    hold_grid = (
        [(hold, 4)]
        if hold is not None
        else [(MAZE_60_HOLD, 4), (6, 4), (4, 2), (3, 2), (8, 4), (2, 2)]
    )

    best_path: list[str] | None = None
    goal_state = None
    best_meta: dict = {
        "start": [start.link_x, start.link_y],
        "path_len": 0,
        "n_cells": 0,
        "expansions": 0,
        "hold": hold_grid[0][0],
        "cell_q": hold_grid[0][1],
    }

    for hold_i, qstep in hold_grid:
        em.set_state(base_state)
        for _ in range(3):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=frame0)

        def cell(x: int, y: int, q: int = qstep) -> tuple[int, int]:
            return (x // q * q, y // q * q)

        s0 = read_snapshot(env.get_ram())
        start_c = cell(s0.link_x, s0.link_y)
        cell_state = {start_c: em.get_state()}
        parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
            start_c: None
        }
        q: deque[tuple[int, int]] = deque([start_c])
        seen = {start_c}
        expansions = 0
        found: list[str] | None = None
        found_state = None

        while q and expansions < 24000 and found is None:
            cur = q.popleft()
            for d in dirs:
                expansions += 1
                em.set_state(cell_state[cur])
                for _ in range(hold_i):
                    env.step(nes_action(d))
                    if assist is not None:
                        assist.apply_env(env, frame=frame0)
                s = read_snapshot(env.get_ram())
                if s.transitioning or s.mode in (4, 6, 7, 16):
                    for _ in range(40):
                        env.step(nes_idle_action())
                        if assist is not None:
                            assist.apply_env(env, frame=frame0)
                    s = read_snapshot(env.get_ram())
                if int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0:
                    path: list[str] = []
                    node: tuple[int, int] | None = cur
                    while node is not None and parent[node] is not None:
                        prev, pd = parent[node]
                        path.append(pd)
                        node = prev
                    path.reverse()
                    path.append(d)
                    found = path
                    found_state = em.get_state()
                    break
                if s.screen != ROOM_L4_STEPLADDER or s.mode == 17:
                    continue
                nc = cell(s.link_x, s.link_y)
                if nc in seen:
                    continue
                seen.add(nc)
                cell_state[nc] = em.get_state()
                parent[nc] = (cur, d)
                q.append(nc)

        best_meta = {
            "hold": hold_i,
            "cell_q": qstep,
            "expansions": expansions,
            "n_cells": len(seen),
            "start": [s0.link_x, s0.link_y],
            "path_len": len(found) if found else 0,
        }
        if found is not None and found_state is not None:
            best_path = found
            goal_state = found_state
            break

    if goal_state is not None:
        em.set_state(goal_state)
        for _ in range(5):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=frame0)
    else:
        em.set_state(base_state)
        for _ in range(5):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=frame0)

    meta = {
        "success": best_path is not None
        and int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0,
        "ladder": int(read_u8(env.get_ram(), ADDR_LADDER)),
        "segment": "level4_stepladder_bfs",
        **best_meta,
    }
    return best_path, meta

def _follow_60_ladder_path(env, path: list[str], *, assist, frame0: int):
    """No-op follow when BFS already restored goal state with ADDR_LADDER.

    Kept for API symmetry with east_32; success is ladder inventory.
    """
    from zelda_i.ram import ADDR_LADDER, read_u8

    frame = frame0
    obs = None
    if int(read_u8(env.get_ram(), ADDR_LADDER)) > 0:
        for _ in range(10):
            obs, *_ = env.step(nes_idle_action())
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
        return obs, frame, True
    # Fallback: try replaying path tokens if goal state was not applied.
    from zelda_i.level4_dungeon import MAZE_60_HOLD

    ladder0 = int(read_u8(env.get_ram(), ADDR_LADDER))
    for d in path:
        for _ in range(MAZE_60_HOLD):
            obs, *_ = env.step(nes_action(d))
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.transitioning or s.mode in (4, 6, 7, 16):
                for _ in range(40):
                    obs, *_ = env.step(nes_idle_action())
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
            if int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0:
                return obs, frame, True
    return obs, frame, int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0

def _settle_play(env, *, assist, frame0: int, max_f: int = 400):
    """Idle through scroll modes until play mode 5 (or timeout)."""
    frame = frame0
    obs = None
    for _ in range(max_f):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.mode in (PLAY_MODE, 5, 9) and not s.transitioning:
            return obs, frame, s
    return obs, frame, read_snapshot(env.get_ram())

def _follow_exit_path(env, path, *, hold: int, assist, frame0: int, dest_room: int):
    """Replay hold-N path tokens; settle scroll into dest_room play."""
    frame = frame0
    obs = None
    start = read_snapshot(env.get_ram()).screen
    for d in path:
        for _ in range(hold):
            obs, *_ = env.step(nes_action(d))
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
            s = read_snapshot(env.get_ram())
            if (
                s.screen != start
                or s.mode in (4, 6, 7, 10, 16)
                or s.transitioning
            ):
                obs, frame, s = _settle_play(env, assist=assist, frame0=frame)
                ok = s.screen == dest_room and s.mode in (PLAY_MODE, 5)
                return obs, frame, ok
    obs, frame, s = _settle_play(env, assist=assist, frame0=frame, max_f=80)
    return obs, frame, s.screen == dest_room and s.mode in (PLAY_MODE, 5)

def _bfs_60_exit_play(env, *, assist, frame0: int):
    """BFS on mode-9 0x60 to 0x32 play (rr-05fz). Returns (path, meta)."""
    from collections import deque

    from zelda_i.level4_dungeon import EXIT_60_HOLD, ROOM_L4_EAST_32, ROOM_L4_STEPLADDER

    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    if s0.screen != ROOM_L4_STEPLADDER and s0.mode != 9:
        return None, {"success": False, "error": "not_on_60"}

    for hold, quant in ((4, 4), (3, 4), (2, 4), (4, 2)):
        st = em.get_state()
        s0 = read_snapshot(env.get_ram())
        q = deque([(s0.link_x // quant * quant, s0.link_y // quant * quant, ())])
        seen = {(s0.link_x // quant * quant, s0.link_y // quant * quant)}
        exp = 0
        found = None
        while q and exp < 15000 and found is None:
            x, y, path = q.popleft()
            exp += 1
            if len(path) > 100:
                continue
            for d in ("RIGHT", "UP", "DOWN", "LEFT"):
                em.set_state(st)
                for pd in path:
                    for _ in range(hold):
                        env.step(nes_action(pd))
                for _ in range(hold):
                    env.step(nes_action(d))
                s = read_snapshot(env.get_ram())
                if s.mode == 17:
                    continue
                if (
                    s.screen != ROOM_L4_STEPLADDER
                    or s.mode in (4, 6, 7, 10, 16)
                    or s.transitioning
                ):
                    # settle without assist (restore search)
                    for _ in range(400):
                        env.step(nes_idle_action())
                        s2 = read_snapshot(env.get_ram())
                        if s2.mode in (PLAY_MODE, 5) and not s2.transitioning:
                            break
                    s2 = read_snapshot(env.get_ram())
                    if s2.screen == ROOM_L4_EAST_32 and s2.mode in (PLAY_MODE, 5):
                        found = list(path) + [d]
                        break
                    continue
                nx, ny = s.link_x // quant * quant, s.link_y // quant * quant
                if (nx, ny) in seen:
                    continue
                if abs(s.link_x - x) + abs(s.link_y - y) < 2:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, path + (d,)))
        em.set_state(st)
        env.step(nes_idle_action())
        if found is not None:
            return found, {
                "success": True,
                "hold": hold,
                "quant": quant,
                "path_len": len(found),
                "exp": exp,
                "n_cells": len(seen),
                "segment": "level4_exit_60_bfs",
            }
    return None, {"success": False, "error": "bfs_miss", "segment": "level4_exit_60_bfs"}

def _bfs_room_exit(env, *, dest: int, assist, frame0: int, hold: int = 4):
    """BFS from current play room to dest room. Returns (path, meta)."""
    from collections import deque

    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    start = s0.screen
    for h, quant in ((hold, 4), (hold, 2), (2, 4), (3, 4)):
        st = em.get_state()
        s0 = read_snapshot(env.get_ram())
        q = deque([(s0.link_x // quant * quant, s0.link_y // quant * quant, ())])
        seen = {(s0.link_x // quant * quant, s0.link_y // quant * quant)}
        exp = 0
        found = None
        while q and exp < 15000 and found is None:
            x, y, path = q.popleft()
            exp += 1
            if len(path) > 100:
                continue
            for d in ("LEFT", "RIGHT", "UP", "DOWN"):
                em.set_state(st)
                for pd in path:
                    for _ in range(h):
                        env.step(nes_action(pd))
                for _ in range(h):
                    env.step(nes_action(d))
                s = read_snapshot(env.get_ram())
                if s.mode == 17:
                    continue
                if s.screen != start or s.mode in (4, 6, 7, 10, 16) or s.transitioning:
                    for _ in range(400):
                        env.step(nes_idle_action())
                        s2 = read_snapshot(env.get_ram())
                        if s2.mode in (PLAY_MODE, 5) and not s2.transitioning:
                            break
                    s2 = read_snapshot(env.get_ram())
                    if s2.screen == dest and s2.mode in (PLAY_MODE, 5):
                        found = list(path) + [d]
                        break
                    continue
                nx, ny = s.link_x // quant * quant, s.link_y // quant * quant
                if (nx, ny) in seen:
                    continue
                if abs(s.link_x - x) + abs(s.link_y - y) < 2:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, path + (d,)))
        em.set_state(st)
        env.step(nes_idle_action())
        if found is not None:
            return found, {
                "success": True,
                "hold": h,
                "quant": quant,
                "path_len": len(found),
                "exp": exp,
                "n_cells": len(seen),
                "dest": f"0x{dest:02x}",
                "segment": "level4_room_exit_bfs",
            }
    return None, {
        "success": False,
        "error": "bfs_miss",
        "dest": f"0x{dest:02x}",
        "segment": "level4_room_exit_bfs",
    }

def run_once(
    *,
    segment: str,
    start_state: str,
    infinite_life: bool,
    save_checkpoint: bool,
    tag: str,
    allow_key_poke: bool = True,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    bead = _BEAD[segment]
    controllers: dict[str, Any] = {}
    try:
        obs, _ = reset_obs(env)
        # stepladder dual-green isolation used 5 idle frames pre-clear (RNG).
        # exit_60 needs ~150 idle after item pickup freeze on Level4Stepladder.
        from zelda_i.level4_dungeon import POST_LADDER_ITEM_SETTLE

        if segment == "stepladder":
            idle_n = 5
        elif segment == "exit_60":
            idle_n = POST_LADDER_ITEM_SETTLE
        else:
            idle_n = 1
        for i in range(idle_n):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=i)

        entry = read_snapshot(env.get_ram())
        entry_fields = _snap_fields(entry)
        frame = idle_n
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

        elif segment == "clear_31":
            # Level4Room31: clear 5× Vire 0x12 on maze (opens RIGHT door).
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_NORTH_30:
                kr = make_key_right_31_controller(clear_vires=True)
                controllers["key_right_31"] = kr
                obs, frame = _run_until(
                    env,
                    kr,
                    assist=assist,
                    max_frames=kr.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not level4_room_31_ready(env.get_ram()):
                    error = "key_right_31_failed"
            elif snap.screen != ROOM_L4_EAST_31:
                error = f"unsupported_start_0x{snap.screen:02x}"
            ctl = make_room_31_clear_controller()
            controllers["clear_31"] = ctl
            if error is None:
                snap = read_snapshot(env.get_ram())
                if snap.screen == ROOM_L4_EAST_31:
                    ctl.phase = DungeonPhase.FIGHT
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_31_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = (
                error is None
                and level4_room_31_cleared(env.get_ram())
                and ctl.success
            )

        elif segment == "east_32":
            # Prefer Level4Room31Cleared; clear first if live Vires.
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L4_EAST_31 and not level4_room_31_cleared(
                env.get_ram()
            ):
                clr = make_room_31_clear_controller()
                clr.phase = DungeonPhase.FIGHT
                controllers["clear_31"] = clr
                obs, frame = _run_until(
                    env,
                    clr,
                    assist=assist,
                    max_frames=ROOM_31_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not level4_room_31_cleared(env.get_ram()):
                    error = "clear_31_failed"
            elif snap.screen == ROOM_L4_NORTH_30:
                kr = make_key_right_31_controller(clear_vires=True)
                controllers["key_right_31"] = kr
                obs, frame = _run_until(
                    env,
                    kr,
                    assist=assist,
                    max_frames=kr.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
                if not level4_room_31_ready(env.get_ram()):
                    error = "key_right_31_failed"
                else:
                    clr = make_room_31_clear_controller()
                    clr.phase = DungeonPhase.FIGHT
                    controllers["clear_31"] = clr
                    obs, frame = _run_until(
                        env,
                        clr,
                        assist=assist,
                        max_frames=ROOM_31_SPEC.max_frames,
                        done=_done_success,
                        frame0=frame,
                    )
                    if not level4_room_31_cleared(env.get_ram()):
                        error = "clear_31_failed"
            elif snap.screen != ROOM_L4_EAST_31 and snap.screen != ROOM_L4_EAST_32:
                error = f"unsupported_start_0x{snap.screen:02x}"
            if error is None and read_snapshot(env.get_ram()).screen == ROOM_L4_EAST_32:
                ok = True
            elif error is None and read_snapshot(env.get_ram()).screen == ROOM_L4_EAST_31:
                path, bfs_meta = _bfs_31_to_east(env, assist=assist, frame0=frame)
                controllers["east_32_bfs"] = bfs_meta
                if path is None:
                    error = "bfs_31_east_failed"
                    ok = False
                else:
                    obs, frame, e32_ok = _follow_31_east_path(
                        env, path, assist=assist, frame0=frame
                    )
                    controllers["east_32"] = {
                        "success": e32_ok,
                        "path": path,
                        "frames": frame,
                        "segment": "level4_east_0x32",
                    }
                    ok = e32_ok and level4_room_32_ready(env.get_ram())
            else:
                ok = False

        elif segment == "clear_32":
            # Level4Room32: clear 2× Zol + 2× LikeLike (ignore 0x2b / 0x68).
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_EAST_32:
                error = f"unsupported_start_0x{snap.screen:02x}"
            ctl = make_room_32_clear_controller()
            controllers["clear_32"] = ctl
            if error is None:
                if snap.screen == ROOM_L4_EAST_32:
                    ctl.phase = DungeonPhase.FIGHT
                obs, frame = _run_until(
                    env,
                    ctl,
                    assist=assist,
                    max_frames=ROOM_32_SPEC.max_frames,
                    done=_done_success,
                    frame0=frame,
                )
            ok = (
                error is None
                and level4_room_32_cleared(env.get_ram())
                and ctl.success
            )

        elif segment == "stepladder":
            # From Level4Room32: clear → push left → stairs 0x60 → ADDR_LADDER.
            # Clear+push+enter use the controller; basement uses live hold4 BFS
            # (scripted path is spawn-relative; entry pose varies).
            # NOTE: do not re-import ROOM_L4_STEPLADDER here — a local import
            # makes the name function-local for all of run_once and breaks
            # exit_60 (UnboundLocalError). Module-level import is enough.
            from zelda_i.level4_dungeon import (
                MAZE_60_HOLD,
                PUSH_32_DIR,
                PUSH_32_HOLD,
                PUSH_32_STAND,
                STAIRS_32_APPROACH,
                STAIRS_32_PUSH,
                STAIRS_32_PUSH_FRAMES,
                StepladderPhase,
            )

            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_L4_EAST_32 and not level4_stepladder_success(
                env.get_ram()
            ):
                error = f"unsupported_start_0x{snap.screen:02x}"

            if error is None and not level4_stepladder_success(env.get_ram()):
                # 1) Clear with raw loop (match dual-green isolation timing).
                if not level4_room_32_cleared(env.get_ram()):
                    clr = make_room_32_clear_controller()
                    clr.phase = DungeonPhase.FIGHT
                    controllers["clear_32"] = clr
                    for _ in range(ROOM_32_SPEC.max_frames):
                        snap = read_snapshot(env.get_ram())
                        act = clr.step(snap)
                        obs, *_ = env.step(act.action)
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                        if clr.success:
                            break
                    if not level4_room_32_cleared(env.get_ram()):
                        error = "clear_32_failed"

                # 2) Controller push + enter (stop on 0x60 / mode 9/16).
                if error is None:
                    ctl = make_stepladder_controller(clear_first=False)
                    ctl.phase = StepladderPhase.ALIGN_PUSH
                    controllers["stepladder_push"] = ctl
                    for _ in range(15000):
                        snap = read_snapshot(env.get_ram())
                        act = ctl.step(snap)
                        obs, *_ = env.step(act.action)
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                        if (
                            snap.screen == ROOM_L4_STEPLADDER
                            or snap.mode in (9, 16)
                        ):
                            break
                        if ctl.phase.name in ("FAILED", "DONE"):
                            break
                    s = read_snapshot(env.get_ram())
                    if s.screen != ROOM_L4_STEPLADDER and s.mode not in (9, 16):
                        error = "stairs_enter_failed"
                        controllers["stepladder_push_final"] = ctl.report()

                # 3) Multi-grid BFS → ADDR_LADDER (goal-state restore).
                if error is None and not level4_stepladder_success(env.get_ram()):
                    path, bfs_meta = _bfs_60_to_ladder(
                        env, assist=assist, frame0=frame
                    )
                    controllers["stepladder_bfs"] = bfs_meta
                    if path is None or not bfs_meta.get("success"):
                        error = "ladder_collect_failed"
                    else:
                        obs, frame, lad_ok = _follow_60_ladder_path(
                            env, path, assist=assist, frame0=frame
                        )
                        controllers["stepladder"] = {
                            "success": lad_ok,
                            "path": path,
                            "frames": frame,
                            "segment": "level4_stepladder",
                            "method": "bfs",
                            "push_stand": list(PUSH_32_STAND),
                            "stairs_approach": list(STAIRS_32_APPROACH),
                        }
                        if not lad_ok:
                            error = "ladder_path_failed"

            ok = error is None and level4_stepladder_success(env.get_ram())

        elif segment == "exit_60":
            # Level4Stepladder mode-9 → clear Keese → BFS → 0x32 play (rr-05fz).
            snap = read_snapshot(env.get_ram())
            if int(read_u8(env.get_ram(), ADDR_LADDER)) <= 0:
                error = "no_ladder"
            elif snap.screen != ROOM_L4_STEPLADDER and snap.mode != 9:
                if not level4_post_ladder_success(env.get_ram()):
                    error = f"unsupported_start_0x{snap.screen:02x}_m{snap.mode}"

            if error is None and not level4_post_ladder_success(env.get_ram()):
                # Clear 4× Keese on 0x60 (type 0x1b).
                for f in range(5000):
                    snap = read_snapshot(env.get_ram())
                    if snap.mode == 8:
                        obs, *_ = env.step(nes_idle_action())
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                        continue
                    keese = [
                        o
                        for o in snap.objects
                        if 1 <= o.slot <= 12 and o.type_id == 0x1B
                    ]
                    if not keese:
                        break
                    tgt = min(
                        keese,
                        key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                    )
                    dx, dy = tgt.x - snap.link_x, tgt.y - snap.link_y
                    if abs(dx) >= abs(dy):
                        d = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        d = "DOWN" if dy > 0 else "UP"
                    act = nes_action(d, "A") if f % 6 < 3 else nes_action(d)
                    obs, *_ = env.step(act)
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)
                for _ in range(20):
                    obs, *_ = env.step(nes_idle_action())
                    frame += 1
                    if assist is not None:
                        assist.apply_env(env, frame=frame)

                # Live BFS exit to 0x32 play (fallback sample path).
                path, bfs_meta = _bfs_60_exit_play(
                    env, assist=assist, frame0=frame
                )
                controllers["exit_60_bfs"] = bfs_meta
                if path is None:
                    path = list(EXIT_60_SAMPLE_PATH)
                    controllers["exit_60_bfs"] = {
                        "success": False,
                        "fallback": "sample_path",
                        "path_len": len(path),
                    }
                hold = int(bfs_meta.get("hold", EXIT_60_HOLD)) if bfs_meta else EXIT_60_HOLD
                obs, frame, exit_ok = _follow_exit_path(
                    env, path, hold=hold, assist=assist, frame0=frame,
                    dest_room=ROOM_L4_EAST_32,
                )
                controllers["exit_60"] = {
                    "success": exit_ok,
                    "path": path,
                    "hold": hold,
                    "frames": frame,
                    "segment": "level4_exit_60",
                }
                if not exit_ok:
                    error = "exit_60_failed"

            ok = error is None and level4_post_ladder_success(env.get_ram())

        elif segment == "west_31":
            # Level4PostLadder 0x32 → BFS LEFT → 0x31 with ladder (rr-05fz).
            snap = read_snapshot(env.get_ram())
            if int(read_u8(env.get_ram(), ADDR_LADDER)) <= 0:
                error = "no_ladder"
            elif snap.screen != ROOM_L4_EAST_32:
                if not level4_west_31_success(env.get_ram()):
                    error = f"unsupported_start_0x{snap.screen:02x}"

            if error is None and not level4_west_31_success(env.get_ram()):
                path, bfs_meta = _bfs_room_exit(
                    env,
                    dest=ROOM_L4_EAST_31,
                    assist=assist,
                    frame0=frame,
                    hold=WEST_31_HOLD,
                )
                controllers["west_31_bfs"] = bfs_meta
                if path is None:
                    path = list(WEST_31_SAMPLE_PATH)
                    controllers["west_31_bfs"] = {
                        "success": False,
                        "fallback": "sample_path",
                        "path_len": len(path),
                    }
                hold = int(bfs_meta.get("hold", WEST_31_HOLD)) if bfs_meta else WEST_31_HOLD
                obs, frame, west_ok = _follow_exit_path(
                    env, path, hold=hold, assist=assist, frame0=frame,
                    dest_room=ROOM_L4_EAST_31,
                )
                controllers["west_31"] = {
                    "success": west_ok,
                    "path": path,
                    "hold": hold,
                    "frames": frame,
                    "segment": "level4_west_31",
                }
                if not west_ok:
                    error = "west_31_failed"

            ok = error is None and level4_west_31_success(env.get_ram())

        elif segment == "map_21":
            # rr-rvae / rr-05fz: 0x31 → 0x30 KEY-UP → 0x20 clear → RIGHT 0x21 map.
            # Assisted first-pass (--infinite-life). Default recon key poke if
            # keys=0 (compass path spent both). Use allow_key_poke=False for
            # natural-key residual (skip-compass route leaves keys≥1).
            from collections import deque

            snap = read_snapshot(env.get_ram())
            if int(read_u8(env.get_ram(), ADDR_LADDER)) <= 0:
                error = "no_ladder"
            elif level4_map_success(env.get_ram()):
                ok = True  # already have map
            else:
                recon_key = False
                if snap.keys < 1:
                    if allow_key_poke:
                        # RECON-ONLY: Level4Room31PostLadder (compass path) keys=0.
                        try:
                            env.unwrapped.data.set_value("keys", 1)
                            recon_key = True
                        except Exception as exc:  # noqa: BLE001
                            error = f"key_poke_fail:{exc!r}"
                    else:
                        error = "no_keys_natural_key_required"
                controllers["map_21_key"] = {
                    "recon_poke": recon_key,
                    "allow_key_poke": allow_key_poke,
                    "keys_after": int(read_snapshot(env.get_ram()).keys),
                }

            if error is None and not level4_map_success(env.get_ram()):
                # Ensure on 0x30 (from 0x31 LEFT or already there).
                snap = read_snapshot(env.get_ram())
                if snap.screen == ROOM_L4_EAST_31:
                    path, bfs_meta = _bfs_room_exit(
                        env,
                        dest=ROOM_L4_NORTH_30,
                        assist=assist,
                        frame0=frame,
                        hold=4,
                    )
                    controllers["map_21_to_30"] = bfs_meta or {}
                    if path is None:
                        # naive LEFT push
                        path = ["LEFT"] * 30
                    obs, frame, ok30 = _follow_exit_path(
                        env,
                        path,
                        hold=4,
                        assist=assist,
                        frame0=frame,
                        dest_room=ROOM_L4_NORTH_30,
                    )
                    if not ok30:
                        error = "no_0x30"

                if error is None:
                    snap = read_snapshot(env.get_ram())
                    if snap.screen != ROOM_L4_NORTH_30:
                        error = f"want_0x30_got_0x{snap.screen:02x}"

                # KEY-UP 0x30 → 0x20 (ladder water + key door; long push)
                if error is None:
                    # Align x≈120 then hold UP through water/key scroll.
                    for _ in range(200):
                        snap = read_snapshot(env.get_ram())
                        if snap.screen != ROOM_L4_NORTH_30:
                            break
                        if abs(snap.link_x - KEY_30_NORTH_X) > 4:
                            d = "RIGHT" if snap.link_x < KEY_30_NORTH_X else "LEFT"
                            obs, *_ = env.step(nes_action(d))
                        else:
                            break
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                    for _ in range(450):
                        snap = read_snapshot(env.get_ram())
                        if (
                            snap.screen == ROOM_L4_WATER_NORTH_20
                            and snap.mode == PLAY_MODE
                            and not snap.transitioning
                        ):
                            break
                        if snap.transitioning or snap.mode in (4, 6, 7, 16):
                            obs, *_ = env.step(nes_action("UP"))
                        else:
                            # re-align x while pushing north
                            if abs(snap.link_x - KEY_30_NORTH_X) > 8:
                                d = (
                                    "RIGHT"
                                    if snap.link_x < KEY_30_NORTH_X
                                    else "LEFT"
                                )
                                obs, *_ = env.step(nes_action(d))
                            else:
                                obs, *_ = env.step(nes_action("UP"))
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                    for _ in range(40):
                        obs, *_ = env.step(nes_idle_action())
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                    snap = read_snapshot(env.get_ram())
                    if snap.screen != ROOM_L4_WATER_NORTH_20:
                        error = (
                            f"key_up_0x30_failed_0x{snap.screen:02x}"
                            f"_keys={snap.keys}_xy=({snap.link_x},{snap.link_y})"
                        )
                    else:
                        controllers["map_21_key_up"] = {
                            "success": True,
                            "keys": int(snap.keys),
                            "xy": [snap.link_x, snap.link_y],
                        }

                # Clear 0x20 Vires
                if error is None:
                    patrol = tuple(
                        (x, y)
                        for y in (93, 109, 125, 141, 157, 173, 189)
                        for x in (48, 80, 112, 144, 176, 200)
                    )
                    spec20 = DungeonRoomSpec(
                        spec_id="l4_map_clear_20",
                        source_room=ROOM_L4_WATER_NORTH_20,
                        room_id=ROOM_L4_WATER_NORTH_20,
                        entry=DoorRoute("DOWN", ((120, 205), (120, 173))),
                        enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE, 0x1B),
                        expected_enemy_count=1,
                        alive_rule=AliveRule.TYPE_AND_HP,
                        type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE, 0x1B),
                        combat=CombatTuning(
                            patrol=patrol,
                            engage_distance=64,
                            attack_phase=4,
                            engage_attack_period=5,
                            engage_attack_hold=3,
                            patrol_attack_period=8,
                            patrol_attack_hold=2,
                        ),
                        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                        max_frames=25000,
                        level=4,
                    )
                    clr = GenericDungeonRoomController(spec20)
                    clr.phase = DungeonPhase.FIGHT
                    for _ in range(25000):
                        snap = read_snapshot(env.get_ram())
                        if snap.mode == 17:
                            error = "death_0x20"
                            break
                        if snap.screen != ROOM_L4_WATER_NORTH_20:
                            break
                        fa = clr.step(snap)
                        obs, *_ = env.step(fa.action)
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                        if clr.success:
                            break
                    controllers["map_21_clear20"] = {
                        "success": bool(clr.success),
                        "frames": clr.frames,
                    }
                    for _ in range(40):
                        obs, *_ = env.step(nes_idle_action())
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)

                # RIGHT 0x20 → 0x21 via state-saving BFS (maze; door bit R often 0)
                if error is None:
                    em = env.unwrapped.em
                    s0 = read_snapshot(env.get_ram())
                    hold_e = 4

                    def _cell4(x: int, y: int) -> tuple[int, int]:
                        return (x // 4 * 4, y // 4 * 4)

                    st = _cell4(s0.link_x, s0.link_y)
                    cs = {st: em.get_state()}
                    parent_e: dict = {st: None}
                    qe: deque = deque([st])
                    seen_e = {st}
                    goal_e = None
                    path_e = None
                    exp_e = 0
                    while qe and exp_e < 12000 and goal_e is None:
                        cur = qe.popleft()
                        for d in ("RIGHT", "UP", "DOWN", "LEFT"):
                            exp_e += 1
                            em.set_state(cs[cur])
                            for _ in range(hold_e):
                                env.step(nes_action(d))
                            s2 = read_snapshot(env.get_ram())
                            if s2.transitioning or s2.mode in (4, 6, 7, 16):
                                for _ in range(350):
                                    env.step(nes_idle_action())
                                s2 = read_snapshot(env.get_ram())
                            if (
                                s2.screen == ROOM_L4_MAP_21
                                and s2.mode == PLAY_MODE
                            ):
                                p = []
                                n = cur
                                while n is not None and parent_e[n] is not None:
                                    pp, pd = parent_e[n]
                                    p.append(pd)
                                    n = pp
                                p.reverse()
                                p.append(d)
                                path_e = p
                                goal_e = em.get_state()
                                break
                            if (
                                s2.screen != ROOM_L4_WATER_NORTH_20
                                or s2.mode != PLAY_MODE
                            ):
                                continue
                            nc = _cell4(s2.link_x, s2.link_y)
                            if nc in seen_e:
                                continue
                            if abs(s2.link_x - cur[0]) + abs(s2.link_y - cur[1]) < 2:
                                continue
                            seen_e.add(nc)
                            cs[nc] = em.get_state()
                            parent_e[nc] = (cur, d)
                            qe.append(nc)
                    controllers["map_21_to_21_bfs"] = {
                        "success": goal_e is not None,
                        "cells": len(seen_e),
                        "exp": exp_e,
                        "path_len": len(path_e) if path_e else 0,
                    }
                    if goal_e is None:
                        error = f"no_0x21_bfs_cells={len(seen_e)}"
                        em.set_state(cs[st])
                    else:
                        em.set_state(goal_e)
                        for _ in range(30):
                            obs, *_ = env.step(nes_idle_action())
                            frame += 1
                            if assist is not None:
                                assist.apply_env(env, frame=frame)
                        controllers["map_21_enter"] = {
                            "success": True,
                            "xy": [
                                read_snapshot(env.get_ram()).link_x,
                                read_snapshot(env.get_ram()).link_y,
                            ],
                            "path": path_e,
                        }

                # Gel thrash (partial clear expands maze) then hold6 BFS to map bit
                if error is None and not level4_map_success(env.get_ram()):
                    patrol = tuple(
                        (x, y)
                        for y in (93, 109, 125, 141, 157, 173, 189)
                        for x in (40, 72, 104, 136, 168, 200)
                    )
                    spec21 = DungeonRoomSpec(
                        spec_id="l4_map_gels_21",
                        source_room=ROOM_L4_MAP_21,
                        room_id=ROOM_L4_MAP_21,
                        entry=DoorRoute("LEFT", ((16, 141), (48, 141))),
                        enemy_types=(GEL_OBJECT_TYPE,),
                        expected_enemy_count=1,
                        alive_rule=AliveRule.TYPE,
                        combat=CombatTuning(
                            patrol=patrol,
                            engage_distance=56,
                            attack_phase=4,
                            engage_attack_period=4,
                            engage_attack_hold=2,
                            patrol_attack_period=8,
                            patrol_attack_hold=2,
                        ),
                        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                        max_frames=15000,
                        level=4,
                    )
                    gclr = GenericDungeonRoomController(spec21)
                    gclr.phase = DungeonPhase.FIGHT
                    for _ in range(15000):
                        snap = read_snapshot(env.get_ram())
                        if snap.mode == 17:
                            error = "death_0x21"
                            break
                        if snap.screen != ROOM_L4_MAP_21:
                            # nudge west re-entry
                            obs, *_ = env.step(nes_action("LEFT"))
                            frame += 1
                            if assist is not None:
                                assist.apply_env(env, frame=frame)
                            continue
                        if level4_map_success(env.get_ram()):
                            break
                        fa = gclr.step(snap)
                        obs, *_ = env.step(fa.action)
                        frame += 1
                        if assist is not None:
                            assist.apply_env(env, frame=frame)
                        if gclr.success:
                            break
                    controllers["map_21_gels"] = {
                        "success": bool(gclr.success),
                        "frames": gclr.frames,
                    }

                if error is None and not level4_map_success(env.get_ram()):
                    # hold6 BFS for ADDR_MAP bit
                    em = env.unwrapped.em
                    s0 = read_snapshot(env.get_ram())

                    def _cell(x: int, y: int) -> tuple[int, int]:
                        return (x // 2 * 2, y // 2 * 2)

                    st = _cell(s0.link_x, s0.link_y)
                    cs = {st: em.get_state()}
                    parent: dict = {st: None}
                    q: deque = deque([st])
                    seen = {st}
                    found_path = None
                    goal_state = None
                    exp = 0
                    while q and exp < 80000 and found_path is None:
                        cur = q.popleft()
                        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                            exp += 1
                            em.set_state(cs[cur])
                            for _ in range(MAP_21_HOLD):
                                env.step(nes_action(d))
                            s2 = read_snapshot(env.get_ram())
                            if level4_map_success(env.get_ram()):
                                p = []
                                n = cur
                                while n is not None and parent[n] is not None:
                                    pp, pd = parent[n]
                                    p.append(pd)
                                    n = pp
                                p.reverse()
                                p.append(d)
                                found_path = p
                                goal_state = em.get_state()
                                break
                            if s2.screen != ROOM_L4_MAP_21 or s2.mode != PLAY_MODE:
                                continue
                            nc = _cell(s2.link_x, s2.link_y)
                            if nc in seen:
                                continue
                            if abs(s2.link_x - cur[0]) + abs(s2.link_y - cur[1]) < 1:
                                continue
                            seen.add(nc)
                            cs[nc] = em.get_state()
                            parent[nc] = (cur, d)
                            q.append(nc)
                    if found_path is None:
                        found_path = list(MAP_21_SAMPLE_PATH)
                        controllers["map_21_bfs"] = {
                            "success": False,
                            "fallback": "sample_path",
                            "cells": len(seen),
                            "exp": exp,
                        }
                        # restore start of BFS
                        em.set_state(cs[st])
                        obs, frame, map_ok = _follow_exit_path(
                            env,
                            found_path,
                            hold=MAP_21_HOLD,
                            assist=assist,
                            frame0=frame,
                            dest_room=ROOM_L4_MAP_21,
                        )
                        # follow_exit_path checks room not map bit — recheck
                        map_ok = level4_map_success(env.get_ram())
                    else:
                        em.set_state(goal_state)
                        for _ in range(20):
                            obs, *_ = env.step(nes_idle_action())
                            frame += 1
                            if assist is not None:
                                assist.apply_env(env, frame=frame)
                        map_ok = level4_map_success(env.get_ram())
                        controllers["map_21_bfs"] = {
                            "success": map_ok,
                            "path": found_path,
                            "hold": MAP_21_HOLD,
                            "cells": len(seen),
                            "exp": exp,
                            "path_len": len(found_path),
                        }
                    controllers["map_21"] = {
                        "success": map_ok,
                        "frames": frame,
                        "segment": "level4_map_21",
                    }
                    if not map_ok:
                        error = "map_bit_not_set"

            ok = error is None and level4_map_success(env.get_ram())

        else:
            error = f"unknown_segment_{segment}"

        snap = read_snapshot(env.get_ram())
        final = _snap_fields(snap)
        try:
            final["ladder"] = int(read_u8(env.get_ram(), ADDR_LADDER))
            final["map"] = int(read_u8(env.get_ram(), ADDR_MAP))
            final["map_l4"] = bool(int(read_u8(env.get_ram(), ADDR_MAP)) & 0x08)
        except Exception:  # noqa: BLE001
            pass
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
                        "allow_key_poke": allow_key_poke,
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
            "allow_key_poke": allow_key_poke,
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
    parser.add_argument(
        "--no-key-poke",
        action="store_true",
        help="map_21: refuse recon keys poke (natural-key residual rr-05fz)",
    )
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
            allow_key_poke=not args.no_key_poke,
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
