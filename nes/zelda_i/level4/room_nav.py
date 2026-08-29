"""Level 4 live maze/BFS nav (isolated checkpoint helpers; not spine)."""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import PLAY_MODE, read_snapshot


def _tick(env, action, *, assist, frame: int):
    obs, *_ = env.step(action)
    frame += 1
    if assist is not None:
        assist.apply_env(env, frame=frame)
    return obs, frame


def _idle(env, *, assist, frame: int, n: int = 1):
    obs = None
    for _ in range(n):
        obs, frame = _tick(env, nes_idle_action(), assist=assist, frame=frame)
    return obs, frame


def _play(s, room: int) -> bool:
    return s.level == 4 and s.screen == room and s.mode == PLAY_MODE and not s.transitioning

def _bfs_50_to_north(env, *, assist, frame0: int, hold: int = 6, long_up: int = 220):
    """Live BFS on 0x50 to a north-band cell that admits long-UP into 0x40."""
    from collections import deque

    from zelda_i.level4.dungeon import ROOM_L4_VIRES_50, ROOM_L4_ZOLS_40
    from zelda_i.level4.maze_path import MAZE_50_HOLD, MAZE_50_LONG_UP

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
    from zelda_i.level4.dungeon import ROOM_L4_ZOLS_40
    from zelda_i.level4.maze_path import MAZE_50_HOLD, MAZE_50_LONG_UP

    frame = frame0
    obs = None
    for d in path:
        for _ in range(MAZE_50_HOLD):
            obs, frame = _tick(env, nes_action(d), assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.transitioning:
                for _ in range(40):
                    obs, frame = _idle(env, assist=assist, frame=frame)
    for _ in range(MAZE_50_LONG_UP + 80):
        obs, frame = _tick(env, nes_action("UP"), assist=assist, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.mode != PLAY_MODE or s.transitioning:
            for _ in range(50):
                obs, frame = _idle(env, assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
        if _play(s, ROOM_L4_ZOLS_40):
            obs, frame = _idle(env, assist=assist, frame=frame, n=20)
            return obs, frame, True
    return obs, frame, False

def _bfs_31_to_east(env, *, assist, frame0: int, hold: int | None = None):
    """Live BFS on cleared 0x31 maze to east door band (free RIGHT → 0x32)."""
    from collections import deque

    from zelda_i.level4.dungeon import (
        MAZE_31_EAST_X_MIN,
        MAZE_31_EAST_Y,
        MAZE_31_EAST_Y_TOL,
        ROOM_L4_EAST_31,
    )
    from zelda_i.level4.stepladder import MAZE_31_CELL_Q, MAZE_31_HOLD

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
    from zelda_i.level4.dungeon import ROOM_L4_EAST_31, ROOM_L4_EAST_32
    from zelda_i.level4.stepladder import MAZE_31_HOLD

    frame = frame0
    obs = None
    for d in path:
        for _ in range(MAZE_31_HOLD):
            obs, frame = _tick(env, nes_action(d), assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.transitioning:
                for _ in range(40):
                    obs, frame = _idle(env, assist=assist, frame=frame)
    for _ in range(200):
        obs, frame = _tick(env, nes_action("RIGHT"), assist=assist, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.transitioning or s.mode in (4, 6, 7) or s.screen != ROOM_L4_EAST_31:
            for _ in range(80):
                obs, frame = _tick(env, nes_action("RIGHT"), assist=assist, frame=frame)
                s = read_snapshot(env.get_ram())
                if _play(s, ROOM_L4_EAST_32):
                    obs, frame = _idle(env, assist=assist, frame=frame, n=20)
                    return obs, frame, True
            s = read_snapshot(env.get_ram())
            if _play(s, ROOM_L4_EAST_32):
                obs, frame = _idle(env, assist=assist, frame=frame, n=20)
                return obs, frame, True
        if _play(s, ROOM_L4_EAST_32):
            obs, frame = _idle(env, assist=assist, frame=frame, n=20)
            return obs, frame, True
    return obs, frame, False

def _scripted_60_ladder(env, *, assist, frame0: int):
    """Settle NW on 0x60 then follow MAZE_60_TO_LADDER hold4 to pedestal."""
    from zelda_i.level4.dungeon import LADDER_60_PICKUP_XY, ROOM_L4_STEPLADDER
    from zelda_i.level4.stepladder import (
        MAZE_60_HOLD,
        MAZE_60_SPAWN_XY,
        MAZE_60_TO_LADDER,
    )
    from zelda_i.ram import ADDR_LADDER, read_u8

    frame = frame0
    obs = None
    ladder0 = int(read_u8(env.get_ram(), ADDR_LADDER))

    # Settle mode-16 → mode-9 NW spawn band.
    for _ in range(200):
        obs, frame = _idle(env, assist=assist, frame=frame)
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
            action = nes_action("RIGHT" if dx > 0 else "LEFT")
        elif abs(dy) > 4:
            action = nes_action("DOWN" if dy > 0 else "UP")
        else:
            break
        obs, frame = _tick(env, action, assist=assist, frame=frame)

    # Follow live BFS path tokens.
    for d in MAZE_60_TO_LADDER:
        for _ in range(MAZE_60_HOLD):
            obs, frame = _tick(env, nes_action(d), assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.transitioning or s.mode in (4, 6, 7, 16):
                for _ in range(40):
                    obs, frame = _idle(env, assist=assist, frame=frame)
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
            action = nes_action("DOWN" if dy > 0 else "UP")
        elif abs(dx) > 4:
            action = nes_action("RIGHT" if dx > 0 else "LEFT")
        else:
            action = nes_idle_action()
        obs, frame = _tick(env, action, assist=assist, frame=frame)

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
    from zelda_i.level4.dungeon import (
        PUSH_32_DIR,
        PUSH_32_STAND,
        ROOM_L4_EAST_32,
        ROOM_L4_STEPLADDER,
        STAIRS_32_APPROACH,
    )
    from zelda_i.level4.stepladder import (
        PUSH_32_HOLD,
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
                    obs, frame = _tick(
                        env,
                        nes_action("RIGHT" if s.link_x < side else "LEFT"),
                        assist=assist,
                        frame=frame,
                    )
                    continue
            dx, dy = tx - s.link_x, ty - s.link_y
            if abs(dx) <= 3 and abs(dy) <= 3:
                return
            if abs(dy) > 3 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
                action = nes_action("DOWN" if dy > 0 else "UP")
            else:
                action = nes_action("RIGHT" if dx > 0 else "LEFT")
            obs, frame = _tick(env, action, assist=assist, frame=frame)

    # Settle after combat clear.
    for _ in range(40):
        obs, frame = _idle(env, assist=assist, frame=frame)

    # West aisle → push stand → hold LEFT.
    _nav(40, PUSH_32_STAND[1])
    _nav(PUSH_32_STAND[0], PUSH_32_STAND[1])
    for _ in range(PUSH_32_HOLD):
        obs, frame = _tick(env, nes_action(PUSH_32_DIR), assist=assist, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_L4_STEPLADDER or s.mode in (9, 16):
            break

    # NE approach → UP into stairs (live dual-green entry).
    if read_snapshot(env.get_ram()).screen == ROOM_L4_EAST_32:
        _nav(STAIRS_32_APPROACH[0], STAIRS_32_APPROACH[1], 500)
        for _ in range(STAIRS_32_PUSH_FRAMES):
            obs, frame = _tick(env, nes_action(STAIRS_32_PUSH), assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.transitioning:
                for _ in range(40):
                    obs, frame = _idle(env, assist=assist, frame=frame)
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

    from zelda_i.level4.dungeon import ROOM_L4_STEPLADDER
    from zelda_i.level4.stepladder import MAZE_60_HOLD
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
            obs, frame = _idle(env, assist=assist, frame=frame)
        return obs, frame, True
    # Fallback: try replaying path tokens if goal state was not applied.
    from zelda_i.level4.stepladder import MAZE_60_HOLD

    ladder0 = int(read_u8(env.get_ram(), ADDR_LADDER))
    for d in path:
        for _ in range(MAZE_60_HOLD):
            obs, frame = _tick(env, nes_action(d), assist=assist, frame=frame)
            s = read_snapshot(env.get_ram())
            if s.transitioning or s.mode in (4, 6, 7, 16):
                for _ in range(40):
                    obs, frame = _idle(env, assist=assist, frame=frame)
            if int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0:
                return obs, frame, True
    return obs, frame, int(read_u8(env.get_ram(), ADDR_LADDER)) > ladder0

def _settle_play(env, *, assist, frame0: int, max_f: int = 400):
    """Idle through scroll modes until play mode 5 (or timeout)."""
    frame = frame0
    obs = None
    for _ in range(max_f):
        obs, frame = _idle(env, assist=assist, frame=frame)
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
            obs, frame = _tick(env, nes_action(d), assist=assist, frame=frame)
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

def _bfs_room_exit(
    env,
    *,
    dest: int,
    assist,
    frame0: int,
    hold: int = 4,
    dirs: tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN"),
    segment: str = "level4_room_exit_bfs",
    hold_grid: tuple[tuple[int, int], ...] | None = None,
):
    """BFS from current play room to dest room. Returns (path, meta)."""
    from collections import deque

    em = env.unwrapped.em
    s0 = read_snapshot(env.get_ram())
    start = s0.screen
    grids = hold_grid or ((hold, 4), (hold, 2), (2, 4), (3, 4))
    for h, quant in grids:
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
            for d in dirs:
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
                "segment": segment,
            }
    return None, {
        "success": False,
        "error": "bfs_miss",
        "dest": f"0x{dest:02x}",
        "segment": segment,
    }


def _bfs_60_exit_play(env, *, assist, frame0: int):
    """BFS on mode-9 0x60 to 0x32 play (rr-05fz). Returns (path, meta)."""
    from zelda_i.level4.dungeon import ROOM_L4_EAST_32, ROOM_L4_STEPLADDER

    s0 = read_snapshot(env.get_ram())
    if s0.screen != ROOM_L4_STEPLADDER and s0.mode != 9:
        return None, {"success": False, "error": "not_on_60"}
    return _bfs_room_exit(
        env,
        dest=ROOM_L4_EAST_32,
        assist=assist,
        frame0=frame0,
        dirs=("RIGHT", "UP", "DOWN", "LEFT"),
        segment="level4_exit_60_bfs",
        hold_grid=((4, 4), (3, 4), (2, 4), (4, 2)),
    )
