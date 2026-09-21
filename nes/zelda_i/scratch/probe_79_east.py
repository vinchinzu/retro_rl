"""0x79 east-gap occupancy. One claim, one act, halt on miss.

Claim: from the west mouth, OccupancyWalker can reach the east scroll
(0x7A). Unknown cells start free; a bump blocks; no path → stand.

Not a y-sweep. Inland 0x68 is not on this hop table.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.common import EDGE_EAST_X, EDGE_NORTH_Y, EDGE_SOUTH_Y
from zelda_i.overworld.path import OverworldPathController, _ow_hop_grid
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.overworld.zd_map import map1_route
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist
from zelda_i.walk.physics import OccupancyWalker, WALK_DELTA

OUT = RECORDINGS_DIR / "scratch_79_east"
STAND_HALT = 24
OCC_CAP = 4000
HOP_CAP = 4000


def _bodies(snap) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for obj in snap.objects:
        if int(getattr(obj, "slot", 0)) == 0:
            continue
        hp = int(getattr(obj, "hp", 0) or 0)
        tid = int(getattr(obj, "type_id", 0) or 0)
        if tid in (0, 0xFF, 0x60):
            continue
        if hp > 200:
            continue
        out.append((int(obj.x), int(obj.y)))
    return tuple(out)


def _edge_goal(grid, start: tuple[int, int], *, axis: str) -> tuple[int, int] | None:
    """Nearest cell on the learned grid that sits on the named scroll line."""
    sx, sy = start
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if axis == "east" and x >= EDGE_EAST_X:
            return (x, y)
        if axis == "north" and y <= EDGE_NORTH_Y:
            return (x, y)
        if axis == "south" and y >= EDGE_SOUTH_Y:
            return (x, y)
        for dx, dy in WALK_DELTA.values():
            cell = (x + dx, y + dy)
            if cell in seen or not grid.passable(*cell):
                continue
            if cell == start:
                continue
            seen.add(cell)
            q.append(cell)
    return None


def _run(env, obs, controller, assist, cap: int) -> tuple[object, int, object]:
    frames = 0
    while frames < cap:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, frames, snap
        act = controller.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frames)
        if getattr(controller, "success", False):
            return obs, frames, read_snapshot(env.get_ram())
        phase = getattr(controller, "phase", None)
        if getattr(phase, "name", "") == "FAILED":
            return obs, frames, read_snapshot(env.get_ram())
    return obs, frames, read_snapshot(env.get_ram())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("east", "north"), default="east")
    args = parser.parse_args()
    axis = args.axis
    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    painted = map1_route()
    hops = painted.hops[:2]
    assert hops[0].target == 0x78 and hops[1].target == 0x79

    obs, _ = reset_obs(env)
    obs, boot = boot_to_ready(env, first_playthrough=True, assist=assist)
    sword = SwordCaveController()
    obs, sword_f, snap = _run(env, obs, sword, assist, SWORD_MAX)
    if not sword.success:
        raise SystemExit(f"sword failed leftover 0x{snap.screen:02X} {snap.link_x},{snap.link_y}")

    walk = OverworldPathController(
        hops=hops,
        require_sword=True,
        farm_below_hearts=0,
        need_rupees=0,
        evade=True,
        max_frames=HOP_CAP,
    )
    obs, hop_f, snap = _run(env, obs, walk, assist, HOP_CAP)
    save_rgb_png(obs, OUT / f"enter_{axis}.png")
    enter = {
        "screen": int(snap.screen),
        "mode": int(snap.mode),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "hop_frames": hop_f,
        "sword_frames": sword_f,
        "boot": boot,
        "hop_ok": bool(walk.success),
        "notes": list(walk.notes),
        "align_y": hops[1].align_y,
    }
    if not (snap.mode == PLAY_MODE and snap.screen == 0x79):
        (OUT / "enter.json").write_text(json.dumps(enter, indent=2) + "\n")
        raise SystemExit(f"not on 0x79: {enter}")

    walker = OccupancyWalker(grid=_ow_hop_grid(), sticky=True, slide=True)
    if axis == "east":
        claim = f"screen=0x79 x>={EDGE_EAST_X} from ({snap.link_x},{snap.link_y})"
        hit_screen, hit_name = 0x7A, "hit_east_0x7a"
    else:
        claim = f"screen=0x79 y<={EDGE_NORTH_Y} from ({snap.link_x},{snap.link_y})"
        hit_screen, hit_name = 0x69, "hit_north_0x69"
    stand = 0
    trail: list[dict] = []
    halt = ""
    for i in range(OCC_CAP):
        snap = read_snapshot(env.get_ram())
        xy = (int(snap.link_x), int(snap.link_y))
        if snap.mode == 17:
            halt = "death"
            break
        if snap.mode == PLAY_MODE and snap.screen == hit_screen:
            halt = hit_name
            break
        if snap.mode != PLAY_MODE or snap.screen != 0x79:
            halt = f"left_0x{snap.screen:02X}_m{snap.mode}"
            break
        bodies = _bodies(snap)
        goal = _edge_goal(walker.grid, xy, axis=axis)
        if goal is None:
            halt = f"miss_no_{axis}_cell"
            break
        direction = walker.next_dir(
            xy,
            goal,
            extra_blocked=bodies,
            transient_occupants=bodies,
        )
        if direction is None:
            stand += 1
            act = nes_idle_action()
            reason = "stand"
        else:
            stand = 0
            act = nes_action(direction)
            reason = direction
        if i % 30 == 0 or reason == "stand":
            trail.append(
                {
                    "i": i,
                    "xy": xy,
                    "goal": goal,
                    "dir": direction,
                    "misses": walker.misses,
                    "blocked": len(walker.grid.blocked),
                    "reason": reason,
                }
            )
        obs, *_ = env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=sword_f + hop_f + i)
        if stand >= STAND_HALT:
            halt = "miss_stand"
            snap = read_snapshot(env.get_ram())
            break
    else:
        halt = "timeout"
        snap = read_snapshot(env.get_ram())

    save_rgb_png(obs, OUT / f"final_{axis}.png")
    leftover = {
        "claim": claim,
        "halt": halt,
        "hit": halt.startswith("hit_"),
        "screen": int(snap.screen),
        "mode": int(snap.mode),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "facing": int(snap.facing),
        "misses": walker.misses,
        "blocked": len(walker.grid.blocked),
        "enter": enter,
        "trail_tail": trail[-12:],
        "blocked_sample": sorted(walker.grid.blocked)[:80],
    }
    (OUT / f"result_{axis}.json").write_text(json.dumps(leftover, indent=2) + "\n")
    print(json.dumps({k: leftover[k] for k in ("claim", "halt", "hit", "screen", "x", "y", "misses", "blocked")}, indent=2))


if __name__ == "__main__":
    main()
