"""0x6A south-gap occupancy. One claim: west-mouth leftover can reach 0x7A."""

from __future__ import annotations

import json
from collections import deque

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.common import EDGE_EAST_X, EDGE_SOUTH_Y
from zelda_i.overworld.shop_p7 import PRE_L1_BOMB_HOPS
from zelda_i.overworld.path import OverworldPathController, _ow_hop_grid
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist
from zelda_i.walk.physics import OccupancyWalker, WALK_DELTA

OUT = RECORDINGS_DIR / "scratch_6a_south"
STAND_HALT = 24
OCC_CAP = 4000
WALK_CAP = 8000


def _bodies(snap) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for obj in snap.objects:
        if int(getattr(obj, "slot", 0)) == 0:
            continue
        hp = int(getattr(obj, "hp", 0) or 0)
        tid = int(getattr(obj, "type_id", 0) or 0)
        if tid in (0, 0xFF, 0x60) or hp > 200:
            continue
        out.append((int(obj.x), int(obj.y)))
    return tuple(out)


def _south_goal(grid, start: tuple[int, int]) -> tuple[int, int] | None:
    """South scroll on the sand, not the west lake column."""
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if y >= EDGE_SOUTH_Y and x >= 176:
            return (x, y)
        for dx, dy in WALK_DELTA.values():
            cell = (x + dx, y + dy)
            if cell in seen or not grid.passable(*cell):
                continue
            seen.add(cell)
            q.append(cell)
    return None


def _drive(env, obs, controller, assist, cap: int, frame0: int) -> tuple:
    frames = 0
    while frames < cap:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, frames, snap
        act = controller.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frame0 + frames)
        if getattr(controller, "success", False):
            return obs, frames, read_snapshot(env.get_ram())
        phase = getattr(controller, "phase", None)
        if getattr(phase, "name", "") == "FAILED":
            return obs, frames, read_snapshot(env.get_ram())
    return obs, frames, read_snapshot(env.get_ram())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    obs, _ = reset_obs(env)
    obs, boot = boot_to_ready(env, first_playthrough=True, assist=assist)
    sword = SwordCaveController()
    obs, sword_f, snap = _drive(env, obs, sword, assist, SWORD_MAX, 0)
    if not sword.success:
        raise SystemExit(f"sword failed 0x{snap.screen:02X}")
    # Hops through arrival on 0x6B (PRE_L1_BOMB_HOPS[8] target).
    walk = OverworldPathController(
        hops=PRE_L1_BOMB_HOPS[:9],
        require_sword=True,
        farm_below_hearts=0,
        need_rupees=0,
        evade=True,
        max_frames=WALK_CAP,
    )
    obs, hop_f, snap = _drive(env, obs, walk, assist, WALK_CAP, sword_f)
    save_rgb_png(obs, OUT / "enter.png")
    if not (snap.mode == PLAY_MODE and snap.screen == 0x6B):
        raise SystemExit(f"not on 0x6B: 0x{snap.screen:02X} ({snap.link_x},{snap.link_y})")

    walker = OccupancyWalker(grid=_ow_hop_grid(), sticky=True, slide=True)
    # Dest leftover (120,189) pushed DOWN into the centre tree row. Spec-block
    # that band so BFS will look for a gap to either side.
    for x in range(64, 176):
        for y in range(185, 220):
            walker.grid.blocked.add((x, y))
    claim = f"screen=0x6B y>={EDGE_SOUTH_Y} from ({snap.link_x},{snap.link_y}) gap"
    stand = 0
    halt = ""
    for i in range(OCC_CAP):
        snap = read_snapshot(env.get_ram())
        xy = (int(snap.link_x), int(snap.link_y))
        if snap.mode == 17:
            halt = "death"
            break
        if snap.mode == PLAY_MODE and snap.screen == 0x7B:
            halt = "hit_south_0x7b"
            break
        if snap.mode != PLAY_MODE or snap.screen != 0x6B:
            halt = f"left_0x{snap.screen:02X}_m{snap.mode}"
            break
        bodies = _bodies(snap)
        goal = _south_goal(walker.grid, xy)
        if goal is None:
            halt = "miss_no_south_cell"
            break
        direction = walker.next_dir(
            xy, goal, extra_blocked=bodies, transient_occupants=bodies
        )
        if direction is None:
            stand += 1
            act = nes_idle_action()
        else:
            stand = 0
            act = nes_action(direction)
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
    save_rgb_png(obs, OUT / "final.png")
    leftover = {
        "claim": claim,
        "halt": halt,
        "hit": halt.startswith("hit_"),
        "screen": int(snap.screen),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "misses": walker.misses,
        "blocked": len(walker.grid.blocked),
        "max_x_blocked": max((c[0] for c in walker.grid.blocked), default=None),
        "enter_hop_frames": hop_f,
    }
    (OUT / "result.json").write_text(json.dumps(leftover, indent=2) + "\n")
    print(json.dumps(leftover, indent=2))


if __name__ == "__main__":
    main()
