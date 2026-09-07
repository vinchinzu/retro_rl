"""Room 0x10 is a `block_stairs` secret room (ROM L7-9 room table: t5&7 == 5,
same as the proven 0x05 / 0x30 push-stairs rooms). Its hidden staircase leads
to cellar 0x4F, whose room-data exit bytes both read 0x10 and whose item byte
is 0x09 = Silver Arrow. So the arrows are DOWN the stairs, not on 0x10's floor.

Clears the room, reads the tile map from cart WRAM (never moves Link to
measure), pushes the 0x68 east, re-reads the map to find the revealed stair
cell, walks onto it and reports the cellar landing + arrow count.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_10_stairs.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from pathlib import Path

from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.tilemap import ascii_room, link_cell, stair_cells
from zelda_i.level9.prefix import make_room10_silver_arrows_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"
BLOCK = 0x68


def block_of(snap):
    return next((o for o in snap.objects if o.type_id == BLOCK), None)


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    combat = make_room10_silver_arrows_controller()
    frame = 0

    def step(action):
        nonlocal frame, obs
        obs, *_ = env.step(action)
        frame += 1
        assist.apply_env(env, frame=frame)
        return read_snapshot(env.get_ram())

    snap = read_snapshot(env.get_ram())
    while not snap.room_all_dead and frame < 8000:
        snap = step(combat.policy(snap).action)
    b = block_of(snap)
    print(f"cleared f{frame} link=({snap.link_x},{snap.link_y}) block={b and (b.x, b.y)}")
    print("--- tiles BEFORE push ---")
    print(ascii_room(env.get_ram()))
    print("stair_cells before:", stair_cells(env.get_ram()))

    # Stand west of the block on its row and shove east until it stops moving.
    b = block_of(snap)
    tx, ty = b.x - 16, b.y - 3
    for _ in range(800):
        if abs(snap.link_x - tx) <= 2 and abs(snap.link_y - ty) <= 2:
            break
        if abs(snap.link_x - tx) > 2:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < ty else "UP"
        snap = step(nes_action(d))
    for _ in range(180):
        snap = step(nes_action("RIGHT"))
    b = block_of(snap)
    print(f"after push f{frame} link=({snap.link_x},{snap.link_y}) block={b and (b.x, b.y)}")
    print("--- tiles AFTER push ---")
    print(ascii_room(env.get_ram()))
    cells = stair_cells(env.get_ram())
    print("stair_cells after:", cells)

    if not cells:
        print("NO STAIRS REVEALED")
        env.close()
        return 1

    sx, sy = cells[0]
    print(f"walking to stair cell {(sx, sy)} via west lane + north band")
    # y=144 row is open x=32..192; x=32 column is open north; y=96 row is open
    # east to the stair cell. Route around the pushed block at (208,144).
    legs = [("LEFT", 32, None), ("UP", None, 93), ("RIGHT", sx, None)]
    for direction, tx, ty in legs:
        for _ in range(900):
            if snap.mode != 5:
                break
            if tx is not None and abs(snap.link_x - tx) <= 2:
                break
            if ty is not None and abs(snap.link_y - ty) <= 2:
                break
            snap = step(nes_action(direction))
        print(f"  leg {direction}: link=({snap.link_x},{snap.link_y}) mode={snap.mode} "
              f"cell={link_cell(snap.link_x, snap.link_y)}")
        if snap.mode != 5:
            break
    for _ in range(240):
        if snap.mode != 5:
            break
        snap = step(nes_idle_action())
    print(f"warp? f{frame} mode={snap.mode} screen=0x{snap.screen:02X} "
          f"link=({snap.link_x},{snap.link_y}) arrows={snap.arrows}")
    for _ in range(300):
        snap = step(nes_idle_action())
        if snap.mode == 9 and not snap.transitioning:
            break
    print(f"cellar settled f{frame} mode={snap.mode} screen=0x{snap.screen:02X} "
          f"link=({snap.link_x},{snap.link_y}) arrows={snap.arrows}")
    print("--- cellar tiles ---")
    print(ascii_room(env.get_ram()))
    # Cellar floor is y=189 (CELLAR_60_FLOOR_Y); the item sits on a centre
    # pedestal, same shape as L6's rod cellar 0x75 (level6/rod.py).
    for _ in range(400):
        if snap.link_y >= 187:
            break
        snap = step(nes_action("DOWN"))
    print(f"on floor link=({snap.link_x},{snap.link_y}) arrows={snap.arrows}")
    print("--- cellar tiles (on floor) ---")
    print(ascii_room(env.get_ram()))
    print("stair_cells:", stair_cells(env.get_ram()))
    save_rgb_png(obs, Path("nes/zelda_i/recordings/l9_cellar4f_floor.png"))
    # Cellar 0x4F (screenshot recordings/l9_cellar4f_floor.png): entry ladder
    # is the west shaft, the bottom corridor runs at y=189, and a second shaft
    # at x=176 climbs into the upper chamber whose floor is y~152. The Silver
    # Arrow sprite sits at ~(128,140) inside that chamber.
    legs = [("RIGHT", 176, None), ("UP", None, 141), ("LEFT", 128, None)]
    for direction, tx, ty in legs:
        for _ in range(600):
            if snap.arrows >= 2:
                break
            if tx is not None and abs(snap.link_x - tx) <= 2:
                break
            if ty is not None and abs(snap.link_y - ty) <= 2:
                break
            snap = step(nes_action(direction))
        print(f"  leg {direction} -> {(tx, ty)}: link=({snap.link_x},{snap.link_y}) "
              f"arrows={snap.arrows}")
    for _ in range(300):
        if snap.arrows >= 2:
            break
        snap = step(nes_idle_action())
    print(f"ARROWS f{frame} arrows={snap.arrows} link=({snap.link_x},{snap.link_y})")
    # Return: back down the chamber shaft, west along the floor, up the entry
    # ladder. Cellar 0x4F's room-data exits both read 0x10.
    ret = [("RIGHT", 176, None), ("DOWN", None, 189), ("LEFT", 48, None),
           ("UP", None, 93)]
    for direction, tx, ty in ret:
        for _ in range(900):
            if snap.mode != 9:
                break
            if tx is not None and abs(snap.link_x - tx) <= 2:
                break
            if ty is not None and abs(snap.link_y - ty) <= 2:
                break
            snap = step(nes_action(direction))
        print(f"  ret {direction} -> {(tx, ty)}: link=({snap.link_x},{snap.link_y}) "
              f"mode={snap.mode} screen=0x{snap.screen:02X}")
        if snap.mode != 9:
            break
    for _ in range(600):
        if snap.mode == 5 and snap.screen == 0x10 and not snap.transitioning:
            break
        snap = step(nes_idle_action() if snap.mode != 9 else nes_action("UP"))
    print(f"BACK f{frame} mode={snap.mode} screen=0x{snap.screen:02X} "
          f"link=({snap.link_x},{snap.link_y}) facing={snap.facing} "
          f"arrows={snap.arrows} dead={snap.room_all_dead}")
    save_rgb_png(obs, Path("nes/zelda_i/recordings/l9_room10_back_from_cellar.png"))
    print(f"RESULT f{frame} arrows={snap.arrows} mode={snap.mode} "
          f"screen=0x{snap.screen:02X} link=({snap.link_x},{snap.link_y})")
    save_rgb_png(obs, Path("nes/zelda_i/recordings/l9_cellar4f_result.png"))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
