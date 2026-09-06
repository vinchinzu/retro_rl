"""H3 recon: from the natural-arrival pin ``OW_L7PondNatural`` (Recorder
already naturally owned and pause-selected, standing on pond ``0x42`` south
shore ``(128,221)``, whistle=1), walk to the blow stand, blow 12xB, wait for
the drain, then dump the read-only cart-WRAM ``$6530`` tile map of the
drained screen and locate the staircase cells, and finally walk onto the
measured staircase and confirm the mode-16 transition into L7 play ``0x79``.

No RAM pokes (writes=0).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_dump_pond_drained.py --tag pond_drain_dump
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import find_cells, STAIR_TILES, read_room_tiles
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import make_assist

PIN = "OW_L7PondNatural"
BLOW_STAND = (128, 189)
WALK_MAX = 2000
BLOW_PRESSES = 12
POST_BLOW_WAIT = 900
WAIT_SAMPLE_EVERY = 20
STAIR_WALK_MAX = 2000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-state", default=PIN)
    parser.add_argument("--tag", default="pond_drain_dump")
    parser.add_argument("--infinite-life", action="store_true", default=True)
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]

    def step(action):
        obs, *_ = env.step(action)
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        resync_custom_state(env, GAME_DIR, GAME, args.from_state)
        snap = read_snapshot(env.get_ram())
        selected = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        print(
            f"start: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
            f"mode={snap.mode} selected_item={selected} f={total[0]}"
        )

        # --- Walk UP from the south-shore arrival to the blow stand ---
        walked = False
        for _ in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            x, y = int(snap.link_x), int(snap.link_y)
            if abs(x - BLOW_STAND[0]) <= 2 and abs(y - BLOW_STAND[1]) <= 2:
                walked = True
                break
            if x < BLOW_STAND[0] - 2:
                act = "RIGHT"
            elif x > BLOW_STAND[0] + 2:
                act = "LEFT"
            elif y > BLOW_STAND[1] + 2:
                act = "UP"
            elif y < BLOW_STAND[1] - 2:
                act = "DOWN"
            else:
                act = None
            snap = step(nes_action(act) if act else nes_idle_action())
        print(f"walked_to_stand={walked} xy=({snap.link_x},{snap.link_y}) f={total[0]}")

        for _ in range(8):
            snap = step(nes_idle_action())

        # --- Blow 12xB ---
        for i in range(BLOW_PRESSES):
            snap = step(nes_action("B"))
        print(f"blown {BLOW_PRESSES}xB, screen=0x{snap.screen:02x} f={total[0]}")

        # --- Wait for the drain, sampling the tile map periodically ---
        first_stair_frame = None
        stair_cells_seen: tuple[tuple[int, int], ...] = ()
        for i in range(POST_BLOW_WAIT):
            snap = step(nes_idle_action())
            if i % WAIT_SAMPLE_EVERY == 0 or i == POST_BLOW_WAIT - 1:
                ram = env.get_ram()
                cells = find_cells(ram, STAIR_TILES)
                if cells and first_stair_frame is None:
                    first_stair_frame = i
                    stair_cells_seen = cells
                    print(f"stairs first visible at post-blow frame {i} (f={total[0]}): {cells}")
        if first_stair_frame is None:
            ram = env.get_ram()
            stair_cells_seen = find_cells(ram, STAIR_TILES)
            print(f"stairs never distinctly seen in {POST_BLOW_WAIT}f wait; final scan: {stair_cells_seen}")

        ram = env.get_ram()
        grid = read_room_tiles(ram)
        print(f"tile map for screen 0x{snap.screen:02x} ({grid.shape[0]}x{grid.shape[1]}):")
        for row_i, row in enumerate(grid):
            print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))
        print(f"stair cells (16x16 origins): {stair_cells_seen}")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_drained_s{snap.screen:02x}.png"
        save_rgb_png(obs_box[0], png)
        print(f"drained_screenshot={png}")

        if not stair_cells_seen:
            print("NO STAIR CELLS FOUND -- aborting walk-on confirmation")
            return

        target = min(
            stair_cells_seen,
            key=lambda c: (c[0] - int(snap.link_x)) ** 2 + (c[1] - int(snap.link_y)) ** 2,
        )
        print(f"walking onto nearest stair cell {target}")
        walked_stairs = False
        for _ in range(STAIR_WALK_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 16:
                walked_stairs = True
                break
            x, y = int(snap.link_x), int(snap.link_y)
            tx, ty = target
            dx, dy = tx - x, ty - y
            if abs(dx) <= 2 and abs(dy) <= 2:
                act = "UP"
            elif abs(dx) >= abs(dy):
                act = "RIGHT" if dx > 0 else "LEFT"
            else:
                act = "DOWN" if dy > 0 else "UP"
            snap = step(nes_action(act))
            if snap.mode == 16:
                walked_stairs = True
                break
        print(
            f"walked_stairs_mode16={walked_stairs} xy=({snap.link_x},{snap.link_y}) "
            f"mode={snap.mode} f={total[0]}"
        )

        settle_frame = None
        for i in range(600):
            snap = step(nes_idle_action())
            if snap.level == 7 and snap.mode == PLAY_MODE and not snap.transitioning:
                settle_frame = i
                break
        print(
            f"settled: level={snap.level} screen=0x{int(snap.screen):02x} "
            f"xy=({snap.link_x},{snap.link_y}) mode={snap.mode} "
            f"settle_frame_after_mode16={settle_frame} f={total[0]}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
