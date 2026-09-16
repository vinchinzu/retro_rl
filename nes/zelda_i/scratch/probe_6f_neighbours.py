"""Scratch probe: what the coast shop screen 0x6F is actually next to.

The walk banks 19R of a 20R pack (``pre_l1_beam4``), so the errand needs an
answer to "arrived short". A top-up has to fight somewhere the ROM will still
spawn a wave, and ``overworld.respawn`` says the corridor behind 0x6F cannot
be one: ``RoomHistory`` is six slots and an out-and-back evicts nothing, so
0x7F..0x7B are all still in the ring on arrival. The screens that are
guaranteed fresh are the ones the walk has never entered — 0x5F north and
0x6E west — and neither has a measured lane.

Same shape as ``probe_coast_lane.py``: drive the real hop table to 0x6F with
the health assist ON (this is geometry, not a survival claim), save the
emulator state on arrival, then for each candidate row/column restore, walk
to it, hold the exit direction, and record whether the screen scrolled and
what was standing there when it did.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_6f_neighbours.py --tag n1

Not a production CLI. 0x5F is ``locations.ladder_heart`` (the water platform
needs the Stepladder); this asks only whether Link can stand on the screen
and fight, not whether he can reach the heart.
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import live_enemies
from zelda_i.overworld.shop_p7 import (
    SHOP_P7_HOPS,
    SHOP_P7_SCREEN,
    ShopP7WalkController,
)
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready
from zelda_i.runner import make_assist

OUT = RECORDINGS_DIR / "scratch_6f_neighbours"
WALK_CAP = 26000
ALIGN_CAP = 420
PUSH_CAP = 520
ALIGN_TOL = 3
SETTLE_FRAMES = 150  # the wave is not live the moment the scroll ends
# Modes that are a *place*, not a scroll. 11 is a cave interior, 12/13 are
# its dialog. A scroll (6/7) is passed through, not reported.
CAVE_MODES = frozenset({11, 12, 13})
_REVERSE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# Rows for a horizontal exit, columns for a vertical one. The 0x6F cave mouth
# is (48, 77), so an UP push near x=48 walks into the shop instead of the
# scroll line — the column sweep starts east of it.
ROWS = tuple(range(77, 206, 8))
COLS = tuple(range(72, 233, 10))

# (from_screen, direction, axis) — the exits off 0x6F worth asking about.
# DOWN is the way the walk arrived and is here as the control: if the sweep
# cannot reproduce a lane it already walked, the sweep is wrong.
EXITS = (
    (SHOP_P7_SCREEN, "UP", "x"),
    (SHOP_P7_SCREEN, "LEFT", "y"),
    (SHOP_P7_SCREEN, "DOWN", "x"),
)


def _run(env, obs, controller, assist, cap: int, frame_base: int):
    frames = 0
    while frames < cap:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, frames, snap
        act = controller.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frame_base + frames)
        if getattr(controller, "success", False):
            return obs, frames, read_snapshot(env.get_ram())
        if getattr(getattr(controller, "phase", None), "name", "") == "FAILED":
            return obs, frames, read_snapshot(env.get_ram())
    return obs, frames, read_snapshot(env.get_ram())


def _census(env, assist, frames: int = SETTLE_FRAMES) -> dict:
    """Stand still and count what spawned. Peak, not total."""
    peak = 0
    types: dict[str, int] = {}
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            break
        bodies = live_enemies(snap)
        if len(bodies) > peak:
            peak = len(bodies)
            types = {}
            for body in bodies:
                key = f"{int(body.type_id):#04x}"
                types[key] = types.get(key, 0) + 1
        env.step(nes_action("A"))  # idle-with-swing: never a direction
        if assist is not None:
            assist.apply_env(env)
    return {"peak_live": peak, "types": types}


def _try_exit(env, assist, screen: int, direction: str, axis: str, want: int) -> dict:
    """Walk to ``want`` on ``axis``, then hold ``direction``. Did it scroll?"""
    snap = read_snapshot(env.get_ram())
    entry = (int(snap.link_x), int(snap.link_y))
    aligned = False
    align_frames = 0
    toward = ("DOWN", "UP") if axis == "y" else ("RIGHT", "LEFT")
    for align_frames in range(1, ALIGN_CAP + 1):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE or int(snap.screen) != screen:
            break
        now = int(snap.link_y) if axis == "y" else int(snap.link_x)
        if abs(now - want) <= ALIGN_TOL:
            aligned = True
            break
        env.step(nes_action(toward[0] if now < want else toward[1]))
        if assist is not None:
            assist.apply_env(env)
    snap = read_snapshot(env.get_ram())
    stood = (int(snap.link_x), int(snap.link_y))
    crossed = ""
    push_frames = 0
    # A scroll is *not* an exit condition: mode leaves 5 for the whole
    # transition and ``$00EB`` only names the new screen once it lands. The
    # first sweep read every crossing as ``mode_6`` for exactly that reason.
    for push_frames in range(1, PUSH_CAP + 1):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            crossed = "death"
            break
        if snap.mode in CAVE_MODES:
            crossed = f"mode_{int(snap.mode)}"  # cave mouth / shop dialog
            break
        if snap.mode == PLAY_MODE and int(snap.screen) != screen:
            crossed = f"0x{int(snap.screen):02X}"
            break
        env.step(nes_action(direction))
        if assist is not None:
            assist.apply_env(env)
    row = {
        "want": want,
        "axis": axis,
        "direction": direction,
        "entry": entry,
        "aligned": aligned,
        "stood": stood,
        "align_frames": align_frames,
        "push_frames": push_frames,
        "crossed": crossed,
        "frames": align_frames + push_frames,
    }
    if crossed.startswith("0x"):
        snap = read_snapshot(env.get_ram())
        row["arrival"] = (int(snap.link_x), int(snap.link_y))
        row["census"] = _census(env, assist)
        # The way home, from where the census left Link. An excursion that
        # cannot come back is not an excursion: the cave mouth is on 0x6F.
        row["home"] = _push_back(env, assist, int(snap.screen), _REVERSE[direction])
    return row


def _push_back(env, assist, from_screen: int, direction: str) -> dict:
    """Hold ``direction`` from wherever the census left Link. Did it come home?"""
    landed = ""
    for frames in range(1, PUSH_CAP + 1):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            landed = "death"
            break
        if snap.mode in CAVE_MODES:
            landed = f"mode_{int(snap.mode)}"
            break
        if snap.mode == PLAY_MODE and int(snap.screen) != from_screen:
            landed = f"0x{int(snap.screen):02X}"
            break
        env.step(nes_action(direction))
        if assist is not None:
            assist.apply_env(env)
    snap = read_snapshot(env.get_ram())
    return {
        "direction": direction,
        "landed": landed,
        "frames": frames,
        "at": (int(snap.link_x), int(snap.link_y)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="n1")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    exits: list[dict] = []
    try:
        obs, _ = reset_obs(env)
        obs, boot = boot_to_ready(env, first_playthrough=True, assist=assist)
        sword = SwordCaveController()
        obs, sword_f, snap = _run(env, obs, sword, assist, SWORD_MAX, boot)
        if not sword.success:
            raise SystemExit(f"sword failed 0x{int(snap.screen):02X}")
        # ``ShopP7WalkController``, not the bare path: 0x79 east is the beach
        # skirt and only that subclass knows it (``_leave_79_east``). The
        # hunter is off — this is geometry, and a fight would spend the boot
        # on the corridor instead of the sweep.
        walk = ShopP7WalkController(
            hops=SHOP_P7_HOPS,
            hunter=None,
            hunt_destination=False,
            max_frames=WALK_CAP,
        )
        obs, hop_f, snap = _run(env, obs, walk, assist, WALK_CAP, boot + sword_f)
        if not (snap.mode == PLAY_MODE and int(snap.screen) == SHOP_P7_SCREEN):
            raise SystemExit(f"walk missed 0x6F: 0x{int(snap.screen):02X} mode {snap.mode}")
        save_rgb_png(env.render(), OUT / f"{args.tag}_6F_arrival.png")
        arrival = env.em.get_state()
        print(json.dumps({"arrival": (int(snap.link_x), int(snap.link_y)),
                          "rupees": int(snap.rupees), "frames": boot + sword_f + hop_f}))

        for screen, direction, axis in EXITS:
            candidates = ROWS if axis == "y" else COLS
            rows = []
            for want in candidates:
                env.em.set_state(arrival)
                row = _try_exit(env, assist, screen, direction, axis, want)
                rows.append(row)
                print(json.dumps({"screen": f"0x{screen:02X}", **{
                    k: v for k, v in row.items() if k != "census"}},
                    default=str), flush=True)
            good = [r for r in rows if r["crossed"].startswith("0x")]
            exits.append(
                {
                    "screen": f"0x{screen:02X}",
                    "direction": direction,
                    "axis": axis,
                    "rows": rows,
                    "lanes": [
                        (r["want"], r["stood"], r["crossed"], r["frames"],
                         r.get("census"))
                        for r in good
                    ],
                }
            )
            (OUT / f"{args.tag}.json").write_text(
                json.dumps({"exits": exits}, indent=2) + "\n"
            )
        env.em.set_state(arrival)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
