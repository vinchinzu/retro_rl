"""Scratch probe: measure the full-health sword shot (the "flying sword").

One claim per run. Phase ``kinematics`` stands Link still on 0x77 after the
sword cave and presses A once per direction, logging object slot ``$0E``
(``ObjState $BA`` / ``ObjX $7E`` / ``ObjY $92`` / ``ObjDir $A6``) every frame
until the shot deactivates. Phase ``partial`` keeps firing while the walk
takes chip damage, so the ``HeartPartial >= $80`` gate is *observed*, never
poked.

ROM (aldonunez ``Z_07.asm`` ``MakeSwordShot``): the shot lives in slot ``$0E``
and spawns only when the sword object (slot ``$0D``) reaches state 3, the
low nibble of ``HeartValues`` equals the high nibble, and ``HeartPartial``
>= ``$80``. ``Z_01.asm CheckMonsterSwordShotOrMagicShotCollision`` gives it
the *same* damage points as the blade ($10 wooden) and the same damage type
(1), so a beam kill feeds ``WorldKillCount`` / ``HelpDropCount`` identically.

Not a production CLI.
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.overworld.zd_map import map1_route
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

OUT = RECORDINGS_DIR / "scratch_beam"
SHOT_SLOT = 0x0E
SWORD_SLOT = 0x0D
ADDR_OBJ_X = 0x0070
ADDR_OBJ_Y = 0x0084
ADDR_OBJ_DIR = 0x0098
ADDR_OBJ_STATE = 0x00AC
ADDR_OBJ_TIMER = 0x0028
ADDR_OBJ_GRID_OFFSET = 0x0394
ADDR_HEALTH = 0x066F
ADDR_PARTIAL = 0x0670
SHOT_WAIT = 200  # frames to watch one shot before giving up on it
DIRS = ("RIGHT", "LEFT", "UP", "DOWN")


def _shot(ram) -> dict:
    return {
        "state": int(ram[ADDR_OBJ_STATE + SHOT_SLOT]),
        "x": int(ram[ADDR_OBJ_X + SHOT_SLOT]),
        "y": int(ram[ADDR_OBJ_Y + SHOT_SLOT]),
        "dir": int(ram[ADDR_OBJ_DIR + SHOT_SLOT]),
        "timer": int(ram[ADDR_OBJ_TIMER + SHOT_SLOT]),
        "grid": int(ram[ADDR_OBJ_GRID_OFFSET + SHOT_SLOT]),
        "sword_state": int(ram[ADDR_OBJ_STATE + SWORD_SLOT]),
    }


def _run(env, obs, controller, cap: int):
    frames = 0
    while frames < cap:
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, frames, snap
        act = controller.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if getattr(controller, "success", False):
            return obs, frames, read_snapshot(env.get_ram())
        if getattr(getattr(controller, "phase", None), "name", "") == "FAILED":
            return obs, frames, read_snapshot(env.get_ram())
    return obs, frames, read_snapshot(env.get_ram())


def _fire_once(env, obs, direction: str) -> tuple[object, dict]:
    """One A edge facing ``direction``; watch slot $0E to deactivation."""
    ram = env.get_ram()
    snap = read_snapshot(ram)
    before = {
        "link": (int(snap.link_x), int(snap.link_y)),
        "facing": int(snap.facing),
        "health": int(ram[ADDR_HEALTH]),
        "partial": int(ram[ADDR_PARTIAL]),
        "shot_state": int(ram[ADDR_OBJ_STATE + SHOT_SLOT]),
    }
    # Face first (a turn frame is not a swing), then the A edge, then idle.
    for _ in range(8):
        obs, *_ = env.step(nes_action(direction))
    obs, *_ = env.step(nes_action(direction, "A"))
    obs, *_ = env.step(nes_idle_action())
    samples: list[dict] = []
    spawned = False
    for i in range(SHOT_WAIT):
        ram = env.get_ram()
        s = _shot(ram)
        s["i"] = i
        if s["state"] != 0:
            spawned = True
            samples.append(s)
        elif spawned:
            break
        else:
            samples.append(s)
        obs, *_ = env.step(nes_idle_action())
    return obs, {
        "direction": direction,
        "before": before,
        "spawned": spawned,
        "samples": samples,
    }


def _summarize(shot: dict) -> dict:
    live = [s for s in shot["samples"] if s["state"] != 0]
    out = {
        "direction": shot["direction"],
        "link": shot["before"]["link"],
        "health": f"0x{shot['before']['health']:02X}",
        "partial": f"0x{shot['before']['partial']:02X}",
        "spawned": shot["spawned"],
        "live_frames": len(live),
    }
    if not live:
        return out
    first, last = live[0], live[-1]
    flying = [s for s in live if s["state"] % 2 == 0]
    out["spawn_xy"] = (first["x"], first["y"])
    out["spawn_offset"] = (
        first["x"] - shot["before"]["link"][0],
        first["y"] - shot["before"]["link"][1],
    )
    out["states"] = sorted({s["state"] for s in live})
    out["end_xy"] = (last["x"], last["y"])
    if flying:
        fly_first, fly_last = flying[0], flying[-1]
        dist = max(
            abs(fly_last["x"] - fly_first["x"]), abs(fly_last["y"] - fly_first["y"])
        )
        out["fly_frames"] = len(flying)
        out["fly_px"] = dist
        out["px_per_frame"] = round(dist / max(len(flying) - 1, 1), 3)
        out["reach_px"] = max(
            abs(fly_last["x"] - shot["before"]["link"][0]),
            abs(fly_last["y"] - shot["before"]["link"][1]),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("kinematics", "lane"), default="kinematics")
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--hops", type=int, default=0, help="map1 hops before firing")
    parser.add_argument("--tag", default="k1")
    args = parser.parse_args()

    if args.phase == "lane":
        run_lane(args.frames, args.tag)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        obs, boot = _boot(env)
        sword = SwordCaveController()
        obs, sword_f, snap = _run(env, obs, sword, SWORD_MAX)
        if not sword.success:
            raise SystemExit(
                f"sword failed leftover 0x{snap.screen:02X} "
                f"({snap.link_x},{snap.link_y})"
            )
        if args.hops:
            walk = OverworldPathController(
                hops=map1_route().hops[: args.hops],
                require_sword=True,
                farm_below_hearts=0,
                need_rupees=0,
                evade=True,
                max_frames=4000,
            )
            obs, _hf, snap = _run(env, obs, walk, 4000)
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            raise SystemExit(f"not playable: mode {snap.mode}")
        save_rgb_png(obs, OUT / f"{args.tag}_stand.png")

        shots = []
        for direction in DIRS:
            obs, shot = _fire_once(env, obs, direction)
            shots.append(shot)
        save_rgb_png(obs, OUT / f"{args.tag}_final.png")
        report = {
            "boot_frames": boot,
            "sword_frames": sword_f,
            "screen": f"0x{int(snap.screen):02X}",
            "link": (int(snap.link_x), int(snap.link_y)),
            "health": f"0x{int(snap.health):02X}",
            "partial": f"0x{int(snap.heart_partial):02X}",
            "summary": [_summarize(s) for s in shots],
            "shots": shots,
        }
        (OUT / f"{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("screen", "link", "health", "partial", "summary")}, indent=2))
    finally:
        env.close()


def _boot(env):
    from zelda_i.route.chain import boot_to_ready

    return boot_to_ready(env, first_playthrough=True)




# --- phase: lane ------------------------------------------------------- #
# "The beam never fired" is two different claims: the gate was shut, or
# nothing ever stood in a 9 px lane. This separates them on the live walk.


def run_lane(frames_cap: int, tag: str) -> dict:
    """Walk the real coast controller and bill every full-health frame."""
    import json as _json

    from zelda_i.beam import BEAM_HALF_WIDTH, beam_offsets, beam_ready
    from zelda_i.combat import bodies_in_box, live_enemies
    from zelda_i.overworld.hunt import HUNT_BOX
    from zelda_i.overworld.prey import PreyPolicy
    from zelda_i.overworld.shop_p7 import make_shop_p7_walk_controller

    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    prey = PreyPolicy()
    try:
        obs, _ = reset_obs(env)
        obs, boot = _boot(env)
        sword = SwordCaveController()
        obs, sword_f, snap = _run(env, obs, sword, SWORD_MAX)
        if not sword.success:
            raise SystemExit("sword failed")
        walk = make_shop_p7_walk_controller()
        ready = 0
        in_lane = 0
        lane_frames: list = []
        disagree: list = []
        aimed_before = 0
        first_chip = None
        perp_hist: dict[int, int] = {}
        by_screen: dict[str, int] = {}
        for i in range(frames_cap):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                break
            if snap.mode == PLAY_MODE and int(snap.level) == 0:
                if first_chip is None and int(snap.heart_partial) < 0x80:
                    first_chip = i
                if beam_ready(snap):
                    ready += 1
                    key = f"0x{int(snap.screen):02X}"
                    by_screen[key] = by_screen.get(key, 0) + 1
                    best = None
                    for obj in bodies_in_box(snap, HUNT_BOX):
                        if prey.skipped(obj):
                            continue
                        for d in ("RIGHT", "LEFT", "UP", "DOWN"):
                            off = beam_offsets(
                                int(snap.link_x), int(snap.link_y), d,
                                int(obj.x), int(obj.y),
                            )
                            if off is None or off[0] <= 0:
                                continue
                            perp = abs(off[1])
                            if best is None or perp < best:
                                best = perp
                    if best is not None:
                        perp_hist[best] = perp_hist.get(best, 0) + 1
                        if best <= BEAM_HALF_WIDTH:
                            in_lane += 1
                            lane_frames.append((i, snap))
            act = walk.step(snap)
            if lane_frames and lane_frames[-1][0] == i:
                hunter = walk.hunter
                aimed = 0 if hunter is None else hunter.beam.aimed
                if aimed == aimed_before and len(disagree) < 8:
                    disagree.append(
                        {
                            "i": i,
                            "reason": act.reason,
                            "screen": f"0x{int(snap.screen):02X}",
                            "link": (int(snap.link_x), int(snap.link_y)),
                            "ready": bool(hunter and beam_ready(snap)),
                            "bodies": [
                                {
                                    "slot": int(o.slot),
                                    "type": f"0x{int(o.type_id):02X}",
                                    "xy": (int(o.x), int(o.y)),
                                    "hp": int(o.hp),
                                }
                                for o in bodies_in_box(snap, HUNT_BOX)
                            ],
                            "live": [
                                (int(o.slot), f"0x{int(o.type_id):02X}",
                                 int(o.x), int(o.y), int(o.hp))
                                for o in live_enemies(snap)
                            ],
                        }
                    )
                aimed_before = aimed
            obs, *_ = env.step(act.action)
            if getattr(walk, "success", False):
                break
        snap = read_snapshot(env.get_ram())
        rep = {
            "frames": i + 1,
            "boot": boot,
            "sword_frames": sword_f,
            "ready_frames": ready,
            "in_lane_frames": in_lane,
            "first_chip_frame": first_chip,
            "ready_by_screen": by_screen,
            "min_perp_hist": dict(sorted(perp_hist.items())),
            "disagree": disagree,
            "leftover": {
                "screen": f"0x{int(snap.screen):02X}",
                "mode": int(snap.mode),
                "xy": (int(snap.link_x), int(snap.link_y)),
                "health": f"0x{int(snap.health):02X}",
                "partial": f"0x{int(snap.heart_partial):02X}",
                "rupees": int(snap.rupees),
            },
            "hunt": {
                k: v
                for k, v in (walk.report().get("hunt") or {}).items()
                if k.startswith("beam_")
            },
        }
        (OUT / f"{tag}_lane.json").write_text(_json.dumps(rep, indent=2) + "\n")
        print(_json.dumps({k: rep[k] for k in rep if k != "min_perp_hist"}, indent=2))
        print("min_perp_hist", rep["min_perp_hist"])
        return rep
    finally:
        env.close()

if __name__ == "__main__":
    main()
