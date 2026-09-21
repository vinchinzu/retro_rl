"""Scratch probe: give ``bomb_topup`` a live frame.

The walk with hunter off + assist ON is the geometry path that already
reached 0x6F (``probe_6f_neighbours.py`` n4). This sitting continues that
boot into ``RupeeTopUpController`` so the stage that has never run gets a
``reason_by_screen`` and a rupee count. Assist ON, hunter off on the walk;
the top-up hunts. Not a Clean claim; not a STATUS.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_topup.py --tag t1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.overworld.shop_p7 import (
    SHOP_P7_BUY_MAX_FRAMES,
    SHOP_P7_HOPS,
    SHOP_P7_PRICE,
    SHOP_P7_SCREEN,
    ShopP7WalkController,
    make_shop_p7_buy_controller,
)
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.overworld.topup import TOPUP_MAX_FRAMES, make_shop_p7_topup_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.route.chain import boot_to_ready, run_controller_stage
from zelda_i.runner import make_assist
from zelda_i.screen_glance import leftover_from_snapshot

OUT = RECORDINGS_DIR / "scratch_topup"
WALK_CAP = 26000
STUCK_SAMPLE = 250


def _glance(snap) -> dict:
    leftover = leftover_from_snapshot(snap)
    leftover["sword"] = int(snap.sword)
    leftover["rupees"] = int(snap.rupees)
    leftover["hearts"] = f"{snap.filled_hearts}/{snap.heart_containers}"
    leftover["hearts_lo"] = int(snap.health) & 0x0F
    leftover["hearts_hi"] = (int(snap.health) >> 4) & 0x0F
    leftover["facing"] = int(snap.facing)
    leftover["level"] = int(snap.level)
    leftover["screen_hex"] = f"0x{int(snap.screen):02X}"
    return leftover


class _Shot:
    """Screenshot every screen/mode change and a compact stuck sample."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.last = None
        self.stuck_at = None
        self.stuck_n = 0
        self.n = 0

    def __call__(self, env, obs, action, frame: int) -> None:
        snap = read_snapshot(env.get_ram())
        key = (int(snap.mode), int(snap.screen))
        xy = (int(snap.link_x), int(snap.link_y))
        take = False
        if key != self.last:
            take = True
            self.last = key
            self.stuck_at = xy
            self.stuck_n = 0
        elif xy == self.stuck_at:
            self.stuck_n += 1
            if self.stuck_n % STUCK_SAMPLE == 0:
                take = True
        else:
            self.stuck_at = xy
            self.stuck_n = 0
        if not take:
            return
        self.n += 1
        name = (
            f"{self.tag}_{self.n:03d}_f{frame}_m{snap.mode:02d}_"
            f"s{snap.screen:02X}_x{snap.link_x}_y{snap.link_y}.png"
        )
        save_rgb_png(env.render(), OUT / name)


def _stage(ctl, name: str) -> dict:
    snap = getattr(ctl, "_nav_snap", None)
    out = {
        "name": name,
        "success": bool(getattr(ctl, "success", False)),
        "phase": getattr(getattr(ctl, "phase", None), "name", None),
        "frames": int(getattr(ctl, "frames", 0)),
        "hop_index": int(getattr(ctl, "hop_index", -1)),
        "notes": list(getattr(ctl, "notes", ())),
        "hits_taken": int(getattr(ctl, "hits_taken", 0)),
    }
    if callable(getattr(ctl, "report", None)):
        out["controller"] = ctl.report()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="t1")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    shot = _Shot(args.tag)
    payload: dict = {"tag": args.tag, "assist": True, "walk_hunter": False}
    try:
        obs, _ = reset_obs(env)
        obs, boot = boot_to_ready(
            env, first_playthrough=True, assist=assist, on_frame=shot
        )
        payload["boot_frames"] = boot

        sword = SwordCaveController()
        obs, sword_res = run_controller_stage(
            env, obs, name="sword_cave", controller=sword,
            max_frames=SWORD_MAX, assist=assist, on_frame=shot, frame_base=boot,
        )
        snap = read_snapshot(env.get_ram())
        payload["sword"] = {**_stage(sword, "sword_cave"), "leftover": _glance(snap)}
        if not sword.success:
            raise SystemExit(f"sword failed 0x{int(snap.screen):02X}")

        walk = ShopP7WalkController(
            hops=SHOP_P7_HOPS,
            hunter=None,
            hunt_destination=False,
            max_frames=WALK_CAP,
        )
        obs, walk_res = run_controller_stage(
            env, obs, name="bomb_walk", controller=walk,
            max_frames=WALK_CAP, assist=assist, on_frame=shot,
            frame_base=sword_res.end_frame,
        )
        snap = read_snapshot(env.get_ram())
        payload["walk"] = {
            **_stage(walk, "bomb_walk"),
            "leftover": _glance(snap),
            "arrived_6f": snap.mode == PLAY_MODE and int(snap.screen) == SHOP_P7_SCREEN,
        }
        save_rgb_png(env.render(), OUT / f"{args.tag}_6F_arrival.png")
        if not payload["walk"]["arrived_6f"]:
            raise SystemExit(
                f"walk missed 0x6F: 0x{int(snap.screen):02X} mode {snap.mode}"
            )
        print(json.dumps({
            "arrival": (int(snap.link_x), int(snap.link_y)),
            "rupees": int(snap.rupees),
            "frames": walk_res.end_frame,
        }), flush=True)

        topup = make_shop_p7_topup_controller(
            shop_screen=SHOP_P7_SCREEN, price=SHOP_P7_PRICE
        )
        obs, topup_res = run_controller_stage(
            env, obs, name="bomb_topup", controller=topup,
            max_frames=TOPUP_MAX_FRAMES, assist=assist, on_frame=shot,
            frame_base=walk_res.end_frame,
        )
        snap = read_snapshot(env.get_ram())
        payload["topup"] = {
            **_stage(topup, "bomb_topup"),
            "leftover": _glance(snap),
            "short": int(snap.rupees) < SHOP_P7_PRICE,
            "home": snap.mode == PLAY_MODE and int(snap.screen) == SHOP_P7_SCREEN,
        }
        save_rgb_png(env.render(), OUT / f"{args.tag}_topup_final.png")
        print(json.dumps({
            "topup_ok": bool(topup.success),
            "rupees": int(snap.rupees),
            "screen": f"0x{int(snap.screen):02X}",
            "frames": topup_res.frames,
            "notes": list(topup.notes),
        }, default=str), flush=True)

        if (
            snap.mode == PLAY_MODE
            and int(snap.screen) == SHOP_P7_SCREEN
            and int(snap.rupees) >= SHOP_P7_PRICE
        ):
            buy = make_shop_p7_buy_controller()
            obs, buy_res = run_controller_stage(
                env, obs, name="bomb_buy", controller=buy,
                max_frames=SHOP_P7_BUY_MAX_FRAMES, assist=assist, on_frame=shot,
                frame_base=topup_res.end_frame,
            )
            snap = read_snapshot(env.get_ram())
            payload["buy"] = {
                **_stage(buy, "bomb_buy"),
                "leftover": _glance(snap),
            }
            save_rgb_png(env.render(), OUT / f"{args.tag}_buy_final.png")
            payload["end_frame"] = buy_res.end_frame

        payload.setdefault("end_frame", topup_res.end_frame)
        payload["assist"] = assist.report()
        payload["pngs"] = shot.n
    finally:
        env.close()
        (OUT / f"{args.tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({
            "wrote": str(OUT / f"{args.tag}.json"),
            "rupees": payload.get("topup", {}).get("leftover", {}).get("rupees"),
            "bombs": payload.get("buy", payload.get("topup", {})).get(
                "leftover", {}
            ).get("bombs"),
            "topup_ok": payload.get("topup", {}).get("success"),
            "buy_ok": payload.get("buy", {}).get("success"),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
