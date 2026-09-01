"""Scratch merchant hunt for first-quest wooden arrows. Not a route claim.

    QT_QPA_PLATFORM=offscreen uv run python \
      nes/zelda_i/scratch/probe_arrow_shop.py --dump
    QT_QPA_PLATFORM=offscreen uv run python \
      nes/zelda_i/scratch/probe_arrow_shop.py --hunt 4A --recon-rupees 200

Recon rupee poke is labeled recon. Do not treat as a route claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_MAGIC_SHIELD,
    ADDR_RUPEES,
    CAVE_MODE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

SCRATCH_DIR = Path(__file__).resolve().parent
DUMP_STATES = (
    "PostSwordStart",
    "Level1ExitOverworld",
    "BFS_4A",
    "At4A",
    "BFS_4B",
    "BFS_6B",
    "BFS_5E",
    "CandleShop5E",
)
# Open first-quest arrow shops (source): E-5 0x44, F-3 0x25, K-5 0x4A, P-7 0x6F.
# 0x5E is live candle (no arrows). 0x6B gathering hyp is not live.
HUNT_STATES = {
    "4A": "BFS_4A",
    "4A_at": "At4A",
    "6B": "BFS_6B",
    "5E": "CandleShop5E",
}
PEDESTALS = (("left", 72), ("mid", 120), ("right", 152))
BUY_Y = 149
# 0x4A live: cave enter mode-16 @ (176,77). Same stairs as CandleShop5E.


def _poke_u8(env: Any, addr: int, value: int) -> str:
    """Direct RAM write in this scratch probe. Not dungeon/ops, not a route helper."""
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "assign"):
            mem.assign(int(addr), "|u1", int(value) & 0xFF)
            return "memory.assign"
        if hasattr(mem, "set_byte"):
            mem.set_byte(int(addr), int(value) & 0xFF)
            return "memory.set_byte"
    except Exception as exc:
        return f"poke_fail={exc!r}"
    em = getattr(env.unwrapped, "em", None)
    if em is not None and hasattr(em, "set_bytes"):
        em.set_bytes(int(addr), bytes([int(value) & 0xFF]))
        return "em.set_bytes"
    return "no_memory_write"


def _objects(snap) -> list[dict[str, int]]:
    return [
        {
            "slot": obj.slot,
            "type": obj.type_id,
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
        }
        for obj in snap.objects
        if obj.type_id
    ]


def leftover_row(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "mode": int(snap.mode),
        "level": int(snap.level),
        "screen": int(snap.screen),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "rupees": int(snap.rupees),
        "sword": int(snap.sword),
        "bow": int(snap.bow),
        "arrows": int(read_u8(ram, ADDR_ARROWS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "shield": int(read_u8(ram, ADDR_MAGIC_SHIELD)),
        "triforce": int(snap.triforce),
        "tile": int(snap.colliding_tile),
        "objects": _objects(snap),
    }


def dump_states(tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out = RECORDINGS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name in DUMP_STATES:
        env = make_env(GAME, name, GAME_DIR, render_mode="rgb_array")
        try:
            obs, _ = reset_obs(env)
            row = leftover_row(env)
            row["state"] = name
            png = out / f"{tag}_{name}.png"
            save_rgb_png(obs, png)
            row["png"] = str(png)
            rows.append(row)
            print(
                f"{name}: screen=0x{row['screen']:02x} mode={row['mode']} "
                f"xy=({row['x']},{row['y']}) rupees={row['rupees']} "
                f"sword={row['sword']} bow={row['bow']} arrows={row['arrows']} "
                f"bombs={row['bombs']} keys={row['keys']} candle={row['candle']}"
            )
        finally:
            env.close()
    path = SCRATCH_DIR / f"{tag}_dump.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {path}")
    return rows


def _step(env: Any, btn: str | None, n: int = 1) -> Any:
    obs = None
    action = nes_idle_action() if btn is None else nes_action(btn)
    for _ in range(n):
        obs, *_ = env.step(action)
    return obs


def hunt_cave(
    *,
    state: str,
    tag: str,
    recon_rupees: int | None,
    max_up: int = 80,
    x_lo: int = 16,
    x_hi: int = 240,
    x_step: int = 8,
) -> dict[str, Any]:
    """Sweep x along the north band and hold UP until mode 11."""
    hits: list[dict[str, Any]] = []
    for x in range(x_lo, x_hi + 1, x_step):
        env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
        try:
            obs, _ = reset_obs(env)
            start = leftover_row(env)
            if recon_rupees is not None and start["rupees"] < recon_rupees:
                _poke_u8(env, ADDR_RUPEES, recon_rupees)
            # North-gap leftovers (BFS_4A y=61) UP-exit to 0x3A. Drop into
            # the clearing first, then align, then UP into the NE cave.
            for _ in range(240):
                snap = read_snapshot(env.get_ram())
                if snap.mode == CAVE_MODE:
                    break
                if snap.link_y < 90:
                    btn = "DOWN"
                elif abs(snap.link_x - x) > 3:
                    btn = "RIGHT" if snap.link_x < x else "LEFT"
                else:
                    btn = "UP"
                obs, *_ = env.step(nes_action(btn))
            for _ in range(max_up):
                snap = read_snapshot(env.get_ram())
                if snap.mode in (CAVE_MODE, 16):
                    break
                obs, *_ = env.step(nes_action("UP"))
            row = leftover_row(env)
            row["aim_x"] = x
            row["start"] = {
                "screen": start["screen"],
                "mode": start["mode"],
                "xy": [start["x"], start["y"]],
                "rupees": start["rupees"],
            }
            if row["mode"] in (CAVE_MODE, 16) or row["level"] != 0:
                png = RECORDINGS_DIR / f"{tag}_{state}_x{x:03d}_mode{row['mode']}.png"
                save_rgb_png(obs, png)
                row["png"] = str(png)
                hits.append(row)
                print(
                    f"HIT aim_x={x} screen=0x{row['screen']:02x} mode={row['mode']} "
                    f"xy=({row['x']},{row['y']}) arrows={row['arrows']} "
                    f"objs={row['objects']}"
                )
                break
        finally:
            env.close()
    return {"state": state, "hits": hits}


def try_buy(
    *,
    state: str,
    tag: str,
    recon_rupees: int,
    cave_x: int,
    pedestals: tuple[tuple[str, int], ...] = (("right", 152),),
    buy_y: int = BUY_Y,
) -> dict[str, Any]:
    """Enter cave at cave_x, settle, touch pedestals, watch ADDR_ARROWS."""
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        start = leftover_row(env)
        poke = None
        if start["rupees"] < recon_rupees:
            poke = _poke_u8(env, ADDR_RUPEES, recon_rupees)
        before_enter = leftover_row(env)
        for _ in range(280):
            snap = read_snapshot(env.get_ram())
            if snap.mode == CAVE_MODE:
                break
            if snap.link_y < 90:
                btn = "DOWN"
            elif abs(snap.link_x - cave_x) > 3:
                btn = "RIGHT" if snap.link_x < cave_x else "LEFT"
            else:
                btn = "UP"
            obs, *_ = env.step(nes_action(btn))
        # Mode-16 transition then cave-bottom settle (CandleShop dialog idle).
        for _ in range(160):
            snap = read_snapshot(env.get_ram())
            if snap.mode == CAVE_MODE and snap.link_y > 200:
                obs, *_ = env.step(nes_idle_action())
            elif snap.mode == CAVE_MODE:
                obs, *_ = env.step(nes_idle_action())
            else:
                obs, *_ = env.step(nes_action("UP"))
        entered = leftover_row(env)
        png = RECORDINGS_DIR / f"{tag}_{state}_cave.png"
        save_rgb_png(obs, png)
        arrows0 = entered["arrows"]
        rupees0 = entered["rupees"]
        buys: list[dict[str, Any]] = []
        for name, px in pedestals:
            # Stairs are a center column; y=213 RIGHT is walled at x≈128.
            # Climb to buy_y, RIGHT under the bomb (mid x=120 y≈137), then UP.
            for _ in range(500):
                snap = read_snapshot(env.get_ram())
                if snap.mode != CAVE_MODE:
                    obs, *_ = env.step(nes_action("UP"))
                    continue
                if snap.link_y > buy_y + 1:
                    btn = "UP"
                elif abs(snap.link_x - px) > 2:
                    btn = "RIGHT" if snap.link_x < px else "LEFT"
                else:
                    btn = "UP"
                obs, *_ = env.step(nes_action(btn))
                now = leftover_row(env)
                if now["arrows"] > arrows0:
                    break
            after = leftover_row(env)
            png_i = RECORDINGS_DIR / f"{tag}_{state}_{name}.png"
            save_rgb_png(obs, png_i)
            delta = {
                "pedestal": name,
                "aim_x": px,
                "arrows": after["arrows"],
                "rupees": after["rupees"],
                "rupees_before": rupees0,
                "bombs": after["bombs"],
                "keys": after["keys"],
                "candle": after["candle"],
                "shield": after["shield"],
                "xy": [after["x"], after["y"]],
                "mode": after["mode"],
                "objects": after["objects"],
                "png": str(png_i),
            }
            buys.append(delta)
            print(f"pedestal {name}: arrows={delta['arrows']} rupees={delta['rupees']} xy={delta['xy']} mode={delta['mode']}")
            if after["arrows"] > arrows0:
                break
        return {
            "recon": True,
            "recon_rupee_poke": poke,
            "cave_x": cave_x,
            "start": start,
            "before_enter": before_enter,
            "entered": entered,
            "cave_png": str(png),
            "buys": buys,
            "arrows_0_to_1": any(b["arrows"] > arrows0 for b in buys),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--hunt", choices=sorted(HUNT_STATES), default=None)
    parser.add_argument("--buy", action="store_true")
    parser.add_argument("--cave-x", type=int, default=176)
    parser.add_argument("--buy-y", type=int, default=165)
    parser.add_argument("--x-lo", type=int, default=16)
    parser.add_argument("--x-hi", type=int, default=240)
    parser.add_argument("--recon-rupees", type=int, default=0)
    parser.add_argument("--tag", default="arrow_shop_probe")
    args = parser.parse_args(argv)
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.dump or args.hunt is None and not args.buy:
        dump_states(args.tag)
        if args.hunt is None and not args.buy:
            return 0
    recon = args.recon_rupees if args.recon_rupees > 0 else None
    if args.hunt:
        hunt = hunt_cave(
            state=HUNT_STATES[args.hunt],
            tag=args.tag,
            recon_rupees=recon,
            x_lo=args.x_lo,
            x_hi=args.x_hi,
        )
        path = SCRATCH_DIR / f"{args.tag}_hunt_{args.hunt}.json"
        path.write_text(json.dumps(hunt, indent=2))
        print(f"wrote {path} hits={len(hunt['hits'])}")
        if args.buy and hunt["hits"]:
            cave_x = int(hunt["hits"][0].get("x") or args.cave_x)
            bought = try_buy(
                state=HUNT_STATES[args.hunt],
                tag=args.tag,
                recon_rupees=args.recon_rupees or 200,
                cave_x=cave_x,
                pedestals=(("right", 152), ("right168", 168)),
                buy_y=args.buy_y,
            )
            path = SCRATCH_DIR / f"{args.tag}_buy_{args.hunt}.json"
            path.write_text(json.dumps(bought, indent=2))
            print(f"wrote {path} arrows_0_to_1={bought.get('arrows_0_to_1')}")
        return 0 if hunt["hits"] else 1
    if args.buy:
        bought = try_buy(
            state=HUNT_STATES.get(args.hunt or "4A", "BFS_4A"),
            tag=args.tag,
            recon_rupees=args.recon_rupees or 200,
            cave_x=args.cave_x,
            pedestals=(("right", 152), ("right168", 168)),
            buy_y=args.buy_y,
        )
        path = SCRATCH_DIR / f"{args.tag}_buy.json"
        path.write_text(json.dumps(bought, indent=2))
        print(f"wrote {path} arrows_0_to_1={bought.get('arrows_0_to_1')}")
        return 0 if bought.get("arrows_0_to_1") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
