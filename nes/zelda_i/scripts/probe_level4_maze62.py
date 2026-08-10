"""Live probe from Level4Room62Cleared: dark-maze coverage, compass bit, exits.

rr-9so0 (tip residual after rr-2ysf). Assisted OK for recon; pure later.

Examples::

    uv run python nes/zelda_i/scripts/probe_level4_maze62.py
    uv run python nes/zelda_i/scripts/probe_level4_maze62.py --from-state Level4Room62Cleared --tag l4_maze
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _NES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, PLAY_MODE, read_snapshot, read_u8

LEVEL4_COMPASS_BIT = 0x08
ROOM = 0x62
DIRS = ("LEFT", "RIGHT", "UP", "DOWN")


def _cell(x: int, y: int, q: int = 8) -> tuple[int, int]:
    return (x // q * q, y // q * q)


def _fields(snap, env) -> dict:
    ram = env.get_ram()
    objs = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12):
            continue
        if o.type_id in (0, 0xFF):
            continue
        objs.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
            }
        )
    return {
        "mode": snap.mode,
        "level": snap.level,
        "sc": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "compass": snap.compass,
        "compass_l4": bool(snap.compass & LEVEL4_COMPASS_BIT),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "room_item_id": snap.room_item_id,
        "room_item": room_item_name(snap.room_item_id),
        "doors": int(snap.cur_opened_doors),
        "open_doorway_mask": int(snap.open_doorway_mask),
        "room_all_dead": snap.room_all_dead,
        "objects": objs,
        "raft": int(read_u8(ram, 0x0660)),
        "triforce": snap.triforce,
    }


def _step(env, assist, button: str | None, frame: int) -> int:
    action = nes_idle_action() if not button else nes_action(button)
    env.step(action)
    frame += 1
    if assist is not None:
        assist.apply_env(env, frame=frame)
    return frame


def _settle(env, assist, frame: int, n: int = 20) -> int:
    for _ in range(n):
        frame = _step(env, assist, None, frame)
    return frame


def run_probe(*, start_state: str, tag: str, seed: int, walk_steps: int) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    random.seed(seed)

    report: dict = {
        "bead": "rr-9so0",
        "start_state": start_state,
        "tag": tag,
        "cells": {},
        "cell_list": [],
        "exits": [],
        "compass_events": [],
        "macros": [],
        "walk": {},
    }

    try:
        env.reset()
        frame = _settle(env, assist, 0)
        snap = read_snapshot(env.get_ram())
        report["start"] = _fields(snap, env)
        print("START", json.dumps(report["start"], indent=2)[:600])

        # Save RGB at start
        try:
            obs = env.render()
            if obs is not None:
                save_rgb_png(RECORDINGS_DIR / f"{tag}_start.png", obs)
        except Exception as exc:  # noqa: BLE001
            report["start_png_error"] = str(exc)

        visited: set[tuple[int, int]] = set()

        def mark(s) -> None:
            if s.screen == ROOM and s.mode == PLAY_MODE:
                c = _cell(s.link_x, s.link_y)
                visited.add(c)
                report["cells"][f"{c[0]},{c[1]}"] = True

        mark(snap)

        # --- Macros: scripted corridor paths through dark maze ---
        macros: list[list[str]] = [
            # notes: vestibule opens DOWN then RIGHT
            ["DOWN"] * 48 + ["RIGHT"] * 96,
            ["DOWN"] * 32 + ["RIGHT"] * 64 + ["UP"] * 32 + ["RIGHT"] * 64,
            ["DOWN"] * 64 + ["RIGHT"] * 48 + ["DOWN"] * 32 + ["RIGHT"] * 80,
            ["RIGHT"] * 24 + ["DOWN"] * 64 + ["RIGHT"] * 96 + ["UP"] * 48,
            ["UP"] * 40 + ["RIGHT"] * 96 + ["DOWN"] * 40 + ["RIGHT"] * 64,
            # zigzag cover
            sum(
                (
                    ["RIGHT"] * 100 + ["DOWN"] * 12 + ["LEFT"] * 100 + ["DOWN"] * 12
                    for _ in range(8)
                ),
                [],
            ),
            sum(
                (
                    ["LEFT"] * 100 + ["UP"] * 12 + ["RIGHT"] * 100 + ["UP"] * 12
                    for _ in range(8)
                ),
                [],
            ),
            # door pushes
            ["RIGHT"] * 220,
            ["UP"] * 220,
            ["DOWN"] * 220,
            ["LEFT"] * 220,
            # SE then N then E (common maze exit pattern)
            ["DOWN"] * 80
            + ["RIGHT"] * 120
            + ["UP"] * 100
            + ["RIGHT"] * 80
            + ["DOWN"] * 40
            + ["RIGHT"] * 60,
            # center sweep
            ["RIGHT"] * 40
            + ["UP"] * 20
            + ["RIGHT"] * 40
            + ["DOWN"] * 40
            + ["RIGHT"] * 60
            + ["UP"] * 60
            + ["LEFT"] * 40
            + ["UP"] * 40
            + ["RIGHT"] * 80,
            # hug south then east door band y141
            ["DOWN"] * 100 + ["RIGHT"] * 180,
            # north band y93
            ["UP"] * 100 + ["RIGHT"] * 180,
            # mid y141 east
            ["RIGHT"] * 20
            + (["UP"] * 2 + ["RIGHT"] * 4) * 10
            + ["RIGHT"] * 100,
        ]

        for mi, macro in enumerate(macros):
            env.reset()
            frame = _settle(env, assist, 0)
            got = False
            exit_info = None
            for button in macro:
                s_before = read_snapshot(env.get_ram())
                c0 = s_before.compass
                sc0 = s_before.screen
                frame = _step(env, assist, button, frame)
                s = read_snapshot(env.get_ram())
                mark(s)
                if (s.compass & LEVEL4_COMPASS_BIT) and not (c0 & LEVEL4_COMPASS_BIT):
                    ev = _fields(s, env) | {"event": "got_compass", "macro": mi, "frame": frame}
                    report["compass_events"].append(ev)
                    print("GOT_COMPASS macro", mi, ev)
                    try:
                        save_rgb_png(
                            RECORDINGS_DIR / f"{tag}_compass_m{mi}.png",
                            env.render(),
                        )
                    except Exception:
                        pass
                    got = True
                    break
                if (
                    s.screen != sc0
                    and s.level == 4
                    and s.mode == PLAY_MODE
                    and not s.transitioning
                ):
                    exit_info = _fields(s, env) | {
                        "from": f"0x{sc0:02x}",
                        "via": button,
                        "macro": mi,
                        "frame": frame,
                    }
                    report["exits"].append(exit_info)
                    print("EXIT macro", mi, exit_info)
                    try:
                        save_rgb_png(
                            RECORDINGS_DIR / f"{tag}_exit_m{mi}_0x{s.screen:02x}.png",
                            env.render(),
                        )
                    except Exception:
                        pass
                    # Explore new room briefly (look for ladder / new items)
                    for _ in range(200):
                        frame = _step(env, assist, random.choice(DIRS), frame)
                        s2 = read_snapshot(env.get_ram())
                        if s2.compass & LEVEL4_COMPASS_BIT and not any(
                            e.get("event") == "got_compass" for e in report["compass_events"]
                        ):
                            report["compass_events"].append(
                                _fields(s2, env)
                                | {"event": "got_compass", "after_exit_macro": mi}
                            )
                            print("GOT_COMPASS after exit", _fields(s2, env))
                        if int(read_u8(env.get_ram(), ADDR_LADDER)) > 0:
                            report["ladder_events"] = report.get("ladder_events", [])
                            report["ladder_events"].append(
                                _fields(s2, env) | {"event": "got_ladder", "macro": mi}
                            )
                            print("GOT_LADDER", _fields(s2, env))
                            break
                    break
            final = _fields(read_snapshot(env.get_ram()), env)
            report["macros"].append(
                {"macro": mi, "got_compass": got, "exit": exit_info, "final": final}
            )

        # --- Long random walk from cleared (coverage + surprises) ---
        env.reset()
        frame = _settle(env, assist, 0)
        cur = "DOWN"
        walk_exits = []
        for step_i in range(walk_steps):
            s0 = read_snapshot(env.get_ram())
            x0, y0, c0, sc0 = s0.link_x, s0.link_y, s0.compass, s0.screen
            frame = _step(env, assist, cur, frame)
            s = read_snapshot(env.get_ram())
            mark(s)
            if abs(s.link_x - x0) + abs(s.link_y - y0) < 1:
                cur = random.choice([d for d in DIRS if d != cur])
            elif step_i % 90 == 0:
                cur = random.choice(DIRS)
            if (s.compass & LEVEL4_COMPASS_BIT) and not (c0 & LEVEL4_COMPASS_BIT):
                report["compass_events"].append(
                    _fields(s, env)
                    | {"event": "got_compass", "walk_step": step_i, "frame": frame}
                )
                print("GOT_COMPASS walk", step_i, _fields(s, env))
                try:
                    save_rgb_png(RECORDINGS_DIR / f"{tag}_compass_walk.png", env.render())
                except Exception:
                    pass
            if (
                s.screen != sc0
                and s.level == 4
                and s.mode == PLAY_MODE
                and not s.transitioning
            ):
                info = _fields(s, env) | {
                    "from": f"0x{sc0:02x}",
                    "via": cur,
                    "walk_step": step_i,
                }
                walk_exits.append(info)
                report["exits"].append(info)
                print("EXIT walk", step_i, info)
                # continue into new room; don't reset (may chain)
                cur = random.choice(DIRS)
            if s.mode == 17:
                env.reset()
                frame = _settle(env, assist, 0)
                cur = "DOWN"
            if int(read_u8(env.get_ram(), ADDR_LADDER)) > 0:
                report["ladder_events"] = report.get("ladder_events", [])
                report["ladder_events"].append(
                    _fields(s, env) | {"event": "got_ladder", "walk_step": step_i}
                )
                print("GOT_LADDER walk", _fields(s, env))
                break

        report["walk"] = {
            "steps": walk_steps,
            "exits": walk_exits,
            "final": _fields(read_snapshot(env.get_ram()), env),
        }

        # --- Targeted grid hunt: from start, try walk-to each coarse target ---
        # Use simple seek with timeout; detect compass change.
        targets = [
            (x, y)
            for y in range(88, 200, 16)
            for x in range(32, 220, 16)
        ]
        compass_hits = []
        for ti, (tx, ty) in enumerate(targets):
            env.reset()
            frame = _settle(env, assist, 0)
            for _ in range(180):
                s = read_snapshot(env.get_ram())
                if s.screen != ROOM:
                    break
                c0 = s.compass
                dx, dy = tx - s.link_x, ty - s.link_y
                if abs(dx) <= 4 and abs(dy) <= 4:
                    break
                # y-first then x
                if abs(dy) > 4:
                    btn = "DOWN" if dy > 0 else "UP"
                else:
                    btn = "RIGHT" if dx > 0 else "LEFT"
                frame = _step(env, assist, btn, frame)
                s2 = read_snapshot(env.get_ram())
                mark(s2)
                if (s2.compass & LEVEL4_COMPASS_BIT) and not (c0 & LEVEL4_COMPASS_BIT):
                    hit = _fields(s2, env) | {
                        "event": "got_compass",
                        "target": [tx, ty],
                        "ti": ti,
                    }
                    compass_hits.append(hit)
                    report["compass_events"].append(hit)
                    print("GOT_COMPASS target", tx, ty, hit)
                    try:
                        save_rgb_png(
                            RECORDINGS_DIR / f"{tag}_compass_t{tx}_{ty}.png",
                            env.render(),
                        )
                    except Exception:
                        pass
                    break
            if compass_hits:
                break  # one hit is enough to know location band

        report["target_hunt_hits"] = compass_hits
        report["cell_list"] = sorted(visited)
        report["n_cells"] = len(visited)
        report["got_compass"] = any(
            e.get("event") == "got_compass" for e in report["compass_events"]
        ) or any(e.get("compass_l4") for e in report["compass_events"])
        # unique exits
        seen = set()
        uniq = []
        for e in report["exits"]:
            k = (e.get("sc"), e.get("from"), e.get("via"))
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        report["unique_exits"] = uniq
        report["final"] = _fields(read_snapshot(env.get_ram()), env)

        out = RECORDINGS_DIR / f"{tag}_probe.json"
        write_json_report(out, report)
        print(
            "WROTE",
            out,
            "n_cells",
            report["n_cells"],
            "got_compass",
            report["got_compass"],
            "n_exits",
            len(uniq),
        )
        print("cells", report["cell_list"])
        print("exits", uniq)
        return report
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default="Level4Room62Cleared")
    ap.add_argument("--tag", default="l4_maze62")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--walk-steps", type=int, default=6000)
    args = ap.parse_args()
    run_probe(
        start_state=args.from_state,
        tag=args.tag,
        seed=args.seed,
        walk_steps=args.walk_steps,
    )


if __name__ == "__main__":
    main()
