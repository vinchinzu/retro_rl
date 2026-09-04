"""Fixture-live Level 7 entrance/pond -> Level 8 bush-screen geometry.

This is rr-6o7.4 development evidence, not a natural post-Level-7 segment.
The true Triforce-fanfare leave is unmeasured.  This probe loads only the
disclosed ``Level7Entrance`` or ``OW_L7Pond`` fixture, uses normal input plus
Survival health refill, and stops on settled overworld 0x6D before candle use.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l7_exit_to_l8_bush.py \
        --from-state Level7Entrance --tag 20260903_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level8.overworld import (
    L7_POND_TO_LEVEL8_BUSH_HOPS,
    L7_POND_TO_LEVEL8_BUSH_SCREENS,
    Level7PondToLevel8BushController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAX_BOMBS,
    ADDR_ROD,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

L7_ENTRY_ROOM = 0x79
L7_POND_SCREEN = 0x42
L8_BUSH_SCREEN = 0x6D
SETTLE_FRAMES = 120


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "sword": int(snap.sword),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
        "rupees": int(snap.rupees),
        "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "rod": int(read_u8(ram, ADDR_ROD)),
        "bow": int(read_u8(ram, ADDR_BOW)),
        "arrows": int(read_u8(ram, ADDR_ARROWS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "magic_key": int(read_u8(ram, ADDR_MAGIC_KEY)),
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
    }


def _glance(env: Any) -> dict[str, Any]:
    snap = read_snapshot(env.get_ram())
    result = leftover_from_snapshot(snap)
    result.update(
        {
            "level": int(snap.level),
            "screen": int(snap.screen),
            "screen_hex": f"0x{snap.screen:02X}",
            "mode": int(snap.mode),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "facing": int(snap.facing),
            "tile": int(snap.colliding_tile),
            "inventory": _inventory(env),
        }
    )
    return result


def _shot(obs: Any, *, tag: str, frame: int, label: str, env: Any) -> Path:
    snap = read_snapshot(env.get_ram())
    path = RECORDINGS_DIR / (
        f"l7_exit_to_l8_bush_{tag}_{label}_f{frame}_"
        f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
    )
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    save_rgb_png(obs, path)
    return path


def _predictions() -> list[dict[str, Any]]:
    claims = [
        {
            "source": "L7:0x79",
            "direction": "DOWN",
            "expected": "OW:0x42",
            "claim": "natural entrance-room exit, no fanfare claim",
        }
    ]
    for source, hop in zip(
        L7_POND_TO_LEVEL8_BUSH_SCREENS,
        L7_POND_TO_LEVEL8_BUSH_HOPS,
    ):
        claims.append(
            {
                "source": f"OW:0x{source:02X}",
                "direction": hop.direction,
                "expected": f"OW:0x{hop.target:02X}",
                "align_x": hop.align_x,
                "align_y": hop.align_y,
                "y_band": list(hop.y_band) if hop.y_band is not None else None,
            }
        )
    return claims


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        default_state="Level7Entrance",
        default_tag="20260903_v1",
        default_trials=1,
    )
    parser.add_argument("--settle-frames", type=int, default=SETTLE_FRAMES)
    args = parser.parse_args()
    if args.trials != 1:
        raise SystemExit("rr-6o7.4 probes exactly one predicted route per run")

    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    controller = Level7PondToLevel8BushController()
    frame = 0
    obs: Any = None
    shots: list[str] = []
    transitions: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "bead": "rr-6o7.4",
        "from_state": args.from_state,
        "infinite_life": bool(args.infinite_life),
        "natural_entry": False,
        "route_eligible": False,
        "status_claim": False,
        "predictions": _predictions(),
        "writes": {
            "controller": 0,
            "inventory": 0,
            "position": 0,
            "room_screen_door": 0,
            "selected_item": 0,
        },
    }
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        if assist is not None:
            assist.apply_env(env, frame=frame)
        start = _glance(env)
        payload["start"] = start
        shots.append(
            str(_shot(obs, tag=args.tag, frame=frame, label="start", env=env))
        )
        last_screen_key = (start["level"], start["screen"])

        for _ in range(controller.max_frames):
            before = _glance(env)
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap)
            obs, *_ = env.step(action.action)
            frame += 1
            if assist is not None:
                assist.apply_env(env, frame=frame)
            after = _glance(env)
            screen_key = (after["level"], after["screen"])
            if screen_key != last_screen_key:
                label = f"transition_{len(transitions):02d}"
                shot = _shot(obs, tag=args.tag, frame=frame, label=label, env=env)
                shots.append(str(shot))
                transitions.append(
                    {
                        "frame": frame,
                        "reason": action.reason,
                        "from": before,
                        "to": after,
                        "screenshot": str(shot),
                    }
                )
                last_screen_key = screen_key
            if frame % 250 == 0 or controller.stuck in {51, 100, 200}:
                samples.append(
                    {
                        "frame": frame,
                        "reason": action.reason,
                        "glance": after,
                        "hop_index": controller.hop_index,
                        "stuck": controller.stuck,
                    }
                )
            if controller.success or controller.failed:
                break

        first_destination = _glance(env)
        payload["first_destination"] = first_destination
        if controller.success:
            for _ in range(max(0, int(args.settle_frames))):
                obs, *_ = env.step(nes_idle_action())
                frame += 1
                if assist is not None:
                    assist.apply_env(env, frame=frame)
        final = _glance(env)
        final_shot = _shot(obs, tag=args.tag, frame=frame, label="final", env=env)
        shots.append(str(final_shot))
        assist_report = assist.report() if assist is not None else {}
        deaths = int(assist_report.get("deaths", 0))
        progression_writes = int(assist_report.get("progression_writes", 0))
        capacity_writes = int(assist_report.get("capacity_writes", 0))
        exact_stop = (
            final["level"] == 0
            and final["mode"] == PLAY_MODE
            and final["screen"] == L8_BUSH_SCREEN
        )
        payload.update(
            {
                "controller": controller.report(),
                "frames": frame,
                "settle_frames": int(args.settle_frames),
                "transitions": transitions,
                "samples": samples[-96:],
                "screenshots": shots,
                "final": final,
                "assist": assist_report,
                "deaths": deaths,
                "progression_writes": progression_writes,
                "capacity_writes": capacity_writes,
                "runtime_integrity": {
                    "initial_fixture_loads": 1,
                    "state_loads_after_start": 0,
                    "controller_writes": 0,
                    "position_writes": 0,
                    "inventory_writes": 0,
                    "room_screen_door_writes": 0,
                    "selected_item_writes": 0,
                },
                "success": bool(controller.success)
                and exact_stop
                and deaths == 0
                and progression_writes == 0
                and capacity_writes == 0,
            }
        )
        if not payload["success"]:
            payload["failed"] = (
                controller.notes[-1]
                if controller.notes
                else "endpoint_or_integrity_miss"
            )
        report = write_report("l7_exit_to_l8_bush_fixture", payload, tag=args.tag)
        print(report)
        print(f"success={payload['success']} failed={payload.get('failed')}")
        print(f"transitions={[t['to']['screen_hex'] for t in transitions]}")
        print(f"final={final}")
        print(f"assist={assist_report}")
    finally:
        env.close()

    raise SystemExit(0 if payload.get("success") else 1)


if __name__ == "__main__":
    main()
