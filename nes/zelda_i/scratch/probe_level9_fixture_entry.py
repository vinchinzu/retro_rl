"""Fixture-only live replay from 0x77 through the left Spectacle Rock.

Prediction for the one allowed trial: the settled 0x05 screenshot has a free
top corridor and a center gap between the paired rocks.  Walk from the east
spawn to ``(120, 93)``, descend the gap, move west below the left rock near
``(72, 173)``, select bombs through the pause menu, press B exactly once, and
hold UP into settled Level 9 room ``0x76``.

This starts from ``Level9OverworldReconFixture`` and is never route evidence.

    QT_QPA_PLATFORM=offscreen uv run python \
      nes/zelda_i/scratch/probe_level9_fixture_entry.py \
      --infinite-life --tag l9_fixture_entry_left_rock_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level9.overworld import Level9FixtureEntryController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEART_PARTIAL,
    ADDR_MAGIC_KEY,
    ADDR_MAX_BOMBS,
    ADDR_RAFT,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist
from zelda_i.screen_glance import leftover_from_snapshot

FROM_STATE = "Level9OverworldReconFixture"
TAG = "l9_fixture_entry_left_rock_v1"


def _glance(env) -> dict[str, int | bool | list[int] | str]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    out = leftover_from_snapshot(snap)
    out.update(
        {
            "level": int(snap.level),
            "screen_hex": f"0x{int(snap.screen):02X}",
            "next_screen": int(snap.next_screen),
            "facing": int(snap.facing),
            "transitioning": bool(snap.transitioning),
            "heart_partial": read_u8(ram, ADDR_HEART_PARTIAL),
            "selected_item": read_u8(ram, ADDR_SELECTED_ITEM),
            "magic_key": read_u8(ram, ADDR_MAGIC_KEY),
            "raft": read_u8(ram, ADDR_RAFT),
            "max_bombs": read_u8(ram, ADDR_MAX_BOMBS),
        }
    )
    return out


def _audited_ram(env) -> dict[str, int]:
    ram = env.get_ram()
    return {
        "bombs": read_u8(ram, ADDR_BOMBS),
        "selected_item": read_u8(ram, ADDR_SELECTED_ITEM),
        "triforce": read_u8(ram, ADDR_TRIFORCE),
        "magic_key": read_u8(ram, ADDR_MAGIC_KEY),
        "raft": read_u8(ram, ADDR_RAFT),
        "max_bombs": read_u8(ram, ADDR_MAX_BOMBS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-state", default=FROM_STATE)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Survival health refill; required for this recon trial.",
    )
    parser.add_argument("--max-frames", type=int, default=12_000)
    args = parser.parse_args()
    if not args.infinite_life:
        parser.error("fixture entry recon requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    controller = Level9FixtureEntryController(max_frames=args.max_frames)
    controller.bind_env(env)
    screenshots: list[str] = []
    samples: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        base = _glance(env)
        base_ram = _audited_ram(env)
        initial_path = RECORDINGS_DIR / f"{args.tag}_initial.png"
        save_rgb_png(env.render(), initial_path)
        screenshots.append(str(initial_path))

        last_phase = controller.phase.name
        last_settled = (int(base["level"]), int(base["screen"]))
        was_transitioning = bool(base["transitioning"])
        rock_saved = False
        bomb_saved = False
        for frame in range(args.max_frames):
            before = read_snapshot(env.get_ram())
            action = controller.step(before)
            env.step(action.action)
            assist.apply_env(env, frame=frame)
            after = read_snapshot(env.get_ram())
            phase = controller.phase.name

            changed = phase != last_phase
            if changed or frame % 250 == 0:
                samples.append(
                    {
                        "frame": frame,
                        "phase": phase,
                        "reason": action.reason,
                        "level": int(after.level),
                        "screen": f"0x{int(after.screen):02X}",
                        "mode": int(after.mode),
                        "xy": [int(after.link_x), int(after.link_y)],
                        "bombs": int(after.bombs),
                        "selected_item": read_u8(env.get_ram(), ADDR_SELECTED_ITEM),
                    }
                )
                last_phase = phase

            transitioning = bool(after.transitioning)
            if transitioning and not was_transitioning:
                path = RECORDINGS_DIR / (
                    f"{args.tag}_transition_f{frame:05d}_"
                    f"l{int(after.level)}_s{int(after.screen):02x}.png"
                )
                save_rgb_png(env.render(), path)
                screenshots.append(str(path))
                transitions.append(
                    {
                        "frame": frame,
                        "kind": "transition_start",
                        "level": int(after.level),
                        "screen": f"0x{int(after.screen):02X}",
                        "mode": int(after.mode),
                        "screenshot": str(path),
                    }
                )
            settled = (int(after.level), int(after.screen))
            if (
                after.mode == PLAY_MODE
                and not transitioning
                and settled != last_settled
            ):
                path = RECORDINGS_DIR / (
                    f"{args.tag}_settled_f{frame:05d}_"
                    f"l{settled[0]}_s{settled[1]:02x}.png"
                )
                save_rgb_png(env.render(), path)
                screenshots.append(str(path))
                transitions.append(
                    {
                        "frame": frame,
                        "kind": "settled",
                        "level": settled[0],
                        "screen": f"0x{settled[1]:02X}",
                        "mode": int(after.mode),
                        "xy": [int(after.link_x), int(after.link_y)],
                        "screenshot": str(path),
                    }
                )
                last_settled = settled
            if (
                not rock_saved
                and after.level == 0
                and after.mode == PLAY_MODE
                and after.screen == 0x05
                and not transitioning
            ):
                path = RECORDINGS_DIR / f"{args.tag}_rock_live_before_bomb.png"
                save_rgb_png(env.render(), path)
                screenshots.append(str(path))
                rock_saved = True
            if (
                not bomb_saved
                and controller.bombs_after is not None
                and controller.blast_wait_frames >= 120
            ):
                path = RECORDINGS_DIR / f"{args.tag}_left_rock_after_bomb.png"
                save_rgb_png(env.render(), path)
                screenshots.append(str(path))
                bomb_saved = True
            was_transitioning = transitioning
            if controller.success or controller.failed:
                break

        final = _glance(env)
        final_ram = _audited_ram(env)
        final_path = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(env.render(), final_path)
        screenshots.append(str(final_path))
        assist_report = assist.report()
        controller_report = controller.report()
        exact_endpoint = (
            controller.success
            and final["level"] == 9
            and final["screen"] == 0x76
            and final["mode"] == PLAY_MODE
            and final["triforce"] == 0xFF
            and final["magic_key"] == 1
            and final["bombs"] == int(base["bombs"]) - 1
            and final["selected_item"] == 1
            and controller.b_presses == 1
            and int(assist_report["deaths"]) == 0
            and int(assist_report["progression_writes"]) == 0
            and int(assist_report["capacity_writes"]) == 0
        )
        payload = {
            "tag": args.tag,
            "fixture": args.from_state,
            "fixture_only": True,
            "natural_entry": False,
            "route_eligible": False,
            "status_claim": False,
            "assist_contract": "nes/zelda_i/docs/ASSIST_CONTRACT.md",
            "hypothesis": (
                "At settled 0x05, use the open top corridor to x=120, descend "
                "the center gap to y=173, stand near (72,173) below the left "
                "rock, pause-select bombs, press B once, then UP to L9 0x76."
            ),
            "prediction": {
                "screen": "0x76",
                "level": 9,
                "mode": PLAY_MODE,
                "bomb_delta": -1,
                "selected_item": 1,
                "do_not_enter": "0x66 Old Man room",
            },
            "success": bool(exact_endpoint),
            "controller": controller_report,
            "base": base,
            "base_ram": base_ram,
            "final": final,
            "final_ram": final_ram,
            "ram_deltas": {
                key: final_ram[key] - value for key, value in base_ram.items()
            },
            "assist": assist_report,
            "runtime_controller_writes": {
                "position": 0,
                "inventory": 0,
                "selected_item": 0,
                "progression": 0,
                "capacity": 0,
                "room": 0,
                "door": 0,
                "object": 0,
            },
            "samples": samples,
            "transitions": transitions,
            "screenshots": screenshots,
        }
        report_path = RECORDINGS_DIR / f"{args.tag}.json"
        report_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({
            "success": payload["success"],
            "controller": controller_report,
            "base": base,
            "final": final,
            "assist": assist_report,
            "report": str(report_path),
            "final_png": str(final_path),
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
