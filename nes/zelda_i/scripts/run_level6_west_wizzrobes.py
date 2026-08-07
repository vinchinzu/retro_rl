"""Isolated pure: Level 6 post-east-key → west 0x78 clear (no Old Man waste).

Default start: ``Level6EastKey`` (0x7a, keys≥1). Path::

    0x7a -LEFT free-> 0x79
    0x79 -LEFT key (fire-bypass y≈157→141) -> 0x78
    clear 5× type 0x24

Trap: do **not** UP from 0x7a (Old Man 0x6a wastes the key).

Stop: ``level6_room_78_clear_success`` (room 0x78, no live 0x24).

Examples::

    uv run python nes/zelda_i/scripts/run_level6_west_wizzrobes.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level6_west_wizzrobes.py --infinite-life --save-state --trials 2
    uv run python nes/zelda_i/scripts/run_level6_west_wizzrobes.py --from-state L6Room_79_keys1 --infinite-life
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level6_dungeon import (
    ROOM_78_SPEC,
    Level6WestWizzrobeController,
    level6_room_78_clear_success,
    make_west_wizzrobe_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    Level6WestKeyDoorController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from retro_harness.nes import nes_action


def _run_free_left_to_79(env, *, assist=None, max_frames: int = 800):
    """0x7a west free door → 0x79. Align y≈141, push LEFT."""
    notes: list[str] = []
    for frame in range(max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == 6
            and snap.screen == LEVEL6_ENTRY_ROOM
            and snap.mode == PLAY_MODE
        ):
            notes.append(f"arrived_79_f{frame}")
            return True, notes, frame
        if snap.mode == 17:
            notes.append("death")
            return False, notes, frame
        # Scroll / settle into 0x79 (mode 6/7 or mid-transition screen).
        if (
            snap.transitioning
            or snap.mode in (2, 3, 4, 6, 7)
            or (
                snap.screen == LEVEL6_ENTRY_ROOM
                and snap.mode != PLAY_MODE
            )
        ):
            env.step(nes_action("LEFT"))
            continue
        if snap.screen != LEVEL6_EAST_KEY_ROOM:
            notes.append(f"unexpected_0x{snap.screen:02x}_m{snap.mode}")
            return False, notes, frame
        if abs(snap.link_y - 141) > 4:
            btn = "DOWN" if snap.link_y < 141 else "UP"
            env.step(nes_action(btn))
        else:
            env.step(nes_action("LEFT"))
    notes.append("timeout")
    return False, notes, max_frames


def _run_west_key(env, *, assist=None, max_frames: int = 5000):
    controller = Level6WestKeyDoorController(max_frames=max_frames)
    for frame in range(max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame)
        action = controller.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        if controller.success or controller.phase.name == "FAILED":
            break
    return controller


def run_once(
    *,
    tag: str = "level6_west_wizzrobes",
    save_checkpoint: bool = False,
    start_state: str = "Level6EastKey",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    west_ctl: Level6WestKeyDoorController | None = None
    fight_ctl: Level6WestWizzrobeController = make_west_wizzrobe_controller()
    free_left_notes: list[str] = []
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        prefix_ok = True
        if (
            entry.screen == LEVEL6_EAST_KEY_ROOM
            and entry.mode == PLAY_MODE
            and entry.keys >= 1
        ):
            ok79, free_left_notes, _ = _run_free_left_to_79(env, assist=assist)
            for _ in range(40):
                if assist is not None:
                    assist.apply_env(env, frame=0)
                obs, *_ = env.step(nes_idle_action())
            mid = read_snapshot(env.get_ram())
            prefix_ok = (
                ok79
                and mid.level == 6
                and mid.screen == LEVEL6_ENTRY_ROOM
                and mid.mode == PLAY_MODE
                and mid.keys >= 1
            )
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "prefix_ok": False,
                    "start_state": start_state,
                    "error": "failed_free_left_7a_to_79",
                    "entry": {
                        "room": entry.screen,
                        "x": entry.link_x,
                        "y": entry.link_y,
                        "keys": entry.keys,
                    },
                    "free_left": free_left_notes,
                    "final": {
                        "room": mid.screen,
                        "x": mid.link_x,
                        "y": mid.link_y,
                        "keys": mid.keys,
                    },
                    "screenshot": str(screenshot),
                    "assist": assist.report() if assist else None,
                }
            west_ctl = _run_west_key(env, assist=assist)
            mid2 = read_snapshot(env.get_ram())
            prefix_ok = (
                west_ctl.success
                and mid2.screen == LEVEL6_WEST_WIZZROBE_ROOM
                and mid2.mode == PLAY_MODE
            )
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "prefix_ok": False,
                    "start_state": start_state,
                    "error": "failed_key_left_79_to_78",
                    "entry": {
                        "room": entry.screen,
                        "keys": entry.keys,
                        "x": entry.link_x,
                        "y": entry.link_y,
                    },
                    "free_left": free_left_notes,
                    "west_door": west_ctl.report(),
                    "final": {
                        "room": mid2.screen,
                        "x": mid2.link_x,
                        "y": mid2.link_y,
                        "keys": mid2.keys,
                    },
                    "screenshot": str(screenshot),
                    "assist": assist.report() if assist else None,
                }
        elif (
            entry.screen == LEVEL6_ENTRY_ROOM
            and entry.mode == PLAY_MODE
            and entry.keys >= 1
        ):
            west_ctl = _run_west_key(env, assist=assist)
            mid2 = read_snapshot(env.get_ram())
            prefix_ok = (
                west_ctl.success
                and mid2.screen == LEVEL6_WEST_WIZZROBE_ROOM
            )
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "prefix_ok": False,
                    "start_state": start_state,
                    "error": "failed_key_left_from_79",
                    "west_door": west_ctl.report() if west_ctl else None,
                    "final": {
                        "room": mid2.screen,
                        "keys": mid2.keys,
                        "x": mid2.link_x,
                        "y": mid2.link_y,
                    },
                    "screenshot": str(screenshot),
                    "assist": assist.report() if assist else None,
                }
        elif entry.screen != LEVEL6_WEST_WIZZROBE_ROOM:
            return {
                "ok": False,
                "prefix_ok": False,
                "start_state": start_state,
                "error": f"unexpected_start_room_0x{entry.screen:02x}",
                "entry": {
                    "room": entry.screen,
                    "x": entry.link_x,
                    "y": entry.link_y,
                    "keys": entry.keys,
                },
            }

        if prefix_ok:
            obs = None
            for frame in range(ROOM_78_SPEC.max_frames):
                if assist is not None:
                    assist.apply_env(env, frame=frame)
                action = fight_ctl.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(action.action)
                if fight_ctl.success or fight_ctl.phase is DungeonPhase.FAILED:
                    break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level6_room_78_clear_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(
                env, GAME_DIR, GAME, "Level6WestWizzrobes"
            )
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level6_west_wizzrobes",
                        "start_state": start_state,
                        "infinite_life": infinite_life,
                    },
                    selected_trial=fight_ctl.report(),
                    natural_entry=False,
                )
            )

        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": False,
            "start_state": start_state,
            "prefix_ok": prefix_ok,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
            },
            "free_left": free_left_notes,
            "west_door": west_ctl.report() if west_ctl else None,
            "controller": fight_ctl.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "open_doorway_mask": snap.open_doorway_mask,
                "live_wizzrobes": len(ROOM_78_SPEC.live_enemies(snap)),
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
            "assist": assist.report() if assist else None,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level6EastKey")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    args = parser.parse_args(argv)

    track = "assisted" if args.infinite_life else "clean"
    reports = [
        run_once(
            tag=f"level6_west_wizzrobes_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} keys={final.get('keys')} "
            f"live={final.get('live_wizzrobes')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"frames={report.get('controller', {}).get('frames')} "
            f"phase={report.get('controller', {}).get('phase')} "
            f"doors={final.get('cur_opened_doors')} "
            f"mask={final.get('open_doorway_mask')}"
        )

    stem = (
        "level6_west_wizzrobes_assisted_isolated"
        if args.infinite_life
        else "level6_west_wizzrobes_isolated"
    )
    output = RECORDINGS_DIR / f"{stem}.json"
    write_json_report(
        output,
        {
            "segment": "level6_west_wizzrobes",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "track": track,
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level6_room_78_clear_success",
            "spec_id": ROOM_78_SPEC.spec_id,
            "entry_policy": (
                "Level6EastKey → LEFT free 0x79 → "
                "Level6WestKeyDoorController y≈157→x32→y141 LEFT key → 0x78"
            ),
            "trap": "Do not UP from 0x7a (Old Man 0x6a wastes key)",
            "next": "0x78 UP → 0x68 compass Zols (5×0x13, RoomItemId 0x16)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
