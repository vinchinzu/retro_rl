"""Isolated pure: clear Level 2 Goriya room 0x5e (5× type 0x06).

Default start: ``Level2_5F`` (0x5f, keys≥1). Prefix:

1. Align y≈141 mid-x on 0x5f
2. Key door LEFT (consumes 1 key) → 0x5e
3. Clear 5× Goriya TYPE_AND_HP via ``ROOM_5E_SPEC`` / ``GenericDungeonRoomController``

Stop: ``level2_room_5e_cleared`` (0 live Goriya, RoomAllDead≥20).

If ``Level2_5F`` lacks keys, prefer regenerating via bomb-north from
``Level2Compass`` (west+east keys carried earlier); do not poke inventory.

Examples::

    uv run python nes/zelda_i/scripts/run_level2_clear5e.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_clear5e.py --save-state --trials 2
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# monorepo root (…/retro_rl); also expose nes/ for package imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _NES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_5E_SPEC,
    level2_room_5e_cleared,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _act(env, direction: str | None) -> None:
    env.step(nes_action(direction) if direction else nes_idle_action())


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _run_controller(env, controller, max_frames: int):
    obs = None
    for _ in range(max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
    return obs


def _enter_5e_key_door(env) -> bool:
    """0x5f key-LEFT: align y≈141 mid-x → pure LEFT through key door.

    Catalog: KEY_DOOR_5F_LEFT (door_y=141, no diamond band).
    """

    def snap():
        return read_snapshot(env.get_ram())

    # Band mid-height, center-ish x.
    for _ in range(400):
        s = snap()
        if s.screen == 0x5E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            _act(env, "LEFT" if s.transitioning else None)
            continue
        if abs(s.link_y - 141) > 4:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
        elif abs(s.link_x - 120) > 8 and s.link_x > 140:
            # Came from south doorway (y≈189): walk to mid x first.
            _act(env, "LEFT" if s.link_x > 120 else "RIGHT")
        elif abs(s.link_x - 120) > 8 and s.link_x < 100:
            _act(env, "RIGHT")
        else:
            break

    # Pure push LEFT through key door.
    for _ in range(600):
        s = snap()
        if s.screen == 0x5E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            _act(env, "LEFT")
            continue
        if abs(s.link_y - 141) > 4:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
        else:
            _act(env, "LEFT")
    s = snap()
    return s.screen == 0x5E and s.mode == PLAY_MODE


def run_once(
    *,
    tag: str = "level2_clear5e",
    save_checkpoint: bool = False,
    start_state: str = "Level2_5F",
    checkpoint_name: str = "Level2_5E",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = GenericDungeonRoomController(ROOM_5E_SPEC)
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        prefix_ok = True
        prefix_error = None
        keys_before_door = entry.keys

        if entry.screen == 0x5E and entry.mode == PLAY_MODE:
            # Already in target (e.g. Level2_5E reload for combat-only).
            controller.phase = DungeonPhase.FIGHT
            obs = _run_controller(env, controller, ROOM_5E_SPEC.max_frames)
        elif entry.screen == 0x5F:
            if entry.keys < 1:
                prefix_ok = False
                prefix_error = "no_keys_on_0x5f"
            else:
                _idle(env, 20)
                if not _enter_5e_key_door(env):
                    prefix_ok = False
                    prefix_error = "failed_key_door_0x5f_to_0x5e"
                else:
                    # Fight only — entry route already done by hand.
                    controller.phase = DungeonPhase.FIGHT
                    obs = _run_controller(
                        env, controller, ROOM_5E_SPEC.max_frames
                    )
        else:
            # Generic controller can ROUTE_ENTRY from source room if start is
            # mid-0x5f-compatible; otherwise report unsupported.
            if entry.screen == ROOM_5E_SPEC.source_room:
                obs = _run_controller(env, controller, ROOM_5E_SPEC.max_frames)
            else:
                prefix_ok = False
                prefix_error = f"unsupported_start_room_0x{entry.screen:02x}"

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level2_room_5e_cleared(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, checkpoint_name)
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
                        "segment": "level2_clear5e",
                        "natural_entry": False,
                        "start_state": start_state,
                    },
                    selected_trial=controller.report(),
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
            "prefix_error": prefix_error,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
                "compass": entry.compass,
                "doors": entry.cur_opened_doors,
            },
            "keys_before_door": keys_before_door,
            "controller": controller.report(),
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
                "live_goriya": len(ROOM_5E_SPEC.live_enemies(snap)),
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level2_5F")
    parser.add_argument(
        "--checkpoint-name",
        default="Level2_5E",
        help="Name for --save-state (default Level2_5E)",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level2_clear5e_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            checkpoint_name=args.checkpoint_name,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        ctrl = report.get("controller") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} "
            f"live={final.get('live_goriya')} "
            f"keys={final.get('keys')} "
            f"all_dead={final.get('room_all_dead')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"frames={ctrl.get('frames')} phase={ctrl.get('phase')} "
            f"max_live={ctrl.get('max_live_enemies')} "
            f"err={report.get('prefix_error')}"
        )

    output = RECORDINGS_DIR / "level2_clear5e_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level2_clear5e",
            "bead": "rr-etl",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level2_room_5e_cleared",
            "spec_id": ROOM_5E_SPEC.spec_id,
            "entry_policy": (
                "Level2_5F keys≥1; align y≈141 mid-x; pure LEFT key door "
                "(KEY_DOOR_5F_LEFT); fight 5× Goriya 0x06 TYPE_AND_HP"
            ),
            "enemies": "5× Goriya type 0x06 TYPE_AND_HP (spawn HP≈48)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
