"""Isolated pure: clear Level 5 room 0x77 (5× Pols Voice) + collect key.

Default start: ``L5_Room_77`` (already in room; keys forced 0 at load if needed
via inventory — runner expects keys=0 at room-ready for FIXED_INVENTORY).

Natural east door from ``Level5Entrance`` is **PARTIAL** (doors=0; wall at
x≈208 does not scroll without door-open residual). Use ``L5_Room_77`` for
combat pure, or ``--poke-doors`` recon from entrance (not Clean STATUS).

Stop: ``level5_room_77_key_success`` (keys≥1, no live type 0x16).

Examples::

    uv run python nes/zelda_i/scripts/run_level5_east_key.py --trials 2
    uv run python nes/zelda_i/scripts/run_level5_east_key.py --save-state
    uv run python nes/zelda_i/scripts/run_level5_east_key.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level5_east_key.py --from-state Level5Entrance --poke-doors --infinite-life
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # monorepo root
_NES_ROOT = Path(__file__).resolve().parents[2]  # nes/
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
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import (
    EAST_DOOR_APPROACH_Y,
    EAST_DOOR_CHANNEL_Y,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_77_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_POLS_77,
    level5_room_77_key_success,
    make_pols_voice_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_KEYS,
    ADDR_OPEN_DOORWAY_MASK,
    PLAY_MODE,
    read_snapshot,
)


def _ensure_door_vars(env) -> None:
    for name, addr in (
        ("doors", ADDR_CUR_OPENED_DOORS),
        ("mask", ADDR_OPEN_DOORWAY_MASK),
    ):
        try:
            env.data.set_variable(name, {"address": addr, "type": "|u1"})
        except Exception:
            pass


def _poke_doors_open(env) -> None:
    """Recon residual: force all door bits open (not Clean)."""
    _ensure_door_vars(env)
    env.data.set_value("doors", 0x0F)
    env.data.set_value("mask", 0x0F)


def _enter_77_from_entry(env, *, assist, max_frames: int = 2500, poke_doors: bool) -> bool:
    """Assisted/recon path into 0x77. Returns True if room-ready on 0x77."""
    import random

    if poke_doors:
        _poke_doors_open(env)
    # Leave south mouth then random-walk with RIGHT bias while doors open.
    for frame in range(max_frames):
        if poke_doors and frame % 5 == 0:
            _poke_doors_open(env)
        if assist is not None:
            assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == 5
            and snap.screen == ROOM_L5_POLS_77
            and snap.mode == PLAY_MODE
        ):
            for _ in range(40):
                env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=0)
            return True
        if snap.link_y > 185:
            act = nes_action("UP")
        elif snap.link_x < 180:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                act = nes_action(
                    "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                )
            else:
                act = nes_action("RIGHT")
        else:
            if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
                act = nes_action(
                    "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
                )
            else:
                act = nes_action("RIGHT")
        # occasional jitter to reproduce door-scroll residual
        if frame % 40 == 0 and snap.link_x > 160:
            act = nes_action(random.choice(["UP", "DOWN", "RIGHT"]))
        env.step(act)
    return False


def run_once(
    *,
    tag: str = "level5_east_key",
    save_checkpoint: bool = False,
    start_state: str = "L5_Room_77",
    infinite_life: bool = False,
    poke_doors: bool = False,
    force_keys_zero: bool = True,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    controller = make_pols_voice_controller()
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        if force_keys_zero:
            env.data.set_value("keys", 0)

        entry = read_snapshot(env.get_ram())
        prefix_ok = True
        if entry.screen == ROOM_L5_ENTRY and entry.mode == PLAY_MODE:
            prefix_ok = _enter_77_from_entry(
                env, assist=assist, poke_doors=poke_doors
            )
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "track": "assisted" if (infinite_life or poke_doors) else "clean",
                    "start_state": start_state,
                    "prefix_ok": False,
                    "entry": {
                        "room": entry.screen,
                        "x": entry.link_x,
                        "y": entry.link_y,
                        "doors": entry.cur_opened_doors,
                        "keys": entry.keys,
                    },
                    "controller": controller.report(),
                    "final": {
                        "room": read_snapshot(env.get_ram()).screen,
                        "keys": read_snapshot(env.get_ram()).keys,
                    },
                    "screenshot": str(screenshot),
                    "note": "east_door_residual_failed",
                }

        for frame in range(ROOM_77_SPEC.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=frame)
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is DungeonPhase.FAILED:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level5_room_77_key_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level5EastKey")
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
                        "segment": "level5_east_key",
                        "natural_entry": False,
                        "start_state": start_state,
                        "poke_doors": poke_doors,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        live = len(ROOM_77_SPEC.live_enemies(snap))
        track = "assisted" if (infinite_life or poke_doors) else "clean"
        return {
            "ok": ok,
            "track": track,
            "start_state": start_state,
            "prefix_ok": prefix_ok,
            "poke_doors": poke_doors,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "doors": entry.cur_opened_doors,
                "keys": int(env.get_ram()[ADDR_KEYS]) if force_keys_zero else entry.keys,
            },
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
                "live_pols": live,
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
    parser.add_argument(
        "--from-state",
        default="L5_Room_77",
        help="Default L5_Room_77; Level5Entrance needs --poke-doors",
    )
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (ASSIST_CONTRACT). Not Clean STATUS.",
    )
    parser.add_argument(
        "--poke-doors",
        action="store_true",
        help="Force cur_opened_doors/mask open to enter 0x77 from 0x76 (recon).",
    )
    parser.add_argument(
        "--keep-keys",
        action="store_true",
        help="Do not force keys=0 at start (default forces 0 for key pickup).",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"l5_east_key_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            poke_doors=args.poke_doors,
            force_keys_zero=not args.keep_keys,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final", {})
        print(
            f"trial={trial} ok={report['ok']} track={report.get('track')} "
            f"room={final.get('room', -1):02X} keys={final.get('keys')} "
            f"live={final.get('live_pols')} "
            f"frames={report['controller'].get('frames')} "
            f"phase={report['controller'].get('phase')}"
        )

    track = "assisted" if (args.infinite_life or args.poke_doors) else "clean"
    output = RECORDINGS_DIR / f"l5_east_key_{track}.json"
    write_json_report(
        output,
        {
            "segment": "level5_east_key",
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": track,
            "track": track,
            "poke_doors": args.poke_doors,
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level5_room_77_key_success",
            "spec_id": ROOM_77_SPEC.spec_id,
            "enemy_type": POLS_VOICE_OBJECT_TYPE,
            "enemy_type_hex": f"0x{POLS_VOICE_OBJECT_TYPE:02x}",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
