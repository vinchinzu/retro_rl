"""Clear Level 5 room 0x77 (5× Pols Voice) and collect its key.

Default start: ``L5_Room_77`` (already in room; keys forced 0 at load for
isolated FIXED_INVENTORY). Predecessor states keep keys.

The natural route first clears 0x66 for a key, returns south to 0x76, then
opens the east key door.  ``Level5Cleared66`` is the route-ready predecessor;
``Level5Entrance`` has zero keys and cannot open this door.

Stop: ``level5_room_77_key_success`` (keys≥1, no live type 0x16).

Examples::

    uv run python nes/zelda_i/scripts/run_level5_east_key.py \
        --from-state Level5Cleared66 --infinite-life --save-state --trials 1
    uv run python nes/zelda_i/scripts/run_level5_east_key.py --trials 2
    uv run python nes/zelda_i/scripts/run_level5_east_key.py --infinite-life --trials 2
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import (
    POLS_VOICE_OBJECT_TYPE,
    ROOM_77_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_POLS_77,
    ROOM_L5_GIBDO_66,
    level5_room_77_key_success,
    make_pols_voice_controller,
)
from zelda_i.level5_path import level5_east_key_step, should_force_keys_zero
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _track_labels(infinite_life: bool) -> tuple[str, str]:
    if infinite_life:
        return "assisted", "survival"
    return "clean", "clean"


def _enter_77_from_entry(
    env, *, assist, max_frames: int = 2500
) -> tuple[bool, object | None, list[dict]]:
    """Deterministically route 0x66→0x76→0x77 and retain a transition trail."""
    obs = None
    trail: list[dict] = []
    last_room = read_snapshot(env.get_ram()).screen
    for frame in range(max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == 5
            and snap.screen == ROOM_L5_POLS_77
            and snap.mode == PLAY_MODE
        ):
            for settle in range(40):
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=frame + 1 + settle)
            return True, obs, trail
        action = level5_east_key_step(snap)
        obs, *_ = env.step(action.action)
        after = read_snapshot(env.get_ram())
        if frame % 250 == 0:
            trail.append(
                {
                    "event": "sample",
                    "frame": frame + 1,
                    "room": after.screen,
                    "room_hex": f"0x{after.screen:02x}",
                    "mode": after.mode,
                    "x": after.link_x,
                    "y": after.link_y,
                    "keys": after.keys,
                    "reason": action.reason,
                }
            )
        if after.screen != last_room:
            trail.append(
                {
                    "event": "transition",
                    "frame": frame + 1,
                    "room": after.screen,
                    "room_hex": f"0x{after.screen:02x}",
                    "mode": after.mode,
                    "x": after.link_x,
                    "y": after.link_y,
                    "keys": after.keys,
                }
            )
            last_room = after.screen
    return False, obs, trail


def run_once(
    *,
    tag: str = "level5_east_key",
    save_checkpoint: bool = False,
    start_state: str = "L5_Room_77",
    infinite_life: bool = False,
    keep_keys: bool = False,
    force_keys_zero: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    controller = make_pols_voice_controller()
    track, intervention_class = _track_labels(infinite_life)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        zero_keys = should_force_keys_zero(
            start_state, keep_keys=keep_keys, force_keys_zero=force_keys_zero
        )
        if zero_keys:
            env.data.set_value("keys", 0)

        entry = read_snapshot(env.get_ram())
        prefix_ok = True
        prefix_trail: list[dict] = []
        if entry.screen in (ROOM_L5_GIBDO_66, ROOM_L5_ENTRY):
            prefix_ok, prefix_obs, prefix_trail = _enter_77_from_entry(
                env, assist=assist
            )
            if prefix_obs is not None:
                obs = prefix_obs
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                fail_snap = read_snapshot(env.get_ram())
                return {
                    "ok": False,
                    "track": track,
                    "intervention_class": intervention_class,
                    "start_state": start_state,
                    "prefix_ok": False,
                    "prefix_trail": prefix_trail,
                    "force_keys_zero": zero_keys,
                    "entry": {
                        "room": entry.screen,
                        "x": entry.link_x,
                        "y": entry.link_y,
                        "doors": entry.cur_opened_doors,
                        "keys": entry.keys,
                    },
                    "controller": controller.report(),
                    "final": {
                        "room": fail_snap.screen,
                        "keys": fail_snap.keys,
                        "x": fail_snap.link_x,
                        "y": fail_snap.link_y,
                        "mode": fail_snap.mode,
                    },
                    "screenshot": str(screenshot),
                    "note": "east_key_prefix_failed",
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
                        GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level5_east_key",
                        "predecessor_entry": start_state != "L5_Room_77",
                        "start_state": start_state,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        live = len(ROOM_77_SPEC.live_enemies(snap))
        return {
            "ok": ok,
            "track": track,
            "intervention_class": intervention_class,
            "start_state": start_state,
            "prefix_ok": prefix_ok,
            "prefix_trail": prefix_trail,
            "force_keys_zero": zero_keys,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "doors": entry.cur_opened_doors,
                "keys": entry.keys,
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
        help="Default L5_Room_77 (isolated); Level5Cleared66 is the route predecessor.",
    )
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (ASSIST_CONTRACT). Not Clean STATUS.",
    )
    parser.add_argument(
        "--keep-keys",
        action="store_true",
        help="Never force keys to 0 (safety; default already keeps keys on predecessors).",
    )
    parser.add_argument(
        "--force-keys-zero",
        action="store_true",
        help="Force keys=0 at load (isolated FIXED_INVENTORY). Off by default on predecessors.",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"l5_east_key_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            keep_keys=args.keep_keys,
            force_keys_zero=args.force_keys_zero,
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

    track, intervention_class = _track_labels(args.infinite_life)
    output = RECORDINGS_DIR / f"l5_east_key_{track}.json"
    write_json_report(
        output,
        {
            "segment": "level5_east_key",
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": intervention_class,
            "track": track,
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
