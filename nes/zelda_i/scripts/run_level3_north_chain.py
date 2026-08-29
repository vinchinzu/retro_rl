"""Isolated pure: Level3WestKey 0x7b → 0x6b (5 Zols) → 0x5b Darknuts.

Clean track from ``Level3WestKey``. North door from 0x7b needs strict
``x≈120`` (|dx|≤4). 0x6b clear uses type-0x13 liveness only (RoomAllDead
residual after wooden-sword hits). Stop: ``level3_reached_5b``.

Examples::

    uv run python nes/zelda_i/scripts/run_level3_north_chain.py --trials 2
    uv run python nes/zelda_i/scripts/run_level3_north_chain.py --save-state
    uv run python nes/zelda_i/scripts/run_level3_north_chain.py --infinite-life --trials 1
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
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import (
    DARKNUT_OBJECT_TYPE,
    ROOM_5B_SPEC,
    ROOM_6B_SPEC,
    Level3NorthChainController,
    level3_reached_5b,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

def run_once(
    *,
    tag: str = "level3_north_chain",
    save_checkpoint: bool = False,
    start_state: str = "Level3WestKey",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = Level3NorthChainController(clear_darknuts=False)
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    max_frames = (
        controller.door.max_frames
        + ROOM_6B_SPEC.max_frames
        + controller.north_exit.max_frames
    )
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())

        for frame in range(max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame + 1)
            if controller.success or controller.phase == "failed":
                break

        # Settle room objects (Darknuts) after scroll into 0x5b.
        for settle in range(180):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=frame + settle + 1)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level3_reached_5b(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level3Darknuts")
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
                        "segment": "level3_north_chain",
                        "natural_entry": False,
                        "start_state": start_state,
                        "intervention_class": intervention,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        live_darknuts = sum(
            1
            for o in snap.objects
            if o.slot >= 1 and o.type_id == DARKNUT_OBJECT_TYPE and o.hp > 0
        )
        return {
            "ok": ok,
            "natural_entry": False,
            "start_state": start_state,
            "intervention_class": intervention,
            "track": track,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
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
                "live_darknuts": live_darknuts,
                "live_zols_6b_spec": len(ROOM_6B_SPEC.live_enemies(snap)),
            },
            "assist": assist.report() if assist else None,
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
    parser.add_argument("--from-state", default="Level3WestKey")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist for recon (not Clean STATUS)",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level3_north_chain_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} track={report['track']} "
            f"room={final['room']:02X} keys={final['keys']} "
            f"darknuts={final['live_darknuts']} "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']}"
        )

    track = "assisted" if args.infinite_life else "clean"
    intervention = "survival" if args.infinite_life else "clean"
    output = RECORDINGS_DIR / "level3_north_chain_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level3_north_chain",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": intervention,
            "track": track,
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level3_reached_5b",
            "spec_id": ROOM_6B_SPEC.spec_id,
            "room_graph": {
                "0x7b": {
                    "role": "west_key_cleared",
                    "north": "0x6b",
                    "north_policy": "UP @ x≈120 (|dx|≤4)",
                },
                "0x6b": {
                    "role": "north_zols",
                    "enemies": "5× Zol type 0x13 (TYPE_AND_HP)",
                    "item": "RoomItemId 0x19 (key drop residual)",
                    "geometry": "diagonal raised blocks",
                    "north": "0x5b",
                    "north_policy": "after type-0x13 clear; UP @ x≈120",
                },
                "0x5b": {
                    "role": "darknuts",
                    "enemies": "3× Darknut type 0x0b HP64",
                    "north": "0x4b (3× Zol + key residual)",
                },
                "0x4b": {
                    "role": "zol_key_next",
                    "status": "graph only",
                },
            },
            "reports": reports,
            "next_room_spec_id": ROOM_5B_SPEC.spec_id,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
