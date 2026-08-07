"""Assisted Survival: Level3Darknuts → ADDR_RAFT (Compass west path).

Durable runner for the LIVE L3 Raft segment (not Clean STATUS)::

    0x5b LEFT → 0x5a LEFT KEY (y≈141 long push) → 0x59 clear DOWN → 0x69
    clear RIGHT @ y≈141 → 0x0f mode-9 channel → Raft (ADDR_RAFT)

Examples::

    uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level3_raft.py --from-state Level3Darknuts --infinite-life
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
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import (
    DARKNUT_OBJECT_TYPE,
    RAFT_CHANNEL_X,
    RAFT_PATH_MAX_FRAMES,
    RAFT_PICKUP_X,
    RAFT_PICKUP_Y,
    Level3RaftPathController,
    level3_has_raft,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_RAFT, PLAY_MODE, read_snapshot, read_u8


def run_once(
    *,
    tag: str = "level3_raft",
    save_checkpoint: bool = False,
    start_state: str = "Level3Darknuts",
    infinite_life: bool = True,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = Level3RaftPathController()
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    max_frames = RAFT_PATH_MAX_FRAMES
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())
        entry_raft = int(read_u8(env.get_ram(), ADDR_RAFT))

        last_reason = ""
        for frame in range(max_frames):
            ram = env.get_ram()
            snap = read_snapshot(ram)
            has_raft = level3_has_raft(ram)
            action = controller.step(snap, has_raft=has_raft)
            last_reason = action.reason
            obs, *_ = env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame + 1)
            if controller.success or controller.failed or controller.phase == "failed":
                break

        # Brief settle after success (pickup / mode settle).
        for settle in range(30):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=frame + settle + 1)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level3_has_raft(ram)
        if ok and not controller.success:
            controller.success = True
            controller.phase = "done"
            controller.notes.append("raft_after_settle")

        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level3Raft")
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
                        "segment": "level3_raft",
                        "natural_entry": False,
                        "start_state": start_state,
                        "intervention_class": intervention,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_assisted.png"
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
                "raft": entry_raft,
            },
            "controller": controller.report(),
            "last_reason": last_reason,
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "bombs": snap.bombs,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "live_darknuts": live_darknuts,
                "raft": int(read_u8(ram, ADDR_RAFT)),
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
    parser.add_argument("--from-state", default="Level3Darknuts")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        default=True,
        help="Survival assist (default on for this residual segment)",
    )
    parser.add_argument(
        "--no-infinite-life",
        action="store_true",
        help="Disable Survival assist (not expected to clear Darknuts Clean)",
    )
    args = parser.parse_args(argv)
    infinite_life = not args.no_infinite_life

    reports = [
        run_once(
            tag=f"level3_raft_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        ctrl = report["controller"]
        print(
            f"trial={trial} ok={report['ok']} track={report['track']} "
            f"room={final['room']:02X} mode={final['mode']} "
            f"raft={final['raft']} "
            f"frames={ctrl['frames']} phase={ctrl['phase']} "
            f"notes={ctrl['notes'][-4:]}"
        )

    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    successes = sum(1 for report in reports if report["ok"])
    output = RECORDINGS_DIR / "level3_raft_assisted.json"
    write_json_report(
        output,
        {
            "segment": "level3_raft",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": intervention,
            "track": track,
            "trials": args.trials,
            "successes": successes,
            "stop_predicate": "level3_has_raft",
            "geometry": {
                "key_door_y": 141,
                "stairs_y": 141,
                "channel_x": RAFT_CHANNEL_X,
                "pickup_xy": [RAFT_PICKUP_X, RAFT_PICKUP_Y],
            },
            "room_graph": {
                "0x5b": {
                    "role": "darknuts_start",
                    "west": "0x5a open (no clear)",
                },
                "0x5a": {
                    "role": "compass",
                    "left_key": "0x59 @ y≈141 long push",
                },
                "0x59": {
                    "role": "west_darknuts",
                    "down": "0x69 after kill-clear",
                },
                "0x69": {
                    "role": "south_darknuts",
                    "right": "0x0f stairs @ y≈141",
                },
                "0x0f": {
                    "role": "raft_passage_mode9",
                    "path": "DOWN y189 → RIGHT x176 → UP → LEFT x136 Raft",
                },
            },
            "reports": reports,
            "checkpoint_name": "Level3Raft",
        },
    )
    print(f"wrote {output} successes={successes}/{args.trials}")
    return 0 if successes == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
