"""Complete Level 1 (The Eagle) and collect Triforce shard 1.

Examples::

    uv run python zelda_i/scripts/run_level1_complete.py --trials 2
    uv run python zelda_i/scripts/run_level1_complete.py \
      --natural-entry --trials 2 --save-state
    uv run python zelda_i/scripts/run_level1_complete.py \
      --natural-entry --room-timing --trials 1
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.chain import run_controller_stage, run_natural_to_milestone
from zelda_i.dungeon import (
    GenericDungeonRoomController,
    ROOM_23_SPEC,
    ROOM_33_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_44_SPEC,
    ROOM_45_SPEC,
    ROOM_52_SPEC,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level1_finish import (
    AQUAMENTUS_MAX_FRAMES,
    BACKTRACK_TO_44_MAX_FRAMES,
    LEVEL1_TRIFORCE_BIT,
    ROOM_42_EXIT_MAX_FRAMES,
    TRIFORCE_MAX_FRAMES,
    Level1BacktrackTo44Controller,
    Level1AquamentusController,
    Level1Room42ExitController,
    Level1TriforceController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, ROOM_TIMINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.room_timer import RoomTimer, bottleneck_visits


def _finish_stages(*, natural_entry: bool):
    room33 = ROOM_33_SPEC
    room23 = ROOM_23_SPEC
    room44 = ROOM_44_SPEC
    room45 = ROOM_45_SPEC
    boss_entry_delay = 109
    if not natural_entry:
        # Checkpoint-isolated runs start from a different emulator RNG stream.
        # Preserve the independently verified checkpoint tunings while the
        # canonical natural-entry chain uses the specs above.
        room33 = replace(
            room33,
            combat=replace(
                room33.combat,
                engage_distance=40,
                attack_phase=0,
            ),
        )
        room23 = replace(
            room23,
            combat=replace(
                room23.combat,
                engage_distance=64,
                attack_phase=0,
            ),
        )
        room44 = replace(
            room44,
            combat=replace(
                room44.combat,
                engage_distance=80,
                attack_phase=6,
            ),
        )
        room45 = replace(
            room45,
            combat=replace(room45.combat, attack_phase=2),
        )
        boss_entry_delay = 0
    return (
        (
            "clear52",
            GenericDungeonRoomController(ROOM_52_SPEC),
            ROOM_52_SPEC.max_frames,
        ),
        (
            "clear42",
            GenericDungeonRoomController(ROOM_42_SPEC),
            ROOM_42_SPEC.max_frames,
        ),
        (
            "exit42",
            Level1Room42ExitController(),
            ROOM_42_EXIT_MAX_FRAMES,
        ),
        (
            "clear43",
            GenericDungeonRoomController(ROOM_43_SPEC),
            ROOM_43_SPEC.max_frames,
        ),
        (
            "clear33_key",
            GenericDungeonRoomController(room33),
            room33.max_frames,
        ),
        (
            "clear23_key",
            GenericDungeonRoomController(room23),
            room23.max_frames,
        ),
        (
            "backtrack44",
            Level1BacktrackTo44Controller(),
            BACKTRACK_TO_44_MAX_FRAMES,
        ),
        (
            "clear44",
            GenericDungeonRoomController(room44),
            room44.max_frames,
        ),
        (
            "clear45_key",
            GenericDungeonRoomController(room45),
            room45.max_frames,
        ),
        (
            "aquamentus_heart",
            Level1AquamentusController(
                entry_delay_frames=boss_entry_delay,
            ),
            AQUAMENTUS_MAX_FRAMES,
        ),
        (
            "triforce_shard_1",
            Level1TriforceController(),
            TRIFORCE_MAX_FRAMES,
        ),
    )


def run_once(
    *,
    natural_entry: bool = False,
    tag: str = "level1_complete",
    save_checkpoint: bool = False,
    room_timing: bool = False,
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1Cleared53"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    prefix = None
    stages = []
    room_timer = RoomTimer() if room_timing else None
    frame_base = 0
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if natural_entry:
            prefix = run_natural_to_milestone(
                env,
                milestone="clear53",
                room_timer=room_timer,
                frame_base=frame_base,
            )
            obs = prefix.obs
            prefix_ok = prefix.success
            frame_base = prefix.end_frame
        else:
            obs, *_ = env.step(nes_idle_action())
            frame_base = 1
            if room_timer is not None:
                room_timer.observe(read_snapshot(env.get_ram()), frame=frame_base)
            prefix_ok = True

        for name, controller, max_frames in _finish_stages(
            natural_entry=natural_entry
        ):
            if not prefix_ok:
                break
            obs, stage = run_controller_stage(
                env,
                obs,
                name=name,
                controller=controller,
                max_frames=max_frames,
                room_timer=room_timer,
                frame_base=frame_base,
            )
            stages.append(stage)
            frame_base = stage.end_frame
            prefix_ok = prefix_ok and stage.success
            if not stage.success:
                break

        snap = read_snapshot(env.get_ram())
        ok = prefix_ok and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level1Complete")
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        None
                        if natural_entry
                        else GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / "Level1Cleared53.state"
                    ),
                    request={
                        "segment": "level1_complete",
                        "natural_entry": natural_entry,
                    },
                    selected_trial={
                        "success": ok,
                        "stages": [stage.report() for stage in stages],
                    },
                    natural_entry=natural_entry,
                )
            )

        label = "natural" if natural_entry else "isolated"
        screenshot = RECORDINGS_DIR / f"{tag}_{label}.png"
        save_rgb_png(obs, screenshot)
        payload = {
            "ok": ok,
            "natural_entry": natural_entry,
            "prefix_ok": prefix.success if prefix else True,
            "prefix": prefix.report() if prefix else None,
            "stages": [stage.report() for stage in stages],
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "health": snap.health,
                "keys": snap.keys,
                "triforce": snap.triforce,
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
        }
        if room_timer is not None:
            room_timer.finalize(frame=frame_base)
            payload["room_timing"] = room_timer.report(
                source=f"run_level1_complete:{tag}",
                extra={
                    "ok": ok,
                    "natural_entry": natural_entry,
                    "final_room": snap.screen,
                    "final_room_hex": f"0x{snap.screen:02X}",
                    "bottlenecks": bottleneck_visits(room_timer.visits, top_n=8),
                },
            )
        return payload
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-entry", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in screen/room hop timing via zelda_i.room_timer; "
            f"writes JSON under {ROOM_TIMINGS_DIR}"
        ),
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            natural_entry=args.natural_entry,
            tag=f"level1_complete_t{trial}",
            save_checkpoint=args.save_state,
            room_timing=args.room_timing,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        failed = next(
            (
                stage["name"]
                for stage in report["stages"]
                if not stage["success"]
            ),
            "-",
        )
        print(
            f"trial={trial} ok={report['ok']} "
            f"prefix_ok={report['prefix_ok']} failed={failed} "
            f"room={final['room']:02X} triforce=0x{final['triforce']:02X}"
        )
        if args.room_timing and "room_timing" in report:
            rt = report["room_timing"]
            print(
                f"  room_timing visits={rt.get('visit_count')} "
                f"dwell={rt.get('total_dwell_frames')} "
                f"transition={rt.get('total_transition_frames')}"
            )

    label = "natural" if args.natural_entry else "isolated"
    output = RECORDINGS_DIR / f"level1_complete_{label}.json"
    write_json_report(
        output,
        {
            "segment": "level1_complete",
            "natural_entry": args.natural_entry,
            "room_timing": args.room_timing,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "triforce & 0x01",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    if args.room_timing:
        ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
        timing_out = ROOM_TIMINGS_DIR / f"level1_complete_{label}_timing.json"
        write_json_report(
            timing_out,
            {
                "segment": "level1_complete",
                "natural_entry": args.natural_entry,
                "trials": args.trials,
                "successes": sum(report["ok"] for report in reports),
                "trial_timings": [
                    r.get("room_timing") for r in reports if "room_timing" in r
                ],
            },
        )
        print(f"room_timing={timing_out}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
