"""Graph pure: enter Level 5 residual room 0x67 from Level5Cleared66.

From cleared 0x66 (doors=0x08) walk RIGHT @ y≈141 → 0x67.
Stop: ``level5_room_67_arrived`` (room-ready, west door bit, Bubbles present).

Bubbles (type 0x40) are sword-immune — no combat clear. Graph node only.

Examples::

    uv run python nes/zelda_i/scripts/run_level5_east67.py --trials 2
    uv run python nes/zelda_i/scripts/run_level5_east67.py --save-state
    uv run python nes/zelda_i/scripts/run_level5_east67.py --infinite-life --trials 1
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
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import (
    BUBBLE_OBJECT_TYPE,
    ROOM_67_SPEC,
    ROOM_L5_EAST_67,
    level5_room_67_arrived,
    make_east_67_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_once(
    *,
    tag: str = "level5_east67",
    save_checkpoint: bool = False,
    start_state: str = "Level5Cleared66",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    controller = make_east_67_controller()
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())

        for frame in range(ROOM_67_SPEC.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=frame)
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.failed:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level5_room_67_arrived(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "L5_Room_67")
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
                        "segment": "level5_east67",
                        "natural_entry": False,
                        "start_state": start_state,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        bubbles = [
            {
                "slot": o.slot,
                "type": o.type_id,
                "hp": o.hp,
                "x": o.x,
                "y": o.y,
            }
            for o in snap.objects
            if 1 <= o.slot <= 10 and o.type_id == BUBBLE_OBJECT_TYPE
        ]
        return {
            "ok": ok,
            "track": "assisted" if infinite_life else "clean",
            "start_state": start_state,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "doors": entry.cur_opened_doors,
            },
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "doors": snap.cur_opened_doors,
                "room_item_id": snap.room_item_id,
                "bubbles": bubbles,
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
    parser.add_argument("--from-state", default="Level5Cleared66")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS).",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"l5_east67_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} "
            f"room={final['room']:02X} doors={final['doors']:#04x} "
            f"bubbles={len(final['bubbles'])} "
            f"frames={report['controller']['frames']}"
        )

    output = RECORDINGS_DIR / "l5_east67_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level5_east67",
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "assisted" if args.infinite_life else "clean",
            "track": "assisted" if args.infinite_life else "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level5_room_67_arrived",
            "spec_id": ROOM_67_SPEC.spec_id,
            "room_id": ROOM_L5_EAST_67,
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
