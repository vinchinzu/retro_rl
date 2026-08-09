"""Isolated pure: Level 2 bomb-north 0x5f → 0x4f from ``Level2_5F``.

Stand **(120, 101)** facing UP, B places bomb (natural inventory; no
selected-item poke). Opens north wall into room **0x4f** play mode 5
(boom candidate RoomItemId 0x1e). Pure geometry — no gel clear.

Stop: ``level2_room_4f_ready`` (level==2, screen==0x4f, mode==5).
ROOM_4F combat + Magical Boomerang collect left to boom pure work.

Examples::

    uv run python nes/zelda_i/scripts/run_level2_bomb_north_5f.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_bomb_north_5f.py --save-state --trials 2
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
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level2_dungeon import (
    BOMB_N_STAND,
    BombNorthPhase,
    make_boom_bomb_north_controller,
    ROOM_L2_BOOM_CANDIDATE,
    level2_room_4f_ready,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot, read_u8

ADDR_SELECTED_ITEM = 0x0656


def run_once(
    *,
    tag: str = "level2_bomb_north_5f",
    save_checkpoint: bool = False,
    start_state: str = "Level2_5F",
    checkpoint_name: str = "Level2_4F",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = make_boom_bomb_north_controller(clear_gels=False)
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())
        entry_sel = read_u8(env.get_ram(), ADDR_SELECTED_ITEM)

        for _ in range(controller.max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is BombNorthPhase.FAILED:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level2_room_4f_ready(ram)
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
                        "segment": "level2_bomb_north_5f",
                        "natural_entry": False,
                        "start_state": start_state,
                        "stand": list(BOMB_N_STAND),
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
            "intervention_class": "clean",
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "bombs": entry.bombs,
                "health": entry.health,
                "compass": entry.compass,
                "selected_item": entry_sel,
                "doors": entry.cur_opened_doors,
            },
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "bombs": snap.bombs,
                "health": snap.health,
                "compass": snap.compass,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "selected_item": read_u8(ram, ADDR_SELECTED_ITEM),
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
        default="Level2_4F",
        help="Name for --save-state (default Level2_4F)",
    )
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level2_bomb_north_5f_t{trial}",
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
            f"room={final.get('room', 0):02X} mode={final.get('mode')} "
            f"bombs={final.get('bombs')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"frames={ctrl.get('frames')} phase={ctrl.get('phase')} "
            f"bomb={ctrl.get('bombs_before_place')}->{ctrl.get('bombs_after_place')}"
        )

    output = RECORDINGS_DIR / "level2_bomb_north_5f_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level2_bomb_north_5f",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level2_room_4f_ready",
            "target_room": f"0x{ROOM_L2_BOOM_CANDIDATE:02x}",
            "stand": list(BOMB_N_STAND),
            "policy": (
                "settle 0x5f (geometry only, no gel clear); walk to (120,101); "
                "face UP; B place bomb (natural B selection, no poke); wait "
                "blast; push UP into 0x4f"
            ),
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
