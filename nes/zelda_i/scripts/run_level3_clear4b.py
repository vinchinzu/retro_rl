"""Isolated: Level3Darknuts 0x5b → 0x4b clear 3× Zol (Clean attempt OK).

North door from 0x5b is **open without Darknut clear**. Stop:
``level3_room_4b_zols_cleared``. Key pickup residual (RoomItemId 0x19 may not
increment inventory — same class as 0x6b).

Examples::

    uv run python nes/zelda_i/scripts/run_level3_clear4b.py --trials 2
    uv run python nes/zelda_i/scripts/run_level3_clear4b.py --infinite-life --trials 1
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
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import GenericDungeonRoomController
from zelda_i.level3_dungeon import (
    NORTH_DOOR_X,
    NORTH_DOOR_X_TOL,
    ROOM_4B_SPEC,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ZOL_KEY_4B,
    level3_room_4b_zols_cleared,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _north_door_step(snap):
    """0x5b → 0x4b UP @ x≈120 (open door, no clear required)."""
    if snap.screen == ROOM_L3_ZOL_KEY_4B and snap.mode == PLAY_MODE:
        return nes_idle_action(), "arrived_4b"
    if snap.transitioning:
        return nes_action("UP"), "scroll"
    if snap.screen != ROOM_L3_DARKNUTS:
        return nes_idle_action(), f"unexpected_0x{snap.screen:02x}"
    if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        d = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return nes_action(d), "align_x"
    return nes_action("UP"), "push_north"


def run_once(
    *,
    tag: str = "level3_clear4b",
    save_checkpoint: bool = False,
    start_state: str = "Level3Darknuts",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    combat = GenericDungeonRoomController(ROOM_4B_SPEC)
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    max_frames = 2000 + ROOM_4B_SPEC.max_frames
    phase = "door"
    door_frames = 0
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())

        for frame in range(max_frames):
            snap = read_snapshot(env.get_ram())
            if phase == "door":
                door_frames += 1
                act, reason = _north_door_step(snap)
                obs, *_ = env.step(act.action if hasattr(act, "action") else act)
                if (
                    snap.screen == ROOM_L3_ZOL_KEY_4B
                    and snap.mode == PLAY_MODE
                    and not snap.transitioning
                ):
                    phase = "combat"
                elif door_frames > 2000:
                    phase = "failed"
            elif phase == "combat":
                action = combat.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(action.action)
                if combat.success:
                    phase = "done"
                    break
                if combat.phase.name == "FAILED":
                    phase = "failed"
                    break
            else:
                break
            if assist is not None:
                assist.apply_env(env, frame=frame + 1)

        for settle in range(60):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=max_frames + settle)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level3_room_4b_zols_cleared(ram)
        checkpoint = None
        if ok and save_checkpoint:
            checkpoint = str(save_state(env, GAME_DIR, GAME, "Level3_4B_Cleared"))
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": False,
            "start_state": start_state,
            "intervention_class": intervention,
            "track": track,
            "phase": phase,
            "door_frames": door_frames,
            "combat": combat.report(),
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
            },
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "room_item_id": snap.room_item_id,
                "live_zols": len(ROOM_4B_SPEC.live_enemies(snap)),
            },
            "assist": assist.report() if assist else None,
            "checkpoint": checkpoint,
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level3Darknuts")
    parser.add_argument("--infinite-life", action="store_true")
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level3_clear4b_t{trial}",
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
            f"room={final['room']:02X} zols={final['live_zols']} "
            f"keys={final['keys']} phase={report['phase']}"
        )

    track = "assisted" if args.infinite_life else "clean"
    output = RECORDINGS_DIR / "level3_clear4b_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level3_clear4b",
            "natural_entry": False,
            "start_state": args.from_state,
            "intervention_class": "survival" if args.infinite_life else "clean",
            "track": track,
            "trials": args.trials,
            "successes": sum(r["ok"] for r in reports),
            "stop_predicate": "level3_room_4b_zols_cleared",
            "spec_id": ROOM_4B_SPEC.spec_id,
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
