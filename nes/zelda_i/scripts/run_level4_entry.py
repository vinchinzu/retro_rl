"""Assisted: Level3Complete+Raft → dock 0x55 → island 0x45 → Level4Entrance.

Durable runner for the LIVE L4 overworld entry segment (not Clean STATUS)::

    Level3Complete fanfare → OW 0x74 raft=1
    → 0x73 → 0x63 E@y≈149 → 0x64 → 0x65 → dock 0x55
    → Raft N → island 0x45 → door UP → level 4 room 0x71

Examples::

    uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level4_entry.py --from-state Level3Complete --infinite-life --dock-only
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
from zelda_i.level4_overworld import (
    LEVEL4,
    LEVEL4_DOCK_SCREEN,
    LEVEL4_ENTRY_ROOM,
    LEVEL4_ISLAND_SCREEN,
    LEVEL4_HOPS_FROM_POST_L3,
    POST_L3_PATH_MAX_FRAMES,
    POST_L3_SETTLE_MAX_FRAMES,
    SCREEN_POST_L3_RETURN,
    OverworldToLevel4Controller,
    PostL3TriforceSettleController,
    has_raft,
    level4_entrance_success,
    on_level4_dock,
    planning_report,
    post_l3_overworld_ready,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_RAFT, PLAY_MODE, read_snapshot, read_u8


def _snap_dict(snap) -> dict:
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "triforce": snap.triforce,
        "health": snap.health,
        "sword": int(snap.has_sword),
    }


def _ensure_post_l3_ow(env, *, assist, tag: str) -> dict:
    """Settle Level3Complete fanfare → OW 0x74 with raft."""
    ram = env.get_ram()
    if post_l3_overworld_ready(ram):
        return {"ok": True, "stage": "already_ow", "frames": 0}

    if not has_raft(ram):
        snap = read_snapshot(ram)
        return {
            "ok": False,
            "stage": "missing_raft",
            "entry": _snap_dict(snap),
            "raft": int(read_u8(ram, ADDR_RAFT)),
        }

    settle = PostL3TriforceSettleController(max_frames=POST_L3_SETTLE_MAX_FRAMES)
    frames = 0
    while frames < POST_L3_SETTLE_MAX_FRAMES:
        ram = env.get_ram()
        snap = read_snapshot(ram)
        act = settle.step(snap, has_raft_flag=has_raft(ram))
        env.step(act.action)
        frames += 1
        if assist is not None and frames % 15 == 0:
            assist.apply_env(env, frame=frames)
        if settle.success:
            break
    ram = env.get_ram()
    ok = post_l3_overworld_ready(ram)
    return {
        "ok": ok,
        "stage": "settle",
        "frames": frames,
        "settle": settle.report(),
        "final": _snap_dict(read_snapshot(ram)),
        "raft": int(read_u8(ram, ADDR_RAFT)),
    }


def run_once(
    *,
    tag: str = "l4_entry",
    save_checkpoint: bool = False,
    start_state: str = "Level3Complete",
    infinite_life: bool = True,
    dock_only: bool = False,
    door_only: bool = False,
    max_frames: int = POST_L3_PATH_MAX_FRAMES,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = _snap_dict(read_snapshot(env.get_ram()))
        entry_raft = int(read_u8(env.get_ram(), ADDR_RAFT))

        pre = _ensure_post_l3_ow(env, assist=assist, tag=tag)
        if not pre.get("ok"):
            snap = read_snapshot(env.get_ram())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_pre_fail.png")
            return {
                "ok": False,
                "bead": "rr-0fx",
                "track": track,
                "intervention_class": intervention,
                "start_state": start_state,
                "stage": pre.get("stage", "pre"),
                "entry": entry,
                "entry_raft": entry_raft,
                "pre": pre,
                "final": _snap_dict(snap),
                "plan": planning_report(),
            }

        # Optional: save post-L3 OW settle as Level3ExitOverworld
        exit_path = None
        if save_checkpoint and post_l3_overworld_ready(env.get_ram()):
            exit_path = str(
                save_state(env, GAME_DIR, GAME, "Level3ExitOverworld")
            )

        hops = LEVEL4_HOPS_FROM_POST_L3
        if dock_only:
            hops = LEVEL4_HOPS_FROM_POST_L3[:5]  # through dock 0x55

        nav = OverworldToLevel4Controller(
            hops=hops,
            require_dungeon=not dock_only and not door_only,
            stop_at_dock=dock_only,
            require_entrance_screen=door_only,
            max_frames=max_frames,
        )
        frames = 0
        trail: list[dict] = []
        last_sc = read_snapshot(env.get_ram()).screen
        last_reason = ""
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_sc or snap.level != 0:
                trail.append({"f": frames, **_snap_dict(snap), "raft": int(read_u8(env.get_ram(), ADDR_RAFT))})
                last_sc = snap.screen
            if nav.success or (
                hasattr(nav.phase, "name") and nav.phase.name == "FAILED"
            ):
                break
            act = nav.step(snap)
            last_reason = act.reason
            obs, *_ = env.step(act.action)
            frames += 1
            if assist is not None and frames % 15 == 0:
                assist.apply_env(env, frame=frames)

        # Brief settle
        for settle in range(40):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None and settle % 15 == 0:
                assist.apply_env(env, frame=frames + settle)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        entered = level4_entrance_success(ram)
        on_dock = on_level4_dock(snap)
        on_island = (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == LEVEL4_ISLAND_SCREEN
        )
        if dock_only:
            ok = on_dock
        elif door_only:
            ok = on_island or entered
        else:
            ok = entered
            if ok and not nav.success:
                nav.success = True
                nav.notes.append("entry_after_settle")

        dock_path = None
        entrance_path = None
        provenance = None
        if ok and save_checkpoint:
            if on_dock or entered or on_island:
                # Save dock if we passed through; re-load not available — save
                # entrance always; dock when dock_only or when still on dock.
                if dock_only and on_dock:
                    p = save_state(env, GAME_DIR, GAME, "OW_L4Dock")
                    dock_path = str(p)
                    provenance = str(
                        write_state_provenance(
                            p,
                            source_state_path=(
                                GAME_DIR
                                / "custom_integrations"
                                / GAME
                                / f"{start_state}.state"
                            ),
                            request={
                                "segment": "l4_dock",
                                "bead": "rr-0fx",
                                "track": track,
                            },
                            selected_trial={
                                "ok": True,
                                "final": _snap_dict(snap),
                                "frames": frames,
                                "nav": nav.report(),
                            },
                        )
                    )
                if entered:
                    p = save_state(env, GAME_DIR, GAME, "Level4Entrance")
                    entrance_path = str(p)
                    provenance = str(
                        write_state_provenance(
                            p,
                            source_state_path=(
                                GAME_DIR
                                / "custom_integrations"
                                / GAME
                                / f"{start_state}.state"
                            ),
                            request={
                                "segment": "l4_entry",
                                "bead": "rr-0fx",
                                "track": track,
                                "intervention_class": intervention,
                            },
                            selected_trial={
                                "ok": True,
                                "entered_level4": True,
                                "entry_room": hex(LEVEL4_ENTRY_ROOM),
                                "final": _snap_dict(snap),
                                "frames": frames,
                                "nav": nav.report(),
                            },
                        )
                    )

        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{'ok' if ok else 'fail'}.png")
        if entered:
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entrance.png")

        return {
            "ok": ok,
            "bead": "rr-0fx",
            "track": track,
            "intervention_class": intervention,
            "natural_entry": False,
            "start_state": start_state,
            "entry": entry,
            "entry_raft": entry_raft,
            "pre": pre,
            "frames": frames,
            "trail": trail[-40:],
            "controller": nav.report(),
            "last_reason": last_reason,
            "final": {
                **_snap_dict(snap),
                "raft": int(read_u8(ram, ADDR_RAFT)),
                "entered_level4": entered,
                "on_dock": on_dock,
                "on_island": on_island,
                "entry_room_live": hex(snap.screen) if entered else None,
            },
            "assist": assist.report() if assist else None,
            "checkpoints": {
                "Level3ExitOverworld": exit_path,
                "OW_L4Dock": dock_path,
                "Level4Entrance": entrance_path,
            },
            "provenance": provenance,
            "plan": planning_report(),
            "live": {
                "post_l3_return": hex(SCREEN_POST_L3_RETURN),
                "dock": hex(LEVEL4_DOCK_SCREEN),
                "island": hex(LEVEL4_ISLAND_SCREEN),
                "entry_room": hex(LEVEL4_ENTRY_ROOM),
                "level": LEVEL4,
            },
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level3Complete")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        default=True,
        help="Survival assist (default on; not Clean STATUS)",
    )
    parser.add_argument(
        "--no-infinite-life",
        action="store_true",
        help="Disable Survival assist",
    )
    parser.add_argument(
        "--dock-only",
        action="store_true",
        help="Stop on raft dock 0x55 (save OW_L4Dock)",
    )
    parser.add_argument(
        "--door-only",
        action="store_true",
        help="Stop on island door screen 0x45 without entering",
    )
    parser.add_argument("--tag", default="l4_entry")
    parser.add_argument("--max-frames", type=int, default=POST_L3_PATH_MAX_FRAMES)
    args = parser.parse_args(argv)
    infinite_life = not args.no_infinite_life

    reports = [
        run_once(
            tag=f"{args.tag}_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=infinite_life,
            dock_only=args.dock_only,
            door_only=args.door_only,
            max_frames=args.max_frames,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final", {})
        ctrl = report.get("controller", {})
        print(
            f"trial={trial} ok={report.get('ok')} track={report.get('track')} "
            f"level={final.get('level')} screen=0x{final.get('screen', 0):02x} "
            f"raft={final.get('raft')} frames={ctrl.get('frames', report.get('frames'))} "
            f"phase={ctrl.get('phase')} notes={ctrl.get('notes', [])[-6:]}"
        )

    successes = sum(1 for r in reports if r.get("ok"))
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    output = RECORDINGS_DIR / "l4_entry_recon.json"
    write_json_report(
        output,
        {
            "segment": "l4_entry",
            "bead": "rr-0fx",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": intervention,
            "track": track,
            "trials": args.trials,
            "successes": successes,
            "pass": successes == args.trials and args.trials > 0,
            "live": {
                "post_l3_return": hex(SCREEN_POST_L3_RETURN),
                "dock": hex(LEVEL4_DOCK_SCREEN),
                "island": hex(LEVEL4_ISLAND_SCREEN),
                "entry_room": hex(LEVEL4_ENTRY_ROOM),
            },
            "plan": planning_report(),
            "reports": reports,
        },
    )
    print(
        f"summary: {successes}/{args.trials} ok track={track} "
        f"report={output}"
    )
    return 0 if successes == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
