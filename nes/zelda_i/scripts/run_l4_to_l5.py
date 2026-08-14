"""Assisted natural predecessor: L4 Triforce → OW → Lost Hills → L5.

The old ``Level5Entrance`` fixture was reached from an early-game overworld
state and therefore lacked the Raft, Stepladder, bombs, and prior shards.  This
runner starts from ``Level4Complete``, idles through the real fanfare return to
island 0x45, rides the Raft back to 0x55, and walks continuously to Level 5
room 0x76.  Survival assist may refill health; no progression state is written.

Example::

    uv run python nes/zelda_i/scripts/run_l4_to_l5.py \
        --infinite-life --trials 2 --save-state
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_overworld import (
    POST_L4_TO_LEVEL5_HOPS,
    OverworldToLevel5Controller,
    level5_entrance_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_RAFT,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

POST_L4_RETURN_SCREEN = 0x45
POST_L4_SETTLE_MAX_FRAMES = 1800
PATH_MAX_FRAMES = 40_000


def _pin(env) -> dict[str, Any]:
    snap = read_snapshot(env.get_ram())
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "triforce": snap.triforce,
        "raft": read_u8(env.get_ram(), ADDR_RAFT),
        "ladder": read_u8(env.get_ram(), ADDR_LADDER),
    }


def _settle_post_l4(env, *, assist: UnlimitedHealthAssist | None) -> dict[str, Any]:
    """Idle L4 mode-18 fanfare until controllable OW island 0x45."""
    trail: list[dict[str, Any]] = []
    last: tuple[int, int, int] | None = None
    for frame in range(POST_L4_SETTLE_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        location = (snap.mode, snap.level, snap.screen)
        if location != last:
            trail.append({"frame": frame, **_pin(env)})
            last = location
        if (
            snap.mode == PLAY_MODE
            and snap.level == 0
            and snap.screen == POST_L4_RETURN_SCREEN
            and not snap.transitioning
        ):
            return {"ok": True, "frames": frame, "trail": trail, "final": _pin(env)}
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=frame + 1)
    return {
        "ok": False,
        "frames": POST_L4_SETTLE_MAX_FRAMES,
        "trail": trail,
        "final": _pin(env),
        "error": "post_l4_settle_timeout",
    }


def run_once(
    *,
    start_state: str,
    infinite_life: bool,
    save_checkpoint: bool,
    tag: str,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = _pin(env)

        settle = _settle_post_l4(env, assist=assist)
        if not settle["ok"]:
            screenshot = RECORDINGS_DIR / f"{tag}_settle_fail.png"
            save_rgb_png(obs, screenshot)
            return {
                "ok": False,
                "error": settle["error"],
                "entry": entry,
                "settle": settle,
                "final": _pin(env),
                "screenshot": str(screenshot),
                "assist": assist.report() if assist else None,
            }

        controller = OverworldToLevel5Controller(
            hops=POST_L4_TO_LEVEL5_HOPS,
            max_frames=PATH_MAX_FRAMES,
        )
        trail: list[dict[str, Any]] = []
        snap = read_snapshot(env.get_ram())
        last_location = (snap.level, snap.screen)
        frames = 0
        while frames < PATH_MAX_FRAMES and not controller.success:
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17 or controller.phase.name == "FAILED":
                break
            action = controller.step(snap)
            obs, *_ = env.step(action.action)
            frames += 1
            if assist is not None:
                assist.apply_env(env, frame=int(settle["frames"]) + frames)
            after = read_snapshot(env.get_ram())
            location = (after.level, after.screen)
            if location != last_location:
                trail.append({"frame": frames, **_pin(env)})
                last_location = location
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR
                    / f"{tag}_f{frames:05d}_L{after.level}_r{after.screen:02x}.png",
                )

        ok = level5_entrance_success(env.get_ram()) and controller.success
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(
                env,
                GAME_DIR,
                GAME,
                "Level5EntranceFromL4",
            )
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
                    ),
                    request={
                        "bead": "rr-28p",
                        "segment": "l4_complete_to_l5_entrance",
                        "track": "assisted" if infinite_life else "clean",
                    },
                    selected_trial={
                        "ok": ok,
                        "settle": settle,
                        "controller": controller.report(),
                        "final": _pin(env),
                    },
                    natural_entry=False,
                )
            )

        screenshot = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, screenshot)
        final = _pin(env)
        return {
            "ok": ok,
            "bead": "rr-28p",
            "track": "assisted" if infinite_life else "clean",
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "entry": entry,
            "settle": settle,
            "path_frames": frames,
            "trail": trail,
            "controller": controller.report(),
            "final": final,
            "inventory_preserved": {
                name: entry[name] == final[name]
                for name in ("raft", "ladder", "bombs", "triforce")
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
    parser.add_argument("--from-state", default="Level4Complete")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--tag", default="l4_to_l5_assisted")
    args = parser.parse_args(argv)

    reports = []
    for trial in range(args.trials):
        tag = f"{args.tag}_t{trial}"
        report = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoint=args.save_state and trial == 0,
            tag=tag,
        )
        reports.append(report)
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"error={report.get('error')} frames={report.get('path_frames')} "
            f"final={(report.get('final') or {}).get('screen_hex')} "
            f"phase={(report.get('controller') or {}).get('phase')}",
            flush=True,
        )

    output = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(
        output,
        {
            "bead": "rr-28p",
            "segment": "l4_complete_to_l5_entrance",
            "natural_entry": False,
            "continuous_emulator_session": True,
            "track": "assisted" if args.infinite_life else "clean",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "trials": reports,
            "successes": sum(bool(report.get("ok")) for report in reports),
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
