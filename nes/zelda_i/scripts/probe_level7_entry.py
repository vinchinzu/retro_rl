"""Level 7 pond probe — plan-only by default, live walk is capability-gated.

Examples::

    uv run python zelda_i/scripts/probe_level7_entry.py --plan-only
    uv run python zelda_i/scripts/probe_level7_entry.py \
        --allow-missing-caps --infinite-life --save-state
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from zelda_i.level7_overworld import (
    LEVEL7_POND_HOPS,
    LEVEL7_TRIFORCE_BIT,
    OverworldToLevel7PondController,
    has_food,
    has_whistle,
    level7_overworld_stop,
    missing_entry_caps,
    planning_report,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_WHISTLE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

DEFAULT_START_STATE = "PostSwordStart"
DEFAULT_MAX_FRAMES = 30000
SAMPLE_PERIOD = 250


def _snap_dict(snap: ZeldaSnapshot, ram) -> dict:
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "heart_containers": snap.heart_containers,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "rupees": snap.rupees,
        "sword": snap.sword,
        "triforce": snap.triforce,
        "whistle": read_u8(ram, ADDR_WHISTLE),
        "food": read_u8(ram, ADDR_FOOD),
        "candle": read_u8(ram, ADDR_CANDLE),
    }


def _print_plan() -> int:
    report = planning_report()
    print("=== Level 7 entry — PLANNING (source hypotheses, not live) ===")
    print(json.dumps(report, indent=2))
    print()
    print("Required entry cap: whistle @", report["ram"]["whistle"])
    print("Mid-dungeon cap: food/bait @", report["ram"]["food"])
    print("Triforce bit:", hex(LEVEL7_TRIFORCE_BIT))
    print("Hypothesized bait shop:", report["screens_hypothesized"]["bait_shop"])
    print("Hypothesized pond:", report["screens_hypothesized"]["pond"])
    print()
    print("Live entry requires real ADDR_WHISTLE from L5 (no Clean poke).")
    print("Optional: walk pond screen without whistle and save OW_L7Pond only.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-only", action="store_true", default=False)
    p.add_argument(
        "--allow-missing-caps",
        action="store_true",
        help="Allow pond-map walk without Whistle (no entry claim)",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--from-state", default=None)
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument("--tag", default="l7_entry")
    args = p.parse_args()

    if args.plan_only or not (
        args.infinite_life
        or args.save_state
        or args.from_state
        or args.allow_missing_caps
    ):
        rc = _print_plan()
        if args.plan_only or not (
            args.infinite_life or args.save_state or args.from_state
        ):
            if not args.plan_only:
                print(
                    "\nRe-run with --allow-missing-caps to map the pond "
                    "without making an entry claim."
                )
            return rc

    from retro_harness.env import make_env, reset_obs, save_state
    from retro_harness.nes import nes_idle_action
    from retro_harness.segment_runner import (
        configure_headless,
        save_rgb_png,
        write_json_report,
    )
    from zelda_i.assist import UnlimitedHealthAssist
    from zelda_i.dungeon_trace import write_state_provenance

    configure_headless()
    start_state = args.from_state or DEFAULT_START_STATE
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if args.infinite_life else None
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        ram = env.get_ram()
        start = _snap_dict(read_snapshot(ram), ram)
        missing = missing_entry_caps(ram)
        refused = bool(missing and not args.allow_missing_caps)
        notes: list[str] = []
        if refused:
            notes.append(f"refused: missing {missing}")
        if not has_food(ram):
            notes.append("food/bait missing: interior clear remains gated")

        nav = OverworldToLevel7PondController(max_frames=args.max_frames)
        trail: list[dict] = [{"f": 0, **start}]
        reason_counts: Counter[str] = Counter()
        screenshots: list[str] = []
        last_signature = (start["level"], start["screen"], start["mode"])
        frames = 0

        while not refused and frames < args.max_frames:
            snap = read_snapshot(env.get_ram())
            signature = (snap.level, snap.screen, snap.mode)
            if signature != last_signature:
                sample = {"f": frames, **_snap_dict(snap, env.get_ram())}
                trail.append(sample)
                path = RECORDINGS_DIR / (
                    f"{args.tag}_f{frames}_lv{snap.level}_sc{snap.screen:02x}"
                    f"_m{snap.mode}.png"
                )
                save_rgb_png(obs, path)
                screenshots.append(str(path))
                last_signature = signature
            elif (
                frames
                and frames % SAMPLE_PERIOD == 0
                and sum("_sample_" in path for path in screenshots) < 12
            ):
                sample = {"f": frames, **_snap_dict(snap, env.get_ram())}
                trail.append(sample)
                path = RECORDINGS_DIR / f"{args.tag}_sample_f{frames}.png"
                save_rgb_png(obs, path)
                screenshots.append(str(path))

            if nav.success or nav.phase.name == "FAILED" or snap.mode == 17:
                break
            action = nav.step(snap)
            reason_counts[action.reason] += 1
            obs, *_ = env.step(action.action)
            frames += 1
            if assist is not None:
                assist.apply_env(env, frame=frames)

        final_snap = read_snapshot(env.get_ram())
        final = _snap_dict(final_snap, env.get_ram())
        success = not refused and level7_overworld_stop(final_snap) and nav.success
        final_path = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(obs, final_path)
        screenshots.append(str(final_path))

        saved_state = None
        provenance = None
        if success and args.save_state:
            path = save_state(env, GAME_DIR, GAME, "OW_L7Pond")
            saved_state = str(path)
            source = GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
            provenance = str(
                write_state_provenance(
                    path,
                    source_state_path=source,
                    request={
                        "segment": "start_to_l7_pond",
                        "bead": "rr-dnp",
                        "track": "assisted" if assist is not None else "clean",
                        "allow_missing_caps": args.allow_missing_caps,
                        "inventory_poke": False,
                        "progression_poke": False,
                    },
                    selected_trial={
                        "success": success,
                        "frames": frames,
                        "start": start,
                        "final": final,
                        "nav": nav.report(),
                    },
                )
            )

        report = {
            "tag": args.tag,
            "track": "assisted" if assist is not None else "clean",
            "assist_contract": "nes/zelda_i/docs/ASSIST_CONTRACT.md",
            "start_state": start_state,
            "start": start,
            "final": final,
            "frames": frames,
            "success": success,
            "refused": refused,
            "missing_entry_caps": missing,
            "allow_missing_caps": args.allow_missing_caps,
            "entry_attempted": False,
            "pond_screen_reached": level7_overworld_stop(final_snap),
            "has_whistle": has_whistle(env.get_ram()),
            "has_food": has_food(env.get_ram()),
            "hops": [
                {"target": h.target, "direction": h.direction} for h in LEVEL7_POND_HOPS
            ],
            "trail": trail,
            "reason_counts": dict(reason_counts),
            "nav": nav.report(),
            "notes": notes,
            "assist": assist.report() if assist is not None else None,
            "controller_writes": {
                "inventory": 0,
                "progression": 0,
                "capacity": 0,
                "position": 0,
                "door": 0,
            },
            "saved_state": saved_state,
            "provenance": provenance,
            "screenshots": screenshots,
        }
        out = RECORDINGS_DIR / f"{args.tag}.json"
        write_json_report(out, report)
        print(json.dumps(report, indent=2))
        return 0 if success else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
