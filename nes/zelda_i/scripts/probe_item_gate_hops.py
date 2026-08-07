"""Assisted probe: walk early OW item-gate hop tables.

Geometry only — does not buy shops or enter caves for items. Reports which
end screens are reached under optional ``--infinite-life``.

Examples::

    # Candle shop near start (0x66), Survival assist
    PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_item_gate_hops.py \\
        --route candle_shop_near --infinite-life --tag ig_candle

    # White sword cave screen (0x0A) + heart-gate report
    PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_item_gate_hops.py \\
        --route white_sword --infinite-life --tag ig_ws

    # Bomb shop (0x6F) via 0x5C maze corridor
    PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_item_gate_hops.py \\
        --route bomb_shop --infinite-life --tag ig_bomb

    # All routes sequentially
    PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_item_gate_hops.py \\
        --route all --infinite-life --tag ig_all
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.item_gate_hops import (
    ITEM_GATE_ROUTES,
    SEGMENT_MAX_FRAMES,
    ItemGateHopController,
    gate_report_snapshot,
    route_for,
    screen_reached,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_one(
    *,
    route_name: str,
    start_state: str,
    max_frames: int,
    infinite_life: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    route = route_for(route_name)

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        entry_snap = read_snapshot(env.get_ram())
        entry = gate_report_snapshot(entry_snap, env.get_ram())
        trail: list[dict] = [{"f": 0, **entry}]
        last_screen = entry["screen"]
        last_level = entry_snap.level

        nav = ItemGateHopController(route_name=route_name, max_frames=max_frames)
        frames = 0
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen or snap.level != last_level:
                trail.append(
                    {"f": frames, **gate_report_snapshot(snap, env.get_ram())}
                )
                last_screen = snap.screen
                last_level = snap.level
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR
                    / f"{tag}_{route_name}_sc{snap.screen:02x}.png",
                )
            if snap.mode == 17:
                break
            if nav.success or (
                hasattr(nav.phase, "name") and nav.phase.name == "FAILED"
            ):
                break
            act = nav.step(snap)
            obs, *_ = env.step(act.action)
            frames += 1
            if assist is not None:
                assist.apply_env(env, frame=frames)
            if nav.success or (
                hasattr(nav.phase, "name") and nav.phase.name == "FAILED"
            ):
                break

        final_snap = read_snapshot(env.get_ram())
        final = gate_report_snapshot(final_snap, env.get_ram())
        # Prefer controller success (handles edge y after scroll); also accept
        # plain screen match with a looser y band for settle screens.
        reached = bool(nav.success) or (
            final_snap.level == 0
            and final_snap.mode == 5
            and final_snap.screen == route.end
        )

        report = {
            "tag": tag,
            "route": route_name,
            "track": track,
            "assist_contract": "nes/zelda_i/docs/ASSIST_CONTRACT.md",
            "start_state": start_state,
            "planned_end": f"0x{route.end:02x}",
            "verification_label": route.verification,
            "requires_note": route.requires_note,
            "hops": [
                {
                    "target": h.target,
                    "direction": h.direction,
                    "align_x": h.align_x,
                    "align_y": h.align_y,
                    "y_band": h.y_band,
                }
                for h in route.hops
            ],
            "screens": [f"0x{s:02x}" for s in route.screens],
            "entry": entry,
            "final": final,
            "trail": trail,
            "nav": nav.report(),
            "frames": frames,
            "end_screen_reached": reached,
            "heart_gate": {
                "min_containers": route.min_heart_containers,
                "entry_containers": entry.get("heart_containers"),
                "final_containers": final.get("heart_containers"),
                "blocks_at_entry": entry.get("white_sword_heart_gate_blocks"),
                "blocks_at_final": final.get("white_sword_heart_gate_blocks"),
                "note": (
                    "infinite-life refills filled hearts only; containers "
                    "unchanged (ASSIST_CONTRACT)"
                ),
            },
            "assist": assist.report() if assist is not None else None,
            "success": reached,
        }
        out = RECORDINGS_DIR / f"{tag}_{route_name}.json"
        write_json_report(out, report)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{route_name}_final.png")
        print(
            f"[{tag}/{route_name}] track={track} reached={reached} "
            f"planned=0x{route.end:02x} final=0x{final['screen']:02x} "
            f"frames={frames} containers={final.get('heart_containers')} "
            f"gate_blocks={final.get('white_sword_heart_gate_blocks')}"
        )
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--route",
        default="candle_shop_near",
        choices=(*sorted(ITEM_GATE_ROUTES), "all"),
        help="Which hop table to walk (or all)",
    )
    p.add_argument("--from-state", default="PostSwordStart")
    p.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    p.add_argument("--tag", default="item_gate")
    args = p.parse_args(argv)

    names = (
        list(ITEM_GATE_ROUTES)
        if args.route == "all"
        else [args.route]
    )
    reports = []
    ok = True
    for name in names:
        rep = run_one(
            route_name=name,
            start_state=args.from_state,
            max_frames=args.max_frames,
            infinite_life=args.infinite_life,
            tag=args.tag,
        )
        reports.append(rep)
        if not rep.get("success"):
            ok = False

    if args.route == "all":
        summary = {
            "tag": args.tag,
            "routes": {
                r["route"]: {
                    "success": r["success"],
                    "final": r["final"]["screen_hex"],
                    "frames": r["frames"],
                    "heart_containers": r["final"].get("heart_containers"),
                }
                for r in reports
            },
        }
        write_json_report(RECORDINGS_DIR / f"{args.tag}_summary.json", summary)
        print(f"[{args.tag}] summary={summary['routes']}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
