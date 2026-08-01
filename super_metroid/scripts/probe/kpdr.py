#!/usr/bin/env python3
"""KPDR probes: active controller route gets Hi-Jump before Kraid.

```bash
# Safer pure composition: Warehouse → Hi-Jump → Warehouse → Kraid entry
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state

# Full door-warp chain Big Pink main → Hi-Jump room (dev)
uv run python super_metroid/scripts/probe/kpdr.py route-to-hijump

# Shorter: Varia state → Hi-Jump (+ grant boots bit)
uv run python super_metroid/scripts/probe/kpdr.py varia-to-hijump

# Single hop from a save state
uv run python super_metroid/scripts/probe/kpdr.py hop hj-room \\
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_hj_shaft.state

# List hop ids
uv run python super_metroid/scripts/probe/kpdr.py list

# Refresh tracker JSON/MD from CSV
uv run python super_metroid/scripts/export/kpdr_tracker.py
```

The ``pure`` subcommand uses controller inputs and resource assists only.
Door-warp and item-grant subcommands remain development-only topology tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    place_samus,
    save_dev_state,
)
from super_metroid.dev.kpdr_dev import (  # noqa: E402
    BIG_PINK_MAIN,
    HJ_COLLECTED_DEV,
    HJ_ENTRY,
    HOP_BY_ID,
    KPDR_TO_HIJUMP,
    VARIA_STATE,
    hop_once,
    route_to_hijump,
    route_varia_to_hijump,
)
from super_metroid.ram import (  # noqa: E402
    parse_counts,
    parse_env_state,
    probe_pin,
    reset_parse_counts,
)
from super_metroid.source_states import (  # noqa: E402
    match_source_by_path,
    suggest_source_path,
    validate_fingerprint,
)
from super_metroid.routes.kpdr_controller import (  # noqa: E402
    play_baby_to_kihunter_return,
    play_bat_to_below_spazer,
    play_business_to_frog_save,
    play_below_spazer_to_west,
    play_big_pink_to_ghz,
    play_east_to_warehouse,
    play_eye_to_baby_return,
    play_frog_save_to_speedway,
    play_glass_to_east,
    play_ghz_to_noob,
    play_kraid_to_eye_return,
    play_kihunter_to_zeela_return,
    play_noob_to_red_tower,
    play_red_tower_to_bat,
    play_red_tower_to_warehouse,
    play_varia_to_kraid,
    play_warehouse_hijump_kraid,
    play_warehouse_to_hijump,
    play_warehouse_to_kraid_with_hijump,
    play_hijump_to_warehouse,
    play_hj_shaft_to_business,
    play_business_to_warehouse,
    play_warehouse_to_business,
    play_warehouse_wall_to_lower_lip,
    play_west_to_glass,
    play_zeela_to_warehouse_return,
)


_REVERSE_ISOLATION_COMMANDS = (
    (
        "K3.3",
        "kraid-to-eye-return",
        "post_varia_to_kraid_pure.state",
        "0xA59F",
    ),
    (
        "K3.4",
        "eye-to-baby-return",
        "post_kraid_to_eye_return.state",
        "0xA56B",
    ),
    (
        "K3.5",
        "baby-to-kihunter-return",
        "post_eye_to_baby_return.state",
        "0xA521",
    ),
    (
        "K3.6",
        "kihunter-to-zeela-return",
        "post_baby_to_kihunter_return.state",
        "0xA4DA",
    ),
)


class _ProbeSession:
    """Minimal ControllerSession for pure play probes.

    Uses ``mode="nav"`` only — pure geometry does not need bank-$7E copies
    every frame. Prefer this over bare full ``parse_env_state`` in hot loops.
    """

    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        return self.state


def _run_pure(
    *,
    source: Path,
    play,
    output: Path | None,
    place_x: int | None = None,
    place_y: int = 171,
    expect_room: int | None = None,
    segment: str = "",
    pin_json: Path | None = None,
) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    session: _ProbeSession | None = None
    reset_parse_counts()
    catalog = match_source_by_path(source)
    expected = expect_room
    if expected is None and catalog is not None:
        expected = catalog.room_id
    try:
        boot_from_state(env, source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
        if place_x is not None:
            place_samus(env, place_x, place_y)
            for _ in range(15):
                env.step(idle_action())
                assist.apply(env.data, parse_env_state(env, mode="nav"))
        session = _ProbeSession(env, assist)
        if expected is not None:
            check = validate_fingerprint(
                session.state,
                expected_room=expected,
                source=catalog,
            )
            if not check.ok:
                report = {
                    "success": False,
                    "error": "source fingerprint failed: " + "; ".join(check.failures),
                    "fingerprint": {
                        "ok": False,
                        "failures": list(check.failures),
                        "sourceId": check.source_id,
                    },
                    "probePin": check.pin,
                    "suggestedSource": (
                        str(suggest_source_path(expected, segment_hint=segment))
                        if expected is not None
                        else None
                    ),
                    "parseCounts": parse_counts(),
                    "controllerOnly": place_x is None,
                    "developmentOnly": place_x is not None,
                }
                if pin_json is not None:
                    pin_json.parent.mkdir(parents=True, exist_ok=True)
                    pin_json.write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                    report["pinJson"] = str(pin_json)
                return report
        # Satisfy ControllerSession protocol used by _hold
        play(session)  # type: ignore[arg-type]
        st = session.state
        if output is not None:
            save_dev_state(env, output)
        pin = probe_pin(st)
        return {
            "success": True,
            "roomIdHex": f"0x{st.room_id:04X}",
            "samusX": st.samus_x,
            "samusY": st.samus_y,
            "pose": st.pose,
            "doorTransition": st.door_transition,
            "frame": session.frame,
            "frames": session.frame,
            "probePin": pin,
            "parseCounts": parse_counts(),
            "statePath": str(output.resolve()) if output else None,
            "developmentOnly": place_x is not None,
            "controllerOnly": place_x is None,
            "placeX": place_x,
            "placeY": place_y if place_x is not None else None,
            "sourceId": catalog.source_id if catalog else None,
        }
    except Exception as exc:  # noqa: BLE001 — probe surface
        st = session.state if session is not None else parse_env_state(env, mode="nav")
        pin = probe_pin(st)
        report = {
            "success": False,
            "error": str(exc),
            "roomIdHex": f"0x{st.room_id:04X}",
            "samusX": st.samus_x,
            "samusY": st.samus_y,
            "pose": st.pose,
            "doorTransition": st.door_transition,
            "frame": session.frame if session is not None else st.frame,
            "frames": session.frame if session is not None else 0,
            "probePin": pin,
            "parseCounts": parse_counts(),
            "controllerOnly": place_x is None,
            "developmentOnly": place_x is not None,
            "sourceId": catalog.source_id if catalog else None,
            # Residual-friendly one-liner for PROCESS schema.
            "residualPinLine": (
                f"room=0x{st.room_id:04X} pose={st.pose} "
                f"x={st.samus_x} y={st.samus_y} "
                f"door_transition={st.door_transition} frames="
                f"{session.frame if session is not None else 0}"
            ),
        }
        if pin_json is not None:
            pin_json.parent.mkdir(parents=True, exist_ok=True)
            pin_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["pinJson"] = str(pin_json)
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List hop ids on KPDR → Hi-Jump warp chain")

    rth = sub.add_parser(
        "route-to-hijump",
        help="Door-warp Big Pink → Kraid → Varia → Hi-Jump room",
    )
    rth.add_argument("--source", type=Path, default=BIG_PINK_MAIN)
    rth.add_argument("--output", type=Path, default=HJ_ENTRY)
    rth.add_argument("--no-save-hops", action="store_true")
    rth.add_argument(
        "--grant-hijump",
        action="store_true",
        help="Grant Hi-Jump item bit at end (dev)",
    )

    vth = sub.add_parser(
        "varia-to-hijump",
        help="Door-warp from Varia state to Hi-Jump room",
    )
    vth.add_argument("--source", type=Path, default=VARIA_STATE)
    vth.add_argument("--output", type=Path, default=HJ_ENTRY)
    vth.add_argument("--no-grant-hijump", action="store_true")

    hop = sub.add_parser("hop", help="Single door-warp hop")
    hop.add_argument(
        "hop_id",
        choices=sorted(HOP_BY_ID),
        help="Hop id (see list)",
    )
    hop.add_argument("--source", type=Path, required=True)
    hop.add_argument("--output", type=Path, default=None)

    col = sub.add_parser(
        "collect-hijump",
        help="Warp Varia→Hi-Jump and grant boots bit (dev)",
    )
    col.add_argument("--source", type=Path, default=VARIA_STATE)
    col.add_argument("--output", type=Path, default=HJ_COLLECTED_DEV)

    sub.add_parser(
        "iso-reverse",
        help="List pure reverse-hop probes and their cataloged source states",
    )

    pure = sub.add_parser(
        "pure",
        help="Controller-only room exit (no door-warp during segment)",
    )
    pure.add_argument(
        "segment",
        choices=(
            "big-pink-to-ghz",
            "ghz-to-noob",
            "noob-to-red",
            "red-to-bat",
            "bat-to-below",
            "below-to-west",
            "west-to-glass",
            "glass-to-east",
            "east-to-warehouse",
            "red-to-warehouse",
            "warehouse-wall",
            "warehouse-to-hijump",
            "hijump-to-warehouse",
            "warehouse-to-kraid-hijump",
            "warehouse-hijump-kraid",
            "hj-shaft-to-business",
            "business-to-warehouse",
            "warehouse-to-business",
            "varia-to-kraid",
            "kraid-to-eye-return",
            "eye-to-baby-return",
            "baby-to-kihunter-return",
            "kihunter-to-zeela-return",
            "zeela-to-warehouse-return",
            "business-to-frog-save",
            "frog-save-to-speedway",
        ),
    )
    pure.add_argument("--source", type=Path, required=True)
    pure.add_argument("--output", type=Path, default=None)
    pure.add_argument(
        "--expect-room",
        type=lambda s: int(s, 0),
        default=None,
        help="Expected entry room id (hex 0x… or int); defaults from SOURCE catalog",
    )
    pure.add_argument(
        "--pin-json",
        type=Path,
        default=None,
        help="On RED / fingerprint fail, write probe pin JSON here",
    )
    pure.add_argument(
        "--place-x",
        type=int,
        default=None,
        help="Dev only: place Samus at this room X before play",
    )
    pure.add_argument(
        "--place-y",
        type=int,
        default=171,
        help="Dev only: place Y used with --place-x (default 171)",
    )
    suggest = sub.add_parser(
        "suggest-source",
        help="Suggest SOURCE_STATES catalog path for a room / segment",
    )
    suggest.add_argument(
        "--room",
        type=lambda s: int(s, 0),
        required=True,
        help="Entry room id (hex 0x… or int)",
    )
    suggest.add_argument(
        "--segment",
        type=str,
        default="",
        help="Optional segment hint (e.g. varia-to-kraid)",
    )

    args = parser.parse_args()

    if args.command == "list":
        for name, door, room, px, py in KPDR_TO_HIJUMP:
            print(f"{name:20} door=0x{door:04X} room=0x{room:04X} place=({px},{py})")
        return

    if args.command == "route-to-hijump":
        report = route_to_hijump(
            source=args.source,
            output=args.output,
            save_hops=not args.no_save_hops,
            grant_hijump=args.grant_hijump,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    if args.command == "varia-to-hijump":
        report = route_varia_to_hijump(
            source=args.source,
            output=args.output,
            grant_hijump=not args.no_grant_hijump,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    if args.command == "hop":
        report = hop_once(
            hop_id=args.hop_id,
            source=args.source,
            output=args.output,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    if args.command == "collect-hijump":
        report = route_varia_to_hijump(
            source=args.source,
            output=args.output,
            grant_hijump=True,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    if args.command == "iso-reverse":
        source_root = "super_metroid/custom_integrations/SuperMetroid-Snes/scratch"
        print("Reverse pure isolation matrix (diagnostic; not continuous evidence):")
        for hop, segment, source_name, room in _REVERSE_ISOLATION_COMMANDS:
            print(f"{hop}: expected source room {room}")
            print(
                "  uv run python super_metroid/scripts/probe/kpdr.py pure "
                f"{segment} --source {source_root}/{source_name}"
            )
        return

    if args.command == "suggest-source":
        from super_metroid.source_states import suggest_sources_for_room

        ranked = suggest_sources_for_room(
            args.room, segment_hint=args.segment, continuous_like_only=False
        )
        payload = {
            "roomIdHex": f"0x{args.room:04X}",
            "segmentHint": args.segment,
            "suggestions": [
                {
                    "sourceId": s.source_id,
                    "path": str(s.path),
                    "roomIdHex": s.room_hex(),
                    "useFor": s.use_for,
                    "continuousLike": s.continuous_like,
                    "exists": s.path.is_file(),
                }
                for s in ranked
            ],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(0 if ranked else 1)

    if args.command == "pure":
        play_fn = {
            "big-pink-to-ghz": play_big_pink_to_ghz,
            "ghz-to-noob": play_ghz_to_noob,
            "noob-to-red": play_noob_to_red_tower,
            "red-to-bat": play_red_tower_to_bat,
            "bat-to-below": play_bat_to_below_spazer,
            "below-to-west": play_below_spazer_to_west,
            "west-to-glass": play_west_to_glass,
            "glass-to-east": play_glass_to_east,
            "east-to-warehouse": play_east_to_warehouse,
            "red-to-warehouse": play_red_tower_to_warehouse,
            "warehouse-wall": play_warehouse_wall_to_lower_lip,
            "warehouse-to-hijump": play_warehouse_to_hijump,
            "hijump-to-warehouse": play_hijump_to_warehouse,
            "warehouse-to-kraid-hijump": play_warehouse_to_kraid_with_hijump,
            "warehouse-hijump-kraid": play_warehouse_hijump_kraid,
            "hj-shaft-to-business": play_hj_shaft_to_business,
            "business-to-warehouse": play_business_to_warehouse,
            "warehouse-to-business": play_warehouse_to_business,
            "varia-to-kraid": play_varia_to_kraid,
            "kraid-to-eye-return": play_kraid_to_eye_return,
            "eye-to-baby-return": play_eye_to_baby_return,
            "baby-to-kihunter-return": play_baby_to_kihunter_return,
            "kihunter-to-zeela-return": play_kihunter_to_zeela_return,
            "zeela-to-warehouse-return": play_zeela_to_warehouse_return,
            "business-to-frog-save": play_business_to_frog_save,
            "frog-save-to-speedway": play_frog_save_to_speedway,
        }[args.segment]
        report = _run_pure(
            source=args.source,
            play=play_fn,
            output=args.output,
            place_x=args.place_x,
            place_y=args.place_y,
            expect_room=args.expect_room,
            segment=args.segment,
            pin_json=args.pin_json,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
