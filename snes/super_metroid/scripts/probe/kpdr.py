#!/usr/bin/env python3
"""KPDR probes: load a pin, play a Skill, compare RAM.

```bash
# Safer pure composition: Warehouse → Hi-Jump → Warehouse → Kraid entry
uv run python snes/super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state

# Chain Ice return + K5 + K6 from the Ice leave pin (not continuous evidence)
uv run python snes/super_metroid/scripts/probe/kpdr.py compose ice-to-moat \\
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_ceres_successor.state

# Remaining K6 hops from the Ice-pin Alpha PB leave
uv run python snes/super_metroid/scripts/probe/kpdr.py compose alpha-pb-to-moat \\
  --source snes/super_metroid/scratch/post_ice_to_alpha_pb_compose.state

# Over-ocean spark from the power-on Moat leave (West Ocean → 0xCA08)
uv run python snes/super_metroid/scripts/probe/kpdr.py compose moat-to-ws \\
  --source snes/super_metroid/scratch/post_moat_poweron.state

# List hop ids
uv run python snes/super_metroid/scripts/probe/kpdr.py list

# Refresh tracker JSON/MD from CSV
uv run python snes/super_metroid/scripts/export/kpdr_tracker.py
```

The ``pure`` subcommand uses controller inputs and resource assists only.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.headed import add_headed_flag, attach_headed, idle_headed  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    place_samus,
    save_dev_state,
)
from super_metroid.ram import (  # noqa: E402
    parse_counts,
    parse_env_state,
    probe_pin,
    reset_parse_counts,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS, get_segment  # noqa: E402
from super_metroid.routes.kpdr.to_bat_cave import (  # noqa: E402
    BubblePhaseStop,
    play_bubble_climb_from_handoff,
    play_bubble_from_top_door,
    play_bubble_to_bat_cave_with_phase_capture,
)
from super_metroid.scripts.probe.red_diag import (  # noqa: E402
    DEFAULT_RING_FRAMES,
    FrameRing,
    attach_red_diag,
    capture_red_artifacts,
    default_red_diag_dir,
    display_path,
)
from super_metroid.source_states import (  # noqa: E402
    match_source_by_path,
    suggest_source_path,
    validate_fingerprint,
)

# CLI short names that do not 1:1 hyphenate a KPDR_SEGMENTS key.
_PURE_ALIASES = {
    "noob-to-red": "noob_to_red_tower",
    "red-to-bat": "red_tower_to_bat",
    "bat-to-below": "bat_to_below_spazer",
    "below-to-west": "below_spazer_to_west",
    "red-to-warehouse": "red_tower_to_warehouse",
    "warehouse-wall": "warehouse_wall_to_lower_lip",
    "warehouse-to-kraid-hijump": "warehouse_to_kraid_with_hijump",
}


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

    When ``ring_frames`` > 0, keeps a short RGB ring for pure RED auto-capture
    (frame dump; no geometry side effects).
    """

    def __init__(
        self,
        env,
        assist: UnlimitedResourcesAssist,
        *,
        ring_frames: int = DEFAULT_RING_FRAMES,
    ) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.frame_ring = FrameRing(maxlen=ring_frames) if ring_frames > 0 else None

    def step(self, action, reason: str = ""):
        del reason
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        if self.frame_ring is not None:
            self.frame_ring.push(obs)
        return self.state


def _capture_pure_red(
    *,
    env,
    state,
    session: _ProbeSession | None,
    segment: str,
    source: Path,
    error: str,
    pin_json: Path | None,
    report: dict[str, object],
    red_diag: bool,
) -> dict[str, object]:
    """Attach auto-captured frame dump + door/PLM snapshot on pure RED."""
    if not red_diag:
        if pin_json is not None:
            pin_json.parent.mkdir(parents=True, exist_ok=True)
            pin_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["pinJson"] = display_path(pin_json)
        return report
    frames = session.frame_ring.frames() if session and session.frame_ring else []
    # Fingerprint fail before play: grab a few live frames for visual context.
    if not frames:
        try:
            for _ in range(3):
                obs, *_ = env.step(idle_action())
                frames.append(obs)
        except Exception:  # noqa: BLE001 — diagnostic best-effort
            frames = []
    out_dir = default_red_diag_dir(segment=segment or "pure")
    try:
        artifacts = capture_red_artifacts(
            env=env,
            state=state,
            frames=frames,
            segment=segment,
            error=error,
            source=display_path(source),
            probe_frames=int(report.get("frames") or 0),
            out_dir=out_dir,
            pin_json=pin_json,
            report=report,
            write_pin=True,
        )
        attach_red_diag(report, artifacts)
    except Exception as diag_exc:  # noqa: BLE001 — never mask the original RED
        report["redDiagError"] = str(diag_exc)
        if pin_json is not None:
            pin_json.parent.mkdir(parents=True, exist_ok=True)
            pin_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["pinJson"] = display_path(pin_json)
    return report


def _play_named_chain(chain: str):
    """Play spine hops for a named post-Ice compose chain."""
    from super_metroid.routes.kpdr.spine import hops_for_tip

    if chain == "ice-to-alpha-pb":
        hops = hops_for_tip("alpha_pb")
    elif chain == "ice-to-moat":
        hops = hops_for_tip("alpha_pb") + hops_for_tip("moat")
    elif chain == "alpha-pb-to-moat":
        hops = hops_for_tip("moat")
    elif chain == "ice-to-ws":
        hops = hops_for_tip("alpha_pb") + hops_for_tip("moat") + hops_for_tip("ws")
    elif chain == "moat-to-ws":
        hops = hops_for_tip("ws")
    elif chain == "ws-to-phantoon":
        hops = hops_for_tip("phantoon")
    elif chain == "ice-to-phantoon":
        hops = (
            hops_for_tip("alpha_pb")
            + hops_for_tip("moat")
            + hops_for_tip("ws")
            + hops_for_tip("phantoon")
        )
    elif chain == "phantoon-to-gravity":
        hops = hops_for_tip("gravity")
    else:
        raise KeyError(chain)

    def play(session):
        for hop in hops:
            hop.play(session)
            if hop.after is not None:
                hop.after(session, [], None)
            st = session.state
            if st.room_id != hop.to_room:
                raise RuntimeError(
                    f"{hop.hop_id}: expected room 0x{hop.to_room:04X}, "
                    f"got 0x{st.room_id:04X} {st}"
                )
            print(
                f"[compose] {hop.hop_id} -> 0x{st.room_id:04X} "
                f"({st.samus_x},{st.samus_y}) p{st.pose} f={session.frame}",
                flush=True,
            )
        return session.state

    play.__name__ = f"play_{chain.replace('-', '_')}"
    play.__qualname__ = play.__name__
    return play


def _headed_hud(env) -> str:
    st = parse_env_state(env, mode="nav")
    max_hp = int(getattr(st, "max_health", 0) or 0)
    return (
        f"BOT  0x{int(st.room_id):04X} ({int(st.samus_x)},{int(st.samus_y)}) "
        f"p{int(st.pose)} gs={int(st.game_state)} hp={int(st.health)}/{max_hp}"
    )


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
    red_diag: bool = True,
    ring_frames: int = DEFAULT_RING_FRAMES,
    phase_capture: bool = False,
    headed: bool = False,
) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    session: _ProbeSession | None = None
    pygame_mod = None
    reset_parse_counts()
    catalog = match_source_by_path(source)
    expected = expect_room
    if expected is None and catalog is not None:
        expected = catalog.room_id
    try:
        assist.attach_env(env)
        if headed:
            pygame_mod = attach_headed(
                env, title=f"SM BOT: {segment or 'pure'}", hud=_headed_hud
            )
        # Open-loop hop bodies desync if we idle after a live pin
        # (human-tape default boot-settle is 0). RAM controllers tolerate 5.
        zero_settle_segments = {
            "bat-to-red",
            "red-to-hellway",
            "hellway-to-caterpillar",
            "caterpillar-to-elevator",
            "elevator-to-kihunter",
            "kihunter-to-moat",
            "ws-to-phantoon",
            "phantoon-to-gravity",
            "gravity-collect",
            "attic-to-west-ocean",
            "west-ocean-to-pancakes",
            "pancakes-to-homing-geemer",
            "homing-geemer-to-bowling",
            "bowling-to-gravity",
        }
        boot_settle = 0 if segment in zero_settle_segments else 5
        boot_from_state(env, source, settle_frames=boot_settle)
        for _ in range(boot_settle):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
        if place_x is not None:
            place_samus(env, place_x, place_y)
            for _ in range(15):
                env.step(idle_action())
                assist.apply(env.data, parse_env_state(env, mode="nav"))
        session = _ProbeSession(
            env,
            assist,
            ring_frames=ring_frames if red_diag else 0,
        )
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
                    "frames": session.frame,
                    "sourceId": catalog.source_id if catalog else None,
                }
                return _capture_pure_red(
                    env=env,
                    state=session.state,
                    session=session,
                    segment=segment,
                    source=source,
                    error=str(report["error"]),
                    pin_json=pin_json,
                    report=report,
                    red_diag=red_diag,
                )
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
    except BubblePhaseStop as phase_stop:
        # Capture / recon early stop — diagnostic success only when requested.
        st = phase_stop.state
        pin = probe_pin(st)
        metrics = dict(phase_stop.metrics)
        dump = metrics.get("dump_phase_c")
        report = {
            "success": bool(phase_capture),
            "phaseStop": phase_stop.phase,
            "phaseCapture": True,
            "phaseCHit": True,
            "error": None if phase_capture else str(phase_stop),
            "roomIdHex": f"0x{st.room_id:04X}",
            "samusX": st.samus_x,
            "samusY": st.samus_y,
            "pose": st.pose,
            "velocityX": int(st.velocity_x),
            "velocityY": int(st.velocity_y),
            "doorTransition": st.door_transition,
            "frame": session.frame if session is not None else st.frame,
            "frames": session.frame if session is not None else 0,
            "probePin": pin,
            "phaseMetrics": metrics,
            "statePath": str(Path(dump).resolve()) if dump else None,
            "parseCounts": parse_counts(),
            "controllerOnly": place_x is None,
            "developmentOnly": True,
            "notHopGreen": True,
            "sourceId": catalog.source_id if catalog else None,
            "residualPinLine": (
                f"room=0x{st.room_id:04X} pose={st.pose} "
                f"x={st.samus_x} y={st.samus_y} "
                f"vx={st.velocity_x} vy={st.velocity_y} "
                f"door_transition={st.door_transition} "
                f"phase_stop={phase_stop.phase} frames="
                f"{session.frame if session is not None else 0}"
            ),
        }
        if pin_json is not None:
            pin_json.parent.mkdir(parents=True, exist_ok=True)
            pin_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["pinJson"] = display_path(pin_json)
        return report
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
        return _capture_pure_red(
            env=env,
            state=st,
            session=session,
            segment=segment,
            source=source,
            error=str(exc),
            pin_json=pin_json,
            report=report,
            red_diag=red_diag,
        )
    finally:
        if headed and pygame_mod is not None:
            idle_headed(env, pygame_mod)
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List pure hop ids (KPDR_SEGMENTS + CLI aliases)")

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
        choices=sorted(
            {k.replace("_", "-") for k in KPDR_SEGMENTS} | set(_PURE_ALIASES)
        ),
    )
    add_headed_flag(pure)
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
        "--no-red-diag",
        action="store_true",
        help="Disable pure-RED frame dump + door/PLM snapshot auto-capture",
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
    pure.add_argument(
        "--start-phase",
        choices=("auto", "full", "climb", "door"),
        default="auto",
        help=(
            "Bubble→Bat only: auto/full natural path; climb skips lower/"
            "repin/launch (Phase-C handoff iteration); door skips to Super door"
        ),
    )
    pure.add_argument(
        "--dump-phase-c",
        type=Path,
        default=None,
        help=(
            "Bubble→Bat only: write first Phase-C usable-right-contact "
            "save-state (dev handoff; not hop GREEN)"
        ),
    )
    pure.add_argument(
        "--stop-at-phase-c",
        action="store_true",
        help=(
            "Bubble→Bat only: stop at first Phase C (capture/recon). "
            "Exit 0 on hit; not hop GREEN to Bat"
        ),
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

    compose = sub.add_parser(
        "compose",
        help="Chain post-Ice spine hops from a pin (not continuous evidence)",
    )
    compose.add_argument(
        "chain",
        choices=(
            "ice-to-alpha-pb",
            "ice-to-moat",
            "alpha-pb-to-moat",
            "ice-to-ws",
            "moat-to-ws",
            "ws-to-phantoon",
            "ice-to-phantoon",
            "phantoon-to-gravity",
        ),
        help=(
            "Named hop chain (alpha_pb, alpha_pb+moat, moat from Alpha PB leave, "
            "Ice→WS, West Ocean spark from Moat leave, WS interior+fight, "
            "or Ice→Phantoon)"
        ),
    )
    compose.add_argument("--source", type=Path, required=True)
    compose.add_argument("--output", type=Path, default=None)
    compose.add_argument(
        "--expect-room",
        type=lambda s: int(s, 0),
        default=None,
        help="Override source fingerprint room (hex 0x… or int)",
    )
    compose.add_argument("--pin-json", type=Path, default=None)
    compose.add_argument("--no-red-diag", action="store_true")
    add_headed_flag(compose)

    args = parser.parse_args()

    if args.command == "list":
        names = {k.replace("_", "-") for k in KPDR_SEGMENTS}
        names.update(_PURE_ALIASES)
        for name in sorted(names):
            print(name)
        return

    if args.command == "iso-reverse":
        source_root = "super_metroid/custom_integrations/SuperMetroid-Snes/scratch"
        print("Reverse pure isolation matrix (diagnostic; not continuous evidence):")
        for hop, segment, source_name, room in _REVERSE_ISOLATION_COMMANDS:
            print(f"{hop}: expected source room {room}")
            print(
                "  uv run python snes/super_metroid/scripts/probe/kpdr.py pure "
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

    if args.command == "compose":
        report = _run_pure(
            source=args.source,
            play=_play_named_chain(args.chain),
            output=args.output,
            expect_room=args.expect_room,
            segment=args.chain,
            pin_json=args.pin_json,
            red_diag=not args.no_red_diag,
            headed=bool(getattr(args, "headed", False)),
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    if args.command == "pure":
        play_fn = get_segment(
            _PURE_ALIASES.get(args.segment, args.segment.replace("-", "_"))
        )
        bubble_phase_opts = (
            args.start_phase != "auto"
            or args.dump_phase_c is not None
            or args.stop_at_phase_c
        )
        if bubble_phase_opts and args.segment != "bubble-to-bat-cave":
            parser.error(
                "--start-phase / --dump-phase-c / --stop-at-phase-c "
                "only apply to bubble-to-bat-cave"
            )
        if args.segment == "bubble-to-bat-cave" and bubble_phase_opts:
            # Map CLI flags to dev helpers — never kwargs on product play.
            phase = (args.start_phase or "auto").strip().lower()
            if phase in ("climb",):
                play_fn = functools.partial(
                    play_bubble_climb_from_handoff,
                    dump_phase_c=args.dump_phase_c,
                    stop_at_phase_c=args.stop_at_phase_c,
                )
            elif phase in ("door",):
                if args.dump_phase_c is not None or args.stop_at_phase_c:
                    parser.error(
                        "--dump-phase-c / --stop-at-phase-c not used with "
                        "--start-phase door"
                    )
                play_fn = play_bubble_from_top_door
            elif phase in ("auto", "full"):
                play_fn = functools.partial(
                    play_bubble_to_bat_cave_with_phase_capture,
                    dump_phase_c=args.dump_phase_c,
                    stop_at_phase_c=args.stop_at_phase_c,
                )
            else:
                parser.error(
                    f"unknown --start-phase {args.start_phase!r} "
                    "(use auto|full|climb|door)"
                )
        report = _run_pure(
            source=args.source,
            play=play_fn,
            output=args.output,
            place_x=args.place_x,
            place_y=args.place_y,
            expect_room=args.expect_room,
            segment=args.segment,
            pin_json=args.pin_json,
            red_diag=not args.no_red_diag,
            phase_capture=bool(args.stop_at_phase_c),
            headed=bool(getattr(args, "headed", False)),
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report.get("success") else 1)

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
