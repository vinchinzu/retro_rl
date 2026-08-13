#!/usr/bin/env python3
"""Human path recorder with optional on-screen route guide.

Records SNES-12 frames + position trace + live anchors to
``tasks/<name>.json``. Product entry: ``./play`` (power-on / resume / varia).

```bash
./snes/super_metroid/play                         # power-on full_start_v1
./snes/super_metroid/play resume full_start_v1    # archives prior take first
uv run python snes/super_metroid/scripts/record/guided_human.py --list
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \\
  snes/super_metroid/tasks/full_start_v1.json --hop 0 --dual --promote-bank
```

Controls: ` toggles autopilot · controller L+R+Select toggles autopilot ·
F5/F1 save + materialize · F6 manual pin · ESC/Q cancel.
Reusing ``--name`` archives prior tape + hop bodies under
``tasks/<name>_segments/sN/``. Start presets: ``super_metroid.start_presets``.
Pipeline: ``docs/tasks/HUMAN_TAPE_PIPELINE.md``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from retro_harness.path_overlay import (  # noqa: E402
    draw_guide_path,
    draw_player_marker,
    nearest_waypoint_index,
    transform_from_session_ctx,
)
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from retro_harness.task_recording import (  # noqa: E402
    RecordedTask,
    pressed_buttons,
    summarize_position_trace,
)
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.human_tape import AnchorRecorder, fingerprint  # noqa: E402
from super_metroid.human_tape.anchors import parse_room_id  # noqa: E402
from super_metroid.human_tape.rta_clock import (  # noqa: E402
    fmt_time as rta_fmt_time,
    resolve_rta_clock,
)
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.rooms.canonical_names import (  # noqa: E402
    load_canonical_names,
    room_name,
)
from super_metroid.routes.kpdr.guides import (  # noqa: E402
    ROUTE_PRESETS,
    guide_for_room,
)

# Start presets live in start_presets.py (keeps this CLI under 1k lines).
from super_metroid.start_presets import (  # noqa: E402
    POWER_ON_STARTS,
    START_PRESETS,
)

# Durable item-seam pins (not overwritten by the next segment's F5 end dump).
_DURABLE_SEAM_PINS: tuple[tuple[int, str, int, str], ...] = (
    # items_mask_required, stem under scratch/, preferred_room, label
    (0x0004, "full_start_v1_morph", 0x9E9F, "Morph"),
    (0x1000, "full_start_v1_bomb", 0x9804, "Bombs"),
    (0x0100, "full_start_v1_hj", 0xA9E5, "HiJump"),
    (0x0001, "full_start_v1_varia", 0xA6E2, "Varia"),
    # Grapple bit 0x4000: collect room vs long return end (pref_room disambiguates).
    (0x4000, "full_start_v1_grapple", 0xAC2B, "Grapple"),
    (0x4000, "full_start_v1_main_street", 0xCFC9, "MainStreet"),
)

# Beam-bit seams (Plasma is collected_beams 0x0008, not an item bit).
_DURABLE_BEAM_SEAM_PINS: tuple[tuple[int, str, int, str], ...] = (
    (0x0008, "full_start_v1_plasma", 0xD2AA, "Plasma"),
)

# Layer 1 camera scroll (WRAM) — same as place_samus in dev/common.py.
ADDR_CAMERA_X = 0x0911
ADDR_CAMERA_Y = 0x0915

SCRATCH = INTEGRATION_DIR / "scratch"
TASKS_DIR = GAME_DIR / "tasks"


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _is_power_on(arg: str) -> bool:
    return arg in POWER_ON_STARTS


def _resolve_state(arg: str) -> Path:
    if _is_power_on(arg):
        raise ValueError(f"power-on start has no state file ({arg})")
    if arg in START_PRESETS:
        rel, _ = START_PRESETS[arg]
        return INTEGRATION_DIR / rel
    path = Path(arg)
    if path.is_file():
        return path.resolve()
    # Allow stem under scratch/ or integration root.
    candidates = [
        SCRATCH / f"{arg}.state",
        SCRATCH / arg,
        INTEGRATION_DIR / f"{arg}.state",
        INTEGRATION_DIR / arg,
        GAME_DIR / arg,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Start state not found: {arg}")


def _trace_row(env, frame: int, action) -> dict[str, object]:
    state = parse_env_state(env, frame=frame, mode="nav")
    return {
        "frame": frame,
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "room": int(state.room_id),
        "room_hex": f"0x{int(state.room_id):04X}",
        "pose": int(state.pose),
        "vx": int(state.velocity_x),
        "vy": int(state.velocity_y),
        "buttons": pressed_buttons(action),
        "energy": int(state.health),
        "missiles": int(state.missiles),
        "supers": int(state.super_missiles),
        "pbs": int(state.power_bombs),
        "selected": int(state.selected_item),
        "door_transition": int(state.door_transition),
        "phase": state.phase.value if hasattr(state.phase, "value") else str(state.phase),
        # Inventory on every row so item-delta / extract works without full RAM.
        "items": int(state.collected_items),
        "beams": int(state.collected_beams),
    }


def _fmt_time(frames: int) -> str:
    """60fps wall-time label: m:ss.mmm"""
    return rta_fmt_time(frames)


def _parse_items_field(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, int):
        return int(raw)
    try:
        return int(str(raw), 0)
    except (TypeError, ValueError):
        return 0


def _write_durable_seam_pins(
    env,
    *,
    end_fp: Mapping[str, Any] | None,
    state_bytes: bytes | None,
) -> list[str]:
    """Copy end state to durable scratch + tasks pins for known item seams."""
    written: list[str] = []
    blob = state_bytes
    if blob is None:
        try:
            blob = env.em.get_state()
        except Exception:
            return written
    items = 0
    beams = 0
    room = 0
    if end_fp:
        items = _parse_items_field(end_fp.get("items"))
        beams = _parse_items_field(end_fp.get("beams"))
        room = parse_room_id(end_fp.get("room")) or 0
    if not items or not beams:
        try:
            st = parse_env_state(env, mode="nav")
            items = items or int(st.collected_items)
            beams = beams or int(st.collected_beams)
            room = room or int(st.room_id)
        except Exception:
            if not items and not beams:
                return written

    def _dump(stem: str, label: str) -> None:
        for dest in (SCRATCH / f"{stem}.state", TASKS_DIR / f"{stem}.state"):
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                written.append(f"{label}→{dest}")
            except OSError as exc:
                written.append(f"{label} FAIL {dest}: {exc}")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    for mask, stem, pref_room, label in _DURABLE_SEAM_PINS:
        if not (items & mask):
            continue
        # Only write when we are in the chozo / collect room (or room unknown).
        if pref_room and room and int(room) != int(pref_room):
            continue
        _dump(stem, label)
    for mask, stem, pref_room, label in _DURABLE_BEAM_SEAM_PINS:
        if not (beams & mask):
            continue
        if pref_room and room and int(room) != int(pref_room):
            continue
        _dump(stem, label)
    return written


def _room_label(room_id: int, names: dict[int, str]) -> str:
    rid = int(room_id)
    return f"{room_name(rid, names=names)} (0x{rid:04X})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from",
        dest="start",
        default="start",
        help=(
            "Start preset name, path/stem, or power-on alias "
            f"(power-on: {', '.join(sorted(POWER_ON_STARTS))}; "
            f"presets: {', '.join(START_PRESETS)}; default: start)"
        ),
    )
    parser.add_argument(
        "--route",
        default=None,
        choices=sorted(ROUTE_PRESETS),
        help=(
            "Guide route preset (waypoints drawn per room). "
            "Default: parlor-left when --from parlor/post-torizo, "
            "else cathedral-to-bat"
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Task name under super_metroid/tasks/ (default: guided_<route>_<ts>)",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable unlimited energy/ammo (harder practice)",
    )
    parser.add_argument(
        "--assist-full",
        action="store_true",
        help=(
            "Always top up energy/ammo (product continuous). "
            "Default practice assist only tops up at 0 and counts top_ups."
        ),
    )
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Record without drawing the route line",
    )
    parser.add_argument(
        "--no-autopilot",
        action="store_true",
        help="Do not load the human-hot-swappable reactive room autopilot",
    )
    parser.add_argument(
        "--autopilot-candidates",
        action="store_true",
        help="Allow unverified candidate room policies (development only)",
    )
    parser.add_argument(
        "--autopilot-policy-dir",
        type=Path,
        help="Override policies/reactive_rooms lookup directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List start presets and routes, then exit",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TASKS_DIR,
        help=f"Task JSON directory (default: {TASKS_DIR})",
    )
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="Disable live room/item gzip anchors (not recommended for long takes)",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help=(
            "Skip post-save materialize (settled hops + *_run_timing.json). "
            "Default ON so each take gets room-split timing structure."
        ),
    )
    parser.add_argument(
        "--bank",
        action="store_true",
        help="With materialize: merge hop records into recordings/skill_bank/bank.json",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help=(
            "If tasks/<name>.json already exists, overwrite without archiving "
            "to tasks/<name>_segments/sN/ (default: archive prior tape first)"
        ),
    )
    args = parser.parse_args()

    if args.list:
        print("Start presets:")
        print(
            f"  {'start':22s} [OK] power-on (no savestate) — intro/title → Ceres → Zebes"
        )
        print(f"    aliases: {', '.join(sorted(POWER_ON_STARTS))}")
        for key, (rel, desc) in START_PRESETS.items():
            path = INTEGRATION_DIR / rel
            mark = "OK" if path.is_file() else "MISSING"
            print(f"  {key:22s} [{mark}] {desc}")
            print(f"    {path}")
        print("\nRoute presets:")
        for key, guides in sorted(ROUTE_PRESETS.items()):
            rooms = " → ".join(g.name or f"0x{g.room_id:04X}" for g in guides)
            print(f"  {key:22s} {rooms}")
        return 0

    power_on = _is_power_on(args.start)

    # Sensible default route from start pin.
    if args.route is None:
        if power_on:
            # Full free-record from title; no guide polyline for the whole game.
            args.route = "cathedral-to-bat"
            args.no_guide = True
        elif args.start in (
            "morph",
            "post-morph",
            "morph-ball",
            "full-start-morph",
            "bomb",
            "bombs",
            "post-bomb",
            "post-bombs",
            "bomb-torizo",
            "full-start-bomb",
            "supers",
            "super",
            "super-missile",
            "spore-super",
            "post-supers",
            "full-start-supers",
            "resume",
            "full-start-pink",
        ):
            # Item-seam free-record; guide polylines off (minimum overlay).
            args.route = "cathedral-to-bat"
            args.no_guide = True
        elif args.start in (
            "varia",
            "post-varia",
            "varia-pickup",
            "varia-end",
            "bubble-human",
            "full-start-bubble",
        ):
            args.route = "cathedral-to-bat"
            args.no_guide = True
        elif args.start in ("parlor", "post-torizo"):
            args.route = "parlor-left"
        elif args.start in ("big-pink", "charge"):
            args.route = "charge-collect-return"
        elif args.start in (
            "charge-to-spazer",
            "below-spazer",
            "spazer",
            "early-spazer",
        ):
            args.route = "early-spazer"
        elif args.start in ("post-spazer-return", "spazer-return"):
            args.route = "spazer-top-drop"
        elif args.start in ("post-spazer", "post-spazer-collect"):
            args.route = "spazer-return-drop"
        elif args.start in ("speed", "post-speed", "speed-collected"):
            args.route = "speed-to-ice-moat"
        elif args.start in ("ice", "post-ice", "ice-collect"):
            # Free-record Ice return; guide polyline is Speed-forward — off by default.
            args.route = "speed-to-ice-moat"
            args.no_guide = True
        elif args.start in ("red-bottom", "red-tower", "post-ice-red"):
            # Free-record Red climb; no dedicated guide polyline.
            args.route = "ws-entrance"
            args.no_guide = True
        elif args.start in (
            "double-chamber",
            "dc",
            "dc-cont",
            "dc-pure",
            "dc-post-missiles",
        ):
            args.route = "double-chamber-to-wave"
        elif args.start in (
            "pre-moat",
            "kihunter-pre-moat",
            "pre-moat-spark",
            "moat-end",
            "alpha-pb-moat-end",
        ):
            # Free-record Moat→WO→ship (prefer bot chain-ws then --from ws-entrance).
            args.route = "west-ocean-to-ws"
            args.no_guide = True
        elif args.start in (
            "west-ocean",
            "post-moat",
            "post-moat-spark",
            "moat-spark",
        ):
            args.route = "west-ocean-to-ws"
        elif args.start in (
            "ws-entrance",
            "post-west-ocean",
            "post-wo-ws",
            "post-ws-spark",
            "wrecked-ship",
        ):
            args.route = "ws-entrance"
        elif args.start in (
            "post-phantoon",
            "phantoon-defeated",
            "post-phant",
        ):
            # Gravity free-record; no dedicated guide polyline yet.
            args.route = "ws-entrance"
            args.no_guide = True
        elif args.start in (
            "post-gravity",
            "post-gravity-caterpillar",
            "gravity",
            "gravity-caterpillar",
            "maridia-start",
            "post-grapple",
            "post-grapple-beam",
            "grapple",
            "main-street",
            "post-main-street",
            "post-grapple-main-street",
            "full-start-main-street",
            "maridia",
            "plasma-beam",
            "plasma",
            "post-plasma",
            "plasma-room",
            "full-start-plasma",
            "golden-torizo",
            "gt",
            "gt-entry",
            "golden-torizo-entry",
            "full-start-gt",
            "metal-pirates",
            "metal-pirate",
            "pirates",
            "mp",
            "full-start-metal-pirates",
            "post-space-jump",
            "space-jump",
            "post-sj",
            "post-draygon",
            "precious",
            "post-draygon-precious",
            "post-space-jump-precious",
            "post-spring-ball",
            "spring-ball",
            "main-hall",
            "ln-main-hall",
            "post-ln-main-hall",
            "lower-norfair",
            "ln-elev-save",
            "post-screw",
            "screw-attack",
            "post-screw-attack",
            "post-ridley",
            "ridley-tank",
            "post-ridley-tank",
            "full-start-ridley",
            "full-start-post-ridley",
            "post-ridley-farming",
            "post-bosses",
            "landing-site-post-bosses",
            "post-bosses-landing",
            "post-ridley-landing",
        ):
            # Grapple / Maridia / LN / post-boss free-record; guide off.
            args.route = "ws-entrance"
            args.no_guide = True
        else:
            args.route = "cathedral-to-bat"

    state_path: Path | None = None
    state_bytes: bytes | None = None
    if not power_on:
        try:
            state_path = _resolve_state(args.start)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not state_path.is_file():
            print(f"ERROR: state missing: {state_path}", file=sys.stderr)
            return 1
        state_bytes = read_state_bytes(state_path)

    task_name = args.name or f"guided_{args.route}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / f"{task_name}.json"
    # Immutable segment archive: reusing --name no longer destroys prior buttons.
    if task_path.is_file() and not args.no_archive:
        try:
            from super_metroid.human_tape.segment_archive import archive_existing_take

            archived = archive_existing_take(task_path)
            if archived is not None:
                print(
                    f"[REC] archived previous take → {archived} "
                    f"(tape.json + join.json; anchors stay under {task_name}_anchors/)",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 — never block record
            print(f"[REC] archive previous take failed (continuing): {exc}", flush=True)

    # Full-run RTA offset (Ceres first control → now). After archive so prior
    # segment is included when continuing full_start_v1 item seams.
    rta_clock = resolve_rta_clock(task_path, include_live_tape=False)
    if power_on:
        # During power-on the live clock starts at 0; Ceres zero is applied once
        # we see the Ceres Elevator boot/enter (see _print_anchor). Until then
        # show local frames only (title/menu not in any% RTA).
        rta_offset = 0
    else:
        rta_offset = int(rta_clock.offset_frames)
    rta_ceres_zero_live: list[int | None] = [None]  # mutable for power-on
    end_state_paths = [
        out_dir / f"{task_name}_end.state",
        SCRATCH / f"{task_name}_end.state",
    ]
    anchors_dir = out_dir / f"{task_name}_anchors"
    anchors_index_path = out_dir / f"{task_name}_anchors.json"
    anchor_rec = AnchorRecorder(
        task_name=task_name,
        anchors_dir=anchors_dir,
        enabled=not args.no_anchors,
    )

    route_guides = ROUTE_PRESETS[args.route]
    route_room_ids = {g.room_id for g in route_guides}

    def _guides_for_active_room(room_id: int) -> list:
        """All route polylines for this room (main + recovery fallbacks)."""
        rid = int(room_id)
        matched = [g for g in route_guides if g.room_id == rid]
        if matched:
            return matched
        g = guide_for_room(rid)
        if g is not None and g.room_id in route_room_ids:
            return [g]
        return []

    def _guide_for_active_room(room_id: int):
        """Primary guide (first) for HUD nearest-waypoint."""
        gs = _guides_for_active_room(room_id)
        return gs[0] if gs else None

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")

    # Practice default: only top up energy/ammo at 0 (skill metric = top_ups).
    # Product continuous still uses refill_when="always" elsewhere.
    assist_refill = "always" if args.assist_full else "at_zero"
    assist = UnlimitedResourcesAssist(
        unlimited_energy=not args.no_assist,
        unlimited_ammo=not args.no_assist,
        refill_when=assist_refill,  # type: ignore[arg-type]
    )
    live_topups = {"energy": 0, "ammo": 0, "last_print": 0}
    if power_on:
        start_label = "power_on"
    elif state_path is not None and state_path.is_relative_to(INTEGRATION_DIR):
        start_label = str(state_path.relative_to(INTEGRATION_DIR))
    else:
        start_label = str(state_path) if state_path is not None else "power_on"
    task = RecordedTask(
        name=task_name,
        start_state=start_label,
    )
    task.metadata["route"] = args.route
    task.metadata["guide_rooms"] = [
        {"room_id": f"0x{g.room_id:04X}", "name": g.name, "points": len(g.points)}
        for g in route_guides
    ]
    task.metadata["source_path"] = start_label
    task.metadata["power_on"] = power_on
    task.metadata["rta_clock"] = {
        **rta_clock.to_dict(),
        "session_offset_frames": int(rta_offset),
        "zero": "ceres_first_control",
    }

    room_names = load_canonical_names()
    live: dict[str, object] = {
        "room": 0,
        "x": 0,
        "y": 0,
        "guide_name": "",
        "nearest": None,
        "cam_x": 0,
        "cam_y": 0,
        "items": 0,
        "anchors": 0,
        "last_anchor": "",
        "last_enter_frame": None,
        "last_enter_name": "",
        "split_delta": "",
    }
    saved = {"ok": False}
    autopilot_box: dict[str, Any] = {}

    def _rta_frames(local_fr: int) -> int:
        """Full-run frames from Ceres first control (any% KPDR clock)."""
        if power_on:
            # Title/menu hold at 0:00 until first Ceres Elevator control.
            if rta_ceres_zero_live[0] is None:
                return 0
            return max(0, int(local_fr) - int(rta_ceres_zero_live[0]))
        return max(0, int(rta_offset) + max(0, int(local_fr)))

    def _print_anchor(a: dict) -> None:
        kind = str(a.get("kind") or "?")
        fr = int(a.get("frame") or 0)
        rid = parse_room_id(a.get("room")) or 0
        label = _room_label(rid, room_names)
        items = a.get("items", "?")
        xy = a.get("xy", "?")
        # Power-on: latch Ceres Elevator as RTA t=0 (first control).
        if (
            power_on
            and rta_ceres_zero_live[0] is None
            and rid == 0xDF45
            and kind in ("boot", "room_enter", "enter")
        ):
            rta_ceres_zero_live[0] = fr
            print(
                f"[RTA] Ceres control zero @ local f{fr} "
                f"(any% clock starts here)",
                flush=True,
            )
        t = _fmt_time(_rta_frames(fr))
        if kind == "room_enter":
            prev_f = live.get("last_enter_frame")
            if isinstance(prev_f, int):
                split = _fmt_time(fr - prev_f)
                prev_n = live.get("last_enter_name") or "?"
                print(
                    f"[ROOM] {label}  t={t} (f{fr})  split=+{split} from {prev_n}  "
                    f"xy={xy}  items={items}",
                    flush=True,
                )
            else:
                print(
                    f"[ROOM] {label}  t={t} (f{fr})  xy={xy}  items={items}",
                    flush=True,
                )
            live["last_enter_frame"] = fr
            live["last_enter_name"] = room_name(rid, names=room_names)
            live["split_delta"] = ""
        elif kind == "item_delta":
            print(
                f"[ITEM] {label}  t={t} (f{fr})  items={items}  xy={xy}",
                flush=True,
            )
        else:
            print(
                f"[ANCHOR] {kind}  {label}  t={t} (f{fr})  xy={xy}  items={items}",
                flush=True,
            )

    def on_step(obs, reward, done, info) -> None:
        del obs, reward, done, info
        action = session.last_action_post_sanitize
        frame = session.frame_count
        # Assist after human step (same pattern as pure probe sessions).
        st = parse_env_state(env, frame=frame, mode="nav")
        assist.apply(env.data, st)
        # Live top-up tally (at_zero practice handicap skill metric).
        if not args.no_assist:
            e_tu = int(assist.telemetry.energy.top_ups)
            a_tu = sum(int(c.top_ups) for c in assist.telemetry.ammo.values())
            total_tu = e_tu + a_tu
            if total_tu > int(live_topups["last_print"]):
                kinds = []
                if e_tu > int(live_topups["energy"]):
                    kinds.append("energy")
                if a_tu > int(live_topups["ammo"]):
                    kinds.append("ammo")
                print(
                    f"[TOPUP] {'+'.join(kinds) or 'resource'}  "
                    f"total={total_tu}  energy={e_tu}  ammo={a_tu}  "
                    f"t={_fmt_time(_rta_frames(frame - 1 if frame > 0 else 0))}",
                    flush=True,
                )
                live_topups["energy"] = e_tu
                live_topups["ammo"] = a_tu
                live_topups["last_print"] = total_tu
        row_frame = frame - 1 if frame > 0 else 0
        row = _trace_row(env, row_frame, action)
        task.append_frame(action, trace_row=row)
        live["room"] = row["room"]
        live["x"] = row["x"]
        live["y"] = row["y"]
        live["items"] = row.get("items", 0)
        ram = env.get_ram()
        live["cam_x"] = _u16(ram, ADDR_CAMERA_X)
        live["cam_y"] = _u16(ram, ADDR_CAMERA_Y)
        guide = _guide_for_active_room(int(row["room"]))
        if guide is not None:
            live["guide_name"] = guide.name
            live["nearest"] = nearest_waypoint_index(guide.points, int(row["x"]), int(row["y"]))
        else:
            live["guide_name"] = ""
            live["nearest"] = None
        # Live anchors: room enter + item-bit change (never rely on full-tape replay).
        new_anchors = anchor_rec.on_frame(env=env, st=st, frame=row_frame)
        if new_anchors:
            live["anchors"] = len(anchor_rec.anchors)
            live["last_anchor"] = new_anchors[-1].get("kind", "")
            for a in new_anchors:
                _print_anchor(a)

    def on_hud(info) -> list[str]:
        del info
        room = int(live["room"] or 0)
        items = int(live.get("items") or 0)
        n = len(task.frames)
        # Minimum overlay: one line — RTA from Ceres + room + items + top-ups.
        # (PlaySession.hud_minimal also drops the yellow F#/FPS banner.)
        line = (
            f"t={_fmt_time(_rta_frames(n))}  "
            f"{_room_label(room, room_names)}  "
            f"({live['x']},{live['y']})  items=0x{items:04X}"
        )
        if not args.no_assist:
            e_tu = int(assist.telemetry.energy.top_ups)
            a_tu = sum(int(c.top_ups) for c in assist.telemetry.ammo.values())
            line += f"  topups={e_tu + a_tu}(e{e_tu}/a{a_tu})"
        lines = [line]
        autopilot = autopilot_box.get("bot")
        if session.bot_active and autopilot is not None:
            lines.append(autopilot.mission_status().summary())
        return lines

    def on_overlay(pg, ctx) -> None:
        if args.no_guide:
            return
        room = int(live["room"] or 0)
        guides = _guides_for_active_room(room)
        if not guides:
            return
        transform = transform_from_session_ctx(
            ctx,
            camera_x=int(live["cam_x"] or 0),
            camera_y=int(live["cam_y"] or 0),
        )
        surface = ctx.get("screen")
        font = ctx.get("font")
        if surface is None:
            return
        # Draw recovery (later entries) first so main path sits on top.
        for i, guide in enumerate(reversed(guides)):
            is_primary = i == len(guides) - 1
            nearest = (
                nearest_waypoint_index(guide.points, int(live["x"] or 0), int(live["y"] or 0))
                if is_primary
                else None
            )
            draw_guide_path(
                pg,
                surface,
                guide.points,
                transform,
                color=guide.color,
                width=3 if is_primary else 2,
                radius=6 if is_primary else 4,
                highlight_index=nearest if isinstance(nearest, int) else 0,
                font=font,
                draw_labels=True,
            )
        draw_player_marker(
            pg,
            surface,
            int(live["x"] or 0),
            int(live["y"] or 0),
            transform,
        )

    def on_key_down(key: int) -> bool:
        # F5/F1: finalize recording (PlaySession F5 normally only quicksaves).
        # F6: manual mid-run pin (next-phase lock without ending the take).
        import pygame

        if key in (pygame.K_F5, pygame.K_F1):
            _finalize(save=True)
            session.running = False
            return True
        if key == pygame.K_F6:
            st = parse_env_state(env, mode="nav")
            frame = max(0, len(task.frames) - 1)
            fp = anchor_rec.manual_pin(env=env, st=st, frame=frame)
            if fp is not None:
                live["anchors"] = len(anchor_rec.anchors)
                live["last_anchor"] = "manual"
                rid = parse_room_id(fp.get("room")) or int(st.room_id)
                print(
                    f"[PIN] {_room_label(rid, room_names)}  t={_fmt_time(frame)} (f{frame})  "
                    f"xy={fp['xy']} items={fp.get('items', '?')} → {fp.get('path')}",
                    flush=True,
                )
            else:
                print("[PIN] skipped (--no-anchors)", flush=True)
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            n = len(task.frames)
            if n > 0:
                print(
                    f"[REC] cancelled — dropping {n} frames of button tape. "
                    f"F5 to save a stitchable segment; ESC only if this take is trash.",
                    flush=True,
                )
            else:
                print("[REC] cancelled", flush=True)
            session.running = False
            return True
        return False

    def on_trigger_save(slot: int) -> None:
        frame = session.save_checkpoint(slot)
        st = parse_env_state(env, mode="nav")
        print(
            f"[CP SAVE {slot}] {_room_label(int(st.room_id), room_names)}  "
            f"t={_fmt_time(frame)} (f{frame})  xy=({int(st.samus_x)},{int(st.samus_y)})  "
            f"items=0x{int(st.collected_items):04X}",
            flush=True,
        )

    def on_trigger_load(slot: int) -> None:
        frame = session.load_checkpoint(slot)
        if frame is None:
            print(f"[CP LOAD {slot}] empty", flush=True)
            return
        # Keep recording aligned with emulator after rewind.
        if len(task.frames) > frame:
            del task.frames[frame:]
        if len(task.trace) > frame:
            del task.trace[frame:]
        st = parse_env_state(env, mode="nav")
        live["room"] = int(st.room_id)
        live["x"] = int(st.samus_x)
        live["y"] = int(st.samus_y)
        live["items"] = int(st.collected_items)
        print(
            f"[CP LOAD {slot}] {_room_label(int(st.room_id), room_names)}  "
            f"t={_fmt_time(frame)} (f{frame})  xy=({int(st.samus_x)},{int(st.samus_y)})  "
            f"tape truncated to {len(task.frames)}f",
            flush=True,
        )

    def _finalize(*, save: bool) -> None:
        if not save or saved["ok"]:
            return
        if not task.frames:
            print("[REC] nothing recorded")
            return
        try:
            task.end_state_data = env.em.get_state()
        except Exception as exc:
            print(f"[REC] end-state capture failed: {exc}")
        # End fingerprint — detect later overwrites / desynced "recovered" pins.
        try:
            end_st = parse_env_state(env, mode="full")
            end_fp = fingerprint(
                frame=max(0, len(task.frames) - 1),
                room_id=int(end_st.room_id),
                x=int(end_st.samus_x),
                y=int(end_st.samus_y),
                pose=int(end_st.pose),
                items=int(end_st.collected_items),
                beams=int(end_st.collected_beams),
                energy=int(end_st.health),
                kind="end",
            )
            task.metadata["end_fingerprint"] = end_fp
            # Also dump end as an anchor for hop extract.
            if anchor_rec.enabled:
                anchor_rec.dump(
                    env=env,
                    st=end_st,
                    frame=max(0, len(task.frames) - 1),
                    kind="end",
                    label="end",
                )
        except Exception as exc:
            print(f"[REC] end fingerprint failed: {exc}")
        task.metadata.update(
            summarize_position_trace(frames=task.frames, trace=task.trace, room_key="room")
        )
        task.metadata["assist"] = {
            "unlimited_energy": not args.no_assist,
            "unlimited_ammo": not args.no_assist,
            "refill_when": assist_refill if not args.no_assist else "off",
            "telemetry": assist.telemetry.to_dict() if hasattr(assist, "telemetry") else {},
        }
        task.metadata["anchors"] = {
            "enabled": anchor_rec.enabled,
            "count": len(anchor_rec.anchors),
            "dir": str(anchors_dir) if anchor_rec.enabled else None,
            "index": str(anchors_index_path) if anchor_rec.enabled else None,
        }
        task.recorded_at = datetime.now().isoformat()
        task.save(task_path, end_state_paths=end_state_paths)
        if anchor_rec.enabled:
            anchor_rec.write_index(
                anchors_index_path,
                extra={
                    "task_json": str(task_path),
                    "end_fingerprint": task.metadata.get("end_fingerprint"),
                    "frame_count": len(task.frames),
                },
            )
        # Mirror a lightweight pointer under recordings/ for discoverability.
        rec_ptr = RECORDINGS_DIR / "human_tasks" / f"{task_name}.json"
        try:
            if task_path.resolve() != rec_ptr.resolve():
                rec_ptr.parent.mkdir(parents=True, exist_ok=True)
                if not rec_ptr.exists():
                    rec_ptr.symlink_to(task_path.resolve())
        except OSError:
            pass
        saved["ok"] = True
        print(f"[REC] saved {task_path} ({len(task.frames)} frames)")
        if not args.no_assist:
            tel = assist.telemetry
            e_tu = int(tel.energy.top_ups)
            a_tu = sum(int(c.top_ups) for c in tel.ammo.values())
            print(
                f"[REC] assist refill={assist_refill}  "
                f"top_ups={e_tu + a_tu} (energy={e_tu} ammo={a_tu})  "
                f"energy_restored={int(tel.energy.restored)}  "
                f"max_hit={int(tel.maximum_single_frame_damage)}",
                flush=True,
            )
        for p in end_state_paths:
            print(f"[REC] end state → {p}")
        end_fp = task.metadata.get("end_fingerprint")
        if end_fp:
            print(
                f"[REC] end pin {end_fp.get('room')} xy={end_fp.get('xy')} "
                f"items={end_fp.get('items')} grapple={end_fp.get('grapple')}"
            )
            local_end = int(end_fp.get("frame") or max(0, len(task.frames) - 1))
            print(
                f"[REC] RTA t={_fmt_time(_rta_frames(local_end))} "
                f"(Ceres-zero any%; local f{local_end})",
                flush=True,
            )
        # Durable item-seam pins (./play morph / ./play bomb / ./play varia).
        try:
            pin_notes = _write_durable_seam_pins(
                env,
                end_fp=end_fp if isinstance(end_fp, Mapping) else None,
                state_bytes=getattr(task, "end_state_data", None),
            )
            for note in pin_notes:
                print(f"[REC] durable pin {note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[REC] durable pin write failed: {exc}", flush=True)
        if anchor_rec.enabled:
            print(
                f"[REC] anchors → {anchors_index_path} "
                f"({len(anchor_rec.anchors)} dumps under {anchors_dir.name}/)"
            )
        # Thin post-process: one library call (do not grow timing logic here).
        if not args.no_materialize:
            try:
                from super_metroid.materialize import materialize_take

                mat = materialize_take(
                    task_path,
                    write=True,
                    write_extract=True,
                    write_run_timing=True,
                    merge_bank=bool(args.bank),
                    stitch=True,
                    stitch_print_table=False,
                )
                if mat.run_timing_path:
                    print(f"[REC] run_timing → {mat.run_timing_path}")
                if mat.extract_path:
                    print(f"[REC] extract → {mat.extract_path}")
                if mat.bank_path:
                    print(f"[REC] skill_bank → {mat.bank_path}")
                summ = (mat.run_timing or {}).get("summary") or {}
                print(
                    f"[REC] materialize rooms={summ.get('room_visits')} "
                    f"items={summ.get('item_splits')} bosses={summ.get('boss_splits')} "
                    f"bank={len(mat.bank_records)}"
                )
            except Exception as exc:
                print(f"[REC] materialize failed (take still saved): {exc}", flush=True)

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME,
        scale=args.scale,
        title=f"Guided REC: {task_name} [{args.route}]",
        bot=None,
        action_size=12,
        base_fps=60,
    )
    session.quiet_checkpoints = True
    # Soft-white one-line HUD (no yellow FPS banner / guide polylines by default).
    session.hud_minimal = True
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_overlay = on_overlay
    session.on_key_down = on_key_down
    session.on_trigger_save = on_trigger_save
    session.on_trigger_load = on_trigger_load
    session.on_close = lambda: None
    if not args.no_autopilot:
        from super_metroid.autopilot import RoomAutopilot

        autopilot = RoomAutopilot(
            env,
            policy_dir=args.autopilot_policy_dir
            if args.autopilot_policy_dir is not None
            else GAME_DIR / "policies" / "reactive_rooms",
            allow_candidates=args.autopilot_candidates,
        )
        autopilot_box["bot"] = autopilot
        # set_bot intentionally leaves this human-controlled until the toggle.
        session.set_bot(autopilot)
    # Closed over by _reset_then_boot for SELECT+L2 pin seed after env.reset.
    session_box: dict[str, Any] = {"s": session}

    def _seed_live_from_env() -> None:
        boot = parse_env_state(env, mode="nav")
        live["room"] = int(boot.room_id)
        live["x"] = int(boot.samus_x)
        live["y"] = int(boot.samus_y)
        ram0 = env.get_ram()
        live["cam_x"] = _u16(ram0, ADDR_CAMERA_X)
        live["cam_y"] = _u16(ram0, ADDR_CAMERA_Y)
        g0 = _guide_for_active_room(int(boot.room_id))
        if g0 is not None:
            live["guide_name"] = g0.name
            live["nearest"] = nearest_waypoint_index(
                g0.points, int(boot.samus_x), int(boot.samus_y)
            )
        else:
            live["guide_name"] = ""
            live["nearest"] = None

    # PlaySession.run() always env.reset() first — re-inject savestate after
    # that reset (unless power-on: leave pure cold boot / title).
    import retro_harness.play_session as _ps_mod

    _orig_reset = _ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        if state_bytes is not None:
            e.em.set_state(state_bytes)
            # Settle door transition / pose fully (Below Spazer needs ~12f for pose 1).
            settle = (
                16
                if args.start
                in (
                    "below-spazer",
                    "spazer",
                    "early-spazer",
                    "post-spazer-return",
                    "spazer-return",
                )
                else 8
            )
            for _ in range(settle):
                obs, _r, _t, _tr, info = step_env(e, idle_action())
        _seed_live_from_env()
        if power_on:
            boot_bits = (
                f"[BOOT] power-on (title/intro) room=0x{int(live['room']):04X} "
                f"xy=({live['x']},{live['y']}) — mash through menus into Ceres"
            )
        else:
            assert state_path is not None
            boot_bits = (
                f"[BOOT] room=0x{int(live['room']):04X} "
                f"xy=({live['x']},{live['y']}) from {state_path.name}"
            )
        if args.start in ("below-spazer", "spazer", "early-spazer"):
            boot_bits += " | Below Spazer pin — keep x>=40 (left door=Bat trap)"
        elif args.start in ("post-spazer-return", "spazer-return"):
            boot_bits += (
                " | top handoff Spazer held — NO RIGHT (Super re-entry); "
                "clean drop to mid y>=220 only"
            )
        elif args.start in ("post-spazer", "post-spazer-collect"):
            boot_bits += (
                " | Spazer Room post-collect — return left, then clean top→mid drop"
            )
        elif args.start in ("charge-to-spazer", "big-pink", "charge"):
            boot_bits += (
                " | start Charge area — collect Charge, play Red→Bat→Below→Spazer"
            )
        elif args.start in (
            "double-chamber",
            "dc",
            "dc-cont",
            "dc-pure",
            "dc-post-missiles",
        ):
            boot_bits += (
                " | DC missile ledge: gate → pack ~x492 free RIGHT past 510 → "
                "runway ~x425 → dash edge 600 → Super → Wave; F5 when Wave held"
            )
        elif args.start in (
            "bubble-save",
            "bubble-save-room",
            "full-start-bubble-save",
        ):
            boot_bits += (
                " | Bubble Save 0xB0DD — leave RIGHT → runway WJ climb; "
                "SELECT+L2 reloads pin (CP1 seeded); live grades: "
                "bubble_save_practice.py"
            )
        print(boot_bits)
        # Seed checkpoint slot 1 with the boot pin so SELECT+L2 works immediately
        # (no need to SELECT+R2 first for practice reloads).
        if state_bytes is not None and session_box.get("s") is not None:
            try:
                session_box["s"].save_checkpoint(1)
                print(
                    "[CP1] boot pin seeded — SELECT+L2 reloads start",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[CP1] seed skipped: {exc}", flush=True)
        return obs, info

    print("=" * 60)
    print(f"GUIDED HUMAN RECORD  route={args.route}")
    print(f"  start: {start_label}")
    print(f"  task:  {task_path}")
    if args.no_assist:
        assist_label = "OFF"
    elif assist_refill == "at_zero":
        assist_label = "ON@0 (ammo@0 · energy≤40)"
    else:
        assist_label = "ON full"
    print(
        f"  guide: {'ON' if not args.no_guide else 'OFF'}  "
        f"assist={assist_label}  "
        f"anchors={'OFF' if args.no_anchors else 'ON'}"
    )
    print(
        "  checkpoints: SELECT+R2 save · SELECT+L2 load "
        "(bare L2/R2 ignored; F2-F4 / Shift+F2-F4 still work; F1=F5=save take)"
    )
    if not args.no_autopilot:
        print("  autopilot: ` or controller L+R+Select toggles at any live frame")
    if not args.no_anchors:
        print(f"  anchors dir: {anchors_dir}")
    if power_on:
        print("  RTA: any% from first Ceres control (title/menu excluded)")
    else:
        print(
            f"  RTA: t0 offset={_fmt_time(rta_offset)} (f{rta_offset}) "
            f"from Ceres · starts at {_fmt_time(rta_offset)}"
        )
        for note in rta_clock.notes[:6]:
            print(f"    · {note}")
    print("  HUD: minimal (one line) · F5/F1=save · F6=pin · ESC/Q=cancel")
    print("=" * 60)

    _ps_mod.reset_env = _reset_then_boot
    try:
        session.run()
    finally:
        _ps_mod.reset_env = _orig_reset
        # session.run already closes env; ignore double-close

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
