#!/usr/bin/env python3
"""Multi-take human practice loop for a single segment (record → reload → repeat).

Default segment is **Double Chamber missile ledge → Super → Wave** under Spazer
mainline continuous-like leave (beams ``0x1004``) — the free+runway geometry
used by pure dual cont-like Wave.

Each take saves a unique task JSON under ``super_metroid/tasks/<series>/`` so
you can keep several clean recordings, review traces, and compare to the bot.

```bash
# Spazer DC missile room — record takes until you quit (ESC on a take)
uv run python snes/super_metroid/scripts/record/practice_takes.py

# Named series + max takes
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment dc-missile-wave --series dc_missile_v1 --max 8

# Pure (beams=0) leave for comparison
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment dc-missile-wave-pure --series dc_pure_v1

# Early Spazer climb (Below Spazer) — guide off
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment early-spazer --series spazer_wj_v1 --no-guide

# List saved takes for a series
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --series dc_missile_v1 --list

# Bot pure check from the same start state (no human window)
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment dc-missile-wave --bot-check

# Single take only (same as guided_human with auto name)
uv run python snes/super_metroid/scripts/record/practice_takes.py --max 1

# K6 West Ocean post-Moat → WS (optional human; product pure-ws is GREEN)
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment west-ocean-to-ws --series west_ocean_ws_v1

# K6 WS Entrance after pure over-ocean / chain-ws → ship free-record → Phantoon
# Refresh pin: west_ocean_spark.py chain-ws  (or pure-ws)
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment ws-entrance --series ws_ship_v1

# Post-Gravity Caterpillar → Grapple side-trek + Maridia free-record
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment post-gravity --series maridia_grapple_v1 --no-guide

# K5 Red Tower → Hellway multi-take (rr-av5s). F5 in Hellway / ESC end series.
# Rank takes: scripts/tools/rank_red_climb_takes.py
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment red-to-hellway --series red_climb_v1
# Live human Red enter (warehouse_to_red_human room_enter f2012):
uv run python snes/super_metroid/scripts/record/practice_takes.py \\
  --segment red-to-hellway-human --series red_climb_human_v1
```

Controls (inside each take — same as guided_human):
  **F5 / F1**  Save this take + end state, advance to next take
  **ESC / Q**  Cancel current take (no save) and **end series**
  ``[`` ``]`` / TAB  Speed / turbo

Human practice assist (unlimited energy/ammo) is **on** by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get("_SNES_IMPORT_ROOT", ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import GAME_DIR, INTEGRATION_DIR  # noqa: E402

TASKS_DIR = GAME_DIR / "tasks"
GUIDED = Path(__file__).resolve().parent / "guided_human.py"
KPDR_PROBE = Path(__file__).resolve().parents[1] / "probe" / "kpdr.py"

# Wave beam bit on collected_beams.
_WAVE_MASK = 0x0001
_ROOM_WAVE = 0xADDE
_ROOM_DOUBLE = 0xADAD


@dataclass(frozen=True)
class Segment:
    """One practice pin + guided route."""

    key: str
    start: str  # guided_human --from preset
    route: str
    description: str
    pure_hop: str | None = None  # kpdr pure hop name for --bot-check
    pure_source_rel: str | None = None  # scratch-relative source for bot
    no_guide_default: bool = False


SEGMENTS: dict[str, Segment] = {
    "dc-missile-wave": Segment(
        key="dc-missile-wave",
        start="double-chamber",
        route="double-chamber-to-wave",
        description=(
            "Double Chamber leave (Spazer cont-like 0x1004) → gate → missile free "
            "→ runway → Super → Wave. Bot-parity source."
        ),
        pure_hop="double-chamber-to-wave",
        pure_source_rel=(
            "scratch/post_single_to_double_chamber_continuous_like.state"
        ),
        # Guide ON by default: take04 main (purple) + floor recover (red).
        no_guide_default=False,
    ),
    "dc-missile-wave-pure": Segment(
        key="dc-missile-wave-pure",
        start="dc-pure",
        route="double-chamber-to-wave",
        description=(
            "Double Chamber pure predecessor leave (often beams=0) → Wave. "
            "Compare vs Spazer cont-like."
        ),
        pure_hop="double-chamber-to-wave",
        pure_source_rel="scratch/post_single_to_double_chamber_pure.state",
        no_guide_default=False,
    ),
    "dc-post-missiles": Segment(
        key="dc-post-missiles",
        start="dc-post-missiles",
        route="double-chamber-to-wave",
        description=(
            "Past-gate post-missile pin only — runway/Super/Wave without gate thrash."
        ),
        pure_hop=None,
        pure_source_rel="scratch/dev_dc_post_missiles.state",
        no_guide_default=False,
    ),
    "early-spazer": Segment(
        key="early-spazer",
        start="charge-to-spazer",
        route="early-spazer",
        description=(
            "Charge Big Pink → play natural into Below Spazer WJ climb + Spazer."
        ),
        pure_hop=None,
        pure_source_rel=None,
        no_guide_default=True,
    ),
    "speed-to-wave": Segment(
        key="speed-to-wave",
        start="speed",
        route="speed-to-wave",
        description="Post-Speed full Wave branch (longer free-record).",
        pure_hop=None,
        pure_source_rel="scratch/post_speed_collected.state",
        no_guide_default=False,
    ),
    "west-ocean-to-ws": Segment(
        key="west-ocean-to-ws",
        start="west-ocean",
        route="west-ocean-to-ws",
        description=(
            "West Ocean post-Moat handoff (0x93FE ~(49,1163)) → green Super → "
            "WS 0xCA08. Optional human practice; product pure-ws is dual GREEN."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_moat_west_ocean_spark.state",
        no_guide_default=False,
    ),
    "ws-entrance": Segment(
        key="ws-entrance",
        start="ws-entrance",
        route="ws-entrance",
        description=(
            "Wrecked Ship Entrance after pure over-ocean / chain-ws "
            "(0xCA08 ~(57,139) gs=8) → ship rooms free-record for Phantoon approach. "
            "Refresh pin: west_ocean_spark.py chain-ws (Kihunter/Moat→WO→WS)."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_west_ocean_ws_spark.state",
        no_guide_default=False,
    ),
    "moat-to-ws": Segment(
        key="moat-to-ws",
        start="pre-moat",
        route="west-ocean-to-ws",
        description=(
            "Kihunter pre-spark (or use bot chain-ws) → Moat spark → WO → WS. "
            "Human free-record full path; prefer bot chain-ws then ws-entrance segment."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_kihunter_pre_moat_spark.state",
        no_guide_default=True,
    ),
    "post-gravity": Segment(
        key="post-gravity",
        start="post-gravity",
        route="ws-entrance",  # placeholder; guide off by default
        description=(
            "Caterpillar 0xA322 post-Gravity return (items 0x3125) → Grapple "
            "side-trek (Red/Crocomire) and/or Maridia free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_gravity_caterpillar.state",
        no_guide_default=True,
    ),
    "post-grapple": Segment(
        key="post-grapple",
        start="post-grapple",
        route="ws-entrance",
        description=(
            "Post-Grapple Beam (items 0x7125) → Croc Escape → Business → "
            "Main Street free-record. Anchors ON; F6 for next-phase pins."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_grapple_beam_human.state",
        no_guide_default=True,
    ),
    "main-street": Segment(
        key="main-street",
        start="main-street",
        route="ws-entrance",
        description=(
            "Main Street 0xCFC9 ~(394,1979) items 0x7125 → Maridia "
            "(Mt Everest / Botwoon path) free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/full_start_v1_main_street.state",
        no_guide_default=True,
    ),
    "golden-torizo": Segment(
        key="golden-torizo",
        start="golden-torizo",
        route="ws-entrance",
        description=(
            "Golden Torizo 0xB283 left door (items 0x7325 beams 0x100F) → "
            "fight / Screw / Ridley free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/full_start_v1_golden_torizo.state",
        no_guide_default=True,
    ),
    "metal-pirates": Segment(
        key="metal-pirates",
        start="metal-pirates",
        route="ws-entrance",
        description=(
            "Metal Pirates 0xB62B right door (items 0x732F Screw) → "
            "clear / Ridley free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/full_start_v1_metal_pirates.state",
        no_guide_default=True,
    ),
    "post-ridley": Segment(
        key="post-ridley",
        start="post-ridley",
        route="ws-entrance",
        description=(
            "Ridley Tank 0xB698 post fight + tank (items 0x732F) → "
            "LN exit / G4 free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/full_start_v1_ridley.state",
        no_guide_default=True,
    ),
    "plasma-beam": Segment(
        key="plasma-beam",
        start="plasma-beam",
        route="ws-entrance",
        description=(
            "Plasma Room 0xD2AA post-collect (items 0x7325 beams 0x100F) → "
            "leave / LN free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/full_start_v1_plasma.state",
        no_guide_default=True,
    ),
    "post-space-jump": Segment(
        key="post-space-jump",
        start="post-space-jump",
        route="ws-entrance",
        description=(
            "Space Jump Room 0xD9AA after collect (items 0x7325) → next "
            "segment free-record (Maridia exit / LN / Plasma path)."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_space_jump.state",
        no_guide_default=True,
    ),
    "post-draygon": Segment(
        key="post-draygon",
        start="post-draygon",
        route="ws-entrance",
        description=(
            "Precious Room 0xD78F post-Draygon+SJ (items 0x7325) F5 end of "
            "maridia_botwoon_path_human."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_draygon_precious.state",
        no_guide_default=True,
    ),
    "main-hall": Segment(
        key="main-hall",
        start="main-hall",
        route="ws-entrance",
        description=(
            "LN Main Hall 0xB236 ~(1152,648) items 0x7327 beams 0x100F → "
            "Ridley / Golden Torizo free-record."
        ),
        pure_hop=None,
        pure_source_rel="scratch/post_ln_main_hall.state",
        no_guide_default=True,
    ),
    # K5 Red Tower climb — multi-take from pure Bat→Red dual pin (rr-av5s).
    "red-to-hellway": Segment(
        key="red-to-hellway",
        start="red-bottom",
        route="ws-entrance",  # placeholder; guide off
        description=(
            "Red Tower bottom 0xA253 pure pin ~(216,2443) → clean climb → "
            "Hellway 0xA2F7. F6 mid seats; F5 in Hellway. Arbitrary takes for "
            "splice board (not open-loop frame concat)."
        ),
        pure_hop="red-to-hellway",
        pure_source_rel="scratch/post_ice_bat_to_red_pure.state",
        no_guide_default=True,
    ),
    # Same climb from live human room_enter (warehouse_to_red chain parity).
    "red-to-hellway-human": Segment(
        key="red-to-hellway-human",
        start=(
            "snes/super_metroid/tasks/warehouse_to_red_human_anchors/"
            "f002012_enter_0xA253_0xA253.state"
        ),
        route="ws-entrance",
        description=(
            "Red enter from warehouse_to_red_human live anchor f2012 "
            "~(216,2443) → Hellway. Prefer when matching human chain enemy phase."
        ),
        pure_hop=None,
        pure_source_rel=None,
        no_guide_default=True,
    ),
    # Bubble Save station → leave RIGHT → save-runway WJ climb (Phase D).
    # Prefer live timing gym: scripts/probe/bubble_save_practice.py
    "bubble-save": Segment(
        key="bubble-save",
        start="bubble-save",
        route="ws-entrance",  # free-record; guide shows Bubble path if present
        description=(
            "Bubble Save 0xB0DD ~(96,152) items 0x1105 → leave RIGHT → "
            "Bubble runway walljump climb. For EARLY/LATE frames_off feedback "
            "use bubble_save_practice.py instead of this multi-take wrapper."
        ),
        pure_hop=None,
        pure_source_rel="scratch/bubble_save.state",
        no_guide_default=False,
    ),
}


def _series_dir(series: str, out_root: Path) -> Path:
    return out_root / series


def _next_take_index(series_dir: Path, series: str) -> int:
    """Next 1-based take number from existing ``{series}_takeNN.json`` files."""
    if not series_dir.is_dir():
        return 1
    best = 0
    prefix = f"{series}_take"
    for p in series_dir.glob(f"{prefix}*.json"):
        stem = p.stem  # e.g. dc_missile_v1_take03
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        if tail.isdigit():
            best = max(best, int(tail))
    return best + 1


def _summarize_take(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: unreadable ({exc})"
    frames = int(data.get("frame_count") or len(data.get("frames") or []))
    trace = data.get("trace") or []
    end = trace[-1] if trace else {}
    room = end.get("room_hex") or (
        f"0x{int(end['room']):04X}" if "room" in end else "?"
    )
    xy = (
        f"({end.get('x')},{end.get('y')})"
        if "x" in end and "y" in end
        else "?"
    )
    mis = end.get("missiles", "?")
    recorded = data.get("recorded_at") or ""
    meta = data.get("metadata") or {}
    rooms_visited = meta.get("rooms") or meta.get("room_sequence")
    extra = ""
    if isinstance(rooms_visited, list) and rooms_visited:
        extra = f" rooms={rooms_visited[-3:]}"
    # Wave success heuristic from end room.
    ok = ""
    try:
        rid = int(str(room).replace("0x", ""), 16) if room != "?" else -1
        if rid == _ROOM_WAVE:
            ok = "  ✓ Wave room"
        elif rid == _ROOM_DOUBLE:
            ok = "  (still Double Chamber)"
    except ValueError:
        pass
    return (
        f"{path.name}: frames={frames} end={room} xy={xy} mis={mis} "
        f"{recorded[:19]}{extra}{ok}"
    )


def _list_takes(series: str, out_root: Path) -> int:
    d = _series_dir(series, out_root)
    if not d.is_dir():
        print(f"No series directory: {d}")
        return 1
    paths = sorted(d.glob(f"{series}_take*.json"))
    if not paths:
        print(f"No takes under {d}")
        return 1
    print(f"Series {series}  ({len(paths)} takes)  dir={d}")
    for p in paths:
        print(f"  {_summarize_take(p)}")
    print()
    print("Replay tip: load task JSON frames into a controller scaffold, or")
    print("  compare end xy/room vs bot pure dual below.")
    return 0


def _bot_check(seg: Segment) -> int:
    if not seg.pure_hop or not seg.pure_source_rel:
        print(
            f"Segment {seg.key} has no pure hop/source for --bot-check.",
            file=sys.stderr,
        )
        return 1
    source = INTEGRATION_DIR / seg.pure_source_rel
    if not source.is_file():
        print(f"ERROR: bot source missing: {source}", file=sys.stderr)
        return 1
    print("=" * 60)
    print(f"BOT CHECK  pure {seg.pure_hop}")
    print(f"  source: {source}")
    print("=" * 60)
    cmd = [
        "uv",
        "run",
        "python",
        str(KPDR_PROBE),
        "pure",
        seg.pure_hop,
        "--source",
        str(source),
        "--no-red-diag",
    ]
    # Dual: run twice
    codes = []
    for i in (1, 2):
        print(f"\n--- bot run {i}/2 ---")
        r = subprocess.run(cmd, cwd=str(ROOT))
        codes.append(r.returncode)
    if all(c == 0 for c in codes):
        print("\n[bot-check] both pure runs exited 0 (inspect stdout for success/room)")
        return 0
    print(f"\n[bot-check] non-zero exits: {codes}", file=sys.stderr)
    return 1


def _run_one_take(
    *,
    seg: Segment,
    take_name: str,
    out_dir: Path,
    no_guide: bool,
    no_assist: bool,
    scale: int,
    start_override: str | None = None,
) -> int:
    """Invoke guided_human for one take. Return process exit code."""
    start = start_override or seg.start
    cmd = [
        "uv",
        "run",
        "python",
        str(GUIDED),
        "--from",
        start,
        "--route",
        seg.route,
        "--name",
        take_name,
        "--out-dir",
        str(out_dir),
        "--scale",
        str(scale),
    ]
    if no_guide:
        cmd.append("--no-guide")
    if no_assist:
        cmd.append("--no-assist")
    print()
    print("=" * 60)
    print(f"TAKE  {take_name}")
    print(f"  segment: {seg.key} — {seg.description}")
    print(f"  start={start}  route={seg.route}")
    print(f"  out: {out_dir / (take_name + '.json')}")
    print("  F5/F1 = save take · ESC/Q = cancel take & end series")
    if seg.key.startswith("red-to-hellway"):
        print("  Red climb: F6 thin seat + ice tiers · F5 in Hellway 0xA2F7")
        print("  Fail mid-climb → ESC (no save) or F5 only if useful mid pin")
    print("=" * 60)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segment",
        default="dc-missile-wave",
        choices=sorted(SEGMENTS),
        help="Practice segment (default: dc-missile-wave Spazer cont-like)",
    )
    parser.add_argument(
        "--series",
        default=None,
        help=(
            "Series stem for take names (default: <segment>_<YYYYMMDD>). "
            "Takes: tasks/<series>/<series>_take01.json …"
        ),
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Max takes this session (default 20; ESC ends early)",
    )
    parser.add_argument(
        "--start-take",
        type=int,
        default=None,
        help="Force first take index (default: continue after last existing)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List takes for --series and exit",
    )
    parser.add_argument(
        "--bot-check",
        action="store_true",
        help="Run pure dual bot probe for segment source (no human window)",
    )
    parser.add_argument(
        "--list-segments",
        action="store_true",
        help="List segment presets and exit",
    )
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Force guide overlay off (default on for some segments)",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Force guide overlay on (waypoints for missile free/runway)",
    )
    parser.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable unlimited energy/ammo",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=TASKS_DIR,
        help=f"Root for series folders (default: {TASKS_DIR})",
    )
    parser.add_argument(
        "--from",
        dest="start_override",
        default=None,
        help=(
            "Override segment start pin (preset name, stem, or path). "
            "Use to boot any Red enter / mid F6 for more climb attempts."
        ),
    )
    args = parser.parse_args()

    if args.list_segments:
        print("Segments:")
        for key, seg in sorted(SEGMENTS.items()):
            print(f"  {key}")
            print(f"    {seg.description}")
            print(f"    --from {seg.start} --route {seg.route}")
            if seg.pure_hop:
                print(f"    bot: pure {seg.pure_hop} ← {seg.pure_source_rel}")
        return 0

    series = args.series or (
        f"{args.segment}_{datetime.now().strftime('%Y%m%d')}"
    )
    out_root = Path(args.out_root)
    if args.list:
        return _list_takes(series, out_root)

    seg = SEGMENTS[args.segment]
    if args.bot_check:
        return _bot_check(seg)

    # Guide: segment default unless forced.
    if args.guide:
        no_guide = False
    elif args.no_guide:
        no_guide = True
    else:
        no_guide = seg.no_guide_default

    series_dir = _series_dir(series, out_root)
    series_dir.mkdir(parents=True, exist_ok=True)
    take_i = args.start_take if args.start_take is not None else _next_take_index(
        series_dir, series
    )

    print("PRACTICE TAKES")
    print(f"  segment: {seg.key}")
    print(f"  series:  {series}")
    print(f"  dir:     {series_dir}")
    print(f"  first:   take{take_i:02d}  max this session: {args.max}")
    print(f"  guide:   {'OFF' if no_guide else 'ON'}  assist={'OFF' if args.no_assist else 'ON'}")
    if args.start_override:
        print(f"  --from:  {args.start_override}  (overrides segment start)")
    print()
    if seg.key.startswith("red-to-hellway"):
        print("Recipe (Red → Hellway, rr-av5s) — see docs/tasks/RED_CLIMB_HUMAN.md:")
        print("  Boot Red bottom ~(216,2443). Climb clean; F6: midplat, thin seat")
        print("  ~(91,587), ice tiers y495/391/295/207, Hellway enter → F5.")
        print("  ESC = discard take (reload next). Rank: rank_red_climb_takes.py")
        print("  Splice = pick best take + hop-replay from live anchors, NOT RLE concat.")
    else:
        print("Recipe (dc-missile-wave / Spazer) — reference take04:")
        print("  P1 purple hop upper y≲180 → gate seat ~(379,139)")
        print("     FALL: red path floor → climb → reseat P2 (do not Super from floor)")
        print("  P2 open gate → past ~(480,139)")
        print("  P3 missile ~x494 free RIGHT+B ~400f past x≥510 (take04 free=406f)")
        print("  P4 runway ~x437 → dash edge ~600")
        print("  P5 launch peak y≈60 → door WJ → Super → Wave 0xADDE → F5")
        print("  paths: routes/kpdr/data/dc_missile_wave_take04_paths.json")
        print("  ref:   tasks/dc_missile_v1/dc_missile_v1_take04.json")
    print()

    saved = 0
    for n in range(args.max):
        take_name = f"{series}_take{take_i:02d}"
        code = _run_one_take(
            seg=seg,
            take_name=take_name,
            out_dir=series_dir,
            no_guide=no_guide,
            no_assist=args.no_assist,
            scale=args.scale,
            start_override=args.start_override,
        )
        task_path = series_dir / f"{take_name}.json"
        if task_path.is_file():
            saved += 1
            print(f"[series] saved {_summarize_take(task_path)}")
            take_i += 1
            continue
        # No save → user cancelled (ESC) or crash; end series.
        if code != 0:
            print(f"[series] take exited {code}; stopping.")
        else:
            print("[series] take not saved (ESC/cancel) — end series.")
        break

    print()
    print(f"[series] done. saved_this_session={saved}  dir={series_dir}")
    print(f"  list: uv run python {Path(__file__).relative_to(ROOT)} "
          f"--series {series} --list")
    if seg.pure_hop:
        print(
            f"  bot:  uv run python {Path(__file__).relative_to(ROOT)} "
            f"--segment {seg.key} --bot-check"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
