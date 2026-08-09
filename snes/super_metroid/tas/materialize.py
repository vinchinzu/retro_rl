"""Materialize control-relative snes12_rle seeds from stage windows / boards.

Slices movie button logs into RLE seeds for RoomStageSpec hops. **No emulator**
— parse only. Status is always ``materialized_unproven`` (not pure / not STATUS).

Hard rules (``docs/TAS_ADAPT.md``):

* Never sanitize L+R (raw SNES-12 via ``compress_snes12_rle``).
* Zebes-first: prefer Landing→Parlor short window, not Ceres thrash tails.
* Product-frame pins from resync runs are **not** movie indices — convert via
  ``movie_start`` / ``movie_index`` when present.

```bash
# Known-good Landing→Parlor seed (~2.2k frames to first Parlor, not 12k thrash)
uv run python -m super_metroid.tas.materialize --stage landing_to_parlor

# From resync board (pins + resync.json movie_index)
uv run python -m super_metroid.tas.materialize \\
  --from-board snes/super_metroid/recordings/tas_import/resync_zebes_rooms \\
  --zebes-only
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.paths import GAME_DIR, RECORDINGS_DIR
from super_metroid.tas.rle import frames_to_snes12_rle_payload, write_snes12_rle_seed
from super_metroid.tas.slice import load_movie_frames, slice_frames
from super_metroid.tas.stages import (
    ANY_LANDING_BODY_HINT,
    ANY_LANDING_MOVIE_START,
    REF_ANY,
    STAGE_CATALOG,
    RoomStageSpec,
    export_room_body_spec,
    get_stage,
    movie_window_from_pins,
    stages_with_tag,
)

TAS_DIR = GAME_DIR / "tas"
BODY_DIR = TAS_DIR / "bodies"
TAS_IMPORT = RECORDINGS_DIR / "tas_import"

# First Parlor enter under product Landing + movie@15000 is ~2192 movie frames
# (product f23740 − f21548). Cap below thrash tail (ANY_LANDING_BODY_HINT=12k).
ANY_LANDING_TO_PARLOR_BODY = 2_500

# Product control pin at Landing (resync / morph spine); product timeline ≠ movie.
PRODUCT_LANDING_FRAME = 21_548

STATUS_MATERIALIZED = "materialized_unproven"

HARD_RULES = [
    "never_sanitize_L+R",
    "assist_off",
    "re_anchor_before_status",
    "pure_first_before_graph_edge",
    "not_status_evidence",
]


def _rel_to_game(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(GAME_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _load_pins(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    if isinstance(raw, dict):
        for key in ("pins", "events", "room_milestones"):
            if isinstance(raw.get(key), list):
                return [p for p in raw[key] if isinstance(p, dict)]
    raise ValueError(f"unrecognized pins shape: {path}")


def _load_resync_meta(board_dir: Path) -> dict[str, Any] | None:
    for name in ("resync.json", "summary.json"):
        p = board_dir / name
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    return None


def _movie_index_from_resync(
    meta: Mapping[str, Any],
    goal_room_id: int,
) -> tuple[int | None, int | None]:
    """Return (movie_start, movie_index_at_goal) from resync rooms list if present."""
    movie_start = meta.get("movie_start")
    if movie_start is None and isinstance(meta.get("meta"), dict):
        movie_start = meta["meta"].get("movie_start")
    if movie_start is not None:
        movie_start = int(movie_start)

    rooms = meta.get("rooms")
    if not isinstance(rooms, list):
        rooms = (meta.get("meta") or {}).get("rooms") if isinstance(meta.get("meta"), dict) else None
    if not isinstance(rooms, list):
        # summary.json nests milestones under annotate / events_summary
        ann = meta.get("events_summary") or meta.get("annotate") or {}
        if isinstance(ann, dict):
            rooms = ann.get("room_milestones")

    goal_mi: int | None = None
    if isinstance(rooms, list):
        for row in rooms:
            if not isinstance(row, Mapping):
                continue
            rid = row.get("room_id")
            if rid is None:
                hx = row.get("room_id_hex") or row.get("room")
                if isinstance(hx, str) and hx.startswith("0x"):
                    rid = int(hx, 16)
            if rid is None:
                continue
            if int(rid) != int(goal_room_id):
                continue
            mi = row.get("movie_index")
            if mi is not None:
                goal_mi = int(mi)
                break
    return movie_start, goal_mi


def _pins_look_product_timeline(
    window: tuple[int, int],
    stage: RoomStageSpec,
) -> bool:
    """Heuristic: pin frames near product Landing / far above movie_start."""
    start, _end = window
    ms = stage.movie_start
    if ms is not None and start >= PRODUCT_LANDING_FRAME - 500:
        # Product morph / resync product frames start ~21k
        return True
    if ms is not None and start < ms:
        # Pin start before known movie splice index → not that movie index
        return True
    return False


def resolve_movie_window(
    stage: RoomStageSpec,
    *,
    pins: Sequence[Mapping[str, Any]] | None = None,
    movie_start: int | None = None,
    body_frames: int | None = None,
    resync_meta: Mapping[str, Any] | None = None,
    after_frame: int = 0,
) -> dict[str, Any]:
    """Resolve ``(movie_start, body_frames)`` for a stage.

    Priority:

    1. Explicit ``movie_start`` + ``body_frames`` kwargs.
    2. Resync meta ``movie_index`` at goal room (product+movie splice runs).
    3. Pin window via ``movie_window_from_pins`` (movie-index pins) or product
       delta converted with stage / resync ``movie_start``.
    4. Stage catalog hints, with Landing→Parlor thrash cap.
    """
    reason = "stage_catalog"
    start = movie_start if movie_start is not None else stage.movie_start
    body = body_frames if body_frames is not None else stage.body_frames
    pin_window: list[int] | None = None
    explicit = movie_start is not None and body_frames is not None

    if not explicit and resync_meta is not None and stage.goal_room_id is not None:
        rs_start, goal_mi = _movie_index_from_resync(resync_meta, stage.goal_room_id)
        if start is None and rs_start is not None:
            start = rs_start
        if goal_mi is not None and (start is not None or rs_start is not None):
            if start is None:
                start = int(rs_start)  # type: ignore[arg-type]
            # movie_index is power-on movie frame at goal enter; body = delta.
            # +32f settle margin so seed includes the enter sample.
            body = max(1, int(goal_mi) - int(start) + 32)
            reason = "resync_movie_index"

    if not explicit and pins and stage.goal_room_id is not None and reason == "stage_catalog":
        win = movie_window_from_pins(
            pins,
            from_room=stage.room_id,
            to_room=stage.goal_room_id,
            after_frame=after_frame,
        )
        # Also accept control pin as from_room start
        if win is None:
            # control kind at from_room then room_enter to_room
            enters = [
                p
                for p in pins
                if p.get("kind") in ("room_enter", "control")
                and int(p.get("frame", 0)) >= after_frame
            ]
            start_f: int | None = None
            for p in enters:
                rid = int(p.get("room_id") or 0)
                if rid == int(stage.room_id) and start_f is None:
                    start_f = int(p["frame"])
                elif (
                    start_f is not None
                    and rid == int(stage.goal_room_id)
                    and int(p["frame"]) > start_f
                ):
                    win = (start_f, int(p["frame"]))
                    break
        if win is not None:
            pin_window = [win[0], win[1]]
            if _pins_look_product_timeline(win, stage):
                # Product timeline: body length is delta; movie_start from stage/resync
                body = win[1] - win[0]
                if start is None:
                    start = stage.movie_start or ANY_LANDING_MOVIE_START
                reason = "product_pin_delta"
            else:
                start = win[0]
                body = win[1] - win[0]
                reason = "movie_pin_window"

    if start is None:
        start = stage.movie_start
    if body is None:
        body = stage.body_frames

    # Landing→Parlor: never materialize the 12k thrash tail by default
    if (
        stage.id == "landing_to_parlor"
        and not explicit
        and reason in ("stage_catalog",)
        and (body is None or body >= ANY_LANDING_BODY_HINT // 2)
    ):
        start = start if start is not None else ANY_LANDING_MOVIE_START
        body = ANY_LANDING_TO_PARLOR_BODY
        reason = "landing_to_parlor_short_default"

    if start is None:
        raise ValueError(
            f"stage {stage.id!r}: cannot resolve movie_start "
            f"(pass --movie-start or set stage.movie_start)"
        )
    if body is None or body <= 0:
        raise ValueError(
            f"stage {stage.id!r}: cannot resolve body_frames "
            f"(pass --body-frames or set stage.body_frames / pins)"
        )

    return {
        "movie_start": int(start),
        "body_frames": int(body),
        "reason": reason,
        "pin_window": pin_window,
    }


def default_out_path(stage_id: str, *, out_dir: Path | None = None) -> Path:
    base = out_dir or BODY_DIR
    return base / f"{stage_id}.json"


def materialize_room_body(
    stage_id: str | RoomStageSpec,
    *,
    pins: Sequence[Mapping[str, Any]] | None = None,
    movie_path: Path | None = None,
    movie_start: int | None = None,
    body_frames: int | None = None,
    out_path: Path | None = None,
    resync_meta: Mapping[str, Any] | None = None,
    frames: list[list[int]] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Slice movie frames for a stage window → snes12_rle seed dict (+ optional write).

    Parameters
    ----------
    stage_id:
        Catalog id or ``RoomStageSpec``.
    pins:
        Annotate pins (room_enter / control). Product-timeline pins need
        ``resync_meta`` or stage ``movie_start`` for conversion.
    movie_path:
        Override stage.movie (default Sniq any% LSMV).
    movie_start / body_frames:
        Explicit window overrides.
    out_path:
        JSON destination; default ``tas/bodies/<stage_id>.json``.
    resync_meta:
        Optional ``resync.json`` / summary dict with ``movie_index`` rooms.
    frames:
        Pre-parsed movie frames (avoids re-parse in batch).
    write:
        When False, return payload without writing.

    Returns
    -------
    dict
        snes12_rle payload plus stage metadata (``status=materialized_unproven``).
    """
    stage = get_stage(stage_id) if isinstance(stage_id, str) else stage_id
    window = resolve_movie_window(
        stage,
        pins=pins,
        movie_start=movie_start,
        body_frames=body_frames,
        resync_meta=resync_meta,
    )
    ms = int(window["movie_start"])
    nbody = int(window["body_frames"])

    movie = Path(movie_path) if movie_path is not None else stage.movie
    if movie is None:
        movie = REF_ANY
    movie = Path(movie)
    if frames is None:
        if not movie.exists():
            raise FileNotFoundError(f"missing ref movie: {movie}")
        frames = load_movie_frames(movie)

    end = ms + nbody
    body = slice_frames(frames, ms, end)

    plan = export_room_body_spec(stage, list(pins or []))
    # Override plan window with resolved materialize window
    plan["movie_start"] = ms
    plan["body_frames"] = nbody
    plan["window_resolve"] = window["reason"]
    if window.get("pin_window"):
        plan["window_from_pins"] = window["pin_window"]

    out = out_path or default_out_path(stage.id)
    rel_movie = _rel_to_game(movie)
    rel_out = _rel_to_game(out)

    extra: dict[str, Any] = {
        "stage_id": stage.id,
        "track": stage.track,
        "control_room_id": stage.room_id,
        "control_room_id_hex": stage.room_hex(),
        "goal_room_id": stage.goal_room_id,
        "goal_room_id_hex": stage.goal_room_hex(),
        "movie": rel_movie,
        "movie_start_index": ms,
        "movie_end_index": end,
        "movie_num_frames": len(frames),
        "body_frames": nbody,
        "window_resolve": window["reason"],
        "status": STATUS_MATERIALIZED,
        "hard_rules": list(HARD_RULES),
        "tech": list(stage.tech),
        "tags": list(stage.tags),
        "note": stage.note,
        "plan": {
            "schema": plan.get("schema"),
            "control": plan.get("control"),
            "goal": plan.get("goal"),
            "window_from_pins": plan.get("window_from_pins"),
            "window_resolve": window["reason"],
        },
        "seed_path": rel_out,
    }

    payload = frames_to_snes12_rle_payload(
        body,
        route_id=f"body_{stage.id}",
        source=f"materialize:{stage.id}@{ms}+{nbody}",
        extra=extra,
    )

    if write:
        write_snes12_rle_seed(out, payload)
        payload["written_path"] = str(out)

    return payload


def materialize_from_board(
    board_dir: Path | str,
    *,
    stage_ids: Sequence[str] | None = None,
    zebes_only: bool = False,
    out_dir: Path | None = None,
    movie_path: Path | None = None,
    frames: list[list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Materialize stage seeds using pins (+ resync meta) under *board_dir*.

    Loads ``pins.json`` when present. Stages without a resolvable movie window
    are skipped with an error entry (not raised).
    """
    board_dir = Path(board_dir)
    pins: list[dict[str, Any]] = []
    pins_path = board_dir / "pins.json"
    if pins_path.is_file():
        pins = _load_pins(pins_path)

    # Prefer extraction_board stage_body_exports for stage list
    board_path = board_dir / "extraction_board.json"
    board: dict[str, Any] | None = None
    if board_path.is_file():
        board = json.loads(board_path.read_text(encoding="utf-8"))

    resync_meta = _load_resync_meta(board_dir)

    if stage_ids is not None:
        ids = list(stage_ids)
    elif zebes_only:
        ids = [s.id for s in stages_with_tag("zebes")]
    elif board and board.get("stage_body_exports"):
        ids = []
        for exp in board["stage_body_exports"]:
            sid = exp.get("stage_id")
            if sid and sid not in ids:
                ids.append(sid)
        # Always include zebes stages if present in catalog and board has landing pins
        for s in stages_with_tag("zebes"):
            if s.id not in ids and s.movie_start is not None:
                ids.append(s.id)
    else:
        # Default: stages with movie_start set
        ids = [
            s.id
            for s in STAGE_CATALOG.values()
            if s.movie_start is not None and s.body_frames is not None
        ]
        if zebes_only:
            ids = [i for i in ids if "zebes" in STAGE_CATALOG[i].tags]

    if zebes_only:
        ids = [i for i in ids if "zebes" in STAGE_CATALOG[i].tags]

    out_root = out_dir or (board_dir / "bodies")
    results: dict[str, dict[str, Any]] = {}
    movie_cache = frames

    for sid in ids:
        stage = get_stage(sid)
        if stage.movie is None and movie_path is None:
            results[sid] = {"error": "no movie", "status": "skipped"}
            continue
        try:
            movie = Path(movie_path) if movie_path else stage.movie
            if movie_cache is None and movie is not None and movie.exists():
                movie_cache = load_movie_frames(movie)
            payload = materialize_room_body(
                stage,
                pins=pins,
                movie_path=movie_path,
                out_path=out_root / f"{sid}.json",
                resync_meta=resync_meta,
                frames=movie_cache,
                write=True,
            )
            results[sid] = {
                "status": payload.get("status"),
                "num_frames": payload.get("num_frames"),
                "movie_start_index": payload.get("movie_start_index"),
                "body_frames": payload.get("body_frames"),
                "window_resolve": payload.get("window_resolve"),
                "path": payload.get("written_path") or str(out_root / f"{sid}.json"),
            }
        except (ValueError, FileNotFoundError, KeyError) as exc:
            results[sid] = {"error": str(exc), "status": "skipped"}

    manifest = {
        "schema": "sm_tas_materialize_board_v1",
        "board_dir": str(board_dir),
        "zebes_only": zebes_only,
        "results": results,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        action="append",
        dest="stages",
        default=None,
        help="Stage id (repeatable). Default: landing_to_parlor",
    )
    p.add_argument(
        "--from-board",
        type=Path,
        default=None,
        help="Annotate / resync dir with pins.json (and optional resync.json)",
    )
    p.add_argument(
        "--zebes-only",
        action="store_true",
        help="Filter to zebes-tagged stages (skip Ceres)",
    )
    p.add_argument(
        "--movie",
        type=Path,
        default=None,
        help="Override ref movie path",
    )
    p.add_argument(
        "--movie-start",
        type=int,
        default=None,
        help="Override movie start index",
    )
    p.add_argument(
        "--body-frames",
        type=int,
        default=None,
        help="Override body length (frames)",
    )
    p.add_argument(
        "--pins",
        type=Path,
        default=None,
        help="pins.json path (else board pins.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (single stage) or directory (batch)",
    )
    p.add_argument(
        "--list-stages",
        action="store_true",
        help="List STAGE_CATALOG and exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve windows only; do not parse movie or write",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_stages:
        for sid, st in STAGE_CATALOG.items():
            tags = ",".join(st.tags)
            print(
                f"{sid:28s} track={st.track:8s} tags={tags:32s} "
                f"ms={st.movie_start} body={st.body_frames}"
            )
        return 0

    if args.from_board is not None:
        board = Path(args.from_board)
        if not board.is_dir():
            # allow relative to GAME_DIR / recordings
            alt = GAME_DIR / board
            if alt.is_dir():
                board = alt
            else:
                alt2 = TAS_IMPORT / board.name
                if alt2.is_dir():
                    board = alt2
        out_dir = args.out if args.out and args.out.suffix != ".json" else None
        if args.dry_run:
            pins: list[dict[str, Any]] = []
            if (board / "pins.json").is_file():
                pins = _load_pins(board / "pins.json")
            meta = _load_resync_meta(board)
            stages = args.stages
            if not stages:
                if args.zebes_only:
                    stages = [s.id for s in stages_with_tag("zebes")]
                else:
                    stages = [
                        s.id
                        for s in STAGE_CATALOG.values()
                        if s.movie_start is not None
                    ]
            if args.zebes_only:
                stages = [
                    i
                    for i in stages
                    if "ceres" not in STAGE_CATALOG[i].tags
                ]
            for sid in stages:
                st = get_stage(sid)
                try:
                    w = resolve_movie_window(
                        st,
                        pins=pins,
                        movie_start=args.movie_start,
                        body_frames=args.body_frames,
                        resync_meta=meta,
                    )
                    print(
                        f"{sid}: movie_start={w['movie_start']} "
                        f"body_frames={w['body_frames']} ({w['reason']})"
                    )
                except ValueError as exc:
                    print(f"{sid}: SKIP {exc}", file=sys.stderr)
            return 0

        results = materialize_from_board(
            board,
            stage_ids=args.stages,
            zebes_only=args.zebes_only,
            out_dir=out_dir or (args.out if args.out and args.out.is_dir() else None),
            movie_path=args.movie,
        )
        for sid, info in results.items():
            if info.get("status") == "skipped" or "error" in info:
                print(f"{sid}: skip — {info.get('error', info)}", file=sys.stderr)
            else:
                print(
                    f"{sid}: {info['num_frames']} frames "
                    f"@ms={info['movie_start_index']} ({info['window_resolve']}) "
                    f"→ {info['path']}"
                )
        return 0 if any(r.get("num_frames") for r in results.values()) else 1

    stages = args.stages or ["landing_to_parlor"]
    if args.zebes_only:
        stages = [s for s in stages if "ceres" not in get_stage(s).tags]

    pins: list[dict[str, Any]] | None = None
    if args.pins is not None:
        pins = _load_pins(args.pins)
    resync_meta = None
    if args.pins and args.pins.parent.is_dir():
        resync_meta = _load_resync_meta(args.pins.parent)

    # Default pins from resync_zebes_rooms when materializing landing_to_parlor
    if pins is None and any(s == "landing_to_parlor" for s in stages):
        default_board = TAS_IMPORT / "resync_zebes_rooms"
        if (default_board / "pins.json").is_file():
            pins = _load_pins(default_board / "pins.json")
            resync_meta = _load_resync_meta(default_board)

    movie_cache: list[list[int]] | None = None
    for i, sid in enumerate(stages):
        stage = get_stage(sid)
        if args.dry_run:
            w = resolve_movie_window(
                stage,
                pins=pins,
                movie_start=args.movie_start,
                body_frames=args.body_frames,
                resync_meta=resync_meta,
            )
            print(
                f"{sid}: movie_start={w['movie_start']} "
                f"body_frames={w['body_frames']} ({w['reason']})"
            )
            continue

        out = args.out
        if out is not None and (len(stages) > 1 or out.is_dir() or out.suffix != ".json"):
            out_dir = out if out.suffix != ".json" else out.parent
            out = out_dir / f"{sid}.json"
        elif out is None:
            out = default_out_path(sid)

        try:
            payload = materialize_room_body(
                stage,
                pins=pins,
                movie_path=args.movie,
                movie_start=args.movie_start,
                body_frames=args.body_frames,
                out_path=out,
                resync_meta=resync_meta,
                frames=movie_cache,
            )
            if movie_cache is None:
                # Keep frames only if we might reuse; re-parse is fine for one stage
                pass
            print(
                f"{sid}: {payload['num_frames']} frames "
                f"@ms={payload['movie_start_index']} "
                f"({payload['window_resolve']}) → {payload.get('written_path', out)}"
            )
        except FileNotFoundError as exc:
            print(f"{sid}: {exc}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"{sid}: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
