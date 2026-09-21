"""Continuous opening spine: power-on through ALTTP's verified room-0x50 tip.

Wraps existing continuous segments. ``--through room_50`` (default) calls
``full_tip.run_to_verified_tip``. ``room_01``, ``room_72``, and ``zelda``
fail closed until they are on the continuous tip. Stairs are not composed
into ``full_tip``; do not run them on a clean power-on. Does not overwrite
``recordings/verified_tip_run.json``. ``--no-video`` is the default.

    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --no-video
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_50 --no-video
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_01 --no-video
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_72 --no-video
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through zelda --no-video
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from alttp.opening_route.escape_graph import N_ROOM_50
from alttp.opening_route.full_tip import FullTipResult, run_to_verified_tip
from alttp.paths import RECORDINGS_DIR
from alttp.ram import AlttpSnapshot
from alttp.startup import build_boot_env

SPINE_THROUGH: tuple[str, ...] = ("room_50", "room_01", "room_72", "zelda")
DEFAULT_THROUGH = "room_50"
ROOM_01_BLOCKER = "room_01 not on continuous tip"
ROOM_72_BLOCKER = "room_72 not on continuous tip; stairs not composed into full_tip"
ZELDA_BLOCKER = "zelda not on continuous tip; $F3CC==1 not measured"
VERIFIED_TIP_RUN = RECORDINGS_DIR / "verified_tip_run.json"

_TipFn = Callable[..., FullTipResult]


def leftover_path(through: str) -> Path:
    """Leftover JSON for one ``--through`` stop. Never the verified-tip artifact."""
    return RECORDINGS_DIR / f"opening_spine_{through}.json"


def leftover_glance(snapshot: AlttpSnapshot) -> dict[str, Any]:
    """RAM glance used as leave proof (no MP4)."""
    return {
        "room_hex": f"0x{snapshot.room_base_id:02X}",
        "module": snapshot.game_mode,
        "submodule": snapshot.submodule,
        "xy": [snapshot.link_x, snapshot.link_y],
        "sword": snapshot.sword_level,
        "follower": snapshot.follower,
        "follower_addr": "$F3CC",
        "keys": snapshot.num_keys,
        "indoors": snapshot.indoors,
        "has_control": snapshot.has_control,
        "has_zelda_follower": snapshot.has_zelda_follower,
    }


def fail_closed_payload(through: str, blocker: str, *, notes: list[str]) -> dict[str, Any]:
    return {
        "kind": "alttp_opening_spine",
        "through": through,
        "ok": False,
        "phase": "fail_closed",
        "frames": 0,
        "blocker": blocker,
        "tip_node": N_ROOM_50,
        "verified_tip": N_ROOM_50,
        "source": "fail_closed",
        "clean_chain": False,
        "continuous": False,
        "leftover": None,
        "video": None,
        "notes": list(notes),
    }


def spine_payload_from_tip(
    result: FullTipResult, *, through: str = DEFAULT_THROUGH
) -> dict[str, Any]:
    return {
        "kind": "alttp_opening_spine",
        "through": through,
        "ok": bool(result.ok),
        "phase": result.phase,
        "frames": result.frames,
        "blocker": result.blocker,
        "tip_node": result.tip_node,
        "verified_tip": N_ROOM_50,
        "source": result.source,
        "clean_chain": result.source == "natural_boot" and bool(result.ok),
        "continuous": through == DEFAULT_THROUGH and bool(result.ok),
        "leftover": leftover_glance(result.snapshot),
        "video": None,
        "notes": list(result.notes),
        "report": result.to_report(),
    }


def run_opening_spine(
    through: str,
    *,
    env: object | None = None,
    close: bool = True,
    run_tip_fn: _TipFn | None = None,
) -> dict[str, Any]:
    """Dispatch one spine stop. Fail-closed stubs never boot the ROM."""
    if through == "room_50":
        tip_fn = run_to_verified_tip if run_tip_fn is None else run_tip_fn
        return spine_payload_from_tip(tip_fn(env, close=close), through=through)
    if through == "room_01":
        return fail_closed_payload(
            through,
            ROOM_01_BLOCKER,
            notes=[
                "0x50 east→0x01 is graph natural_entry, not continuous.",
                "Fail-closed stub; ROM not booted. Did not STATUS-promote.",
            ],
        )
    if through == "room_72":
        return fail_closed_payload(
            through,
            ROOM_72_BLOCKER,
            notes=[
                "0x01 stairs→0x72 is not continuous even if natural_entry.",
                "Fail-closed stub; ROM not booted. Did not STATUS-promote.",
            ],
        )
    if through == "zelda":
        return fail_closed_payload(
            through,
            ZELDA_BLOCKER,
            notes=[
                "Follower $F3CC==1 is not measured on the continuous path.",
                "Fail-closed stub; ROM not booted. Did not STATUS-promote.",
            ],
        )
    known = ", ".join(SPINE_THROUGH)
    raise ValueError(f"unknown --through {through!r}; known: {known}")


def write_leftover(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    banned = VERIFIED_TIP_RUN.expanduser().resolve()
    if resolved == banned:
        raise RuntimeError(
            "refusing to overwrite recordings/verified_tip_run.json; "
            "use recordings/opening_spine_<through>.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through",
        choices=SPINE_THROUGH,
        default=DEFAULT_THROUGH,
        help=(
            "Spine stop. room_50 is the verified continuous tip. "
            "room_01, room_72, and zelda fail closed until composed into full_tip."
        ),
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record MP4 (default off). Encode is not wired; leftover JSON only.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Leftover JSON path (default: recordings/opening_spine_<through>.json)",
    )
    return parser


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_headless()
    report_path = args.report if args.report is not None else leftover_path(args.through)
    env: object | None = None
    try:
        if args.through == "room_50":
            env = build_boot_env()
        payload = run_opening_spine(args.through, env=env, close=False)
        payload["video"] = None
        if args.video:
            notes = list(payload.get("notes") or [])
            notes.append(
                "--video requested; MP4 encode is not wired on this CLI (leftover JSON only)."
            )
            payload["notes"] = notes
    finally:
        if env is not None:
            env.close()  # type: ignore[attr-defined]

    write_leftover(report_path, payload)
    print(f"Wrote {report_path}")
    print(
        f"ok={payload['ok']} through={payload['through']} "
        f"phase={payload.get('phase')} frames={payload.get('frames', 0)} "
        f"blocker={payload.get('blocker')!r}"
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
