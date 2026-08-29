"""Pure HappyLee track isolation (track 3).

Parked: no hybrid / natural_82 / skills / flamexx writes. Live 8-3 search
CLI is deleted (git restores). Isolation gates stay so other tracks cannot
be overwritten from this directory.
"""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from smb.paths import GAME_DIR, MODELS_DIR, RECORDINGS_DIR
from smb.tas.slice import (
    DEFAULT_FM2,
    HL_8_1_FM2_START,
    HL_8_1_LEAVE_FRAMES,
    HL_8_2_FM2_START,
    HL_8_2_LEAVE_FRAMES,
)

PURE_HL_MODELS = MODELS_DIR / "pure_hl"
PURE_HL_EVIDENCE = RECORDINGS_DIR / "tas_import" / "pure_hl"
PURE_HL_FM2 = DEFAULT_FM2
PURE_8_3_GATE = PURE_HL_MODELS / "gate_8_3_leave.json"
PURE_8_3_SEED = PURE_HL_MODELS / "smb_8_3_pure_hl.json"
TRACK_NAME = "pure_hl"

_FORBIDDEN_WRITE_GLOBS = (
    "smb_1_1_to_ending*",
    "smb_happylee_hybrid*",
    "smb_*_natural*",
    "smb_8_3_stitchless*",
    "smb_8_3_natural*",
    "smb_8_4_flamexx*",
    "smb_8_3_happylee_slice.json",
    "smb_8_1_happylee_slice.json",
    "smb_8_2_happylee_slice.json",
)

TRACK_RULES = (
    "HappyLee #1715M FM2 only",
    "no natural_82 splice",
    "no flamexx",
    "no skill macros",
    "no 8-4 until pure 8-3 leave verified",
    f"write only under {PURE_HL_MODELS.relative_to(GAME_DIR)} "
    f"and {PURE_HL_EVIDENCE.relative_to(GAME_DIR)}",
)


def ensure_pure_dirs() -> None:
    PURE_HL_MODELS.mkdir(parents=True, exist_ok=True)
    PURE_HL_EVIDENCE.mkdir(parents=True, exist_ok=True)


def assert_pure_write_path(path: Path) -> Path:
    """Raise if *path* is outside pure_hl trees or matches forbidden names."""
    path = path.resolve()
    allowed = (PURE_HL_MODELS.resolve(), PURE_HL_EVIDENCE.resolve())
    if not any(path == d or d in path.parents for d in allowed):
        raise RuntimeError(
            f"pure_hl refuse write outside track dirs: {path}\n"
            f"allowed: {PURE_HL_MODELS}, {PURE_HL_EVIDENCE}"
        )
    for pat in _FORBIDDEN_WRITE_GLOBS:
        if fnmatch(path.name, pat):
            raise RuntimeError(
                f"pure_hl refuse write to protected name {path.name!r} ({pat})"
            )
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_pure_dirs()
    path = assert_pure_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def pure_8_3_gate_open() -> bool:
    if not PURE_8_3_GATE.exists():
        return False
    try:
        data = json.loads(PURE_8_3_GATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("verified_leave_8_4_control")) and bool(
        data.get("pure_fm2_only")
    )


def track_status() -> dict[str, Any]:
    ensure_pure_dirs()
    open_gate = pure_8_3_gate_open()
    return {
        "track": TRACK_NAME,
        "rules": list(TRACK_RULES),
        "fm2": str(PURE_HL_FM2),
        "models_dir": str(PURE_HL_MODELS),
        "evidence_dir": str(PURE_HL_EVIDENCE),
        "gate_8_3_open": open_gate,
        "gate_file": str(PURE_8_3_GATE),
        "gate_exists": PURE_8_3_GATE.exists(),
        "pure_8_3_seed": str(PURE_8_3_SEED),
        "pure_8_3_seed_exists": PURE_8_3_SEED.exists(),
        "hl_indices": {
            "8_1_start": HL_8_1_FM2_START,
            "8_1_leave": HL_8_1_LEAVE_FRAMES,
            "8_2_start": HL_8_2_FM2_START,
            "8_2_leave": HL_8_2_LEAVE_FRAMES,
            "8_3_start": None,
        },
        "blocked": {
            "pure_8_4": not open_gate,
            "reason": None
            if open_gate
            else "pure 8-3 leave not verified — do not start 8-4",
        },
        "do_not_touch": [
            "models/smb_1_1_to_ending_natural_82.json",
            "models/smb_happylee_hybrid_v2_fx84.json",
            "models/smb_8_3_stitchless_skills_leave.json",
            "models/smb_8_3_natural_for_hl_hybrid.json",
            "models/smb_8_4_flamexx_slice.json",
        ],
    }


def select_leave_fan(
    unique: list[dict[str, Any]],
    *,
    top_leaves: int = 5,
    default_si82: int | None = None,
) -> list[dict[str, Any]]:
    """Fastest leave classes first, then default SI, then slower diversity."""
    if not unique:
        return []
    sorted_u = sorted(unique, key=lambda r: (r["leave82"], r["si82"]))
    fan: list[dict[str, Any]] = list(sorted_u[: max(1, top_leaves)])
    if default_si82 is not None:
        for row in sorted_u:
            if int(row["si82"]) == int(default_si82) and row not in fan:
                fan.append(row)
                break
    for row in reversed(sorted_u):
        if row not in fan and len(fan) < top_leaves + 2:
            fan.append(row)
    return fan


def refuse_8_4_until_gate() -> dict[str, Any]:
    open_gate = pure_8_3_gate_open()
    return {
        "allowed": open_gate,
        "gate_file": str(PURE_8_3_GATE),
        "message": (
            "pure 8-4 search allowed"
            if open_gate
            else "BLOCKED: pure 8-3 leave not verified — fix 8-3 sync first"
        ),
    }
