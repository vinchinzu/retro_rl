"""Pure HappyLee reproduction track (track 3).

Isolated from:
  - Clean ``natural_82`` (M8 STATUS)
  - Hybrid v2 (HL…8-2 + natural 8-3 + flamexx 8-4)
  - Stitchless skills 8-3 leave

Rules
-----
- Only HappyLee #1715M FM2 frames (no flamexx, no natural_82, no skill macros).
- Seeds write **only** under ``models/pure_hl/``.
- Evidence writes **only** under ``recordings/tas_import/pure_hl/``.
- **Do not start pure 8-4** until pure 8-3 leaves to 8-4 control and verifies.
- No advance past a desync: re-gate / re-search / fix phase first.

CLI: ``uv run python -m smb.scripts.pure_hl …``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.ram import PLAYER_STATE_DYING, reached_ending, read_snapshot
from smb.tas.fm2 import frames_to_nes9_rle_payload, parse_fm2
from smb.tas.replay import IDLE, get_state, idle_until, set_state, to_action9
from smb.tas.chain import reach_8_1_control_after_hl_w8
from smb.tas.slice import (
    DEFAULT_FM2,
    DEFAULT_HL_1_1,
    HL_8_1_FM2_START,
    HL_8_1_LEAVE_FRAMES,
    HL_8_2_FM2_START,
    HL_8_2_LEAVE_FRAMES,
    is_8_2_control,
    is_8_3_control,
    is_8_4_control,
    probe_8_1_from_control,
    probe_8_2_from_control,
    probe_8_3_from_control,
)
from smb.tas.stages import snap_fingerprint

# ---------------------------------------------------------------------------
# Isolation paths (hard rules)
# ---------------------------------------------------------------------------

PURE_HL_MODELS = MODELS_DIR / "pure_hl"
PURE_HL_EVIDENCE = RECORDINGS_DIR / "tas_import" / "pure_hl"
PURE_HL_FM2 = DEFAULT_FM2  # HappyLee only

# Existing seeds we must never overwrite (other tracks).
_FORBIDDEN_WRITE_GLOBS = (
    "smb_1_1_to_ending*",
    "smb_happylee_hybrid*",
    "smb_*_natural*",
    "smb_8_3_stitchless*",
    "smb_8_3_natural*",
    "smb_8_4_flamexx*",
    "smb_8_3_happylee_slice.json",  # historically non-pure mid-heal
    "smb_8_1_happylee_slice.json",  # shared verified; pure track re-exports own copy
    "smb_8_2_happylee_slice.json",
)

TRACK_NAME = "pure_hl"
TRACK_RULES = (
    "HappyLee #1715M FM2 only",
    "no natural_82 splice",
    "no flamexx",
    "no skill macros",
    "no 8-4 until pure 8-3 leave verified",
    f"write only under {PURE_HL_MODELS.relative_to(GAME_DIR)} "
    f"and {PURE_HL_EVIDENCE.relative_to(GAME_DIR)}",
)

# Gate file: written only when pure 8-3 → 8-4 control is verified.
PURE_8_3_GATE = PURE_HL_MODELS / "gate_8_3_leave.json"
PURE_8_3_SEED = PURE_HL_MODELS / "smb_8_3_pure_hl.json"


def ensure_pure_dirs() -> None:
    PURE_HL_MODELS.mkdir(parents=True, exist_ok=True)
    PURE_HL_EVIDENCE.mkdir(parents=True, exist_ok=True)


def assert_pure_write_path(path: Path) -> Path:
    """Raise if *path* is outside pure_hl trees or matches forbidden names."""
    from fnmatch import fnmatch

    path = path.resolve()
    allowed = (PURE_HL_MODELS.resolve(), PURE_HL_EVIDENCE.resolve())
    if not any(path == d or d in path.parents for d in allowed):
        raise RuntimeError(
            f"pure_hl refuse write outside track dirs: {path}\n"
            f"allowed: {PURE_HL_MODELS}, {PURE_HL_EVIDENCE}"
        )
    # Extra: never write a filename that collides with other-track seeds.
    name = path.name
    for pat in _FORBIDDEN_WRITE_GLOBS:
        if fnmatch(name, pat):
            raise RuntimeError(
                f"pure_hl refuse write to protected name {name!r} ({pat})"
            )
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_pure_dirs()
    path = assert_pure_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def pure_8_3_gate_open() -> bool:
    """True only when a verified pure 8-3 leave gate file exists."""
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
    return {
        "track": TRACK_NAME,
        "rules": list(TRACK_RULES),
        "fm2": str(PURE_HL_FM2),
        "models_dir": str(PURE_HL_MODELS),
        "evidence_dir": str(PURE_HL_EVIDENCE),
        "gate_8_3_open": pure_8_3_gate_open(),
        "gate_file": str(PURE_8_3_GATE),
        "gate_exists": PURE_8_3_GATE.exists(),
        "pure_8_3_seed": str(PURE_8_3_SEED),
        "pure_8_3_seed_exists": PURE_8_3_SEED.exists(),
        "hl_indices": {
            "8_1_start": HL_8_1_FM2_START,
            "8_1_leave": HL_8_1_LEAVE_FRAMES,
            "8_2_start": HL_8_2_FM2_START,
            "8_2_leave": HL_8_2_LEAVE_FRAMES,
            "8_3_start": None,  # open until pure leave
        },
        "blocked": {
            "pure_8_4": not pure_8_3_gate_open(),
            "reason": (
                None
                if pure_8_3_gate_open()
                else "pure 8-3 leave not verified — do not start 8-4"
            ),
        },
        "do_not_touch": [
            "models/smb_1_1_to_ending_natural_82.json",
            "models/smb_happylee_hybrid_v2_fx84.json",
            "models/smb_8_3_stitchless_skills_leave.json",
            "models/smb_8_3_natural_for_hl_hybrid.json",
            "models/smb_8_4_flamexx_slice.json",
        ],
    }


@dataclass
class PureChainTo83:
    """HL pure predecessor through 8-3 control (idle blackouts)."""

    success: bool
    leave_8_1: int | None = None
    wait_8_2: int | None = None
    leave_8_2: int | None = None
    wait_8_3: int | None = None
    control_8_3_fp: dict[str, int] | None = None
    ctrl_wait_8_1: int | None = None
    error: str | None = None
    approx_total_to_8_3_control: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "ctrl_wait_8_1": self.ctrl_wait_8_1,
            "leave_8_1": self.leave_8_1,
            "wait_8_2": self.wait_8_2,
            "leave_8_2": self.leave_8_2,
            "wait_8_3": self.wait_8_3,
            "control_8_3_fp": self.control_8_3_fp,
            "approx_total_to_8_3_control": self.approx_total_to_8_3_control,
            "error": self.error,
            "track": TRACK_NAME,
            "pure_fm2_only": True,
        }


def build_pure_to_8_3_control(
    env,
    fm2: list[list[int]],
    *,
    start_8_1: int = HL_8_1_FM2_START,
    start_8_2: int = HL_8_2_FM2_START,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
) -> PureChainTo83:
    """Drive env from Level1_1 HL chain to first ``is_8_3_control`` frame."""
    pred = reach_8_1_control_after_hl_w8(env, fm2_path=PURE_HL_FM2, hl_1_1_path=hl_1_1_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        return PureChainTo83(
            success=False,
            error=f"8_1_control_failed:{pred.get('stage')}",
            ctrl_wait_8_1=pred.get("ctrl_wait_8_1"),
        )

    lives = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(env, fm2, start_8_1, start_lives=lives)
    if not tr81.ok:
        return PureChainTo83(
            success=False,
            error="8_1_body_failed",
            leave_8_1=tr81.leave_frame,
            ctrl_wait_8_1=pred.get("ctrl_wait_8_1"),
        )

    wait82, snap82 = idle_until(env, is_8_2_control)
    if not is_8_2_control(snap82):
        return PureChainTo83(
            success=False,
            error="8_2_control_timeout",
            leave_8_1=tr81.leave_frame,
            wait_8_2=wait82,
            ctrl_wait_8_1=pred.get("ctrl_wait_8_1"),
        )

    tr82 = probe_8_2_from_control(env, fm2, start_8_2, start_lives=int(snap82.lives))
    if not tr82.ok:
        return PureChainTo83(
            success=False,
            error="8_2_body_failed",
            leave_8_1=tr81.leave_frame,
            wait_8_2=wait82,
            leave_8_2=tr82.leave_frame,
            ctrl_wait_8_1=pred.get("ctrl_wait_8_1"),
        )

    wait83, snap83 = idle_until(env, is_8_3_control)
    if not is_8_3_control(snap83):
        return PureChainTo83(
            success=False,
            error="8_3_control_timeout",
            leave_8_1=tr81.leave_frame,
            wait_8_2=wait82,
            leave_8_2=tr82.leave_frame,
            wait_8_3=wait83,
            ctrl_wait_8_1=pred.get("ctrl_wait_8_1"),
        )

    # Cumulative through leave-8-2 + wait83 (policy-ish; predecessor uses HL seeds).
    base_keys = (
        "leave_1_1",
        "ctrl_wait_1_2",
        "w4",
        "ctrl_wait_4_1",
        "leave_4_1",
        "ctrl_wait_4_2",
        "w8",
    )
    base = sum(int(pred.get(k) or 0) for k in base_keys)
    wait81 = int(pred.get("ctrl_wait_8_1") or 0)
    total = (
        base
        + wait81
        + int(tr81.leave_frame)
        + wait82
        + int(tr82.leave_frame)
        + wait83
    )

    return PureChainTo83(
        success=True,
        leave_8_1=tr81.leave_frame,
        wait_8_2=wait82,
        leave_8_2=tr82.leave_frame,
        wait_8_3=wait83,
        control_8_3_fp=snap_fingerprint(snap83),
        ctrl_wait_8_1=wait81,
        approx_total_to_8_3_control=total,
    )


def verify_to_8_3_control(
    *,
    fm2_path: Path = PURE_HL_FM2,
    start_8_1: int = HL_8_1_FM2_START,
    start_8_2: int = HL_8_2_FM2_START,
    write_evidence: bool = True,
) -> dict[str, Any]:
    """Fresh Level1_1 rebuild: pure HL → 8-3 control. Evidence only."""
    from retro_harness.env import make_env

    fm2 = parse_fm2(fm2_path).frames
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    chain = build_pure_to_8_3_control(
        env, fm2, start_8_1=start_8_1, start_8_2=start_8_2
    )
    env.close()
    report = {
        **chain.to_dict(),
        "fm2": str(fm2_path),
        "si_8_1": start_8_1,
        "si_8_2": start_8_2,
        "gate_8_3_open": pure_8_3_gate_open(),
        "next": (
            "run probe-83 / search-83 for pure FM2 leave"
            if chain.success
            else "fix predecessor before any 8-3 body work"
        ),
    }
    if write_evidence:
        write_json(PURE_HL_EVIDENCE / "verify_to_8_3_control.json", report)
    return report


def probe_continuous_from_8_2(
    env,
    fm2: list[list[int]],
    start_idx: int,
    *,
    start_lives: int,
    max_play: int = 6000,
) -> dict[str, Any]:
    """From 8-2 control, play pure FM2 continuously (keeps blackout inputs).

    Records 8-3 enter/control, 8-4 leave/control, death, ending.
    """
    body = fm2[start_idx:]
    n = min(len(body), max_play)
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    exits: list[dict[str, Any]] = []
    max_x = 0
    max_x_83 = 0
    death = ending = None
    enter83 = enter84 = ctrl83_at = leave83 = None
    for i in range(n):
        env.step(to_action9(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        w, l = int(snap.world), int(snap.level)
        if 0 < px < 20_000:
            max_x = max(max_x, px)
            # Only count in-level 8-3 progress after control (ignore castle exit x).
            if (
                ctrl83_at is not None
                and w == 7
                and l == 2
                and 0 < px < 4000
            ):
                max_x_83 = max(max_x_83, px)
        key = (w, l)
        if key != last:
            exits.append(
                {
                    "i": i + 1,
                    "from": list(last),
                    "to": list(key),
                    "x": px,
                    "t": int(snap.timer),
                }
            )
            last = key
            if key == (7, 2) and enter83 is None:
                enter83 = i + 1
            if key == (7, 3) and enter84 is None:
                enter84 = i + 1
                leave83 = i + 1
        if is_8_3_control(snap) and ctrl83_at is None:
            ctrl83_at = i + 1
        if reached_ending(ram, start_lives=start_lives):
            ending = i + 1
            break
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = {
                "i": i + 1,
                "w": w + 1,
                "l": l + 1,
                "x": px,
                "t": int(snap.timer),
                "ps": int(snap.player_state),
            }
            break
    return {
        "si": start_idx,
        "ending": ending,
        "death": death,
        "exits": exits[:16],
        "max_x": max_x,
        "max_x_83": max_x_83,
        "enter83": enter83,
        "ctrl83_at": ctrl83_at,
        "leave83": leave83,
        "enter84": enter84,
        "left_83": leave83 is not None,
        "played": ending or (death or {}).get("i") or n,
        "pure_fm2_only": True,
    }


def probe_8_3_continuous(
    *,
    fm2_path: Path = PURE_HL_FM2,
    start_8_1: int = HL_8_1_FM2_START,
    si_8_2: int = HL_8_2_FM2_START,
    max_play: int = 6000,
    write_evidence: bool = True,
) -> dict[str, Any]:
    """Single continuous pure FM2 trial from 8-2 control (diagnostic)."""
    from retro_harness.env import make_env

    fm2 = parse_fm2(fm2_path).frames
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    pred = reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {"success": False, "error": "8_1_control_failed", "pred_stage": pred.get("stage")}

    lives81 = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(env, fm2, start_8_1, start_lives=lives81)
    if not tr81.ok:
        env.close()
        return {"success": False, "error": "8_1_body_failed", "probe": tr81.to_dict()}

    wait82, snap82 = idle_until(env, is_8_2_control)
    if not is_8_2_control(snap82):
        env.close()
        return {"success": False, "error": "8_2_control_timeout", "wait82": wait82}

    lives82 = int(snap82.lives)
    trial = probe_continuous_from_8_2(
        env, fm2, si_8_2, start_lives=lives82, max_play=max_play
    )
    env.close()
    report = {
        "track": TRACK_NAME,
        "mode": "continuous_from_8_2_control",
        "pure_fm2_only": True,
        "si_8_1": start_8_1,
        "si_8_2": si_8_2,
        "leave_8_1": tr81.leave_frame,
        "wait_8_2": wait82,
        "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
        **trial,
        "success_leave_83": bool(trial.get("left_83")),
        "note": (
            "Pure continuous FM2 keeps inter-level blackout inputs. "
            "If left_83 is false, phase is still wrong — do not start 8-4."
        ),
    }
    if write_evidence:
        write_json(PURE_HL_EVIDENCE / "probe_8_3_continuous.json", report)
    return report



def discover_leave_classes_8_2(
    env,
    fm2: list[list[int]],
    st82,
    *,
    lives82: int,
    si82_min: int,
    si82_max: int,
    si82_step: int = 2,
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan pure FM2 8-2 starts for leave → 8-3 control phase classes.

    Returns ``(all_hits, unique_by_leave82_timer)``. Pure HappyLee only.
    """
    hits: list[dict[str, Any]] = []
    for si82 in range(si82_min, si82_max + 1, si82_step):
        set_state(env, st82)
        tr = probe_8_2_from_control(env, fm2, si82, start_lives=lives82)
        if not tr.ok or tr.leave_frame is None:
            continue
        wait83, snap83 = idle_until(env, is_8_3_control)
        if not is_8_3_control(snap83):
            continue
        fp = snap_fingerprint(snap83)
        row = {
            "si82": si82,
            "leave82": int(tr.leave_frame),
            "wait83": int(wait83),
            "control_8_3_fp": fp,
            "timer": int(fp.get("timer", snap83.timer)),
            "player_x": int(fp.get("player_x", snap83.player_x)),
        }
        hits.append(row)
        if log:
            log(
                f"  leave82 si={si82} leave={tr.leave_frame} wait83={wait83} "
                f"t={row['timer']} x={row['player_x']}"
            )

    seen: set[tuple[int, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(hits, key=lambda r: (r["leave82"], r["si82"])):
        key = (int(row["leave82"]), int(row["timer"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return hits, unique


def select_leave_fan(
    unique: list[dict[str, Any]],
    *,
    top_leaves: int = 5,
    default_si82: int | None = None,
) -> list[dict[str, Any]]:
    """Pick phase-diverse leave classes: fastest first, then slower diversity."""
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


def _gated_si_search(
    env,
    fm2: list[list[int]],
    st83,
    *,
    lives83: int,
    si_min: int,
    si_max: int,
    si_step: int,
    lead_idles: tuple[int, ...],
    max_play: int,
    si82: int,
    leave82: int,
    wait83: int,
    ctrl_fp: dict[str, int],
    stop_on_hit: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    """Grid pure FM2 SI + lead at a fixed 8-3 control state. Pure inputs only."""
    hits: list[dict[str, Any]] = []
    best_prog: dict[str, Any] | None = None
    n_trials = 0
    for lead in lead_idles:
        for si in range(si_min, si_max + 1, si_step):
            n_trials += 1
            set_state(env, st83)
            for _ in range(lead):
                env.step(IDLE)
            tr = probe_8_3_from_control(
                env, fm2, si, max_play=max_play, start_lives=lives83
            )
            row: dict[str, Any] = {
                "si": si,
                "lead": lead,
                "si82": si82,
                "leave82": leave82,
                "wait83": wait83,
                "leave": tr.leave_frame,
                "death": tr.death,
                "max_x": tr.max_x,
                "exits": tr.exits,
                "control_8_3_fp": ctrl_fp,
            }
            if tr.ok:
                hits.append(row)
                if log:
                    log(
                        f"  HIT gated si={si} lead={lead} leave={tr.leave_frame} "
                        f"si82={si82} leave82={leave82}"
                    )
                if stop_on_hit:
                    return hits, best_prog, n_trials
            else:
                improved = best_prog is None or (tr.max_x or 0) > (
                    best_prog.get("max_x") or 0
                )
                if improved:
                    best_prog = row
                if log and (improved or n_trials % 25 == 0):
                    log(
                        f"  prog si={si} lead={lead} max_x={tr.max_x} "
                        f"death={tr.death} n={n_trials} si82={si82} t={ctrl_fp.get('timer')}"
                    )
    return hits, best_prog, n_trials


def search_pure_8_3(
    *,
    fm2_path: Path = PURE_HL_FM2,
    start_8_1: int = HL_8_1_FM2_START,
    start_8_2: int = HL_8_2_FM2_START,
    # Gated SI search (after idle to 8-3 control)
    si_min: int = 12950,
    si_max: int = 13650,
    si_step: int = 1,
    lead_idles: tuple[int, ...] = (0, 1, 2),
    max_play: int = 3500,
    # Multi-leave 8-2 fan → re-gate 8-3 (phase classes by leave82/timer)
    multi_leave: bool = False,
    si82_min: int = 10840,
    si82_max: int = 10940,
    si82_step: int = 2,
    top_leaves: int = 5,
    # Also try continuous from nearby 8-2 starts
    cont_si82_min: int | None = None,
    cont_si82_max: int | None = None,
    cont_si82_step: int = 2,
    cont_max_play: int = 5500,
    progress: Callable[[str], None] | None = None,
    write_evidence: bool = True,
    export_on_hit: bool = True,
    stop_on_hit: bool = True,
) -> dict[str, Any]:
    """Search pure HappyLee FM2 for 8-3 → 8-4 leave. Never writes hybrid paths.

    Modes:
    1. **Gated**: HL…8-2 leave → idle → 8-3 control → grid SI + lead idle
    2. **Multi-leave**: fan distinct 8-2 leave phase classes, re-gate each
    3. **Continuous** (optional): from 8-2 control play FM2 without re-gate
    """
    from retro_harness.env import make_env

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    fm2 = parse_fm2(fm2_path).frames
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    log("pure_hl: build to 8-1 control…")
    pred = reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {"success": False, "error": "8_1_control_failed"}

    lives81 = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(env, fm2, start_8_1, start_lives=lives81)
    if not tr81.ok:
        env.close()
        return {"success": False, "error": "8_1_body_failed", "probe": tr81.to_dict()}

    wait82, snap82 = idle_until(env, is_8_2_control)
    if not is_8_2_control(snap82):
        env.close()
        return {"success": False, "error": "8_2_control_timeout"}
    lives82 = int(snap82.lives)
    st82 = get_state(env)
    log(f"pure_hl: 8-2 control wait82={wait82} leave81={tr81.leave_frame}")

    report: dict[str, Any] = {
        "track": TRACK_NAME,
        "pure_fm2_only": True,
        "no_hybrid": True,
        "no_flamexx": True,
        "no_natural_82": True,
        "no_skill_macros": True,
        "si_8_1": start_8_1,
        "si_8_2_default": start_8_2,
        "leave_8_1": tr81.leave_frame,
        "wait_8_2": wait82,
        "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
        "multi_leave": multi_leave,
        "leave_classes_82": [],
        "unique_leave_classes": [],
        "fan_classes": [],
        "gated_hits": [],
        "gated_best_progress": None,
        "cont_hits": [],
        "cont_best": None,
        "exported": None,
        "gate_written": False,
    }

    if cont_si82_min is not None and cont_si82_max is not None:
        cmin, cmax, cstep = cont_si82_min, cont_si82_max, cont_si82_step
    elif multi_leave:
        cmin, cmax, cstep = si82_min, si82_max, si82_step
    else:
        cmin = cmax = cstep = None  # type: ignore[assignment]

    cont_best = None
    if cmin is not None and cmax is not None:
        log(f"pure_hl: continuous scan si82={cmin}..{cmax} step={cstep}")
        for si in range(int(cmin), int(cmax) + 1, int(cstep)):
            set_state(env, st82)
            trial = probe_continuous_from_8_2(
                env, fm2, si, start_lives=lives82, max_play=cont_max_play
            )
            if trial.get("left_83") or (trial.get("max_x_83") or 0) > 800:
                report["cont_hits"].append(trial)
                log(
                    f"  cont si82={si} left83={trial.get('left_83')} "
                    f"max_x83={trial.get('max_x_83')} death={trial.get('death')}"
                )
            if cont_best is None or (
                (trial.get("left_83") and not cont_best.get("left_83"))
                or (
                    trial.get("left_83") == cont_best.get("left_83")
                    and (trial.get("max_x_83") or 0) > (cont_best.get("max_x_83") or 0)
                )
                or (
                    not cont_best.get("left_83")
                    and (trial.get("max_x_83") or 0) > (cont_best.get("max_x_83") or 0)
                )
            ):
                cont_best = trial
            if trial.get("left_83") and stop_on_hit:
                log(f"pure_hl: CONTINUOUS LEAVE 8-3 at si82={si}")
                break
        report["cont_best"] = cont_best
        if cont_best and cont_best.get("left_83"):
            log(f"pure_hl: CONTINUOUS LEAVE 8-3 at si82={cont_best['si']}")

    if multi_leave:
        log(
            f"pure_hl: multi-leave discover si82={si82_min}..{si82_max} "
            f"step={si82_step}"
        )
        all_leaves, unique = discover_leave_classes_8_2(
            env,
            fm2,
            st82,
            lives82=lives82,
            si82_min=si82_min,
            si82_max=si82_max,
            si82_step=si82_step,
            log=log,
        )
        fan = select_leave_fan(
            unique, top_leaves=top_leaves, default_si82=start_8_2
        )
        report["leave_classes_82"] = all_leaves
        report["unique_leave_classes"] = unique
        report["fan_classes"] = fan
        log(
            f"pure_hl: leave hits={len(all_leaves)} unique={len(unique)} "
            f"fan={len(fan)}"
        )
        if not fan:
            env.close()
            report["error"] = "no_8_2_leave_classes"
            if write_evidence:
                write_json(PURE_HL_EVIDENCE / "search_8_3.json", report)
            return report
    else:
        fan = [
            {
                "si82": start_8_2,
                "leave82": None,
                "wait83": None,
                "control_8_3_fp": None,
                "timer": None,
            }
        ]
        report["fan_classes"] = fan

    hits: list[dict[str, Any]] = []
    best_prog: dict[str, Any] | None = None
    n_trials = 0
    found = bool(cont_best and cont_best.get("left_83"))

    for leave in fan:
        if found and stop_on_hit:
            break
        si82 = int(leave["si82"])
        log(
            f"=== fan si82={si82} leave82={leave.get('leave82')} "
            f"t={leave.get('timer')} ==="
        )
        set_state(env, st82)
        tr82 = probe_8_2_from_control(env, fm2, si82, start_lives=lives82)
        if not tr82.ok or tr82.leave_frame is None:
            log(f"  8-2 body failed si82={si82}")
            continue
        wait83, snap83 = idle_until(env, is_8_3_control)
        if not is_8_3_control(snap83):
            log(f"  8-3 control timeout after leave82={tr82.leave_frame}")
            continue
        lives83 = int(snap83.lives)
        st83 = get_state(env)
        ctrl_fp = snap_fingerprint(snap83)
        leave82 = int(tr82.leave_frame)
        if report.get("control_8_3_fp") is None:
            report["wait_8_3"] = wait83
            report["leave_8_2"] = leave82
            report["control_8_3_fp"] = ctrl_fp
            report["si_8_2_used"] = si82
        log(
            f"  8-3 control wait83={wait83} leave82={leave82} "
            f"t={snap83.timer} x={snap83.player_x}"
        )

        class_hits, class_best, class_n = _gated_si_search(
            env,
            fm2,
            st83,
            lives83=lives83,
            si_min=si_min,
            si_max=si_max,
            si_step=si_step,
            lead_idles=lead_idles,
            max_play=max_play,
            si82=si82,
            leave82=leave82,
            wait83=wait83,
            ctrl_fp=ctrl_fp,
            stop_on_hit=stop_on_hit,
            log=log,
        )
        n_trials += class_n
        hits.extend(class_hits)
        if class_best is not None and (
            best_prog is None
            or (class_best.get("max_x") or 0) > (best_prog.get("max_x") or 0)
        ):
            best_prog = class_best

        if class_hits:
            found = True
            hit = class_hits[0]
            if export_on_hit and report.get("exported") is None:
                exp = export_pure_8_3(
                    start_idx=int(hit["si"]),
                    leave_frames=int(hit["leave"]),
                    lead_idle=int(hit["lead"]),
                    ctrl_fp=ctrl_fp,
                    leave_8_2=leave82,
                    wait_8_3=wait83,
                    start_8_2=si82,
                    start_8_1=start_8_1,
                    verify=True,
                    verify_trials=2,
                )
                report["exported"] = exp
                report["gate_written"] = bool(exp.get("success"))
                report["wait_8_3"] = wait83
                report["leave_8_2"] = leave82
                report["control_8_3_fp"] = ctrl_fp
                report["si_8_2_used"] = si82
            if stop_on_hit:
                break

    env.close()
    report["gated_hits"] = hits
    report["gated_best_progress"] = best_prog
    report["n_gated_trials"] = n_trials
    report["n_gated_hits"] = len(hits)
    report["success"] = bool(hits) or bool(
        report.get("cont_best") and report["cont_best"].get("left_83")
    )
    if (
        report["success"]
        and not hits
        and report.get("cont_best")
        and report["cont_best"].get("left_83")
        and export_on_hit
        and report.get("exported") is None
    ):
        report["continuous_leave_only"] = True
        report["next"] = (
            "continuous pure FM2 left 8-3 — export gated body from control "
            "offset or promote continuous chain carefully; re-verify 2/2"
        )
    elif report["success"] and report.get("gate_written"):
        report["next"] = (
            "pure 8-3 leave verified — 8-4 search now allowed (pure HL only)"
        )
    elif report["success"]:
        report["next"] = (
            "gated leave hit — confirm gate_8_3_leave.json "
            "verified_leave_8_4_control=true (2/2)"
        )
    else:
        report["next"] = (
            "still desynced — expand multi-leave SI82 / SI83 / leads / continuous; "
            "do NOT start pure 8-4"
        )

    if write_evidence:
        write_json(PURE_HL_EVIDENCE / "search_8_3.json", report)
    return report


def export_pure_8_3(
    *,
    start_idx: int,
    leave_frames: int,
    lead_idle: int = 0,
    ctrl_fp: dict[str, int] | None = None,
    leave_8_2: int | None = None,
    wait_8_3: int | None = None,
    start_8_2: int = HL_8_2_FM2_START,
    start_8_1: int = HL_8_1_FM2_START,
    fm2_path: Path = PURE_HL_FM2,
    verify: bool = True,
    verify_trials: int = 2,
) -> dict[str, Any]:
    """Write pure 8-3 seed + open gate. Paths only under models/pure_hl/.

    ``start_8_2`` must match the multi-leave class that produced the hit so
    verify rebuilds the same 8-3 control phase (not the canonical default only).
    """
    ensure_pure_dirs()
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[start_idx : start_idx + leave_frames]]
    if not frames:
        raise ValueError("empty pure 8-3 body")

    meta = {
        "track": TRACK_NAME,
        "pure_fm2_only": True,
        "no_hybrid": True,
        "no_natural_82": True,
        "no_flamexx": True,
        "no_skill_macros": True,
        "start_state": "8-3_control_after_hl_8_2_pure",
        "target": "8_4_control",
        "body_frames": leave_frames,
        "leave_frames": leave_frames,
        "fm2": str(fm2_path),
        "fm2_start_index": start_idx,
        "lead_idle_before_body": lead_idle,
        "leave_8_2": leave_8_2,
        "wait_8_3": wait_8_3,
        "si_8_2": start_8_2,
        "si_8_1": start_8_1,
        "control_8_3_fp": ctrl_fp,
        "verified_leave_8_4_control": not verify,
        "note": (
            "Pure HappyLee 8-3 body. Do not sanitize L+R. "
            "Gate opens only after leave to 8-4 control."
        ),
    }
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id="smb_8_3_pure_hl",
        source="HappyLee warps #1715M pure track (no hybrid)",
        extra=meta,
    )
    out = assert_pure_write_path(PURE_8_3_SEED)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    gate = {
        "track": TRACK_NAME,
        "verified_leave_8_4_control": True if not verify else False,
        "pure_fm2_only": True,
        "fm2_start_index": start_idx,
        "leave_frames": leave_frames,
        "lead_idle_before_body": lead_idle,
        "si_8_2": start_8_2,
        "si_8_1": start_8_1,
        "seed": str(out),
        "control_8_3_fp": ctrl_fp,
        "leave_8_2": leave_8_2,
        "wait_8_3": wait_8_3,
        "8_4_blocked_until_gate": False if not verify else True,
        "verify_trials": [],
    }

    if verify:
        from retro_harness.env import make_env

        n_ok = 0
        for trial_i in range(max(1, verify_trials)):
            env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
            env.reset()
            chain = build_pure_to_8_3_control(
                env, fm2, start_8_1=start_8_1, start_8_2=start_8_2
            )
            if not chain.success:
                env.close()
                gate["verify_trials"].append(
                    {"trial": trial_i, "ok": False, "error": chain.error}
                )
                continue

            for _ in range(lead_idle):
                env.step(IDLE)
            lives = int(read_snapshot(env.get_ram(), 0).lives)
            tr = probe_8_3_from_control(
                env, fm2, start_idx, max_play=leave_frames + 50, start_lives=lives
            )
            ok_leave = tr.ok
            ok_84 = False
            wait84 = None
            fp84 = None
            if ok_leave:
                wait84, snap84 = idle_until(env, is_8_4_control, max_wait=400)
                ok_84 = is_8_4_control(snap84)
                fp84 = snap_fingerprint(snap84) if ok_84 else None
            env.close()
            trial_ok = bool(ok_leave and ok_84)
            if trial_ok:
                n_ok += 1
            gate["verify_trials"].append(
                {
                    "trial": trial_i,
                    "ok": trial_ok,
                    "leave": tr.leave_frame,
                    "death": tr.death,
                    "max_x": tr.max_x,
                    "wait_8_4": wait84,
                    "control_8_4_fp": fp84,
                }
            )
            gate["probe_leave"] = tr.leave_frame
            gate["probe_death"] = tr.death
            gate["probe_max_x"] = tr.max_x
            if ok_84:
                gate["wait_8_4"] = wait84
                gate["control_8_4_fp"] = fp84

        gate["verified_leave_8_4_control"] = n_ok >= max(1, verify_trials)
        gate["verify_n_ok"] = n_ok
        gate["verify_n_trials"] = max(1, verify_trials)
        gate["8_4_blocked_until_gate"] = not gate["verified_leave_8_4_control"]

    write_json(PURE_8_3_GATE, gate)
    write_json(PURE_HL_EVIDENCE / "export_8_3.json", {"seed": str(out), "gate": gate})
    return {
        "success": bool(gate.get("verified_leave_8_4_control")),
        "seed": str(out),
        "gate": gate,
        "path": str(out),
    }


def refuse_8_4_until_gate() -> dict[str, Any]:
    """Hard block for pure 8-4 work until 8-3 gate opens."""
    return {
        "allowed": pure_8_3_gate_open(),
        "gate_file": str(PURE_8_3_GATE),
        "message": (
            "pure 8-4 search allowed"
            if pure_8_3_gate_open()
            else "BLOCKED: pure 8-3 leave not verified — fix 8-3 sync first"
        ),
    }
