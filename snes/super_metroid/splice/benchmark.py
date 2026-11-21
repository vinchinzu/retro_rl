"""Standard KPDR Any% segment benchmarking against reference splits.

Extracts exact frame-level split timestamps from any Clean master tape
(``master_tape_*_clean.{json,npz}``), compares them against the standard
23-split KPDR Any% reference (``0:56:28.88`` Late Ice / ``57:11.75`` PB) and
the ``< 55:00`` (198,000-frame) end goal, breaks down policy/skill coverage
per segment, ranks the highest-leverage time-loss hotspots, and records
longitudinal progress in ``recordings/speed_benchmarks/history.json``.

Route-order flexibility:
  The reference splits follow KPDR Any% Late Ice (Hi-Jump after Varia, Ice
  after Plasma). Runs with Early Hi-Jump or Early Ice still align every
  split by exact milestone identity, flag route-order shifts explicitly, and
  report both reference-order cumulative comparisons and actual-run-order
  segment durations so route policy changes stay apples-to-apples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR

FPS_CONVENTION = 60.0
SUB55_BUDGET_FRAMES = 198_000  # 55:00.00 @ 60 fps (Ceres control -> Ship)
BENCHMARK_DIR = GAME_DIR / "recordings" / "speed_benchmarks"
HISTORY_PATH = BENCHMARK_DIR / "history.json"
DEFAULT_CHECKPOINT_DIR = SCRATCH_STATE_DIR / "clean_checkpoints"

# Item & beam masks ($09A4 collected items, $09A8 collected beams).
_ITEM_VARIA = 0x0001
_ITEM_MORPH = 0x0004
_ITEM_GRAVITY = 0x0020
_ITEM_HI_JUMP = 0x0100
_ITEM_SPACE_JUMP = 0x0200
_ITEM_BOMBS = 0x1000
_ITEM_SPEED = 0x2000

_BEAM_WAVE = 0x0001
_BEAM_ICE = 0x0002
_BEAM_PLASMA = 0x0008
_BEAM_CHARGE = 0x1000
_HYPER_BEAM_EQUIPPED = 0x1009


@dataclass(frozen=True)
class ReferenceSplit:
    """One standard KPDR Any% split from the target reference image."""

    id: str
    label: str
    ref_pb_seconds: float      # "BEST" column in reference image (57:11.75 PB)
    ref_run_seconds: float     # "TIME" column in reference image (56:28.88 run)
    movable: bool = False      # e.g. Ice Beam / Hi-Jump order varies by route


def parse_rta_seconds(text: str) -> float:
    """Parse ``MM:SS.cc`` or ``H:MM:SS.cc`` into seconds."""
    parts = text.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"invalid RTA timestamp: {text!r}")


def format_rta(seconds: float | None, *, signed: bool = False) -> str:
    """Format seconds as ``H:MM:SS.cc`` (or ``MM:SS.cc`` under 1 hour)."""
    if seconds is None:
        return "—"
    sign = ""
    if seconds < 0:
        sign = "-"
        seconds = -seconds
    elif signed:
        sign = "+"
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h > 0:
        return f"{sign}{h}:{m:02d}:{s:02d}.{cs:02d}"
    return f"{sign}{m:02d}:{s:02d}.{cs:02d}"


def frames_to_seconds(frames: int | None) -> float | None:
    if frames is None:
        return None
    return frames / FPS_CONVENTION


# Standard 23-split KPDR Any% Late Ice reference (from user's target splits image).
KPDR_REFERENCE_SPLITS: tuple[ReferenceSplit, ...] = (
    ReferenceSplit("ceres_station", "Ceres Station", parse_rta_seconds("01:31.01"), parse_rta_seconds("01:31.31")),
    ReferenceSplit("morph_ball", "Morph Ball", parse_rta_seconds("03:09.70"), parse_rta_seconds("03:14.13")),
    ReferenceSplit("first_missiles", "First Missiles", parse_rta_seconds("03:32.05"), parse_rta_seconds("03:37.43")),
    ReferenceSplit("bomb", "Bomb", parse_rta_seconds("05:15.55"), parse_rta_seconds("05:20.16")),
    ReferenceSplit("first_super", "First Super", parse_rta_seconds("07:26.53"), parse_rta_seconds("07:42.70")),
    ReferenceSplit("charge_beam", "Charge Beam", parse_rta_seconds("08:23.64"), parse_rta_seconds("08:40.72")),
    ReferenceSplit("kraid", "Kraid", parse_rta_seconds("11:01.95"), parse_rta_seconds("11:49.74")),
    ReferenceSplit("varia_suit", "Varia Suit", parse_rta_seconds("11:10.23"), parse_rta_seconds("11:55.53")),
    ReferenceSplit("hi_jump_boots", "Hi-Jump Boots", parse_rta_seconds("13:35.22"), parse_rta_seconds("13:57.35"), movable=True),
    ReferenceSplit("speed_booster", "Speed Booster", parse_rta_seconds("15:37.83"), parse_rta_seconds("15:53.96")),
    ReferenceSplit("wave_beam", "Wave Beam", parse_rta_seconds("16:50.65"), parse_rta_seconds("16:58.57")),
    ReferenceSplit("first_power_bomb", "First Power Bomb", parse_rta_seconds("19:36.53"), parse_rta_seconds("19:55.84")),
    ReferenceSplit("phantoon", "Phantoon", parse_rta_seconds("22:50.49"), parse_rta_seconds("22:58.00")),
    ReferenceSplit("gravity_suit", "Gravity Suit", parse_rta_seconds("25:24.66"), parse_rta_seconds("25:37.25")),
    ReferenceSplit("draygon", "Draygon", parse_rta_seconds("32:32.07"), parse_rta_seconds("32:06.96")),
    ReferenceSplit("space_jump", "Space Jump", parse_rta_seconds("32:39.50"), parse_rta_seconds("32:16.90")),
    ReferenceSplit("plasma_beam", "Plasma Beam", parse_rta_seconds("34:31.35"), parse_rta_seconds("34:17.08")),
    ReferenceSplit("ice_beam", "Ice Beam", parse_rta_seconds("36:52.19"), parse_rta_seconds("36:31.28"), movable=True),
    ReferenceSplit("ridley", "Ridley", parse_rta_seconds("42:30.10"), parse_rta_seconds("42:10.16")),
    ReferenceSplit("g4", "G4", parse_rta_seconds("48:02.93"), parse_rta_seconds("47:40.78")),
    ReferenceSplit("mother_brain_1", "Mother Brain 1", parse_rta_seconds("52:27.36"), parse_rta_seconds("51:59.42")),
    ReferenceSplit("mother_brain_2", "Mother Brain 2", parse_rta_seconds("54:16.79"), parse_rta_seconds("53:34.65")),
    ReferenceSplit("ship", "Ship", parse_rta_seconds("57:11.75"), parse_rta_seconds("56:28.88")),
)

REFERENCE_BY_ID: dict[str, ReferenceSplit] = {s.id: s for s in KPDR_REFERENCE_SPLITS}


def splits_sidecar_path(stem: Path) -> Path:
    return stem.parent / f"{stem.name}_splits.json"


def checkpoint_dir_for(tape_sha256: str) -> Path:
    return DEFAULT_CHECKPOINT_DIR / tape_sha256


def save_checkpoint(path: Path, state: bytes) -> None:
    from retro_harness.env import write_state_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete_on_close=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.close()
        write_state_bytes(temp_path, state)
        temp_path.replace(path)


def _u16(ram: Any, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def scan_master_splits(
    stem: Path,
    *,
    save_checkpoints: bool = False,
    overwrite_checkpoints: bool = False,
    checkpoint_dir: Path | None = None,
    write_sidecar: bool = True,
) -> dict[str, Any]:
    """Replay a Clean master tape from power-on and extract exact RAM splits.

    Enforces loadless, write-free replay from ``retro.State.NONE``. Also
    optionally persists ``rowNNN_leave.state`` checkpoints for every row so
    any segment can be A/B benchmarked locally without re-running from boot.
    """
    import numpy as np
    from super_metroid.dev.common import make_dev_env
    from super_metroid.ram import (
        ADDR_BOSS_BITS,
        ADDR_COLLECTED_BEAMS,
        ADDR_COLLECTED_ITEMS,
        ADDR_DOOR_TRANSITION,
        ADDR_ENEMY0_HP,
        ADDR_EQUIPPED_BEAMS,
        ADDR_GAME_STATE,
        ADDR_MAX_MISSILES,
        ADDR_MAX_POWER_BOMBS,
        ADDR_MAX_SUPER_MISSILES,
        ADDR_ROOM_ID,
        GS_CERES_LEAVE,
        GS_ORDINARY,
        credits_complete,
        read_wram_u8,
    )
    from super_metroid.splice.assemble import _guard_writes

    report = json.loads(stem.with_suffix(".json").read_text())
    actions = np.load(stem.with_suffix(".npz"))["actions"]
    tape_sha256 = hashlib.sha256(actions.tobytes()).hexdigest()
    if tape_sha256 != report["tape_sha256"]:
        raise ValueError(f"tape digest mismatch for {stem}")

    ckpt_dir = checkpoint_dir or checkpoint_dir_for(tape_sha256)
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = make_dev_env()
    env.reset()
    restore_writes = _guard_writes(env)

    rows = report["rows"]

    splits_abs: dict[str, int] = {}
    split_rows: dict[str, int] = {}
    ceres_control_frame: int | None = None
    mb1_seen = False
    mb2_seen = False
    checkpoints_written = 0

    def record(split_id: str, frame: int, row_num: int) -> None:
        if split_id not in splits_abs:
            splits_abs[split_id] = frame
            split_rows[split_id] = row_num

    try:
        frame = 0
        for r in rows:
            row_num = int(r["row"])
            count = int(r["frames"])
            chunk = actions[frame : frame + count]
            for action in chunk:
                env.step(action)
                frame += 1
                ram = env.get_ram()
                room_id = _u16(ram, ADDR_ROOM_ID)
                gs = _u16(ram, ADDR_GAME_STATE)
                door = _u16(ram, ADDR_DOOR_TRANSITION)

                if ceres_control_frame is None and room_id == 0xDF45 and gs == GS_ORDINARY and door == 0:
                    ceres_control_frame = frame

                if ceres_control_frame is not None:
                    # Ceres Station split: Ceres elevator/escape cutscene begins.
                    if "ceres_station" not in splits_abs and (gs in GS_CERES_LEAVE or room_id == 0x91F8):
                        record("ceres_station", frame, row_num)

                    items = _u16(ram, ADDR_COLLECTED_ITEMS)
                    beams = _u16(ram, ADDR_COLLECTED_BEAMS)
                    if items & _ITEM_MORPH:
                        record("morph_ball", frame, row_num)
                    if _u16(ram, ADDR_MAX_MISSILES) > 0:
                        record("first_missiles", frame, row_num)
                    if items & _ITEM_BOMBS:
                        record("bomb", frame, row_num)
                    if _u16(ram, ADDR_MAX_SUPER_MISSILES) > 0:
                        record("first_super", frame, row_num)
                    if beams & _BEAM_CHARGE:
                        record("charge_beam", frame, row_num)
                    if items & _ITEM_VARIA:
                        record("varia_suit", frame, row_num)
                    if items & _ITEM_HI_JUMP:
                        record("hi_jump_boots", frame, row_num)
                    if items & _ITEM_SPEED:
                        record("speed_booster", frame, row_num)
                    if beams & _BEAM_WAVE:
                        record("wave_beam", frame, row_num)
                    if _u16(ram, ADDR_MAX_POWER_BOMBS) > 0:
                        record("first_power_bomb", frame, row_num)
                    if items & _ITEM_GRAVITY:
                        record("gravity_suit", frame, row_num)
                    if items & _ITEM_SPACE_JUMP:
                        record("space_jump", frame, row_num)
                    if beams & _BEAM_PLASMA:
                        record("plasma_beam", frame, row_num)
                    if beams & _BEAM_ICE:
                        record("ice_beam", frame, row_num)

                    # Boss flags live in high WRAM ($7E:D828..); only peek in boss rooms.
                    if room_id == 0xA59F and "kraid" not in splits_abs:
                        if read_wram_u8(env, ADDR_BOSS_BITS + 1) & 0x01:
                            record("kraid", frame, row_num)
                    elif room_id == 0xCD13 and "phantoon" not in splits_abs:
                        if read_wram_u8(env, ADDR_BOSS_BITS + 3) & 0x01:
                            record("phantoon", frame, row_num)
                    elif room_id == 0xDA60 and "draygon" not in splits_abs:
                        if read_wram_u8(env, ADDR_BOSS_BITS + 4) & 0x01:
                            record("draygon", frame, row_num)
                    elif room_id == 0xB32E and "ridley" not in splits_abs:
                        if read_wram_u8(env, ADDR_BOSS_BITS + 2) & 0x01:
                            record("ridley", frame, row_num)
                    elif room_id == 0xDAAE and "g4" not in splits_abs:
                        # G4 split: entering Tourian First Room after Statues Room.
                        record("g4", frame, row_num)
                    elif room_id == 0xDD58 and gs == GS_ORDINARY and door == 0:
                        # Slot 1 HP ($0FCC) is Mother Brain's head:
                        # MB1 (3000 -> 0), MB2 (18000 -> 0), MB3 (36000 -> 0).
                        hp1 = _u16(ram, ADDR_ENEMY0_HP + 0x40)
                        if not mb1_seen and hp1 == 3000:
                            mb1_seen = True
                        elif mb1_seen and "mother_brain_1" not in splits_abs and hp1 == 0:
                            record("mother_brain_1", frame, row_num)
                        if "mother_brain_1" in splits_abs and not mb2_seen and hp1 == 18000:
                            mb2_seen = True
                        elif mb2_seen and "mother_brain_2" not in splits_abs and hp1 == 0:
                            record("mother_brain_2", frame, row_num)
                    elif room_id == 0x91F8 and "mother_brain_2" in splits_abs and "ship" not in splits_abs:
                        if gs in (38, 39):
                            record("ship", frame, row_num)

            if save_checkpoints:
                ckpt_path = ckpt_dir / f"row{row_num:03d}_leave.state"
                if overwrite_checkpoints or not ckpt_path.exists():
                    save_checkpoint(ckpt_path, env.em.get_state())
                    checkpoints_written += 1

        # Credits tail (if present).
        for action in actions[frame:]:
            env.step(action)
            frame += 1
            if "ship" not in splits_abs:
                gs = _u16(env.get_ram(), ADDR_GAME_STATE)
                if gs in (38, 39):
                    record("ship", frame, rows[-1]["row"] if rows else 0)

        credits_done = bool(credits_complete(env))
    finally:
        restore_writes()
        env.close()

    if ceres_control_frame is None:
        ceres_control_frame = int(rows[0]["frames"]) if rows else 0

    rta_splits: dict[str, dict[str, Any]] = {}
    for split_id, abs_frame in splits_abs.items():
        rta_frames = max(0, abs_frame - ceres_control_frame)
        rta_splits[split_id] = {
            "abs_frame": abs_frame,
            "rta_frames": rta_frames,
            "rta_seconds": round(rta_frames / FPS_CONVENTION, 3),
            "rta": format_rta(rta_frames / FPS_CONVENTION),
            "row": split_rows.get(split_id),
        }

    ship_abs = splits_abs.get("ship")
    rta_ship_frames = (ship_abs - ceres_control_frame) if ship_abs is not None else None
    result = {
        "schema_version": 1,
        "kind": "kpdr_splits_scan",
        "stem": str(stem),
        "tape_sha256": tape_sha256,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_frames": len(actions),
        "ceres_control_frame": ceres_control_frame,
        "ship_abs_frame": ship_abs,
        "rta_ship_frames": rta_ship_frames,
        "rta_ship_seconds": round(rta_ship_frames / FPS_CONVENTION, 3) if rta_ship_frames is not None else None,
        "rta_ship": format_rta(frames_to_seconds(rta_ship_frames)),
        "credits_complete": credits_done,
        "credits_complete_rta_frames": len(actions) - ceres_control_frame,
        "checkpoints_written": checkpoints_written,
        "splits": rta_splits,
    }
    if write_sidecar:
        splits_sidecar_path(stem).write_text(json.dumps(result, indent=2) + "\n")
    return result


def estimate_splits_from_ledger(report: Mapping[str, Any]) -> dict[str, Any]:
    """Instant offline split estimate from a master JSON ledger (no ROM boot).

    Used as a fast fallback when no ``*_splits.json`` RAM scan exists yet.
    """
    rows = report.get("rows") or []
    ceres_start = int(rows[0]["frames"]) if rows else 0
    cum = 0
    splits: dict[str, dict[str, Any]] = {}

    def mark(split_id: str, abs_f: int, row_num: int) -> None:
        if split_id not in splits:
            rta_f = max(0, abs_f - ceres_start)
            splits[split_id] = {
                "abs_frame": abs_f,
                "rta_frames": rta_f,
                "rta_seconds": round(rta_f / FPS_CONVENTION, 3),
                "rta": format_rta(rta_f / FPS_CONVENTION),
                "row": row_num,
                "estimated": True,
            }

    room_triggers = {
        "0x9E9F": ("morph_ball", lambda r: r.get("to_room") is None),
        "0xA107": ("first_missiles", lambda r: True),
        "0x9804": ("bomb", lambda r: r.get("to_room") is None),
        "0x9B5B": ("first_super", lambda r: r.get("to_room") is None),
        "0x9D19": ("charge_beam", lambda r: r.get("to_room") == "0x9E52"),
        "0xA59F": ("kraid", lambda r: r.get("to_room") == "0xA6E2"),
        "0xA6E2": ("varia_suit", lambda r: r.get("to_room") is None),
        "0xA9E5": ("hi_jump_boots", lambda r: True),
        "0xAD1B": ("speed_booster", lambda r: True),
        "0xADDE": ("wave_beam", lambda r: r.get("to_room") is None),
        "0xA890": ("ice_beam", lambda r: r.get("to_room") is None),
        "0xA3AE": ("first_power_bomb", lambda r: r.get("to_room") is None),
        "0xCD13": ("phantoon", lambda r: r.get("to_room") == "0xCC6F"),
        "0xCE40": ("gravity_suit", lambda r: r.get("to_room") is None),
        "0xDA60": ("draygon", lambda r: r.get("to_room") == "0xD9AA"),
        "0xD9AA": ("space_jump", lambda r: r.get("to_room") is None),
        "0xD2AA": ("plasma_beam", lambda r: r.get("to_room") is None),
        "0xB32E": ("ridley", lambda r: r.get("to_room") == "0xB698"),
        "0xA66A": ("g4", lambda r: r.get("to_room") == "0xDAAE"),
        "0xDD58": ("mother_brain_2", lambda r: r.get("to_room") is None),
    }
    for r in rows:
        cum += int(r["frames"])
        rm = r.get("room", "")
        if rm == "0xDF8D" and r.get("to_room") == "0xDF45":
            mark("ceres_station", cum, r["row"])
        trig = room_triggers.get(rm)
        if trig and trig[1](r):
            mark(trig[0], cum, r["row"])
        if r.get("goal_phase") == "ending_or_credits":
            mark("ship", cum, r["row"])

    ship_f = splits["ship"]["rta_frames"] if "ship" in splits else None
    return {
        "schema_version": 1,
        "kind": "kpdr_splits_estimated",
        "tape_sha256": report.get("tape_sha256"),
        "total_frames": int(report.get("total_frames", cum)),
        "ceres_control_frame": ceres_start,
        "rta_ship_frames": ship_f,
        "rta_ship_seconds": round(ship_f / FPS_CONVENTION, 3) if ship_f is not None else None,
        "rta_ship": format_rta(frames_to_seconds(ship_f)),
        "credits_complete": bool(report.get("credits_complete", False)),
        "splits": splits,
    }


def _row_kind_counts(rows: Sequence[Mapping[str, Any]], coverage_by_row: Mapping[int, str] | None) -> dict[str, Any]:
    counts = {"controller": 0, "searched_buttons": 0, "human_retimed": 0, "human_verbatim": 0}
    for r in rows:
        n = int(r["frames"])
        kind = (coverage_by_row or {}).get(int(r["row"]))
        if kind is None:
            method = str(r.get("method", ""))
            if method == "room-controller" or r.get("controller_id"):
                kind = "controller"
            elif method == "tape+idle:0":
                kind = "human_verbatim"
            elif method.startswith(("tape+idle", "tape-trim")):
                kind = "human_retimed"
            else:
                kind = "searched_buttons"
        counts[kind] += n
    total = sum(counts.values()) or 1
    return {
        "total_frames": sum(counts.values()),
        "skill_frames": counts["controller"],
        "skill_pct": round(100.0 * counts["controller"] / total, 1),
        "counts": counts,
    }


def _per_row_coverage_kinds(stem: Path, board: Path | None, rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """Compute button-diff provenance kind for each row (matches splice chain)."""
    import numpy as np
    from super_metroid.splice.assemble import _chain_body, _chain_rows

    out: dict[int, str] = {}
    tape_path = stem.with_suffix(".npz")
    if not tape_path.is_file() or board is None or not board.is_file():
        return out
    try:
        tape = np.load(tape_path)["actions"]
        cache: dict[str, Any] = {}
        bodies: dict[int, Any] = {}
        for hop in _chain_rows(board):
            if hop.get("tape") and not hop.get("controller_id"):
                bodies[hop["row"]] = _chain_body(hop, cache)[0]
    except (OSError, ValueError, KeyError):
        return out
    offset = 0
    for r in rows:
        n = int(r["frames"])
        row_num = int(r["row"])
        method = str(r.get("method", ""))
        got = tape[offset : offset + n]
        body = bodies.get(row_num)
        offset += n
        if method == "room-controller" or r.get("controller_id"):
            out[row_num] = "controller"
        elif body is not None:
            trim = int(method.split(":")[1]) if method.startswith("tape-trim:") else 0
            delay = int(method.split(":")[1]) if method.startswith("tape+idle:") else 0
            keep = body[trim:]
            same = (
                len(got) >= delay + len(keep)
                and not got[:delay].any()
                and np.array_equal(got[delay : delay + len(keep)], keep)
            )
            if same:
                out[row_num] = "human_verbatim" if not (delay or trim) else "human_retimed"
            else:
                out[row_num] = "searched_buttons"
        else:
            out[row_num] = "searched_buttons"
    return out


def rank_hotspots(rows: Sequence[Mapping[str, Any]], *, top_n: int = 15) -> list[dict[str, Any]]:
    """Rank the biggest time-loss single rooms and obvious detour clusters."""
    hotspots: list[dict[str, Any]] = []
    for r in rows:
        if r.get("room") == "0x0000":
            continue
        frames = int(r["frames"])
        if frames < 1500:
            continue
        hotspots.append({
            "row": int(r["row"]),
            "room": r["room"],
            "to_room": r.get("to_room"),
            "name": r.get("name", "?"),
            "frames": frames,
            "seconds": round(frames / FPS_CONVENTION, 2),
            "method": r.get("method", ""),
            "energy_end": r.get("energy"),
        })
    hotspots.sort(key=lambda h: h["frames"], reverse=True)
    return hotspots[:top_n]


def build_benchmark_report(
    stem: Path,
    *,
    board: Path | None = None,
    splits_data: Mapping[str, Any] | None = None,
    top_hotspots: int = 15,
) -> dict[str, Any]:
    """Compare a master tape against the standard 0:56:28.88 reference & <55:00 goal."""
    report = json.loads(stem.with_suffix(".json").read_text())
    rows = report.get("rows") or []
    board_path = board or (Path(report["board"]) if report.get("board") else None)
    if board_path is not None and not board_path.is_absolute():
        candidate = GAME_DIR.parents[1] / board_path
        if candidate.is_file():
            board_path = candidate

    if splits_data is None:
        sidecar = splits_sidecar_path(stem)
        if sidecar.is_file():
            loaded = json.loads(sidecar.read_text())
            if loaded.get("tape_sha256") == report.get("tape_sha256"):
                splits_data = loaded
    if splits_data is None:
        splits_data = estimate_splits_from_ledger(report)

    our_splits: Mapping[str, Mapping[str, Any]] = splits_data.get("splits", {})
    coverage_by_row = _per_row_coverage_kinds(stem, board_path, rows)

    # Detect actual order of splits in our run.
    present_in_order = sorted(
        ((sid, info["rta_seconds"]) for sid, info in our_splits.items() if sid in REFERENCE_BY_ID),
        key=lambda item: item[1],
    )
    actual_order = [sid for sid, _ in present_in_order]
    ref_order = [s.id for s in KPDR_REFERENCE_SPLITS]
    order_shifts: list[str] = []
    for idx, sid in enumerate(actual_order):
        if sid in ref_order:
            expected_before = [x for x in ref_order[: ref_order.index(sid)] if x in our_splits]
            actual_before = actual_order[:idx]
            if any(x not in actual_before for x in expected_before):
                order_shifts.append(sid)

    # 1. Reference-order comparison table (1:1 with user's uploaded image).
    ref_table: list[dict[str, Any]] = []
    prev_ref_run = prev_ref_pb = 0.0
    for ref in KPDR_REFERENCE_SPLITS:
        seg_ref = ref.ref_run_seconds - prev_ref_run
        seg_pb = ref.ref_pb_seconds - prev_ref_pb
        prev_ref_run = ref.ref_run_seconds
        prev_ref_pb = ref.ref_pb_seconds
        ours = our_splits.get(ref.id)
        our_sec = None if ours is None else int(ours["rta_frames"]) / FPS_CONVENTION
        cum_delta = None if our_sec is None else round(our_sec - ref.ref_run_seconds, 2)
        ref_table.append({
            "id": ref.id,
            "label": ref.label,
            "movable": ref.movable,
            "order_shifted": ref.id in order_shifts,
            "ref_pb_seconds": round(ref.ref_pb_seconds, 2),
            "ref_run_seconds": round(ref.ref_run_seconds, 2),
            "ref_seg_seconds": round(seg_ref, 2),
            "ref_pb_seg_seconds": round(seg_pb, 2),
            "our_rta_seconds": None if our_sec is None else round(our_sec, 2),
            "our_rta_frames": None if ours is None else int(ours["rta_frames"]),
            "our_row": None if ours is None else ours.get("row"),
            "delta_vs_ref_seconds": cum_delta,
        })

    # 2. Actual run-order segment breakdown with per-segment skill coverage.
    actual_segments: list[dict[str, Any]] = []
    prev_frame = 0
    prev_sid = None
    ceres_frame = int(splits_data.get("ceres_control_frame", 0))
    row_bounds = []
    offset = -ceres_frame
    for row in rows:
        end = offset + int(row["frames"])
        row_bounds.append((offset, end, row))
        offset = end
    for sid in actual_order:
        split = our_splits[sid]
        end_frame = int(split["rta_frames"])
        ref = REFERENCE_BY_ID[sid]
        ref_idx = ref_order.index(sid)
        ref_prev = ref_order[ref_idx - 1] if ref_idx else None
        seg_rows = [
            dict(row, frames=min(end_frame, end) - max(prev_frame, start))
            for start, end, row in row_bounds
            if max(prev_frame, start) < min(end_frame, end)
        ]
        cov = _row_kind_counts(seg_rows, coverage_by_row)
        actual_segments.append({
            "id": sid,
            "label": ref.label,
            "start_row": seg_rows[0]["row"] if seg_rows else split.get("row"),
            "end_row": split.get("row"),
            "row_count": len(seg_rows),
            "rta_seconds": round(end_frame / FPS_CONVENTION, 2),
            "seg_frames": end_frame - prev_frame,
            "seg_seconds": round((end_frame - prev_frame) / FPS_CONVENTION, 2),
            "same_start_split": prev_sid == ref_prev,
            "skill_pct": cov["skill_pct"],
            "skill_frames": cov["skill_frames"],
            "total_frames": cov["total_frames"],
            "counts": cov["counts"],
        })
        prev_frame = end_frame
        prev_sid = sid

    non_boot_rows = [r for r in rows if r.get("room") != "0x0000"]
    overall_cov = _row_kind_counts(non_boot_rows, coverage_by_row)

    rta_ship_frames = splits_data.get("rta_ship_frames")
    rta_ship_sec = frames_to_seconds(rta_ship_frames)
    ref_total_sec = KPDR_REFERENCE_SPLITS[-1].ref_run_seconds
    sub55_sec = SUB55_BUDGET_FRAMES / FPS_CONVENTION

    return {
        "schema_version": 1,
        "kind": "super_metroid_speed_benchmark",
        "stem": str(stem),
        "tape_sha256": report.get("tape_sha256"),
        "source_kind": splits_data.get("kind"),
        "credits_complete": bool(splits_data.get("credits_complete")),
        "total_frames_from_power_on": int(report.get("total_frames", 0)),
        "ceres_control_frame": int(splits_data.get("ceres_control_frame", 0)),
        "rta_ship_frames": rta_ship_frames,
        "rta_ship_seconds": None if rta_ship_sec is None else round(rta_ship_sec, 2),
        "rta_ship": format_rta(rta_ship_sec),
        "ref_total_seconds": round(ref_total_sec, 2),
        "ref_total": format_rta(ref_total_sec),
        "delta_vs_ref_seconds": None if rta_ship_sec is None else round(rta_ship_sec - ref_total_sec, 2),
        "sub55_target_seconds": sub55_sec,
        "delta_vs_sub55_seconds": None if rta_ship_sec is None else round(rta_ship_sec - sub55_sec, 2),
        "route_order_shifts": order_shifts,
        "overall_policy_coverage": overall_cov,
        "reference_comparison": ref_table,
        "actual_segments": actual_segments,
        "hotspots": rank_hotspots(rows, top_n=top_hotspots),
    }


def record_benchmark_history(
    bench: Mapping[str, Any],
    *,
    label: str,
    notes: str = "",
    history_path: Path = HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Append a verified benchmark snapshot to the longitudinal history log."""
    if bench["source_kind"] != "kpdr_splits_scan" or not bench["credits_complete"] or bench["rta_ship_frames"] is None:
        raise ValueError("Benchmark history requires a completed ROM split scan")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if history_path.is_file():
        history = json.loads(history_path.read_text())

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "stem": bench["stem"],
        "tape_sha256": bench["tape_sha256"],
        "source_kind": bench["source_kind"],
        "credits_complete": bench["credits_complete"],
        "rta_ship_frames": bench["rta_ship_frames"],
        "rta_ship_seconds": bench["rta_ship_seconds"],
        "rta_ship": bench["rta_ship"],
        "delta_vs_ref_seconds": bench["delta_vs_ref_seconds"],
        "delta_vs_sub55_seconds": bench["delta_vs_sub55_seconds"],
        "skill_pct": bench["overall_policy_coverage"]["skill_pct"],
        "skill_frames": bench["overall_policy_coverage"]["skill_frames"],
        "notes": notes,
        "splits": {
            row["id"]: row["our_rta_seconds"]
            for row in bench["reference_comparison"]
            if row["our_rta_seconds"] is not None
        },
        "split_frames": {
            row["id"]: row["our_rta_frames"]
            for row in bench["reference_comparison"]
            if row["our_rta_frames"] is not None
        },
    }
    # Replace existing entry with same label + tape hash to stay idempotent.
    history = [
        h for h in history
        if not (h.get("label") == label and h.get("tape_sha256") == entry["tape_sha256"])
    ]
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    return history


def format_benchmark_report(bench: Mapping[str, Any], *, history: Sequence[Mapping[str, Any]] = ()) -> str:
    """Render a clean Markdown benchmark report comparing against the 56:28.88 image."""
    lines: list[str] = []
    cov = bench["overall_policy_coverage"]
    counts = cov["counts"]
    total = cov["total_frames"] or 1
    ship_frames = bench["rta_ship_frames"]
    rta = "Ship split pending" if ship_frames is None else f"`{bench['rta_ship']}` ({ship_frames:,}f @ 60 fps)"
    lines.append(f"# Super Metroid Speed Benchmark — `{Path(bench['stem']).name}`")
    lines.append("")
    lines.append(
        f"- **RTA (Ceres → Ship):** {rta} "
        f"| **Credits complete:** `{bench['credits_complete']}` "
        f"| **Source:** `{bench['source_kind']}`"
    )
    lines.append(
        f"- **vs. Reference Image (`0:56:28.88` KPDR Late Ice):** "
        f"`{format_rta(bench['delta_vs_ref_seconds'], signed=True)}` "
        f"| **vs. Sub-55:00 Goal (`55:00.00` / 198,000f):** "
        f"`{format_rta(bench['delta_vs_sub55_seconds'], signed=True)}`"
    )
    lines.append(
        f"- **Policy Coverage (gameplay frames):** "
        f"Skill/Controller **{cov['skill_pct']}%** ({counts['controller']:,}f) · "
        f"Searched **{100.0 * counts['searched_buttons'] / total:.1f}%** ({counts['searched_buttons']:,}f) · "
        f"Human Retimed **{100.0 * counts['human_retimed'] / total:.1f}%** ({counts['human_retimed']:,}f) · "
        f"Human Verbatim **{100.0 * counts['human_verbatim'] / total:.1f}%** ({counts['human_verbatim']:,}f)"
    )
    if bench["route_order_shifts"]:
        lines.append(
            "- **Route-order shifts vs Late Ice image:** "
            + ", ".join(f"`{s}`" for s in bench["route_order_shifts"])
            + " (collected earlier on current route; see Actual Run Order table for true segment durations)"
        )
    lines.append("")
    lines.append("## Standard KPDR Splits vs. Reference Image (`0:56:28.88`)")
    lines.append("")
    lines.append("| # | Split | Ref PB | Ref Run (`56:28`) | Our RTA | Cum `+/-` | Row | Notes |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for idx, row in enumerate(bench["reference_comparison"], start=1):
        note = "route shifted (early)" if row["order_shifted"] else ("movable" if row["movable"] else "")
        r_str = str(row["our_row"]) if row["our_row"] is not None else "—"
        lines.append(
            f"| {idx} | {row['label']} | {format_rta(row['ref_pb_seconds'])} "
            f"| {format_rta(row['ref_run_seconds'])} | {format_rta(row['our_rta_seconds'])} "
            f"| {format_rta(row['delta_vs_ref_seconds'], signed=True)} | {r_str} | {note} |"
        )

    lines.append("")
    lines.append("## Actual Run-Order Segments & Skill Coverage")
    lines.append("")
    lines.append("† The route reaches this milestone from a different preceding split.")
    lines.append("")
    lines.append("| Segment (ends at) | Rows | Ours | Ref PB | Δ vs PB | Ref 56:28 | Δ vs run | Skill % | Frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ref_by_id = {row["id"]: row for row in bench["reference_comparison"]}
    for seg in bench["actual_segments"]:
        ref = ref_by_id[seg["id"]]
        pb_sec = ref["ref_pb_seg_seconds"]
        run_sec = ref["ref_seg_seconds"]
        seg_sec = seg["seg_frames"] / FPS_CONVENTION
        delta = "" if seg["same_start_split"] else " †"
        lines.append(
            f"| {seg['label']}{delta} | {seg['start_row']}–{seg['end_row']} ({seg['row_count']}) "
            f"| {format_rta(seg_sec)} | {format_rta(pb_sec)} "
            f"| {format_rta(seg_sec - pb_sec, signed=True)} | {format_rta(run_sec)} "
            f"| {format_rta(seg_sec - run_sec, signed=True)} "
            f"| {seg['skill_pct']:5.1f}% | {seg['seg_frames']:,} |"
        )

    lines.append("")
    lines.append("## Top Time-Loss Hotspots (Single Rows)")
    lines.append("")
    lines.append("| Row | Room | Time | Frames | Method | End E |")
    lines.append("|---:|---|---:|---:|---|---:|")
    for h in bench["hotspots"]:
        lines.append(
            f"| {h['row']} | {h['name']} (`{h['room']}`→`{h['to_room'] or 'pickup'}`) "
            f"| {format_rta(h['seconds'])} | {h['frames']:,} | `{h['method']}` | {h['energy_end']} |"
        )

    if history:
        lines.append("")
        lines.append("## Longitudinal Progress History")
        lines.append("")
        lines.append("| Label | RTA (Ceres→Ship) | Saved vs Baseline | vs Sub-55 | Skill % | Notes |")
        lines.append("|---|---:|---:|---:|---:|---|")
        base_sec = history[0].get("rta_ship_seconds")
        for h in history:
            cur = h.get("rta_ship_seconds")
            saved = (base_sec - cur) if (base_sec is not None and cur is not None) else None
            saved_str = format_rta(-saved, signed=True) if saved is not None else "—"
            lines.append(
                f"| `{h['label']}` | {h['rta_ship']} | {saved_str} "
                f"| {format_rta(h.get('delta_vs_sub55_seconds'), signed=True)} "
                f"| {h.get('skill_pct', 0.0):.1f}% | {h.get('notes', '')} |"
            )
    return "\n".join(lines) + "\n"
