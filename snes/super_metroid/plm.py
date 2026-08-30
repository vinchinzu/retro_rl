"""Live PLM + Samus-projectile snapshots (shot-block hit evidence).

Low WRAM only (``env.get_ram()``). PLM instruction change or ID→0 is the
hit; a spawned projectile is the beam. Differential until a route pins
the exact Wave-block PLM IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from super_metroid.combat.enemies.scan import enemies_from_ram
from super_metroid.paths import SCRATCH_STATE_DIR

ADDR_ROOM_WIDTH = 0x07A5
ADDR_ROOM_HEIGHT = 0x07A7
ADDR_PLM_ID = 0x1C37
ADDR_PLM_BLOCK = 0x1C87
ADDR_PLM_INST = 0x1D27
N_PLMS = 40
ADDR_SAMUS_PROJ_TYPE = 0x0C04
ADDR_SAMUS_PROJ_X = 0x0C18
ADDR_SAMUS_PROJ_Y = 0x0C4A
N_SAMUS_PROJECTILES = 10
NEAR_PX = 80
LIP_SHOT_TRACE = SCRATCH_STATE_DIR / "ws_main_lip_shot.json"
# Wave / shot blocks are clipdata. A hit *spawns* one of these PLMs; they
# are not preloaded ids that go 0. Take02–05 lip UP+X at ~(1223,1860)
# first spawn 0xD080 then 0xD074 / 0xD078.
SHOT_BLOCK_PLM_IDS = frozenset({0xD074, 0xD078, 0xD080})


def _u16(ram: np.ndarray, addr: int) -> int:
    if addr + 1 >= len(ram):
        return 0
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def session_ram(session: Any) -> np.ndarray | None:
    env = getattr(session, "env", None)
    if env is None:
        return None
    try:
        ram = env.get_ram()
    except Exception:
        return None
    if ram is None:
        return None
    return np.asarray(ram, dtype=np.uint8)


def plm_block_pixels(
    block: int, room_width_blocks: int
) -> tuple[int, int, int, int]:
    """Map ``$1C87`` to ``(bx, by, px, py)``.

    ``$1C87`` is a byte offset into two-byte level data, not a cell
    index. Bank ``$84`` divides by two before converting to block
    coordinates.
    """
    width = max(int(room_width_blocks), 1)
    cell = int(block) // 2
    bx = cell % width
    by = cell // width
    return bx, by, bx * 16 + 8, by * 16 + 8


def snapshot_plms(ram: np.ndarray, n: int = N_PLMS) -> tuple[dict[str, int], ...]:
    """Active PLM slots: id, instruction pointer, block index, pixel xy."""
    width = _u16(ram, ADDR_ROOM_WIDTH) or 1
    rows: list[dict[str, int]] = []
    for i in range(n):
        pid = _u16(ram, ADDR_PLM_ID + i * 2)
        if not pid:
            continue
        block = _u16(ram, ADDR_PLM_BLOCK + i * 2)
        inst = _u16(ram, ADDR_PLM_INST + i * 2)
        bx, by, px, py = plm_block_pixels(block, width)
        rows.append(
            {
                "i": i,
                "id": pid,
                "inst": inst,
                "block": block,
                "bx": bx,
                "by": by,
                "px": px,
                "py": py,
            }
        )
    return tuple(rows)


def snapshot_projectiles(ram: np.ndarray) -> tuple[dict[str, int], ...]:
    """Live Samus projectiles (type 0 = empty)."""
    rows: list[dict[str, int]] = []
    for slot in range(N_SAMUS_PROJECTILES):
        kind = _u16(ram, ADDR_SAMUS_PROJ_TYPE + slot * 2)
        if not kind:
            continue
        rows.append(
            {
                "slot": slot,
                "type": kind,
                "x": _u16(ram, ADDR_SAMUS_PROJ_X + slot * 2),
                "y": _u16(ram, ADDR_SAMUS_PROJ_Y + slot * 2),
            }
        )
    return tuple(rows)


def diff_plms(
    before: tuple[dict[str, int], ...],
    after: tuple[dict[str, int], ...],
) -> tuple[dict[str, object], ...]:
    """Per-slot id/inst/block changes. Missing after-slot = destroyed (id 0)."""
    prev = {int(row["i"]): row for row in before}
    nxt = {int(row["i"]): row for row in after}
    changes: list[dict[str, object]] = []
    for i in sorted(set(prev) | set(nxt)):
        a = prev.get(i)
        b = nxt.get(i)
        if a == b:
            continue
        hit = b is None or (
            a is not None
            and (int(a["id"]) != int(b["id"]) or int(a["inst"]) != int(b["inst"]))
        )
        changes.append(
            {
                "i": i,
                "before": a,
                "after": b,
                "destroyed": b is None,
                "hit": hit,
            }
        )
    return tuple(changes)


def near_samus(row: dict[str, int], samus_x: int, samus_y: int, slack: int = NEAR_PX) -> bool:
    """True if the PLM pixel is next to or above Samus (UP shot band)."""
    px, py = int(row["px"]), int(row["py"])
    if abs(px - int(samus_x)) > slack:
        return False
    return py <= int(samus_y) + 16


def plms_from_compact(rows: list[list[int]] | tuple[list[int], ...] | None) -> tuple[dict[str, int], ...]:
    """Tape ``plms`` rows ``[i, id, px, py, inst]`` → snapshot dicts."""
    out: list[dict[str, int]] = []
    for row in rows or ():
        if len(row) < 5:
            continue
        out.append(
            {
                "i": int(row[0]),
                "id": int(row[1]),
                "px": int(row[2]),
                "py": int(row[3]),
                "inst": int(row[4]),
                "block": 0,
                "bx": 0,
                "by": 0,
            }
        )
    return tuple(out)


def shot_block_spawns(
    before: tuple[dict[str, int], ...],
    after: tuple[dict[str, int], ...],
) -> tuple[dict[str, int], ...]:
    """New 0xD074 / 0xD078 / 0xD080 ids. Empty ``before`` is a seed, not a hit.

    Take02 adds slots 30/31. Live Main Shaft often *reuses* an existing
    slot index (id 0xC842 → 0xD080). Count id changes, not only new ``i``.
    """
    if not before:
        return ()
    prev_id = {int(row["i"]): int(row["id"]) for row in before}
    spawned: list[dict[str, int]] = []
    for row in after:
        pid = int(row["id"])
        if pid not in SHOT_BLOCK_PLM_IDS:
            continue
        if prev_id.get(int(row["i"])) == pid:
            continue
        spawned.append(row)
    return tuple(spawned)


def nearby_hits(
    changes: tuple[dict[str, object], ...],
    samus_x: int,
    samus_y: int,
) -> tuple[dict[str, object], ...]:
    hits: list[dict[str, object]] = []
    for ch in changes:
        if not ch.get("hit"):
            continue
        loc = ch.get("before") or ch.get("after")
        if not isinstance(loc, dict):
            continue
        if near_samus(loc, samus_x, samus_y):
            hits.append(ch)
    return tuple(hits)


def append_shot_event(
    session: Any,
    before_plms: tuple[dict[str, int], ...],
    before_projs: tuple[dict[str, int], ...],
    trace: list[dict[str, object]],
    *,
    buttons: tuple[str, ...],
    samus_x: int,
    samus_y: int,
) -> bool:
    """Diff after a step. Returns True when a nearby PLM was hit."""
    ram = session_ram(session)
    if ram is None:
        return False
    after_plms = snapshot_plms(ram)
    after_projs = snapshot_projectiles(ram)
    changes = diff_plms(before_plms, after_plms)
    hits = nearby_hits(changes, samus_x, samus_y)
    beam = bool(after_projs) or bool(before_projs)
    if not changes and not after_projs and not before_projs:
        return False
    trace.append(
        {
            "frame": int(getattr(session, "frame", 0)),
            "buttons": list(buttons),
            "xy": [int(samus_x), int(samus_y)],
            "beam": beam,
            "projectiles": [
                {**p, "type_hex": f"0x{int(p['type']):04X}"} for p in after_projs
            ],
            "plm_changes": [
                {
                    "i": c["i"],
                    "destroyed": c["destroyed"],
                    "hit": c["hit"],
                    "id_before": (
                        f"0x{int(c['before']['id']):04X}" if c["before"] else None
                    ),
                    "id_after": (
                        f"0x{int(c['after']['id']):04X}" if c["after"] else None
                    ),
                    "inst_before": (
                        f"0x{int(c['before']['inst']):04X}" if c["before"] else None
                    ),
                    "inst_after": (
                        f"0x{int(c['after']['inst']):04X}" if c["after"] else None
                    ),
                    "px": (c["before"] or c["after"] or {}).get("px"),
                    "py": (c["before"] or c["after"] or {}).get("py"),
                    "near": c in hits,
                }
                for c in changes
            ],
        }
    )
    return bool(hits)


def coverage_trace(ram: np.ndarray | None) -> dict[str, list[list[int]]]:
    """Compact per-frame enemies / PLMs / Samus projectiles for human tapes.

    Lists stay small (live slots only) so 4–5 takes can be diffed for missing
    skills: shoot-up (``projs``), block breaks (``plms`` id/inst), Atomic
    pathing (``enemies``).
    """
    if ram is None or len(ram) < ADDR_PLM_ID:
        return {"enemies": [], "plms": [], "projs": []}
    buf = np.asarray(ram, dtype=np.uint8)
    enemies = enemies_from_ram(buf)
    return {
        "enemies": [
            [e.slot, e.enemy_id, e.x, e.y, e.hp, e.freeze_timer] for e in enemies
        ],
        "plms": [
            [p["i"], p["id"], p["px"], p["py"], p["inst"]] for p in snapshot_plms(buf)
        ],
        "projs": [
            [p["slot"], p["type"], p["x"], p["y"]] for p in snapshot_projectiles(buf)
        ],
    }


def dump_shot_trace(
    trace: list[dict[str, object]],
    *,
    path: Path | None = None,
    extra: dict[str, object] | None = None,
) -> Path | None:
    """Overwrite the named scratch JSON. No-op on an empty trace."""
    if not trace:
        return None
    dest = path or LIP_SHOT_TRACE
    dest.parent.mkdir(parents=True, exist_ok=True)
    beam_frames = sum(1 for row in trace if row.get("beam"))
    near_hits = sum(
        1
        for row in trace
        for ch in row.get("plm_changes", [])
        if isinstance(ch, dict) and ch.get("near") and ch.get("hit")
    )
    payload = {
        "kind": "ws_main_lip_shot",
        "beam_frames": beam_frames,
        "near_plm_hits": near_hits,
        "events": trace,
    }
    if extra:
        payload.update(extra)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[LIP SHOT] beam_frames={beam_frames} near_plm_hits={near_hits} -> {dest}"
    )
    return dest


__all__ = [
    "ADDR_PLM_BLOCK",
    "ADDR_PLM_ID",
    "ADDR_PLM_INST",
    "ADDR_ROOM_WIDTH",
    "ADDR_SAMUS_PROJ_TYPE",
    "ADDR_SAMUS_PROJ_X",
    "ADDR_SAMUS_PROJ_Y",
    "LIP_SHOT_TRACE",
    "N_PLMS",
    "NEAR_PX",
    "SHOT_BLOCK_PLM_IDS",
    "append_shot_event",
    "coverage_trace",
    "diff_plms",
    "dump_shot_trace",
    "near_samus",
    "nearby_hits",
    "plm_block_pixels",
    "plms_from_compact",
    "session_ram",
    "shot_block_spawns",
    "snapshot_plms",
    "snapshot_projectiles",
]
