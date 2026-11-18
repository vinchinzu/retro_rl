"""Try human inputs from the corresponding source entry on the Clean chain board.

The ROM gets each candidate's exact board slice and the solver's leave check.
Source states can have assisted resources. Only the master replay counts a
room toward Clean completion.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.human_tape.product_chain import build_clean_chain_board
from super_metroid.paths import GAME_DIR
from super_metroid.splice.assemble import _chain_body, _guard_writes, _try_chain_body

OUT = GAME_DIR / "recordings" / "tape_census_clean.json"


def audit_board(board: dict[str, Any]) -> dict[str, Any]:
    """Check the full planned route using source traces and physical topology."""
    graph = json.loads((GAME_DIR / "maps" / "full_room_graph.json").read_text())
    rooms = {room["roomId"] for room in graph["rooms"]}
    edges = {(edge["source"]["roomId"], edge["target"]["roomId"])
             for edge in graph["edges"] if not edge["impossible"]}
    cache = {}
    problems = []
    route_rooms = set()
    crossings = 0
    target = 0
    hops = board["hops"]
    for index, row in enumerate(hops):
        if row["row"] != index:
            problems.append(f"row {index}: incorrect row ID {row['row']}")
        source = int(row["room"], 16)
        if index == 0 and source == 0x5555:
            continue  # Reset metadata belongs to the following boot slice.
        if source != target:
            problems.append(f"row {index}: previous leave was 0x{target:04X}, entry is {row['room']}")
        target = int(row.get("to_room") or row["room"], 16)
        route_rooms.update((source, target))
        if source and source != target:
            crossings += 1
            if (source, target) not in edges:
                problems.append(f"row {index}: no physical link {row['room']} -> {row['to_room']}")
        if row.get("tape"):
            try:
                if source and index and row["tape"] == hops[index - 1].get("tape"):
                    if row["source_start"] != hops[index - 1]["source_end"] + 1:
                        raise ValueError("input slice does not follow the previous slice on the same tape")
                actions, leave = _chain_body(row, cache)
                if actions.shape[1:] != (12,) or not np.isin(actions, (0, 1)).all():
                    raise ValueError("invalid SNES-12 input slice")
                if leave["room"] != target:
                    raise ValueError("tape slice ends in another room")
                if row.get("goal_phase") and leave.get("phase") != row["goal_phase"]:
                    raise ValueError("tape slice ends in another phase")
            except (ValueError, OSError, IndexError, KeyError) as exc:
                problems.append(f"row {index}: {exc}")
        elif not row.get("controller_id"):
            problems.append(f"row {index}: no tape or controller")
    if not hops or target != 0x91F8 or hops[-1].get("goal_phase") != "ending_or_credits":
        problems.append("route does not end at Landing Site ship launch")
    off_route = rooms - route_rooms
    return {
        "board_rows": len(hops), "source_tapes": len(cache),
        "planned_crossings": crossings,
        "route_rooms": len(route_rooms & rooms), "catalog_rooms": len(rooms),
        "off_route_room_groups": dict(sorted(Counter(
            room["area"] for room in graph["rooms"] if room["roomId"] in off_route
        ).items())),
        "problems": problems,
    }


def census_segment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one emulator and one decoded source tape per worker job."""
    env = make_dev_env()
    cache = {}
    results = []
    try:
        for row in rows:
            result = {key: row.get(key) for key in ("row", "segment", "room", "name", "to_room")}
            results.append(result)
            if row.get("controller_id") or row.get("mode") == "missing":
                result["status"] = "controller" if row.get("controller_id") else "missing"
                continue
            if row["room"] == "0x5555":
                result["status"] = "boot_metadata"
                continue
            if not row.get("anchor_path") and row["room"] != "0x0000":
                result["status"] = "no_exact_entry"
                continue
            try:
                body, final = _chain_body(row, cache)
                if row["room"] == "0x0000":
                    env.reset()
                else:
                    entry = boot_from_state(env, GAME_DIR / row["anchor_path"], settle_frames=0)
                    sample = final["reference"][0]
                    # Shared archive filenames (particularly boot pins) may
                    # have been reused by another take. Do not grade that pin.
                    if (entry.room_id, entry.samus_x, entry.samus_y, entry.pose,
                        entry.health, entry.missiles, entry.super_missiles, entry.power_bombs,
                        entry.collected_items, entry.collected_beams) != (
                        sample["room"], sample["x"], sample["y"], sample["pose"],
                        sample["energy"], sample["missiles"], sample["supers"], sample["pbs"],
                        sample["items"], sample["beams"]):
                        result["status"] = "entry_mismatch"
                        continue
                restore_writes = _guard_writes(env)
                try:
                    _grade(env, row, body, final, result)
                finally:
                    restore_writes()
            except (ValueError, OSError, IndexError, KeyError, RuntimeError) as exc:
                result.update(status="error", error=str(exc))
    finally:
        env.close()
    return results


def _grade(env: Any, row: dict, body: np.ndarray, final: dict, result: dict) -> None:
    from super_metroid.ram import parse_env_state

    saved = env.em.get_state()
    for delay in range(13):
        if delay:
            env.em.set_state(saved)
        presses = np.concatenate((np.zeros((delay, 12), dtype=np.int8), body))
        chosen = _try_chain_body(env, presses, row, final)
        if chosen is not None:
            result.update(status="retimed" if delay else "as_recorded", delay=delay,
                          frames=len(chosen), energy=parse_env_state(env, mode="nav").health)
            return
    result["status"] = "red"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", help="inclusive board row range, e.g. 40-60")
    parser.add_argument("--board", type=Path, help="use a saved Clean board (default rebuilds from source)")
    parser.add_argument("--audit", action="store_true", help="validate the full board against the tapes and room graph without booting")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    board = json.loads(args.board.read_text()) if args.board else build_clean_chain_board()
    if board.get("kind") != "super_metroid_clean_chain_candidates":
        parser.error("expected the Clean chain candidate board")
    if args.audit:
        result = audit_board(board)
        print(json.dumps(result, indent=2))
        raise SystemExit(bool(result["problems"]))
    hops = board["hops"]
    if args.rows:
        low, _, high = args.rows.partition("-")
        hops = hops[int(low) : int(high or low) + 1]
    segments = {}
    for row in hops:
        segments.setdefault(row.get("tape"), []).append(row)
    with Pool(args.workers) as pool:
        rows = sorted((r for group in pool.map(census_segment, segments.values()) for r in group),
                      key=lambda r: r["row"])
    out = args.out or (OUT.with_stem(f"{OUT.stem}_{hops[0]['row']}_{hops[-1]['row']}")
                       if args.rows and hops else OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print("Source-entry results", dict(Counter(row["status"] for row in rows)))
    for row in rows:
        if row["status"] in {"red", "error"}:
            print(f"{row['status']} {row['row']:3d} {row['segment']} {row['name']} {row.get('error', '')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
