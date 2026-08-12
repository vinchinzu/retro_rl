#!/usr/bin/env python3
"""Classify an interact from an existing tape or a live pin. Do not record.

    # Offline: first held-item change on a traced recording
    uv run python -m harvest.scripts.interact_scan tape mountain_grape_stand

    # Decomp: what box will we see?
    uv run python -m harvest.scripts.interact_scan search grape

    # Live stand pin: nearby NPCs + one A
    HEADLESS=1 uv run python -m harvest.scripts.interact_scan tap --state <pin>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harvest.paths import PROJECT_DIR, TASKS_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from harvest.core.animal_status import read_held_item
from harvest.core.interact import classify_interact, first_held_change, held_name
from harvest.core.npc_catalog import (
    current_dialogue_registers,
    game_objects,
    search_text_records,
    text_record_for_id,
)
from harvest.core.ram_catalog import read_ram_value
from harvest.tasks.nav import get_pos_from_ram, make_action


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _text_payload(text_id: int) -> dict:
    rec = text_record_for_id(text_id)
    if rec is None:
        return {"text_id": text_id, "text_id_hex": f"0x{text_id:04X}", "text": ""}
    return rec.to_dict(compact=False)


def _cmd_tape(name: str) -> dict:
    path = Path(name)
    if not path.suffix:
        path = Path(TASKS_DIR) / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = data.get("trace") or []
    if not trace:
        return {
            "mode": "tape",
            "task": path.name,
            "error": "no RAM trace — replay once, do not re-record",
            "replay": (
                "HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe "
                f"--mode replay --task {path.stem}"
            ),
        }
    change = first_held_change(trace)
    if change is None:
        return {"mode": "tape", "task": path.name, "error": "held_item never changed"}
    # Buttons around the change (A / Down) so we do not guess facing.
    idx = next(i for i, row in enumerate(trace) if row.get("frame") == change.get("frame"))
    window = trace[max(0, idx - 4) : min(len(trace), idx + 80)]
    inputs = [
        {
            "frame": r.get("frame"),
            "buttons": r.get("buttons") or [],
            "held": r.get("held_item"),
            "lock": r.get("input_lock"),
            "pos": [r.get("x"), r.get("y")],
        }
        for r in window
        if r.get("buttons")
    ]
    kind = classify_interact(
        held_before=int(change["held_before"]),
        held_after=int(change["held_after"]),
        lock_after=int(change.get("input_lock") or 1),
        text_choices=("Don't eat",) if int(change["held_after"]) in {1, 2, 3, 4, 5} else (),
    )
    print(
        f"[SCAN] {path.stem} first held {change['held_before']}→{change['held_after']} "
        f"({change['held_name']}) f={change.get('frame')} "
        f"pos=({change.get('x')},{change.get('y')}) class={kind}"
    )
    return {
        "mode": "tape",
        "task": path.stem,
        "class": kind,
        "first_held_change": change,
        "nearby_inputs": inputs[:16],
    }


def _cmd_search(query: str) -> dict:
    hits = search_text_records(query)
    print(f"[SCAN] {len(hits)} text hits for {query!r}")
    for rec in hits:
        preview = rec.text.replace("\n", " / ")[:80]
        print(f"  0x{rec.text_id:04X} {rec.category} {preview}")
    return {"mode": "search", "query": query, "hits": [h.to_dict() for h in hits]}


def _live_row(ram) -> dict:
    pos = get_pos_from_ram(ram)
    held = int(read_held_item(ram))
    dlg = current_dialogue_registers(ram)
    rec = text_record_for_id(int(dlg["text_id"]))
    objects = []
    for obj in game_objects(ram):
        if obj.is_player:
            continue
        objects.append(
            {
                "label": obj.label,
                "kind": obj.kind,
                "sprite": f"0x{obj.sprite_table_idx:04X}",
                "tile": list(obj.tile),
                "pixel": list(obj.pixel),
            }
        )
    return {
        "tilemap": int(read_ram_value(ram, "tilemap", raw=True)),
        "x": int(pos.x),
        "y": int(pos.y),
        "tx": int(pos.x) // 16,
        "ty": int(pos.y) // 16,
        "held": held,
        "held_name": held_name(held),
        "lock": int(dlg["input_lock"]),
        "text_id": int(dlg["text_id"]),
        "text_id_hex": f"0x{int(dlg['text_id']):04X}",
        "text": rec.text if rec else "",
        "choices": list(rec.choices) if rec else [],
        "cursor": int(dlg["menu_cursor"]),
        "objects": objects[:12],
    }


def _cmd_tap(state: str) -> dict:
    _configure_headless()
    from harvest.runtime.retro_setup import make_harvest_env

    env = make_harvest_env(state)
    env.reset()
    before = _live_row(env.get_ram())
    npc_near = any(
        obj["kind"] == "npc_candidate"
        and abs(obj["tile"][0] - before["tx"]) + abs(obj["tile"][1] - before["ty"]) <= 2
        for obj in before["objects"]
    )
    print(
        f"[SCAN] before held={before['held_name']} lock={before['lock']} "
        f"pos=({before['x']},{before['y']}) npcs={sum(1 for o in before['objects'] if o['kind']=='npc_candidate')}"
    )
    env.step(make_action(a=True))
    for _ in range(20):
        env.step(make_action())
    after = _live_row(env.get_ram())
    kind = classify_interact(
        held_before=before["held"],
        held_after=after["held"],
        lock_after=after["lock"],
        text_choices=after["choices"] or after["text"].splitlines(),
        npc_in_face=npc_near,
    )
    print(
        f"[SCAN] after  held={after['held_name']} lock={after['lock']} "
        f"text={after['text_id_hex']} class={kind}"
    )
    if after["text"]:
        print(after["text"])
    return {"mode": "tap", "state": state, "class": kind, "before": before, "after": after}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tape", help="First held-item change on a traced recording")
    t.add_argument("task")
    s = sub.add_parser("search", help="Grep UnlinkedText via the pointer table")
    s.add_argument("query")
    a = sub.add_parser("tap", help="One live A from a save-state pin")
    a.add_argument("--state", required=True)
    a.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "interact_scan.json",
    )
    args = p.parse_args()
    if args.cmd == "tape":
        report = _cmd_tape(args.task)
    elif args.cmd == "search":
        report = _cmd_search(args.query)
    else:
        report = _cmd_tap(args.state)
    out = getattr(args, "out", None)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    elif args.cmd != "search":
        print(json.dumps(report, indent=2)[:2000])
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
