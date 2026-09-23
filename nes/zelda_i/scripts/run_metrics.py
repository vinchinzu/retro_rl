"""One milestone row from a spine report: frames, flutter, drops, assists.

    uv run python nes/zelda_i/scripts/run_metrics.py nes/zelda_i/recordings/full_poweron12.json

Paste the row into ``docs/RUN_METRICS.md``. The report must carry ``ledger``
(``spine/ledger.py``; every ``run_survival_spine`` report does).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

POKED = ("bombs", "keys", "rupees")


def assist_totals(report: dict) -> dict[str, object]:
    """Units each Survival poke granted (``to - from``), plus the heart refill."""
    out = {field: 0 for field in POKED}
    for write in (report.get("inventory_assist") or {}).get("writes") or []:
        if write.get("field") in out:
            out[write["field"]] += int(write["to"]) - int(write.get("from", 0))
    assist = report.get("assist") or {}
    out["refill"] = int((assist.get("health") or {}).get("writes", 0))
    out["kind"] = str(assist.get("kind") or "off")
    out["hits"] = int(assist.get("damage_events", 0))
    return out


def row(path: Path) -> str:
    report = json.loads(path.read_text())
    ledger = report["ledger"]
    drops = ledger["drops"]
    kinds = drops["by_kind"]
    picked = sum(v.get("picked", 0) for v in kinds.values())
    missed_hearts = sum(
        1 for d in drops["missed_rows"] if d["kind"] in ("heart", "fairy")
    )
    a = assist_totals(report)
    slow = ledger["slowest_visits"][0]
    # Books added 2026-09-23; older reports leave these cells blank.
    damage = ledger.get("damage", "")
    items = ledger.get("room_items")
    missed = (
        " ".join(f"{r['room']}={r['item']}" for r in items["missed"]) or "none"
        if items is not None
        else ""
    )
    return (
        f"| {path.stem} | {'ok' if report['ok'] else report.get('failed_stage')} ({a['kind']}) "
        f"| {report.get('through')} | {ledger['frames']} | {ledger['flutters']} "
        f"| {picked}/{drops['total']} | {missed_hearts} "
        f"| {a['hits']} / {a['refill']} | {a['bombs']} / {a['keys']} / {a['rupees']} "
        f"| {slow['room']} {slow['frames']}f | {damage} | {missed} |"
    )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(row(Path(arg)))
