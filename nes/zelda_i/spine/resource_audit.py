"""Read-only bomb/key/health supply audit of a continuous spine report.

    uv run python -m zelda_i.spine.resource_audit nes/zelda_i/recordings/lastheart_poweron37.json

Reports observations, not a counterfactual Clean route: removing an assist
changes combat, drops and subsequent inventory. Use the listed rooms and gates
as replay targets, then measure again from power-on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def audit(report: dict[str, Any]) -> dict[str, Any]:
    ledger = report.get("ledger") or {}
    gains = ledger.get("gains") or []
    spends = ledger.get("spends") or []
    writes = (report.get("inventory_assist") or {}).get("writes") or []
    shops = []
    for stage in report.get("stages") or []:
        ctl = stage.get("controller") or {}
        price = int(ctl.get("price") or 0)
        if price <= 0:
            continue
        name = stage.get("name", "?")
        bought = [
            g for g in gains
            if g.get("source") == "play"
            and int(stage.get("frame_base", 0)) < int(g.get("frame", 0)) <= int(stage.get("end_frame", 0))
            and g.get("field") in ("bombs", "keys", "potion", "arrows", "food", "candle", "ring")
        ]
        notes = ctl.get("notes") or []
        skipped = any("nothing_to_buy" in str(note) for note in notes)
        shops.append({
            "stage": name,
            "price": price,
            "success": bool(stage.get("success")),
            "item_gains": bought,
            "rupees_at_buy": ctl.get("rupees_at_buy"),
            "rupees_out": (ctl.get("leftover") or {}).get("rupees"),
            "outcome": (
                "bought" if stage.get("success") and bought
                else "skipped" if skipped
                else "already_supplied" if stage.get("success") and not ctl.get("min_item_gain")
                else "no_measured_buy"
            ),
        })
    gates = [
        {
            "stage": w.get("stage") or "unlabelled gate",
            "field": w["field"],
            "before": int(w.get("from", 0)),
            "after": int(w["to"]),
            "granted": int(w["to"]) - int(w.get("from", 0)),
        }
        for w in writes if w.get("field") in ("bombs", "keys")
    ]
    missed_drops = [
        d for d in (ledger.get("drops") or {}).get("missed_rows") or []
        if d.get("kind") in ("bomb", "heart", "fairy")
    ]
    missed_items = [
        r for r in (ledger.get("room_items") or {}).get("missed") or []
        if r.get("item") in ("bombs", "key", "heart_container")
    ]
    return {
        "shops": shops,
        "unbought_stages": [s for s in shops if s["outcome"] != "bought"],
        "assist_gates": gates,
        "spend_totals": {
            field: (sum(int(s["from"]) - int(s["to"]) for s in spends if s["field"] == field)
                    if "spends" in ledger else None)
            for field in ("bombs", "keys")
        },
        "assist_totals": {
            field: sum(g["granted"] for g in gates if g["field"] == field)
            for field in ("bombs", "keys")
        },
        "missed_drops": missed_drops,
        "missed_room_items": missed_items,
        "forced_drop_windows": ledger.get("forced_drop_windows") or [],
        "health_refills": int(((report.get("assist") or {}).get("health") or {}).get("writes", 0)),
    }


def main(paths: list[str]) -> int:
    if not paths:
        raise SystemExit("usage: python -m zelda_i.spine.resource_audit REPORT.json [...]")
    for name in paths:
        result = audit(json.loads(Path(name).read_text()))
        print(f"{name}: assist bombs={result['assist_totals']['bombs']} "
              f"keys={result['assist_totals']['keys']} health={result['health_refills']} "
              f"observed spend bombs={result['spend_totals']['bombs']} "
              f"keys={result['spend_totals']['keys']}")
        for shop in result["shops"]:
            gained = ",".join(f"{g['field']}:{g['from']}->{g['to']}" for g in shop["item_gains"]) or "-"
            print(f"  buy {shop['stage']}: {shop['outcome']} price={shop['price']} gain={gained}")
        for gate in result["assist_gates"]:
            print(f"  gate {gate['stage']}: {gate['field']} {gate['before']}->{gate['after']}")
        for drop in result["missed_drops"]:
            if drop["kind"] == "bomb":
                bankable = drop.get("bankable", "unknown")
                print(f"  missed bomb drop {drop['room']} f{drop['frame']} "
                      f"{drop['outcome']} bankable={bankable}")
        for item in result["missed_room_items"]:
            print(f"  missed room item {item['room']} {item['item']}")
        for window in result["forced_drop_windows"]:
            print(f"  bomb-drop window {window['room']} f{window['frame']} "
                  f"fairy_preempts={window['fairy_preempts_next_kill']}")
        missed_heals = sum(d["kind"] in ("heart", "fairy") for d in result["missed_drops"])
        print(f"  missed health drops={missed_heals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
