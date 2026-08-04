#!/usr/bin/env python3
"""Export KPDR progress tracker to JSON + markdown summary (chartable).

```bash
uv run python snes/super_metroid/scripts/export/kpdr_tracker.py
uv run python snes/super_metroid/scripts/export/kpdr_tracker.py --csv path --json out.json
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
DEFAULT_CSV = ROOT / "snes" / "super_metroid" / "docs" / "routes" / "KPDR_TRACKER.csv"
DEFAULT_JSON = ROOT / "snes" / "super_metroid" / "maps" / "kpdr_tracker.json"
DEFAULT_MD = ROOT / "snes" / "super_metroid" / "docs" / "routes" / "KPDR_TRACKER.md"

# status ranks for progress bars / charts
STATUS_RANK = {
    "continuous": 4,
    "controller_dev": 3,
    "dev_fight": 2,
    "dev_warp": 2,
    "dev_grant": 2,
    "dev_item": 2,
    "optional": 1,
    "open": 0,
    "future": 0,
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    by_status = Counter(r["status"] for r in rows)
    by_seg = Counter(r["seg_id"].split(".")[0] for r in rows)
    to_kraid_entry = [r for r in rows if r["seg_id"].startswith(("K0", "K1", "K2"))]
    ranks = [STATUS_RANK.get(r["status"], 0) for r in to_kraid_entry]
    max_rank = 4 * len(to_kraid_entry) if to_kraid_entry else 1
    progress_pct = round(100.0 * sum(ranks) / max_rank, 1)
    return {
        "totalSegments": len(rows),
        "toKraidEntrySegments": len(to_kraid_entry),
        "statusCounts": dict(sorted(by_status.items())),
        "majorSegCounts": dict(sorted(by_seg.items())),
        "kraidEntryPathProgressPct": progress_pct,
        "chartSeries": {
            "status": [{"status": k, "count": v} for k, v in sorted(by_status.items())],
            "byMajorSeg": [
                {
                    "seg": seg,
                    "segments": sum(1 for r in rows if r["seg_id"].startswith(seg)),
                    "doneish": sum(
                        1
                        for r in rows
                        if r["seg_id"].startswith(seg)
                        and STATUS_RANK.get(r["status"], 0) >= 2
                    ),
                }
                for seg in sorted(by_seg)
            ],
        },
    }


def to_markdown(rows: list[dict[str, str]], summary: dict[str, object]) -> str:
    lines = [
        "# KPDR progress tracker",
        "",
        "Machine-readable source: `KPDR_TRACKER.csv` · JSON: `maps/kpdr_tracker.json`.",
        "Regenerate: `uv run python snes/super_metroid/scripts/export/kpdr_tracker.py`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total segments | {summary['totalSegments']} |",
        f"| Super → Kraid-entry segments | {summary['toKraidEntrySegments']} |",
        f"| Kraid-entry path progress (weighted) | {summary['kraidEntryPathProgressPct']}% |",
        "",
        "### Status counts",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for k, v in summary["statusCounts"].items():  # type: ignore[union-attr]
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "### Chart series (status)",
        "",
        "```",
    ]
    counts = summary["statusCounts"]  # type: ignore[assignment]
    max_c = max(counts.values()) if counts else 1
    for k, v in counts.items():
        bar = "#" * max(1, int(20 * v / max_c))
        lines.append(f"{k:16} {v:3} {bar}")
    lines += [
        "```",
        "",
        "## Segment table (Super → Kraid-entry focus)",
        "",
        "| # | Seg | Room | Status | Layer | Item/Boss | Anchor |",
        "|--:|-----|------|--------|-------|-----------|--------|",
    ]
    for r in rows:
        if not r["seg_id"].startswith(("K0", "K1", "K2")):
            continue
        lines.append(
            f"| {r['order']} | `{r['seg_id']}` | {r['room_id_hex']} {r['room_name']} "
            f"| **{r['status']}** | {r['layer']} | {r['item_or_boss']} "
            f"| `{r['anchor_state']}` |"
        )
    lines += [
        "",
        "## Full route (later KPDR)",
        "",
        "| # | Seg | Room | Status | Notes |",
        "|--:|-----|------|--------|-------|",
    ]
    for r in rows:
        if r["seg_id"].startswith(("K0", "K1", "K2")):
            continue
        lines.append(
            f"| {r['order']} | `{r['seg_id']}` | {r['room_id_hex']} {r['room_name']} "
            f"| {r['status']} | {r['notes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = load_rows(args.csv)
    summary = summarize(rows)
    payload = {
        "sourceCsv": str(args.csv.relative_to(ROOT)),
        "summary": summary,
        "segments": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(to_markdown(rows, summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "csv": str(args.csv),
                "json": str(args.json),
                "md": str(args.md),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
