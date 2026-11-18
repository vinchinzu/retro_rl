"""Final scorecard for a power-on (or pin) run that ends at the credits.

    .venv/bin/python -m harvest.tools.final_score recordings/e2e/<tag>/report.json
    .venv/bin/python -m harvest.tools.final_score <report.json> --out FINAL.md
    .venv/bin/python -m harvest.tools.final_score <A.json> <B.json>        # B - A delta

It answers four questions in one page: did the run reach the credits cleanly, how much
cash did it end with, what is the ranch score and which terms are short, and what did the
run cost (frames, rescues, bad days). The numbers all come from the run report that
``run_to_day2`` writes; ``y1_bench --sales`` stays the place for per-season sales detail
and is appended here as an appendix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from harvest.planner import run_strategy
from harvest.tools import y1_bench

COW_PRICE_G = 5000
HEN_PRICE_G = 1000  # the bench headline counts a hen at purchase price, resale is 500G
WALLET_CAP_G = 9_999_990

# Ranch Mastery terms in the order of the ROM calculator: (oracle key, label, max points).
TERMS: tuple[tuple[str, str, int], ...] = (
    ("money_score", "Money", 78),
    ("cows_score", "Cows (3 each)", 36),
    ("chickens_score", "Chickens (3 each)", 36),
    ("power_berries_score", "Power berries", 50),
    ("bachelorettes_score", "Hearts, five girls", 155),
    ("shipped_crops_score", "Shipped crops, four kinds", 124),
    ("happiness_score", "Player happiness", 31),
    ("house_upgrades_score", "House upgrades", 32),
    ("marriage_score", "Marriage", 32),
    ("children_score", "Children", 32),
    ("special_items_score", "Special items", 43),
    ("ranch_development_score", "Pasture and development", 50),
    ("cow_happiness_score", "Cow happiness (25 a cow)", 300),
)
# Where each term's missing points come from: (strategy that works on it, lever and status).
# Static on purpose: it is the route map, so a scorecard shows what to turn on next.
LEVERS: dict[str, tuple[str, str]] = {
    "Money": ("-", "capped at 99,840G; everything above it is spendable surplus"),
    "Cows (3 each)": ("score_herd", "cow_target_later; a cow is 3 + up to 25 happiness points; last feedable buy Y3 Spring D10"),
    "Chickens (3 each)": ("score", "final_flock_target; hatch toward 12 after Y3 Spring D10, subject to pasture and feed"),
    "Power berries": ("score", "scheduled native sources; spent spawn bits cannot recover an uneaten berry (F6)"),
    "Hearts, five girls": ("-", "each girl at 496-511 hearts; Eve takes morning eggs in either remodeled house (F9 ladders)"),
    "Shipped crops, four kinds": ("score", "stop at 496 a crop (window floor), tomatoes in Summer; turnip shelf verified; harvest earlier"),
    "Player happiness": ("-", "needs 992; dated sources in F7"),
    "House upgrades": ("-", "done in every run"),
    "Marriage": ("-", "done in every run"),
    "Children": ("-", "second child needs the wife bit by Y2 Fall D9 and hearts >= 650 (F4)"),
    "Special items": ("-", "clock and turtle shell (F6)"),
    "Pasture and development": ("-", "2,950 counted tiles (99.4% of the 2,968-cell field, every cell hoed first); "
                                     "328 bags is the floor, the raster plan needs 364-391 (see the terrain table)"),
    "Cow happiness (25 a cow)": ("score_herd", "25 a cow at happiness >= 200; needs 50 fed care days"),
}
RATING_BANDS = ((0, 199, "below Ranch Master"), (200, 399, "Ranch Master"), (400, 599, "good"),
                (600, 799, "great"), (800, 998, "near perfect"), (999, 999, "PERFECT"))
SCORE_CAP = 999
RANCH_MASTER_AT = 200


def _g(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}G"


def _n(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _band(score: int | None) -> str:
    if score is None:
        return "n/a"
    for low, high, name in RATING_BANDS:
        if low <= score <= high:
            return name
    return "n/a"


def _ending(report: Mapping[str, Any]) -> dict[str, Any]:
    ending = report.get("ending") or {}
    final = ending.get("final") or {}
    oracle = final.get("oracle") or {}
    return {
        "finished": bool(ending.get("finished")),
        "scene_index": final.get("scene_index"),
        "score": final.get("ranch_mastery_0x25"),
        "score_at_0x20": final.get("ranch_mastery_0x20"),
        "oracle": oracle,
    }


def verdict(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    """CLEAN, CREDITS (reached the credits but not clean) or INCOMPLETE, with the reasons."""
    ending = _ending(report)
    rows = y1_bench.day_rows(report)
    rescues = sum(row["rescues"] for row in rows)
    audit = report.get("clean_run") or {}
    reasons: list[str] = []
    if not (report.get("success") and ending["finished"] and ending["scene_index"] == 0x25):
        reasons.append(f"no credits: {report.get('reason') or 'unknown'} at {report.get('end_date') or '?'}")
        return "INCOMPLETE", reasons
    if rescues:
        reasons.append(f"{rescues} rescue(s)")
    for key in ("mid_run_state_loads", "ram_writes"):
        if int(audit.get(key) or 0):
            reasons.append(f"{key} = {audit.get(key)}")
    if audit.get("assists"):
        reasons.append(f"assists {audit.get('assists')}")
    if audit.get("intervention_class") != "Clean":
        reasons.append(f"intervention {audit.get('intervention_class')}")
    start = "power-on" if (report.get("power_on") or {}).get("completed") else f"pin {report.get('state') or '?'}"
    reasons.append(f"start: {start}") if not reasons else reasons.append(f"start: {start}")
    return ("CLEAN" if len(reasons) == 1 else "CREDITS"), reasons


def net_worth(report: Mapping[str, Any]) -> dict[str, int | None]:
    eco = report.get("end_economy") or {}
    end = report.get("end") or {}
    wallet = eco.get("money", end.get("money"))
    cows = eco.get("cows", end.get("num_cows"))
    hens = eco.get("hens", end.get("num_chickens"))
    if wallet is None:
        return {"wallet": None, "cows": cows, "hens": hens, "net_worth": None}
    total = int(wallet) + COW_PRICE_G * int(cows or 0) + HEN_PRICE_G * int(hens or 0)
    return {"wallet": int(wallet), "cows": cows, "hens": hens, "net_worth": total}


def headline(report: Mapping[str, Any]) -> dict[str, Any]:
    ending = _ending(report)
    eco = report.get("end_economy") or {}
    worth = net_worth(report)
    bin_gold = eco.get("bin_gold")
    wallet = worth["wallet"]
    rows = y1_bench.day_rows(report)
    state, reasons = verdict(report)
    summary = report.get("journal_summary") or {}
    floor = report.get("floor_report") or {}
    video = report.get("video") or {}
    return {
        "verdict": state,
        "verdict_reasons": reasons,
        "end_date": report.get("end_date"),
        "wallet": wallet,
        "closing_bin_gold": bin_gold,
        "wallet_with_bin": None if wallet is None or bin_gold is None else int(wallet) + int(bin_gold),
        "cows": worth["cows"],
        "hens": worth["hens"],
        "net_worth": worth["net_worth"],
        "ranch_score": ending["score"],
        "ranch_band": _band(ending["score"]),
        "ranch_master": None if ending["score"] is None else ending["score"] >= RANCH_MASTER_AT,
        "perfect": None if ending["score"] is None else ending["score"] >= SCORE_CAP,
        "strategy": report.get("strategy") or "cash (not recorded)",
        "frames": report.get("frames"),
        "wall_seconds": report.get("wall_seconds"),
        "days": summary.get("overnights", report.get("days_completed")),
        "rescues": sum(row["rescues"] for row in rows),
        "bad_days": floor.get("total_bad_days", sum(1 for row in rows if row["bad_day"])),
        "day_failures": len(report.get("day_failures") or []),
        "video_parts": len(video.get("parts") or []),
        "video_hours": None if not video.get("duration_seconds") else round(video["duration_seconds"] / 3600, 2),
    }


def breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    oracle = _ending(report)["oracle"]
    out = []
    for key, label, maximum in TERMS:
        value = oracle.get(key)
        out.append({"term": label, "points": value, "max": maximum,
                    "gap": None if value is None else maximum - int(value)})
    return out


def counters(report: Mapping[str, Any]) -> dict[str, Any]:
    """Shipped-crop counters (nine-bit in the ROM: 496..511 is the full 31 points)."""
    shipped = (report.get("end_economy") or {}).get("shipped") or {}
    detail = _ending(report)["oracle"].get("crop_ship_details") or {}
    names = {"corn": "corn", "tomato": "tomatoes", "potato": "potatoes", "turnip": "turnips"}
    rows = {}
    for key, oracle_key in names.items():
        count = shipped.get(key)
        rows[key] = {"shipped": count, "points": detail.get(oracle_key),
                     "wrapped": None if count is None else int(count) > 511}
    return rows


def terrain_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-morning terrain ledger from the day journal: season starts plus the last morning."""
    raws = [raw for raw in report.get("day_journal") or [] if isinstance(raw, Mapping)]
    rows = y1_bench.day_rows(report)
    picked: list[dict[str, Any]] = []
    for row, raw in zip(rows, raws):
        ledger = (raw.get("development_census") or {}).get("ledger")
        if not ledger:
            continue
        picked.append({"label": row["label"], "first": row["day"] == 1, **ledger})
    if not picked:
        return []
    keep = [item for item in picked[:-1] if item["first"]] + [picked[-1]]
    return keep


def render_terrain(report: Mapping[str, Any]) -> list[str]:
    rows = terrain_rows(report)
    if not rows:
        return []
    last = rows[-1]
    lines = [
        "## Farm development (terrain worked)", "",
        f"The ROM counts a tile below 0xA0 that is fence or at least 0x1D, and scores `min(100, counted * 10 // 295) >> 1`. "
        f"Hoed soil does not count: it only takes the seed. 50 points need **{_n(last['need'])}** counted tiles, "
        f"{last['required_pct']}% of the {_n(last['pool'])}-cell field, so at most {_n(last['slack'])} cells may stay bare "
        "and every other cell is hoed, then sown. A bag sows a 3x3 (nine tiles at best, 500G).", "",
        "| Morning | Worked | Counted | Pts | To go | Hoe left | Bags (min / plan) | Plan cost | Unhoed in plan | Full 3x3 grass | planted_grass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['label']} | {item['worked_pct']}% | {_n(item['counted'])} | {item['points']} | {_n(item['to_go'])} "
            f"| {_n(item['hoe_swings_left'])} | {_n(item['bags_min'])} / {_n(item['bags_plan'])} | {_g(item['bag_cost_g'])} "
            f"| {_n(item['plan_unhoed'])} | {_n(item['full_grass_blocks'])} | {_n(item['planted_grass'])} |"
        )
    lines.append("")
    return lines


def render_score_history(report: Mapping[str, Any]) -> list[str]:
    """Expose every score window from measured morning ledgers, including partial pin runs."""
    ledgers = [row["score_ledger"] for row in report.get("day_journal") or []
               if isinstance(row, Mapping) and isinstance(row.get("score_ledger"), Mapping)
               and row["score_ledger"].get("terms")]
    if not ledgers:
        return []
    first, last = ledgers[0]["terms"], ledgers[-1]["terms"]
    lines = ["## Score windows across the replay", "",
             "First and last recorded mornings after sleep; the first is not the loaded save. "
             "These RAM-model points are separate from the ROM ending score above.", "",
             "| Term | First | Last | Change | Raw last | Full-score window | Gap |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for term, row in last.items():
        old = (first.get(term) or {}).get("value")
        new = row.get("value")
        delta = "n/a" if old is None or new is None else f"{int(new) - int(old):+d}"
        window = f"{_n(row.get('lo'))}–{_n(row.get('hi'))}"
        lines.append(f"| {term} | {_n(old)} | {_n(new)} | {delta} | {_n(row.get('raw'))} | {window} | {_n(row.get('gap'))} |")
    return [*lines, ""]


def render_idle(report: Mapping[str, Any]) -> list[str]:
    """Frames the run asked for nothing and nothing moved (see WorkFrameLedger)."""
    metrics = report.get("work_metrics") or {}
    idle = metrics.get("idle")
    if not idle:
        return []
    total = idle["input_frames"] + idle["idle_frames"]
    runs = idle["dead_runs"]
    floor = idle["dead_run_min_frames"]
    lines = ["## Idle frames", "",
             "Counted before every emulator step. *Dead* means no button held and the map and pixel "
             f"position equal the previous frame's. Runs of {floor}+ frames are listed.", "",
             "| Frames | Button held | No button | Dead | Dead runs | Dead in runs | Longest run |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| {_n(total)} | {_n(idle['input_frames'])} | {_n(idle['idle_frames'])} | {_n(idle['dead_frames'])} | "
             f"{_n(len(runs))} | {_n(sum(r['frames'] for r in runs))} | "
             f"{_n(max((r['frames'] for r in runs), default=0))} |", ""]
    seasons = ("Sp", "Su", "Fa", "Wi")
    by_phase: dict[str, list[int]] = {}
    for r in runs:
        row = by_phase.setdefault(r["phase"], [0, 0])
        row[0] += 1
        row[1] += r["frames"]
    if by_phase:
        lines += ["| Phase | Dead runs | Frames |", "| --- | ---: | ---: |"]
        for phase, (count, frames) in sorted(by_phase.items(), key=lambda kv: -kv[1][1])[:10]:
            lines.append(f"| {phase} | {_n(count)} | {_n(frames)} |")
        lines += ["", "Longest runs:", ""]
        for r in sorted(runs, key=lambda r: -r["frames"])[:8]:
            y, se, d = r["date"]
            lines.append(f"- {_n(r['frames'])} frames ({r['frames'] / 60:.1f} s) in {r['phase']}, "
                         f"Y{y + 1} {seasons[se % 4]} D{d}, map {r['tilemap']:#04x}, from frame {_n(r['start_frame'])}")
        lines.append("")
    return lines


def render(report: Mapping[str, Any], name: str) -> str:
    h = headline(report)
    ending = _ending(report)
    oracle = ending["oracle"]
    lines: list[str] = [f"# Final scorecard: {name}", ""]
    reasons = f" ({'; '.join(h['verdict_reasons'])})" if h["verdict_reasons"] else ""
    lines += [f"**{h['verdict']}**{reasons}. Ended {h['end_date']}.", ""]
    lines += [
        "| Headline | Value |", "| --- | ---: |",
        f"| Strategy | {h['strategy']} |",
        f"| Total cash (wallet) | **{_g(h['wallet'])}** |",
        f"| Unpaid bin at the credits | {_g(h['closing_bin_gold'])} |",
        f"| Cash plus bin | {_g(h['wallet_with_bin'])} |",
        f"| Net worth (cash + 5,000G a cow + 1,000G a hen) | {_g(h['net_worth'])} |",
        f"| Ranch score (ROM, ending index 0x25) | **{_n(h['ranch_score'])} / 999** ({h['ranch_band']}) |",
        f"| Cows / hens | {_n(h['cows'])} / {_n(h['hens'])} |",
        "",
    ]
    if oracle:
        lines += ["## Ranch score breakdown", "", "| Term | Points | Max | Gap |", "| --- | ---: | ---: | ---: |"]
        for row in breakdown(report):
            lines.append(f"| {row['term']} | {_n(row['points'])} | {row['max']} | {_n(row['gap'])} |")
        total = oracle.get("total_score")
        lines += [f"| **Total** (capped at 999) | **{_n(total)}** | 999 | {_n(None if total is None else 999 - int(total))} |", ""]
        details = oracle.get("bachelorette_details") or {}
        if details:
            lines += ["Hearts: " + ", ".join(f"{k} {v}" for k, v in details.items())
                      + ". Cow happiness by slot: " + ", ".join(str(v) for v in oracle.get("cow_happiness_details") or []) + ".", ""]
        missing = oracle.get("missing_milestones") or []
        if missing:
            lines += ["Short of full marks:", *[f"- {item}" for item in missing], ""]
    else:
        lines += ["No ending breakdown in the report (the run did not reach the score screen).", ""]
    lines += render_score_history(report)
    lines += render_terrain(report)
    crops = counters(report)
    lines += ["## Shipped crops", "",
              "The ROM counter is nine bits: one point per 16 shipped, 31 points at 496 to 511, "
              "and a count past 511 wraps back to 0.", "",
              "| Crop | Shipped | Points | Past 511 |", "| --- | ---: | ---: | :---: |"]
    for key, row in crops.items():
        lines.append(f"| {key} | {_n(row['shipped'])} | {_n(row['points'])} | {'WRAPPED' if row['wrapped'] else 'no'} |")
    lines.append("")
    shipped = {key: row["shipped"] for key, row in crops.items()}
    lost = run_strategy.points_lost_to_wrap({k: v for k, v in shipped.items() if v is not None})
    lines += [f"Points lost to counters that wrapped past 511: **{lost}** "
              f"(the score strategy stops each crop at 496, 15 below the wrap, for 31 points).", ""]
    if oracle:
        lines += ["## Where the missing points are", "",
                  "| Term | Gap | Strategy | Lever |", "| --- | ---: | --- | --- |"]
        for row in breakdown(report):
            if row["gap"]:
                who, lever = LEVERS.get(row["term"], ("-", ""))
                lines.append(f"| {row['term']} | {row['gap']} | {who} | {lever} |")
        lines.append("")
    lines += [
        "## Run cost", "",
        "| Frames | Wall | Game days | Rescues | Bad days | Day failures |", "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {_n(h['frames'])} | {_n(h['wall_seconds'])} s | {_n(h['days'])} | {_n(h['rescues'])} | "
        f"{_n(h['bad_days'])} | {_n(h['day_failures'])} |", "",
    ]
    lines += render_idle(report)
    audit = report.get("clean_run") or {}
    lines += [f"Intervention: {audit.get('intervention_class', 'n/a')}, initial loads {audit.get('initial_state_loads', 'n/a')}, "
              f"mid-run loads {audit.get('mid_run_state_loads', 'n/a')}, RAM writes {audit.get('ram_writes', 'n/a')}, "
              f"assists {audit.get('assists') or 'none'}.", ""]
    video = report.get("video") or {}
    if video:
        lines += [f"Video: {len(video.get('parts') or [])} part(s), {video.get('frames', 'n/a')} frames "
                  f"({_n(video.get('duration_seconds'))} s at {video.get('fps', 'n/a')} fps), first part `{video.get('path', 'n/a')}`.", ""]
    card = y1_bench.scorecard(dict(report), name)
    lines += ["## Sales by season", "", "```text", y1_bench.render_sales(card).rstrip("\n"), "```", ""]
    return "\n".join(lines)


def compare(first: Mapping[str, Any], second: Mapping[str, Any], names: tuple[str, str]) -> str:
    a, b = headline(first), headline(second)
    keys = ("wallet", "net_worth", "ranch_score", "cows", "hens", "frames", "rescues", "bad_days")
    out = [f"## B ({names[1]}) minus A ({names[0]})", "", "| Metric | A | B | B - A |", "| --- | ---: | ---: | ---: |"]
    for key in keys:
        x, y = a[key], b[key]
        delta = "n/a" if x is None or y is None else f"{int(y) - int(x):+,}"
        out.append(f"| {key} | {_n(x)} | {_n(y)} | {delta} |")
    ta = {row["term"]: row["points"] for row in breakdown(first)}
    tb = {row["term"]: row["points"] for row in breakdown(second)}
    out += ["", "| Score term | A | B | B - A |", "| --- | ---: | ---: | ---: |"]
    for term, _, _ in ((label, 0, 0) for _, label, _ in TERMS):
        x, y = ta.get(term), tb.get(term)
        delta = "n/a" if x is None or y is None else f"{int(y) - int(x):+,}"
        out.append(f"| {term} | {_n(x)} | {_n(y)} | {delta} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final scorecard from run_to_day2 report(s).")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = parser.parse_args(argv)
    if len(args.reports) > 2:
        parser.error("give one report or two")
    loaded = [y1_bench.load_report(path) for path in args.reports]
    names = tuple(path.parent.name if path.stem == "report" else path.stem for path in args.reports)
    text = "\n".join(render(report, name) for report, name in zip(loaded, names))
    if len(loaded) == 2:
        text += "\n" + compare(loaded[0], loaded[1], (names[0], names[1]))
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
