"""Year-1 benchmark scorecard from a ``run_to_day2`` report JSON.

``python -m harvest.tools.y1_bench report.json [other.json] [--out card.md|card.json]``

One report prints one scorecard. A second report prints side-by-side numbers
and the delta (second minus first). The input is the report, never the log.
Every field a report lacks prints ``n/a``: older reports have no ``economy``
or ``eod`` journal fields, so chickens, brush, milker and bin gold go ``n/a``
there while cows (the journal's ``cows`` rows) and purchases (phase reasons)
still resolve.

Reading rules worth knowing before trusting a number:

* Journal ``money`` is the wallet on the morning after the day (the night
  payout is in). ``shipping_money`` is read after the payout and is always 0,
  so bin gold comes from ``eod.bin_gold``, the pre-sleep read.
* Animal gold is ``bin gold - crop gold - forage gold`` on days with cow or
  coop chores. Crop gold is the lifetime crop counters' delta times the ship
  price. Forage gold is the ``shipping_deposit`` of mushroom and grape phases.
  The residual is milk and eggs (tier and count unknown), so it is an estimate.
* ``planted`` sums ``establish_planted`` and ``planted=`` in phase reasons. The
  establish phase reports "sequence complete", so it is usually 0: ``n/a``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from harvest.core.shipping_credit import SHIPPED_CROP_FIELDS
from harvest.planner.livestock_econ import CHICKEN_PRICE, CHICKEN_SALE_PRICE, COW_PRICE, SHIP_PRICE_G
from harvest.tools.pr999_report import SEASONS, calendar_index

NA = "n/a"
TOP_FAILURES = 10
SEASON_TAGS = tuple(name[:2] for name in SEASONS)  # Sp Su Fa Wi

_ANIMAL_BUY = re.compile(r"^(cow|chicken) (\d+)->(\d+) money (\d+)->(\d+)")
_SEED_BUY = re.compile(r"^bought (\w+?)_seeds (\d+)->(\d+) money (\d+)->(\d+)")
_COW_CHORES = re.compile(r"milk=(\d+) ship=(\d+) fed=(\d+)")
_COOP_CHORES = re.compile(r"fed=(\d+) egg=(\w+) ship=(\w+)")
_HEN_SALE = re.compile(r"chickens (\d+)->(\d+) ")
_PLANTED = re.compile(r"planted=(\d+)")
_END_DATE = re.compile(r"^Y(\d+)S(\d+)D(\d+)$")


def _int(value: Any) -> int | None:
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _num(value: Any) -> str:
    number = _int(value)
    return NA if number is None else f"{number:,}"


def _signed(value: Any) -> str:
    number = _int(value)
    return NA if number is None else f"{number:+,}"


def _label(year: int, season: int, day: int) -> str:
    tag = SEASON_TAGS[season] if 0 <= season < len(SEASON_TAGS) else str(season)
    return f"Y{year + 1} {tag} D{day}"


def _sum(values: Sequence[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


# --- per-day rows ---------------------------------------------------------


def _snapshot(row: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    snap = row.get(key)
    return snap if isinstance(snap, Mapping) else None


def _cow_rows_owned(row: Mapping[str, Any]) -> int | None:
    cows = row.get("cows")
    if not isinstance(cows, list):
        return None
    return sum(1 for cow in cows if isinstance(cow, Mapping) and cow.get("exists"))


def _phase_reasons(row: Mapping[str, Any], phase: str) -> list[str]:
    return [
        str(pr.get("reason") or "")
        for pr in row.get("phase_results") or []
        if pr.get("phase") == phase and pr.get("status") == "success"
    ]


def _purchases(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for pr in row.get("phase_results") or []:
        if pr.get("status") != "success":
            continue
        reason = str(pr.get("reason") or "")
        animal = _ANIMAL_BUY.match(reason)
        seed = _SEED_BUY.match(reason)
        if animal:
            kind, before, after, wallet0, wallet1 = animal.groups()
            found.append({"kind": kind, "before": int(before), "after": int(after),
                          "spent": int(wallet0) - int(wallet1)})
        elif seed:
            kind, before, after, wallet0, wallet1 = seed.groups()
            found.append({"kind": f"{kind}_seeds", "before": int(before), "after": int(after),
                          "spent": int(wallet0) - int(wallet1)})
    return found


def _hens_sold(row: Mapping[str, Any]) -> int:
    """Hens the day's sale events took.

    The event succeeds on a drop it saw, or on a request flag the ROM already
    cleared on the walk home, which reads ``n->n``: one hen either way.
    """
    sold = 0
    for reason in _phase_reasons(row, "SELL_CHICKEN"):
        match = _HEN_SALE.search(reason)
        sold += max(1, int(match.group(1)) - int(match.group(2))) if match else 1
    return sold


def _planted(row: Mapping[str, Any]) -> int:
    total = _int(row.get("establish_planted")) or 0
    for pr in row.get("phase_results") or []:
        if pr.get("status") == "success" and pr.get("phase") in {"CROP_ESTABLISH", "CROP_WATER"}:
            match = _PLANTED.search(str(pr.get("reason") or ""))
            total += int(match.group(1)) if match else 0
    return total


def _forage_gold(row: Mapping[str, Any]) -> int:
    """Gold the day's forage phases (mushroom, grape) put in the bin."""

    total = 0
    for pr in row.get("phase_results") or []:
        deposit = pr.get("shipping_deposit")
        if isinstance(deposit, Mapping):
            total += max(0, (_int(deposit.get("shipping_money_after")) or 0)
                         - (_int(deposit.get("shipping_money_before")) or 0))
    return total


def _crop_counts(prev: Mapping[str, Any] | None, now: Mapping[str, Any] | None) -> dict[str, int] | None:
    if not prev or not now:
        return None
    before, after = prev.get("shipped"), now.get("shipped")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if any(_int(snap.get(kind)) is None for snap in (before, after) for kind in SHIPPED_CROP_FIELDS):
        return None
    counts = {kind: int(after[kind]) - int(before[kind]) for kind in SHIPPED_CROP_FIELDS}
    return counts if all(value >= 0 for value in counts.values()) else None


def _crop_gold(prev: Mapping[str, Any] | None, now: Mapping[str, Any] | None) -> int | None:
    """Gold of the crops dropped between two morning reads (lifetime counters)."""
    counts = _crop_counts(prev, now)
    return None if counts is None else sum(n * SHIP_PRICE_G[kind] for kind, n in counts.items())


def day_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One derived row per journal day. Years count season wraparounds."""

    start = report.get("start") if isinstance(report.get("start"), Mapping) else {}
    year = _int(start.get("year")) or 0
    previous_season: int | None = None
    previous_eco = _snapshot(report, "start_economy")
    flock_log = _int(start.get("num_chickens"))
    rows: list[dict[str, Any]] = []
    for raw in report.get("day_journal") or []:
        if not isinstance(raw, Mapping):
            continue
        season = _int(raw.get("plan_season")) or 0
        day = _int(raw.get("plan_day")) or 1
        if previous_season is not None and season < previous_season:
            year += 1
        previous_season = season
        eco, eod = _snapshot(raw, "economy"), _snapshot(raw, "eod")
        end_season = _int(raw.get("end_season"))
        end_year = year + 1 if end_season is not None and end_season < season else year
        gates = _snapshot(raw, "gates")
        end_of_day = eod or eco
        cow_chores = _phase_reasons(raw, "COW_CHORES")
        coop_chores = _phase_reasons(raw, "COOP_CHORES")
        cow_match = [m for m in map(_COW_CHORES.search, cow_chores) if m]
        coop_match = [m for m in map(_COOP_CHORES.search, coop_chores) if m]
        cows = _first(end_of_day and end_of_day.get("cows"), _cow_rows_owned(raw))
        hens = end_of_day.get("hens") if end_of_day else None
        crop_gold = _crop_gold(previous_eco, eco)
        for buy in _purchases(raw):
            if buy["kind"] == "chicken":
                flock_log = buy["after"]
        forage_gold = _forage_gold(raw)
        bin_gold = _int(eod.get("bin_gold")) if eod else None
        animal_day = bool(cow_chores or coop_chores)
        rows.append({
            "year": year, "season": season, "day": day,
            "label": _label(year, season, day),
            "index": calendar_index(year, season, day),
            "money": _int(raw.get("money")),
            "harvested": _int(raw.get("harvested_count")) or 0,
            "shipped": _int(raw.get("shipped_count")) or 0,
            "planted": _planted(raw),
            "bin_gold": bin_gold,
            "crop_gold": crop_gold,
            "crop_shipped": _crop_counts(previous_eco, eco),
            "forage_gold": forage_gold,
            "animal_gold": (bin_gold - crop_gold - forage_gold
                            if None not in (bin_gold, crop_gold) and animal_day else None),
            "animal_day": animal_day,
            "coop_day": bool(coop_chores),
            "milk_shipped": sum(int(m.group(2)) for m in cow_match),
            "egg_ship_day": any(m.group(3) == "True" for m in coop_match),
            "hens_sold": _hens_sold(raw),
            "cows": _int(cows),
            "chickens": _int(end_of_day.get("chickens")) if end_of_day else None,
            "hens": _int(hens),
            "morning_cows": _int(_first(eco and eco.get("cows"), _cow_rows_owned(raw))),
            "morning_chickens": _int(eco.get("chickens")) if eco else None,
            "planted_grass_morning": _int(eco.get("planted_grass")) if eco else None,
            "hay_morning": _int(eco.get("hay")) if eco else None,
            "flock_log": flock_log,
            "brush_used": any("brush=True" in text for text in cow_chores),
            "milk_made": sum(int(m.group(1)) for m in cow_match),
            "cows_fed": _int(eod.get("fed_cows")) if eod else (max((int(m.group(3)) for m in cow_match), default=0)),
            "hens_fed": _int(eod.get("fed_chickens")) if eod else (max((int(m.group(1)) for m in coop_match), default=0)),
            "brush": eco.get("brush") if eco else None,
            "milker": eco.get("milker") if eco else None,
            "bad_day": bool(raw.get("bad_day")),
            "floor_reason": str(raw.get("floor_reason") or ""),
            "rescues": _int(raw.get("rescues")) or 0,
            "purchases": _purchases(raw),
            "phase_results": raw.get("phase_results") or [],
            "gates": gates,
            "gate_label": _label(end_year, _first(end_season, season), _int(raw.get("end_day")) or day),
        })
        previous_eco = eco
    return rows


# --- sections -------------------------------------------------------------


def integrity(report: Mapping[str, Any]) -> dict[str, Any]:
    clean = report.get("clean_run") if isinstance(report.get("clean_run"), Mapping) else {}
    power_on = isinstance(report.get("power_on"), Mapping)
    state = report.get("state")
    loads0, loads1 = _int(clean.get("initial_state_loads")), _int(clean.get("mid_run_state_loads"))
    writes, assists = _int(clean.get("ram_writes")), clean.get("assists")
    problems = []
    if not clean:
        problems.append("no clean_run audit")
    if clean.get("intervention_class") not in (None, "Clean"):
        problems.append(f"class {clean.get('intervention_class')}")
    if loads1:
        problems.append(f"{loads1} mid-run load(s)")
    if writes:
        problems.append(f"{writes} RAM write(s)")
    if assists:
        problems.append(f"assists {dict(assists)}")
    if not power_on and clean:
        problems.append(f"pin start {state}" if state else "no power-on record")
    elif power_on and loads0:
        problems.append(f"{loads0} initial load(s) on power-on")
    return {
        "start": "power-on" if power_on else (f"pin {state}" if state else NA),
        "power_on": power_on,
        "intervention_class": clean.get("intervention_class"),
        "initial_state_loads": loads0, "mid_run_state_loads": loads1,
        "ram_writes": writes, "assists": dict(assists) if isinstance(assists, Mapping) else assists,
        "success": report.get("success"), "terminal": report.get("terminal"),
        "reason": report.get("reason"), "end_date": report.get("end_date"),
        "verdict": "CLEAN" if not problems else "; ".join(problems),
        "comparable": power_on and not problems and bool(report.get("success")),
    }


def cost(report: Mapping[str, Any]) -> dict[str, Any]:
    start = report.get("start") if isinstance(report.get("start"), Mapping) else {}
    end = report.get("end") if isinstance(report.get("end"), Mapping) else {}
    days = None
    if all(_int(d.get(k)) is not None for d in (start, end) for k in ("year", "season", "day")):
        days = (calendar_index(_int(end["year"]), _int(end["season"]), _int(end["day"]))
                - calendar_index(_int(start["year"]), _int(start["season"]), _int(start["day"])))
    frames, planner = _int(report.get("frames")), _int(report.get("planner_frames"))
    wall = report.get("wall_seconds")
    return {
        "frames": frames, "planner_frames": planner, "wall_seconds": wall, "game_days": days,
        "frames_per_day": round(frames / days) if frames is not None and days else None,
        "planner_frames_per_day": round(planner / days) if planner is not None and days else None,
        "wall_per_day": round(float(wall) / days, 1) if wall is not None and days else None,
    }


def headline(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = report.get("start") if isinstance(report.get("start"), Mapping) else {}
    end = report.get("end") if isinstance(report.get("end"), Mapping) else {}
    last = rows[-1] if rows else {}
    wallet = _int(_first(end.get("money"), last.get("money")))
    cows = _int(_first(end.get("num_cows"), last.get("morning_cows")))
    chickens = _int(_first(end.get("num_chickens"), last.get("morning_chickens")))
    parts = {
        "wallet": wallet,
        "chickens": chickens, "chicken_gold": None if chickens is None else chickens * CHICKEN_PRICE,
        "cows": cows, "cow_gold": None if cows is None else cows * COW_PRICE,
    }
    known = None not in (wallet, parts["chicken_gold"], parts["cow_gold"])
    parts["net_worth"] = wallet + parts["chicken_gold"] + parts["cow_gold"] if known else None
    # The headline counts a hen at her 1,000G price; the shop buys her back for 500G.
    parts["resale_net_worth"] = (
        wallet + chickens * CHICKEN_SALE_PRICE + parts["cow_gold"] if known else None
    )
    year_end = _int(end.get("year"))
    if (_int(start.get("year")) or 0) >= 1:
        state = "NOT A Y1 RUN"
    else:
        state = "COMPLETE" if year_end is not None and year_end >= 1 else "PARTIAL"
    found = _END_DATE.match(str(report.get("end_date") or ""))
    parts["as_of"] = _label(int(found[1]) - 1, int(found[2]), int(found[3])) if found else None
    parts["year_state"] = state
    parts["year_complete"] = state == "COMPLETE"
    parts["start_wallet"] = _int(start.get("money"))
    parts["planted_grass"] = _int(_first(end.get("planted_grass"), last.get("planted_grass_morning")))
    parts["hay"] = _int(_first(end.get("stored_grass"), last.get("hay_morning")))
    return parts


def seasons(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = report.get("start") if isinstance(report.get("start"), Mapping) else {}
    ledger = _snapshot(report, "sales_ledger") or {}
    observed = {(item["year"], item["season"]): item["bin_gold"]
                for item in ledger.get("seasons", []) if ledger.get("source") == "per_frame_ram"}
    prior = _int(start.get("money"))
    out: list[dict[str, Any]] = []
    keys = list(dict.fromkeys((row["year"], row["season"]) for row in rows))
    for year, season in keys:
        chunk = [row for row in rows if (row["year"], row["season"]) == (year, season)]
        last = chunk[-1]
        harvested = sum(row["harvested"] for row in chunk)
        planted = sum(row["planted"] for row in chunk)
        wallet = last["money"]
        out.append({
            "label": _label(year, season, 1).rsplit(" ", 1)[0],
            "days": len(chunk),
            "wallet": wallet,
            "gold": None if wallet is None or prior is None else wallet - prior,
            "planted": None if planted == 0 and harvested > 0 else planted,
            "harvested": harvested,
            "shipped": sum(row["shipped"] for row in chunk),
            "bin_gold": observed.get((year, season), _sum([row["bin_gold"] for row in chunk])),
            "animal_gold": _sum([row["animal_gold"] for row in chunk]),
            "cows": last["morning_cows"],
            "chickens": _first(last["morning_chickens"], last["flock_log"]),
            "chickens_from_log": last["morning_chickens"] is None and last["flock_log"] is not None,
            "bad_days": sum(1 for row in chunk if row["bad_day"]),
        })
        prior = wallet if wallet is not None else prior
    return out


def _first_label(rows: list[dict[str, Any]], pick) -> dict[str, Any] | None:
    for row in rows:
        if pick(row):
            return {"label": row["label"], "index": row["index"]}
    return None


def animals(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = report.get("start") if isinstance(report.get("start"), Mapping) else {}

    def bought(kind: str) -> list[dict[str, Any]]:
        return [
            {"label": row["label"], "index": row["index"], "before": buy["before"],
             "after": buy["after"], "spent": buy["spent"]}
            for row in rows for buy in row["purchases"]
            if buy["kind"] == kind and buy["after"] > buy["before"]
        ]

    def census_first(field: str, start_key: str) -> dict[str, Any] | None:
        """No purchase receipt: first census morning with the animal, if none at the start."""
        if _int(start.get(start_key)):
            return None
        found = _first_label(rows, lambda row: (row[field] or 0) > 0)
        return None if found is None else {**found, "source": "census"}

    cow_buys, hen_buys = bought("cow"), bought("chicken")
    in_coop = _first_label(rows, lambda row: row["coop_day"])
    animal_rows = [row for row in rows if row["animal_day"]]
    milk = sum(row["milk_shipped"] for row in rows)
    gold_rows = [row["animal_gold"] for row in animal_rows]
    return {
        "cows_at_start": _int(start.get("num_cows")),
        "chickens_at_start": _int(start.get("num_chickens")),
        "cow_buys": cow_buys, "hen_buys": hen_buys,
        "first_cow": _first(cow_buys[0] if cow_buys else None, census_first("morning_cows", "num_cows")),
        "first_hen": _first(hen_buys[0] if hen_buys else None, census_first("morning_chickens", "num_chickens")),
        "first_in_coop": in_coop,
        "brush": _first_label(rows, lambda row: row["brush"]),
        "milker": _first_label(rows, lambda row: row["milker"]),
        "brush_first_used": _first_label(rows, lambda row: row["brush_used"]),
        "milker_first_used": _first_label(rows, lambda row: row["milk_made"] > 0),
        "economy_fields": any(row["brush"] is not None for row in rows),
        "milk_shipped": milk,
        "egg_ship_days": sum(1 for row in rows if row["egg_ship_day"]),
        "hens_sold": sum(row["hens_sold"] for row in rows),
        "first_hen_sale": _first_label(rows, lambda row: row["hens_sold"] > 0),
        "animal_gold_est": _sum(gold_rows),
        "animal_gold_days": sum(1 for value in gold_rows if value is not None),
        "animal_days": len(animal_rows),
        "bin_gold_anomalies": sum(1 for value in gold_rows if value is not None and value < 0),
        "forage_gold": sum(row["forage_gold"] for row in rows),
        "unexplained_bin_gold": _sum([
            None if row["animal_day"] or row["bin_gold"] is None or row["crop_gold"] is None
            else row["bin_gold"] - row["crop_gold"] - row["forage_gold"] for row in rows
        ]),
    }


def gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Why the animals and tools waited: ``livestock_econ.ranch_gates`` on each morning read."""

    read = [row for row in rows if row["gates"] and "cow" in row["gates"]]
    cow = Counter(str(row["gates"]["cow"]) for row in read if row["gates"]["cow"])
    hen = Counter(str(row["gates"]["hen"]) for row in read if row["gates"].get("hen"))
    cow_open = next((row["gate_label"] for row in read if row["gates"]["cow"] is None), None)
    trip = next((row["gate_label"] for row in read if row["gates"].get("tool_trip")), None)
    return {"mornings": len(read), "cow_blocks": cow.most_common(), "hen_blocks": hen.most_common(),
            "cow_open": cow_open, "tool_trip": trip}


def care(rows: list[dict[str, Any]]) -> dict[str, Any]:
    owned = [row for row in rows if (row["cows"] or 0) + (row["hens"] or 0) > 0]
    fed = [
        row for row in owned
        if (row["cows_fed"] or 0) >= (row["cows"] or 0)
        and (row["hens"] is None or (row["hens_fed"] or 0) >= row["hens"])
    ]
    fed_ids = {id(row) for row in fed}
    return {
        "animal_days": len(owned), "fully_fed_days": len(fed),
        "pct": round(100 * len(fed) / len(owned), 1) if owned else None,
        "hens_known": any(row["hens"] is not None for row in rows),
        "unfed_days": [row["label"] for row in owned if id(row) not in fed_ids][:8],
    }


def reliability(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    for row in rows:
        for pr in row["phase_results"]:
            if pr.get("status") in {"failure", "blocked"}:
                failures[str(pr.get("phase", "?"))] += 1
            elif pr.get("status") == "skipped":
                skips[str(pr.get("phase", "?"))] += 1
    kinds = Counter(
        f"{fail.get('phase', '?')}/{fail.get('status', '?')}"
        for fail in report.get("day_failures") or [] if isinstance(fail, Mapping)
    )
    if not rows:
        summary = report.get("journal_summary")
        failures.update((summary or {}).get("phase_failure_counts") or {})
    floor = report.get("floor_report") if isinstance(report.get("floor_report"), Mapping) else {}
    trips = [row for row in rows if row["floor_reason"]]
    return {
        "bad_days": sum(1 for row in rows if row["bad_day"]),
        "floor_report_bad_days": _int(floor.get("total_bad_days")),
        "floor_trips": len(trips), "floor_trip_days": [row["label"] for row in trips][:8],
        "rescues": sum(row["rescues"] for row in rows),
        "day_failures": len(report.get("day_failures") or []),
        "phase_failures_total": sum(failures.values()),
        "phase_failures_top": failures.most_common(TOP_FAILURES),
        "phase_skips_total": sum(skips.values()),
        "phase_skips_top": skips.most_common(5),
        "day_failure_kinds": kinds.most_common(),
    }


def spending(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bags: Counter[str] = Counter()
    gold: Counter[str] = Counter()
    for row in rows:
        for buy in row["purchases"]:
            if buy["kind"].endswith("_seeds"):
                bags[buy["kind"][:-6]] += max(0, buy["after"] - buy["before"])
                gold[buy["kind"][:-6]] += buy["spent"]
    return {"seed_bags": dict(bags), "seed_gold": dict(gold), "seed_gold_total": sum(gold.values())}


def completion(report: Mapping[str, Any]) -> dict[str, Any]:
    """Credits with a score and Perfect are separate checks."""
    ending = _snapshot(report, "ending") or {}
    final = _snapshot(ending, "final") or {}
    audit = _snapshot(report, "clean_run") or {}
    power_on = _snapshot(report, "power_on") or {}
    credits = bool(report.get("success") and report.get("terminal") and ending.get("finished")
                   and final.get("terminal") and final.get("scene_index") == 0x25)
    score = _int(final.get("ranch_mastery_0x25"))
    clean = (power_on.get("completed") is True
             and all(_int(audit.get(key)) == 0 for key in
                     ("initial_state_loads", "mid_run_state_loads", "ram_writes"))
             and audit.get("assists") == {} and audit.get("intervention_class") == "Clean")
    return {"credits_complete": credits, "score": score, "clean_power_on": clean,
            "power_on_credits_with_score": credits and clean and score is not None,
            "perfect_999": credits and clean and score == 999}


def sales(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Seasonal gross shipping, receipt sales, and wallet reconciliation.

    Per-frame RAM beats sparse end-of-day snapshots. Legacy reports retain
    observed subtotals and coverage; they cannot claim complete sales totals.
    Milk/egg gold remains a residual, never a measured split of the two.
    """
    ledger = _snapshot(report, "sales_ledger") or {}
    measured = ledger.get("source") == "per_frame_ram" and bool(ledger.get("observations"))
    live = {(item["year"], item["season"]): item for item in ledger.get("seasons", [])}
    keys = sorted({*((row["year"], row["season"]) for row in rows), *live})
    first, last = _snapshot(report, "start") or {}, _snapshot(report, "end") or {}
    bounds_known = all(_int(snap.get(key)) is not None for snap in (first, last) for key in ("year", "season"))
    excluded = []
    if bounds_known:
        lo, hi = (first["year"], first["season"]), (last["year"], last["season"])
        excluded = [live[key] for key in keys if key in live and not lo <= key <= hi]
        keys = [key for key in keys if lo <= key <= hi]
    prior = _int((_snapshot(report, "start") or {}).get("money"))
    closing_bin = _int((_snapshot(report, "end_economy") or {}).get("bin_gold"))
    out = []
    for index, (year, season) in enumerate(keys):
        chunk = [row for row in rows if (row["year"], row["season"]) == (year, season)]
        observed = live.get((year, season)) if measured else None
        counts = (dict(observed["crop_shipped"]) if observed else {
            kind: _sum([row["crop_shipped"][kind] if row["crop_shipped"] else None for row in chunk])
            for kind in SHIPPED_CROP_FIELDS})
        crop_gold = {kind: None if n is None else n * SHIP_PRICE_G[kind] for kind, n in counts.items()}
        crops = _sum(list(crop_gold.values()))
        bin_gold = observed["bin_gold"] if observed else _sum([row["bin_gold"] for row in chunk])
        forage = sum(row["forage_gold"] for row in chunk)
        animal = _sum([row["animal_gold"] for row in chunk])
        if not any(row["animal_day"] for row in chunk):
            animal = 0
        hens = sum(row["hens_sold"] for row in chunk)
        gross = None if bin_gold is None else bin_gold + hens * CHICKEN_SALE_PRICE
        unpaid = closing_bin if index == len(keys) - 1 else 0
        paid = None if None in (bin_gold, closing_bin) else max(0, bin_gold - unpaid) + hens * CHICKEN_SALE_PRICE
        bought = sum(buy["spent"] for row in chunk for buy in row["purchases"])
        wallet = chunk[-1]["money"] if chunk else None
        if index == len(keys) - 1:
            wallet = _first((_snapshot(report, "end") or {}).get("money"), wallet)
        delta = None if None in (wallet, prior) else wallet - prior
        out.append({
            "year": year, "season": season, "label": f"Y{year + 1} {SEASONS[season]}",
            "source": "frame ledger" if observed else "journal subtotal",
            "bin_days": sum(row["bin_gold"] is not None for row in chunk), "journal_days": len(chunk),
            "crop_shipped": counts, "crop_gold": crop_gold, "crop_gold_total": crops,
            "bin_gold": bin_gold, "forage_gold_receipts": forage,
            "milk_egg_gold_est": animal,
            "unattributed_bin_gold": (None if None in (bin_gold, crops, animal)
                                      else bin_gold - crops - forage - animal),
            "hens_sold_receipts": hens, "hen_sales_gold_receipts": hens * CHICKEN_SALE_PRICE,
            "gross_sales_gold": gross, "purchases_gold_receipts": bought,
            "paid_sales_gold": paid,
            "wallet_start": prior, "wallet_end": wallet, "wallet_delta": delta,
            "other_net_cash": None if None in (delta, gross) else delta - gross + bought,
        })
        prior = wallet
    endpoints_match = all(
        all((_snapshot(report, endpoint) or {}).get(key) == (_snapshot(ledger, endpoint) or {}).get(key)
            for key in ("year", "season", "day"))
        for endpoint in ("start", "end")
    )
    bin_total = _sum([item["bin_gold"] for item in out])
    hen_total = sum(item["hen_sales_gold_receipts"] for item in out)
    paid_bin = None if None in (bin_total, closing_bin) else max(0, bin_total - closing_bin)
    outside_sales = any(item["bin_gold"] or any(item["crop_shipped"].values()) for item in excluded)
    return {"seasons": out, "counter_decreases": ledger.get("counter_decreases", []),
            "excluded_calendar_rows": excluded, "closing_unpaid_bin_gold": closing_bin,
            "paid_shipping_gold": paid_bin,
            "paid_sales_gold": None if paid_bin is None else paid_bin + hen_total,
            "complete_shipping_observation": bool(measured and endpoints_match and set(keys) <= live.keys()
                                                  and not outside_sales and not ledger.get("counter_decreases")),
            "bin_gold": bin_total,
            "hen_sales_gold_receipts": hen_total,
            "gross_sales_gold": _sum([item["gross_sales_gold"] for item in out])}


def farm_work(report: Mapping[str, Any]) -> dict[str, Any]:
    """Frame budgets and explicit crop gates; missing receipts stay unknown."""
    phases: dict[str, dict[str, Any]] = {}
    ledger = report.get("work_metrics", {})
    seen: set[tuple] = set()
    duplicates: list[dict] = []
    for day in report.get("day_journal", ()):
        for row in day.get("phase_results", ()):
            name = str(row.get("phase", ""))
            work = row.get("work")
            if not work:
                continue
            start, end = row.get("start_frame"), row.get("end_frame")
            ident = (name, start, end)
            if isinstance(start, int) and isinstance(end, int) and end > start:
                if ident in seen:
                    duplicates.append({"phase": name, "start_frame": start, "end_frame": end, "plan_day": day.get("plan_day")})
                    continue
                seen.add(ident)
            stats = phases.setdefault(name, {"visits": 0, "frames": 0, "counts": {},
                                              "wet_gate_pass": 0, "wet_gate_fail": 0,
                                              "wet_gate_unknown": 0, "interrupted": 0,
                                              "plant_gate_pass": 0, "plant_gate_fail": 0,
                                              "plant_gate_unknown": 0})
            stats["visits"] += 1
            stats["frames"] += int(row.get("frames", 0))
            stats["interrupted"] += row.get("status") == "interrupted"
            for key, value in work.get("counts", {}).items():
                stats["counts"][key] = stats["counts"].get(key, 0) + int(value)
            if name == "CROP_ESTABLISH":
                before, after = work.get("before", {}).get("crops"), work.get("after", {}).get("crops")
                planted = None if before is None or after is None else max(0, after["crop"] - before["crop"])
                if planted is not None:
                    stats["counts"]["planted_tiles"] = stats["counts"].get("planted_tiles", 0) + planted
                gate = "unknown" if planted is None or row.get("status") == "no_work" else "pass" if planted > 0 and row.get("status") == "success" else "fail"
                stats[f"plant_gate_{gate}"] += 1
            wet = work.get("after", {}).get("gates", {}).get("all_growing_crops_wet")
            stats["wet_gate_pass" if wet is True else "wet_gate_fail" if wet is False else "wet_gate_unknown"] += 1
    sales = report.get("sales_ledger")
    confirmed = (sum(sum(row.get("crop_shipped", {}).values()) for row in sales.get("seasons", ()))
                 if sales and not sales.get("counter_decreases") else None)
    harvest = phases.get("HARVEST_ROUTE")
    if harvest is not None:
        harvest["counts"]["confirmed_shipped_count"] = confirmed
        reported = harvest["counts"].get("shipped_count", 0)
        harvest["shipment_delta"] = reported - confirmed if confirmed is not None else None
    for name, stats in phases.items():
        stats["receipt_frames"] = stats["frames"]
        stats["frames"] = ledger.get("phase_frames", {}).get(name, stats["frames"])
        key = ("confirmed_shipped_count" if name == "HARVEST_ROUTE" else
               next((key for key in ("watered_count", "planted_tiles") if stats["counts"].get(key)), None))
        units = stats["counts"].get(key, 0)
        stats["unit"] = {"confirmed_shipped_count": "confirmed_shipped_crop", "watered_count": "watered_tile", "planted_tiles": "planted_tile"}.get(key)
        stats["frames_per_unit"] = round(stats["frames"] / units, 1) if units else None
    wallet: dict[str, int] = {}
    for row in ledger.get("segments", ()):
        before, after = row.get("before", {}).get("money"), row.get("after", {}).get("money")
        if before is not None and after is not None and before != after:
            wallet[row["phase"]] = wallet.get(row["phase"], 0) + after - before
    opening, closing = report.get("start", {}).get("money"), report.get("end", {}).get("money")
    balanced = sum(wallet.values()) == closing - opening if ledger.get("segments") and opening is not None and closing is not None else None
    return {"phases": phases, "attributed_frames": ledger.get("frames"),
            "phase_frames": ledger.get("phase_frames", {}), "duplicate_receipts": duplicates,
            "wallet_delta_by_phase": wallet, "wallet_balanced": balanced}


def relationships(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Measured affection and the ROM's wrapped score; absent old fields stay unknown."""
    start, end = report.get("start", {}), report.get("end", {})
    result = {}
    for girl in ("maria", "ann", "nina", "ellen", "eve"):
        before, after = _int(start.get(f"{girl}_hearts")), _int(end.get(f"{girl}_hearts"))
        result[girl] = {"before": before, "after": after,
                        "delta": None if before is None or after is None else after - before,
                        "points": None if after is None else (after & 0x1FF) >> 4,
                        "in_window": None if after is None else 496 <= after <= 511}
    return result


def scorecard(report: Mapping[str, Any], name: str = "") -> dict[str, Any]:
    rows = day_rows(report)
    seasonal = seasons(report, rows)
    return {
        "name": name,
        "integrity": integrity(report),
        "cost": cost(report),
        "headline": headline(report, rows),
        "journal_days": len(rows),
        "eod_days": sum(1 for row in rows if row["bin_gold"] is not None),
        "seasons": seasonal,
        "animals": animals(report, rows),
        "gates": gates(rows),
        "care": care(rows),
        "reliability": reliability(report, rows),
        "spending": spending(rows),
        "completion": completion(report),
        "sales": sales(report, rows),
        "farm_work": farm_work(report),
        "relationships": relationships(report),
        "totals": {
            "harvested": sum(row["harvested"] for row in rows),
            "shipped": sum(row["shipped"] for row in rows),
            "bin_gold": _sum([season["bin_gold"] for season in seasonal]),
        },
    }


# --- text -----------------------------------------------------------------


def _when(item: Mapping[str, Any] | None, *, absent: str = NA) -> str:
    if not item:
        return absent
    return f"{item['label']} (day {item['index'] + 1}){', census' if item.get('source') else ''}"


def _buys(items: list[dict[str, Any]]) -> str:
    return ", ".join(f"{b['label']} {b['before']}->{b['after']}" for b in items[:6]) or "none"


def render_sales(card: Mapping[str, Any]) -> str:
    done, sold = card["completion"], card["sales"]
    out = [f"# Seasonal sales: {card['name']}", "",
           f"Clean power-on credits with score: {done['power_on_credits_with_score']}. "
           f"Score: {_num(done['score'])}. Perfect 999: {done['perfect_999']}.", "",
           "All gold is G. Crop cells are quantity / gold. Gross is bin shipping plus "
           "receipted hen sales, before spending. Milk + eggs is a residual estimate on measured chore days; "
           "forage and hen sales come from phase receipts.", "",
           "| Season | Turnip | Potato | Tomato | Corn | Crop G | Forage G | Milk + egg G (est.) | Unattributed bin G | Bin G | Hen sales G | Gross G | Paid G |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for season in sold["seasons"]:
        crops = [f"{_num(season['crop_shipped'][kind])} / {_num(season['crop_gold'][kind])}"
                 for kind in ("turnip", "potato", "tomato", "corn")]
        values = [season[key] for key in ("crop_gold_total", "forage_gold_receipts", "milk_egg_gold_est", "unattributed_bin_gold",
                                          "bin_gold", "hen_sales_gold_receipts", "gross_sales_gold", "paid_sales_gold")]
        out.append("| " + " | ".join([season["label"], *crops, *map(_num, values)]) + " |")
    out += ["", f"Total bin shipping: {_num(sold['bin_gold'])}G; receipted hen sales: "
            f"{_num(sold['hen_sales_gold_receipts'])}G; recorded gross sales: {_num(sold['gross_sales_gold'])}G.", "",
            f"Closing unpaid bin: {_num(sold['closing_unpaid_bin_gold'])}G. "
            f"Paid shipping from observed deposits: {_num(sold['paid_shipping_gold'])}G; "
            f"paid sales including receipted hens: {_num(sold['paid_sales_gold'])}G.", "",
            "| Season | Source | Bin snapshots / journal rows | Wallet start | Wallet end | Wallet change | Purchase receipts G | Other net cash G |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for season in sold["seasons"]:
        out.append("| " + " | ".join([
            season["label"], season["source"], f"{season['bin_days']}/{season['journal_days']}",
            _num(season["wallet_start"]), _num(season["wallet_end"]), _signed(season["wallet_delta"]),
            _num(season["purchases_gold_receipts"]), _signed(season["other_net_cash"]),
        ]) + " |")
    out += ["", "Other net cash = wallet change - recorded gross sales + purchase receipts. "
            "It includes unitemized spending and income, so it is not a profit figure.", ""]
    if not sold["complete_shipping_observation"]:
        out += ["**Incomplete shipping observation:** missing snapshots, unjournaled days, "
                "counter decreases or sales outside the run's calendar prevent a complete total.", ""]
    return "\n".join(out)


def render(card: Mapping[str, Any]) -> str:
    ig, co, hl, an, ca, rl, sp = (
        card[key] for key in ("integrity", "cost", "headline", "animals", "care", "reliability", "spending")
    )
    out = [f"Y1 BENCH {card['name']}".rstrip()]
    out.append(f"  start {ig['start']}  verdict {ig['verdict']}  comparable {'yes' if ig['comparable'] else 'NO'}")
    out.append(
        f"  success={ig['success']} terminal={ig['terminal']} end={ig['end_date'] or NA} "
        f"reason={str(ig['reason'])[:48]!r}"
    )
    out.append(
        f"  loads init/mid {_num(ig['initial_state_loads'])}/{_num(ig['mid_run_state_loads'])}  "
        f"ram_writes {_num(ig['ram_writes'])}  assists {ig['assists'] if ig['assists'] is not None else NA}"
    )
    out.append(
        f"  frames {_num(co['frames'])} over {_num(co['game_days'])} days = "
        f"{_num(co['frames_per_day'])}/day (planner {_num(co['planner_frames_per_day'])}/day)  "
        f"wall {_num(co['wall_seconds'])}s ({co['wall_per_day'] if co['wall_per_day'] is not None else NA}s/day)"
    )
    out.append("")
    out.append(
        f"HEADLINE net worth {_num(hl['net_worth'])}  [{hl['year_state']}, as of {hl['as_of'] or NA}]"
        f"   frames/day {_num(co['frames_per_day'])}"
    )
    out.append(
        f"  wallet {_num(hl['wallet'])} + chickens {_num(hl['chickens'])} x {CHICKEN_PRICE:,} = "
        f"{_num(hl['chicken_gold'])} + cows {_num(hl['cows'])} x {COW_PRICE:,} = {_num(hl['cow_gold'])}"
        f"   pasture grass {_num(hl['planted_grass'])}, hay {_num(hl['hay'])}"
    )
    out.append(f"  at resale ({CHICKEN_SALE_PRICE} per hen) {_num(hl['resale_net_worth'])}")
    out.append("  affection / ROM points: " + ", ".join(
        f"{girl} {_num(row['after'])}/{_num(row['points'])}" for girl, row in card.get("relationships", {}).items()))
    out.append("")
    head = f"  {'season':<10}{'days':>5}{'wallet':>9}{'+gold':>9}{'plant':>6}{'harv':>6}{'ship':>6}{'binG':>8}{'animG':>7}{'cow':>4}{'chk':>4}{'bad':>4}"
    out.append(head)
    for s in card["seasons"]:
        out.append(
            f"  {s['label']:<10}{s['days']:>5}{_num(s['wallet']):>9}{_signed(s['gold']):>9}"
            f"{_num(s['planted']):>6}{s['harvested']:>6}{s['shipped']:>6}{_num(s['bin_gold']):>8}"
            f"{_num(s['animal_gold']):>7}{_num(s['cows']):>4}"
            f"{('~' if s['chickens_from_log'] else '') + _num(s['chickens']):>4}{s['bad_days']:>4}"
        )
    out.append(
        f"  total: harvested {card['totals']['harvested']} shipped {card['totals']['shipped']} "
        f"bin gold {_num(card['totals']['bin_gold'])} "
        f"({'frame ledger; ' if card['sales']['complete_shipping_observation'] else ''}"
        f"{card['eod_days']}/{card['journal_days']} days read pre-sleep; --sales for breakdown)"
    )
    if any(s["chickens_from_log"] for s in card["seasons"]):
        out.append("  ~ flock from hen purchase receipts (no economy fields): excludes hatches and sales")
    out.append("")
    out.append("ANIMALS")
    out.append(
        f"  first cow {_when(an['first_cow'], absent='none' if an['cows_at_start'] in (None, 0) else 'owned at start')}"
        f"   first hen {_when(an['first_hen'], absent='none' if an['chickens_at_start'] in (None, 0) else 'owned at start')}   in coop {_when(an['first_in_coop'], absent='none')}"
    )
    if an["economy_fields"]:
        out.append(f"  brush owned {_when(an['brush'], absent='never')}   milker owned {_when(an['milker'], absent='never')}")
    else:
        out.append(
            f"  brush owned {NA}   milker owned {NA}   (no economy fields; first used: brush "
            f"{_when(an['brush_first_used'], absent='never')}, milker {_when(an['milker_first_used'], absent='never')})"
        )
    out.append(
        f"  cow buys {_buys(an['cow_buys'])}   hen buys {_buys(an['hen_buys'])}   "
        f"hens sold {an['hens_sold']}" + (f", first {_when(an['first_hen_sale'])}" if an["hens_sold"] else "")
    )
    gold = _num(an["animal_gold_est"])
    note = f"{an['animal_gold_days']}/{an['animal_days']} chore days measured" if an["animal_gold_est"] is not None else "needs eod fields"
    out.append(
        f"  milk shipped {an['milk_shipped']}  egg-ship days {an['egg_ship_days']}  "
        f"milk+egg gold est {gold} ({note})"
    )
    out.append(f"  forage gold (mushroom/grape deposits) {_num(an['forage_gold'])}")
    gt = card["gates"]
    if gt["mornings"]:
        blocks = ", ".join(f"{name} {n}" for name, n in gt["cow_blocks"]) or "none"
        out.append(
            f"  cow gate first open {gt['cow_open'] or 'never'}   tool trip first allowed "
            f"{gt['tool_trip'] or 'never'}   ({gt['mornings']} morning reads)"
        )
        out.append(f"  cow waited on: {blocks}")
    if an["bin_gold_anomalies"]:
        out.append(f"  WARNING {an['bin_gold_anomalies']} chore day(s) had bin gold below crop + forage gold")
    if an["unexplained_bin_gold"]:
        out.append(f"  WARNING {_num(an['unexplained_bin_gold'])}G of bin gold on non-chore days is not crop or forage")
    pct = NA if ca["pct"] is None else f"{ca['pct']}%"
    scope = "cows+hens" if ca["hens_known"] else "cows only"
    out.append(
        f"  care: fully fed {ca['fully_fed_days']}/{ca['animal_days']} animal-days = {pct} ({scope})"
    )
    out.append("")
    out.append("RELIABILITY")
    out.append(
        f"  bad days {rl['bad_days']} (floor report {_num(rl['floor_report_bad_days'])})  "
        f"floor trips {rl['floor_trips']}  rescues {rl['rescues']}  day failures {rl['day_failures']}"
    )
    if rl["floor_trip_days"]:
        out.append(f"  floor trips on {', '.join(rl['floor_trip_days'])}")
    top = ", ".join(f"{name} {n}" for name, n in rl["phase_failures_top"]) or "none"
    out.append(f"  phase failures {rl['phase_failures_total']}: {top}")
    kinds = ", ".join(f"{name} {n}" for name, n in rl["day_failure_kinds"]) or "none"
    skipped = ", ".join(f"{name} {n}" for name, n in rl["phase_skips_top"]) or "none"
    out.append(f"  day failures by phase/status: {kinds}")
    out.append(f"  phase skips {rl['phase_skips_total']}: {skipped}")
    seeds = ", ".join(f"{k} {n} bags/{sp['seed_gold'][k]:,}G" for k, n in sp["seed_bags"].items()) or "none seen"
    out.append(f"  seeds bought: {seeds}")
    work = card.get("farm_work", {})
    out.append(f"WORK attributed frames {_num(work.get('attributed_frames'))}")
    out.append(f"  wallet trace balances: {work.get('wallet_balanced')}; changes {work.get('wallet_delta_by_phase', {})}")
    if work.get("duplicate_receipts"):
        out.append(f"  duplicate receipts excluded: {len(work['duplicate_receipts'])}")
    for name, row in work.get("phases", {}).items():
        if name in {"CROP_WATER", "CROP_ESTABLISH", "HARVEST_ROUTE"}:
            out.append(f"  {name}: {row['frames']}f, {row['counts']}, "
                       f"{_num(row['frames_per_unit'])}f/{row['unit'] or 'unit'}")
            gate = "wet" if name == "CROP_WATER" else "plant" if name == "CROP_ESTABLISH" else None
            if gate:
                out.append(f"    {gate} gate pass/fail/unknown "
                           f"{row[gate + '_gate_pass']}/{row[gate + '_gate_fail']}/{row[gate + '_gate_unknown']}")
            elif name == "HARVEST_ROUTE":
                out.append(f"    reported minus native shipments {_num(row.get('shipment_delta'))}")
    return "\n".join(out) + "\n"


# --- comparison -----------------------------------------------------------


def flat(card: Mapping[str, Any]) -> dict[str, Any]:
    hl, co, an, ca, rl = (card[key] for key in ("headline", "cost", "animals", "care", "reliability"))
    metrics: dict[str, Any] = {
        "net worth": hl["net_worth"], "net worth (resale)": hl["resale_net_worth"], "wallet": hl["wallet"],
        "cows": hl["cows"], "chickens": hl["chickens"], "hens sold": an["hens_sold"],
        "frames/day": co["frames_per_day"], "wall s": co["wall_seconds"],
        "harvested": card["totals"]["harvested"], "shipped": card["totals"]["shipped"],
        "bin gold": card["totals"]["bin_gold"], "milk+egg gold est": an["animal_gold_est"],
        "first cow day": an["first_cow"]["index"] + 1 if an["first_cow"] else None,
        "first hen day": an["first_hen"]["index"] + 1 if an["first_hen"] else None,
        "care %": ca["pct"], "bad days": rl["bad_days"], "floor trips": rl["floor_trips"],
        "rescues": rl["rescues"], "phase failures": rl["phase_failures_total"],
    }
    for s in card["seasons"]:
        metrics[f"wallet {s['label']}"] = s["wallet"]
    for girl, row in card.get("relationships", {}).items():
        metrics[f"{girl} affection"] = row["after"]
    for name, row in card.get("farm_work", {}).get("phases", {}).items():
        if name in {"CROP_WATER", "CROP_ESTABLISH", "HARVEST_ROUTE"}:
            metrics[f"frames {name}"] = row["frames"]
            metrics[f"frames/unit {name}"] = row["frames_per_unit"]
            metrics[f"wet gate fails {name}"] = row["wet_gate_fail"]
    return metrics


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}"
    return _num(value)


def deltas(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    a, b = flat(first), flat(second)
    out: dict[str, Any] = {}
    for key in list(dict.fromkeys([*a, *b])):
        x, y = a.get(key), b.get(key)
        ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        out[key] = round(y - x, 1) if ok else None
    return out


def render_compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    a, b, d = flat(first), flat(second), deltas(first, second)
    out = [
        f"Y1 BENCH COMPARE  A={first['name']}  B={second['name']}",
        f"  A {first['integrity']['start']}, {first['integrity']['verdict']}",
        f"  B {second['integrity']['start']}, {second['integrity']['verdict']}",
        f"  {'metric':<22}{'A':>12}{'B':>12}{'B-A':>12}",
    ]
    for key in d:
        delta = NA if d[key] is None else (f"{d[key]:+,.0f}" if d[key] == int(d[key]) else f"{d[key]:+,.1f}")
        out.append(f"  {key:<22}{_cell(a.get(key)):>12}{_cell(b.get(key)):>12}{delta:>12}")
    return "\n".join(out) + "\n"


# --- CLI ------------------------------------------------------------------


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit(f"{path}: not a run report object")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Year-1 benchmark scorecard from run_to_day2 report(s).")
    parser.add_argument("reports", nargs="+", type=Path, help="report.json [other.json]")
    parser.add_argument("--out", type=Path, default=None, help="write .json (data) or .md/.txt (text)")
    parser.add_argument("--sales", action="store_true", help="seasonal sales breakdown and completion checks")
    args = parser.parse_args(argv)
    if len(args.reports) > 2:
        parser.error("give one report or two")
    cards = [scorecard(load_report(path), path.stem) for path in args.reports]
    text = "".join((render_sales(card) if args.sales else render(card)) + "\n" for card in cards)
    if len(cards) == 2:
        comparison = render_compare(*cards)
        text += "```text\n" + comparison + "```\n" if args.sales else comparison
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix == ".json":
            body: Any = cards[0] if len(cards) == 1 else {"a": cards[0], "b": cards[1], "delta": deltas(*cards)}
            args.out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        else:
            args.out.write_text(text if args.sales else "```text\n" + text.rstrip("\n") + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
