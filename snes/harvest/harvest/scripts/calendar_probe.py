"""Calendar what-if probe: can the planner survive day (year, season, day)?

Dev-only scouting, never evidence. From a real house pin it mints a what-if pin
set on the EVENING BEFORE the target (pre-sleep, weekday consistent, forecast
byte for the target written), then runs the normal multi-day planner for two
overnights: the ROM's own night code produces the target morning, then the
planner plays the target day. Reports pass/fail, the day journal and frames.

  python -m harvest.scripts.calendar_probe --dates Y1:fall:12 Y2:spring:1 \
      --base Y1_Summer_D13_Morning --jobs 3 --tag fest
  python -m harvest.scripts.calendar_probe --preset festivals --jobs 3

Date spec: ``Y<year>:<season>:<day>[:<weather>]`` with weather one of
sunny/rain/snow/hurricane/fair (default: the ROM forecast for the date, i.e.
festival codes, else sunny). ``--mint-only`` writes the pins and stops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harvest.paths import GAME_DIR, PROJECT_DIR

SEASONS = ("spring", "summer", "fall", "winter")
WEATHER_CODES = {"sunny": 0, "rain": 1, "snow": 2, "hurricane": 3, "fair": 4}
EVE_CLOCK = (18, 30)
OUT_DIR = PROJECT_DIR / "recordings" / "lanes"
DEFAULT_BASE = "Y1_Summer_D13_Morning"
DEFAULT_MAX_FRAMES = 140_000


def parse_spec(spec: str) -> tuple[int, int, int, str | None]:
    parts = spec.split(":")
    year = int(parts[0].lstrip("Yy"))
    season = SEASONS.index(parts[1].lower()) if not parts[1].isdigit() else int(parts[1])
    return year, season, int(parts[2]), (parts[3].lower() if len(parts) > 3 else None)


def date_tag(year: int, season: int, day: int, weather: str | None) -> str:
    return f"Y{year}_{SEASONS[season]}_D{day}" + (f"_{weather}" if weather else "")


def eve_of(year: int, season: int, day: int) -> tuple[int, int, int]:
    """The (year, season, day) whose night rolls into the target morning."""
    if day > 1:
        return year, season, day - 1
    if season > 0:
        return year, season - 1, 30
    return year - 1, 3, 30


def mint_whatif(
    base: str, year: int, season: int, day: int, weather: str | None, name: str
) -> Path:
    """Write pin ``name``: base pin re-dated to the eve before the target."""
    from harvest.core.harvest_state import HarvestStateDocument
    from harvest.planner.day_phase_calendar import forecast_code_for_date

    eve_year, eve_season, eve_day = eve_of(year, season, day)
    if eve_year < 1:
        raise SystemExit("target has no eve: Year 1 Spring D1 is the game start")
    code = WEATHER_CODES[weather] if weather else forecast_code_for_date(year - 1, season, day)
    doc = HarvestStateDocument.load(base)
    doc.set_calendar_date(game_year=eve_year, season=eve_season, day=eve_day)
    doc.set_clock(hour=EVE_CLOCK[0], minute=EVE_CLOCK[1], second=0)
    # SetWeatherFlags copies the forecast byte into today's flags at rollover.
    doc.set_scalar_value("weather_tomorrow", code)
    # A base morning may carry a storm/festival flag word; the night code
    # rewrites it, but stale bits must not read as "yesterday was a hurricane".
    doc.mutable_state.write_u16(0x196, 0)
    return doc.save_as(GAME_DIR / f"{name}.state")


def run_one(
    spec: str, base: str, tag: str, max_frames: int, days: int, mint_only: bool
) -> dict:
    year, season, day, weather = parse_spec(spec)
    ident = date_tag(year, season, day, weather)
    pin = f"LN_calendar_wi_{ident}"
    row: dict = {"spec": spec, "id": ident, "base": base, "pin": pin}
    try:
        row["pin_path"] = str(mint_whatif(base, year, season, day, weather, pin))
    except Exception as exc:  # noqa: BLE001 - report per-date, keep the batch going
        return {**row, "success": False, "reason": f"mint failed: {exc}"}
    if mint_only:
        return {**row, "success": None, "reason": "mint-only"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"cal_{tag}_{ident}.json"
    log = OUT_DIR / f"cal_{tag}_{ident}.log"
    cmd = [
        sys.executable, "-m", "harvest.scripts.run_to_day2",
        "--state", pin, "--days", str(days), "--max-frames", str(max_frames),
        "--progress-every", "20000", "--out", str(out),
    ]
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ).returncode
    row.update(rc=rc, wall_s=round(time.monotonic() - t0, 1), log=str(log), out=str(out))
    if not out.exists():
        return {**row, "success": False, "reason": f"no report (rc={rc})"}
    rep = json.loads(out.read_text(encoding="utf-8"))
    journal = rep.get("day_journal") or []
    prev_step = 0
    rows = []
    for r in journal:
        res = r.get("phase_results") or []
        by = lambda st: [x.get("phase") for x in res if x.get("status") == st]  # noqa: E731
        rows.append({
            "date": f"{r.get('plan_season')}:{r.get('plan_day')}->{r.get('end_season')}:{r.get('end_day')}",
            "ok": by("success"), "skipped": by("skipped"), "no_work": by("no_work"),
            "failed": by("failure"), "planned": r.get("planned_phases"),
            "storm": r.get("is_storm"), "festival": r.get("festival_name"),
            "sleep": r.get("sleep_reason"), "frames": int(r.get("step") or 0) - prev_step,
            "money": r.get("money"), "deferred": [d.get("phase") + ":" + str(d.get("reason"))
                                                    for d in r.get("deferred") or []],
        })
        prev_step = int(r.get("step") or 0)
    row.update(
        success=bool(rep.get("success")),
        reason=rep.get("reason"),
        frames=rep.get("frames"),
        end=f"{rep['end'].get('season')}:{rep['end'].get('day')}" if rep.get("end") else None,
        journal=rows,
        day_row=rows[-1] if rows else {},
        day_failures=rep.get("day_failures"),
    )
    return row


PRESETS = {
    # Verified ROM festival days (WeatherTomorrow, bank_82 $828CF9) + Y1 specials.
    "festivals": ["Y1:spring:23", "Y1:fall:12", "Y1:fall:20", "Y1:winter:10",
                  "Y1:winter:24", "Y2:spring:1", "Y1:winter:8:fair"],
    "weekdays": [f"Y1:{s}:{d}" for s in SEASONS for d in (7, 13)],
    "weather": ["Y1:spring:15:rain", "Y1:summer:15:rain", "Y1:fall:15:rain",
                "Y1:winter:15:rain", "Y1:winter:16:snow", "Y1:summer:17:hurricane"],
    "rollovers": ["Y1:summer:1", "Y1:fall:1", "Y1:winter:1", "Y2:spring:1"],
    "y2": ["Y2:spring:1", "Y2:spring:2", "Y2:spring:23", "Y2:summer:13", "Y2:fall:12",
           "Y2:winter:10", "Y2:winter:13"],
    "y3": ["Y3:spring:1", "Y3:spring:23", "Y3:summer:13", "Y3:summer:29", "Y3:fall:12",
           "Y3:winter:10", "Y3:winter:13"],
}


def to_markdown(rows: list[dict]) -> str:
    lines = ["| date | result | frames | phases ok | skipped/failed | note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        d = r.get("day_row") or {}
        ok = ",".join(d.get("ok") or []) if isinstance(d.get("ok"), list) else ""
        bad = ",".join((d.get("skipped") or []) + (d.get("failed") or [])) if d else ""
        res = "PASS" if r.get("success") else ("mint" if r.get("success") is None else "FAIL")
        lines.append(f"| {r['id']} | {res} | {r.get('frames', '')} | {ok} | {bad} | "
                     f"{(r.get('reason') or '')[:60]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dates", nargs="*", default=[], help="Y<n>:<season>:<day>[:<weather>]")
    p.add_argument("--preset", choices=sorted(PRESETS), action="append", default=[])
    p.add_argument("--base", default=DEFAULT_BASE, help="house-morning pin to re-date")
    p.add_argument("--tag", default="probe")
    p.add_argument("--jobs", type=int, default=1, help="parallel emulators (max 3)")
    p.add_argument("--days", type=int, default=2, help="overnights: eve sleep + target day")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument("--mint-only", action="store_true")
    args = p.parse_args(argv)
    specs = list(args.dates)
    for name in args.preset:
        specs.extend(PRESETS[name])
    if not specs:
        p.error("give --dates or --preset")
    jobs = max(1, min(3, args.jobs))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(
            lambda s: run_one(s, args.base, args.tag, args.max_frames, args.days, args.mint_only),
            specs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = OUT_DIR / f"cal_matrix_{args.tag}.json"
    matrix.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    md = to_markdown(rows)
    (OUT_DIR / f"cal_matrix_{args.tag}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"[CAL] matrix -> {matrix}")
    return 0 if all(r.get("success") in (True, None) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
