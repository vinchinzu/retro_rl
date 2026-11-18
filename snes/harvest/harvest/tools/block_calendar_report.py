"""Block calendar report: the Year-1 plot plan as JSON and a map page, and a run checked against it.

    .venv/bin/python -m harvest.tools.block_calendar_report plan --out recordings/block_calendar
    .venv/bin/python -m harvest.tools.block_calendar_report track <run>/report.json --out recordings/block_calendar

``plan`` writes ``block_plan.json`` (schema ``harvest.block_calendar/1``, from ``planner.block_calendar``)
and ``block_plan.html``. ``track`` adds a run: per day the forecast against the journal (wallet,
shipped by crop, harvest frames), and every ``ship_log`` record (``harvest.ship_log/1``, written by
``HarvestTask``) matched to its planned crop slot, so the measured pickup-to-ship frames sit beside
the frames the plan assumed. Both write the same page; ``track`` fills its run columns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

TRACK_SCHEMA = "harvest.block_track/1"

# ── Plan ──────────────────────────────────────────────────────────────


def make_plan(core: int, late: int, pin: Optional[str]) -> dict:
    from harvest.planner import block_calendar as bc

    walkable = bc.open_field_walkable
    source = "static field"
    if pin:
        try:
            walkable, source = bc.pin_walkable(pin), pin
        except (FileNotFoundError, OSError):
            pass
    plan = bc.build_plan(walkable, core=core, late=late)
    plan["geometry_source"] = source
    plan["sweep"] = [r for r in bc.sweep(walkable, cores=range(8, 19), lates=(0, 2, 4), tomatoes=(0,))][:12]
    plan["farm"] = farm_cells(walkable)
    return plan


def farm_cells(walkable, box=(2, 25, 28, 50)) -> dict:
    """Open cells of the drawn window, for the map's walls."""
    x0, y0, x1, y1 = box
    return {"box": list(box), "open": [[x, y] for y in range(y0, y1) for x in range(x0, x1) if walkable((x, y))]}


# ── Track ─────────────────────────────────────────────────────────────


def slot_index(plan: dict) -> Dict[tuple, dict]:
    index = {}
    for block in plan["blocks"]:
        for crop in block["crops"]:
            index[tuple(crop["tile"])] = {"block": tuple(block["center"]), "rank": block["rank"],
                                          "role": block["role"], **crop}
    return index


SEASON_NAMES = ("Spring", "Summer", "Fall", "Winter")


def track(plan: dict, report: dict, label: str) -> dict:
    """Forecast vs journal per day, and the ship log matched to planned slots."""
    from harvest.planner.block_calendar import PICK_FRAMES, RUN_FRAMES_PER_TILE, THROW_FRAMES

    forecast = {(d["season"], d["day"]): d for d in plan["forecast"]["days"]}
    slots = slot_index(plan)
    days, records = [], []
    season_wallets: Dict[str, int] = {}
    for row in report.get("day_journal", []):
        key = (row["plan_season"], row["plan_day"])
        if key[1] == 30 or key == max((r["plan_season"], r["plan_day"]) for r in report.get("day_journal", [])):
            season_wallets[SEASON_NAMES[key[0]]] = row.get("money")
        if key[0] > 1:
            continue                     # the plan covers Spring and Summer; later seasons only feed season_wallets
        shipped: Dict[str, int] = defaultdict(int)
        harvest_frames = 0
        for phase in row.get("phase_results", []):
            if phase.get("phase") != "HARVEST_ROUTE":
                continue
            harvest_frames += int(phase.get("frames") or 0)
            for rec in phase.get("ship_log") or []:
                shipped[rec.get("crop", "?")] += 1
                slot = slots.get(tuple(rec["target"]))
                walk = slot["walk_tiles"] if slot else None
                records.append({
                    "season": key[0], "day": key[1], **rec,
                    "block": list(slot["block"]) if slot else None,
                    "rank": slot["rank"] if slot else None,
                    "approach_frames": rec["f_pick"] - rec["f_select"],
                    "carry_frames": rec["f_ship"] - rec["f_pick"],
                    "cycle_frames": rec["f_ship"] - rec["f_select"],
                    "planned_cycle_frames": slot["cycle_frames"] if slot else None,
                    "planned_approach_frames": PICK_FRAMES + walk * RUN_FRAMES_PER_TILE if walk is not None else None,
                    "planned_carry_frames": THROW_FRAMES + walk * RUN_FRAMES_PER_TILE if walk is not None else None,
                })
        # Trips come from the phase reason ("2/2 shipped"): the journal's shipping_deposit of reports older than
        # 2026-11-02 holds only the last loop of a two-trip day.
        price = {0: 150, 1: 200}.get(key[0], 0)
        forage = {"trips": 0, "shipped_g": 0, "skipped": ""}
        for phase in row.get("phase_results", []):
            if phase.get("phase") != "MOUNTAIN_BERRY":
                continue
            done = re.search(r"(\d+)/\d+ shipped", str(phase.get("reason") or ""))
            if done and phase.get("status") == "success":
                forage["trips"] += int(done.group(1))
            else:
                forage["skipped"] = str(phase.get("reason") or phase.get("status") or "")[:60]
        forage["shipped_g"] = forage["trips"] * price
        fc = forecast.get(key, {})
        days.append({
            "season": key[0], "day": key[1], "money_night": row.get("money"),
            "forecast_money_night": fc.get("money_night"), "shipped": dict(shipped),
            "forecast_shipped": fc.get("shipped", {}), "harvest_frames": harvest_frames,
            "bad_day": bool(row.get("bad_day")), "storm": bool(row.get("is_storm")),
            "forage": forage, "forecast_forage": fc.get("forage"),
        })
    per_block: Dict[str, dict] = {}
    for rec in records:
        if rec["block"] is None:
            continue
        b = per_block.setdefault(str(rec["block"]), {"block": rec["block"], "rank": rec["rank"], "n": 0,
                                                       "cycle": 0, "planned": 0, "paid_g": 0})
        b["n"] += 1
        b["cycle"] += rec["cycle_frames"]
        b["planned"] += rec["planned_cycle_frames"] or 0
        b["paid_g"] += rec.get("paid_g", 0)
    for b in per_block.values():
        b["mean_cycle"] = round(b["cycle"] / b["n"], 1)
        b["mean_planned"] = round(b["planned"] / b["n"], 1)
        b["ratio"] = round(b["cycle"] / b["planned"], 2) if b["planned"] else None
    return {"schema": TRACK_SCHEMA, "label": label, "days": days, "ship_log": records,
            "season_wallets": season_wallets,
            "per_block": sorted(per_block.values(), key=lambda b: b["rank"])}


# ── Page ──────────────────────────────────────────────────────────────


def write_page(path: Path, plan: dict, tracks: List[dict]) -> None:
    data = json.dumps({"plan": plan, "tracks": tracks}, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(PAGE.replace("__DATA__", data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_track = sub.add_parser("track")
    p_track.add_argument("reports", nargs="+", type=Path)
    p_track.add_argument("--plan", type=Path, help="block_plan.json (default: <out>/block_plan.json, else rebuilt)")
    for p in (p_plan, p_track):
        p.add_argument("--out", type=Path, default=Path("recordings/block_calendar"))
        p.add_argument("--core", type=int, default=12)
        p.add_argument("--late", type=int, default=2)
        p.add_argument("--pin", default="LN_d2_perfect_poweron")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "block_plan.json"
    tracks: List[dict] = []
    if args.cmd == "track" and (args.plan or plan_path).is_file():
        plan = json.loads((args.plan or plan_path).read_text())
    else:
        plan = make_plan(args.core, args.late, args.pin)
        plan_path.write_text(json.dumps(plan, indent=1))
        print(f"wrote {plan_path}")
    if args.cmd == "track":
        for report_path in args.reports:
            t = track(plan, json.loads(report_path.read_text()), report_path.parent.name)
            tracks.append(t)
            out = args.out / f"block_track_{t['label']}.json"
            out.write_text(json.dumps(t, indent=1))
            print(f"wrote {out}: {len(t['days'])} days, {len(t['ship_log'])} ship records")
            for b in t["per_block"]:
                print(f"  block {tuple(b['block'])} rank {b['rank']:2d}: {b['n']:3d} crops, "
                      f"cycle {b['mean_cycle']:6.1f} vs plan {b['mean_planned']:6.1f} (x{b['ratio']})")
    page = args.out / "block_plan.html"
    write_page(page, plan, tracks)
    print(f"wrote {page}")


PAGE = r"""<title>Block Calendar Y1</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&family=Pixelify+Sans:wght@500;700&display=swap">
<style>
/* Layout: one reading column; the map and charts get the full column, tables scroll inside their own boxes. */
:root{
  --ground:#eef1ea; --panel:#f8faf5; --ink:#1c241f; --muted:#5d6b61; --rule:#cfd8cb;
  --lane:#dfe5d6; --wall:#7b8579; --soil:#b89566; --corn:#d7a91c; --tomato:#c4472f; --potato:#8c6a43;
  --water:#3c7fa8; --accent:#2f6b4f; --good:#2f7d4a; --warn:#b4632a;
  --display:"Pixelify Sans",ui-monospace,monospace; --body:"Archivo",system-ui,sans-serif;
  --data:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#141a16; --panel:#1b231e; --ink:#e3eadf; --muted:#9aa89d; --rule:#2e3a32;
  --lane:#26302a; --wall:#56625a; --soil:#8a6c47; --corn:#e2b93a; --tomato:#e0644a; --potato:#b08a5c;
  --water:#5aa0cc; --accent:#6fbf93; --good:#6fbf8a; --warn:#e09254; color-scheme:dark}}
:root[data-theme="dark"]{
  --ground:#141a16; --panel:#1b231e; --ink:#e3eadf; --muted:#9aa89d; --rule:#2e3a32;
  --lane:#26302a; --wall:#56625a; --soil:#8a6c47; --corn:#e2b93a; --tomato:#e0644a; --potato:#b08a5c;
  --water:#5aa0cc; --accent:#6fbf93; --good:#6fbf8a; --warn:#e09254; color-scheme:dark}
body{background:var(--ground);color:var(--ink);font:15px/1.55 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 56px}
main{max-width:1060px;margin:0 auto;display:grid;gap:34px}
h1,h2{font-family:var(--display);font-weight:700;text-wrap:balance;margin:0;letter-spacing:.01em}
h1{font-size:2.1rem;line-height:1.1}
h2{font-size:1.25rem}
p{margin:0;max-width:68ch}
.lede{color:var(--muted);margin-top:8px}
section{display:grid;gap:12px;min-width:0}
.kpis{display:flex;flex-wrap:wrap;gap:10px 28px;font-variant-numeric:tabular-nums}
.kpi b{display:block;font:500 1.35rem/1.2 var(--data)}
.kpi span{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px;min-width:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font:13px/1.4 var(--data);font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{padding:5px 10px;border-bottom:1px solid var(--rule);text-align:right}
th{font:600 11px/1.3 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
td:first-child,th:first-child{text-align:left}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font:500 11px/1.5 var(--body);color:var(--ground)}
.chip.core{background:var(--accent)} .chip.late{background:var(--tomato)} .chip.off{background:var(--wall)}
.chip.corn{background:var(--corn);color:#1c241f} .chip.tomato{background:var(--tomato)} .chip.potato{background:var(--potato)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.85rem;color:var(--muted)}
.controls button{font:500 .8rem var(--body);padding:5px 11px;border:1px solid var(--rule);border-radius:4px;
  background:var(--panel);color:var(--ink);cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink)}
.controls button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
svg text{font-family:var(--data)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8rem;color:var(--muted)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.tip{min-height:1.5em;font:13px var(--data);color:var(--muted)}
.note{font-size:.85rem;color:var(--muted)}
code{font:13px var(--data)}
dl{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:4px 16px;margin:0;font-size:.88rem}
dt{font-family:var(--data);color:var(--accent)} dd{margin:0;min-width:0}
@media (max-width:560px){h1{font-size:1.6rem} dl{grid-template-columns:1fr}}
@media (prefers-reduced-motion:no-preference){.cell{transition:opacity .15s}}
</style>
<main>
  <header>
    <h1>Block Calendar Y1</h1>
    <p class="lede" id="lede"></p>
  </header>
  <section>
    <div class="kpis" id="kpis"></div>
  </section>
  <section>
    <h2>Farm map</h2>
    <p class="note">Each block is a 3x3 with the centre planted and one side cell left bare as a notch. Pairs face each other across a shared lane. Numbers are frames from the bin to the pick and back (pick and throw included).</p>
    <div class="controls" role="group" aria-label="Map labels">
      <span>Label crops by</span>
      <button id="lab-plan" aria-pressed="true">planned frames</button>
      <button id="lab-run" aria-pressed="false">measured frames</button>
      <button id="lab-walk" aria-pressed="false">walk tiles</button>
    </div>
    <div class="panel scroll"><svg id="map" role="img" aria-label="Farm plot map"></svg></div>
    <div class="legend">
      <span><i style="background:var(--corn)"></i>core block, corn after its last potato</span>
      <span><i style="background:var(--tomato)"></i>late block, potato to Summer D5/D6, then tomato</span>
      <span><i style="background:var(--lane)"></i>lane / notch (stand)</span>
      <span><i style="background:var(--water)"></i>bin (F2) and its four stands</span>
    </div>
    <div class="tip" id="tip">Hover a crop for its slot.</div>
  </section>
  <section>
    <h2>Calendar</h2>
    <div class="panel scroll"><table id="cal"></table></div>
  </section>
  <section>
    <h2>Forecast</h2>
    <p class="note" id="fc-note"></p>
    <div class="panel scroll"><svg id="chart" role="img" aria-label="Wallet forecast"></svg></div>
    <div class="panel scroll"><svg id="ship" role="img" aria-label="Crops shipped per day"></svg></div>
  </section>
  <section>
    <h2>Forage</h2>
    <p class="note">The ROM gives a mountain pickup tile a different item each season (<code>DATA8_82CFB4</code>): grapes in Spring (150G), the green fruit in Summer (200G, one tile at (28,10)), mushrooms in Fall (150G, and a poison mushroom at 200G). A trip is one house-to-bin loop of 3,300-3,700 frames and costs no crop work while nothing is ripe to ship: watering, sowing and clearing run on after 17:00. So the plan sends the farmer up the mountain first on every day without a harvest (two grapes a day while two loops fit before the bin closes), and keeps harvest days for the bin.</p>
    <div class="panel scroll"><table id="forage"></table></div>
  </section>
  <section>
    <h2>Grass and cows by Summer's end</h2>
    <p class="note">Estimate only, not run on the ROM. Grass bought and sown in Summer (only Spring and Summer bags count toward the shop's 16 tiles per animal), sized to feed every animal hay to Year-2 Spring D14. One cow a day once cash covers 5,000G plus a 1,000G reserve. Milk from 21 nights after the buy, happiness +4 a day.</p>
    <div class="panel scroll"><table id="herd"></table></div>
  </section>
  <section id="run-section" hidden>
    <h2>Run against the plan</h2>
    <p class="note" id="run-note"></p>
    <div class="panel scroll"><table id="runblocks"></table></div>
    <div class="panel scroll"><table id="rundays"></table></div>
  </section>
  <section>
    <h2>Data</h2>
    <dl id="schema"></dl>
  </section>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const P = D.plan, T = D.tracks || [], RUN = T[T.length - 1];
const $ = id => document.getElementById(id);
const fmt = n => n == null ? '–' : Math.round(n).toLocaleString('en-US');
const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const A = P.assumptions, F = P.forecast.totals;
const core = P.blocks.filter(b => b.role === 'core'), late = P.blocks.filter(b => b.role === 'late');
$('lede').textContent = `${core.length} core blocks nearest the bin, ${late.length} late potato blocks behind them. ` +
  `Forecast from the Day-3 pin (Spring D${A.start.day}, ${fmt(A.start.money)}G) with ${A.max_water_tiles} watered tiles a day at most. Geometry: ${P.geometry_source}.`;
const kp = [
  [fmt(F.money_spring_end) + 'G', 'wallet, Spring D30'],
  [fmt(F.money_end) + 'G', 'wallet, Summer D30'],
  [fmt(F.summer_shipped), 'Summer crops shipped'],
  [`${F.shipped.potato}/${F.shipped.corn}/${F.shipped.tomato}`, 'potato / corn / tomato'],
  [fmt(F.seed_g) + 'G', 'seed bags'],
  [`${fmt(F.forage_gold)}G / ${F.forage_trips}`, 'forage gold / trips'],
];
for (const [v, l] of kp) { const d = document.createElement('div'); d.className = 'kpi'; d.innerHTML = `<b>${v}</b><span>${l}</span>`; $('kpis').appendChild(d); }

// ── Map ──
const S = 22, [X0, Y0, X1, Y1] = P.farm.box, W = (X1 - X0) * S, H = (Y1 - Y0) * S;
const map = $('map'); map.setAttribute('viewBox', `0 0 ${W} ${H}`); map.setAttribute('width', W); map.setAttribute('height', H);
const open = new Set(P.farm.open.map(([x, y]) => x + ',' + y));
const crops = new Map(); P.blocks.forEach(b => b.crops.forEach(c => crops.set(c.tile.join(','), {b, c})));
const notches = new Map(); P.blocks.filter(b => b.role !== 'off' && b.notch_cell).forEach(b => notches.set(b.notch_cell.join(','), b));
const measured = new Map();
if (RUN) for (const r of RUN.ship_log) { const k = r.target.join(','); const m = measured.get(k) || {n: 0, s: 0}; m.n++; m.s += r.cycle_frames; measured.set(k, m); }
const binF2 = new Set(['8,29', '9,29', '8,30', '9,30']);
const stands = new Set(P.bin_stands.map(s => s.join(',')));
const labels = [];
for (let y = Y0; y < Y1; y++) for (let x = X0; x < X1; x++) {
  const k = x + ',' + y, px = (x - X0) * S, py = (y - Y0) * S;
  const hit = crops.get(k);
  let fill = open.has(k) ? 'var(--lane)' : 'var(--wall)';
  if (binF2.has(k)) fill = 'var(--water)';
  if (hit && hit.b.role !== 'off') fill = hit.b.role === 'late' ? 'var(--tomato)' : 'var(--corn)';
  else if (hit) fill = 'var(--soil)';
  const r = el('rect', {x: px + .5, y: py + .5, width: S - 1, height: S - 1, rx: 2, fill, class: 'cell',
    opacity: hit && hit.b.role === 'off' ? .35 : 1}, map);
  if (stands.has(k)) el('circle', {cx: px + S / 2, cy: py + S / 2, r: 5, fill: 'var(--water)'}, map);
  if (notches.has(k)) el('rect', {x: px + 5, y: py + 5, width: S - 10, height: S - 10, rx: 1, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.2, 'stroke-dasharray': '2 2'}, map);
  if (hit && hit.b.role !== 'off') {
    const t = el('text', {x: px + S / 2, y: py + S / 2 + 3.5, 'text-anchor': 'middle', 'font-size': 8.5, fill: '#1c241f'}, map, '');
    labels.push({t, hit, k});
    r.addEventListener('mouseenter', () => {
      const m = measured.get(k);
      $('tip').textContent = `crop ${k} · block (${hit.b.center}) ${hit.b.notch} notch, rank ${hit.b.rank} ${hit.b.role} · stand (${hit.c.stand}) facing ${hit.c.face} · ` +
        `${hit.c.walk_tiles} tiles to the bin · plan ${hit.c.cycle_frames} f` + (m ? ` · measured ${Math.round(m.s / m.n)} f over ${m.n}` : '');
    });
  }
}
for (let x = X0; x < X1; x += 4) el('text', {x: (x - X0) * S + 2, y: 9, 'font-size': 8, fill: 'var(--muted)'}, map, x);
for (let y = Y0; y < Y1; y += 4) el('text', {x: 2, y: (y - Y0) * S + 9, 'font-size': 8, fill: 'var(--muted)'}, map, y);
function relabel(mode) {
  for (const {t, hit, k} of labels) {
    const m = measured.get(k);
    t.textContent = mode === 'walk' ? hit.c.walk_tiles : mode === 'run' ? (m ? Math.round(m.s / m.n) : '') : hit.c.cycle_frames;
  }
  for (const id of ['plan', 'run', 'walk']) $('lab-' + id).setAttribute('aria-pressed', id === mode);
}
for (const id of ['plan', 'run', 'walk']) $('lab-' + id).addEventListener('click', () => relabel(id));
if (!RUN) $('lab-run').disabled = true;
relabel('plan');

// ── Calendar ──
const cal = $('cal');
cal.innerHTML = '<tr><th>Block</th><th>Rank</th><th>Notch</th><th>Role</th><th>Last potato sow</th><th>Potato ripe</th><th>Summer bag</th><th>Live crops</th><th>Mean frames</th></tr>' +
  P.blocks.filter(b => b.role !== 'off').map(b => `<tr><td>(${b.center})</td><td>${b.rank}</td><td>${b.notch}</td>` +
    `<td><span class="chip ${b.role}">${b.role}</span></td><td>Spring D${b.spring_last_sow}</td><td>Summer D${b.potato_ripe_summer_day}</td>` +
    `<td>${b.summer_crop ? `<span class="chip ${b.summer_crop}">${b.summer_crop}</span> D${b.summer_plant_day}` : '–'}</td>` +
    `<td>${b.crops.filter(c => c.walk_tiles != null).length}/8</td><td>${b.mean_cycle_frames}</td></tr>`).join('');

// ── Forecast charts ──
const days = P.forecast.days, runDays = RUN ? RUN.days : [];
const idx = d => d.season * 30 + d.day;
const runBy2 = new Map(runDays.map(d => [idx(d), d]));
function lineChart(svg, series, yLabel) {
  const w = 1000, h = 260, l = 64, r = 16, t = 14, b = 30;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', w); svg.setAttribute('height', h);
  const xs = days.map(idx), xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(...series.flatMap(s => s.pts.map(p => p[1]))) * 1.05 || 1;
  const X = v => l + (v - xmin) / (xmax - xmin) * (w - l - r), Y = v => h - b - v / ymax * (h - t - b);
  const step = ymax > 50000 ? 20000 : ymax > 10000 ? 5000 : 1000;
  for (let v = 0; v <= ymax; v += step) {
    el('line', {x1: l, x2: w - r, y1: Y(v), y2: Y(v), stroke: 'var(--rule)'}, svg);
    el('text', {x: l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)'}, svg, fmt(v));
  }
  for (const v of [3, 10, 20, 31, 40, 50, 60]) if (v >= xmin && v <= xmax) {
    el('text', {x: X(v), y: h - 10, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)'}, svg, (v > 30 ? 'Su ' : 'Sp ') + (v > 30 ? v - 30 : v));
  }
  el('line', {x1: X(30.5), x2: X(30.5), y1: t, y2: h - b, stroke: 'var(--muted)', 'stroke-dasharray': '3 3'}, svg);
  for (const s of series) {
    el('polyline', {points: s.pts.map(([x, y]) => `${X(x)},${Y(y)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.w || 2}, svg);
    const last = s.pts[s.pts.length - 1];
    if (last) { el('circle', {cx: X(last[0]), cy: Y(last[1]), r: 3.5, fill: s.color}, svg);
      el('text', {x: X(last[0]) - 6, y: Y(last[1]) - 8, 'text-anchor': 'end', 'font-size': 11, fill: s.color}, svg, `${s.name} ${fmt(last[1])}`); }
  }
  el('text', {x: l, y: t - 2, 'font-size': 11, fill: 'var(--muted)'}, svg, yLabel);
}
const series = [{name: 'forecast', color: 'var(--accent)', pts: days.map(d => [idx(d), d.money_night])}];
if (RUN) series.push({name: RUN.label, color: 'var(--warn)', w: 2.5, pts: runDays.filter(d => d.money_night != null).map(d => [idx(d), d.money_night])});
lineChart($('chart'), series, 'wallet at night (G)');
$('fc-note').textContent = `The forecast ships ripe crops nearest-first until the 06:00-17:00 clock (${fmt(A.ship_day_ticks)} frames) runs out, with ${A.pick_frames} frames to pick, ${A.throw_frames} to throw and ${A.run_frames_per_tile} a tile running. Watering, sowing and the spa run after 17:00, which costs no clock.`;
(function shipChart() {
  const svg = $('ship'), w = 1000, h = 200, l = 64, r = 16, t = 14, b = 30;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', w); svg.setAttribute('height', h);
  const n = days.length, bw = (w - l - r) / n;
  const tot = d => Object.values(d.shipped || {}).reduce((a, v) => a + v, 0);
  const ymax = Math.max(10, ...days.map(tot), ...runDays.map(tot)) * 1.1;
  const Y = v => h - b - v / ymax * (h - t - b);
  for (let v = 0; v <= ymax; v += 20) { el('line', {x1: l, x2: w - r, y1: Y(v), y2: Y(v), stroke: 'var(--rule)'}, svg);
    el('text', {x: l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)'}, svg, v); }
  const runBy = new Map(runDays.map(d => [idx(d), d]));
  days.forEach((d, i) => {
    let y0 = 0; const x = l + i * bw;
    for (const c of ['potato', 'corn', 'tomato']) { const v = (d.shipped || {})[c] || 0; if (!v) continue;
      el('rect', {x: x + 1, y: Y(y0 + v), width: Math.max(1, bw * .55), height: Y(y0) - Y(y0 + v), fill: `var(--${c})`}, svg); y0 += v; }
    const rd = runBy.get(idx(d)); if (rd) { const v = tot(rd); if (v) el('rect', {x: x + bw * .58, y: Y(v), width: Math.max(1, bw * .32), height: Y(0) - Y(v), fill: 'var(--ink)', opacity: .55}, svg); }
    if (d.day % 5 === 0) el('text', {x: x + bw / 2, y: h - 10, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)'}, svg, (d.season ? 'Su ' : 'Sp ') + d.day);
  });
  el('text', {x: l, y: t - 2, 'font-size': 11, fill: 'var(--muted)'}, svg, 'crops shipped per day: forecast by crop' + (RUN ? `, ${RUN.label} in grey` : ''));
})();

// ── Forage ──
(function forageTable() {
  $('forage').innerHTML = '<tr><th>Day</th><th>Item</th><th>Trips</th><th>Gold</th><th>Run: trips</th><th>Run: gold</th><th>Run: why not</th></tr>' +
    days.map(d => { const f = d.forage, r = runBy2.get(idx(d)), rf = r && r.forage;
      return `<tr><td>${d.season ? 'Summer' : 'Spring'} D${d.day}</td><td>${f ? f.kind.replace('_', ' ') : '–'}</td><td>${f ? f.trips : 0}</td><td>${f ? fmt(f.gold) : '–'}</td>` +
        `<td>${rf ? rf.trips : ''}</td><td>${rf ? fmt(rf.shipped_g) : ''}</td><td>${rf && !rf.trips && f ? rf.skipped : ''}</td></tr>`; }).join('');
})();

// ── Herd ──
const herd = P.herd.options;
$('herd').innerHTML = '<tr><th>Cows</th><th>Grass tiles</th><th>Bags</th><th>Grass bought</th><th>Cow buy days</th><th>Spend</th><th>Milk to Y1 Winter D30</th><th>Milk to Y2 Summer D30</th><th>Net to Y2 Summer</th></tr>' +
  herd.map(o => `<tr><td>${o.cows}${o.cows === P.herd.best_cows ? ' ◂' : ''}</td><td>${o.grass_tiles}</td><td>${o.grass_bags}</td>` +
    `<td>${o.grass_buy_day ? 'Su D' + (o.grass_buy_day - 30) : '–'}</td><td>${(o.buy_days || []).map(d => 'D' + (d - 30)).join(' ') || '–'}</td>` +
    `<td>${fmt(o.spend_g)}</td><td>${fmt(o.milk_to_y1_winter_end)}</td><td>${fmt(o.milk_to_y2_summer_end)}</td><td>${fmt(o.net_to_y2_summer_end)}</td></tr>`).join('');

// ── Run ──
if (RUN) {
  $('run-section').hidden = false;
  const recs = RUN.ship_log, mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  $('run-note').textContent = `${RUN.label}: ${recs.length} crops in the ship log. Measured pickup-to-ship cycle ${fmt(mean(recs.map(r => r.cycle_frames)))} frames against ${fmt(mean(recs.filter(r => r.planned_cycle_frames).map(r => r.planned_cycle_frames)))} planned for the same slots.` +
    Object.entries(RUN.season_wallets || {}).map(([k, v]) => ` ${k} D30 wallet ${fmt(v)}G.`).join('');
  $('runblocks').innerHTML = '<tr><th>Block</th><th>Rank</th><th>Crops shipped</th><th>Measured frames</th><th>Planned frames</th><th>Ratio</th><th>Paid</th></tr>' +
    RUN.per_block.map(b => `<tr><td>(${b.block})</td><td>${b.rank}</td><td>${b.n}</td><td>${b.mean_cycle}</td><td>${b.mean_planned}</td>` +
      `<td style="color:${b.ratio > 1.3 ? 'var(--warn)' : 'var(--good)'}">${b.ratio}x</td><td>${fmt(b.paid_g)}</td></tr>`).join('');
  $('rundays').innerHTML = '<tr><th>Day</th><th>Wallet</th><th>Forecast</th><th>Shipped</th><th>Forecast shipped</th><th>Harvest frames</th><th>Flags</th></tr>' +
    runDays.map(d => { const s = o => Object.entries(o || {}).map(([k, v]) => `${v} ${k}`).join(', ') || '–';
      return `<tr><td>${d.season ? 'Summer' : 'Spring'} D${d.day}</td><td>${fmt(d.money_night)}</td><td>${fmt(d.forecast_money_night)}</td><td>${s(d.shipped)}</td><td>${s(d.forecast_shipped)}</td><td>${fmt(d.harvest_frames)}</td><td>${[d.storm ? 'storm' : '', d.bad_day ? 'bad day' : ''].filter(Boolean).join(' ') || ''}</td></tr>`; }).join('');
}

// ── Schema ──
const schema = [
  ['block_plan.json', 'schema harvest.block_calendar/1: assumptions, bin_stands, blocks[], forecast.days[], forecast.totals, herd.options[], sweep[]'],
  ['blocks[]', 'center [x,y], notch east|west|ring, notch_cell, layout_name, rank, role core|late|off, spring_last_sow, potato_ripe_summer_day, summer_crop, summer_plant_day, mean_cycle_frames, crops[]'],
  ['crops[]', 'tile, stand, face, walk_tiles (stand to nearest bin stand, BFS over lanes and notches), cycle_frames = pick + throw + 2 x walk x 8'],
  ['forecast.days[]', 'season, day, weekday, shop, bags{crop:n}, sown[], ripe, shipped{crop:n}, shipped_by_rank, ship_frames, ship_ticks_free, left_ripe, gold, forage{kind,trips,gold}|null, money_morning, money_night, water_tiles'],
  ['ship_log record', 'schema harvest.ship_log/1, one per bin drop in HARVEST_ROUTE phase rows of report.json: target, crop, from, stand, face, stand_tries, ship_stand, ship_face, f_select, f_pick, f_throw, f_ship (emulator frames), tick_* (ROM clock: (hour*15+minute)*60+second), paid_g, at_horse'],
  ['block_track_<run>.json', 'schema harvest.block_track/1: days[] (wallet and shipped vs forecast, harvest frames), ship_log[] (each record plus block, rank, approach_frames, carry_frames, cycle_frames and the planned values), per_block[]'],
];
$('schema').innerHTML = schema.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
</script>
"""


if __name__ == "__main__":
    main()
