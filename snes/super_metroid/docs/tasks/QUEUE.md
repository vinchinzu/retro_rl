# Queue

**Source of truth for ready work:** monorepo beads —

```bash
bd ready -l super_metroid
bd show <id>
```

This file is a **human snapshot** only. Update it when the tip changes; do not
invent tickets here that are not in `bd`. Planner owns STATUS, natural entry,
and continuous promote. Executors take **one bead per session**.

## Live tip

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → **Wave Beam** (K4.10) | **continuous green** | **136,361f** ×2 → `0xADDE` beams `0x1005` (default tip) |
| Speed Booster (K4.5) | previous tip | **130,388f** ×2 Spazer dual (prefix) |
| Spazer warehouse dual | prefix green | **89,416 + 90,904f** beams `0x1004` |
| Clean bombs/Torizo | secondary GREEN | **49,321f** ×2 — [CLEAN_TRACK.md](../CLEAN_TRACK.md) |
| ★ Product next | Pure **Snake → Ice PLM** (prefer 2WJ) | `rr-5if` (Gate/Acid/Snake dual GREEN) |
| Tape | Full Speed→Wave→Ice→Moat human | `rr-dbu.12` **GREEN** 39,711f |
| Tracker | Ice pure stack | `rr-dbu.11` (blocked on PLM hop) |
| Agent optional | consolidate · duck-type · Clean STATUS · speed start Spazer | P3 |

```text
✅ Continuous --to wave dual + STATUS — 136361f exact
✅ Human tape Speed→Wave→Ice→Moat 39711f (rr-dbu.12) — notes: no Spazer start; 2WJ climb
✅ Residual purge · guide_paths split
✅ Pure Business → Ice Gate  ← rr-fg3 dual GREEN
✅ Gate → Acid  ← rr-9t4 dual GREEN
✅ Acid → Snake  ← rr-5cf dual GREEN 652f ×2
▶  Snake → Ice PLM (prefer 2WJ)  ← rr-5if
⬜  Continuous --to ice  ← rr-dbu.7
⬜  K5 Alpha PB · Moat approach  ← rr-dbu.8 · rr-dbu.9
```

## Critical path (product)

```text
rr-dbu.12 (human tape)
  → rr-dbu.11 (Ice pure, routes/kpdr/ice/)
    → rr-dbu.7 (--to ice)
      → rr-dbu.8 (K5)
        → rr-dbu.9 (Moat approach → rr-hhj spark pin GREEN)
          → Wrecked Ship … ending
```

**Do not invent Ice hops without tape.** Hygiene Pass B does not block product.

## K6 / shinespark (side track — not tip gate)

| Doc / script | Role |
|--------------|------|
| [SHINE_PRACTICE.md](SHINE_PRACTICE.md) | **Index** — LS drill, Moat pure, West Ocean pure, follow-ups |
| `scripts/probe/shine_practice.py` | `drill` / `human` / `diagnose` / `demo` |
| `scripts/probe/moat_spark_watch.py` | Moat pure → `post_moat_west_ocean_spark.state` |
| `scripts/probe/west_ocean_spark.py` | Edge spark → `0xC98E` (not green WS yet) |

## Process pointers

| Doc | Role |
|-----|------|
| `bd ready -l super_metroid` | Ready / blocked work |
| [`PROCESS.md`](PROCESS.md) | Roles, pure-first, residuals, hot modules, dual-track |
| [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) | Hard in-room geometry playbook |
| [`TASK_TEMPLATE.md`](../TASK_TEMPLATE.md) | Card scaffold (geometry residuals) |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`STATUS.md`](../STATUS.md) | Verified tip claims |
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | Full ticket buffer (history / long tail) |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip board |
| Root [`AGENTS.md`](../../../../AGENTS.md) | Monorepo beads rules |
