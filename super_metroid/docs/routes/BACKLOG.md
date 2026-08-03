# Backlog — Super Metroid full assisted clear

Machine source: [`BACKLOG.csv`](BACKLOG.csv) (**~308** tickets; includes CLEAN +
Cathedral pure stack).

Decomposed from KPDR spine + boss pipeline + dual-track practice + structure
debt + **parallel Clean** (no energy/ammo) early tips. Target depth ~200–310
so executors always have atomic pure/graph/compose cards.

## Summary

| Status | Count (approx) |
|--------|------:|
| `open` | 271 |
| `ready` | 11 |
| `done` | 13 |
| `parked` | 13 |

| Epic | Count |
|------|------:|
| `K4` | 55 |
| `K7` | 41 |
| `K9` | 41 |
| `PRACTICE` | 38 |
| `K6` | 31 |
| `K8` | 25 |
| `COMPOSE` | 21 |
| `CLEAN` | 11 |
| `K5` | 10 |
| `ARCH` | 10 |
| `DOCS` | 7 |
| `OPTIONAL` | 7 |
| `BOSS-INFRA` | 6 |
| `DONE` | 5 |

| Kind | Count |
|------|------:|
| `pure` | 94 |
| `graph` | 41 |
| `compose` | 39 |
| `practice` | 38 |
| `boss` | 32 |
| `stabilize` | 18 |
| `status` | 13 |
| `docs` | 10 |
| `arch` | 10 |
| `infra` | 9 |

Kinds lean pure → graph → compose → practice/boss. Epic weight is **K4-heavy**
(next spine), then K7/K9/practice/K6, with ARCH/DOCS/BOSS-INFRA/CLEAN parallel.

## ★ Ready now

Living cards: [`docs/tasks/QUEUE.md`](../tasks/QUEUE.md) ·
triage: [`docs/tasks/TRIAGE.md`](../tasks/TRIAGE.md).

| Ticket | Title | Own files | Living card |
|--------|-------|-----------|-------------|
| **`SM-K4-CATH-03`** | ★ Pure Cathedral→Rising Tide (serial spine) | `routes/kpdr/k4_norfair.py` | [`SM-K4-CATH-03`](../tasks/SM-K4-CATH-03.md) |
| `SM-CLEAN-BOMBS` | Continuous bombs/Torizo Clean (parallel) | continuous CLI (`*_clean`) | [`SM-CLEAN-BOMBS`](../tasks/SM-CLEAN-BOMBS.md) |
| `SM-ARCH-HOPS-MODULE` | Extract hop tables → `routes/kpdr/hops.py` | `hops.py` + continuous import | [`SM-ARCH-HOPS-MODULE`](../tasks/SM-ARCH-HOPS-MODULE.md) |
| `SM-ARCH-RED-DIAG` | Pure RED clip + PLM/door snapshot | probe diagnostics | [`SM-ARCH-RED-DIAG`](../tasks/SM-ARCH-RED-DIAG.md) |

**Done / parked (do not re-dispatch as spine):**

| Ticket | Status | Notes |
|--------|--------|-------|
| `SM-K4-CATH-01` / `02` | pure **GREEN** | Living residuals may archive after CATH-03 lands |
| `SM-K4.1-PURE` / Speedway pure | pure **GREEN**, **parked** post-Speed | Not first Bubble path |
| `SM-K4.2-PURE` Speedway→Farm | **RED** without Speed | Parked until post-Speed |
| Wave-11 PATH / BOSS / ARCH cards | closed 2026-08-01 | See [`WAVE-11.md`](../tasks/WAVE-11.md) |

## Parallel Clean (P2 — does not block assisted spine)

Contract: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). ★ product tip: Bomb Torizo
exit with no energy/ammo assists. **Infra done** (artifacts / CLI / integrity).

| Ticket | Title | Status |
|--------|-------|--------|
| `SM-CLEAN-CONTRACT` … `INTEGRITY` | Docs + `_clean` + `--clean` + integrity | **done** |
| `SM-CLEAN-MORPH` | Continuous morph Clean | **done** (27,074f green) |
| `SM-CLEAN-BOMBS` | ★ Continuous bombs/Torizo Clean | **ready** (missiles detour green; BT existing model) |
| `SM-CLEAN-BT-ECONOMY` | One-knob if clean BT RED | gated |
| `SM-CLEAN-STAB` / `STATUS` | Dual re-verify + STATUS secondary | after bombs GREEN |

## P0 open (tip-critical — Cathedral first Bubble)

| Ticket | Title | Depends |
|--------|-------|---------|
| `SM-K4-CATH-03` | Pure Cathedral→Rising Tide | CATH-02 green |
| `SM-K4-CATH-04` | Pure Rising Tide→Bubble | CATH-03 |
| Bubble→Speed / Wave / Ice pure stack | Geometry + sources | CATH-04 |
| Graph + compose tips (`--to` bubble/speed/wave/ice) | After each pure green | pure stack |
| Stabilize + STATUS per tip | Planner-serial | compose green |
| `SM-K4.1-PURE` / Speedway | **Parked** post-Speed shortcut | — |

CSV still lists historical Speedway farm rows; living spine is Cathedral —
see [`TRIAGE.md`](../tasks/TRIAGE.md) and [`SM-K4-REPATH-CATH-note.md`](../tasks/SM-K4-REPATH-CATH-note.md).

## Epic order (product)

```text
DONE (K0–K4.0 continuous)
  → K4 Cathedral → Bubble → Speed/Wave/Ice   ★ YOU ARE HERE (CATH-03)
  → K5 Alpha PB
  → K6 Moat → Phantoon → Gravity
  → K7 Maridia → Botwoon → Draygon → Space Jump
  → K8 Lower Norfair → Ridley
  → K9 G4 → Tourian → MB → Escape → Credits  (M8)
Parallel: PRACTICE dual-track · ARCH structure · BOSS-INFRA primitives
Parallel: CLEAN (no energy/ammo) → Bomb Torizo tip
Parked: Speedway→Farm until post-Speed · OPTIONAL (Pink PB, Charge, Croc, …)
```

## Ticket kinds (recipe)

| Kind | Meaning |
|------|---------|
| `pure` | Controller from continuous-like source |
| `graph` | Progression edge `controller_dev` |
| `compose` | Catalog tip + continuous hops |
| `stabilize` | Pure re-verify + continuous re-record |
| `status` | Planner STATUS/tracker promote |
| `boss` | Catalog / phases / closeout strategy |
| `practice` | Dual-track room farm (not continuous) |
| `arch` | Structure debt (planner-serial) |
| `docs` / `infra` | Rollups, sources, dispatch |

**Sizing:** one pure hop or one residual change per card; 30–90 min sessions.
Do not mix pure + continuous + STATUS in one mega-card.

Living cards (markdown) for active work live in [`docs/tasks/`](../tasks/).
Wave board: [`WAVE-11.md`](../tasks/WAVE-11.md). Triage:
[`TRIAGE.md`](../tasks/TRIAGE.md). Historical residuals / completed farm cards:
[`docs/tasks/archive/`](../tasks/archive/).
