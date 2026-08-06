# Backlog — Super Metroid full assisted clear

Machine source: [`BACKLOG.csv`](BACKLOG.csv) (**~319** tickets; includes CLEAN +
Cathedral pure stack + Early Spazer / 100% ladder).

Decomposed from KPDR spine + boss pipeline + dual-track practice + structure
debt + **parallel Clean** (no energy/ammo) early tips. Target depth ~200–310
so executors always have atomic pure/graph/compose cards.

## Summary

| Status | Count (approx) |
|--------|------:|
| `open` | 271 |
| `ready` | 14 |
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
| `SPAZER` | 11 |
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

Living cards: [`docs/tasks/QUEUE.md`](../tasks/QUEUE.md).

| Ticket | Title | Own files | Living card |
|--------|-------|-----------|-------------|
| **Bat → Speed Hall pure** | ★ Next serial pure hop | `routes/kpdr/` (scaffold) | [QUEUE.md](../tasks/QUEUE.md) |
| `SM-CLEAN-BOMBS` | Continuous bombs/Torizo Clean (parallel) | continuous CLI (`*_clean`) | [`SM-CLEAN-BOMBS`](../tasks/SM-CLEAN-BOMBS.md) |
| `SM-ARCH-RED-DIAG` | Pure RED clip + PLM/door snapshot | probe diagnostics | [BACKLOG.csv](BACKLOG.csv) |

**Done / parked (do not re-dispatch as spine):**

| Ticket | Status | Notes |
|--------|--------|-------|
| `SM-K4-CATH-01`…`04` + Bubble→Bat pure | pure **GREEN**; continuous `bat_cave` | Primary tip promoted |
| `SM-ARCH-HOPS-MODULE` / `SM-ARCH-TIP-SPEC` | **done** | `hops.py` + `PostSupersTipSpec` + alias bind |
| `SM-K4.1-PURE` / Speedway pure | pure **GREEN**, **parked** post-Speed | Not first Bubble path |
| `SM-K4.2-PURE` Speedway→Farm | **RED** without Speed | Parked until post-Speed |
| Wave-11 PATH / BOSS / ARCH cards | closed 2026-08-01 | See [QUEUE.md](../tasks/QUEUE.md) |

## Parallel Early Spazer + 100% (P2/P3 — does not block K4 spine)

Epic: [TRACK_100.md](TRACK_100.md). Below Spazer
`0xA408` is already continuous; insert Spazer Room `0xA447` collect with
walljump-capable approach, secondary tip, then fold into default continuous.

| Ticket | Title | Status |
|--------|-------|--------|
| `SM-SPAZER-SCAFFOLD` | Module + `ROOM_SPAZER` | **ready** |
| `SM-SPAZER-SRC` | Continuous-like Below Spazer source | **ready** |
| `SM-SPAZER-PURE` | Pure collect + return (walljump) | open |
| `SM-SPAZER-GRAPH` … `STATUS` | Graph → tip → dual integrity → tracker | open |
| `SM-SPAZER-POLICY` | Later policies prefer Spazer when held | open |
| `SM-SPAZER-FOLD` | Fold into default continuous spine | open (after STAB) |
| `SM-100-TRACK` | 100% item/map/boss board | **ready** |
| `SM-OPT-SPAZER` | Old one-liner | **parked / superseded** |

## Parallel Clean (P2 — does not block assisted spine)

Contract: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). ★ product tip: Bomb Torizo
exit with no energy/ammo assists. **Infra done** (artifacts / CLI / integrity).

| Ticket | Title | Status |
|--------|-------|--------|
| `SM-CLEAN-CONTRACT` … `INTEGRITY` | Docs + `_clean` + `--clean` + integrity | **done** |
| `SM-CLEAN-MORPH` | Continuous morph Clean | **done** (27,074f green) |
| `SM-CLEAN-BOMBS` / `BT-ECONOMY` | Continuous bombs/Torizo Clean | **done** dual 49,321f (hybrid BT) |
| `SM-CLEAN-BT-ECONOMY` | One-knob if clean BT RED | gated |
| `SM-CLEAN-STAB` / `STATUS` | Dual re-verify + STATUS secondary | after bombs GREEN |

## P0 open (tip-critical — Cathedral first Bubble)

| Ticket | Title | Status |
|--------|-------|--------|
| `SM-K4-CATH-01`…`04` | Cathedral climb → first Bubble | **done** pure GREEN |
| `SM-K4.4-PURE` / R19 | Bubble → Bat Cave | **done** pure GREEN 2012f |
| ★ Bat → Speed Hall pure | Next serial pure hop | **ready** from `post_bubble_to_bat_pure` |
| Speed / Wave / Ice pure stack | Geometry + sources | after Bat→Speed |
| Graph + compose tips (`--to` bubble/speed/wave/ice) | After each pure green | pure stack |
| Stabilize + STATUS per tip | Planner-serial | compose green |
| `SM-K4.1-PURE` / Speedway | **Parked** post-Speed shortcut | — |

CSV still lists historical Speedway farm rows; living spine is Cathedral —
see [QUEUE.md](../tasks/QUEUE.md) and [MILESTONES.md](MILESTONES.md).

## Epic order (product)

```text
DONE (K0–K4.0 continuous)
  → K4 Cathedral → Bubble → Bat pure GREEN; ★ Bat → Speed → Wave/Ice
  → K5 Alpha PB
  → K6 Moat → Phantoon → Gravity
  → K7 Maridia → Botwoon → Draygon → Space Jump
  → K8 Lower Norfair → Ridley
  → K9 G4 → Tourian → MB → Escape → Credits  (M8)
Parallel: PRACTICE dual-track · ARCH structure · BOSS-INFRA primitives
Parallel: CLEAN (no energy/ammo) → Bomb Torizo tip
Parked: Speedway→Farm until post-Speed · OPTIONAL (Pink PB, Charge, Croc, …)
Parallel: SPAZER early walljump detour → tip → fold · 100% board scaffold
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
Wave board / triage: [`QUEUE.md`](../tasks/QUEUE.md). Delete completed cards
and residuals rather than keeping archive trees.
