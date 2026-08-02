# Backlog — Super Metroid full assisted clear

Machine source: [`BACKLOG.csv`](BACKLOG.csv) (**293** tickets).

Decomposed from KPDR spine + boss pipeline + dual-track practice + structure debt.
Target depth ~200–300 so executors always have atomic pure/graph/compose cards.

## Summary

| Status | Count |
|--------|------:|
| `open` | 270 |
| `ready` | 11 |
| `parked` | 7 |
| `done` | 5 |

| Epic | Count |
|------|------:|
| `K4` | 51 |
| `K7` | 41 |
| `K9` | 41 |
| `PRACTICE` | 38 |
| `K6` | 31 |
| `K8` | 25 |
| `COMPOSE` | 21 |
| `K5` | 10 |
| `ARCH` | 10 |
| `DOCS` | 7 |
| `OPTIONAL` | 7 |
| `BOSS-INFRA` | 6 |
| `DONE` | 5 |

| Kind | Count |
|------|------:|
| `pure` | 93 |
| `graph` | 41 |
| `compose` | 35 |
| `practice` | 34 |
| `boss` | 32 |
| `stabilize` | 17 |
| `status` | 12 |
| `arch` | 10 |
| `docs` | 8 |
| `infra` | 6 |

## ★ Ready now

Living cards: [`docs/tasks/WAVE-11.md`](../tasks/WAVE-11.md) ·
triage: [`docs/tasks/TRIAGE.md`](../tasks/TRIAGE.md).

| Ticket | Title | Own files | Living card |
|--------|-------|-----------|-------------|
| `SM-K4.1-PURE` | Pure Frog Save→Speedway (residual may be GREEN) | `routes/kpdr/k4_norfair.py` | [`SM-K4.1-PURE`](../tasks/SM-K4.1-PURE.md) / [`SM-K4-SPEEDWAY-PURE`](../tasks/SM-K4-SPEEDWAY-PURE.md) |
| `SM-K4-SPEEDWAY-SRC` | Fingerprint Speedway pure successor | `docs/SOURCE_STATES.md` | [`SM-K4-SPEEDWAY-SRC`](../tasks/SM-K4-SPEEDWAY-SRC.md) |
| `SM-PATH-ROOM-W01a`…`d` | Path-room clears (Speedway / Bubble / Speed Hall / Single Chamber) | `policies/room_clears/` (one each) | Wave-11 PATH cards |
| `SM-BOSS-PRIM-LANE` | Lane-hold window primitive | `combat/primitives.py` | [`SM-BOSS-PRIM-LANE`](../tasks/SM-BOSS-PRIM-LANE.md) |
| `SM-BOSS-NATURAL-ENTRY-CLI` | Standardize capture-natural CLI | `combat/` + thin probe CLI | [`SM-BOSS-NATURAL-ENTRY-CLI`](../tasks/SM-BOSS-NATURAL-ENTRY-CLI.md) |
| `SM-ARCH-HOPS-MODULE` | Extract hop tables → `routes/kpdr/hops.py` | `hops.py` + continuous import | [`SM-ARCH-HOPS-MODULE`](../tasks/SM-ARCH-HOPS-MODULE.md) |
| `SM-ARCH-RED-DIAG` | Pure RED clip + PLM/door snapshot | probe diagnostics | [`SM-ARCH-RED-DIAG`](../tasks/SM-ARCH-RED-DIAG.md) |

## P0 open (tip-critical)

| Ticket | Title | Depends |
|--------|-------|---------|
| `SM-K4.1-PURE` | Pure Frog Save→Speedway | `` |
| `SM-K4-SPEEDWAY-SRC` | Catalog Speedway successor source | pure green |
| `SM-K4.1-GRAPH` | Graph edge Frog Save→Speedway controller_dev | `SM-K4.1-PURE` |
| `SM-K4.2-PURE` | Pure Speedway→farm approach | `SM-K4.1-PURE` |
| `SM-K4.2-GRAPH` | Graph edge Speedway→farm approach controller_dev | `SM-K4.2-PURE` |
| `SM-K4.3-PURE` | Pure Approach→Bubble Mountain | `SM-K4.2-PURE` |
| `SM-K4.3-GRAPH` | Graph edge Approach→Bubble Mountain controller_dev | `SM-K4.3-PURE` |
| `SM-K4-TIP-SPEEDWAY` | Continuous tip --to speedway (or intermediate) | pure stack green |
| `SM-K4-TIP-SPEED` | Continuous tip --to speed | `SM-K4.6-PURE` |
| `SM-K4-TIP-WAVE` | Continuous tip --to wave | `SM-K4.10-PURE` |
| `SM-K4-TIP-ICE` | Continuous tip --to ice | `SM-K4.15-PURE` |
| `SM-K4-STAB-SPEED` | Stabilize wave pure+continuous Speed stack | `SM-K4-TIP-SPEED` |
| `SM-K4-STAB-ICE` | Stabilize wave pure+continuous Ice stack | `SM-K4-TIP-ICE` |
| `SM-K4-STATUS-SPEED` | STATUS promote Speed tip | `SM-K4-STAB-SPEED` |
| `SM-K4-STATUS-ICE` | STATUS promote Ice tip | `SM-K4-STAB-ICE` |

## Epic order (product)

```text
DONE (K0–K4.0 continuous)
  → K4 Speed/Wave/Ice
  → K5 Alpha PB
  → K6 Moat → Phantoon → Gravity
  → K7 Maridia → Botwoon → Draygon → Space Jump
  → K8 Lower Norfair → Ridley
  → K9 G4 → Tourian → MB → Escape → Credits  (M8)
Parallel: PRACTICE dual-track · ARCH structure · BOSS-INFRA
Parked: OPTIONAL (Pink PB, Charge return, Croc, GT, …)
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

Living cards (markdown) for active work live in [`docs/tasks/`](../tasks/).
Wave board: [`WAVE-11.md`](../tasks/WAVE-11.md). Triage:
[`TRIAGE.md`](../tasks/TRIAGE.md). Historical residuals / completed farm cards:
[`docs/tasks/archive/`](../tasks/archive/).

