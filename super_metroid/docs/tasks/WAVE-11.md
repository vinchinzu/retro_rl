# Wave-11 — multi-agent dispatch board (2026-08-01)

**Status: CLOSED** — all ready slots residual GREEN (parallel width-2 Grok batches).
Schema: [`TASK_TEMPLATE.md`](../TASK_TEMPLATE.md). Process: [`PROCESS.md`](PROCESS.md).
Live tip: [`QUEUE.md`](QUEUE.md). Triage: [`TRIAGE.md`](TRIAGE.md).

**Rules (unchanged):** own-files only · residual → next card ID + one change ·
width ≤ 8 · never mix continuous tip / `STATUS.md` into a farm batch · pure-first.

## Serial spine

| Slot | Card | Status |
|------|------|--------|
| 1 | [`SM-K4.1-PURE`](SM-K4.1-PURE.md) ≡ [`SM-K4-SPEEDWAY-PURE`](SM-K4-SPEEDWAY-PURE.md) | **GREEN** (~295f → `0xB106`) |
| 2 | [`SM-K4-SPEEDWAY-SRC`](SM-K4-SPEEDWAY-SRC.md) | **GREEN** — `post_frog_save_to_speedway_pure` cataloged |
| 3 | `SM-K4.2-PURE` / `SM-K4.1-GRAPH` | **Next** — open pure Speedway→farm; graph is planner-serial |

## Parallel dual-track (all GREEN)

| Slot | Card | Result |
|------|------|--------|
| P1 | [`SM-PATH-ROOM-W01a`](SM-PATH-ROOM-W01a.md) | practice promote Frog Speedway |
| P2 | [`SM-PATH-ROOM-W01b`](SM-PATH-ROOM-W01b.md) | practice promote Bubble Mountain |
| P3 | [`SM-PATH-ROOM-W01c`](SM-PATH-ROOM-W01c.md) | practice promote Speed Booster Hall |
| P4 | [`SM-PATH-ROOM-W01d`](SM-PATH-ROOM-W01d.md) | practice promote Single Chamber bottom (R1 open) |
| P5 | [`SM-BOSS-PRIM-LANE`](SM-BOSS-PRIM-LANE.md) | `lane_hold_window` + unit tests |
| P6 | [`SM-BOSS-NATURAL-ENTRY-CLI`](SM-BOSS-NATURAL-ENTRY-CLI.md) | multi-boss `capture-natural` CLI |
| P7 | [`SM-ARCH-HOPS-MODULE`](SM-ARCH-HOPS-MODULE.md) | `routes/kpdr/hops.py` extract; continuous import rewire |
| P8 | [`SM-ARCH-RED-DIAG`](SM-ARCH-RED-DIAG.md) | pure RED frame dump + door snapshot |
| + | [`SM-BOSS-UNIT-MATRIX`](SM-BOSS-UNIT-MATRIX.md) | catalog×strategy matrix tests (follow-on) |

## Hotspots (do not assign in parallel)

| Module / doc | Why |
|--------------|-----|
| `routes/continuous.py` | tip order / hops composition (import-only extract landed) |
| `docs/STATUS.md` | integrity claims |
| `progression.py` / `routes/catalog.py` | graph promote |
| `business_climb.py` / `varia_return.py` / climb return geometry | serialized knobs |

## Closeout

1. Residuals filed under `docs/tasks/*-residual.md` for every Wave-11 slot.
2. **Planner next:** open `SM-K4.2-PURE` card if missing → pure Speedway→farm → graph → tip compose → continuous stabilize → STATUS.
3. Archive completed cards under `archive/` once successors exist.
4. Wave-12 candidates: `SM-BOSS-PRIM-SPRAY`, `SM-ARCH-TIP-SPEC`, `SM-PATH-ROOM-W01d-R1`, more PATH_ROOM IDs.
