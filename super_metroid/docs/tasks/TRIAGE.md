# Super Metroid backlog review & triage

**As of:** 2026-08-01 (STATUS + BACKLOG.csv + plan/PROCESS + Wave-11 cards).

No public GitHub Issues — the live backlog lives in docs:

| Board | Role |
|-------|------|
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | ~288 atomic tickets |
| [`routes/BACKLOG.md`](../routes/BACKLOG.md) | Epic summary + ready list |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip status |
| [`routes/KPDR_TRACKER.csv`](../routes/KPDR_TRACKER.csv) | Per-segment spine |
| [`research/PATH_ROOM_BOARD.md`](../research/PATH_ROOM_BOARD.md) | Path-room play status |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`QUEUE.md`](QUEUE.md) | Live executor wave |
| [`WAVE-11.md`](WAVE-11.md) | Current multi-agent dispatch |

## Current state

| Fact | Value |
|------|--------|
| Maturity | **M5** (route-building, Bronze, resource-assisted) |
| Best continuous tip | Power-on → **Frog Savestation** `0xB167` (K4.0) |
| Frames | **114,923f** (~31.9 min), dual integrity green |
| Integrity | 0 state loads / 0 progression or capacity writes / 0 deaths |
| Default continuous tip | `frog` |
| Prior continuous greens | Business return, Varia (K3), Kraid entry, Hi-Jump, Warehouse, Red Tower, Spore Super, … |
| Target | Assisted power-on → ending/credits (M7 dry-run with skips → M8 real fights) |
| Backlog size | ~288 tickets (+ tracker / path board / source catalog) |
| Process | Pure-first + one-knob + dual-track; planner owns continuous + STATUS |

## Critical path (spine — sequential for integrity)

Pure → graph → catalog/tip → continuous re-record → STATUS promote.
Do **not** door-warp past open hops and claim continuous progress.

| Priority | Ticket / work | Notes |
|----------|---------------|-------|
| **P0 ready** | `SM-K4.1-PURE` / `SM-K4-SPEEDWAY-PURE` Frog Save → Speedway | Source: `scratch/post_frog_continuous.state`. Residual may already be pure GREEN → next `SM-K4-SPEEDWAY-SRC` |
| P0 | K4 pure stack Speedway → farm → Bubble → Speed Hall → Speed Booster (+ return) | Then Wave / Ice pure stack |
| P0 | Compose + stabilize continuous tips (`--to speedway` / `speed` / `wave` / `ice`) + STATUS | Planner-serial after pure green |
| P1 | K5 Alpha PB (post-Ice natural collect) | First competitive PB on KPDR |
| P1 | K6 Moat → West Ocean → WS → natural Phantoon + fight + Gravity | Per [`BOSS_PIPELINE.md`](../BOSS_PIPELINE.md) |
| Later | K7 Maridia/Botwoon/Draygon · K8 LN/Ridley · K9 Tourian/MB + escape → credits | Hard blockers: Zebetite regen, escape geometry/timer, bank `$7E` events |

## Parallel tracks (do not block the spine)

### 1. Room policy farm / practice (~34 PRACTICE tickets)

Dual-track via `farm_room_waves.sh` + `ROOM_WORK_QUEUE` / `PATH_ROOM_BOARD`.
Prioritize ~107 completion-path rooms. Produces reusable policies for later
spine promotion. Main non-blocking progress engine.

Wave-11 atoms: `SM-PATH-ROOM-W01a`…`W01d` (see [`WAVE-11.md`](WAVE-11.md)).

### 2. Boss pipeline foundations (BOSS-INFRA + early boss cards)

Catalog, `BossStrategy` / `BossEvidence`, natural-entry capture harness, shared
combat primitives (lane-hold, phase machine, spray). **Kraid** is the living
continuous template. Prepare Phantoon unit harnesses before natural entry;
full fight policies stay gated on natural entry.

Wave-11 atoms: `SM-BOSS-PRIM-LANE`, `SM-BOSS-NATURAL-ENTRY-CLI`.

### 3. Architecture / structure debt (~10 ARCH tickets)

Hop tables out of `continuous.py`, typed path-summary, selective WRAM /
StateCache, pure-RED diagnostics, tip-wiring hygiene. Pays off as runs
lengthen (multi-hour scale).

Wave-11 atoms: `SM-ARCH-HOPS-MODULE`, `SM-ARCH-RED-DIAG`.

### 4. Optimizations on already-green segments

High-dwell tightening (Business↔Warehouse, HJ shaft, Climb, Spore). Multi-run
stability after any spine change. Source-state catalog expansion.

### 5. Optional / parked (only if cheap)

Conventional Charge return, Pure Pink PB, ship-first Phantoon, Crocomire side,
vision BC until gold. Forced progression writes: **never** on continuous path.

## Near-term focus (1–2 weeks)

| Lane | Action |
|------|--------|
| **Spine (serial)** | Land / re-verify Frog→Speedway pure → SRC → graph → continuous tip → re-verify `--to frog` + new intermediate tip |
| **Parallel** | Path-room farm waves; BossCatalog + combat primitives; 1–2 ARCH items that reduce continuous friction |
| **Hygiene** | After spine geometry: pure + continuous **stabilize** before STATUS |
| **Do not** | Expand topology/warp product work (Track A done). Do not fake continuous with warps or progression writes |

## Maturity path

| Gate | Meaning |
|------|---------|
| **M6** | Full route graph with owners + predicates (most pure/graph tickets) |
| **M7** | Continuous dry-run power-on → credits (boss skips allowed) |
| **M8** | Verified ending/credits with real boss fights + video/manifest evidence |

## “Not all sequential”

The KPDR spine is sequential for integrity. The bulk of ~288 tickets (room
farms, boss infra, ARCH, practice waves, side collections) are dual-track and
should advance in parallel. Enforcement: [`PROCESS.md`](PROCESS.md) pure-first,
residual schema with next-card ID, own-files conflict check in dispatch.
