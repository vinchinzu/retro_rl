# Super Metroid backlog review & triage

**As of:** 2026-08-02 (STATUS + BACKLOG.csv + plan/PROCESS + Cathedral repath).

No public GitHub Issues — the live backlog lives in docs:

| Board | Role |
|-------|------|
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | ~308 atomic tickets |
| [`routes/BACKLOG.md`](../routes/BACKLOG.md) | Epic summary + ready list |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip status |
| [`routes/KPDR_TRACKER.csv`](../routes/KPDR_TRACKER.csv) | Per-segment spine |
| [`research/PATH_ROOM_BOARD.md`](../research/PATH_ROOM_BOARD.md) | Path-room play status |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`QUEUE.md`](QUEUE.md) | Live executor wave |
| [`WAVE-11.md`](WAVE-11.md) | Closed multi-agent dispatch (2026-08-01) |
| [`plan.md`](../plan.md) | Strategy, risks, accelerators |

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
| Backlog size | **~308** tickets (K4-heavy; then K7/K9/practice/K6; ARCH/DOCS/BOSS-INFRA/CLEAN parallel) |
| Status mix | Mostly `open`; small `ready`; few `parked`/`done` |
| Kinds | pure → graph → compose → practice/boss lean |
| Process | Pure-first + one-knob + dual-track; planner owns continuous + STATUS |
| Clean track | Morph **green** (27,074f); ★ bombs/Torizo next — parallel only |

## Critical path (spine — sequential for integrity)

Pure → graph → catalog/tip → continuous re-record → STATUS promote.
Do **not** door-warp past open hops and claim continuous progress.
Pure-first discipline is **non-negotiable**.

| Priority | Ticket / work | Notes |
|----------|---------------|-------|
| **P0 ready** | **[`SM-K4.4-PURE-R11`](SM-K4.4-PURE-R11.md)** Bubble → Bat pure (Phase C→D ladder) | CATH-01…04 GREEN; R10 PARTIAL; ladder [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) |
| P0 | After Bat GREEN: Speed Hall → Speed / Wave / Ice pure | Graph + compose only after each pure green |
| P0 | Continuous tip extensions (`--to` bubble / speed / wave / ice) + stabilize + STATUS | Planner-serial; short stabilize after each tip |
| Parked | Frog Save → Speedway pure (**GREEN**) → Farm | Post-Speed shortcut only (Boost Blocks need Speed) |
| P1 | K5 Alpha PB (post-Ice natural collect) | First competitive PB on KPDR |
| P1 | K6 Moat → West Ocean → WS → natural Phantoon + fight + Gravity | Per [`BOSS_PIPELINE.md`](../BOSS_PIPELINE.md) |
| Later | K7 Maridia/Botwoon/Draygon/SJ · K8 LN/Ridley · K9 Tourian/MB + escape → credits | Hard blockers deferred: Zebetite regen, escape geometry/timer, bank `$7E` events |

Cathedral path (first Bubble, no Speed):

```text
Business 0xA7DE
  → Cathedral Entrance 0xA7B3   CATH-01 GREEN ~959f
  → Cathedral 0xA788            CATH-02 GREEN ~909f
  → Rising Tide 0xAFA3          ★ CATH-03
  → Bubble Mountain 0xACB3      CATH-04
  → Bat Cave → Speed Hall → Speed → Wave / Ice
```

## Parallel tracks (do not block the spine)

Width ≤ 8; own-files only; never claim continuous greens from these lanes.

### 1. Room policy farm / practice

Dual-track via `farm_room_waves.sh` + `ROOM_WORK_QUEUE` / `PATH_ROOM_BOARD`.
Prioritize ~107 completion-path rooms. Produces reusable policies for later
spine promotion. Main non-blocking progress engine.

### 2b. Early Spazer + 100% track (P2/P3 parallel)

Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md). Collect Spazer `0xA447` early via
Below Spazer detour (walljump-capable red-room geometry). Secondary continuous
tip first; **fold** into default spine only after pure + dual integrity
(`SM-SPAZER-FOLD`). Seeds eventual **100%** board (`SM-100-TRACK`).

| Priority | Work | Notes |
|----------|------|-------|
| P2 ready | `SM-SPAZER-SCAFFOLD` / `SM-SPAZER-SRC` | Disjoint from Bubble hot modules |
| P2 open | pure → graph → compose → stab → status | Pure-first |
| P2 open | `SM-SPAZER-POLICY` | Prefer Spazer when held (later combat) |
| P2 open | `SM-SPAZER-FOLD` | Planner-serial continuous insert |
| P3 ready | `SM-100-TRACK` | Item/map/boss checklist only |

Does **not** block Cathedral/Bubble P0. Supersedes parked `SM-OPT-SPAZER`.

### 2. Clean track — non-assist early tips (P2 parallel)

**Bronze / Clean:** no energy refill, no ammo refill. ★ Target continuous
power-on → **Bomb Torizo** exit. Morph green validates the contract.

Contract: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). Tickets: `SM-CLEAN-*`.

| Priority | Work | Notes |
|----------|------|-------|
| P2 done | Artifacts / CLI / integrity / contract / Morph | Morph clean 27,074f green 2026-08-02 |
| P2 ready | `SM-CLEAN-BOMBS` ★ | Missiles detour clean green; BT = existing model |
| P3 gated | `SM-CLEAN-BT-ECONOMY` | Only if clean BT RED |
| P4 parked | Clean spore / supers | After Clean BT green only |

**Do not** change default CLI assists or primary STATUS tip. Clean failures do
not demote assisted continuous greens. Keep artifacts and CLI strictly
separated (`*_clean` stems only).

### 3. Architecture / structure debt (~10 ARCH tickets)

Highest leverage for cheaper pure→continuous cycles:

1. Hop tables / tip runners → data-driven + `routes/kpdr/hops.py`
2. Selective WRAM / StateCache enforcement (linter or test gate)
3. Pure-RED diagnostics + source-state catalog polish
4. Graph API cleanup (typed path summary)

Cards: `SM-ARCH-HOPS-MODULE`, `SM-ARCH-RED-DIAG`, … — planner-serial on
`continuous.py` / `progression.py` / `catalog.py`.

### 4. Boss pipeline foundations (BOSS-INFRA only)

Catalog, `BossStrategy` / `BossEvidence`, natural-entry capture harness, shared
combat primitives. **Kraid** is the living continuous template. Full fight
policies stay gated on natural entry — primitives only until then.

### 5. Optimizations on already-green segments

High-dwell tightening (Business↔Warehouse, HJ shaft, Climb, Spore) is useful
but **secondary** to the pure stack. Multi-run stability after any spine change.

### 6. Optional / parked (only if cheap)

Conventional Charge return, Pure Pink PB, ship-first Phantoon, Crocomire side,
vision BC until gold. Forced progression writes: **never** on continuous path.

## Key issues / risks (none catastrophic)

| Risk | Notes |
|------|-------|
| Long-horizon nav fragility | Secondary to new pure hops; tighten offline after tips land |
| Architecture debt | Cloned tip surface, multi-registry tips, full WRAM in hot paths, large files — slows pure→continuous cycles |
| Residual / card proliferation | Enforce archive-after-successor + one-knob residual schema ([PROCESS.md](PROCESS.md)) |
| Process drift | Practice/Clean claiming continuous or mutating default assists = hard fail |
| Endgame notes | Zebetites, escape geometry, timer/WRAM — correctly deferred |

No major integrity regressions on the continuous spine; Climb loop / Spore
fight issues already cleaned.

## Near-term focus (1–2 weeks)

| Lane | Action |
|------|--------|
| **Spine (serial)** | Dispatch **CATH-03** → CATH-04 / Bubble → Speed/Wave/Ice pure → graph → compose → stabilize → STATUS |
| **After each continuous tip** | Force short stabilize wave before more knobs |
| **Parallel Clean** | ★ `SM-CLEAN-BOMBS` (own artifacts; no default assist change) |
| **Parallel** | Path-room farm waves; boss primitives only; 1–2 ARCH items that reduce continuous friction |
| **Hygiene** | After 1–2 continuous tips: boards/STATUS hygiene-only commit |
| **Do not** | Expand topology/warp product work. Do not fake continuous with warps or progression writes. Do not overwrite assisted baselines with clean runs |

**Ticket sizing:** one pure hop or one residual change per card; 30–90 min
sessions. Recipe acceptance: source fingerprint → target room, no
placement/warp, residual with exact next-card ID.

## Maturity path

| Gate | Meaning |
|------|---------|
| **M6** | Full route graph with owners + predicates (most pure/graph tickets) |
| **M7** | Continuous dry-run power-on → credits (boss skips allowed) |
| **M8** | Verified ending/credits with real boss fights + video/manifest evidence |

## “Not all sequential”

The KPDR spine is sequential for integrity. The bulk of ~308 tickets (room
farms, boss infra, ARCH, practice waves, Clean, side collections) are dual-track
and should advance in parallel. Enforcement: [`PROCESS.md`](PROCESS.md)
pure-first, residual schema with next-card ID, archive-after-successor,
own-files conflict check in dispatch.

Overall: system is disciplined at **M5** with a verified continuous tip past
Varia/Business/Frog and a clean parallel track emerging. Main accelerators are
finishing the immediate pure stack, landing continuous tips cleanly, and
chipping highest-leverage ARCH so each hop is cheaper.
