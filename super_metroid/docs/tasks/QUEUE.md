# Super Metroid task queue (OpenCode executors)

Planner (Grok / human) owns design, natural-entry judgment, STATUS, and
integrity. Executors (OpenCode + Luna / Flash) take **one card per session**.

Dispatch from repo root:

```bash
./super_metroid/scripts/dispatch_opencode.sh SM-K4-06B
./super_metroid/scripts/dispatch_opencode.sh SM-PURE-ISO SM-GRAPH-GUARD SM-WRAP-DRAY
```

## Wave 3 — honest pass (2026-07-31)

Wave 3 is an **honest pass**, not an inflated continuous pass.

| Card | Goal | Model | Status | Notes |
|------|------|-------|--------|-------|
| SM-DOOR-PHASE | Phase + Y-sweep door recon | Luna | **done ✓** | Diagnostic only; never left 0xA59F |
| SM-TIGHTEN-01B | business settle 20→5 | Luna | **done ✓ code / continuous RED** | Planner continuous failed; **reverted to 20f** |
| SM-TIGHTEN-02B | HJ bomb tunnel A + settle B | Luna | **done ✓ code** | Appeared to clear into Business on failed run; full tip re-verify pending |
| SM-TIGHTEN-03 | terminator dwell report | Flash | **done ✓** | Report only |
| SM-TIGHTEN-04 | green shaft dwell report | Flash | **done ✓** | Report only |
| SM-BOTW-01 | Botwoon scaffold + wrap | Luna | **done ✓** | Dev only |
| SM-DRAY-01 | Draygon module + tests | Luna | **done ✓ scoped** | No wrap (residual → SM-WRAP-DRAY) |
| SM-K3-TRACK | Tracker reverse notes | Flash | **done ✓** | No false continuous |

**Planner continuous re-record (gate):**

1. After Wave-3 tightens (01B 20→5 + 02B knobs): `--to kraid --no-video`
   **FAILED** @ business climb:
   - `business_1227_land` — expected y=1227, actual **y=1419 floor** (~f90895)
   - Floor-recover retry `business_1339_ground` — y=1291 (~f91487)
2. **Root cause:** SM-TIGHTEN-01B 20→5 settles (lip unstable). Settles restored
   to **20f** (planner).
3. Re-record after revert: **GREEN** — outcome `kraid_entry`, **97,139f**,
   integrity 0 state loads / 0 progression writes, natural Kraid entry.
   Dwell (post-02B, 01B reverted): `business_to_warehouse` **2,257f** (unchanged
   vs prior report), `hj_shaft_to_business` **1,871f** (was ~1,885f on varia
   baseline report — ~14f band only; **no STATUS savings claim**).

**Still planner gates (not Wave-3 claims):**

- Continuous re-record green (`--to kraid`, then `--to varia`) + dwell compare
- Pure door geometry `kraid_to_eye_return` green

Earlier Wave 1–2 done cards remain in history below.

## Wave 4 — aggressive stress (model + harness failure points)

Intent: slightly harder cards — live continuous controllers, pure-green
pressure, isolation harness, anti-inflation tests. **Continuous re-record and
pure door geometry remain planner gates.** Cards must not STATUS-promote.

| Card | Goal | Model | Status | Stress target |
|------|------|-------|--------|---------------|
| SM-K4-06B | One-primitive short-hop Y approach pure | Luna | **done ✓ residual RED** | Pure still pin pose 82 x=37 y=307 door_transition=0 |
| SM-DOOR-BLUE | Blue-door RAM diagnostic | Luna | **done ✓** | 1200f samples; no open-state field; door never transitions |
| SM-PURE-ISO | Pure isolation probes for climb/HJ | Luna | **done ✓** | CLI wired; business pure **RED** @ 1339_ground y=1419 |
| SM-TIGHTEN-01C | Safer settle 20→12 + pure-first | Luna | **done ✓ code** | 8 settles→12f; pure CLI race (not wired yet in its session) |
| SM-TIGHTEN-P2 | Setup jumps 4→3 | Luna | **done ✓ code** | Both setup loops → `RIGHT,LEFT,LEFT` |
| SM-TIGHTEN-02C | Gray-door Recipe C | Luna | **done ✓ code** | First 4f X then RIGHT+B |
| SM-TIGHTEN-04A | Main-shaft entry guarded settle | Luna | **done ✓ code** | 1000f → 360f poll x118–126 |
| SM-TIGHTEN-03B | Terminator exit idle trim | Luna | **done ✓ code** | exit timeout 900→600 |
| SM-GRAPH-GUARD | Unit locks vs verification inflation | Luna | **done ✓** | kraid_to_eye stays unverified (27 tests) |
| SM-WRAP-DRAY | Draygon wrap residual | Luna | **done ✓** | wrap + export + tests |
| SM-K4-R-01 | eye→baby pure or blocked residual | Luna | **done ✓ blocked** | No chained eye source; door-shot scaffold only |

### Wave 4 honest rollup (2026-07-31)

All 11 sessions **EXIT:0**. Unit suite post-wave: **69 passed** (controller /
progression / k4 / kpdr_dev / draygon / boss / post_spore).

**Failure points found (value of aggressive wave):**

1. **Pure door geometry still blocked** — short-hop approach alone did not open
   left Kraid exit; same class pin as DOOR-PHASE (pose 82 / x≈37 / y≈307 /
   door_transition=0). Not inflated to pure green.
2. **Harness gap closed** — `business-to-warehouse` pure isolation now exists
   and **reproduces climb fail** at `business_1339_ground` (y=1419) from
   `continuous_like_business_climb_entry.state` in ~1k frames.
3. **Parallel file races accepted** — 01C+P2 both landed on `business_climb`
   (12f settle **and** 3 setup jumps). 04A+03B both landed on
   `spore_spawn_controller`. Continuous tip is **unverified** under this stack.
4. **01C pure-first intent weakened by race** — 01C session saw pure choice
   missing (PURE-ISO not finished); optional pure was not a real gate.

**Still planner gates (no STATUS / continuous claim):**

```bash
# Climb stack (01C+P2+02C) — expected stress:
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video

# Early spine (04A+03B):
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video

# Door geometry still open:
# next one-primitive card after 06B residual (landing/settle timing only)
```

If continuous fails: roll back climb settles 12→20 and/or setup 3→4 first;
gray Recipe C independently; shaft/terminator knobs independently.

### Planner climb isolation (post Wave-4)

| Knobs | Pure business-to-warehouse |
|-------|----------------------------|
| 3 setup `R,L,L` + 12f settle | **RED** `business_1339_ground` y=1419 |
| 4 setup + 12f settle | **GREEN** ~3721f → 0xA6A1 |

**Climb knobs (current, continuous-green once):** setup `LEFT,LEFT,RIGHT` (P2B)
+ platform settles **12f** (01C). Wave-4 P2 `RIGHT,LEFT,LEFT` pure-red; do not
restore that tuple.

**04A main_shaft guarded settle:** continuous **RED** (x=128 pose=0 miss band)
→ **reverted** to fixed 1000f hold. Do not re-enable without wider band + pure
isolation.

## Wave 5 — next stress batch (2026-07-31)

| Card | Goal | Model | Status | Notes |
|------|------|-------|--------|-------|
| SM-K4-06C | Hop/settle timing only pure | Luna | **done ✓ residual RED** | hop/settle 24/20 best; still door_transition=0 |
| SM-K4-06D | Weapon/missile door diagnostic | Luna | **done ✓** | beam/missile/super all no transition |
| SM-DOOR-PLM | Read-only PLM/door field or blocked | Luna | **done ✓ blocked** | no safe PLM field; no invent |
| SM-TIGHTEN-P2B | 3-jump re-try pure-first gate | Luna | **done ✓ pure GREEN** | `LEFT,LEFT,RIGHT` kept; pure ~3467f |
| SM-CLIMB-MATRIX | Pure A/B matrix report | Luna | **partial** | race w/ P2B; EXIT0, report incomplete |
| SM-HJ-SRC | Capture HJ shaft pure source | Luna | **done ✓ partial** | state 0xAA41; pure RED @ ensure_morph |
| SM-TIGHTEN-01D | Lock 12f settle docs | Luna | **done ✓** | settles 12f documented |
| SM-TIGHTEN-05 | Spore fight dwell report | Flash | **done ✓** | report only |
| SM-TIGHTEN-06 | Bomb Torizo dwell report | Flash | **done ✓** | report only |
| SM-GRAPH-NEXT | Reverse path block tests | Luna | **done ✓** | progression tests green (25) |
| SM-PHAN-02 | Phantoon unit expand | Luna | **done ✓** | unit tests |

### Wave 5 planner continuous (2026-07-31)

| Event | Result |
|-------|--------|
| After 04A guard in tree | **RED** `main_shaft_entry_settle` x=128 pose=0 (04A **reverted**) |
| After 04A revert + P2B `L,L,R` + 12f settle + 02B/02C | **GREEN** kraid_entry **96,924f**, 0 loads / 0 prog |
| Pure business (final knobs) | **GREEN** ~3467f |
| Dwell vs prior green 97,139f | total **−215f**; business **2,006** (was 2,257, **−251**); hj_shaft **1,835** (was 1,871, **−36**) |

**STATUS:** dwell deltas measured, **not** promoted to STATUS.md (multi-run stability still open; door pure still red).

**Serialize note:** P2B won the climb race (`L,L,R` + 12f). CLIMB-MATRIX concurrent residual incomplete.

### Wave 4 parallelism

- **OK parallel (disjoint):** DOOR-BLUE · PURE-ISO · GRAPH-GUARD · WRAP-DRAY ·
  (reports none this wave)
- **Serialize on business_climb:** 01C then P2 (not same session)
- **Serialize on hijump_return:** 02C alone vs other HJ edits
- **Serialize on spore_spawn_controller:** 04A then 03B (or opposite; not parallel)
- **Serialize geometry:** 06B before R-01 source expectation
- **Never parallel** with STATUS / continuous compose (planner only)

### Recommended dispatch order

1. Parallel batch A: `SM-PURE-ISO` `SM-GRAPH-GUARD` `SM-WRAP-DRAY` `SM-DOOR-BLUE`
2. Geometry: `SM-K4-06B` (then R-01 if source appears)
3. Efficiency (after A, preferably after pure-iso): `SM-TIGHTEN-01C` → planner
   continuous; only then `SM-TIGHTEN-P2` / `02C`
4. Early-spine efficiency: `SM-TIGHTEN-04A` → `SM-TIGHTEN-03B` (planner
   `--to spore` between or after)

## Wave 1–2 archive (done)

| Card | Goal | Model | Status | Notes |
|------|------|-------|--------|-------|
| SM-K4-01 | Lock reverse-return edge contract | Luna | **done** | |
| SM-K4-02 | Scaffold `play_kraid_to_eye_return` | Luna | **done** | |
| SM-K4-03 | Tracker rows K4 reverse | Flash | **done** | |
| SM-K4-04 | path_verification unit tests | Luna | **done** | |
| SM-K4-05 | Offline high-dwell rank | Flash | **done** | |
| SM-K4-06 | Geometry pure green eye return | Luna | **partial** | Still red @ door; → 06B |
| SM-K4-R-SCAFFOLD | 4 reverse hop scaffolds | Luna | **done** | |
| SM-PHAN-01 | Phantoon scaffold | Luna | **done** | |
| SM-BT-UNIT | Bomb Torizo unit tests | Luna | **done** | |
| SM-KRAID-UNIT | Kraid combat unit tests | Luna | **done** | |
| SM-TIGHTEN-01/02 | Dwell analysis reports | Flash | **done** | |
| SM-GRAPH-BRANCH | Wave+Ice path tests | Luna | **done** | |
| SM-DOOR-RECON | Left-door recon | Luna | **done** | pin pose 138 |

## Residual planner work (not cards)

1. **Hard block:** pure-green `kraid_to_eye_return` from
   `scratch/post_varia_to_kraid_pure.state`. Next executor attempt: SM-K4-06B
   (one short-hop primitive only). Still a planner gate to promote graph.
2. After pure green: promote graph edge → `controller_dev` only; re-run
   progression tests; tracker K3.3 → controller_dev.
3. Natural sources for eye→baby→… only after K3.3 pure green.
4. Continuous compose + tip recipe + integrity only after reverse spine
   pure-green end-to-end to Business.
5. **Continuous re-record gate:** `--to kraid` green after 01B revert (97,139f).
   Still needed before STATUS: `--to varia` re-record; multi-run stability on
   02B knobs; any Wave-4 efficiency patch re-record. No STATUS savings from
   ~14f hj_shaft band noise.
6. Do not STATUS-promote from scaffolds, diagnostics, or unit locks.
7. Phantoon / Botwoon / Draygon: scaffolds only until natural ship/Maridia entry.

## Model pick

| Card shape | Prefer |
|------------|--------|
| Tracker / docs / offline report / import tests | Flash |
| Controller scaffold, unit contracts, registration | Luna |
| Geometry pure green with explicit source state | Luna (bounded); planner reviews residual |
| Live continuous-spine efficiency patches | Luna; continuous verify = planner |
| Natural-entry design, continuous, STATUS | Planner only |

## Done criteria for a card

Executor final message must include **super-clean residual**: files changed,
verify paste, acceptance checklist, residual risks, planner next, explicit
non-claims. Planner marks QUEUE status and decides promotion.

**Force-pass ban:** pure geometry and continuous integrity are never “green”
from scaffolding, diagnostics, or unit tests alone.
