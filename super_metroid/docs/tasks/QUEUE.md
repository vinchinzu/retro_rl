# Super Metroid task queue (OpenCode executors)

Planner (Grok / human) owns design, natural-entry judgment, STATUS, and
integrity. Executors (OpenCode + Luna / Flash) take **one card per session**.

Process rules (pure-first, stabilize waves, residual schema, metrics):
[`PROCESS.md`](PROCESS.md) · template [`docs/TASK_TEMPLATE.md`](../TASK_TEMPLATE.md)
· sources [`docs/SOURCE_STATES.md`](../SOURCE_STATES.md).

Dispatch from repo root:

```bash
./super_metroid/scripts/dispatch_opencode.sh SM-K4-06B
./super_metroid/scripts/dispatch_opencode.sh SM-PURE-ISO SM-GRAPH-GUARD SM-WRAP-DRAY
./super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS
# Luna + max thinking (default variant=max):
./super_metroid/scripts/dispatch_opencode.sh --luna --variant max SM-ROOM-SEG-01
```

### Wave 10 dual-track room farm (continuous tip relaxed)

Parallel **segment-only** farm. Continuous tip / STATUS / `routes/kpdr/*`
spine are **parked** for this wave — agents work one room problem each so
ownership never collides.

```bash
# 8 agents × up to 10 rounds (~hours); Luna + max thinking
./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
# or wall-clock cap:
./super_metroid/scripts/farm_room_waves.sh --rounds 20 --parallel 8 --deadline-hours 2
# dry-run card selection only:
./super_metroid/scripts/farm_room_waves.sh --rounds 1 --parallel 8 --dry-run
```

| Knob | Value |
|------|-------|
| Model | `openrouter/openai/gpt-5.6-luna` |
| Thinking | `--variant max` |
| Parallelism | 8 (one problem / agent) |
| Between rounds | wait EXIT → residual rollup → path guard → fresh sessions → next cards |
| Collision guard | each `SM-ROOM-SEG-NN` owns only its policy + state |
| Continuous tip | **relaxed / parked** — not a farm card |

Card generator: `scripts/generate_room_segment_cards.py`  
Farm logs: `docs/tasks/logs/farm/` (gitignored)

## Process gates (non-negotiable)

1. **Pure-first:** spine controller change → pure green from continuous-like
   source **before** continuous re-record.
2. **Stabilize wave:** after implement/stress knobs land, pure re-verify +
   continuous re-record **before** stacking more interacting knobs.
3. **One knob / residual:** each card one change; residual ends with next
   card ID + one change (PROCESS residual schema).
4. **Serialize hot modules:** `business_climb`, `hijump_return`, spore
   controller, `varia_return` geometry, `continuous.py` / `STATUS.md`.
5. **Dual track:** `ROOM_WORK_QUEUE` practice ≠ KPDR continuous integrity.
6. **Force-pass ban:** scaffolds / units / diagnostics never claim pure or
   continuous green.

## Wave 10 — dual-track room farm (**open**, continuous tip relaxed)

Intent: knock out easy/standard practice rooms with **8-wide** OpenCode Luna
(`--variant max`). Continuous tip / STATUS / spine geometry **parked** for
this wave. One agent = one `SM-ROOM-SEG-NN` problem (no file collisions).

| Knob | Value |
|------|-------|
| Parallel | 8 |
| Rounds | up to 10 (or `--deadline-hours 2`) |
| Model | `openrouter/openai/gpt-5.6-luna` + `variant=max` |
| Generator | `scripts/generate_room_segment_cards.py` |
| Orchestrator | `scripts/farm_room_waves.sh` |
| Skip | Wave-9 parked residuals + already-verified policies |

```bash
./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8 --deadline-hours 2
```

Between rounds: wait EXIT → residual rollup → path guard → fresh sessions →
next batch. Practice greens are dual-track only.

Spine tip work (warehouse→business R-04B, continuous post-Varia) stays
**planner-serial**, outside this farm.

---

## Wave 6 — stabilize + process seed (**closed**, 2026-07-31)

Intent: close Wave-5 continuous/dwell honesty, seed process tooling, extract
primitives. **No new post-Varia continuous tip until stabilize exits green.**

| Card | Goal | Model | Status | Notes |
|------|------|-------|--------|-------|
| SM-STAB-KRAID | Planner continuous `--to kraid` re-verify | Planner | **done ✓ GREEN** | **96,924f** kraid_entry; 0 loads / 0 prog; matches Wave-5 |
| SM-STAB-VARIA | Planner continuous `--to varia` re-verify | Planner | **done ✓ GREEN** | **104,382f** varia_collected; 0 loads / 0 prog (**+2,428f** vs STATUS 101,954 — no savings) |
| SM-ROLLUP-STATUS | STATUS/tracker/board **proposal** | Flash | **done ✓** | `SM-ROLLUP-STATUS-proposal.md` — planner apply selectively |
| SM-PRIM-01 | Extract `settle_hold` (12f) | Luna | **done ✓** | residual → SM-PRIM-01B call-site migrate |
| SM-PRIM-02 | Extract `short_hop` (24/20) | Luna | **done ✓** | residual → SM-PRIM-02B call-site migrate |
| SM-K4-06E | Jump-enter pure door residual | Planner | **done ✓ GREEN** | Pure ~610f → 0xA56B; graph `controller_dev`; residual `SM-K4-06E-residual.md` |
| SM-SRC-EYE | Capture eye-exit source after pure door green | Planner | **done ✓** | `scratch/post_kraid_to_eye_return.state` (0xA56B) |

### Wave-6 honest rollup

**Exit gate:** kraid + varia continuous integrity green ✓ · rollup proposal ✓ ·
2 primitives extracted ✓ · pure door was RED at wave open (later fixed in
Wave 7 planner pass — see below).

| Continuous | Frames | Outcome | Integrity |
|------------|-------:|---------|-----------|
| `--to kraid` | **96,924** | kraid_entry | 0 / 0 |
| `--to varia` | **104,382** | varia_collected | 0 / 0 |

**Dwell (both reports agree on climb band):**

| Split | Wave-6 kraid | Wave-6 varia | Notes |
|-------|-------------:|-------------:|-------|
| `business_to_warehouse` | 2,006 | 2,006 | multi-run match; −251 vs older 2,257 |
| `hj_shaft_to_business` | 1,835 | 1,835 | multi-run match; −36 vs ~1,871 |

**Do not STATUS-promote frame totals:** varia tip is **slower** than STATUS
baseline (101,954 → 104,382). Climb dwell multi-run is interesting but
savings claims wait until total tip is ≤ baseline or multi-run mean is
documented deliberately.

## Wave 7 — reverse pure chain (**closed implement**, 2026-07-31)

Intent: pure-green post-Varia reverse hops one at a time; graph `controller_dev`
only; **no continuous compose** until Business.

| Card | Goal | Model | Status | Notes |
|------|------|-------|--------|-------|
| SM-K4-06E | Jump-enter Kraid→eye pure | Planner | **done ✓ GREEN** | ~610f; Y-band (not floor spin) |
| SM-SRC-EYE | Capture eye source | Planner | **done ✓** | `post_kraid_to_eye_return.state` |
| SM-K4-R-01B | Eye→baby pure | Planner | **done ✓ GREEN** | ~651f jump mid-room |
| SM-K4-R-01C | Baby→kihunter pure | Planner | **done ✓ GREEN** | ~1248f supers clear gray lock |
| SM-K4-R-02 | Kihunter→zeela pure | Luna | **done ✓ RED residual** | Climb works; wrong door → Baby `0xA521` |
| SM-K4-R-02B | Door-window avoid Baby hatch | Luna | **→ Wave 8** | residual fix |
| SM-K4-R-03 | Zeela→warehouse pure | Luna | **blocked** | After R-02B source capture |
| SM-K4-R-GRAPH | Tracker/graph lock | Flash | **done ✓ partial** | K3.3–K3.5 ok; K3.6 stays open |
| SM-PRIM-01B | `settle_hold` migrate | Luna | **done ✓** | `big_pink_shaft.py` → 01C |
| SM-PRIM-02B | `short_hop` migrate | Luna | **→ Wave 8** | green_hill.py |
| SM-ROLLUP-APPLY | Selective honesty notes | Planner | **pending** | No 104,382 frame promote |

### Reverse pure chain (live)

| Hop | Pure | Frames | Graph | Source out |
|-----|------|-------:|-------|------------|
| varia→kraid | GREEN (prior) | — | controller_dev | `post_varia_to_kraid_pure` |
| kraid→eye | **GREEN** | ~610 | controller_dev | `post_kraid_to_eye_return` |
| eye→baby | **GREEN** | ~651 | controller_dev | `post_eye_to_baby_return` |
| baby→kihunter | **GREEN** | ~1248 | controller_dev | `post_baby_to_kihunter_return` |
| kihunter→zeela | **RED** | — | unverified | climb OK; upper door → Baby `0xA521` |
| zeela→warehouse | blocked | — | unverified | needs R-02B |

**Still no** post-Varia continuous tip until reverse pure spine reaches Business.

---

## Wave 8 — multi-track epics (**closed implement**, 2026-07-31)

Intent: harder parallel work; pure continuous **not** required for most tracks.

| Metric | Value |
|--------|------:|
| Sessions | **20** EXIT:0 |
| Unit recheck post-wave | **101 passed** (boss + scaffold + progression + controller) |
| Pure geometry | R-02B **RED** (climb OK; door → Baby `0xA521`) |
| Continuous | **n/a** (none attempted) |
| False continuous claims | **0** |

### Track rollup

| Track | Outcome |
|-------|---------|
| A reverse tip | R-02B RED → **SM-K4-R-02C** + recon |
| B bosses | Ridley/MB/Croc/GT/Escape scaffolds GREEN; Phan/Botw/Dray refine GREEN |
| C scaffolds | K4 Norfair + Moat GREEN; Pink PB PARTIAL |
| D practice | Boulder/Crab/Ice PARTIAL/RED — residuals filed |
| E harness/docs | PRIM-01C GREEN; PRIM-02B PARTIAL; graph guard GREEN; SRC inventory GREEN; pure-iso note GREEN |

---

## Wave 8b / 8c / 8d — residual fan-out (**closed**, 2026-07-31)

Intent: multi-session fan-out. Spine tip R-02E/F + climb recon + practice.
**All sessions EXIT:0.** Critical tip still RED — climb needs redesign.

### Critical / geometry

| Card | Goal | Model | Status |
|------|------|-------|--------|
| SM-K4-R-02C | Post-climb Zeela door-window only | Luna | **done ✓ RED** |
| SM-KIHUNTER-RECON | Door-band diagnostic report | Luna | **done ✓** (x=96..160) |
| SM-K4-R-02D | Real climb guard + recon Zeela band | Luna | **done ✓ RED** (x=357 y=395) |
| SM-K4-R-02E | Lower-alcove launch (x 360–420, cap x&lt;480) | Luna | **done ✓ RED** (x=470 y=395) |
| SM-K4-R-02F | Vertical launch cadence only | Luna | **done ✓ RED** (best_min_y=371) |
| SM-KIHUNTER-CLIMB-RECON | Climb launch grid (108 trials) | Luna | **done ✓ RED** (all min_y=371; no upper land) |
| SM-K4-R-03 | Zeela→warehouse pure | Luna | **blocked** — spine climb needs planner redesign |

### Boss / wrap

| Card | Goal | Model | Status |
|------|------|-------|--------|
| SM-BOSS-WRAP-01 | protocol wraps for new bosses | Luna | **done ✓** |
| SM-BOSS-UNIT-MATRIX | catalog × strategy matrix tests | Luna | **done ✓** |

### Practice residuals + next easy

| Card | Goal | Status |
|------|------|--------|
| SM-ROOM-EASY-01-R1 | Boulder door-entry one-knob | **done ✓ PARTIAL** |
| SM-ROOM-EASY-01-R2 | Boulder door-shot residual | **done ✓ PARTIAL** (same pin x=85) → switch Crab Hole |
| SM-ROOM-EASY-02 | Crab Hole first isolate | **done ✓ RED** (wrong exit `0xCF80`) |
| SM-ROOM-ICE-TUT-R1 | Ice left-exit one-knob | **done ✓ PARTIAL** |
| SM-ROOM-ICE-TUT-R2 | Ice pose-138 / left door residual | **done ✓ PARTIAL** → R3 |
| SM-ROOM-EASY-03 | Grapple Tutorial 2 | **done ✓ RED** |
| SM-ROOM-EASY-03-R1 | Grapple left-exit residual | **done ✓ RED** → R2 |
| SM-ROOM-METAL | Metal Pirates | **done ✓ RED** |
| SM-ROOM-METAL-01 | Metal combat-clear residual | **done ✓ RED** (enemy0_hp=1800) → Super Missile tactic |

### More scaffolds / primitives

| Card | Goal | Status |
|------|------|--------|
| SM-ALPHA-PB-01 | Alpha PB pure scaffold | **done ✓** (scaffold) |
| SM-WS-01 | Wrecked Ship approach scaffold | **done ✓** (scaffold) |
| SM-CHARGE-01 | Charge return optional scaffold | **done ✓** (scaffold) |
| SM-PRIM-01D | settle_hold → red_tower | **done ✓** |
| SM-PRIM-01E | settle_hold → kraid_approach | **done ✓** |
| SM-PRIM-02C | vertical hop primitive | **done ✓ GREEN** (`vertical_hop`) |
| SM-ROLLUP-STATUS-8B | Wave 8b honesty proposal | **done ✓** (proposal only) |

### Wave 8c/d honest spine pin

| Card | Result | Pin |
|------|--------|-----|
| R-02E | RED | timeout `0xA4DA` x=470 y=395 |
| R-02F | RED | timeout; `best_min_y=371` |
| CLIMB-RECON | RED diagnostic | 108/108 `min_y=371`; no upper land; no Baby |

**Exit gate:** stop stacking one-knob cadence cards. Climb redesign is Wave 9
planner work. R-03 stays blocked. STATUS not promoted.

---

### Wave 9 honest rollup

**Exit gate (updated after tip residual):** kihunter→zeela pure **GREEN** ~1716f + graph `controller_dev`. R-03 floor-left RED; R-03B reverse-drop class RED at second-drop stall x=122 y=409. Practice: Ice/Crab/Grapple parked; Metal fixture unlocked (combat still RED); BOOT-01 4 fixtures GREEN. Continuous post-Varia: **not attempted**.

| Metric | Value |
|--------|------:|
| Critical spine redesign | **GREEN** ~1716f (maneuver-class switch worked) |
| Next hop (zeela→warehouse) | **RED** — second-drop stall; R-03C live |
| Graph | `kihunter_to_zeela_return` = `controller_dev` |
| Practice rooms | parked residuals + 4 new scaffolds + Metal combat |
| Continuous post-Varia | **Blocked** — R-03C + warehouse→business + more |
| False continuous claims | **0** |
| STATUS 104,382 | **Not promoted** |

**Do not STATUS-promote:** reverse pure incomplete; no continuous tip total.

---

## Wave 9 — climb redesign + practice fan-out (**open**, 2026-07-31)

Intent: **planner redesign** of Kihunter alcove climb (maneuver class change);
parallel dual-track practice residuals. No continuous post-Varia. No R-02G
cadence spam.

### Critical / spine

| Card | Goal | Model | Status |
|------|------|-------|--------|
| **SM-K4-R-CLIMB-REDESIGN** | Wall → mid ledge → bomb hole x≈376 → Zeela | Planner | **done ✓ GREEN** ~1716f; source `post_kihunter_to_zeela_return` |
| SM-K4-R-03 | Zeela→warehouse pure | Luna | **done ✓ RED** floor-left pin x=19 y=395 dt=1 → R-03B |
| SM-K4-R-03B | Reverse of forward Zeela drops | Luna | **done ✓ RED** second-drop stall x=122 y=409 → R-03C |
| SM-K4-R-03C | Second-drop climb only (Hi-Jump) | Luna | **done ✓ RED** still floor y=395 x=89 |
| SM-ZEELA-CLIMB-RECON | Diagnostic climb grid (24 trials) | Luna | **done ✓** best_min_y=331 `forward_drop_reverse_shot` |
| SM-K4-R-03D | forward-drop reverse-shot climb class | Luna | **done ✓ RED** floor-door x=20 y=396 dt=1 |
| SM-K4-R-03E | Anti floor-left during reverse-shot climb | Luna | **done ✓ RED** no floor-door; pin x=41 y=395 → **PLANNER-GATE** |
| SM-K4-R-GRAPH-B | Promote `kihunter_to_zeela_return` → `controller_dev` | Flash | **done ✓ GREEN** |
| **SM-K4-R-ZEELA-REDESIGN** | Planner full Zeela reverse | Planner | **done ✓ GREEN** ~1800f; source `post_zeela_to_warehouse_return` |
| SM-K4-R-04 | Pure warehouse→business CLI + probe | Planner | **done ✓ RED** reverse stack → **R-04B** |

### Practice fan-out (disjoint) — tip residual EXIT:0; Wave 9b live

| Card | Goal | Model | Status |
|------|------|-------|--------|
| SM-ROOM-ICE-TUT-R3 | Ice `jumpx7` pose-138 residual | Luna | **parked** same pin x=277 |
| SM-ROOM-ICE-TUT-PARK | Park Ice one-knob chain (docs) | Flash | **done ✓ GREEN** |
| SM-ROOM-EASY-03-R2 | Grapple `land_shoot` approach residual | Luna | **done ✓ RED** → R3 |
| SM-ROOM-EASY-03-R3 | Grapple door open/entry only | Luna | **parked** same pin x=21 pose-138 → PLANNER-GATE |
| SM-ROOM-EASY-02B | Crab Hole wrong-exit residual | Luna | **done ✓ RED** → 02C |
| SM-ROOM-EASY-02C | Crab path-select top-left | Luna | **parked** still `0xCF80` → planner rewrite |
| SM-ROOM-EASY-PARK | Park Crab + Grapple (docs) | Flash | **done ✓** parked; planner rewrite placeholders in note |
| SM-ROOM-METAL-02 | Metal Super Missile combat tactic | Luna | **done ✓ RED** fixture gate |
| SM-ROOM-METAL-03 | Metal fixture supers capacity | Luna | **done ✓ PARTIAL** max_supers=5; combat RED |
| SM-ROOM-METAL-04 | Metal Super combat after fixture unlock | Luna | **done ✓ PARTIAL** enemy0_hp=1800 → park |
| SM-ROOM-BOOT-01 | Bootstrap next unstarted easy item rooms | Luna | **done ✓ GREEN** 4 fixtures |
| SM-ROOM-SCAFFOLD-HOPPER | Hopper E-Tank scaffold | Luna | **done ✓ GREEN** promoted |
| SM-ROOM-SCAFFOLD-BILLY | Billy Mays' scaffold | Luna | **done ✓ GREEN** promoted |
| SM-ROOM-SCAFFOLD-ICE | Ice Beam Room scaffold | Luna | **done ✓ RED** → ICE-R1 |
| SM-ROOM-SCAFFOLD-SPAZER | Spazer Room scaffold | Luna | **done ✓ RED** → SPAZER-01 |
| SM-ROOM-ICE-R1 | Ice collect residual | Luna | **done ✓ RED** beams=0 → R2 parked |
| SM-ROOM-SPAZER-01 | Spazer residual | Luna | **done ✓ RED** → PLANNER-GATE park |
| SM-ROLLUP-STATUS-9 | Wave 9 honesty proposal | Flash | **done ✓ GREEN** proposal only |

### Planner gates (still)

| Gate | Action |
|------|--------|
| Continuous post-Varia | Blocked until reverse pure → Business |
| STATUS 104,382 | Do not promote |
| Graph → continuous | Never from pure alone |
| **kihunter→zeela climb** | **GREEN** pure + graph `controller_dev` |
| **zeela→warehouse** | **GREEN** pure ~1800f + graph `controller_dev` (SM-K4-R-ZEELA-REDESIGN) |
| SM-K4-R-03…E | closed RED history; redesign landed green |
| SM-K4-R-04 | **RED** reverse warehouse stack; next **SM-K4-R-04B** planner |
| Ice / Crab / Grapple / Metal combat | **parked** |
| Hopper + Billy practice | **GREEN** promoted (dual-track only) |

### Reverse pure chain (live after Wave 9 redesign)

| Hop | Pure | Frames | Source out |
|-----|------|-------:|------------|
| varia→kraid | GREEN (prior) | — | `post_varia_to_kraid_pure` |
| kraid→eye | GREEN | ~610 | `post_kraid_to_eye_return` |
| eye→baby | GREEN | ~651 | `post_eye_to_baby_return` |
| baby→kihunter | GREEN | ~1248 | `post_baby_to_kihunter_return` |
| **kihunter→zeela** | **GREEN** | **~1716** | `post_kihunter_to_zeela_return` |
| **zeela→warehouse** | **GREEN** | **~1800** | `post_zeela_to_warehouse_return` |
| warehouse→business | **RED** | — | reverse stack; R-04B planner |

### Wave 9 spine tip residual (honest) — updated after redesign

R-03→R-03E closed RED. **SM-K4-R-ZEELA-REDESIGN pure GREEN ~1800f** (mid
RIGHT-bias reverse-shot → crouch-load lip → hop plant → shotblock clear →
wall-spin top → left Warehouse door). Graph `zeela_to_warehouse_return` =
`controller_dev`. Source captured.

R-04 pure CLI wired; **RED** at warehouse right ledge x≈728 — elevator-only
controller cannot cross super-stack (open from left; passage at y≈139). Next:
**SM-K4-R-04B** planner redesign of warehouse reverse approach.

**No continuous post-Varia. No STATUS 104,382. No false pure-green.**

```bash
# Zeela reverse pure (green):
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state

# R-04 still RED — do not claim warehouse→business until R-04B:
# uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
#   --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
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

1. ~~**Hard block:** pure-green `kraid_to_eye_return`~~ **done SM-K4-06E**
   (~610f, jump-enter Y band; graph `controller_dev`; eye source captured).
2. Pure chain reverse: eye→baby→kihunter→zeela→warehouse from chained sources
   (`post_kraid_to_eye_return` first).
3. Continuous compose + tip recipe + integrity only after reverse spine
   pure-green end-to-end to Business.
5. **Stabilize gate (Wave 6):** re-record `--to kraid` and `--to varia`.
   Wave-5 single green kraid **96,924f** / dwell −215f — **not** STATUS until
   multi-run stable. No STATUS savings from band noise.
6. Do not STATUS-promote from scaffolds, diagnostics, or unit locks.
   Use `SM-ROLLUP-STATUS` for proposal lag cleanup.
7. Phantoon / Botwoon / Draygon: scaffolds only until natural ship/Maridia entry.
8. Primitive extract (`SM-PRIM-01` / `02`) after green tightens — raise cheap
   agent hit rate before more geometry.

## Model pick

| Card shape | Prefer |
|------------|--------|
| Tracker / docs / offline report / STATUS proposal / rollup | Flash |
| Controller scaffold, unit contracts, registration, primitives | Luna |
| Geometry pure green with explicit source state | Luna (bounded); planner reviews residual |
| Live continuous-spine efficiency patches | Luna; continuous verify = planner |
| Natural-entry design, continuous, STATUS apply | Planner only |

## Done criteria for a card

Executor final message must include the **PROCESS residual schema**: files
changed, verify paste, acceptance checklist, residual risks, **Next card ID +
one change + source state**, explicit non-claims, probe pin if geometry.
Planner marks QUEUE status and decides promotion.

**Force-pass ban:** pure geometry and continuous integrity are never “green”
from scaffolding, diagnostics, or unit tests alone.
