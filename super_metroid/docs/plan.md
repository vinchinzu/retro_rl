# Plan — Super Metroid assisted full clear

Shared workflow:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../../snes_oneshot/docs/FULL_RUN_PROCESS.md).
Assist semantics: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
Verified facts: [STATUS.md](STATUS.md). Layers/contracts: [ARCHITECTURE.md](ARCHITECTURE.md).
Executor process: [tasks/PROCESS.md](tasks/PROCESS.md).

## Strategy

Unlimited energy and ammo make combat and hazard attrition secondary. The hard
problem remains long-horizon navigation: room identity, door/elevator
transitions, item requirements, movement abilities, boss/event state,
backtracking, and recovery from positional stalls. Long continuous runs will
grow to multi-hour frame counts — structure and selective RAM matter as much as
geometry.

**Clear rooms by play.** Each hop on the completion path must be crossed with a
controller or room policy (natural door exit). Door-warps are topology
diagnostics only — never route evidence. Living inventory:

**[PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md)** (regenerate with
`scripts/export/path_room_board.py`).

Do not start with a monolithic full-run coordinate script. Grow one hop at a
time from the furthest played room. Continuous tip extension follows the recipe
in [ARCHITECTURE.md](ARCHITECTURE.md) (pure → graph → catalog → hops → record
→ STATUS).

**Boss fights stay deferred** until natural *entry* to that boss room exists on
the played chain. Continuous acceptance still requires natural boss flags and
zero progression writes. Pipeline: [BOSS_PIPELINE.md](BOSS_PIPELINE.md).

**Agent discipline is non-negotiable:** pure-first, one-knob, residual schema
with next-card ID + one change, dual-track (spine continuous vs room practice).
Do not relax these rules for scale — they are the core defense against dark
poking. See [tasks/PROCESS.md](tasks/PROCESS.md).

---

## Current inventory (2026-08-02)

**Status board:** [routes/MILESTONES.md](routes/MILESTONES.md) ·
**Backlog (~308 tickets):** [routes/BACKLOG.csv](routes/BACKLOG.csv) ·
**Spine segments:** [routes/KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) ·
**Live triage:** [tasks/TRIAGE.md](tasks/TRIAGE.md) · [tasks/QUEUE.md](tasks/QUEUE.md).

### Verified continuous (M5)

| Artifact | Coverage |
|----------|----------|
| `recordings/start_to_varia.{json,mp4}` | Power-on → **Varia Suit** (KPDR K3, post-Kraid) |
| Frames | **101,954** @ 60 fps (~28.3 min); integrity **0** state loads / **0** progression writes |
| Prefixes | Spore Supers 73,251f · Red Tower 80,445f · Warehouse 83,512f · Hi-Jump 87,696f · Kraid entry ~97k |
| Controllers | `routes/kpdr/` + `combat/kraid.py` (`play_kraid_entry_to_varia`) |
| `recordings/start_to_business*.json` | Power-on → **Business Center return** (KPDR K3→K4) |
| Frames | **113,723** @ 60 fps (~31.6 min), two matching integrity-green no-video runs |
| Integrity | 0 state loads / progression writes / capacity writes / deaths |
| `recordings/start_to_frog_save*.json` | Power-on → **Frog Savestation** (KPDR K4.0) |
| Frames | **114,923** @ 60 fps (~31.9 min), two matching integrity-green no-video runs |
| Integrity | 0 state loads / progression writes / capacity writes / deaths |
| Checkpoint | `scratch/post_frog_continuous.state` (tip) · `scratch/post_business_continuous.state` (Cathedral source) |

Reproduce: `scripts/record/continuous.py --to frog` (also `--to business|varia|kraid|hijump|warehouse|…`).

### Backlog model (healthy, atomic)

~**308** tickets in [BACKLOG.csv](routes/BACKLOG.csv) (epic summary:
[BACKLOG.md](routes/BACKLOG.md)). Status mix is mostly `open`, a small `ready`
set, a few `parked`/`done`. Kinds lean pure → graph → compose → practice/boss.
Epic weight: **K4-heavy** (next spine), then K7/K9/practice/K6, with ARCH /
DOCS / BOSS-INFRA / CLEAN parallel.

Each remaining spine hop stays **pure → graph → compose → stabilize →
status**. Living executor markdown is only for ready/in-flight cards
(`docs/tasks/`); the CSV is the full queue. **Ticket size rule:** one pure hop
or one residual change per card; STATUS/docs updates are planner-owned or tiny
follow-ons. Prefer 30–90 min agent sessions. Avoid mega-cards that mix pure +
continuous + STATUS.

### Post-Varia / K4 (continuous through Frog Save; Cathedral repath)

| Piece | Status |
|-------|--------|
| Reverse pure spine | Varia→…→Business green from accepted Varia checkpoint (9,343f) |
| Continuous tip | **Done:** power-on → Frog Save 114,923f ×2 integrity green |
| First Bubble repath | **Cathedral climb** (no Speed). Speedway→Farm needs Speed (parked) |
| Pure stack | CATH-01 **GREEN** (~959f) · CATH-02 **GREEN** (~909f) · ★ **CATH-03** open |
| Frog Save → Speedway | Pure **GREEN** (~295f); **post-Speed shortcut only**, not first Bubble |
| K4 forward after Bubble | Bat Cave → Speed Hall → Speed → Wave → Ice → (K5) Alpha PB |
| Bosses past Kraid | Unit/scaffold or dev-warp only until natural continuous entry |
| Ending / credits | Open (Zebetites / escape / timer correctly deferred) |

Live residual board: [tasks/QUEUE.md](tasks/QUEUE.md). Source states:
[SOURCE_STATES.md](SOURCE_STATES.md). Repath note:
[tasks/SM-K4-REPATH-CATH-note.md](tasks/SM-K4-REPATH-CATH-note.md).

### Research topology (not continuous)

| Piece | Status |
|-------|--------|
| `maps/full_room_graph.json` | 261 rooms, 583 directed edges, 22/22 completion legs have room paths |
| `maps/full_route_hops.json` | **~199 door hops** across all 22 legs (~**107 unique rooms**) |
| `dev/route_dev.py` + `probe_route.py` | Full 22-leg hop runner proven (dev); fights skipped; `developmentOnly` |
| Mid/late dev states | `dev_power_bombs_collected`, `dev_phantoon_entry`, `dev_route_*` anchors through finish |

### Continuous gap (first missing natural progress)

```text
[VERIFIED] power-on ──► Frog Savestation 0xB167 (K4.0)  + Business return 0xA7DE
                              │
                    ★ GAP: first Bubble via Cathedral (no Speed)
                       Business → 0xA7B3 → 0xA788 → 0xAFA3 → Bubble 0xACB3
                       pure: CATH-01✓ CATH-02✓ → ★ CATH-03 → CATH-04
                              │ then continuous tips: speedway?/bubble → speed → wave → ice
                              │ then Alpha PB → Moat → WS → Phantoon → …
                              │
[PARKED] Frog Save → Speedway → Farm → Bubble  (requires speed_booster)
[DEV ONLY] hybrid tour / door-warp chain (not continuous evidence)
```

### Room-policy + boss maturity

| Layer | Count / note |
|-------|----------------|
| Continuous segments | Full KPDR spine through Frog Save via `continuous.py --to` tips |
| Verified room_clears | Growing via dual-track `ROOM_WORK_QUEUE` + `farm_room_waves.sh` |
| Bulk scaffolds | 262 catalog problems; practice greens ≠ continuous evidence |
| Boss policies | Spore + Bomb Torizo + **Kraid→Varia continuous**; catalog + `BossStrategy` in `combat/`. Vision BC parked — see [BOSS_PIPELINE.md](BOSS_PIPELINE.md) |

**Practice track (parallel to continuous KPDR):** easy/standard room farms and
combat unit scaffolds while the spine advances. Never mix spine knobs into
farm waves.

---

## Two product tracks

Keep these separate so topology probes do not pollute continuous acceptance.

| Track | Goal | Integrity rules | Bosses |
|-------|------|-----------------|--------|
| **A — Topology probe** | Hop table / door-warp walk for connectivity only | Dev warps allowed; **label** `developmentOnly`; not route evidence | Skip bits OK for topology |
| **B — Played room spine** | Every path hop crossed by controller/policy; grow continuous chain | Assist contract for continuous claims; natural entries preferred | Entry by play first; fights later |
| **C — Room practice (dual-track)** | Isolated doorway segments / policies | Own-files only; not continuous evidence | Unit/scaffold only |

**Primary product path is Track B.** Track A already proved 22-leg connectivity;
do not invest further in warp tours as a substitute for playing rooms. Track C
raises practice % without blocking the tip.

How far we are (play, not warps):

| Layer | Furthest |
|-------|----------|
| Continuous | **Frog Savestation `0xB167`** (`start_to_frog_save`, **114,923f** twice) |
| Controller dev | Cathedral pure: CATH-01/02 green; CATH-03 scaffold open |
| ★ Next hop | Cathedral → Rising Tide → Bubble → Speed → Wave → Ice → Alpha PB → Phantoon |

**Continuous spine:** [ROUTE_KPDR.md](routes/ROUTE_KPDR.md) (K→P→D→R).
Hop table / waves: [PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md) (topology only).

---

## Critical path & dual-track (triage)

**Serial spine (integrity — pure-first is non-negotiable):**

| Priority | Work | Notes |
|----------|------|-------|
| **P0 now** | Finish pure stack **CATH-03+** → Bubble → Speed/Wave/Ice pure | Then graph edge + compose tip + dual re-record + STATUS |
| P0 | Stabilize continuous tip extensions (`--to` bubble/speed/wave/ice) | Short stabilize wave after each continuous promotion |
| P1 | K5 Alpha PB (natural post-Ice) | First competitive PB on KPDR |
| P1 | K6 Moat → Wrecked Ship → natural Phantoon + fight + Gravity | Per [BOSS_PIPELINE.md](BOSS_PIPELINE.md) |
| Later | Maridia (Botwoon/Draygon/SJ) → LN/Ridley → G4/Tourian/MB → Escape/Credits | Endgame notes deferred until natural entry |

**Parallel (safe dual-track — width ≤ 8, own-files only):**

| Lane | Work | Rule |
|------|------|------|
| Practice | `ROOM_WORK_QUEUE` + `farm_room_waves.sh` | Practice ≠ continuous integrity |
| Clean | Bombs/Torizo next (`SM-CLEAN-BOMBS`); Morph green | Separate `*_clean` artifacts; no default CLI assist mutation |
| ARCH | Tip-spec/hops extract, selective WRAM/StateCache, graph cleanup | Planner-serial on hot modules |
| Boss infra | Primitives / catalog / capture CLI only | Full fights only after natural entry |
| Sources | Catalog + diagnostics | Capture cards, not free geometry pokes |

Do **not** let Clean or farm waves claim continuous greens or change default
assists. Treat dual-track violations as hard fails ([PROCESS.md](tasks/PROCESS.md)).

---

## Key risks (none catastrophic)

| Risk | Mitigation |
|------|------------|
| Long-horizon nav fragility / high-dwell segments | Tighten offline secondary; stabilize after each tip |
| Architecture debt (cloned tip runners, multi-registry tips, full WRAM in hot paths, large files, source diagnostics) | Highest-leverage ARCH first — see Structure plan below |
| Residual / card proliferation | Archive-after-successor + one-knob residual schema (PROCESS) |
| Process drift (practice claiming continuous) | Dual-track gates; planner owns STATUS |
| Endgame (Zebetites regen, escape geometry, timer/WRAM) | Correctly deferred until natural entry |

No major integrity regressions on the continuous spine; historical issues
(Climb loop, Spore fight) already cleaned.

---

## Ticket sizing & plan accelerators

**Keep tickets atomic for agentic work:** pure hop, one-knob residual, focused
unit tests, clear acceptance (source fingerprint → target room, no
placement/warp, residual with exact next-card ID).

**Near-term (1–2 weeks):**

1. **Serial spine:** dispatch CATH-03 (then CATH-04 / Bubble stack) → green
   residual → planner lands graph/catalog/continuous tip + dual re-verify →
   STATUS / MILESTONES / KPDR_TRACKER.
2. After each continuous promotion, force a short **stabilize** wave before
   more knobs.
3. Parallel width ≤ 8 on own-files practice / ARCH / CLEAN only.
4. After Speed/Wave/Ice continuous tips land, push Alpha PB and first boss
   natural-entry work.

**Plan improvements (process + structure):**

| Improvement | Why | Where |
|-------------|-----|-------|
| Residual lifecycle: living cards only once successor exists | Prune residual pile; archive aggressively | [PROCESS.md](tasks/PROCESS.md) § residual lifecycle |
| Collapse tip runners / hop tables → data-driven + `hops.py` | Shrink `continuous.py` surface / merge pain | Structure §2 · `SM-ARCH-HOPS-MODULE` |
| Selective RAM / StateCache linter or test gate | Stop full parses in hot paths as runs lengthen | Structure §1 |
| Clean track isolation (artifacts + CLI) | Morph green validates contract; deepen carefully | [CLEAN_TRACK.md](CLEAN_TRACK.md) |
| Hygiene-only board commit after 1–2 tips | Regenerate matrix / boards without geometry | Flash rollup |

Main accelerators: finish the immediate pure stack, land continuous tips
cleanly, chip highest-leverage ARCH so each hop is cheaper. Focus serial
effort on the **K4 pure → continuous ladder**; everything else is parallel fuel.

---

## Structure & API plan (efficiency for whole-game length)

Current layers (CLI → `routes/continuous` + catalog + segment → pure kpdr
controllers → progression graph → ram/assist → combat) are the right
boundaries. Tip-extension recipe is clear. **After Frog Save continuous
acceptance (2026-08-01), structure debt is explicit** — do not keep paying a
tip tax into 2k-line modules without decomposition. Full map:
[ARCHITECTURE.md](ARCHITECTURE.md) (known structural debt snapshot).

Planner-serial when touching `continuous.py` / `progression.py` / `catalog.py`.
Product geometry (Speedway pure, etc.) stays dual-track parallel.

### 1. Selective RAM + StateCache enforcement (highest leverage)

Prefer `read_wram_u8/u16` / `peek_wram` + `StateCache` over full-bank
`parse_env_state` (especially `mode="full"`). Tight controller loops risk
accidental full copies as run length grows.

- [x] Pure probes use `mode="nav"`; `StateCache` hits/misses + `parse_counts()`.
- [x] `probe_pin()` residual helper; pure RED reports pin + optional `--pin-json`.
- [x] Cache-local parse counters on `StateCache.stats()` / `reset_stats()`
  (process-global `parse_counts` remains for probe rollups).
- [ ] Profile frame time on long `--to frog` / full runs (report WRAM-copy rate).
- [ ] Optional linter: forbid bare full `parse_env_state` inside `routes/kpdr/`.

### 2. Declarative continuous composition (**priority structure debt**)

Post-Supers tips are data-driven via `PostSupersTipSpec` (parent + hops +
report fields). Early morph→supers runners remain bespoke. Hop tables still
live in `continuous.py` (extract later if file size is the pain).

- [x] Scaffold script `scripts/scaffold_tip.py` (stub + residual + checklist).
- [x] Integrity-green `--state-output` checkpoint path on late tips.
- [x] **Tip-spec table** drives post-Supers `run_to()` / play chain (no new
  clone runner pair per tip; thin wrappers keep historical names).
- [ ] Move hop tables out of continuous (e.g. `routes/kpdr/hops.py` or per-tip
  modules) so continuous stays dispatch + composition only.
- [x] `ContinuousTip.supports_checkpoint` (and similar) instead of string
  allowlists in `run_to`.
- [x] Keep `ContinuousSession` / `HopExecutor` contracts stable.
- [x] Fix docstring drift in continuous/catalog when default tip moves
  (default tip is Frog Save / K4.0, not Varia).
- [x] Clean-track control plane: `resolve_clean_resources`, morph via
  `finish_report`, no `inspect.signature` in `run_to` (see
  [tasks/SM-ARCH-CLEAN-TRACK-residual.md](tasks/SM-ARCH-CLEAN-TRACK-residual.md)).
- [ ] Optional: one harness for early morph→supers runners (delete remaining
  clean/assist/finish copy).

### 3. Source-state & pure-probe diagnostics

`SOURCE_STATES.md` + continuous-like scratch states + code catalog. Strengthen:

- [x] Code catalog `source_states.py` + room fingerprint validation on pure load.
- [x] `kpdr.py suggest-source` + pure `--expect-room` / catalog match.
- [x] On pure RED: pin fields + optional pin JSON (`door_transition`, pose/x/y).
- [ ] Short video clip + PLM/door RAM snapshot on RED.
- [ ] Dispatch auto-suggest `--source` from card schema (pre-dispatch).
- [ ] Provenance on checkpoints (parent tip, command, capabilities).

### 4. Primitive library growth + promotion discipline

Grow `controller_common` (short-hop Y-approach, guarded settles, climb launches,
door-shot windows, etc.) aggressively **once a second consumer exists**.
Promote only after proven pure (+ continuous when on spine). Same for combat
primitives under `BossStrategy`.

**Controller structure (from tip review):**

- [x] Warehouse: explicit `entry_mode` (`auto` / `left_elevator` /
  `right_reverse_stack`) chosen once at hop start — not mid-loop rediscovery.
- [x] Super-stack open helper dedupe (`_open_warehouse_stack(face=…)`).
- [x] Zeela return: named phases; module docs match `continuous` verification.
- [ ] Prefer `wait_ordinary_room` handoff bands (`y_range` etc.) over hoping
  the next hop survives airborne settles.
- [ ] Remove thin warehouse helper aliases (`_hold = hold`) when touching other
  kpdr modules that still use them.

### 5. Graph first-class (**API collapse**)

Make `RoomProgressionGraph` the source of truth for next-hop suggestions,
work-queue ranking, and verification. **Do not keep twin helpers.**

- [x] Landed helpers: `suggest_pure_work`, `pure_gate` (usable for cards).
- [x] **Collapse** into one rank table (`VERIFICATION_RANK`) + `path_summary`
  (`min_verification=`) + `suggest_edges` (prefer/exclude filters);
  `pure_gate` / `path_verification` / `suggest_*` are thin wrappers.
- [ ] Typed path-summary model instead of ad-hoc `dict[str, object]`.
- [ ] Stop mechanical multi-line DoorEdge reformats that only inflate
  `progression.py` (~1.8k); extract edge data if needed.
- [ ] Work-queue / tracker export reads graph verification + dwell ranks.
- [ ] Planner “next pure” CLI: unified path summary + SOURCE suggest together.

### 6. Hygiene (from root `ARCHITECTURE_AND_CLEANUP_PLAN.md`)

- [ ] Keep `legacy/` and `dev/` (door-warps) strictly fenced.
- [ ] Normalize artifact naming (semantic states, not opaque).
- [ ] Extract shared adventure patterns into `adventure_common` **only after**
  SM + ALTTP both prove the abstraction (room/door graphs, inventory prereqs,
  event flags, path replanning).
- [x] Remove thin warehouse helper aliases (`_hold = hold`) in warehouse.py;
  call shared helpers directly (other kpdr modules still use aliases).

These keep Segment / HopExecutor / ContinuousSession contracts while making the
spine cheaper to extend and run. Detail: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Agent system plan (Luna / Flash / planner)

Existing system already limits dark poking:

| Role | Owns |
|------|------|
| **Luna** | Tests + controller scaffold + bounded geometry with named continuous-like source |
| **Flash** | Tracker/docs/dwell reports + STATUS **proposals** |
| **Planner** | STATUS apply, continuous composition, natural-entry judgment, integrity |

Cards are atomic (recipe step, own-files only, one-knob, pure-first, residual
schema). Dispatch: `dispatch_opencode.sh` / `farm_room_waves.sh`.

### Targeted process improvements

- [ ] Stronger pre-dispatch schema validation + auto-skeleton residual.md.
- [ ] Mandatory residual metrics: frames, dwell, exact pose/x/y/`door_transition` pin.
- [ ] Auto-suggest source state from room + required capabilities.
- [ ] Ownership / file-locking declarations so parallel waves stay safe
  (extend existing dispatch conflict check).
- [ ] On RED pure: richer diagnostics (replay clip, PLM/door RAM snapshot).
- [ ] Keep continuous / STATUS / hot modules (`business_climb`, `varia_return`,
  spore, etc.) serialized or planner-only.
- [ ] Scale dual-track room-segment farming while the spine advances; Luna
  clears non-interacting rooms/combat units in parallel.

**Do not** relax pure-first / one-knob / residual rules.

---

## Roadmap: run-to-Kraid-and-beyond → whole-game M8

Follow the maturity ladder and [BOSS_PIPELINE.md](BOSS_PIPELINE.md). Natural
entry is non-negotiable for continuous evidence.

### Immediate (product tip + structure)

- [x] Continuous reverse spine through Business + Frog Save (K4.0 tip).
- [x] K4 repath decision: first Bubble = **Cathedral climb** (no Speed);
  Speedway→Farm parked until post-Speed.
- [x] Cathedral pure CATH-01 / CATH-02 green from continuous-like sources.
- [ ] **★ K4 forward product:** CATH-03 → Bubble pure stack, then Speed →
  Wave → Ice → Alpha PB (`controller_dev` first; continuous only after
  power-on integrity). Recipe: pure → graph → compose → stabilize.
- [x] **Structure (planner-serial):** tip-spec post-Supers composition, graph
  API collapse, checkpoint flags, Warehouse/Zeela lineage hygiene, cache-local
  parse stats — remaining: hop-table extract, typed path summary, full-run
  profile, full-parse linter. Cards in [`tasks/QUEUE.md`](tasks/QUEUE.md).
- [ ] Re-record continuous tips after each stabilize wave that touches spine
  knobs; promote only after integrity + multi-run dwell honesty.

### Near-term continuous spine (K4 → ship)

1. Pure Cathedral → Bubble, then continuous tips through Speed / Wave / Ice.
2. K5 Alpha PB (natural) → ship access + natural Phantoon entry.
3. Sequential bosses per pipeline (Phantoon → Botwoon → Draygon → Ridley → …
   → Mother Brain + escape/credits). Each requires natural doorway entry on the
   continuous chain, BossCatalog + strategy, closeout, then continuous
   promotion. **Kraid/Varia is the living template.**

### Parallel dual-track

- Room practice / policies for remaining critical-path and high-value rooms
  ([ROOM_WORK_QUEUE](routes/ROOM_WORK_QUEUE.md)).
- Combat unit scaffolds and shared primitives (no full fights before natural entry).
- Clean track: Morph green; ★ bombs/Torizo — [CLEAN_TRACK.md](CLEAN_TRACK.md).
- Early Spazer + 100% (opened 2026-08-03): parallel walljump Spazer detour
  (`0xA447`) → secondary tip → continuous fold; 100% board scaffold —
  [tasks/SPAZER_EARLY.md](tasks/SPAZER_EARLY.md). Does not block K4 Bubble.
- Non-blocking side content (Crocomire, Golden Torizo, etc.) only after main
  spine advances.

### Maturity targets

| Gate | Target |
|------|--------|
| **M6** | Complete route graph with owners/predicates |
| **M7** | Continuous dry-run invariants (full power-on → credits path; resource assists only) |
| **M8** | Verified capture + ending evidence |

### Medium-term program

- Once SM graphs + inventory/event handling are solid (and ALTTP is advancing),
  promote `adventure_common`.
- **Clean track (opened 2026-08-01):** parallel Bronze/Clean continuous tips —
  no energy, no ammo — starting at Morph → **Bomb Torizo**. Contract + tickets:
  [CLEAN_TRACK.md](CLEAN_TRACK.md). Does not replace assisted M8 target.
- Observation class migration (Bronze → Silver) as a separate workstream after
  continuous reliability.
- Broader retro_rl horizons (Final Fight M8, platformers, NES parity) must not
  dilute the SM continuous spine.

**Prioritization heuristic:** close active trunk bottlenecks first, prove shared
packages, prefer natural-entry + clean evidence (same as root ROADMAP).

---

## Track A — end-to-end room-tour video (A0–A2 core done)

**Deliverable:** `recordings/full_route_tour.mp4` (+ JSON report) that:

1. Starts from a known state (power-on continuous prefix *or* Ceres/Landing boot).
2. Walks **all 22 completion legs** using `maps/full_route_hops.json`.
3. Visits ~107 rooms; holds each room briefly so the video is watchable.
4. Skips boss fights via `skip_boss` / `mark_all_major_bosses`.
5. Ends at Landing Site finish `0x91F8` (and, if cheap, idle into ship/credits
   probe — not required for “all rooms”).
6. Manifest marks every progression write and every state load.

### A0 — Wire full hop runner (1–2 days) — **done**

- [x] Extend `dev/route_dev.py` to load `full_route_hops.json` (not only late).
- [x] Define `FULL_LEG_ORDER` = all 22 legs from `completionSequence`.
- [x] Handle the **one null door hop**: Ceres ship `0xDF45 → 0x91F8`
  substituted with door `0x896A` (Parlor→Landing Site) via
  `NULL_DOOR_SUBSTITUTES`.
- [x] At each item/boss anchor, apply progressive flag/loadout via
  `apply_anchor_progress` (early legs) + full loadout after Morph.
- [x] CLI: `probe_route.py full` / `full-tour [--video PATH]
  [--frames-per-room N] [--report PATH]`.
- [x] Tests: hop chain room continuity for all 22 legs; null-door documented
  (`tests/test_route_dev.py`, 12 tests).

**Acceptance met:** `probe_route.py full` returns success through
`landing_site_finish` with hop success on every non-null door (full emulator
run of all 22 legs).

### A1 — Early/mid loadout gates on the tour (1 day) — **done**

Door colors and elevators need inventory/events even when fighting is skipped.
Implemented as progressive `apply_anchor_progress`:

| Anchor leave | Grant / set (dev) |
|--------------|-------------------|
| morph_ball | Morph bit |
| first_missile | Missile capacity ≥ 5 |
| bomb_torizo | Bombs + Torizo bit |
| spore_spawn | Spore bit + Super capacity |
| early_power_bombs | PB capacity |
| kraid | Kraid bit + Varia if needed for heat later |
| speed_booster / ice_beam | Speed, Ice (and beams as hop table needs) |
| phantoon…ridley | Existing `ROUTE_ITEMS` / `ROUTE_BEAMS` + boss bits |
| mother_brain | Event 0x0E + escape-room placement |

### A2 — Record the tour video (1 day) — **done** (core path + hybrid)

- [x] Frame writer on every hop settle + short walk/idle in-room
  (`--frames-per-room`, default 36).
- [x] Hybrid splice: continuous `start_to_supers` prefix *then* resume
  from Super room with warps for the rest
  (`probe_route.py full-hybrid` → `recordings/full_route_hybrid.{mp4,json}`).
- [x] Report: per-leg room list, hop success, frames, flags written, label
  `developmentOnly: true`.

**Acceptance:** `probe_route.py full-tour` writes
`recordings/full_route_tour.mp4` + `.json` by default. Hybrid path:
`full-hybrid` concatenates continuous Super prefix + warp suffix. Both are
development-only — not continuous acceptance.

### A3 — Escape / credits glance (optional, 1–2 days)

Not required for “all rooms,” but cheap polish for the same video:

- [ ] After MB skip: place in Escape 1 pipes, warp Escape 1–4 → Climb → Parlor → LS.
- [ ] Probe ship interaction / game-state ending; if credits RAM is reachable
  without a real MB fight, append a short credits clip. If not, stop at LS
  and document credits as Track B.

---

## Track B — continuous room spine (after A0 skeleton)

Replace door-warps with natural room policies **in order**. Bosses remain
skippable in a “spine dry run” mode until fight scripts land; final acceptance
requires real fights.

### B1 — Continuous Super → KPDR Brinstar (no early Pink PB)

Natural suffix from verified Spore Super entry — **KPDR K0–K1**:

```text
0x9B5B Super collect → 0xA0A4 → 0x9D19 Big Pink (+ Charge)
  → GHZ 0x9E52 → Noob 0x9FBA → Red Tower 0xA253
```

Authoritative board: [ROUTE_KPDR.md](routes/ROUTE_KPDR.md).
Legacy Pink-PB / ship-first notes: [ROUTE_SUPERS_TO_PHANTOON.md](archive/routes/ROUTE_SUPERS_TO_PHANTOON.md) (archived).

- [x] Super shaft descent + Chozo collect (capacity 0 → 5) from
  `natural_post_spore_spawn` — `kpdr.play_super_room_collect`.
- [x] Continuous power-on → Super collect dry report
  (`recordings/start_to_supers.json`, **73,251** frames after Spore fight
  re-record + Climb early-fall splice; was 92,424 then 74,421).
- [x] Opt-in continuous room timing seam
  (`continuous.py --to supers|red_tower --room-timing`).
- [x] Climb early-fall splice in `pit_to_post_torizo`: drop policy
  `[2138:3308)` thrash loops (left-wall peak ~y=1970 → fall to y=2067);
  Climb dwell **4,339 → 3,169**; continuous Supers integrity green.
- [ ] Next timing experiments (pure nav): Parlor→Terminator (3,350),
  Parlor→Flyway (2,627), Green Elev→Dachora (2,660).
- [x] Bottom gate bomb + door shot → farming `0xA0A4` (dev from post-Spore).
- [x] Farming green Super door → Big Pink `0x9D19` (dev).
- [x] Big Pink farm-pocket **crest** + tunnel → main x≲750 (dev controller).
- [x] **Continuous power-on through farming / Big Pink main / GHZ / Noob /
  Red Tower** (`continuous.py --to red_tower`, **80,445** frames, integrity green).
- [x] Collapse per-tip `start_to_*.py` record scripts into one
  `scripts/record/continuous.py --to <tip>` + `run_to()` dispatcher.
- [ ] **Charge Beam** return in Big Pink (natural collect works; a conventional
  return to the route is not ready; do not route an infinite bomb jump).
- [x] Natural Big Pink main → GHZ green door controller-only.
- [x] GHZ → Noob → Red Tower controller-only (no Pink PB): upper Noob
  pit-block bridge, GHZ pillar/blue-gate shot, and both natural door spawns
  compose. Natural Big Pink→Red composition is 3,478 frames.
- [x] Pink PB maze work (partial) **parked** — not required for KPDR; first PB
  is **Alpha `0xA3AE` after Ice** (segment K5).
- [x] Dev ship bridge (`skip-to-red` / `ship-route`) kept as topology only.

### B2 — KPDR safety order: Hi-Jump → Kraid → Speed/Wave/Ice → Alpha PB → Phantoon

**Chosen continuous order: KPDR (B2b-style), not ship-first.**

```text
Red Tower → Warehouse → Business Center → Hi-Jump 0xA9E5
  → Warehouse → Kraid → Varia
  → Bubble Mountain → Speed → Wave → Ice
  → Alpha PB 0xA3AE → elev → Moat → WS → Phantoon → Gravity
```

- [x] K2 prefix: Red Tower → Bat → Below Spazer → West/Glass/East →
  Warehouse Entrance (continuous **83,512f**).
- [x] K2 safety detour: Warehouse→Business→Hi-Jump Shaft→Hi-Jump Room;
  collect the E-Tank and Boots from real PLMs (continuous Hi-Jump **87,696f**).
- [x] K2 return: Hi-Jump ledges→ordinary bomb tunnel→Business→Warehouse.
  No infinite bomb jump is used or required.
- [x] K2 approach: three-Super Warehouse wall→Zeela→Kihunter→Baby
  Kraid→Eye Door→natural Kraid-room entry (continuous Kraid **~97k**).
- [x] K3 boss-only: Super-spray fight + rear door + real Varia PLM from
  doorway entry (`play_kraid_fight_to_varia`).
- [x] K3 continuous: power-on → Varia Suit (`--to varia`, **104,382f**, integrity green).
- [x] K4 return pure: reverse hops post-Varia → Business (9,343f from accepted Varia checkpoint).
- [x] K4 continuous: power-on → Business return (`--to business`, 113,723f twice, integrity green).
- [x] K4.0 forward: Business → Frog Save (`--to frog`, 114,923f twice, integrity green).
- [x] K4 repath: first Bubble = Cathedral (CATH-01/02 pure green).
- [ ] K4 forward pure + continuous: CATH-03 → Rising Tide → Bubble → Speed → Wave → Ice.
- [ ] K4 post-Speed shortcut (parked): Frog Save → Speedway → Farm → Bubble.
- [ ] K5: Alpha PB collect (preferred first Power Bombs).
- [ ] K6: Moat / Ocean / WS / Phantoon / Gravity by play.
- [ ] Document / promote KPDR edges in `progression.py` as verification advances
  (`controller_dev` → `continuous`).

Ship-first / PRKD remains out of continuous scope; hop-table warps stay Track A.

### B3 — In-room policy factory (parallel)

Stop hand-authoring only:

1. Scaffold from catalog waypoints (`run_room_problem.py scaffold`).
2. Replay from natural entry state captured from predecessor.
3. Promote to `verified_development_state` then to continuous graph edge.
4. Priority queue = **rooms on the completion path first** (~107), not all 262.

High-value path rooms (from hop table + known hard geometry):

| Priority | Rooms / legs |
|----------|----------------|
| P0 | Super room, PB room, Red Tower, Moat, West Ocean, WS shaft |
| P1 | Warehouse / Kraid approach, Norfair heat halls if on continuous path |
| P2 | Maridia sand / Botwoon hall, LN exit, Statues, Tourian metroids |
| P3 | Escape pipes / Climb / Parlor return |

### B4 — Boss scripts (pipeline in [`BOSS_PIPELINE.md`](BOSS_PIPELINE.md))

**Rule:** no new boss fight until natural entry to that room exists on the
played continuous chain. Never write boss/event/item RAM to claim a win.

**Phase 0 foundations (do before next fight code):** full `BossCatalogEntry`
registry, `BossStrategy`/`BossEvidence` protocol, generalized natural-entry
capture, shared `combat/primitives.py`, continuous tip hooks, probe CLI
template. Living template: Kraid fight → rear door → Varia.

| Priority | Boss | Dev status | Continuous need |
|----------|------|------------|-----------------|
| 0 | Bomb Torizo | Continuous (replay) | done |
| 0 | Spore Spawn | Continuous | done |
| 1 | **Kraid** | Fight + Varia continuous | **done** (`--to varia`, 101,954f) |
| 2 | Phantoon | entry state only | fight + WS power restore |
| 3 | Botwoon | skip bit only | fight |
| 4 | Draygon | skip bit only | fight + Space Jump collect |
| 5 | Crocomire | skip / side | fight (acid push; non-HP) |
| 6 | Ridley | skip bit only | fight |
| 7 | Golden Torizo | optional / side | multi-phase practice |
| 8 | Mother Brain | room entry + spray probes | zebetite, phases, escape init |
| — | Escape → credits | warp hop chain | timer, ship, ending/credits RAM |

Per-boss checklist and promotion criteria: [`BOSS_PIPELINE.md`](BOSS_PIPELINE.md).

### B5 — Full continuous dry run → verified capture

Promotion order (same as historical Phase 6):

1. Segment from natural entry
2. Multi-milestone suffix
3. Power-on dry run with boss skips (spine integrity)
4. Power-on dry run with real bosses
5. Credits evidence + video (M8)

---

## Gap checklist (what blocks “done”)

| Gap | Blocks | Track |
|-----|--------|-------|
| ~~Full hop runner not wired~~ | — | **A0 done** |
| ~~Ceres ship null door~~ (sub `0x896A`) | — | **A0 done** |
| ~~Progressive loadout on early tour legs~~ | — | **A1 done** |
| ~~Tour video recorder~~ (`full-tour`) | — | **A2 done** |
| ~~Hybrid continuous-prefix + warp tour~~ | — | **A2 done** (`full-hybrid`) |
| ~~Natural Super collect~~ | — | **B1 done** (continuous) |
| ~~Continuous Super → Red Tower → … → Varia~~ | — | **B1–B2 done** (K3 tip) |
| ~~Post-Varia reverse pure → Business~~ | **Done:** `start_to_business` 113,723f ×2 | B2 |
| ~~K4.0 Frog Save continuous~~ | **Done:** 114,923f ×2 | B2 |
| K4 Cathedral → Bubble pure + continuous tips | Continuous past Norfair items | B2 |
| Natural ship + Phantoon entry | Continuous to WS | B2 / B4 |
| Remaining path room policies | Continuous room running | B3 dual-track |
| Boss fight scripts past Kraid | True clear | B4 |
| Escape timer + credits predicate | M8 ending | B4–B5 |
| ~~Clone Super+ tip runners~~ | — | **partial:** `PostSupersTipSpec` landed; hop extract open (§2) |
| ~~Twin graph path APIs~~ | — | **partial:** collapse landed; typed summary open (§5) |
| ~~Lineage special-cases (Warehouse/Zeela)~~ | — | **done** (entry_mode + named phases, §4) |
| Selective-RAM profile / pure RED clip | Whole-game efficiency | structure plan §1–3 |

---

## Recommended execution order (next waves)

Live triage + dispatch: [`tasks/TRIAGE.md`](tasks/TRIAGE.md) ·
[`tasks/WAVE-11.md`](tasks/WAVE-11.md) · [`tasks/QUEUE.md`](tasks/QUEUE.md).

```text
Track A topology — DONE (stop expanding warp product work)
  ✓  full hop runner + full-tour / full-hybrid diagnostics

Track B — play every path room (PRIMARY)
  ✓  Continuous Kraid + Varia (K3 tip @ 104,382f)
  ✓  Continuous Varia return → Business (K3→K4 tip @ 113,723f ×2)
  ✓  Continuous Business → Frog Save (K4.0 tip @ 114,923f ×2)
  ✓  Cathedral repath; CATH-01/02 pure green
  NOW  ★ SM-K4-CATH-03 pure → CATH-04 Bubble → Speed/Wave/Ice pure stack
       → graph → compose continuous tips → stabilize → STATUS
  THEN K5 Alpha PB → Moat → WS → natural Phantoon + Gravity
  Parallel dual-track: room farm · Clean bombs · Early Spazer/100% · 1–2 ARCH · boss primitives
  ARCH priority: hops module extract, selective-RAM gate, pure RED diag
  Parked: Speedway→Farm until post-Speed (Boost Blocks)

Later   Boss policies: Phantoon → Botwoon → Draygon → Ridley → MB + escape
Maturity M6 graph owners → M7 full dry-run invariants → M8 credits capture
```

Do **not** door-warp past open hops to fake progress. Measure furthest played
room; fix that hop; repeat. Boss work follows [`BOSS_PIPELINE.md`](BOSS_PIPELINE.md).
Live dispatch board: [`tasks/TRIAGE.md`](tasks/TRIAGE.md).

---

## Phase 0 — contract and scaffold

- [x] Record ROM path and hash.
- [x] Define allowed resource writes and forbidden progression writes.
- [x] Define continuous completion at ending/credits, not final-boss HP zero.
- [x] Create the integration files, ROM link, typed state, and tests.
- [x] Choose and document the initial start condition: `retro.State.NONE`,
  fresh file A selected through the title flow.

Acceptance met: the integration boots the expected ROM and the contract is
represented in tests and report fields.

## Phase 1 — boot and core RAM

Map with probe evidence:

- game/menu/control mode
- area, room, door/elevator transition
- player X/Y, velocity, pose, grounded/control flags
- current/max energy and reserves
- current/capacity for each ammo type
- equipment/item bitsets
- boss/event/collected-item bits
- death/game over
- ending/credits state

Use the continuous reset boot trace as acceptance evidence. Development states
may be added later, but are not part of the accepted route.

Acceptance met: repeated reset runs reach the same Ceres control predicate at
frame 10,860 without a state load.

## Phase 2 — route graph and first natural suffix

Represent milestones as data:

```text
milestone
  entry predicate
  required inventory/events
  room/door target
  policy owner
  completion predicate
  timeout
  recovery state
```

Start with:

1. power-on/menu → first controllable Ceres room
2. Ceres traversal → escape/transition
3. Zebes arrival → first required upgrade
4. first upgrade → first ammo unlock
5. first ammo unlock → next route gate

Prefix acceptance met through Morph Ball from the state produced by every real
predecessor. Continuous acceptance later extended through Bomb Torizo and
Spore Spawn (see STATUS).

## Phase 3 — navigation primitives

Build only primitives demonstrated by two or more rooms:

- approach and activate door/elevator
- run/jump across a room
- recover from wall, ledge, and platform stalls
- aim/shoot a door or obstacle
- traverse vertical shafts
- select and use naturally unlocked ammo
- fight or bypass an enemy
- boss-specific policy

Watchdogs use room/door/inventory/event progress, not player coordinates alone.
Every recovery action has a bounded budget and a regression state.

## Phase 4 — route expansion

Grow verified suffixes through:

- early required movement/combat upgrades
- early bosses and major area transitions
- midgame traversal/backtracking
- late-game access requirements
- final area and bosses
- endgame escape
- ending/credits

Maintain a route-requirement table. An item or boss flag is considered
required only when a real transition demonstrates the dependency.

## Phase 5 — assist validation

Before long chains:

- verify energy refill never changes maximum energy or item flags
- verify every ammo type stays locked at zero capacity until collected
- verify refill stops during transitions, menus, death, and scripted sequences
- verify damage and ammo use are measured before refill
- verify progression-write count remains zero

Test ordinary combat, environmental damage, an ammo door/obstacle, a room
transition, a boss transition, and a scripted sequence.

## Phase 6 — chain and full dry runs

Promotion order:

1. segment from clean state
2. segment from natural entry
3. two-milestone suffix
4. area suffix
5. late-game suffix through ending
6. full power-on dry run
7. final capture

Candidate reports and logs must not overwrite the last successful baseline.
Abort early on milestone timeout, route regression, forbidden write, invalid
assist write, or prolonged no-progress.

## Initial metrics

- completion milestone and furthest room
- total frames and split time per milestone
- room/door transitions
- item and boss/event acquisition frames
- deaths
- energy restored and write count
- ammo restored/writes by type
- action-reason counts by room/segment
- maximum no-progress interval
- state loads and progression writes

## Implementation checklist

1. [x] Scaffold the integration around `roms/SuperMetroid.sfc`.
2. [x] Boot headlessly and identify the first controllable frame.
3. [x] Populate `docs/ram_map.md` with source and live-route evidence.
4. [x] Implement phase-guarded, capacity-preserving unlimited ammo.
5. [x] Clear all of Ceres continuously from power-on.
6. [x] Continue from the natural Zebes entry through Morph Ball.
7. [x] Extend through both early Missiles, Climb return, and Bomb Torizo/Bombs.
8. [x] Extend post-Torizo through Terminator/Green Brinstar, defeat Spore
   Spawn, and exit naturally to the Spore Super room.
9. [x] Merge full reference topology and editor geometry into 262 canonical
   room-development problems.
10. [x] Validate save-state teleport and natural target-room settlement on two
    queue-1 door clears plus Flyway.
11. [x] **Late route skeleton (dev, fights skipped):** Phantoon → Gravity →
    Botwoon → Draygon → Ridley → Statues → Tourian → MB → Escape → Landing
    Site (`dev/route_dev.py`, `maps/late_game_route_hops.json`).
12. [x] **Kraid defeated (dev):** Super spray; state `dev_kraid_defeated`.
13. [x] **Power Bombs (dev):** door-warp collect → `dev_power_bombs_collected`.
14. [x] **Ship route → Phantoon entry (dev):** `dev_phantoon_entry`.
15. [x] **A0** Full 22-leg hop runner from `full_route_hops.json` (Ceres null
    door → `0x896A`).
16. [x] **A1–A2** Progressive loadout + `full-tour` + `full-hybrid`
    (continuous Super prefix splice; bosses skipped; `developmentOnly`).
17. [x] **B1** Continuous Super → Red Tower → Warehouse (early KPDR spine).
18. [x] **B2 K2–K3** Continuous Hi-Jump → Kraid entry → Varia Suit.
19. [~] **B2 K4** Post-Varia reverse pure + continuous K4 (Bubble→…→Alpha PB).
20. [ ] **B2 K5–K6 / B3** Ship rooms + path-priority room policies (dual-track).
21. [~] **B4** Boss fights: Kraid continuous done; Phantoon → … → MB + escape open.
22. [ ] **B5** Continuous dry run → credits video (M7/M8).
23. [~] **Structure plan** Partial: source catalog, scaffold, pure pins,
    tip-spec post-Supers composition, graph API collapse (`path_summary` /
    `suggest_edges`), checkpoint flags, Warehouse/Zeela lineage, cache-local
    parse stats. **Open:** hop-table extract, typed path summary, full-run
    RAM profile / pure RED clip / linter (see Structure & API plan above).
