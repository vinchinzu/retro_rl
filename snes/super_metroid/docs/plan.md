# Plan — Super Metroid assisted full clear

Verified facts: [STATUS.md](STATUS.md). Shared workflow:
[`docs/FULL_RUN_PROCESS.md`](../../../docs/FULL_RUN_PROCESS.md).
Assist: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Layers: [ARCHITECTURE.md](ARCHITECTURE.md).
Executor: [tasks/PROCESS.md](tasks/PROCESS.md) · `bd ready -l super_metroid`
([tasks/QUEUE.md](tasks/QUEUE.md) snapshot).

## Strategy

Unlimited energy and ammo make combat attrition secondary. The hard problem is
long-horizon navigation: room identity, doors/elevators, item gates, movement
abilities, boss/event state, backtracking, and stall recovery.

**Clear rooms by play.** Each hop on the completion path must be crossed with a
controller or room policy (natural door exit). Door-warps are topology
diagnostics only — never route evidence. Living hop board:
[research/PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md).

Grow one hop at a time from the furthest played room. Continuous tip extension
recipe ([ARCHITECTURE.md](ARCHITECTURE.md)):

```text
pure → graph edge → catalog → hops → continuous compose → dual re-verify → STATUS
```

**Boss fights stay deferred** until natural *entry* to that boss room exists on
the played chain. Continuous acceptance still requires natural boss flags and
zero progression writes. Pipeline: [BOSS_PIPELINE.md](BOSS_PIPELINE.md).

**Agent discipline (non-negotiable):** pure-first, one-knob residual, residual
schema with next-card ID + one change, dual-track (spine continuous vs room
practice). Do not relax for scale. See [tasks/PROCESS.md](tasks/PROCESS.md).

**Ticket size:** one pure hop or one residual change per card; prefer 30–90 min
sessions. STATUS/docs updates are planner-owned or tiny follow-ons.

---

## Current focus

| Priority | Work | Beads |
|----------|------|-------|
| **★ P0 stabilize** | Stabilize wave after Speed continuous | `rr-07b` |
| **★ P1 pure** | Pure Speed return → Bubble | `rr-g4i` |
| P0 done | Continuous `--to speed` dual + STATUS promote | `rr-d20` ✓ / `rr-cd0` ✓ |
| P0 pure (done) | Bat→Hall + Hall→Speed collect pure green | closed |
| P0 done | Spazer warehouse dual + STATUS promote | `rr-jx9` ✓ / `rr-4wg` |
| P0 done | HJ Spazer pillar unequip fix; Business→Cath door KB escape | closed |
| P1 pure stack | Wave → Ice (pure green each) after Speed return | after `rr-g4i` |
| P1 optional | Dual Spazer `bat_cave` tip STATUS (historical single 127,806f) | optional |
| P1 | K5 **Alpha PB** (natural post-Ice) | later |
| Parallel Clean | ★ **bombs / Torizo Clean** — [CLEAN_TRACK.md](CLEAN_TRACK.md) | `rr-siz` |

**Parked:** Frog Save → Speedway → Farm → Bubble (needs Speed / Boost Blocks).
**Do not:** re-open CATH pure, Bat→Speed pure, or Speed dual (green residuals +
STATUS tip promoted); do not claim pure Speed return until `rr-g4i` green.
Default continuous tip is **`speed`** (**130,388f** ×2).

Live work: `bd ready -l super_metroid` · snapshot [tasks/QUEUE.md](tasks/QUEUE.md).
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).
---

## Ceres classic L↔R arm-pump (opt-in shave) — notes 2026-08-05

**Policy:** when a faster prefix desyncs a later leg, **re-solve with WRAM** —
do not blind-restore product open-loop. Speed every section; re-pin tails.

**Code:** `routes/kpdr/early_spine.py` — `_ceres_arm_pump_*`,
`_ceres_reactive_magnet_escape` / `_ceres_reactive_falling` /
`_ceres_reactive_elev_climb` / `_ceres_elev_top_to_ship`; `play_ceres_*` on
morph spine. BB elev downstream: 1f `$0E16` parity + reactive board fallback
in `play_elevator_to_morph_room`. Unit: `tests/test_ceres_arm_pump.py`.

### Verified facts

| Fact | Detail |
|------|--------|
| Product morph | GREEN **27,074f**; ridley **16,414**; zebes_landing **21,799** |
| Arm-pump outbound | Ridley **16,181** (~**233f** shave) |
| Falling→elev | Mid-trans y≈139 fake; **gs=8 → bottom y≈651** |
| Ledge pin | **y=571 pose=2**; s0 re-solve = **ledge walk LEFT** (not blind LEFT+A 70) |
| Shaft s2–s10 | Product-like band → land **~x189 y171 pose 9** (short of wall) |
| **SM-CERES-ELEV-TOP (solved)** | Walk RIGHT → **x211 y171 pose 137**, idle plant, product **LEFT+A 38 + LEFT 25** through pad **x≈145 y75** → **gs 32** leave |
| Arm-pump morph | Dual GREEN **26,824f** ×2 (`morph_arm_pump_probe.json` + `_reverify.json`) |
| Shave vs old product | **250f** (27,074 → 26,824); BB elev needs 1f `$0E16` parity (flag toggles/frame) |

### STATUS

Morph tip frames updated to **26,824** after dual integrity-green. No further
Ceres elev residual — next opt-in is more outbound arm-pump (optional).

### Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --no-video
```

---

## Finish Spazer K2.2 (mainline — always collect)

**Product path:** `play_below_spazer_to_west` → `play_spazer_detour` always when
Spazer missing (floor entry included). **No Charge-only West skip.** Continuous
warehouse without Spazer bit is RED until residual pure is green — intentional.

**Done (do not re-prove):**

| Fact | Evidence |
|------|----------|
| Charge on continuous K1 | `play_big_pink_to_ghz`; `below_spazer_with_charge.json` **84,880f** |
| Spazer door / collect / return pure | `below-spazer-to-spazer`, `spazer-collect`, `spazer-return-to-below` |
| Mid band → West pure | `spazer-top-to-west` from y≥220 |
| Mainline wired | `play_below_spazer_to_west` always → `play_spazer_detour` |
| Historical Charge-only West | `warehouse_with_charge.*` **85,992f** beams `0x1000` — **not** product |

**Done this epic:**

1. Climb / top→West / morph-tunnel Super door / pure detour — **GREEN**.
2. **SM-SPAZER-CONT** — continuous `--to warehouse` **GREEN** **89,416f**,
   beams **`0x1004`**, integrity 0 loads/prog/deaths. Floor Cacatac clear
   (Charge-cadence UP+X) before spin — spike knockoff was continuous fail.
   Video: `recordings/warehouse_with_spazer.mp4` (from supers frame).
3. **SM-SPAZER-CONT dual** (`rr-jx9`) — second warehouse integrity match
   **90,904f**, beams `0x1004`, room `0xA6A1`, outcome `warehouse_entry`
   (`warehouse_with_spazer_dual.json`). Frame +1,488 = Spore combat variance.
4. **SM-SPAZER-STATUS** (`rr-4wg`) — STATUS/MILESTONES warehouse Spazer dual
   promoted (2026-08-06). Later folded into Speed dual STATUS (`rr-cd0`).
5. **Speed dual + STATUS** (`rr-d20` / `rr-cd0`) — continuous `--to speed`
   **130,388f** ×2 exact match, beams `0x1004`, items `0x3105`, room
   `0xAD1B`; `DEFAULT_CONTINUOUS_TIP = speed`.

**Optional / later:** dual Spazer `bat_cave` tip STATUS alone (single
**127,806f** in `bat_cave_spazer_cwu.json`) — superseded by Speed dual tip.

**Sources:** [tasks/EARLY_SPAZER_HUMAN.md](tasks/EARLY_SPAZER_HUMAN.md) ·
[tasks/SM-SPAZER-HUMAN-CHUNKS.md](tasks/SM-SPAZER-HUMAN-CHUNKS.md).

---

## Open work by epic

### K4 remaining (Norfair items)

- [x] Pure **Bat → Speed Hall** (residual GREEN)
- [x] Pure Speed Hall → Speed Booster room + collect (residual GREEN)
- [x] Spine graph edges continuous for Speed tip (`--to speed` wired)
- [x] Continuous compose + dual re-verify for `speed` (`rr-d20`, 130388f ×2)
- [x] STATUS promote `speed` (`rr-cd0`, default CLI tip)
- [ ] Stabilize wave after Speed continuous (`rr-07b`)
- [ ] Pure Speed return → Bubble (`rr-g4i`) → Wave → Ice chain
- [ ] Continuous compose + dual for `wave` / `ice` tips
- [ ] (Parked) Speedway → Farm → Bubble post-Speed shortcut

### K5 — Alpha PB

- [ ] Natural Alpha PB `0xA3AE` collect after Ice (not Pink PB)
- [ ] Continuous tip + integrity dual-run
- [ ] Graph verification promotion

### K6 — Ship / Phantoon / Gravity

- [x] Moat shinespark pure from `post_kihunter_pre_moat_spark` → West Ocean
  — residual [tasks/SM-MOAT-SHINESPARK-residual.md](tasks/SM-MOAT-SHINESPARK-residual.md)
  (store→spin→UP unspin→spark + RIGHT+X door; probe hop + controller pure GREEN;
  West handoff `scratch/post_moat_west_ocean_spark.state`; harness B=dash A=jump;
  **not** continuous / STATUS)
- [ ] Moat → West Ocean → Wrecked Ship by play
- [ ] Natural Phantoon entry → fight → Gravity
- [ ] Continuous tips only after natural doorway entry

### K7 — Maridia

- [ ] Tube break → Botwoon → Draygon → Space Jump (natural entry each)

### K8 — Lower Norfair / Ridley

- [ ] LN entry → Ridley natural entry + fight

### K9 — Tourian / MB / Escape / Credits

- [ ] G4 statues → Tourian → Mother Brain (zebetites, phases)
- [ ] Escape timer + geometry + ship / ending-credits evidence (M8)

Boss order and phase rules: [BOSS_PIPELINE.md](BOSS_PIPELINE.md). Template:
Kraid → Varia continuous.

### PRACTICE (dual-track, planner opt-in)

- [ ] `ROOM_WORK_QUEUE` + `farm_room_waves.sh` when planner opts in (not default P0)
- [ ] Combat unit scaffolds (no full fights before natural entry)
- [ ] Early Spazer walljump detour + 100% board (parallel; does not block K4)
  — [routes/TRACK_100.md](routes/TRACK_100.md)

Practice greens ≠ continuous evidence and **not** product next-work
(`STATUS` + `tasks/QUEUE.md` own that). Own-files only; width ≤ 8.

### CLEAN (parallel)

- [x] Morph Clean continuous green (was 27,074f; assisted tip now 26,824f — clean re-verify open)
- [ ] ★ Bombs / Torizo Clean (`SM-CLEAN-BOMBS`)
- [ ] Later Clean tips only after bombs green
- Never mutate default CLI assists or demote assisted baselines

### ARCH (planner-serial on hot modules)

Highest leverage for whole-game length — see Structure below and
[ARCHITECTURE.md](ARCHITECTURE.md).

### Maturity targets

| Gate | Target | Notes |
|------|--------|-------|
| **M5** | Bronze observation; resource-assisted continuous tip | **Current** (Bat Cave) |
| **M6** | Complete route graph with owners/predicates | In progress |
| **M7** | Continuous dry-run invariants (power-on → credits path) | Open |
| **M8** | Verified capture + ending/credits evidence | Open |

Observation-class migration (Bronze → Silver) is a separate workstream after
continuous reliability.

---

## Structure & API (open only)

Current layers (CLI → continuous + catalog + segment → pure kpdr → graph →
ram/assist → combat) are correct. Planner-serial when touching
`continuous.py` / `progression.py` / `catalog.py`.

### Selective RAM + StateCache

- [ ] Profile frame time on long tips / full runs (WRAM-copy rate)
- [ ] Optional linter: forbid bare full `parse_env_state` inside `routes/kpdr/`

### Declarative continuous composition

- [x] Move hop tables out of continuous (`routes/kpdr/hops.py` + tip-spec bind)
- [x] Early morph→supers extracted to `routes/early_continuous.py`; shared
  prefix conditions + room-timing helpers in `runtime`
- [x] Morph/Ceres SpineHop orchestration (`routes/kpdr/early_spine.py`); seeds unchanged
- [x] Controller shims deleted (`kpdr_controller` / `post_spore_controller`)
- [x] Fold bombs/spore/supers play onto SpineHop (`early_post_morph.py`; no timing risk)
- [ ] Optional: further collapse early run_* finish_report boilerplate

### Source-state & pure-probe diagnostics

- [ ] Short video clip + PLM/door RAM snapshot on pure RED
- [ ] Dispatch auto-suggest `--source` from card schema
- [ ] Provenance on checkpoints (parent tip, command, capabilities)

### Controller structure

- [x] Bubble skills extraction (`routes/skills/` + hop policy `bubble_to_bat`; product `to_bat_cave`)
- [x] K4 knockback / Super-door plant frames → `routes/skills/knockback.py` + `door.super_door_pressure_frame`
- [ ] Prefer `wait_ordinary_room` handoff bands (`y_range` etc.) over airborne
  settle hope
- [x] Remove thin helper aliases (`_hold = hold`) from KPDR segment modules
- [x] Rename room-named KPDR segments to hop names (`pink_to_ghz`, `red_stack`, `to_kraid`, …)
- [x] De-nest progression stage tables (`progression/stages/`)
- [x] Continuous `*RunReport` aliases removed; Super+ tip aliases in `post_supers_aliases.py`

### Graph first-class

- [ ] Typed path-summary model (not ad-hoc `dict[str, object]`)
- [ ] Stop mechanical multi-line DoorEdge reformats; extract edge data if needed
- [ ] Work-queue / tracker export reads graph verification + dwell ranks
- [ ] Planner “next pure” CLI: path summary + SOURCE suggest together

### Hygiene

- [ ] Keep `legacy/` and `dev/` (door-warps) strictly fenced
- [ ] Normalize artifact naming (semantic states)
- [ ] Promote shared adventure patterns to `retro_harness.adventure` **only after**
  SM + ALTTP both prove the abstraction

### Agent process improvements

- [ ] Stronger pre-dispatch schema validation + auto-skeleton residual.md
- [ ] Mandatory residual metrics: frames, dwell, exact pose/x/y/`door_transition`
- [ ] Ownership / file-locking so parallel waves stay safe
- [ ] Dual-track room farming remains planner opt-in (never starves serial pure)

**Do not** relax pure-first / one-knob / residual rules.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Long-horizon nav fragility / high-dwell segments | Tighten offline secondary; stabilize after each tip |
| Architecture debt (multi-registry tip wire, full WRAM hot paths) | Highest-leverage ARCH first |
| Residual / card proliferation | Archive-after-successor + one-knob schema |
| Process drift (practice claiming continuous) | Dual-track gates; planner owns STATUS |
| Endgame (Zebetites regen, escape geometry, timer/WRAM) | Deferred until natural entry |

## Non-goals (now)

- Door-warp / hybrid tours as continuous evidence (Track A topology **done** —
  stop expanding warp product work)
- Ship-first / PRKD continuous route
- Pink PB maze (first PB is Alpha after Ice)
- Vision-based boss combat in `legacy/`
- Claiming Clean greens as the program M5/M8 assisted gate
- Full endgame fight code before natural entry

---

## Recommended next waves

```text
★ NOW   Bat → Speed Hall pure → Speed/Wave/Ice pure stack
        → graph → compose continuous tips → stabilize → STATUS
THEN    K5 Alpha PB → Moat → WS → natural Phantoon + Gravity
Parallel Clean bombs · Early Spazer/100% · 1–2 ARCH · boss primitives
Opt-in  Room farm (metrics only; not product next-work)
Parked  Speedway→Farm until post-Speed
Later   Botwoon → Draygon → Ridley → MB + escape → credits (M8)
```

Live dispatch: [tasks/QUEUE.md](tasks/QUEUE.md).
Milestone board: [routes/MILESTONES.md](routes/MILESTONES.md) ·
Backlog: [routes/BACKLOG.csv](routes/BACKLOG.csv).

---

## Pointers

| Doc | Role |
|-----|------|
| [STATUS.md](STATUS.md) | Verified tip + prefix frames |
| [tasks/PROCESS.md](tasks/PROCESS.md) | Pure-first / residual / dual-track |
| [tasks/QUEUE.md](tasks/QUEUE.md) | Live cards |
| [routes/BACKLOG.csv](routes/BACKLOG.csv) | Full ticket inventory |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers + structural debt |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss natural-entry rules |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean track process |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | KPDR route text |
| [research/PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md) | Hop topology |
