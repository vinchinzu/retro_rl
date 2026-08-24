# Plan — Super Metroid assisted full clear

Verified facts: [STATUS.md](STATUS.md). Assist:
[ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Layers:
[ARCHITECTURE.md](ARCHITECTURE.md). Tracker: `bd ready -l super_metroid`.

**Doc consolidation (2026-08-18):** deleted closed/green hop residuals,
`tasks/QUEUE.md`, `tasks/PROCESS.md`, `TASK_TEMPLATE.md`, `CODE_REVIEW.md`,
`docs/research/*` essays, and duplicate route CSVs (`BACKLOG`,
`MILESTONES.csv`, `TRACK_100.csv`). Kept STATUS, plan, ASSIST_CONTRACT,
ram_map, ROUTE_KPDR, MILESTONES.md, KPDR_TRACKER (code-owned CSV), and
the two open-tip residuals (`rr-dbu.8`, `rr-av5s`). Do not recreate a
QUEUE. Do not rewrite the route or claim a new tip.

**Program role:** Super Metroid is **substrate A** of the solver flagship
triangle (SM + ALTTP + SMZ3). Pure room policies and capability edges are Layer 1
skills the shared item-logic planner will sequence; SMZ3 is the seed-abstract
proof. See [`docs/SOLVER_ARCHITECTURE.md`](../../../docs/SOLVER_ARCHITECTURE.md).

## Strategy

Unlimited energy and ammo make combat attrition secondary. The hard problem is
long-horizon navigation: room identity, doors/elevators, item gates, movement
abilities, boss/event state, backtracking, and stall recovery.

**Clear rooms by play.** Each hop on the completion path must be crossed with a
controller or room policy (natural door exit). Door-warps are topology
diagnostics only — never route evidence. Living hop geometry:
[routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md).
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
practice). Do not relax for scale. See `AGENTS.md` (pure-first).

**Ticket size:** one pure hop or one residual change per card; prefer 30–90 min
sessions. STATUS/docs updates are planner-owned or tiny follow-ons.

---

## Current focus

| Priority | Work | Beads |
|----------|------|-------|
| **★ Product next** | WS Main Shaft → basement pure (`0xCAF6` → `0xCC6F`) | `rr-4btp` · residual `tasks/rr-ahjo-residual.md` |
| Done compose + return | Ice tip 11 hops (return + Ice pure); continuous RED climb | `rr-kxge` compose · `rr-dbu.7` wire |
| Done pure return | Wave→Business 7/7 dual GREEN | `rr-vqv3` |
| Done pure stack | Business→Gate→Acid→Snake→Ice PLM dual GREEN | `rr-dbu.11` · hops `rr-fg3` `rr-9t4` `rr-5cf` `rr-5if` |
| Product after Ice tip | K5 → Moat approach | `rr-dbu.8` → `.9` |
| Done | Human full tape 39,711f Wave+Ice+Moat | `rr-dbu.12` · [SM-SPEED-ICE-MOAT-HUMAN.md](tasks/SM-SPEED-ICE-MOAT-HUMAN.md) |
| Optional | Policy consolidate; duck-type; Clean STATUS; speed start Spazer | P3 |
| Parallel Clean | bombs/Torizo Clean **GREEN** 49,321f ×2 | polish `rr-3z8` |
| Done | Wave continuous **136,361f** + hygiene Pass A/B essential | |

**Critical path to credits:** Ice dual continuous is **GREEN**
(`rr-kxge` closed). Power-on `--to moat` is scratch dual-green
(**175526f** ×2 `0x93FE` `(49,1163)` p1, rr-2r06). Power-on `--to ws`
is scratch dual-green (**176141f** ×2 `0xCA08` `(57,139)` p1, rr-p2bw).
Entrance → Main Shaft is scratch dual-green (**403f** ×2 `0xCAF6`
`(1063,907)` p9, rr-ahjo). Next: Main Shaft → basement `0xCC6F` (`rr-4btp`;
`play_ws_main_to_basement` is still a scaffold). Planner STATUS for
`moat` / `ws` is a follow-on — default CLI stays `ice`.
**Do not STATUS-promote past Ice without a planner STATUS pass.**

**Parked:** Frog Save → Speedway → Farm → Bubble (post-Speed shortcut);
spore clean; Pass B.3 deep consolidate.

Default continuous tip is **`ice`** (**148,167f** ×2). Wave / Speed remain
valid prefixes. Hygiene Pass B does **not** block product.

Live work: `bd ready -l super_metroid`.
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).
---

## Ceres opener (TAS boot) + arm-pump — 2026-08-07

**Policy:** when a faster prefix desyncs a later leg, **re-solve with WRAM** —
do not blind-restore product open-loop. Speed every section; re-pin tails.

**Code:** `routes/kpdr/ceres/` + `early_spine.play_boot_to_ceres`. TAS movies under
`tas/ref/` (Sniq any% #3653M). Unit: `tests/test_ceres_arm_pump.py`.

### Improvement table — boot / first Ceres control

| Milestone | Frames | Δ frames | Δ seconds (@ 60.0988) | Notes |
|-----------|-------:|---------:|----------------------:|-------|
| Legacy open-loop boot (`_boot_spans` sum) | 10,860 | — | — | Fixed title idle 2,100f + intro mash |
| Legacy first `gs=8` elev (probe) | 10,642 | — | — | Control during last boot spans |
| **TAS-style mash first `gs=8`** | **8,479** | **−2,163** | **−36.0 s** | START/A period-1 → A-every-other (Sniq pattern) |
| TAS boot hop end (+elev settle +plant) | ~8,572 | **−2,288** vs 10,860 | **−38.1 s** | y 0→72 pad settle required for outbound |
| Sniq any% first B+RIGHT (ref movie) | 8,639 | — | — | lsnes movie; not same core |

### Improvement table — morph spine (published / probe)

| Milestone | Frames | Δ frames | Δ seconds | Notes |
|-----------|-------:|---------:|----------:|-------|
| Product morph (pre arm-pump) | 27,074 | — | — | `morph.json` |
| Arm-pump morph dual GREEN | **26,824** | **−250** | **−4.2 s** | ridley 16,181; landing 21,548 |
| TAS boot → ridley (probe) | 13,671 | **−2,510** vs 16,181 | **−41.8 s** | Outbound holds after settle |
| TAS boot full morph dual | **27,494** ×2 | **+670** vs product | **+11.1 s** | Elev GREEN; BB elev + morph reseed cost |

### Improvement table — Ceres Ridley fight (same enter pin)

Public RTA: [wiki Ridley § Ceres Station](https://wiki.supermetroid.run/Ridley#Ceres_Station) — escape starts at energy **< 30**; five tail hits at the right wall. Probe: `scripts/probe/ceres_ridley_combat.py bench`. Seconds @ 60.0988.

| Policy | Frames | Seconds | Clock | Hits | Notes |
|--------|-------:|--------:|------:|-----:|-------|
| wait (left-door idle) | 3,212 | 53.445 | 00:53.53 | 8 | Previous product (`ceres_ridley_natural_countdown`) |
| **tail_tank** | **1,936** | **32.214** | **00:32.27** | 5–6 | Right wall; first hit f606; countdown same pin |
| Δ | **−1,276** | **−21.232** | −00:21.27 | | Same `ceres_ridley_enter.state` |

**Tail-tank is product** (flag removed). Same-pin fight bench is still
`scratch/ceres_ridley_bench.json`. Elev hop from the tail-tank leave pin
(`0xDF45:0xDF8D->0x91F8:0x0000`, `ceres_elev_enter.state`) is the open
card: takeoff windows live in ``takeoff.PlatformHop`` (shared, every room),
not a Ceres-only hop type and not frame hillclimb. Pin seats 571 / 475 /
363; live climb still no ship.
KPDR Ceres Station goal is **1:35** from first elev control. Do not
STATUS-promote from the pin bench. Probe:
`scripts/probe/ceres_elev_escape.py`. Residual:
`docs/plan.md` § Ceres arm-pump.

### Elev re-pin findings (`rr-14u`, 2026-08-07)

| Fact | Detail |
|------|--------|
| Ledge pin | TAS vs legacy: **identical** WRAM at left seat (`x45 y571 pose 138 x_sub=0`) |
| Desync cause | **Absolute-frame debris phase** in shaft (not subpixel at pin) |
| Product shaft | Open-loop s2–s10 works when phase matches; thrash hops burn timer |
| Phase search | Idle **0** = legacy green; idle **14** = TAS elev clear (probe scan) |
| Falling | Keep product **walk** into door (arm-pump mid desyncs elev entry) |
| Human (Kentroid) | Shaft ≈432f mostly `LEFT+B` + short A (spin); not used as restore |
| Sniq TAS | Short `LEFT+A` / `LEFT+B` pulses near Ceres end — reference only |
| Happy medium | Product wall-spin spans + phase idle list + top residual; no hop thrash |

### Verified facts (still true)

| Fact | Detail |
|------|--------|
| Arm-pump morph dual | GREEN **26,824f** ×2 (legacy boot + arm-pump escape) |
| TAS morph dual | GREEN **27,494f** ×2 (probe; elev + landing OK) |
| Falling→elev | Mid-trans y≈139 fake; **gs=8 → bottom y≈651** — do not LEFT-walk on ghost y |
| Elev top | x211 y171 pose 137 → LEFT+A → ship pad |
| Product boot | `_BOOT_STYLE = "legacy"` — do **not** flip to `tas` until TAS ≤ 26,824 |

### STATUS / next

- **Shipped:** tail-tank Ceres Ridley on the spine; elev platform-hop body
  (center 475 land; ship leave still open). BB elev parity
  retry + reactive board, morph seed pad-return reseed; falling timeout 700f.
- **Product stays legacy 26,824.** TAS path dual-green but **+670f** from BB elev
  seed misses + morph reseed — reclaim before `_BOOT_STYLE = "tas"` / STATUS promote.
- **Follow-on:** first-try BB elev under TAS boot (`$0E16` phase) so morph seed
  hits product tape without reseed.

### Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --no-video
# TAS probe: set early_spine._BOOT_STYLE = "tas" then same CLI
# TAS movies: uv run python -m super_metroid.tas.fetch_refs
```

### TAS 100% reference foundation (2026-08-07) — not STATUS

Power-on Sniq 100% (`sniq_100_full`, 222 789f) annotated under harness: first
control **11 183** @ Ceres elev; **106** room_enter / **18** desync / **5**
deaths; **never leaves Ceres** (items/beams stay `0x0000`). Same desync class
as any%. Artifacts (gitignored): `recordings/tas_import/sniq_100_full/` +
`extraction_board.json`. Tooling: `tas/stages.py` (RoomStageSpec),
`tas/extract_hops.py` (skills/graph board). Follow-ons: `rr-ni19` seed
materialize, `rr-d7mq` skill pure probes. **Product tip remains Wave; Ice pure
`rr-5cf` is still P0.** Playbook: [`TAS_ADAPT.md`](TAS_ADAPT.md).

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
   `0xADDE`; `DEFAULT_CONTINUOUS_TIP = wave` (Speed prefix still 130,388f).

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

- [x] Natural Alpha PB `0xA3AE` collect after Ice (not Pink PB)
- [x] Scratch dual `--to moat` **175526f** ×2 (rr-2r06; default CLI still `ice`)
- [x] Scratch dual `--to ws` **176141f** ×2 (rr-p2bw; default CLI still `ice`)
- [ ] Planner STATUS promote `--to moat` / `--to alpha_pb` / `--to ws`

### K6 — Ship / Phantoon / Gravity

- [x] Moat shinespark pure from `post_kihunter_pre_moat_spark` → West Ocean
  (store→spin→UP unspin→spark + RIGHT+X door; probe hop + controller pure GREEN;
  West handoff `scratch/post_moat_west_ocean_spark.state`; harness B=dash A=jump;
  residual purged after pure green; **not** continuous / STATUS)
- [x] Landing Site shine practice gym + diagnose/drill
  — [tasks/SHINE_PRACTICE.md](tasks/SHINE_PRACTICE.md)
  (`scripts/probe/shine_practice.py` human/drill/demo; store trap documented)
- [x] West Ocean edge-turn-hop pure → mid-right door `0xC98E` (Bowling)
  — [tasks/SHINE_PRACTICE.md](tasks/SHINE_PRACTICE.md) / `west_ocean_spark.py pure`
  (practice only; free-place spit bootstrap)
- [x] West Ocean over-ocean spark → green Super WS `0xCA08` pure
  — `west_ocean_spark.py pure-ws` / `play_west_ocean_over_ocean_spark`
  (natural Moat handoff ~(49,1163); stutter dual-green from the power-on
  `--to moat` leave **627f** ×2 probe / **615f** ×2 spine hop; pin
  `scratch/post_moat_poweron_wo_to_ws.state`; `--to ws` scratch dual
  **176141f** ×2, **not** STATUS)
- [x] Product WS pin + human record setup (`--from ws-entrance` /
  `practice_takes --segment ws-entrance`) for ship free-record
- [x] Human Gravity path + tail pin Caterpillar `0xA322` items `0x3125`
  (`scratch/post_gravity_caterpillar.state`; `--from post-gravity`)
- [ ] Natural climb onto West Ocean dry spit (only if reusing edge-bowling path)
- [x] Moat → West Ocean → Wrecked Ship pure compose (`play_moat_to_ws` /
  `west_ocean_spark.py chain-ws` / `record_pure_chain --preset moat-to-ws`;
  dual pin sources; **pin-only** — not power-on continuous STATUS)
- [x] Compose wired to Phantoon ship recording (`--from ws-entrance` after
  `chain-ws`; `phantoon_combat` ← `ws_ship_human_end` ← Gravity free-record)
- [ ] Natural Phantoon entry → fight → Gravity (continuous power-on compose)

- [x] Grapple side-trek + Maridia free-record from post-gravity pin
  (`tasks/maridia_grapple_human.json` 44039f → Main Street trace end;
  Grapple ~f24720 items `0x7125`; hops extract offline;
  Main Street **binary end pin LOST** — re-lock with anchors from
  `--from post-grapple`; see `docs/tasks/SM-MARIDIA-GRAPPLE-HUMAN.md`)
- [x] Anti-desync human recording: live room/item anchors + F6 + end fingerprint
  (`human_tape.py` / `guided_human` default ON / `extract_human_tape.py`)
- [x] Re-lock Main Street pin from post-grapple with anchors + F6
  (`tasks/maridia_main_street_human` **14170f**, end `0xCFC9` ~(391,1979)
  items `0x7125`, pin `scratch/post_grapple_main_street.state`, end_fp OK;
  `--from main-street`)
- [x] Fix room_enter anchor swallow during door_transition (`human_tape.py`)
- [ ] Continuous tips only after natural doorway entry

### K7 — Maridia

- [x] Human Main Street → Botwoon → Draygon → Space Jump free-record
  (`tasks/maridia_botwoon_path_human` **58670f**, SJ @ f52049 items `0x7325`,
  pins `post_space_jump` / `post_draygon_precious`; shape only — sloppy grapple;
  `docs/tasks/SM-MARIDIA-BOTWOON-HUMAN.md`)
- [ ] Tube / Everest / Botwoon / Draygon pure + continuous (natural entry each)
  (human start: `--from main-street`; post-SJ: `--from post-space-jump`;
  post-Plasma: `--from plasma-beam` / `scratch/full_start_v1_plasma.state`)

### K8 — Lower Norfair / Ridley

- [x] Human post-SJ → Spring + Plasma → LN Main Hall free-record
  (`tasks/post_sj_exit_human` **80368f**, end `0xB236` ~(1152,648) items
  `0x7327` beams `0x100F`; pin `post_ln_main_hall` / `--from main-hall`;
  `docs/tasks/SM-POST-SJ-EXIT-HUMAN.md`)
- [x] Human Main Hall → Screw → Ridley → Landing Site free-record
  (`tasks/post-main-hall` **121220f**, Screw f10857 items `0x732F`, Ridley
  Norfair bit 6→7, end `0x91F8` ~(1152,1088); pins `post_bosses_landing_site`
  / `post_screw_attack` / `post_ridley_tank`; `--from post-bosses`;
  `docs/tasks/SM-POST-MAIN-HALL-HUMAN.md`)
- [ ] LN pure geometry + Ridley combat natural-entry (shape from human tape)

### K9 — Tourian / MB / Escape / Credits

- [ ] Human G4 statues → Tourian → Mother Brain free-record
  (start: `--from post-bosses`)
- [ ] G4 statues → Tourian → Mother Brain pure + continuous (zebetites, phases)
- [ ] Escape timer + geometry + ship / ending-credits evidence (M8)

Boss order and phase rules: [BOSS_PIPELINE.md](BOSS_PIPELINE.md). Template:
Kraid → Varia continuous.

### PRACTICE (dual-track, planner opt-in)

- [ ] `ROOM_WORK_QUEUE` + `farm_room_waves.sh` when planner opts in (not default P0)
- [ ] Combat unit scaffolds (no full fights before natural entry)
- [ ] Early Spazer walljump detour + 100% board (parallel; does not block K4)
  — [routes/TRACK_100.md](routes/TRACK_100.md)

Practice greens ≠ continuous evidence and **not** product next-work
(`STATUS` + beads own that). Own-files only; width ≤ 8.

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
- [x] Continuous `*RunReport` aliases removed; Super+ tips via `play_tip` / `run_to` only

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

Live dispatch: `bd ready -l super_metroid`.
Milestone names: [routes/MILESTONES.md](routes/MILESTONES.md).

---

## Pointers

| Doc | Role |
|-----|------|
| [STATUS.md](STATUS.md) | Verified tip + prefix frames |
| `bd ready -l super_metroid` | Ready / in-flight work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers + structural debt |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss natural-entry rules |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean track process |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | KPDR route text |
