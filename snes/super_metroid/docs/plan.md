# Plan — Super Metroid assisted full clear

Verified facts: [STATUS.md](STATUS.md). Shared workflow:
[`docs/FULL_RUN_PROCESS.md`](../../../docs/FULL_RUN_PROCESS.md).
Assist: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Layers: [ARCHITECTURE.md](ARCHITECTURE.md).
Executor: [tasks/PROCESS.md](tasks/PROCESS.md) · [tasks/QUEUE.md](tasks/QUEUE.md).

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

| Priority | Work |
|----------|------|
| **★ P0 pure** | **Bat → Speed Hall** from `post_bubble_to_bat_pure` / `post_bat_cave_continuous` |
| P0 pure stack | Speed Hall → Speed → Wave → Ice (pure green each before compose) |
| P0 continuous | After pure green: graph → compose tips (`speed` / `wave` / `ice`) → dual re-verify → STATUS |
| P1 | K5 **Alpha PB** (natural post-Ice; first competitive PB on KPDR) |
| Parallel Clean | Morph green; ★ **bombs / Torizo Clean** — [CLEAN_TRACK.md](CLEAN_TRACK.md) |
| Parallel (opt-in) | Room farm (planner only) · 1–2 ARCH · boss primitives (own-files only) |

**Parked:** Frog Save → Speedway → Farm → Bubble (needs Speed / Boost Blocks).
**Do not:** treat CATH pure as open (CATH-01…04 + Bubble→Bat pure are green);
default continuous tip is already `bat_cave` (not Frog).

Live residual board: [tasks/QUEUE.md](tasks/QUEUE.md).
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).

---

## Open work by epic

### K4 remaining (Norfair items)

- [ ] Pure **Bat → Speed Hall** (★ next)
- [ ] Pure Speed Hall → Speed Booster room + collect
- [ ] Pure Speed → Wave → Ice chain
- [ ] Graph edges to `continuous` after each pure green
- [ ] Continuous compose + dual re-verify for `speed` / `wave` / `ice` tips
- [ ] Short stabilize wave after each continuous promotion
- [ ] (Parked) Speedway → Farm → Bubble post-Speed shortcut

### K5 — Alpha PB

- [ ] Natural Alpha PB `0xA3AE` collect after Ice (not Pink PB)
- [ ] Continuous tip + integrity dual-run
- [ ] Graph verification promotion

### K6 — Ship / Phantoon / Gravity

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

- [x] Morph Clean continuous green (27,074f)
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
