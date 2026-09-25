# Plan: Super Metroid

Verified facts: [STATUS.md](STATUS.md). Language:
[`CONTEXT.md`](../CONTEXT.md). Assist:
[ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Tracker:
`bd ready -l super_metroid -l spine`.

**Doc home:** STATUS is verified facts; this file is future work; beads
are ready/in-flight. Session loop lives in `.grok/skills/sm-session/`
(not a QUEUE, not PROCESS.md). One living residual per open tip. Do not
rewrite the route or claim a new tip from a pin bench.

**Program role:** Beat vanilla Super Metroid with a **skill API**, equal token
weight with Harvest. Solver/SMZ3 is downstream. See
[`CONTEXT.md`](../CONTEXT.md) and
[`docs/SOLVER_ARCHITECTURE.md`](../../../docs/SOLVER_ARCHITECTURE.md).

## Strategy

**Survival** (energy + unlocked ammo) makes combat attrition secondary. The
hard problem is long-horizon navigation. The campaign is a skillset. TAS and
human **tapes** are guidelines; trash them once the skill exists.

One **living tip**. Pin is for building; rung green is power-on to that tip.
Scratch, Clean, and practice are parallel, not a second product.

**Chip** is one room: tape → skill. **Sync** is a clean **Join** into the next
room (doorway pause / a few frames allowed). If the leftover will not Join:
one multi-room Skill, a smaller checkbox, or write the miss and end the
Sitting. Drop the split, not Gravity. Full power-on dual at milestones
(Gravity, a new living tip, credits) and before **Publish**, not every slop
hop.

**Clear rooms by play.** Door-warps are topology diagnostics only. Recipe
([ARCHITECTURE.md](ARCHITECTURE.md)):

```text
tape/TAS guideline → hop dual-green → SpineHop → power-on compose → Sync next
```

**Boss fights stay deferred** until natural *entry* to that boss room exists on
the played chain. Pipeline: [BOSS_PIPELINE.md](BOSS_PIPELINE.md).

**Agent discipline:** `.grok/skills/sm-session/` (one bead, one knob, never
halt, no STATUS from a pin). Do not relax for scale.

**Ticket size:** one hop, or drop the split (merge rooms / shrink checkbox /
write the miss). Prefer 30–90 min sessions. STATUS/docs updates are
planner-owned.

---

## Current focus

| Priority | Work | Beads |
|----------|------|-------|
| **★ Product next** | Gravity on the Phantoon tip (power-on) | `rr-kw8t` |
| Living tip | `--to phantoon` **195,336f** ×2 | STATUS `rr-b926` |
| Parallel | Chip prefix slop under Sync | not a second tip |
| Parked | TAS/oracle, 100%, Clean spore+ | not `spine` |

**Rungs:** Phantoon (living tip) → Gravity → Maridia → LN+Ridley → Tourian+credits
→ rewrite toward sub-hour.

**Critical path:** Phantoon is the living tip. Next compose is Gravity.
The residual owns the pin. Do not name that pin in STATUS.
[tasks/rr-kw8t-residual.md](tasks/rr-kw8t-residual.md).

Live work: `bd ready -l super_metroid -l spine`.
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).

Parallel room work, human-tape bootstrap, permissive scaffold assists, and
safe ten-agent speed waves are specified in
[PARALLEL_SPINE_PLAN.md](PARALLEL_SPINE_PLAN.md). Read it before dispatching
work that starts from archived room pins, changes a room already on the spine,
or attempts a tape-backed continuous chain. Its scaffold track is development
evidence, not the Survival living tip.

### Ceres elevator (parallel Chip)

Not Gravity and not a second tip. Closed climb notes live in
[tasks/rr-gzsd-residual.md](tasks/rr-gzsd-residual.md). Still open: wall
jumps between shaft rungs (slower than the TAS elevator), and the
`CERES_ELEV_MAX_FRAMES` check at the end of `play_ceres_escape_to_landing`
never fires because `RouteSession.step` replaces `session.info` each frame.
`_ceres_elev_budget` inside the climb still uses a local start. Do not
STATUS. Do not change `DEFAULT_CONTINUOUS_TIP`.

---

## Next rungs

### Gravity (NOW)

Natural Phantoon leave / WS power-on → Gravity Suit (`rr-kw8t`). Residual
owns pin, checkbox, and probe CLI. Power-on green is the rung.

### After Gravity

- Maridia: Tube / Everest / Botwoon / Draygon / Space Jump, each from natural
  entry. Human shape: [tasks/SM-MARIDIA-BOTWOON-HUMAN.md](tasks/SM-MARIDIA-BOTWOON-HUMAN.md).
- LN + Ridley from natural entry. Human shape:
  [tasks/SM-POST-SJ-EXIT-HUMAN.md](tasks/SM-POST-SJ-EXIT-HUMAN.md),
  [tasks/SM-POST-MAIN-HALL-HUMAN.md](tasks/SM-POST-MAIN-HALL-HUMAN.md).
- Tourian / Mother Brain / escape / credits (M8). Human G4→Tourian tape is
  still open (`--from post-bosses`).

Tapes are guidelines. Continuous tips only after natural doorway entry.

### Parked (not spine)

- Prefix Chip/slop under Sync. Morph is **24,187f**. Ceres TAS-speed
  elevator is the parallel Chip above; do not freeze Gravity for it.
- Planner STATUS for prefix CI `--to moat` (`rr-g3nj`) and Ice dual
  (`rr-ucl9`). Not a second living tip.
- TAS/oracle, 100% board ([routes/TRACK_100.md](routes/TRACK_100.md)).
- Practice `ROOM_WORK_QUEUE` when the planner opts in.

### Clean (parallel)

Morph Clean **26,824f** and Bombs/Torizo Clean **49,321f** ×2 are green.
Next is Spore Clean, parked. Never mutate default CLI assists.
[CLEAN_TRACK.md](CLEAN_TRACK.md).

---

## Maturity targets

| Gate | Target | Notes |
|------|--------|-------|
| **M5** | Bronze observation; Survival continuous tip | **Current** (Phantoon). Sticker, not a rung. |
| **M6** | Complete route graph with owners/predicates | In progress |
| **M7** | Continuous dry-run invariants (power-on → credits path) | Open |
| **M8** | Verified capture + ending/credits evidence | Open |

Observation-class migration (Bronze → Silver) is a separate workstream after
continuous reliability.

---

## Structure (open)

Current layers (CLI → continuous + catalog + segment → pure kpdr → graph →
ram/assist → combat) are correct. Planner-serial when touching
`continuous.py` / `progression.py` / `catalog.py`. See
[ARCHITECTURE.md](ARCHITECTURE.md).

- Profile frame time on long tips (WRAM-copy rate); optional linter against
  bare full `parse_env_state` inside `routes/kpdr/`.
- Prefer `wait_ordinary_room` handoff bands over airborne settle hope.
- Typed path-summary model; work-queue export from graph verification.
- Keep `legacy/` and `dev/` (door-warps) strictly fenced.
- Promote shared adventure patterns to `retro_harness.adventure` only after
  SM + ALTTP both prove the abstraction.

**Do not** relax pure-first / one-knob / residual rules.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Long-horizon nav fragility / high-dwell segments | Tighten offline secondary; stabilize after each tip |
| Architecture debt (multi-registry tip wire, full WRAM hot paths) | Highest-leverage ARCH first |
| Residual / card proliferation | One living residual; delete closed-hop cards |
| Process drift (practice claiming continuous) | Dual-track gates; planner owns STATUS |
| Endgame (Zebetites regen, escape geometry, timer/WRAM) | Deferred until natural entry |

## Non-goals (now)

- Door-warp / hybrid tours as continuous evidence
- Ship-first / PRKD continuous route
- Pink PB maze (first PB is Alpha after Ice)
- Vision-based boss combat in `legacy/`
- Claiming Clean greens as the program M5/M8 assisted gate
- Full endgame fight code before natural entry

---

## Recommended next waves

```text
★ NOW   Gravity on the Phantoon tip (rr-kw8t). Power-on is the rung.
THEN    Maridia (tube / Botwoon / Draygon / SJ) on the tip
        LN + Ridley → Tourian + MB + escape + credits
Parallel Chip prefix slop under Sync (drop the split, not Gravity)
Publish every 20–30 working sessions to the living tip (not a calendar)
Later   rewrite toward sub-hour; drop convenience majors; 100%
```

Live dispatch: `bd ready -l super_metroid`.
Milestone names: [routes/MILESTONES.md](routes/MILESTONES.md).

---

## Pointers

| Doc | Role |
|-----|------|
| [`CONTEXT.md`](../CONTEXT.md) | Language |
| [STATUS.md](STATUS.md) | Verified tip + prefix frames |
| `bd ready -l super_metroid` | Ready / in-flight work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers + structural debt |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss natural-entry rules |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean track process |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | KPDR route text |
