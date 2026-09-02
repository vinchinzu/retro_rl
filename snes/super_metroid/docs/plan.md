# Plan — Super Metroid

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

**Critical path:** Phantoon is the living tip. Next compose is Gravity from
`scratch/post_phantoon_leave.state`. Living residual:
[tasks/rr-kw8t-residual.md](tasks/rr-kw8t-residual.md).

Live work: `bd ready -l super_metroid -l spine`.
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).

Parallel room work, human-tape bootstrap, permissive scaffold assists, and
safe ten-agent speed waves are specified in
[PARALLEL_SPINE_PLAN.md](PARALLEL_SPINE_PLAN.md). Read it before dispatching
work that starts from archived room pins, changes a room already on the spine,
or attempts a tape-backed continuous chain. Its scaffold track is development
evidence, not the Survival living tip.

### Ceres TAS-speed elevator (parallel Chip)

Not Gravity. Not a second tip. Play from
`routes/kpdr/ceres/data/ceres_first_control.state` only. Leftover magnet /
falling seats are gone.

Rungs are entry → 475 → 363 → 267 → 171 → ship pad. Miss raises. Hard fail
over 2500f (`CERES_ELEV_MAX_FRAMES`). No 571 checkpoint recover. TAS
`elev_to_landing` is **2246f**. A 3349f station that remapped y=651 as
success was a gaslight.

Falling leave lands **(216, 633) pose 26 vd=1 vy=4 inv=6** against TAS
**(216, 632) pose 25 vy=+4 inv=36**. x=45 was never debris: it is the Ceres
door enemy `$E23F` (`CERES_DOOR_ID`), shut for ~16f after the ledge. Walking
it is pose 138 / movement type 21 — momentum 0 and a y=108 ceiling bonk, in
the air as much as on the ground. The door policy crouches it out east of
x=45, then runs LEFT and jumps at `_CERES_FALLING_DOOR_JUMP_X = 33` so the
leave is the 4th air frame. Door leave WRAM is frozen into the elev dest:
dest y is Falling y + 512.

`_ceres_fast_entry_window` gates `momentum_x >= 1`. That is the measured
floor, not a widened band: ground momentum caps at 2.75 (`$0B46`/`$0B48`)
however long the runway and halves once to 1.375 on the second airborne
frame, while the y band needs air frame 3. Eight takeoffs — spin, A-tap, aim,
shoot, UP, L-pump, and turnaround jumps carrying 4.375 — all halve the same
way, so no leave out of this door can carry 2 into the window. The other
clauses (y band, spin pose, `vd == 1`, `vy > 0`, `inv > 0`) are unchanged.

**The climb is one wall jump then three ledge hops, and it is not Sniq's
tape.** Sniq's `elev_wj` inputs (slice `sniq_100_ceres_open`, TAS frames
13074–13148) replay from this entry to the right *places* — the 475 seat, the
y=404 latch height — but never latch: they release A for one frame and
stable-retro needs two. Measured shape, `scratch/ceres_elev_wj`:

| rung | how | seat | cost |
|------|-----|------|------|
| entry → 475 | RIGHT+A 16f up the x=211 wall, LEFT 2f with A released, LEFT+A kick (pose 132), hold A 34f | (156, 475) | 75f |
| 475 → 363 | walk x=137, spin jump RIGHT | (189, 363) | 75f |
| 363 → 267 | walk x=191, spin jump LEFT | (107, 267) | 71f |
| 267 → 171 | walk x=144, spin jump LEFT | (66, 171) | 96f |
| 171 → pad | walk x=48, spin jump RIGHT, pad walk | (113, 75) gs 32 | 67f |

The entry arrives four air frames into its spin jump, so the entry rise alone
tops out at y=608 and the wall jump is what reaches 475. Above 475 a full
ground spin jump rises 111px against gaps of 112/96/96, so no further wall
jump is needed — what has to be right is the launch x. Each constant is the
middle of its measured band (475: 130–144, 363: 185–211, 267: 132–156, 171:
41–55); off the band the jump clips a ledge lip and drops back down the
shaft, which raises.

Entry → Ceres success (gs 32) is **384f**. Station is green end to end:
`success=True ridley=3255 landing=7876`.

**Open, and the next knob: it is not TAS speed.** Elevator gs 8 → Landing
gs 8 is **2575f** against the TAS 2246f (`tas_elev=False`; the report's
synthesized row prints 3071f because it counts our 496f Landing settle loop,
which the TAS clock does not). The +329 is structural: four ledge hops at
~70f of jump arc each against the TAS wall-jump chain's 31f plant-to-plant.
Closing it needs wall jumps *between* rungs. The shaft does have a chimney —
x=155, walls both sides, y 371–404, Samus pins there for 15f — but a
release-2 kick out of it (5 launch x × 2 spin dirs × 7 contact frames × 2
kick dirs) reaches y=323 and never lands a rung seat. That is where to start.

**Also open (planner-owned):** the `CERES_ELEV_MAX_FRAMES` guard at the end
of `play_ceres_escape_to_landing` is dead. It reads
`session.info["ceres_elev_start"]`, and `RouteSession.step` overwrites
`self.info` with the env's step info on every frame, so the lookup is always
`None` and the hard fail never fires. It is also mis-clocked: it compares a
span that includes the 496f settle loop against a cap set on the
settle-excluding TAS clock, so even a TAS-perfect climb could not pass it.
`_ceres_elev_budget` inside the climb uses a local start and does work. Left
untouched: making it fire flips `station` red and kills the JSON report
(`_write_json` is after the raise), which is a reporting decision, not this
knob.

```bash
PYTHONPATH=snes uv run python -m super_metroid.routes.kpdr.ceres.spine station
```

Do not STATUS. Do not change `DEFAULT_CONTINUOUS_TIP`.

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
