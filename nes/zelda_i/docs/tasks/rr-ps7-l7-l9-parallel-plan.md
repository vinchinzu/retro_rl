# Parallel build plan — Zelda I Levels 7–9

**Status:** implementation plan; no route claim.  The continuous Survival tip
is still the recovered Level 6 Compass boundary in
[`rr-tne2-residual.md`](rr-tne2-residual.md).  This plan does not supersede
that residual or promote Clean `STATUS.md`.

## Outcome

Finish the Survival-assisted full-game route without assigning a fresh agent to
every room.  Natural-entry proof remains sequential, but route discovery,
topology decoding, controller construction, and fixture-live verification run
in parallel by dungeon.

The implementation should need only ten public acceptance checkpoints after
Level 6:

| Dungeon | Public cumulative `--through` targets |
|---------|---------------------------------------|
| Level 7 | `level7-entry`, `level7-red-candle`, `level7` |
| Level 8 | `level8-entry`, `level8-magic-key`, `level8` |
| Level 9 | `level9-entry`, `level9-silver-arrows`, `level9-patra`, `level9-credits` |

Rooms remain named internal stages in the report.  A red therefore identifies
the exact room/policy that failed without making that room another public CLI
target, bead, checkpoint, or agent session.

## Facts at the planning boundary

- `rr-tne2` is the only route tip.  Fresh power-on `--through level6-compass`
  is 1/1 at play `0x68` `(120,205)`, TF `0x1F`, keys 4, bombs 8, Bow 1,
  full `0x66` health.  Finish L6 using the existing plan and residual.
- Level 7 has a partial start-based pond walk through OW `0x53`, but no pond,
  entry, dungeon, Red Candle, or shard checkpoint.  Its interior room IDs are
  source hypotheses.
- Level 8 has a start-based assisted path to bush screen `0x6D` and a mapped
  Blue Candle shop.  The bush has not opened and no dungeon room is live.
- Level 9 has live fixture-only suffix evidence from room `0x41` through Patra
  `0x52`, Ganon `0x42`, Zelda `0x32`, and credits.  Its natural entry-to-suffix
  route is unbuilt.  Every existing `*ReconFixture` remains
  `route_eligible=false`.
- `spine/survival.py` ends at L6.  L7/L8 have no dungeon/spine modules.  L9's
  large `stairs*` modules are fixture-recon implementations, not a natural
  spine.
- `bd ready -l zelda_i -l spine` currently has no unblocked work.  The bead
  graph must distinguish parallel recon/build children from sequential
  promotion children instead of removing dungeon dependencies from the epics.

## Core execution model

### Evidence ladder

Every stage carries one of these labels; never collapse them into a boolean
"green":

1. **Hypothesis** — walkthrough or ROM/topology claim; no live proof.
2. **Fixture-live** — controller works from a composed or loaded recon fixture.
   Useful implementation evidence, always `route_eligible=false`.
3. **Natural-segment** — begins at the real predecessor checkpoint with its
   earned inventory, but may have loaded that checkpoint at process start.
4. **Spine-green** — fresh power-on cumulative `run_survival_spine`, measured
   post-reset state loads 0, exact endpoint and inventory contract satisfied.

Parallel agents may advance work through fixture-live.  Only the integrator may
promote it to natural-segment or spine-green.

### Deep module seam

Each new level exposes the same small interface already used by Level 6:

```text
Lx_THROUGH
Lx_STOPS
continue_levelx_spine(env, run, through, run_stages, ...)
```

The implementation hides room controllers behind chapter factories that
return fresh controllers and named stages.  `spine/survival.py` learns only the
interface above.  Tests and the continuous composer use the same seam.

Use the established roles:

- `levelX/dungeon.py`: room specs and stop predicates only;
- `levelX/path.py` or a purpose-named path module: one-frame navigation policy;
- `levelX/hops.py`: internal `SpineHop` chapter rows and controller factories;
- `levelX/spine.py`: the public interface and cumulative attachment;
- `levelX/overworld.py`: stable OW facts/hops, not dungeon loops.

Do not mint room-specific public `--through` names.  Do not add another shared
abstraction until both L7 and L8 demonstrate the same missing behavior;
`SpineHop`, `attach_hops`, `OccupancyWalker`, and `overworld.path` are already
the shared modules.

## Persistent agent lanes and ownership

Use one long-lived agent per dungeon, plus the integrator.  Agents keep their
lane across reds instead of handing each room to a new session.

| Lane | Exclusive writes | Must not write |
|------|------------------|----------------|
| Integrator / current tip | `spine/survival.py`, `scripts/run_survival_spine.py`, shared RAM/assist/audit code, route catalogs, bead export, current L6 files/residual | L7–L9 controller implementations while their owners are active |
| L7 | `level7/**`, `docs/LEVEL7_ROUTE.md`, L7 tests and `l7_*` recon artifacts | shared spine, shared RAM, L8/L9 |
| L8 | `level8/**`, `docs/LEVEL8_ROUTE.md`, L8 tests and `l8_*` recon artifacts | shared spine, shared RAM, L7/L9 |
| L9 | `level9/**`, `door_graph/level9_exits.py`, `docs/LEVEL9_ROUTE.md`, L9 tests and `l9_*` recon artifacts | shared spine, shared RAM, L7/L8 |

`level8/overworld.py` is already 638 lines.  Freeze it for mainline work and
put the post-L7 Red-Candle entry chapter in a cohesive `level8/entry.py` module;
do not add more knobs to the large file.  The old Blue Candle shop controller
remains fallback-only.

Likewise, do not add natural-route behavior to L9's 560–720 line fixture
modules.  Keep those as recon adapters and build the natural prefix through new
role modules (`dungeon.py`, `natural_path.py`, `hops.py`, `spine.py`).

Parallel agents do not commit, export beads, or edit `.beads` concurrently in
the shared worktree.  The integrator serializes file review, bead updates,
export, and commits by lane.  Existing unrelated dirty-worktree changes are
out of scope.

## Work decomposition

### Lane 0 — finish and stabilize Level 6

Continue `rr-tne2` exactly from the current residual.  Before attaching L7:

1. Recompose the L6 body from `level6-compass` through the existing cumulative
   Gohma chain.
2. Add the natural heart, north `0x0C`, and TF `0x20` endpoint.
3. Measure post-reset state loads with `AuditedEnv`; remove/fail closed on the
   L4 Gleeok restore fallback in continuous mode.
4. Record the settled post-fanfare OW screen and full handoff inventory rather
   than assuming OW `0x22`.

Required L7 handoff packet: room/screen, mode, x/y, TF `0x3F`, Whistle, Rod,
Bow/arrows, Food, Candle, rupees, keys, bombs, heart containers, selected item,
deaths, every assist write, and measured state-load count.

### Lane 7 — Demon, three chapters

#### L7-A: topology + natural entry preparation

- Decode the first-quest L7 room/door/stair graph offline before emulator
  probing.  Mark every unobserved room as hypothesis.
- Replace the start-`0x77` pond route with a controller built from the measured
  post-L6 OW leftover.
- Build one deterministic 60R plan and natural Bait purchase.  Bait is required
  for the Hungry Goriya; no Food or rupee write is allowed.
- Reach the pond, select and use the naturally earned Whistle, drain it, and
  enter the live L7 entry room.

Public gate `level7-entry`: play Level 7 in the observed entry room, TF `0x3F`,
Whistle owned, Food owned, and predecessor inventory preserved.

#### L7-B: entry through Red Candle

- Implement a multi-room chapter from the entry through bomb/key routing,
  Digdogger policies, the Hungry Goriya, and the tip-of-nose stairs.
- Prefer bomb walls over the source's fifth lock; report every natural key
  gain/spend.  Survival bomb/key **count** top-up is allowed only at a verified
  gate and must retain existing telemetry.
- Prove Food is consumed by the natural gate and Red Candle changes
  `ADDR_CANDLE` to 2 naturally.

Public gate `level7-red-candle`: exact live room/mode, Candle 2, Whistle still
owned, TF still `0x3F`, and no item/progression/capacity write.

#### L7-C: Red Candle through Aquamentus and shard

- Compose the forced Digdogger and Aquamentus suffix from L7-B.
- Collect the natural heart container, then shard `0x40`.
- Preserve the exact settled post-fanfare OW leave for L8.

Public gate `level7`: TF `0x3F→0x7F`, exactly one natural heart-container
increase, full hearts, Candle 2, deaths 0, and no mid-run state load.

### Lane 8 — Lion, three chapters

The canonical path inherits the natural Red Candle from L7.  Do **not** put the
old 60R Blue Candle farm on the mainline.  It is only a fail-closed fallback if
the incoming route contract unexpectedly has Candle 0; such a mismatch should
normally fail L7 instead.

#### L8-A: post-L7 OW, bush burn, live entry

- Build from the measured post-L7 OW leftover, not the start-based hop table.
- Reach verified bush screen `0x6D` through the live `0x5C` maze geometry.
- Select the already-owned Candle through normal pause/input behavior; do not
  write `selected_item`.
- Solve the exact bush tile/fire facing.  Require Candle owned before B,
  observe `ADDR_CANDLE_USED`, mode-16 mouth, then live Level 8 play.
- Fix the current false positive: exhausting the burn budget while still on
  `0x6D` is failure, never success.

Public gate `level8-entry`: observed L8 entry room, TF `0x7F`, Candle 2, and a
natural burn/transition.

#### L8-B: entry through Magical Key

- Decode and then live-confirm the room/door graph.  Room IDs inferred from a
  walkthrough grid remain hypotheses until RAM observes them.
- Take the route to the Magical Key; it is the deliberate investment that
  removes Level 9's ordinary key bottleneck.
- Skip the Book of Magic on the minimum full-clear route unless live topology
  proves its detour cheaper than bypassing it.
- Reuse/parameterize Gohma policy without carrying L6's one-time wooden-arrow
  poke or Level 6 room checks.  Bow and wooden arrows must arrive naturally
  from the cumulative route.

Public gate `level8-magic-key`: `ADDR_MAGIC_KEY` changes naturally, TF remains
`0x7F`, and the exact incoming/outgoing key and bomb counts are recorded.

#### L8-C: Magic Key through four-head Gleeok and shard

- Compose the return/passage/boss chapter from L8-B.
- Live-confirm the four-head Gleeok object type before parameterizing the
  shared fight policy; do not assume the absent type is `0x45`.
- Collect the natural heart and shard.  Skip optional Book/Map/Compass work
  that is not on the selected route.

Public gate `level8`: TF `0x7F→0xFF`, Magic Key owned, exactly one natural
heart-container increase, full hearts, deaths 0, and no mid-run state load.

### Lane 9 — Death Mountain, four chapters

#### L9-A: topology selection and natural entry

- Decode the complete first-quest L9 graph from ROM, including stair/cellar
  endpoints, before choosing a route.
- Select the shortest route from entry `0x76` to natural Silver Arrows and a
  proven join into the existing suffix, assuming full TF and Magic Key.
- Red Ring is optional under Survival health refill and is excluded from the
  minimum route unless the selected topology passes it at negligible cost.
- Build post-L8 OW → Spectacle Rock `0x05` → natural bomb entrance → Old Man
  full-TF gate from the measured L8 leftover.

Public gate `level9-entry`: play L9 room `0x76`, TF exactly `0xFF`, bombs
natural/declared, Magic Key owned, and no room/door/TF write.

#### L9-B: entry through Silver Arrows

- Implement the selected multi-room prefix using Magical Key routing.
- Acquire Silver Arrows naturally (`ADDR_ARROWS == 2`).  Do not inherit the
  fixture's full-loadout writes.
- Keep all new specs/policies out of the large fixture stair modules.

Public gate `level9-silver-arrows`: exact live room/mode, arrows 2, Bow owned,
TF `0xFF`, and no inventory/progression/capacity write.

#### L9-C: natural join to the proven suffix

- Work backward and forward from the selected topology until the cumulative
  prefix naturally enters final Patra room `0x52` with body `0x47`, eight eyes
  `0x25`, and the north door closed.
- Reuse the proven `0x41→0x31→0x30→0x67→0x04→0x03→0x52` fixture suffix only
  if the selected natural route actually joins it.
- `rr-yxy6` (`0x51` statue diamond) is conditional.  Do not spend sessions on
  it merely because it is the current backward recon leaf; solve it only if
  the decoded natural route requires `0x51→0x41`.
- Every reused recon stage loses `fixture_only` only after recomposition from
  the natural predecessor.  Never relabel old fixture evidence.

Public gate `level9-patra`: live uncleared Patra `0x52` from the natural
cumulative prefix, Silver Arrows 2, and zero post-reset state loads.

#### L9-D: Patra, Ganon, Zelda, credits

- Reuse the already live Patra/Ganon/Zelda policies through their existing
  interfaces without adding inventory or door writes.
- Final Patra must naturally open north; Ganon must receive registered sword
  hits and the natural Silver Arrow; `$0672` must become nonzero naturally.
- Collect the Power Triforce, enter Zelda `0x32`, clear the guard fires, and
  stop on updating mode `0x13`, submode 3 or 4.

Public gate `level9-credits`: credits/final-page predicate, deaths 0,
continuous power-on session, measured post-reset state loads 0, and no
progression/capacity write.  Encode the single watchable MP4 only after this
gate; all development trials use `--no-video`.

## Bead decomposition — chapters, not rooms

Keep the existing epic dependency chain for promotion.  Add only three child
issues under each of L7 and L8, and three under L9:

| Epic | Recon/build children that may run now | Promotion child |
|------|----------------------------------------|-----------------|
| `rr-8t4` L7 | topology + entry; two internal controller chapters | cumulative L6→L7 acceptance, depends on `rr-tne2`/L6 completion |
| `rr-6o7` L8 | Red-Candle burn + topology; Magic-Key/boss chapters | cumulative L7→L8 acceptance, depends on L7 promotion |
| `rr-sz8` L9 | natural topology/prefix; Silver-Arrows-to-suffix join | cumulative L8→credits acceptance, depends on L8 promotion |

Recon/build children are explicitly labeled `recon` and
`route_eligible=false`; they do not need the prior dungeon's promotion child.
Promotion children retain the sequential dependency.  Do not remove the epic
dependencies and do not spawn one child per room.  Keep `rr-sz8.1/.2/.3/.4`
as historical fixture work; none can close the natural L9 promotion.

## Parallel waves

Wave A chapter beads created on 2026-08-30:

| Lane | Recon/build chapters | Sequential promotion |
|------|--------------------------|----------------------|
| L7 | `rr-8t4.1`, `rr-8t4.2` | `rr-8t4.3` |
| L8 | `rr-6o7.1`, `rr-6o7.2` | `rr-6o7.3` |
| L9 | `rr-sz8.5`, `rr-sz8.6` | `rr-sz8.7` |

The first recon chapter in each lane is explicitly assigned in progress for
Wave A. Dotted child issues inherit their epic's blocker in `bd ready`, so the
lane assignment—not removal of the epic dependency—authorizes fixture-live
work while the promotion children remain sequential.

### Wave A — immediately, while L6 is active

- Integrator: finish `rr-tne2` and the measured continuous-session audit.
- L7 owner: topology, post-L6-relative entry controller interface, Bait plan,
  and fixture-live L7 chapters.
- L8 owner: solve Red-Candle bush entry, decode topology, and fixture-live
  Magic-Key/boss chapters.
- L9 owner: decode natural topology, select the route/suffix join, and make the
  existing ending suffix callable without fixture inventory setup.

All dungeon agents stop at fixture-live.  Their output is code plus an endpoint
contract, not a route claim.

### Wave B — promote one dungeon at a time

1. Attach L7's `spine.py`; run the three cumulative public targets in order.
   Stop on the first red and return that internal stage to the same L7 owner.
2. Feed the measured L7 leave/inventory to L8; attach and promote its three
   public targets the same way.
3. Feed the measured L8 leave/inventory to L9; attach and promote its four
   public targets.

Do not create a new agent for a red room.  The persistent dungeon owner fixes
the reported internal stage and the integrator reruns the same public target.
Three serial reds on that stage block the chapter and trigger topology/policy
retargeting.

### Wave C — final audit

- Run one fresh power-on `level9-credits` trial with `--no-video`.
- Require every chapter report and assist audit to compose into the top-level
  report.
- Only after that succeeds, make the one watchable capture.
- Leave Clean M5/STATUS unchanged; Clean combat and resource hardening is a
  later damage-heatmap pass.

## Chapter handoff contract

Each owner returns one compact handoff per chapter:

```text
chapter id and evidence label
exact predecessor: level, room/screen, mode, x/y
required inventory/capabilities
ordered internal stage names and controller factories
exact endpoint predicate
expected inventory deltas: keys, bombs, items, TF, containers
known dead beliefs / first missed RAM claim
fixture provenance and route_eligible=false, when applicable
files changed and the one public target that consumes the chapter
```

No controller may default keys/bombs to zero, poke a door, grant an item, or
silently extend a timeout.  Occupancy misses block and replan; no path means
stand.  One-frame clips live in a path module with semantic reasons.

## Acceptance contract for every public target

- fresh power-on Survival run from the real cumulative predecessor;
- exact level, room/screen, mode, x/y band, and endpoint event;
- exact TF and earned-item deltas;
- keys/bombs and selected item recorded before and after every gate;
- full hearts with `lo==hi`, accepted container count, and deaths 0;
- measured post-reset state loads exactly 0;
- `progression_writes=0` and `capacity_writes=0`;
- only existing documented Survival exceptions, each present in telemetry;
- final PNG and `screen_glance` agree with RAM;
- `status_claim=false` until the separate Clean program gate.

Loaded/materialized fixtures may prove implementation only.  They never save a
route checkpoint, close a promotion bead, or inherit a natural/continuous
label.

## First build action

Create the nine chapter-level child issues above, assign the three recon/build
lanes, and keep promotion dependencies sequential.  The integrator continues
the already-documented `level6-clear68` target while L7–L9 agents build to
fixture-live behind their exclusive module seams.

This planning pass did not run tests or emulator trials.
