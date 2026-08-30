# Parallel spine plan — tape scaffold to reactive room Skills

## Outcome

Build one no-human-input, no-state-load emulator session from power-on through
the end of credits as quickly as practical by using the existing human tapes
as temporary room controllers and a separately disclosed, permissive assist
profile. Use that run to prove the splice machinery. Then replace tape rooms
with reactive **Skills**, integrate ten independently improved route ranges,
and optimize the resulting continuous run without letting a change such as an
Ice climb rewrite regress neighboring speed-walk rooms.

This plan adds a development track; it does not redefine **Finish** or replace
the one living **Tip**. The published primary remains Bronze / Survival under
[ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). The first tape-backed credits run is
called the **Scaffold chain** and must be labeled development-only until its
writes and runtime controllers satisfy the Survival contract.

## What already exists

Reuse these modules; do not build a second route engine.

| Capability | Existing source | Present limitation |
|---|---|---|
| One continuous runner | `routes/tips.py` `play_hops` / `run_tip` | Product order is still hand-wired through several registries |
| Product hop contract | `routes/kpdr/spine_types.py` `SpineHop` | Entry/leave provenance and candidate identity are not one typed record |
| Human tape materialization | `materialize.py`, `human_tape/` | Timing stitch is not replay; several tools expose overlapping schemas |
| Room bodies and pins | `tasks/full_start_v1_segments/sN/` | Gitignored/local, absolute paths occur, and availability is not preflighted |
| Work inventory | `tasks/PRODUCT_CHAIN_HOP_BOARD.json` | Generated/local; 282 hops, 282 anchors, but only 13 bank-dual-green and 2 policies at the latest inspection |
| Candidate ranking | `skill_bank.py` | Bank selection does not itself prove predecessor/successor **Join** |
| Reactive playback | `reactive_policy.py`, `autopilot.py` | Only one compiled policy currently exists; most rooms fall back to human control |
| Live rejoin | `room_adapter.search_live_adapter` | Not yet the standard compose path for every candidate kind |
| Exit grading | `leave_specs.py`, `hop_glance.py` | Specs are not bound into one immutable room task/candidate manifest |
| Source validation | `source_states.py` | Provenance, command, capabilities, and binary digests remain open work |
| Survival writes | `assist.py` | No isolated, audited development profile for enemy/boss HP reduction |

The useful local tape chain is materially better than older gap docs imply.
`full_start_v1_segments/s23` contains live anchors and bodies for Attic → West
Ocean → Pancakes and Wavers → Homing Geemer → Bowling Alley → Gravity. Segments
`s24` onward cover Gravity/Grapple, Maridia, Lower Norfair, and Ridley ranges;
the live `full_start_v1` and G4/Tourian tapes cover the ending. Before opening
new recording work, regenerate and inspect the product board. Treat
`FULL_STITCH_GAPS.md` claims about missing Gravity anchors as stale until the
preflight below proves otherwise.

## Design decision: one deep splice module

Create a game-local module, provisionally `super_metroid.splice`, with three
entry points and one CLI. It is a planning/verification module over the
existing `tips.play_hops` runner, not another runner.

```python
prepared = splice.prepare(task_id)                 # validate and materialize start
report = splice.grade(prepared, candidate_ref)     # replay, Join, successor probe
assembly = splice.assemble(route_id, selection)    # execute in one emulator session
```

The interface includes invariants and errors, not only type signatures:

- `prepare` fails closed before boot when the ROM/core/start-state digest,
  inventory, boss/event bits, room, pose/position/velocity band, predecessor,
  intended exit, or required artifact is missing or mismatched.
- `grade` runs from the exact immutable start, records every intervention,
  and returns evidence. It never edits the product manifest or promotes itself.
- `assemble` accepts only candidates whose intervention profile matches the
  assembly profile. It drives the existing `SpineHop`/`play_hops` path in one
  emulator session. It never loads a room state during the active assembly.
- A candidate is `replay_green` when it clears twice from its development
  anchor. It is `sync_green` only after its actual predecessor leave starts it
  and its exact leave passes the successor's `LeaveSpec` twice. Only
  `sync_green` is route-ready.
- A room candidate owns ordinary settled entry in its room through ordinary
  settled **Join** in the next room. This removes ambiguous ownership of door
  input and transition frames. Hard in-room phase pins accelerate work but do
  not change the external room contract.
- If A's new leave cannot start B, A+B become one change. The old A remains
  selected until the pair is green.

The seam has real adapters because behavior actually varies:

1. `TapeCandidate`: one settled room slice, with bounded live projection and
   `search_live_adapter` recovery; temporary Scaffold use only.
2. `ControllerCandidate`: a game-local reactive `play_*` Skill.
3. `ReactivePolicyCandidate`: compiled multi-trajectory room policy.
4. `BossCandidate`: thin movement adapter around `BossStrategy` after natural
   boss-room entry.

Hide tape slicing, state materialization, fingerprint/digest validation,
door-transition ownership, bounded adapter search, `hop_glance`, intervention
telemetry, failure capture, bank lookup, and report writing behind this seam.
Tests and callers should use the same three entry points. Keep internal
adapters private until a second external consumer exists.

## Canonical manifests

### Route manifest

Add one checked-in, declarative route manifest generated from the chosen
`TipSpec`/`SpineHop` chain and late tape inventory. Each ordered edge contains:

- stable `task_id` and `hop_key`;
- current room, predecessor room, intended next room or terminal goal;
- required items, beams, capacities, boss/event bits, and route variant;
- entry contract and successor `LeaveSpec` identity/digest;
- allowed candidate kinds and selected candidate id per intervention profile;
- owner package and integration order;
- maximum frames/no-progress budget;
- predecessor and successor task ids.

Generate task cards and assembly tables from this manifest. Do not hand-copy
the same hop order into a new table, `source_states.py`, the bank, and the CLI.
The existing product board remains an input/migration oracle until the route
manifest can regenerate it.

### Room task card

Every parallel worker receives one generated card containing all information
needed by a Grok-4.6-class agent:

- one checkbox: make `task_id` `sync_green` or leave an exact residual;
- immutable entry state content digest and repo-relative path;
- source tape digest, segment/hop/frame bounds, and source intervention data;
- full entry fingerprint including room, x/y, pose, velocity/subpixels,
  momentum, door kinematics, inventory, beams, capacities, boss/event bits,
  enemy phase summary, and prior room;
- exact exit/Join predicate and the next task's entry contract;
- candidate adapter kind, allowed intervention profile, timeout, and commands;
- owned source paths plus a unique candidate artifact directory;
- forbidden hot files and explicit non-claims;
- completion report fields and the next boot after RED.

Cards are immutable for a dispatch wave. A planner issues a new card revision
when a predecessor leave changes.

### Candidate artifact

Write candidates append-only under a game-local ignored artifact store, keyed
by content digest. Check in only compact manifests/reports when appropriate.
Each candidate binds:

- source/task/card/ROM/core/start-state/controller/tape digests;
- candidate kind and implementation identity;
- entry and final fingerprints;
- two replay rows and two predecessor→candidate→successor Join rows;
- frame count, no-progress maximum, action reasons, and failure class;
- every memory write as frame/address/entity/old/new/reason;
- leftover state path and screenshot/trace on RED;
- parent candidate id, so speed comparisons remain reversible.

Never mutate a shared `bank.json` from worker branches. Workers emit candidate
manifests. A single planner-owned rollup selects and writes bank/route indexes.

## Intervention tracks

Keep three explicit profiles and reject mixed evidence:

| Profile | Runtime use | Allowed writes | Claim |
|---|---|---|---|
| `clean` | Later privilege removal | none | Existing Clean rules |
| `survival` | Primary product | current energy + naturally unlocked ammo only | Eligible for living Tip after normal gates |
| `scaffold` | Fast first bot credits chain | Survival plus an allowlisted live-enemy HP clamp | Development-only, never STATUS/Finish |

Before implementing `scaffold`, extend `ASSIST_CONTRACT.md` and the report
schema. Conservatively label it Bronze / Progression-assisted development,
even though it must not write item, door, boss, event, room, timer, position,
or capacity state.

The HP clamp is not a generic `enemy0_hp = 0` switch. It must:

- be allowlisted by room, species/boss, spawn state, and phase;
- change a live target from positive HP to `1`, once per eligible phase, so a
  real controller hit triggers the game's death/phase/event logic;
- handle multi-slot enemies and multi-phase bosses explicitly;
- fail closed for unknown layouts and suspend during scripted transitions;
- log every write and expose counts by room/entity;
- be removable per task without changing the candidate interface.

Use the clamp to unblock traversal and validate splicing, not to skip doors,
items, boss flags, escape, or credits. A room whose phase logic breaks under
the clamp gets a normal reactive boss Skill immediately.

## Parallel ownership and integration

Workers may edit only their declared controller/Skill owner and their unique
candidate directory. They do not edit `routes/tips.py`, `spine_hops.py`,
`tip_segments.py`, `catalog.py`, `progression/`, `assist.py`, shared manifests,
STATUS, or another worker's residual. Shared primitives are planner-serial;
workers request them in their report. Promotion is also planner-serial.

Use one worktree/branch per task card. The coordinator records a lease with
task id, card revision, branch, owner paths, and expiry. Two active leases may
not overlap source paths. Artifact directories never overlap. Ten agents may
work independently from immutable room starts; only one coordinator may
integrate in route order.

An isolated speed improvement follows this transaction:

```text
selected candidate A0
  → worker grades A1 from the same entry digest
  → A1 beats A0 and replay-greens
  → coordinator runs predecessor → A1 → successor twice
  → select A1 only if Sync and regression budgets pass
  → otherwise retain A0 and couple the seam card
```

This is how an Ice-climb agent improves Ice without modifying the speed-walk
run. The branch submits `A1`; it never rewrites the shared spine or re-pins
unrelated rooms. Selection is data, rollback is selecting `A0`, and any new
leave invalidates only the immediate successor card.

## Delivery phases

### Phase 0 — freeze the evidence and remove false blockers

1. Snapshot hashes and availability for every `full_start_v1_segments/sN`
   tape, anchor index, state, body, join record, late G4/Tourian tape, ROM, and
   emulator core. Rewrite absolute paths as repo-relative references at load.
2. Regenerate the product-chain board and report missing/corrupt artifacts,
   duplicate hop keys, impossible inventory transitions, and stale docs.
3. Establish a recoverable backup/export for the gitignored tape/state corpus.
4. Mark old `gravity_path_human` as an oracle only; prefer anchored s23/s24
   material unless preflight rejects it.

**Done:** a fresh process can resolve every selected artifact by digest before
an emulator boots, and the exact first uncovered route edge is known.

### Phase 1 — minimal splice MVP

1. Land the route/task/candidate schemas and `prepare`, `grade`, `assemble`.
2. Adapt existing `human_tape`, `SkillBank`, `hop_glance`, `source_states`, and
   `tips.play_hops`; replace overlapping orchestration rather than layering a
   new runner beside it.
3. Make RED always save a leftover package; make GREEN always include the next
   room's Join grade.
4. Add planner-only selection/rollup and a read-only board command.

**Done:** one existing two-room slice can select tape and controller variants,
run both through the same interface, and roll back selection without code edits.

### Phase 2 — Attic → Gravity pilot while Main Shaft remains serial

Use `full_start_v1_segments/s23` room anchors to work ahead of the living
Main Shaft residual:

1. Attic `0xCA52` → West Ocean `0x93FE` (kill-all gray door; Scaffold HP clamp
   is allowed here).
2. West Ocean → Pancakes and Wavers `0x9461`.
3. Pancakes/Wavers → Homing Geemer `0x968F`.
4. Homing Geemer → Bowling `0xC98E`.
5. Split the 5015-frame Bowling hop internally, but retain one external
   natural-entry→Gravity contract.
6. Gravity entry, natural PLM collect, and settled post-collect leave.

These candidates may replay-green in parallel from archived anchors. They do
not become route-ready until the current Main Shaft controller reaches Attic
and the coordinator re-runs each real neighbor Join in order. Keep
`rr-kw8t`/Main Shaft implementation serial and preserve its dirty files.

**Done:** actual Phantoon leave → Main Shaft → s23 chain → natural Gravity
collect assembles twice in one session. Only the Survival version can advance
the living Tip; the Scaffold version proves tooling and remains scratch.

### Phase 3 — first Scaffold chain to credits

Assemble the fastest available candidate per room, preferring in order:

1. existing continuous `ControllerCandidate`;
2. verified reactive policy;
3. tape candidate with bounded live rejoin;
4. a focused reactive rewrite when tape cannot Join.

Use the anchored item-seam ranges as ten parallel preparation lanes, while
the coordinator stitches them serially:

1. Attic → Gravity (`s23`)
2. Gravity → Grapple (`s24`)
3. Grapple → Main Street (`s25`)
4. Main Street → Space Jump (`s26`)
5. Space Jump → Plasma (`s27`)
6. Plasma → Golden Torizo (`s29`, with `s28` treated as superseded if marked)
7. Golden Torizo → Screw Attack (`s30`)
8. Screw Attack → Metal Pirates (`s31`)
9. Metal Pirates → post-Ridley (`s32`)
10. Ridley → G4 → Tourian → Mother Brain → escape → ship/credits (live
    `full_start_v1` plus the anchored G4/Tourian tapes)

Each lane advances one room at a time and can return several independent
candidate manifests. Bosses may use the Scaffold clamp; route progression,
items, doors, escape, and credits still occur naturally in the running game.

**Done:** one power-on, autonomous, zero-state-load Scaffold session reaches
the end of credits with a complete intervention ledger and room split table.
It is a splice milestone, not Finish.

### Phase 4 — Gut after first credits

Delete or fold duplicate tape/compose orchestration so the three-entry splice
interface is the test surface. Generate the board, cards, CLI help, candidate
selection, and `SpineHop` projections from one route manifest. Keep source
files near the 1000-LOC soft ceiling by merging behavior into the owning deep
module or deleting replaced paths; do not create sibling wrappers.

Promote a repeated movement primitive only after two room Skills consume it.
Bosses remain behind `BossStrategy`; enemy behavior remains the cooperative
Overlay. Archive human tapes after their derived Skills, counterexamples, and
provenance are recoverable.

**Done:** deleting the splice module would force validation, candidate choice,
and Join logic back into many callers; callers themselves remain small.

### Phase 5 — ten-agent reactive conversion and Speed waves

Dispatch ten non-overlapping cards chosen by highest continuous dwell and
failure cost, not simply geographic order. Each card must convert a selected
tape/open-loop candidate into a condition-robust Skill or improve an existing
Skill. Require entry perturbations, takeover at 25/50/75%, boundary/recovery
fixtures, and successor Join. Keep at least one counterexample for every
observed failure class.

For Speed, A/B from identical entry digests and compare:

- frames and RTA delta;
- Join rate across natural/boundary/recovery starts;
- no-progress maximum and retries;
- assist writes/damage;
- immediate successor performance.

The coordinator integrates passing candidates in route order in batches of at
most three adjacent changes, then runs the narrowest prefix/suffix compose.
Run a full milestone dual at Gravity, new living tips, credits, and before
Publish—not after each isolated Chip.

**Done:** all selected rooms have reactive Skills, the ten-way wave stitches
into one continuous Survival chain, and the old candidate remains selectable
for every rejected speedup.

### Phase 6 — privilege reduction

Remove Scaffold HP clamps room by room. A room graduates to Survival only when
the same route manifest selects a candidate that greens with resource refill
alone. Later remove resource writes under the existing Clean track. Never
change evidence labels in place; produce a new assembly report per profile.

**Done:** power-on → credits is dual-green under Survival and eligible for the
normal STATUS/Finish process; Clean remains independent.

## Promotion gates

| Gate | Required evidence |
|---|---|
| Prepared | All artifacts and fingerprints resolve by digest |
| Replay | Candidate reaches its LeaveSpec twice from the same immutable room start |
| Reactive | Boundary/recovery/takeover cases meet the declared rate and timeout |
| Sync | Real predecessor → candidate → immediate successor passes twice |
| Range | Every room in an item-seam lane passes in one emulator session |
| Scaffold credits | Power-on → end credits, no state loads/human input, full write ledger |
| Survival credits | Same, with only contracted energy/unlocked-ammo writes |
| Speed promotion | Faster under identical start plus Sync/regression gates |

## Blocking bugs to resolve first

1. Local/gitignored tapes and state pins have no durable availability contract;
   absolute paths in indexes make worktree/host movement fragile.
2. `FULL_STITCH_GAPS.md` and deferred beads describe Gravity anchors as
   missing, while the newer `full_start_v1` product board reports complete
   anchors. The generated inventory must be authoritative.
3. `PRODUCT_CHAIN_HOP_BOARD.json` is a useful work list but not a checked-in
   route contract and currently reports almost no reactive policy coverage.
4. The skill bank, product board, source catalog, `LeaveSpec`, and spine each
   carry pieces of candidate identity; no single record proves all of them.
5. `human_tape.stitch` is timing-only. Existing compose verifies isolated
   pin/body replay by rebooting each hop's archived pin; despite its name, it
   is not natural-entry, no-load composition. There is no universal continuous
   candidate selector/executor.
6. Entry fingerprints omit or inconsistently preserve velocity/subpixels,
   door kinematics, enemy phase, source/core hashes, and assist-profile digest.
7. Worker output can collide in shared bank JSON, residuals, scratch report
   names, and hot spine registries; leases and planner-only rollup are absent.
8. The current assist controller cannot express or audit a scoped one-hit
   development profile, and boss phase layouts cannot safely share one generic
   HP write.
9. A replay-green room can still poison its successor; selection has no atomic
   predecessor/candidate/successor transaction today.
10. Old long tapes are not automatically safe runtime controllers. Any room
    that cannot bounded-rejoin and Join must be converted before the first
    autonomous Scaffold chain.
11. `RoomAutopilot` is wired to interactive recording, not `tips.play_hops`;
    the continuous runner cannot yet dispatch a reactive Skill. Its registry
    also selects by current/from room without a required intended-exit Goal,
    which is ambiguous in multi-exit rooms.
12. Current tape and reactive green checks are weaker than canonical
    `LeaveSpec`/`hop_glance` Join: destination-room arrival can pass before
    ordinary settle, and a final in-room item hop may pass on loose xy alone.
13. `HopSkillRecord.dual_green` is a mutable boolean rather than a receipt
    bound to pin/body/profile/ROM/core digests; changing a body does not
    inherently invalidate its green.
14. The generated product board contains invalid/unsettled pseudo-room rows
    such as `0x5555`/`0x0000`, uses inconsistent `start`/`leave` seam keys, and
    ranks a globally short row rather than dependency order from the Tip.
15. Reactive variants do not yet bind intended exit, beams/equipped state,
    capacities, boss/event lineage, or enemy phase strongly enough for
    deterministic route selection and recovery.

## Stop conditions and non-claims

- No worker promotes STATUS, changes `DEFAULT_CONTINUOUS_TIP`, or overwrites a
  selected baseline.
- No active assembly uses save-state loads, human takeover, door warp, item or
  boss/event writes, or frame concatenation between tapes.
- A pin replay, theoretical PB, stitched timing report, Scaffold clear, or
  practice-room green is not a Survival continuous claim.
- Main Shaft stays serial until Attic Join is green; later rooms may work ahead
  from immutable anchors.
- If three attempts share the same miss class, issue a new trajectory/phase
  card or convert to a reactive Skill; do not farm the same timing knob.
- Run no ten-agent wave until Phase 1 card validation prevents overlapping
  ownership and missing artifact dispatches.

## First executable backlog

1. Artifact/digest preflight and stale-gap report.
2. Route/task/candidate schemas plus a read-only card generator.
3. `prepare` around existing source/anchor validation.
4. `grade` around replay + `hop_glance` + successor smoke.
5. Planner-only candidate selection and `assemble` through `tips.play_hops`.
6. Attic and Bowling tape adapters from s23; keep Main Shaft serial.
7. Scaffold assist contract + one allowlisted ordinary-enemy pilot.
8. Phantoon-leave→Gravity Scaffold range assemble.
9. Generate ten item-seam lane inventories and ownership leases.
10. Run the first autonomous Scaffold credits chain, then open Gut/reactive
    conversion and Speed waves.
