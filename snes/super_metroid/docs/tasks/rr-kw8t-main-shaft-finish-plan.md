# rr-kw8t Main Shaft finish and diagnosis plan

Static audit written 2026-08-29. This pass did not boot the ROM, run a probe,
or run tests. Findings below are therefore split into confirmed static defects
and runtime hypotheses. The first execution step is to build a red-capable,
bounded feedback loop; no controller change should be called a fix before that
loop reproduces and then clears the exact miss.

## Scope and invariant

Finish the powered Wrecked Ship Main Shaft hop from the existing natural
predecessor pin to ordinary Attic:

```text
post_ws_basement_to_main.state
  -> grate_seat                 GREEN x2
  -> west_super at y~1675       GREEN x2, natural entry
  -> planted 1543 stairs        RED, first diagnostic target
  -> mid_climb at y~680         RED
  -> attic_seat
  -> Attic room 0xCA52          full-hop GREEN only here
  -> prove the next Sync seam
```

Preserve the already-green lower route. Do not retune the pit, grate departure,
or west-super phase while diagnosing the 1675-to-1543 conversion. Start from
`scratch/post_ws_main_west_super.state` for iteration, then return to
`scratch/post_ws_basement_to_main.state` for natural-entry proof. An airborne
leftover is evidence, never a planted phase pin.

The relevant controller files already have uncommitted worktree edits. Review
and preserve those edits before implementing this plan; do not replace them
from `HEAD`.

## What is actually failing

The current route is not failing because the first jump lacks height. It starts
near `(1060, 1675)`, reaches `(1171, 1503) p84`, and therefore rises above the
1543 stair height. It fails to turn that airborne progress into a stable stair
landing. At timeout it is around `(1154, 1561) p76`, overlapping an Atomic at
`(1155, 1561)`, with all four Atomics still at 250 HP and freeze timer 0.

The dedicated probe records only minimum y and the final enemy snapshot. It
cannot currently answer when stair contact occurred, which controller owned
the landing frames, what target the ice overlay selected, whether a shot had a
valid line, or whether the run repeated one deterministic cycle for most of its
3,600-frame budget.

## Ranked findings

### P0 — controller geometry contradicts every successful human take

This is a confirmed static mismatch and the leading causal hypothesis.

`SHAFT_HOPS` tells the controller to launch right from the 1675 ledge with
`RIGHT+B+A`, then launch left from an x window of `1120..1180` at the 1543
stairs. Successful human takes 02–05 use a materially different route:

| Take | First launch | Stable traversal / next launch |
|------|--------------|--------------------------------|
| 02 | `(1062,1675)`: A, then RIGHT+A | run to about x1252 at y1547; B+A, then B+LEFT+A |
| 03 | `(1061,1675)`: RIGHT+A | launch left near x1259 at y1547 |
| 04 | `(1076,1632)`: RIGHT+A | launch left near x1259 at y1547 |
| 05 | `(1067,1675)`: A, then RIGHT+A | launch left near x1259 at y1547 |

The human route does not turn left around x1150. It crosses the stair shelf to
the far right before the next launch. The current guessed window turns at the
same place where the moving Atomic can intersect Samus. The human first launch
also omits B, while the generic shaft action holds B throughout.

Do not tune more jump height. Reconstruct the successful seat, traversal, and
takeoff from the tapes and make the controller reproduce those observable
states.

### P0 — the ice overlay can steal the frames needed to land

This is a high-confidence runtime hypothesis, not yet a reproduced cause.

In the shaft, any non-`None` result from `ice_keepaway_action` replaces
`climb_action` for that frame. That includes an empty tuple used as an ice-wait
decision. Target selection is nearest-enemy distance rather than phase intent
or line of fire. Near the recorded peak, the fixed Covern around `(1160,1468)`
can be nearer than the Atomic. The generic shot-seat predicate checks
horizontal distance but not vertical alignment, and the aiming rule fires
horizontally when a target is only about 35 pixels above Samus. At the final
Atomic overlap, it can also charge/release horizontally without an escape
action. The zero HP/freeze change in the saved red artifact is consistent with
a non-progressing overlay loop.

Successful take 02 uses no X input during this immediate shaft conversion.
Avoidance through the tape-derived far-right stair traversal is therefore the
first policy to try. Do not make “ice the Atomic at overlap” the default plan.
If combat is later necessary, invoke it only from a stable grounded seat,
select a phase-specific enemy slot, require an actual shot line, and define an
escape action for live overlap.

### P0 — PLM pixel coordinates are decoded at twice the true position

This is a confirmed measurement bug. Both `plm.snapshot_plms` and the private
PLM reader in `scripts/probe/ws_main.py` treat `$1C87` as a block-cell index.
It is a byte offset into two-byte level data. The game divides it by two before
dividing by room width; spawn code multiplies `(y * width + x)` by two before
storing it. See the source-level routines in the
[Super Metroid bank $84 disassembly](https://github.com/InsaneFirebat/sm_disassembly/blob/main/src/bank_84.asm#L135-L156).

The correct conversion is:

```text
cell = plm_block_index // 2
bx = cell % room_width_blocks
by = cell // room_width_blocks
px = bx * 16 + 8
py = by * 16 + 8
```

Current tests encode the wrong interpretation, so they must be corrected rather
than treated as evidence. This bug makes PLM proximity, screenshots, and
“near Samus” diagnostics misleading. It does not directly explain the present
1543 failure: the existing lip-hit latch keys on PLM ID appearance rather than
the derived coordinates, and successful take 02 does not fire during the
1675-to-1543 conversion.

### P1 — “PLM state” currently conflates three different evidence levels

The code exposes low-WRAM active PLM slots; `red_diag.py` and the gate recon say
PLM state is blocked; the collision helper says live bank-$7F clipdata is not
mapped. Resolve this wording rather than making a blanket claim:

1. Active PLM slot ID, instruction pointer, and byte-offset position in low
   WRAM can be source-validated.
2. Live level/BTS collision data in bank `$7F` is not currently available
   through the existing navigation snapshot.
3. A semantic claim such as “this breakable brick is open” still needs a
   position-specific PLM transition, dynamic collision proof, or visual /
   past-obstacle proof.

Correct the active-slot coordinates first. Only investigate breakable-brick or
dynamic collision state if the tape-matched landing still fails. There is no
current positive evidence that a breakable brick blocks the first stair seam.

### P1 — progress and acceptance are too coarse

`mid_climb` only accepts the much higher y~680 band. The failed seam at y1543
has no stable predicate, so minimum y is being used as a proxy for progress.
That violates the hard-room rule that a phase handoff must include usable
position, pose, and velocity.

Add a private diagnostic predicate for `stairs_1543`, derived from the human
tapes. Require Main Shaft room, an x/y stair band, a planted/grounded pose, and
near-zero vertical velocity. Keep the public six-phase route contract unless a
new public phase is needed by a second consumer. Never save or resume from the
current `(1154,1561) p76` airborne overlap as if it were green.

### P1 — high-WRAM fields in navigation reports are untrustworthy

The route correctly checks Phantoon using the bank-$7E helper, but generic
navigation snapshots can show garbage or zero for high WRAM event/boss fields,
and the final report derives `boss` from that snapshot. This can make a good
entry look bad or a bad entry look good in diagnostics.

For this probe, either omit high-WRAM boss/event claims or read `$7E:D82B`
through the trusted helper and label it separately. Do not use raw
`env.get_ram()` offsets above the core's reliable range.

### P2 — secondary consistency and lifecycle defects

- Covern metadata declares 300 max HP while all recorded shaft traces and test
  fixtures show 80. Reconcile it before using HP to infer target progress.
- `lip_hit` and previous PLM IDs are initialized inside each `climb_until`
  call, so phase boundaries can forget a lower-shaft block transition. This is
  not the present red seam because natural `west_super` is already green, but
  it can make direct and composed runs diverge. Preserve phase state in one
  route context or re-establish it from trustworthy observations after the
  Main Shaft blocker is cleared.
- The 3,600-frame budget lacks a repeated-state/cycle classifier, allowing a
  deterministic deadlock to consume the full budget without new evidence.

## Execution plan

### 1. Build the red-capable feedback loop before changing behavior

Extend the existing Main Shaft probe rather than adding another sibling probe.
Use a short circular trace around the first failed conversion. Per decision,
capture:

- frame, room, x/y/subpixels, velocity, momentum, pose, facing, movement type;
- classified region, current hop/window, chosen buttons, and decision reason;
- selected enemy slot/species/x/y/HP/freeze plus target-relative dx/dy;
- beam charge, projectile count, and active-PLM additions/removals;
- events `launch_1675`, `stairs_contact`, `stairs_planted`,
  `enemy_interrupted`, `relaunch`, and `repeated_cycle`.

Stop and save at the first stable 1543 seat or at the first classified repeated
cycle. Keep only the last roughly 180–300 frames plus event summaries. A red
artifact must identify the first divergence, not merely the final timeout.

Mine takes 02–05 into a compact fixture of stable seats, takeoff frames, peaks,
and input runs for the shaft above y1675. Compare the bot trace against take 02
at these transitions. Treat `stairs_planted` as the immediate assertion and
`mid_climb` y~680 as the phase outcome.

This audit intentionally did not run the future loop. When execution begins,
first reproduce the existing miss once from the natural west-super pin with
the new trace enabled. If it does not reproduce, stop and explain the changed
precondition instead of tuning against a different failure.

### 2. Repair observability before trusting PLM conclusions

In one isolated change:

1. Divide the PLM block byte offset by two in the shared decoder and the
   duplicated Wrecked Ship probe reader.
2. Replace the test fixture that canonizes the wrong conversion with examples
   containing odd/even cell positions and a real captured Wrecked Ship slot.
3. Add a source comment naming `$1C87` as a byte offset into two-byte level
   data.
4. Distinguish active-slot telemetry from unavailable bank-$7F collision data
   in diagnostic output/docs.
5. Remove or replace the generic `boss` field in this probe with a trusted
   bank-$7E read.

Do not change the controller in the same experiment. Reproduce the red once
with corrected telemetry so subsequent evidence has valid coordinates.

### 3. Replace the guessed first stair conversion with the human route

Make one behavior change at a time:

1. Reproduce the human first launch: A/RIGHT+A without forcing B for the entire
   arc.
2. During committed flight and landing, keep movement ownership in the shaft
   controller; do not let generic ice targeting preempt the landing.
3. Require a planted 1543 stair state, then traverse right toward the
   tape-derived x~1252–1259 takeoff.
4. Launch left from that far-right window using the take-derived input. Delete
   or replace the current guessed `1120..1180` window; do not widen both windows
   until something happens.

After each change, compare the same events to take 02. A useful result is a
stable stair seat or a new, earlier, well-classified divergence. A smaller
minimum y by itself is not progress.

Only if tape-derived avoidance still fails should the next isolated experiment
be combat. Pre-clear a named Atomic from a stable seat, validate line of fire,
and require an HP decrease or freeze-timer increase. If neither changes within
a short charge/release budget, relinquish combat ownership and escape; never
wait indefinitely at overlap.

### 4. Derive the rest of the shaft instead of extending guessed hops

Once `stairs_1543` is stable, extract each next grounded seat and takeoff from
the successful tapes through y~680. For each seam:

1. Define a private stable-seat predicate from multiple successful takes.
2. Implement only the next seat-to-seat transition.
3. Reproduce from the natural prior seat, not an airborne minimum-y dump.
4. Record exact exit x/y/pose/velocity and the first divergence on red.
5. Move upward only after the seam is repeatable.

Keep orchestration in the existing Main Shaft composer. Do not create a family
of sibling controller modules or let a source file grow past the repo's soft
limit; remove obsolete guessed geometry when the tape-derived policy replaces
it.

### 5. Finish Attic and prove the successor seam

After the natural `west_super -> mid_climb` phase is green:

1. Prove `mid_climb -> attic_seat` with a planted, usable Attic-side seat.
2. Prove `attic_seat -> attic_door` exits Main Shaft into ordinary Attic
   `0xCA52`, game state 8.
3. Dual the complete Main Shaft hop from
   `post_ws_basement_to_main.state`; phase pins are acceleration only.
4. Start the next Attic-to-Sync hop from the exact Main Shaft exit twice. This
   is the successor proof required before promoting the hop.
5. Update the residual, bead, evidence, and status only to the highest gate
   actually cleared. Do not change `DEFAULT_CONTINUOUS_TIP` or claim Gravity
   from a Main Shaft-only result.

## Future verification ladder (not run during this audit)

Run the narrowest layer that can falsify each change when implementation
begins:

1. ROM-free unit checks for PLM conversion, stable-seat predicates, target
   priority/line-of-fire, and action ownership.
2. One diagnostic run: natural west-super pin to stable 1543 stairs.
3. Repeat and dual: natural west-super pin to public `mid_climb` y~680.
4. Phase-local upper-shaft iteration from the first naturally captured planted
   seat, followed by natural-entry recomposition.
5. Full Main Shaft dual from `post_ws_basement_to_main.state` to Attic.
6. Successor Attic-to-Sync dual from the exact full-hop exit.

For each ROM run, record source hash, start observation, target predicate, frame
count, end observation, and artifact paths. On a red, save the earliest stable
divergence and the bounded trace; do not merely increase the frame budget.

## Acceptance criteria

- The diagnostic loop reproduces the original miss and identifies the first
  loss of progress with controller reason and enemy/PLM context.
- PLM coordinates use the source-correct byte-offset conversion; no breakable
  brick claim relies on the old doubled coordinates.
- The bot plants the 1543 stairs in a tape-supported state without ending in a
  live-enemy overlap.
- Natural `west_super -> mid_climb` reaches the existing y~680 contract twice.
- The complete natural Main Shaft hop reaches Attic `0xCA52` twice from
  `post_ws_basement_to_main.state`.
- The exact Attic exit starts the Sync successor successfully twice.
- Lower green phases remain unchanged and no intermediate phase pin is called
  hop-green.

## Process audit

### Keep

- Natural-predecessor pins, dual confirmation, and exact held exits.
- Separate phase green versus full-hop green, with explicit non-claims.
- Successful human tapes with frame-by-frame state, enemies, PLMs, projectiles,
  and inputs.
- The rule that a usable contact needs position, pose, and velocity—not only a
  height record.
- One living residual and one living tip.

### Change

- Mine successful tapes before writing geometry for the next red seam. The
  lower departure was data-derived; the upper `SHAFT_HOPS` list was not.
- Make the first red reproduction and its exact symptom the unit of work.
  Minimum y and final screenshots are supporting evidence, not a feedback loop.
- Log action ownership and decision reason whenever overlays can preempt route
  movement.
- Test contracts and transitions, not only the controller's current button
  tuple or source-code strings.
- Source-validate WRAM semantics before encoding them in fixtures. A passing
  fixture can preserve a decoding bug.
- After three identical misses, stop widening geometry or adding combat. Save
  the earliest stable seat and compare against the successful tape.
- Prefer an observed avoidance route over an invented enemy interaction. The
  tape shows whether combat is actually part of the seam.

### Decision rule for the current red

The next implementation should begin with observability plus the tape-derived
far-right stair traversal. PLM coordinate repair is required for trustworthy
diagnostics, but breakable-brick investigation is gated behind a failure of the
corrected human trajectory. Enemy freezing is a fallback experiment, not the
current default.

## Non-claims for this planning pass

- No emulator, probe, replay, or test was run.
- No controller or WRAM code was changed.
- No phase or hop was promoted.
- No bead was claimed, closed, exported, or committed.
- The runtime cause is ranked, not declared proven.
