# rr-kw8t Main Shaft finish and diagnosis plan

Static audit refreshed 2026-08-30. This pass did not boot the ROM, run a
probe, run tests, or change controller code. It inspected the current
uncommitted controller, the latest exact-red dual report, and successful human
takes 02–05. Runtime claims below remain hypotheses until the existing probe
reproduces them with a bounded decision trace.

## Outcome and current gate

Finish the powered Wrecked Ship Main Shaft hop from its natural predecessor to
ordinary Attic:

```text
post_ws_basement_to_main.state
  -> grate_seat                         GREEN x2
  -> west_super                         GREEN x2, natural entry
  -> mid_climb (1101, 651) p9           GREEN x2, natural west_super
  -> clear the 572 ceiling              RED, first target
  -> clear/traverse the y=523 ledge      not encoded robustly
  -> upper platforms / attic_seat        not proved
  -> Attic 0xCA52 gs=8                   full-hop GREEN only here
  -> prove the exact successor Sync
```

The living failure is no longer the 1675→1543 conversion described by the
older version of this plan. That seam and the remainder of the lower shaft
through the planted 651 seat are dual-green. Do not retune the pit, grate,
west-super, 1543, 1130, 1019, 827, or 651 approach while fixing the upper
shaft.

Iterate from
`custom_integrations/SuperMetroid-Snes/scratch/post_ws_main_mid_climb.state`,
then recompose from `post_ws_basement_to_main.state`. Neither phase pin is hop
GREEN.

The relevant route files are already part of a large uncommitted move from
`routes/kpdr/k6/` to `routes/kpdr/wrecked_ship/`. Preserve those edits. Keep
the fix in the existing geometry/action/shaft composer; do not create another
probe or sibling controller.

## Evidence already present

The latest report
`custom_integrations/SuperMetroid-Snes/scratch/ws_main_to_attic_dual.json` is
an exact deterministic RED from the mid-climb pin:

| Fact | Observation |
|------|-------------|
| Both runs | **5,369f**, same final state |
| Final | Main Shaft `0xCAF6`, `(1234,587)` p3, planted |
| Progress | minimum y **572** at `(1164,572)` p26; not Attic |
| Beam | charge **53** at timeout |
| Upper enemies | Atomics `0xE9FF` at `(1221,588)` and `(1209,588)`, both 50 HP |
| PLM result | no recorded proof that the upper shot blocks opened |

Successful take 02 reaches the same wall at `(1243,587)`, uses alternating
`UP+X` / `UP` taps, observes three upper `0xD074` / `0xD080` PLM spawns,
walks left to `(1231,587)`, and jumps. It then lands near `(1204,523)`,
walks left while firing several short horizontal bursts, and produces more
shot-block PLM transitions before climbing from the left side. Takes 04 and 05
also produce three upper-wall PLM spawns before leaving the wall.

The compact human-tape PLM pixel coordinates predate the corrected `$1C87`
decoder and must not be copied as live coordinates. The current shared decoder
correctly divides the byte offset by two. Re-capture or derive the actual
upper-block coordinates with the corrected decoder.

## Confirmed static defects

### P0 — the upper shot action never owns a release cadence

`_slope_651_action` returns `shoot_up_action()` continuously while
`ceiling_open` is false. That helper is always `("UP", "X")`. The knockback
branch independently does the same thing. Unlike the lower grate and shared
ceiling-door controllers, neither branch uses beam charge or a local shot
cycle to release X.

This directly matches the exact-red artifact: the controller plants at the
right wall, never latches clearance, consumes the 3,600-frame climb budget,
and exits still holding charge 53. The first fix experiment is a tape-derived
tap/release cadence scoped to this wall, not a larger frame budget.

### P0 — one boolean is weaker than the observed clearance contract

`ceiling_open` becomes true on the first new member of
`{0xD074, 0xD078, 0xD080}` seen while Samus is anywhere in the broad 651 band.
That loses both position and count. Successful takes break three blocks at the
upper wall before the first jump, and then break several more while traversing
the y=523 ledge.

A single family-ID spawn is therefore neither necessary proof that the right
upper block changed nor sufficient proof that the route is clear. Replace the
boolean with phase-local, position-qualified evidence. At minimum record the
spawned slot, ID, corrected block position, and the number/set of upper-wall
blocks cleared. Do not jump merely because an unrelated family ID appeared.

### P0 — the blocker is mislabeled in the residual

The two moving enemies at the failed wall are Atomics (`0xE9FF`), not Coverns
(`0xEA3F`). The latest report places them at `(1221,588)` and `(1209,588)`
with 50 HP. The human upper-wall shot window has the corresponding Atomics
farther left and/or below the vertical firing lane and still at 250 HP.

Correct the diagnosis before selecting a combat policy. The existing 651
overlay skip means movement owns this seam; generic Covern keepaway is not the
fix. The phase difference is relevant, but the missing X-release cycle must be
fixed and observed before enemy handling is added.

### P1 — knockback can hide the event that unlocks the route

`climb_until` checks `is_knockback` before `_update_lip_hit`. At the 587
wall, the knockback branch can shoot continuously and `continue`, skipping
PLM sampling for that frame. A transient shot-block spawn can therefore occur
without updating `ceiling_open`, especially while an Atomic repeatedly
contacts Samus.

PLM/event observation must occur before early controller branches, or the
upper shot sequence must retain its own last snapshot across knockback. One
route state must own shot cadence, observed block transitions, and recovery.

### P1 — the y=523 successor is absent from the current policy

Opening the first ceiling is not the end of the upper-shaft problem. Human
take 02 lands at y=523 and crosses left with short `LEFT+X` bursts that
produce additional shot-block PLMs. The current special 651 controller
releases ownership after a grounded y=523 landing, after which generic hop
selection has no contract for clearing that row.

Without an explicit `upper_523` transition, a fix can green the first jump and
immediately fail on the next obstacle. Treat `587 wall -> planted 523 -> left
block run -> upper takeoff` as one seam unless a naturally held y=523 seat can
be proved usable by both sides.

### P1 — the current red artifact is deterministic but not diagnostic

The dual report proves repeatability, but it saves only the final state and
minimum y. It does not show button ownership, X press/release edges, charge,
projectile flight, coordinate-qualified PLM transitions, or the first Atomic
interception. The 3,600-frame timeout turns a short deadlock into a 5,369-frame
report.

The existing probe should stop on the first repeated planted-wall cycle and
save a short ring buffer. Do not diagnose future reds from final screenshots
alone.

## Ranked runtime hypotheses

Test these in order, one variable at a time:

1. **Missing X release is the immediate cause.** If the wall uses the human
   tap/release rhythm, projectiles and upper PLM transitions will appear and
   the controller will leave `(1234,587)`.
2. **The first spawn causes a premature jump.** If clearance requires the
   observed upper block set rather than one family ID, the jump will stop
   bonking at y=572 and plant the y=523 ledge.
3. **Natural bot enemy phase blocks an otherwise-correct shot cycle.** If
   Atomics are the remaining cause, correct taps will damage/freeze them or
   fail to reach the expected upper blocks while the trace shows projectile
   interception. Waiting for a clear lane or removing a named Atomic will
   change that result.
4. **Knockback loses the successful PLM event.** If so, the trace will show an
   upper PLM spawn during a frame where the knockback branch bypasses the
   latch; moving observation ahead of action dispatch will retain it.
5. **The first ceiling greens but y=523 is the next independent blocker.** If
   so, the controller will plant y=523 and then cycle without the horizontal
   block transitions present in takes 02–05.

Do not start with a new jump window, wider geometry, or more timeout. Those do
not falsify the leading hypotheses.

## Execution plan

### 1. Tighten the existing red loop

Extend `scripts/probe/ws_main_climb.py`; do not add another probe. From the
natural mid-climb pin, retain roughly the last 240 frames and emit events for:

- stable 651 plant, 587 wall contact, and y=523 plant;
- action owner/reason and buttons, especially X press and release edges;
- beam charge plus projectile slot/type/position;
- Atomics by slot, x/y, HP, freeze, and overlap with Samus/projectiles;
- PLM additions/ID changes with corrected block coordinates;
- `upper_block_1..N`, `wall_clear`, `jump_587`, `land_523`, and repeated
  cycle.

Stop RED after a bounded planted-wall cycle with no new projectile or
upper-block event; stop GREEN for the first experiment only at a planted y=523
seat. Preserve the full public `attic_seat` predicate for the phase result.

The first future run must reproduce the current exact miss before behavior
changes. If it does not, stop and identify the changed source state rather than
tuning a different failure.

### 2. Give the wall a local tap/release shot state

Use a counter that begins on stable right-wall contact, not global session
frame. Reproduce the observed rhythm: short `UP+X` bursts separated by `UP`
release frames. The action must make charge fall and a projectile leave before
starting another burst.

Apply the same state through brief knockback so contact cannot reset it to an
eternal hold. Keep PLM observation active on every frame. Do not mix enemy
policy into this experiment.

Success for this step is coordinate-qualified upper-block transitions and a
controlled departure from the wall; minimum y alone is not success.

### 3. Replace `ceiling_open` with an upper-clearance contract

From corrected live telemetry and takes 02–05, identify the three upper-wall
block positions. Track a small set/count of those positions within the current
climb context. Only allow the left takeoff when the required set is clear.

Keep lower `lip_hit` state separate from upper clearance; they are different
obstacles and should not share a semantic latch. Ignore family-ID spawns
outside the upper-wall coordinate band. If live PLMs expire before the third
event, retain the observed cleared-position set for the phase.

### 4. Handle Atomics only if the corrected shot contract still fails

First try the correct shot cadence with no new combat. If a bounded trace
shows repeated projectile interception, compare these isolated experiments:

1. wait at the stable 651 seat until both upper Atomics leave the shot/takeoff
   corridor;
2. from a stable seat, target the specific blocking Atomic and require HP or
   freeze change before returning ownership to movement;
3. prefer killing over freezing if a frozen Atomic becomes a solid obstacle in
   the only passage.

Each experiment needs a short exit condition. Never wait or charge indefinitely
at `(1234,587)`. Do not re-enable the broad generic ice overlay across the
whole 651 arc; it would steal the already-green slope run.

### 5. Encode the y=523 block run from tape evidence

After the 587 jump plants near `(1204,523)`, add a private stable-seat
predicate and reproduce the leftward tap-fire traverse. Qualify each required
block transition by position. The human sequence reaches the left side around
x1095–1077 before the next committed jump.

Keep this in the existing Main Shaft composer. If the landing cannot start the
traverse twice from the exact held exit, keep the wall jump and ledge traversal
as one change. Do not publish a y=523 phase pin as hop GREEN.

### 6. Derive the final upper platforms, then use the existing Attic door

Mine multiple successful takes for stable seats and departures above y=523.
Take 02 gives the rough ladder:

```text
y523 left takeoff
  -> y443 plant
  -> y363 plant / right launch
  -> y171 plant / left launch
  -> top seat around (1123,91)
  -> shared attic_door_action
  -> Attic 0xCA52 gs=8
```

For each transition, require room, x/y, planted pose, and near-zero vertical
velocity. Remove superseded guessed hop geometry as the tape-derived
transitions land. Do not add controller modules or grow a parallel route.

### 7. Recompose in increasing scope

After the local upper seam is repeatable:

1. dual `mid_climb -> attic_seat` from the natural mid-climb pin;
2. dual the complete Main Shaft hop from `post_ws_basement_to_main.state` to
   ordinary Attic `0xCA52`;
3. start the exact Attic successor from both full-hop exits and prove Sync;
4. update the residual/bead only to the highest gate actually cleared;
5. later run the Gravity milestone power-on dual. Main Shaft alone does not
   change the living tip or prove Gravity.

## Future verification ladder

No verification was run during this planning pass. When implementation starts,
use the narrowest falsifier for each change:

1. contract checks for local shot cadence, upper-block qualification, and
   latch persistence through knockback;
2. one traced mid-climb-pin run to coordinate-qualified wall clearance;
3. repeat/dual to planted y=523 and then to `attic_seat`;
4. full natural Main Shaft dual to Attic;
5. successor Sync dual from the exact full-hop exits;
6. Gravity milestone power-on dual only after the local chain is complete.

Every ROM artifact should record source state/hash, start observation, target
predicate, frames, end observation, block events, and trace path. On RED, save
the earliest classified divergence instead of increasing the budget.

## Acceptance criteria

- The bounded trace reproduces the planted `(1234,587)` failure and identifies
  its first missing release, projectile, or block event.
- The upper-wall controller visibly presses and releases X; it cannot time out
  with one continuous charge hold.
- Upper clearance is based on the required corrected block positions, not one
  unqualified PLM family ID.
- Atomics and Coverns are named by their actual RAM IDs in evidence and policy.
- The exact 587 departure plants y=523 and the successor block traverse starts
  from that held state twice.
- `mid_climb -> attic_seat` clears twice from the natural mid-climb pin.
- The complete hop reaches ordinary Attic `0xCA52`, gs=8 twice from
  `post_ws_basement_to_main.state`.
- Both exact Attic exits start the successor successfully.
- Lower green phases remain unchanged, no intermediate pin is called hop
  GREEN, and no Main Shaft-only result changes the living tip.

## Non-claims for this planning pass

- No emulator, probe, replay, or test was run.
- No controller, artifact, residual, STATUS, bead, or route wiring was changed.
- No phase, hop, Gravity rung, or living tip was promoted.
- Existing uncommitted Wrecked Ship edits were preserved.
- The static defects are confirmed; their runtime ordering remains the ranked
  hypothesis list above until the bounded loop is executed.
