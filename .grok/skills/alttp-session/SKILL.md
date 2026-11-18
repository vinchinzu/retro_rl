---
name: alttp-session
description: >
  Named ALttP sitting: one room controller, one predecessor replay, two
  attempts, then stop. Use when working in snes/alttp, starting a room hop,
  asked for a B9 or Eastern controller, or running /alttp-session.
---

# ALttP session

Planner owns `docs/STATUS.md`. State-load greens are not continuous claims.
Size: [CODING_STANDARDS.md](../../../CODING_STANDARDS.md) (~1000 LOC, merge or delete).
Budget, result fields, and the B9 assessment:
[snes/alttp/docs/plan.md](../../../snes/alttp/docs/plan.md).

The current user message names the sitting. A finished script does not.

## Sitting

1. Read the pins, fixture, and living residual already on disk.
   Done: one hypothesis names the predecessor pin and the leave glance.
2. Put that hypothesis in one room controller. The controller owns the
   movement loop and the RAM checks. Collision-safe steps share the attempt.
   When two notes disagree, compare the integer fields.
   Done: that controller file is the policy under test.
3. Replay it from the real predecessor.
   Done: the replay records room, layer, xy, HP, and either a new transition
   or the first prediction miss.
4. Stop at two attempts or 15 minutes. The second attempt is a different
   hypothesis. Two tries with no new transition and no replayable policy
   end the sitting. Report the blocker.

Write one compact result into the living residual, using the fields in the
plan's run contract. A new pin is evidence. Progress is a verified transition,
a replayable segment, or a closed maturity gate.

Old `alttp-*.rhai` scripts are in `snes/alttp/scratch/retired-workflows/` and
are not registered. Run one only when the user names that file.

## Loop

1. `bd ready -l alttp -l spine` when a database exists. Claim exactly one.
   No database: skip. Do not run `bd init`.
   Done: one bead is `in_progress`, or this step was skipped.
2. Overwrite the living residual named in
   [`snes/alttp/AGENTS.md`](../../../snes/alttp/AGENTS.md) under
   `snes/alttp/docs/tasks/`.
   Done: that file holds this sitting's leftover.
3. Leave proof is a RAM glance (room hex, module/submodule, x/y band, sword,
   `$F3CC` follower, keys). Save-state pins are not leave proof.
   Done: the glance matches that leftover.

Natural-entry: a hop is route-ready only from the real predecessor continuous
state. Ladder: `planned` → `isolated` → `natural_entry` → `continuous`.

## Non-claims

Did not STATUS-promote. Did not treat a save-state pin as power-on. Did not
write `$F3CC` / sword / keys. Did not copy approach xy into Python. Did not
overwrite `recordings/verified_tip_run.json` on a red. Did not start another
script from this sitting.
