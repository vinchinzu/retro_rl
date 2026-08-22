---
name: zelda-assisted-route
description: Advance the Zelda I NES full-game route under Survival health refill, especially unexplored overworld or dungeon geometry, stuck navigation, and checkpoint composition. Use for zelda_i route work where screenshots and RAM traces should replace repeated random probes. Do not use to claim or tune Clean results.
---

# Zelda Assisted Route

Advance one route boundary at a time with deterministic input, visual evidence,
and the existing earned-capacity health refill.

## Establish the boundary

1. Read `nes/zelda_i/AGENTS.md`, `docs/ASSIST_CONTRACT.md`, and the current
   `docs/plan.md` handoff.
2. Run `bd ready -l zelda_i` and inspect the active tip issue. Claim only one
   issue.
3. Start from the real predecessor checkpoint. Record level, room, mode, x/y,
   keys, bombs, items, heart containers, and Triforce bits before acting.
   Predecessor inventory is sacred: do not default-zero keys or poke doors on a
   route runner. Isolated combat fixtures (`L5_Room_77`) may zero keys; route
   checkpoints may not.
4. Use `--infinite-life` for first-pass route and puzzle work. Require the
   assist report to show `progression_writes=0` and `capacity_writes=0`.
   Heart refill is always on. Bomb/key **count** top-up is a documented
   Survival shortcut (`docs/ASSIST_CONTRACT.md`) until a farm pass exists —
   never write `max_bombs` or grant undiscovered items. Door pokes are recon
   and cannot write route checkpoints.

## Predict, then act

Every live step needs a falsifiable RAM claim (`zelda_i.predict` /
`retro_harness.predict`). A miss names the wrong belief; it is not a reason
to extend a timeout.

Offline first (no emulator):

1. Room sequence: `door_graph.bfs_path` under inventory caps.
2. In-room walk: `walk_physics.OccupancyWalker` grades `move DX,DY`
   (`retro_harness.predict.grade_claims`). A stuck miss blocks the cell
   ahead and replans. No path → stand. Halt at the first unrecoverable
   miss; do not hunt or probe.
3. Door clips (LEFT+UP residual) stay one-frame policies in `level*_path.py`.

## Run a screenshot-first loop

1. Form one geometry hypothesis from the last PNG plus sampled RAM. Treat a
   walkthrough or TAS as a hypothesis, never as proof.
2. Put one-frame door/nav policy in `level*_path.py` (follow
   `level3_path.west_door_step`). `level*_dungeon.py` stays specs + stop
   predicates. Scripts do not grow path loops. Give actions semantic reasons
   and add unit tests for boundary coordinates.
3. Run one emulator trial. Save a screenshot on every room/screen transition,
   the final frame, and a compact sample every roughly 250 stuck frames.
4. On failure, inspect the final screenshot and the last coordinate/reason
   samples before editing. Change one thing, then rerun.
5. Never add random jitter, silently extend timeouts, or repeat an unchanged
   policy. Never poke doors, undiscovered items, or progression for a route
   claim. Bomb/key count top-up is the documented Survival shortcut until
   farming; label any other inventory poke as recon.

## Promote a segment

1. Require an exact stop predicate, preserved predecessor inventory, zero
   deaths, and a successful controller report.
2. Save checkpoint plus provenance only after success. A state-loaded segment
   remains development evidence even when it begins at the real predecessor.
3. Compose the new segment from its predecessor before moving deeper. Keep the
   assisted result out of the Clean program gate.
4. Put reusable game logic in `nes/zelda_i/`; promote to `retro_harness` only
   after a second game consumes it.

## Stop cleanly

Update `docs/plan.md` with the exact next command, expected transitions, last
observed failure, and evidence paths. After a verified segment, also refresh
the Next sections in `AGENTS.md` and `docs/STATUS.md`. If `LEVELN_ROUTE.md`
still says poke, that is a bug. Put only verified facts plus the single
program maturity gate in `docs/STATUS.md`. Run the narrow tests, update the
active bead, `bd sync`, and commit code with `.beads/issues.jsonl`. Do not push
unless requested.
