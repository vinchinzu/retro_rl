---
name: predict-path
description: >
  Predict the next RAM pose before a live act, grade it, and plan room paths
  offline. Use for Zelda I occupancy/door-graph work, SMB approx residual
  search, or any retro_rl route that is wasting emulator probes. Halt a plan
  at the first miss; do not batch exploration.
---

# Predict, then path

Shared loop for Zelda I and SMB. Grammar: `retro_harness.predict`.
Unknown clauses miss (not a weak "pose changed" fallback).

## Gate

1. Write a claim the next frame can contradict (`x=`, `move DX,DY`,
   `screen=0xNN`, `x≈120±4`). Empty claim is a refuse.
2. Act once. Grade against RAM. A miss is the useful result — it dates the
   belief that broke.
3. Batch only mechanics that already held. Planned sequences stop at the
   first miss.

## Offline before live

| Game | Search model | Live halt |
|------|----------------|-----------|
| Zelda I | `door_graph.bfs_path` + `walk.physics.OccupancyWalker` | occupancy miss → block that cell → replan; no path → stand (do not probe) |
| SMB | `smb.approx.step` / `rollout` | first missed `Grade` / `first_miss_index` (not residual wrapped as `halt_plan`) |

Do not probe the emulator to feel a path BFS can close. Do not hill-climb
the emulator to discover a jump the stepper already models.

One-page notes stay Verified / Assumed / Plan. Dead beliefs stay next to the
evidence that killed them.
