# Benchmark Status

Last updated: 2026-03-05.

This file defines the repo's benchmark ladder and records the current state of each track. The point of the ladder is to separate "works autonomously" from "works under tighter observational constraints" without blocking progress.

## Benchmark Tiers

### Bronze

Autonomous and reproducible, with very few restrictions.

- No human input after the attempt starts.
- Actions must still go through legal controller input.
- Read-only RAM access is allowed.
- Game-specific heuristics, room graphs, reward shaping, specialist models, scripted menus, and save-state curricula are allowed.

Use Bronze to make the harness work end to end.

Imitation learning is allowed. For Bronze, it is often a good fit for menus, wake-up sequences, tricky exits, and early route seeds.

### Silver

A stronger runtime benchmark with much less privileged information.

- No human input after the attempt starts.
- No RAM writes, teleports, or mid-run emulator-state mutation.
- Runtime control should rely mainly on pixels plus generic harness state.
- RAM may still be used for training rewards, offline labeling, resets, benchmark bookkeeping, and limited generic runtime safety checks.
- A full hardcoded route should not be the entire solution.

Use Silver to prove the system is becoming game-playing software instead of a debug script.

Imitation learning is still allowed at Silver. The benchmark claim depends on runtime inputs and outputs during the attempt, not on whether demos were used in training.

### Gold

Pure runtime benchmark.

- The runtime agent observes only pixels, action history, and its own memory.
- The runtime agent outputs only controller input.
- No RAM or emulator-internal state is available in the decision loop.
- Success is judged from a published start state and objective.

Use Gold to measure actual "play the game from the screen" progress.

Imitation learning is also allowed at Gold, but the live agent must still remain pixels-in and controller-out.

## Rules For A Valid Benchmark

- Publish the start state or deterministic reset sequence.
- Publish the success condition.
- Log attempt count, success rate, and completion time or frame count.
- Report the claimed tier next to every result.
- If training used privileged signals, say so separately from the runtime tier.
- Loading a state or mutating emulator memory during an active attempt invalidates that attempt.

## Repo Status Board

| Track | Benchmark target | Bronze | Silver | Gold | Notes |
|-------|------------------|--------|--------|------|-------|
| `fighters_common` | Win a fight from a fight-ready state | Achieved for SF2 and MK1; SSF2 and MK2 in progress | Unclaimed | Unclaimed | Repo notes document 500K-step SF2 and MK1 runs at 100% win rate in the current match benchmark setup. |
| `alttp/` | Fixed-state top-down room and dungeon benchmarks | Working Bronze baseline | Not started | Not started | Verified Bronze benchmarks now cover fresh-profile boot, wake/exit flow, Link's House overworld start, GT room clear, and a route to Hyrule Castle. |
| `harvest/` | Farm-clear task from a fixed morning state | Partial | Not started | Not started | The bot can autoplay farm-clearing tasks from prepared states, but this is not yet framed as a root-level tiered benchmark. |
| `super_metroid_rl/` | Segment and chained-route completion | Partial | Not started | Not started | Recording, segment training, and route scripts exist; a full autonomous run benchmark is not yet claimed. |
| `donkey_kong_country/` | Level or route completion from published states | Infrastructure only | Not started | Not started | Play, autosplit, replay, and recording support exist; no documented autonomous clear benchmark yet. |
| `super_mario_bros/` | Segment and route completion | Partial | Not started | Not started | Platformer route tooling is present in the repo, but the benchmark ladder is not normalized at the root level. |

## Priority: Top-Down Adventure Harness

The next shared abstraction target should be ALTTP-like games. They force the harness to handle navigation, interaction, combat, room transitions, and long-horizon objectives in a way that should also transfer to other top-down adventures.

### Proposed ALTTP Benchmark Ladder

| Stage | Bronze | Silver | Gold | Current status |
|-------|--------|--------|------|----------------|
| Boot and enter a controllable state | Auto-load a clean state or menu sequence with RAM-aware validation | Same, but no game-specific state mutation during the attempt | Pixel-only confirmation of reaching control | Bronze working: cold boot, blank SRAM, new-slot creation, wake-up, and house exit are benchmarked |
| Single-room navigation | Reach a target tile or door using RAM position/collision | Vision-led navigation with generic stuck recovery | Pixels only | Not benchmarked |
| Single-room combat | Clear a room using RAM HP / sprite slots | Mostly vision-driven combat with limited generic runtime help | Pixels only | Arena environment exists; tier reporting does not |
| Room-to-room traversal | Use room graph plus RAM coordinates | Hybrid navigation with generic transition handling | Pixels only | Bronze working for the opening route: Link's House rainy overworld start to Hyrule Castle |
| Dungeon objective | Complete a published dungeon or item objective | Same objective with limited privileged runtime state | Pixels only | Not started |

### Shared Harness Work Needed Next

- Add a root benchmark runner that standardizes attempts, metrics, video/log artifacts, and tier labels.
- Move recording and replay features from `harvest/` and `super_metroid_rl/` into `retro_harness/`.
- Add top-down adventure adapters for room id, position, facing, interaction lock, dialogue, transition state, and combat state.
- Add reusable macro-actions for movement, facing, attack, interact, hold-through-transition, and stuck recovery.
- Add pixel-dataset capture and replay-friendly evaluation paths for Gold work.

## Practical Next Step

If the repo is going to bias toward top-down adventure harnesses first, the next concrete deliverable should be:

1. A shared benchmark runner in `retro_harness/`.
2. An ALTTP fixed-state benchmark suite with at least:
   - one room-navigation task
   - one room-clear task
   - one multi-room traversal task
3. Tier labels attached to every reported result.

That Bronze baseline now exists for the opening ALTTP route. The next useful move is to expand from the opening castle path into additional overworld segments, interior objectives, and recovery-heavy tasks without abandoning the same benchmark runner and artifact pipeline.
