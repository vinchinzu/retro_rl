# Plan — Donkey Kong Country

Facts: [STATUS.md](STATUS.md). Commands: [../AGENTS.md](../AGENTS.md).

## Next milestones

1. **M2** — Document remaining RAM for goal/bonus/death transitions.
2. **M3** — Scripted clear of Jungle Hijinks from a fixed level state.
3. **M4** — Natural-entry clear of the next level from Jungle Hijinks exit.
4. Wire `retro_harness.platformer` level config if not already sufficient for route eval.

## Work queue

### Autosplit + timing

- Lock level start using in-game timer + movement change.
- Confirm level ID → name mapping for the first few levels.
- Add RAM for transitions (goal, bonus, death) to tighten splits.

### Recording + training

- Standardize recording paths/metadata with shared harness writers.
- Optional CLI for listing recordings and converting to MP4.
