# Plan: Donkey Kong Country

Facts: [STATUS.md](STATUS.md). Commands: [../AGENTS.md](../AGENTS.md).

## Next milestones

1. M2. Document remaining RAM for goal, bonus, and death transitions.
2. M3. Scripted clear of Jungle Hijinks from a fixed level state.
3. M4. Natural-entry clear of the next level from the Jungle Hijinks exit.
4. Wire `retro_harness.platformer` level config if the current config is not enough for route eval.

## Work queue

### Autosplit and timing

- Lock level start using the in-game timer plus a movement change.
- Confirm level id to name for the first few levels.
- Add RAM for goal, bonus, and death so splits are tighter.

### Recording and training

- Standardize recording paths and metadata with the shared harness writers.
- Optional CLI for listing recordings and converting to MP4.
