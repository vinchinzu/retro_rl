# TASK SM-BOSS-NATURAL-ENTRY-CLI: Standardize capture-natural CLI

## Recipe step
docs | efficiency

## Model
Flash

## Wave type
implement

## Own files only
- `combat/natural_entry.py` (helpers / argparse surface)
- `scripts/probe/` thin CLI wrapper **if** needed (one shared entry, not per-boss sprawl)
- optional residual: `docs/tasks/SM-BOSS-NATURAL-ENTRY-CLI-residual.md`
- optional short usage note in residual or `docs/BOSS_PIPELINE.md` **only if**
  already editing pipeline docs is required for the CLI — prefer residual doc

## Context (minimal)
- Bomb Torizo already has `capture-natural` / `prove-natural` patterns
- Goal: consistent `capture-natural <boss>` that records room + pose + door
  settle **without** progression writes
- Full fights remain deferred until natural entry on continuous chain
- Wave board: `docs/tasks/WAVE-11.md`

## Read first
- `combat/natural_entry.py`
- `scripts/probe/bomb_torizo_combat.py` (capture-natural pattern)
- `docs/BOSS_PIPELINE.md` § natural-entry capture
- `combat/features.py` catalog IDs (read only)

## Do
1. Provide a consistent `capture-natural <boss>` entry point (or document the
   single existing path + thin multi-boss dispatch) that records room + pose +
   door settle without progression writes.
2. Document usage for Phantoon / Botwoon / etc. (residual or BOSS_PIPELINE
   bullet only — no STATUS).
3. Residual → next boss catalog card (`SM-BOSS-UNIT-MATRIX` or planner pick).

## Do not
- Write boss flags or start a fight as “green”
- Touch `continuous.py` / STATUS
- Claim natural entry on the continuous chain

## Acceptance
- [ ] CLI or documented entry path works for at least one non-BT boss id
      (or honest BLOCKED residual naming missing catalog)
- [ ] No progression / boss-bit forges in the capture path
- [ ] Residual next card ID + one change

## Verify commands
```bash
# Prefer help / dry path that does not require a long continuous prefix:
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py --help
# If a multi-boss CLI is added:
# uv run python super_metroid/scripts/probe/<cli>.py capture-natural --help
```

## Done when
Residual filed. Capture is infrastructure only — not continuous evidence.
