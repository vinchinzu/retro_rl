# TASK SM-CLEAN-BOMBS: Continuous power-on → Bomb Torizo exit (Clean) ★

## Recipe step
compose + record (★ Clean tip)

## Model
Luna → planner review

## Wave type
implement

## Own files only
- residual: `docs/tasks/SM-CLEAN-BOMBS-residual.md`
- only if fight/economy fails: `combat/bomb_torizo.py` **or** early policy
  segments — **one knob**, then assisted bombs re-verify (see
  `SM-CLEAN-BT-ECONOMY`)

Depends: clean infra green (artifacts + integrity); morph clean **done**.

## Context
- Assisted bombs continuous is already green (hash-pinned BT path).
- Clean disables ammo refill; BT fight + early rooms use natural capacity
  (10 missiles after detour).
- **Must not** overwrite `recordings/start_to_bomb_torizo.json`.
- Primary STATUS tip stays Frog assisted.
- **Prefix logged 2026-08-02:** both Missile packs green on clean at assisted
  frames (27,928 / 29,690). Pit settle fixed for beam-selected clean detour.
  **Do not re-solve BT here** — existing combat model / hash pin owns the fight
  (tank damage OK; deaths zero + zero resource writes). Residual:
  [`SM-CLEAN-BOMBS-residual.md`](SM-CLEAN-BOMBS-residual.md).

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/routes/START_TO_BOMBS.md`
- `combat/bomb_torizo.py`
- `routes/continuous.py` (`run_start_to_bombs`)

## Do
1. Run clean continuous `--to bombs` with clean artifact paths.
2. Require splits: morph, both missile expansions, bombs, BT defeat, BT exit,
   Parlor settle — same as assisted.
3. Clean integrity: zero ammo/energy writes; zero loads/progression/capacity.
4. If RED: residual → `SM-CLEAN-BT-ECONOMY` with **one** failure mode (ammo
   dry / death / stall frame).
5. If GREEN: dual re-verify (`SM-CLEAN-STAB`) then STATUS secondary
   (`SM-CLEAN-STATUS`) — planner.

## Acceptance
- [ ] Clean report success + bomb_torizo outcome
- [ ] Resource writes all zero
- [ ] Assisted bombs baseline files unchanged
- [ ] Residual PROCESS fields

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video
# re-verify once path stable:
uv run python super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video \
  --report super_metroid/recordings/start_to_bomb_torizo_clean_reverify.json
```

## Residual routing
- GREEN → `SM-CLEAN-STAB` then `SM-CLEAN-STATUS`
- RED ammo/death in BT → `SM-CLEAN-BT-ECONOMY`
- RED earlier room → name room + one controller knob card
