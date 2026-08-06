# TASK SM-CLEAN-BOMBS: Continuous power-on → Bomb Torizo exit (Clean) ★

## Recipe step
compose + record (★ Clean tip)

## Model
Luna → planner review

## Wave type
implement

## Own files only
- residual: optional `docs/tasks/SM-CLEAN-BOMBS-residual.md` (delete after close)
- only if fight/economy fails: `combat/bomb_torizo.py` **or** early policy
  segments — **one knob**, then assisted bombs re-verify (see
  `SM-CLEAN-BT-ECONOMY`)

Depends: clean infra green (artifacts + integrity); morph clean **done**.

## Context
- Assisted bombs continuous is already green (hash-pinned BT path).
- Clean disables ammo refill; BT fight + early rooms use natural capacity
  (10 missiles after detour).
- **Must not** overwrite `recordings/bombs.json`.
- Primary STATUS tip stays Bat Cave assisted.
- **Prefix logged 2026-08-02:** both Missile packs green on clean at assisted
  frames (27,928 / 29,690). Pit settle fixed for beam-selected clean detour.
  **Do not re-solve BT here** — existing combat model / hash pin owns the fight
  (tank damage OK; deaths zero + zero resource writes). Residual:
  (missile prefix green; residual deleted in hygiene pass).

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/routes/START_TO_BOMBS.md`
- `combat/bomb_torizo.py`
- `routes/continuous.py` (`run_bombs`)

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
- [ ] Clean report success + bomb_torizo outcome — **RED** 2026-08-06
- [x] Resource writes all zero (failed run still Clean intervention)
- [x] Assisted bombs baseline files unchanged
- [x] Residual PROCESS fields → `SM-CLEAN-BOMBS-residual.md` / `SM-CLEAN-BT-ECONOMY`

## Verify commands
```bash
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video
# re-verify once path stable:
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video \
  --report snes/super_metroid/recordings/bombs_clean_reverify.json
```

## Residual routing
- GREEN → `SM-CLEAN-STAB` then `SM-CLEAN-STATUS`
- RED ammo/death in BT → `SM-CLEAN-BT-ECONOMY`
- RED earlier room → name room + one controller knob card
