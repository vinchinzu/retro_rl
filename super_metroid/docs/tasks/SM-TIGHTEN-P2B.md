# TASK SM-TIGHTEN-P2B: Re-try setup jumps 4→3 with **mandatory pure green** gate

## Recipe step
efficiency implement (pure-first — continuous still planner)

## Model
Luna

## Own files only
- `routes/kpdr/business_climb.py` (**edit** setup jump loops only)
- `docs/tasks/SM-TIGHTEN-P2B-note.md` (**create**)

Do **not** edit settles (leave current 12f or whatever is in tree), runup,
continuous, STATUS.

## Context (honest Wave-4 failure)
- Wave-4 P2 applied `("RIGHT","LEFT","LEFT")` (3 jumps).
- Pure `business-to-warehouse` from
  `scratch/continuous_like_business_climb_entry.state` **RED**:
  `business_1339_ground` y=1419 floor @ f957.
- Planner **reverted** setup to 4 jumps; pure then **GREEN** (~3721f → warehouse).
- Settles remain 12f (01C) and pure-green with 4-jump setup.
- This card re-attempts 3 jumps **only if pure stays green**. If pure red,
  **revert immediately** before EXIT and residual the fail pin.

## Read first
- `docs/tasks/SM-TIGHTEN-P2-note.md`
- `docs/tasks/SM-TIGHTEN-01-report.md` P2
- `routes/kpdr/business_climb.py` setup loops
- `docs/tasks/SM-PURE-ISO-note.md`

## Do
1. Confirm current setup is 4-tuple. Change **only** both setup loops to
   **one** 3-jump candidate. Prefer a different order than failed Wave-4 if
   you have a reason; document choice. Candidates:
   - `("RIGHT", "LEFT", "LEFT")` — known pure-red on this source
   - `("LEFT", "LEFT", "RIGHT")`
   - `("LEFT", "RIGHT", "LEFT")`
2. **Immediately** run pure (required, not optional):
   ```bash
   uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
     --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
   ```
3. If exit ≠ 0: **revert to 4-jump** `("RIGHT","LEFT","LEFT","RIGHT")`, re-run
   pure to confirm green, residual RED attempt + pin. Card still “done” with
   honest fail.
4. If pure green: leave 3-jump in tree; residual demands planner continuous
   `--to kraid` before any claim.

## Residual required
- Tuple tried + pure exit + pin if fail
- Final tree state (3 or 4 jumps)
- Continuous gate command

## Do not
- Change settles/runup
- Leave pure-red 3-jump in tree
- continuous / STATUS claim

## Acceptance
- [ ] Pure green with 3-jump **or** pure-red attempt + 4-jump restored + pure green
- [ ] pytest controller_common green
- [ ] Residual complete

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```
