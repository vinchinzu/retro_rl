# TASK SM-TIGHTEN-01B: Implement business climb settle trim (P1 only)

## Recipe step
efficiency implement (bounded patch — **not** STATUS / continuous promotion)

## Model
Luna

## Own files only
- `routes/kpdr/business_climb.py` (**edit**)
- optional: `docs/tasks/SM-TIGHTEN-01B-note.md` (**create**, short residual note)

Do **not** edit continuous.py, STATUS.md, hijump_return.py, progression, tracker.

## Context
- Analysis: `docs/tasks/SM-TIGHTEN-01-report.md` **P1 only**
- Split: `business_to_warehouse` ~2,257f on `start_to_varia`
- **P1 recipe:** replace fixed `_hold(session, 20, reason="business_NNNN_settle")`
  platform settles with a shorter idle (recommend **5f**) so `_wait_standing_y`
  (already present after each settle) starts polling sooner
- **Do not** implement P2 (setup jumps 4→3) or P3 (runup_907) in this card
- Continuous re-record (`--to kraid`) is ~27 min — **planner residual**, not
  required for card EXIT (document it clearly)

## Read first (all)
- `docs/tasks/SM-TIGHTEN-01-report.md` (P1 section + caveats)
- `routes/kpdr/business_climb.py` (full `_business_high_jump_platforms` + callers)
- `routes/controller_common.py` (`wait_standing_y` / related if present)
- `docs/STATUS.md` business-climb continuous-hardening notes (read only)
- Grep `business_.*_settle` in `business_climb.py`

## Do (thorough)
1. In `business_climb.py`, for each platform settle that is exactly
   `_hold(session, 20, reason="business_NNNN_settle")` inside the high-jump
   platform climb (1339, 1227, 1147, 987, 907, 843, 779, elevator — match report):
   - Change the hold from **20 → 5** frames
   - Keep the reason string suffix `_settle` so dwell tooling still labels them
   - **Do not** remove `_wait_standing_y` / standing gates
   - Leave `business_1067_settle` at 30f unless it is clearly the same pattern
     and the report groups it with the 20f settles — if unsure, leave 30f
   - Do **not** change `runup_907`, setup jump count, floor recover settles,
     or elevator_center_settle unless they are exact 20f platform settles in P1 scope
2. Re-read the full function after edit; ensure hop logic / gates unchanged
3. Run import/unit smoke:
   ```bash
   uv run pytest super_metroid/tests/test_controller_common.py -q
   ```
4. Write residual note (final message and optional short `SM-TIGHTEN-01B-note.md`):
   - Lines/reasons changed
   - Speculative save band from report (~160f) — **not claimed**
   - Planner must re-record:
     `uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video`
     then `split_dwell.py` on `start_to_kraid.json` for `business_to_warehouse`
   - If continuous fails after this patch, revert settles to 20f (document)

## Residual required (super-clean)
- Exact list of reason labels touched
- What was **not** changed (P2/P3, continuous, STATUS)
- Continuous verify command for planner
- Risk: continuous natural-entry variance on climb lips

## Do not
- P2/P3 patches
- continuous.py / STATUS edits
- Claim frame savings without re-record
- Progression RAM writes

## Acceptance
- [ ] Only settle-hold durations reduced as specified
- [ ] pytest controller_common green
- [ ] Diff summary + residual continuous verify path

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
rg -n "business_.*_settle" super_metroid/routes/kpdr/business_climb.py
# planner only (do not run unless card explicitly asks and time allows):
# uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
```
